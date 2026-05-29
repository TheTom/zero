//! The interactive REPL: wires an input source, key decoder, line editor, and a
//! [`Backend`] into Claude-Code-style inline rendering.
//!
//! "Inline" means output is printed in normal flow, so the terminal emulator's
//! own scrollback works exactly as users expect — we only take over the current
//! input block, redrawing it (one or more rows) in place as it is edited.
//!
//! [`App`] is generic over its input ([`Input`]) and output ([`Write`]) so the
//! whole loop is testable with scripted bytes and a captured buffer; the binary
//! instantiates it with a real terminal and stdout.

use crate::editor::LineEditor;
use crate::key::{decode_keys, Key};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;
use zero_core::backend::{Backend, StreamEvent};
use zero_core::clock::{format_duration, Stopwatch};
use zero_core::config::Config;
use zero_core::discovery::Discovered;
use zero_core::message::{Conversation, Message, Role};
use zero_core::openai::OpenAiBackend;
use zero_core::servers::ServerStore;
use zero_core::session::SessionLog;

/// Prefix drawn before continuation rows of a multiline input (aligns under the
/// prompt). Same display width as the prompt.
const CONT: &str = "  ";

/// A source of input bytes. `read` returns 0 on a poll timeout (not EOF).
/// `RawTerminal` implements this (see `term.rs`); tests use a scripted source.
pub trait Input {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// How `/clip` copies text out. The real one shells to the system clipboard;
/// tests inject a fake.
pub type ClipboardFn = Box<dyn FnMut(&str) -> io::Result<()>>;

/// Result of handling one key: keep looping or tear down.
#[derive(Debug, PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

/// Reverse incremental history search (`^R`) state.
#[derive(Debug, Default)]
struct Search {
    query: String,
    /// Index into history of the current match, if any.
    idx: Option<usize>,
}

/// The running terminal application.
pub struct App<I: Input, W: Write> {
    input: I,
    out: W,
    editor: LineEditor,
    conv: Conversation,
    backend: Box<dyn Backend>,
    log: Option<SessionLog<std::fs::File>>,
    prompt: String,
    /// Row offset (within the input block) where the cursor was left after the
    /// last render — so the next redraw knows how far up to go to clear.
    cursor_row: usize,
    /// True after a `^C` on an empty line; a second `^C` then exits.
    ctrl_c_armed: bool,
    /// True after one `Esc`; a second `Esc` clears the line.
    esc_pending: bool,
    /// `Some` while in `^R` reverse-search mode.
    search: Option<Search>,
    /// A dangerous shell command awaiting `y/N` confirmation.
    pending_shell: Option<String>,
    /// Human-readable backend/config summary, shown by `/config`.
    info: String,
    /// Live config, mutated by `/connect` and persisted to `config_path`.
    config: Config,
    config_path: Option<PathBuf>,
    servers_path: Option<PathBuf>,
    /// Servers found by the last `/scan`, for `/connect <n>`.
    scan_results: Vec<Discovered>,
    /// The last assistant response (raw markdown), for `/clip`.
    last_reply: String,
    /// How `/clip` copies — the system clipboard by default; swappable in tests.
    clipboard: ClipboardFn,
}

impl<I: Input, W: Write> App<I, W> {
    /// Build an app over an input source, an output sink, and a backend.
    pub fn new(
        input: I,
        out: W,
        backend: Box<dyn Backend>,
        log: Option<SessionLog<std::fs::File>>,
    ) -> Self {
        App {
            input,
            out,
            editor: LineEditor::new(),
            conv: Conversation::new(),
            backend,
            log,
            prompt: "› ".to_string(),
            cursor_row: 0,
            ctrl_c_armed: false,
            esc_pending: false,
            search: None,
            pending_shell: None,
            info: String::new(),
            config: Config::default(),
            config_path: None,
            servers_path: None,
            scan_results: Vec::new(),
            last_reply: String::new(),
            clipboard: Box::new(clipboard_copy),
        }
    }

    /// Override how `/clip` copies (tests inject a fake to avoid the real
    /// system clipboard).
    pub fn set_clipboard(&mut self, f: ClipboardFn) {
        self.clipboard = f;
    }

    /// Set the summary shown by `/config` (backend, model, config path).
    pub fn set_info(&mut self, info: impl Into<String>) {
        self.info = info.into();
    }

    /// Provide the live config and where to persist config + known servers, so
    /// `/connect` can attach to a discovered server and remember it.
    pub fn set_config(
        &mut self,
        config: Config,
        config_path: Option<PathBuf>,
        servers_path: Option<PathBuf>,
    ) {
        self.config = config;
        self.config_path = config_path;
        self.servers_path = servers_path;
    }

    /// Run the event loop until the user quits.
    pub fn run(&mut self) -> io::Result<()> {
        // Ask the terminal to report disambiguated keys (kitty keyboard
        // protocol). On terminals that support it, Shift+Enter then arrives as
        // `ESC [ 13 ; 2 u`; on others this is silently ignored. Popped in finish.
        self.out.write_all(b"\x1b[>1u")?;
        self.print_banner()?;
        self.redraw_input()?;

        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = self.input.read(&mut buf)?;
            if n == 0 {
                // A lone ESC that never grew into a sequence is a real Esc press.
                if pending == [0x1b] {
                    pending.clear();
                    self.dispatch(Key::Esc)?; // Esc never quits, only clears/arms
                    self.redraw_if_idle()?;
                }
                continue;
            }
            pending.extend_from_slice(&buf[..n]);
            let (keys, consumed) = decode_keys(&pending);
            pending.drain(0..consumed);
            for key in keys {
                if self.dispatch(key)? == Flow::Quit {
                    return self.finish();
                }
            }
            self.redraw_if_idle()?;
        }
    }

    /// Redraw the input line — unless a submode (search / shell confirm) owns
    /// the screen, in which case it renders itself and a redraw would clobber it.
    fn redraw_if_idle(&mut self) -> io::Result<()> {
        if self.search.is_none() && self.pending_shell.is_none() {
            self.redraw_input()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        // Pop the kitty keyboard-protocol flags we pushed in run().
        self.out.write_all(b"\x1b[<u")?;
        self.write_text("\n")?;
        Ok(())
    }

    /// Route a key, honoring submodes (confirm, search) and double-press combos.
    fn dispatch(&mut self, key: Key) -> io::Result<Flow> {
        if self.pending_shell.is_some() {
            return self.handle_confirm_key(key);
        }
        if self.search.is_some() {
            return self.handle_search_key(key);
        }
        // Reset double-press latches unless this key continues the combo.
        if key != Key::Ctrl('c') {
            self.ctrl_c_armed = false;
        }
        if key != Key::Esc {
            self.esc_pending = false;
        }
        self.handle_key(key)
    }

    fn handle_key(&mut self, key: Key) -> io::Result<Flow> {
        match key {
            Key::Ctrl('c') => return self.on_ctrl_c(),
            Key::Esc => return self.on_esc(),
            Key::Ctrl('d') if self.editor.is_empty() => return Ok(Flow::Quit),
            Key::Ctrl('d') => self.editor.delete(),
            Key::Ctrl('r') => self.enter_search()?,
            Key::Ctrl('l') => self.write_text("\x1b[2J\x1b[H")?,
            Key::Ctrl('a') | Key::Home => self.editor.home(),
            Key::Ctrl('e') | Key::End => self.editor.end(),
            Key::Ctrl('u') => self.editor.kill_to_start(),
            Key::Ctrl('k') => self.editor.kill_to_end(),
            Key::Ctrl('w') => self.editor.kill_word(),
            Key::Ctrl('b') => self.editor.left(),
            Key::Ctrl('f') => self.editor.right(),
            Key::WordLeft => self.editor.word_left(),
            Key::WordRight => self.editor.word_right(),
            Key::ShiftEnter => self.editor.insert_newline(),
            Key::Ctrl(_) | Key::Tab => {} // unmapped; ignore
            Key::Enter => return self.on_submit(),
            Key::Backspace => self.editor.backspace(),
            Key::Delete => self.editor.delete(),
            Key::Left => self.editor.left(),
            Key::Right => self.editor.right(),
            // Up/Down move between input lines first, then fall back to history.
            Key::Up => {
                if !self.editor.line_up() {
                    self.editor.history_prev();
                }
            }
            Key::Down => {
                if !self.editor.line_down() {
                    self.editor.history_next();
                }
            }
            Key::Char(c) => self.editor.insert(c),
        }
        Ok(Flow::Continue)
    }

    /// `^C`: clear a non-empty line; on an empty line, arm, then exit on a
    /// second press. Prevents an accidental single keystroke from quitting.
    fn on_ctrl_c(&mut self) -> io::Result<Flow> {
        if !self.editor.is_empty() {
            self.editor.clear();
            self.ctrl_c_armed = false;
            return Ok(Flow::Continue);
        }
        if self.ctrl_c_armed {
            self.write_text("\n^C\n")?;
            return Ok(Flow::Quit);
        }
        self.ctrl_c_armed = true;
        self.clear_input_block()?;
        self.write_text("\x1b[2m(press ^C again to exit)\x1b[0m\n")?;
        self.cursor_row = 0;
        Ok(Flow::Continue)
    }

    /// `Esc`: first press arms; second clears the whole input.
    fn on_esc(&mut self) -> io::Result<Flow> {
        if self.esc_pending {
            self.esc_pending = false;
            self.editor.clear();
        } else {
            self.esc_pending = true;
        }
        Ok(Flow::Continue)
    }

    fn on_submit(&mut self) -> io::Result<Flow> {
        let text = self.editor.submit();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            self.clear_input_block()?;
            self.cursor_row = 0;
            return Ok(Flow::Continue);
        }
        if matches!(trimmed, "/quit" | "/exit") {
            return Ok(Flow::Quit);
        }
        // Bare `exit`/`quit` (shell muscle memory) — don't send it to the model
        // and don't silently quit; nudge toward the real exit and arm ^C.
        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            self.echo_committed(&text)?;
            self.write_text("\x1b[2m(press ^C again to exit)\x1b[0m\n")?;
            self.ctrl_c_armed = true;
            self.cursor_row = 0;
            return Ok(Flow::Continue);
        }
        // `!cmd` — run a shell command inline (gated by the safety classifier).
        if let Some(rest) = trimmed.strip_prefix('!') {
            let cmd = rest.trim().to_string();
            self.echo_committed(&text)?;
            if cmd.is_empty() {
                return Ok(Flow::Continue);
            }
            let verdict = zero_core::safety::classify(&cmd);
            if verdict.is_dangerous() {
                let reason = verdict.reason.unwrap_or("destructive command");
                self.write_text(&format!(
                    "\x1b[33m⚠ {reason}\x1b[0m — run anyway? \x1b[2m[y/N]\x1b[0m "
                ))?;
                self.pending_shell = Some(cmd);
                self.cursor_row = 0;
                return Ok(Flow::Continue);
            }
            self.run_shell(&cmd)?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/help" {
            self.echo_committed(&text)?;
            self.print_help()?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/config" {
            self.echo_committed(&text)?;
            let info = if self.info.is_empty() {
                "no backend configured (stub)".to_string()
            } else {
                self.info.clone()
            };
            self.write_text(&format!("\x1b[2m{info}\x1b[0m\n"))?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/scan" {
            self.echo_committed(&text)?;
            self.write_text("\x1b[2mscanning local network…\x1b[0m\n")?;
            self.out.flush()?;
            let results = zero_core::discovery::scan(Duration::from_millis(300));
            self.apply_scan(results)?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/servers" {
            self.echo_committed(&text)?;
            self.print_servers()?;
            return Ok(Flow::Continue);
        }
        if let Some(rest) = trimmed.strip_prefix("/connect") {
            self.echo_committed(&text)?;
            let n = rest.trim().parse::<usize>().unwrap_or(0);
            self.connect_index(n)?;
            return Ok(Flow::Continue);
        }
        if let Some(rest) = trimmed.strip_prefix("/model") {
            self.echo_committed(&text)?;
            self.set_model(rest.trim())?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/clip" {
            self.echo_committed(&text)?;
            self.clip_last()?;
            return Ok(Flow::Continue);
        }

        self.echo_committed(&text)?;
        self.conv.push(Message::user(&text));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::User, &text);
        }

        // Disjoint field borrows let the sink write while the backend is read.
        let (reply, elapsed) = stream_reply(self.backend.as_ref(), &mut self.out, &self.conv)?;

        self.conv.push(Message::assistant(&reply));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::Assistant, &reply);
            let _ = log.record_turn_done(elapsed.as_millis());
        }
        self.last_reply = reply; // remembered for /clip

        // Honest, measured elapsed — dimmed, never an estimate.
        self.write_text(&format!("\n\x1b[2m  {}\x1b[0m\n", format_duration(elapsed)))?;
        Ok(Flow::Continue)
    }

    /// Copy the last assistant response to the system clipboard (`/clip`).
    fn clip_last(&mut self) -> io::Result<()> {
        if self.last_reply.trim().is_empty() {
            return self.write_text("\x1b[2mnothing to copy yet\x1b[0m\n");
        }
        let reply = self.last_reply.clone();
        let n = reply.chars().count();
        match (self.clipboard)(&reply) {
            Ok(()) => self.write_text(&format!("\x1b[2mcopied {n} chars to clipboard\x1b[0m\n")),
            Err(e) => self.write_text(&format!("\x1b[31mclip failed: {e}\x1b[0m\n")),
        }
    }

    // --- shell mode (!) ---------------------------------------------------

    /// Confirm-mode key handler for a pending dangerous shell command.
    fn handle_confirm_key(&mut self, key: Key) -> io::Result<Flow> {
        let cmd = self.pending_shell.take().expect("a command is pending");
        match key {
            Key::Char('y') | Key::Char('Y') => {
                self.write_text("\n")?;
                self.run_shell(&cmd)?;
            }
            _ => {
                self.write_text("\x1b[2mcancelled\x1b[0m\n")?;
                self.cursor_row = 0;
            }
        }
        Ok(Flow::Continue)
    }

    /// Run `cmd` via `sh -c`, print its output, exit code, and measured time.
    fn run_shell(&mut self, cmd: &str) -> io::Result<()> {
        let sw = Stopwatch::start();
        match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                if !stdout.is_empty() {
                    self.write_text(&stdout)?;
                    if !stdout.ends_with('\n') {
                        self.write_text("\n")?;
                    }
                }
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stderr.is_empty() {
                    self.write_text(&format!("\x1b[31m{stderr}\x1b[0m"))?;
                    if !stderr.ends_with('\n') {
                        self.write_text("\n")?;
                    }
                }
                let code = out.status.code().unwrap_or(-1);
                if let Some(log) = self.log.as_mut() {
                    let _ = log.record_meta("shell", cmd);
                }
                self.write_text(&format!(
                    "\x1b[2m[exit {code} · {}]\x1b[0m\n",
                    sw.elapsed_human()
                ))?;
            }
            Err(e) => self.write_text(&format!("\x1b[31m[shell error: {e}]\x1b[0m\n"))?,
        }
        self.cursor_row = 0;
        Ok(())
    }

    // --- network discovery (/scan, /connect, /servers) -------------------

    /// Record scan results, refresh the saved server list, and print a picker.
    fn apply_scan(&mut self, results: Vec<Discovered>) -> io::Result<()> {
        if let Some(path) = &self.servers_path {
            let mut store = ServerStore::load(path).unwrap_or_default();
            for d in &results {
                store.upsert(d);
            }
            let _ = store.save(path);
        }
        self.scan_results = results;
        let targets = self.connect_targets();
        if targets.is_empty() {
            self.write_text(
                "\x1b[2mno OpenAI-compatible servers found (loopback or LAN)\x1b[0m\n",
            )?;
            return Ok(());
        }
        // Each model on each server is its own connectable row.
        let mut out = String::from("\x1b[1mdiscovered models\x1b[0m\n");
        for (i, (url, model)) in targets.iter().enumerate() {
            let m = if model.is_empty() {
                "(no models)"
            } else {
                model
            };
            out.push_str(&format!("  {}) {m}  \x1b[2m{url}\x1b[0m\n", i + 1));
        }
        out.push_str("\x1b[2muse /connect <n> to attach\x1b[0m\n");
        self.write_text(&out)?;
        Ok(())
    }

    /// Flatten the last scan into one `(url, model)` row per model — so a host
    /// serving several models offers several pick choices.
    fn connect_targets(&self) -> Vec<(String, String)> {
        let mut targets = Vec::new();
        for d in &self.scan_results {
            if d.models.is_empty() {
                targets.push((d.base_url.clone(), String::new()));
            } else {
                for m in &d.models {
                    targets.push((d.base_url.clone(), m.clone()));
                }
            }
        }
        targets
    }

    /// Attach to the nth discovered (server, model) pair (1-based): swap the
    /// live backend and persist the choice to the config file.
    fn connect_index(&mut self, n: usize) -> io::Result<()> {
        let targets = self.connect_targets();
        let Some((url, model)) = n.checked_sub(1).and_then(|i| targets.get(i)).cloned() else {
            self.write_text("\x1b[31mno such entry — run /scan first\x1b[0m\n")?;
            return Ok(());
        };
        self.config.base_url = Some(url);
        self.config.model = model;
        self.rebuild_backend();
        self.write_text(&format!(
            "\x1b[2mconnected: {}\x1b[0m\n",
            self.config.summary()
        ))
    }

    /// Set the model on the current endpoint (`/model <name>`), or show it.
    fn set_model(&mut self, name: &str) -> io::Result<()> {
        if name.is_empty() {
            let cur = if self.config.model.is_empty() {
                "(none)"
            } else {
                &self.config.model
            };
            return self.write_text(&format!("\x1b[2mmodel: {cur}\x1b[0m\n"));
        }
        self.config.model = name.to_string();
        self.rebuild_backend();
        self.write_text(&format!("\x1b[2mmodel set: {}\x1b[0m\n", self.config.model))
    }

    /// Rebuild the live backend from the current config and persist it.
    fn rebuild_backend(&mut self) {
        if let Some(b) = OpenAiBackend::from_config(&self.config) {
            self.backend = Box::new(b);
        }
        if let Some(path) = &self.config_path {
            let _ = self.config.save(path);
        }
        self.info = self.config.summary();
    }

    /// List the locally-saved servers (`~/.zero/servers.json`).
    fn print_servers(&mut self) -> io::Result<()> {
        let store = self
            .servers_path
            .as_ref()
            .and_then(|p| ServerStore::load(p).ok())
            .unwrap_or_default();
        if store.servers.is_empty() {
            self.write_text("\x1b[2mno saved servers — run /scan\x1b[0m\n")?;
            return Ok(());
        }
        self.write_text("\x1b[1msaved servers\x1b[0m\n")?;
        for s in &store.servers {
            let model = s.model.as_deref().unwrap_or("(none selected)");
            self.write_text(&format!("  {model}  \x1b[2m{}\x1b[0m\n", s.base_url))?;
        }
        Ok(())
    }

    // --- reverse history search (^R) -------------------------------------

    fn enter_search(&mut self) -> io::Result<()> {
        self.clear_input_block()?;
        self.cursor_row = 0;
        let s = Search::default();
        self.render_search(&s)?;
        self.search = Some(s);
        Ok(())
    }

    fn handle_search_key(&mut self, key: Key) -> io::Result<Flow> {
        let mut s = self.search.take().expect("in search mode");
        match key {
            Key::Enter => {
                // Accept the match into the line (does not submit immediately).
                if let Some(i) = s.idx {
                    let hit = self.editor.history()[i].clone();
                    self.editor.set_text(&hit);
                }
                self.clear_input_block()?;
                return Ok(Flow::Continue);
            }
            Key::Esc | Key::Ctrl('c') | Key::Ctrl('g') => {
                // Cancel: leave the line untouched.
                self.clear_input_block()?;
                return Ok(Flow::Continue);
            }
            Key::Ctrl('r') => s.idx = self.search_from(&s.query, s.idx),
            Key::Char(c) => {
                s.query.push(c);
                s.idx = self.search_from(&s.query, None);
            }
            Key::Backspace => {
                s.query.pop();
                s.idx = self.search_from(&s.query, None);
            }
            _ => {}
        }
        self.render_search(&s)?;
        self.search = Some(s);
        Ok(Flow::Continue)
    }

    /// Most recent history index whose entry contains `query`, searching strictly
    /// older than `before` when given (for repeated `^R`).
    fn search_from(&self, query: &str, before: Option<usize>) -> Option<usize> {
        if query.is_empty() {
            return None;
        }
        let hist = self.editor.history();
        let start = match before {
            Some(0) => return None,
            Some(b) => b - 1,
            None => hist.len().checked_sub(1)?,
        };
        (0..=start).rev().find(|&i| hist[i].contains(query))
    }

    fn render_search(&mut self, s: &Search) -> io::Result<()> {
        let shown = s
            .idx
            .map(|i| self.editor.history()[i].as_str())
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("");
        // Match readline: "(failed reverse-i-search)" when a non-empty query has
        // no match.
        let label = if s.idx.is_none() && !s.query.is_empty() {
            "failed reverse-i-search"
        } else {
            "reverse-i-search"
        };
        let line = format!(
            "\r\x1b[K\x1b[2m({label})`\x1b[0m{}\x1b[2m`:\x1b[0m {}",
            s.query, shown
        );
        self.out.write_all(line.as_bytes())?;
        self.out.flush()
    }

    // --- rendering --------------------------------------------------------

    /// Print the committed input line(s) as static scrollback.
    fn echo_committed(&mut self, text: &str) -> io::Result<()> {
        self.clear_input_block()?;
        self.cursor_row = 0;
        for (i, line) in text.split('\n').enumerate() {
            let prefix = if i == 0 {
                self.prompt.clone()
            } else {
                CONT.to_string()
            };
            self.write_text(&format!("{prefix}{line}\n"))?;
        }
        Ok(())
    }

    /// Move to the top-left of the current input block and clear downward.
    fn clear_input_block(&mut self) -> io::Result<()> {
        if self.cursor_row > 0 {
            write!(self.out, "\x1b[{}A", self.cursor_row)?;
        }
        self.out.write_all(b"\r\x1b[J")?;
        Ok(())
    }

    /// Redraw the (possibly multi-row) input block and position the cursor.
    fn redraw_input(&mut self) -> io::Result<()> {
        self.clear_input_block()?;
        let text = self.editor.text();
        let lines: Vec<&str> = text.split('\n').collect();
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                self.out.write_all(b"\r\n")?;
            }
            let prefix = if i == 0 { self.prompt.as_str() } else { CONT };
            write!(self.out, "{prefix}{line}")?;
        }
        // Cursor is now at the end of the last row; move it to its logical spot.
        let (trow, tcol) = self.cursor_rowcol();
        let prefix_w = if trow == 0 {
            self.prompt.chars().count()
        } else {
            CONT.chars().count()
        };
        let bottom = lines.len() - 1;
        if bottom > trow {
            write!(self.out, "\x1b[{}A", bottom - trow)?;
        }
        self.out.write_all(b"\r")?;
        // The prompt/continuation prefix is always >= 1 column wide, so the
        // cursor always needs advancing past it.
        let col = prefix_w + tcol;
        write!(self.out, "\x1b[{col}C")?;
        self.cursor_row = trow;
        self.out.flush()
    }

    /// (row, column-in-chars) of the cursor within the input text.
    fn cursor_rowcol(&self) -> (usize, usize) {
        let cur = self.editor.cursor();
        let (mut row, mut col) = (0usize, 0usize);
        for (i, ch) in self.editor.text().chars().enumerate() {
            if i >= cur {
                break;
            }
            if ch == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    fn print_banner(&mut self) -> io::Result<()> {
        let banner = format!(
            "\x1b[1m{}\x1b[0m — local-first AI terminal  \x1b[2m({})\x1b[0m\n\
             \x1b[2m/help for commands · ! for shell · ^C twice to quit\x1b[0m\n\n",
            zero_core::brand::name(),
            self.backend.name()
        );
        self.write_text(&banner)
    }

    fn print_help(&mut self) -> io::Result<()> {
        // Hand-aligned single column: bold section headers, cyan keys, dim
        // descriptions. `H` marks headers, `K` marks key rows.
        const ROWS: &[(char, &str, &str)] = &[
            ('H', "Commands", ""),
            ('K', "/help", "show this help"),
            ('K', "/quit  /exit", "leave Zero"),
            ('K', "/config", "show the active backend and model"),
            ('K', "/scan", "find model servers on your network"),
            ('K', "/connect <n>", "attach to a discovered model"),
            ('K', "/model <name>", "switch model on the current endpoint"),
            ('K', "/servers", "list saved servers"),
            ('K', "/clip", "copy the last response to the clipboard"),
            (
                'K',
                "!<cmd>",
                "run a shell command — dangerous ones ask first",
            ),
            ('H', "Editing", ""),
            ('K', "^A  ^E   Home End", "start / end of line"),
            ('K', "^B  ^F", "back / forward one char"),
            ('K', "⌥←  ⌥→", "back / forward one word"),
            ('K', "^W", "delete the word before the cursor"),
            ('K', "^U  ^K", "kill to start / end of line"),
            ('K', "^L", "clear the screen"),
            ('H', "Multiline & history", ""),
            ('K', "^J", "insert a newline (works everywhere)"),
            (
                'K',
                "⇧⏎  ⌥⏎",
                "insert a newline (on terminals that support it)",
            ),
            ('K', "⏎", "submit"),
            ('K', "↑  ↓", "move between input lines, else recall history"),
            ('K', "^R", "reverse history search"),
            ('H', "Exit", ""),
            ('K', "Esc Esc", "clear the line"),
            (
                'K',
                "^C",
                "clear the line; on an empty line, ^C again quits",
            ),
        ];
        let mut out = String::from("\n");
        for (kind, key, desc) in ROWS {
            match kind {
                'H' => out.push_str(&format!("\x1b[1m{key}\x1b[0m\n")),
                _ => out.push_str(&format!(
                    "  \x1b[36m{key:<18}\x1b[0m \x1b[2m{desc}\x1b[0m\n"
                )),
            }
        }
        out.push('\n');
        self.write_text(&out)
    }

    /// Write text, translating `\n` to `\r\n` for raw mode.
    fn write_text(&mut self, s: &str) -> io::Result<()> {
        write_raw(&mut self.out, s)
    }
}

/// Copy `text` to the system clipboard via the first available tool
/// (`pbcopy` on macOS, `wl-copy`/`xclip` on Linux). Errors if none are found.
fn clipboard_copy(text: &str) -> io::Result<()> {
    const CANDIDATES: &[(&str, &[&str])] = &[
        ("pbcopy", &[]),
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
    ];
    copy_with(CANDIDATES, text)
}

/// Pipe `text` to the first `candidates` command that spawns. Factored out so
/// tests can pass a harmless command instead of the real system clipboard.
fn copy_with(candidates: &[(&str, &[&str])], text: &str) -> io::Result<()> {
    use std::process::{Command, Stdio};
    for (cmd, args) in candidates {
        match Command::new(cmd)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    stdin.write_all(text.as_bytes())?;
                }
                let _ = child.wait();
                return Ok(());
            }
            Err(_) => continue, // tool not installed; try the next
        }
    }
    Err(io::Error::other(
        "no clipboard tool found (install pbcopy, wl-copy, or xclip)",
    ))
}

/// Translate `\n` → `\r\n` (raw mode needs the carriage return) and write.
fn write_raw<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    if s.contains('\n') {
        w.write_all(s.replace('\n', "\r\n").as_bytes())
    } else {
        w.write_all(s.as_bytes())
    }?;
    Ok(())
}

/// Stream a reply from `backend`, echoing tokens live, returning the full text
/// plus measured elapsed time. Free function so the caller can pass disjoint
/// `&mut out` and `&backend` borrows.
fn stream_reply<W: Write>(
    backend: &dyn Backend,
    out: &mut W,
    conv: &Conversation,
) -> io::Result<(String, Duration)> {
    let mut reply = String::new();
    let mut io_err: Option<io::Error> = None;
    // `reply` keeps the raw markdown (for the model + /clip); `md` renders it to
    // ANSI for live display so `**bold**`, `*italic*`, `` `code` `` look right.
    let mut md = crate::markdown::MarkdownStream::new();

    write_raw(
        out,
        &format!("\x1b[2m{}›\x1b[0m ", zero_core::brand::slug()),
    )?;
    out.flush()?;

    let sw = Stopwatch::start();
    let stream_res = backend.stream(conv, &mut |ev| {
        if io_err.is_some() {
            return;
        }
        if let StreamEvent::Token(t) = ev {
            reply.push_str(&t);
            let rendered = md.feed(&t);
            if let Err(e) = write_raw(out, &rendered).and_then(|()| out.flush()) {
                io_err = Some(e);
            }
        }
    });
    let elapsed = sw.elapsed();

    if let Some(e) = io_err {
        return Err(e);
    }
    // Close any styling left open at the end of the response.
    write_raw(out, &md.finish())?;
    if let Err(e) = stream_res {
        write_raw(out, &format!("\n\x1b[31m[{e}]\x1b[0m"))?;
    }
    Ok((reply, elapsed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zero_core::backend::{BackendError, StubBackend};

    /// Input that yields a fixed byte script on the first read, then 0 forever.
    struct ScriptedInput {
        bytes: Vec<u8>,
        done: bool,
    }
    impl ScriptedInput {
        fn new(bytes: &[u8]) -> Self {
            ScriptedInput {
                bytes: bytes.to_vec(),
                done: false,
            }
        }
    }
    impl Input for ScriptedInput {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.done {
                return Ok(0);
            }
            let n = self.bytes.len().min(buf.len());
            buf[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes.drain(..n);
            if self.bytes.is_empty() {
                self.done = true;
            }
            Ok(n)
        }
    }

    /// Input yielding a sequence of chunks (one per read), then 0 forever.
    struct MultiInput {
        chunks: std::collections::VecDeque<Vec<u8>>,
    }
    impl MultiInput {
        fn new(chunks: &[&[u8]]) -> Self {
            MultiInput {
                chunks: chunks.iter().map(|c| c.to_vec()).collect(),
            }
        }
    }
    impl Input for MultiInput {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let Some(chunk) = self.chunks.pop_front() else {
                return Ok(0);
            };
            let n = chunk.len().min(buf.len());
            buf[..n].copy_from_slice(&chunk[..n]);
            Ok(n)
        }
    }

    /// A writer that succeeds `ok` times then fails — to exercise I/O errors.
    struct FlakyWriter {
        ok: usize,
    }
    impl Write for FlakyWriter {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            if self.ok == 0 {
                return Err(io::Error::other("write failed"));
            }
            self.ok -= 1;
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A backend that errors mid-stream, to exercise the error path.
    struct FailBackend;
    impl Backend for FailBackend {
        fn name(&self) -> &str {
            "fail"
        }
        fn stream(
            &self,
            _conv: &Conversation,
            _sink: &mut dyn FnMut(StreamEvent),
        ) -> Result<(), BackendError> {
            Err(BackendError("boom".to_string()))
        }
    }

    fn app(script: &[u8]) -> App<ScriptedInput, Vec<u8>> {
        App::new(
            ScriptedInput::new(script),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        )
    }

    fn rendered(a: &App<ScriptedInput, Vec<u8>>) -> String {
        String::from_utf8(a.out.clone()).unwrap()
    }

    #[test]
    fn scripted_input_returns_zero_after_exhaustion() {
        let mut si = ScriptedInput::new(b"ab");
        let mut b = [0u8; 8];
        assert_eq!(si.read(&mut b).unwrap(), 2);
        assert_eq!(si.read(&mut b).unwrap(), 0);
    }

    #[test]
    fn multi_input_returns_zero_when_drained() {
        let mut mi = MultiInput::new(&[b"x"]);
        let mut b = [0u8; 8];
        assert_eq!(mi.read(&mut b).unwrap(), 1);
        assert_eq!(mi.read(&mut b).unwrap(), 0);
    }

    #[test]
    fn write_raw_translates_newlines() {
        let mut buf = Vec::new();
        write_raw(&mut buf, "a\nb\n").unwrap();
        assert_eq!(buf, b"a\r\nb\r\n");
    }

    #[test]
    fn write_raw_passthrough_without_newline() {
        let mut buf = Vec::new();
        write_raw(&mut buf, "plain").unwrap();
        assert_eq!(buf, b"plain");
    }

    #[test]
    fn stream_reply_collects_text_and_writes_label() {
        let mut conv = Conversation::new();
        conv.push(Message::user("ping"));
        let mut out = Vec::new();
        let (reply, _) = stream_reply(&StubBackend::instant(), &mut out, &conv).unwrap();
        assert!(reply.contains("ping"));
        let painted = String::from_utf8(out).unwrap();
        assert!(painted.contains("›"));
        assert!(painted.contains("ping"));
    }

    #[test]
    fn stream_reply_renders_backend_error() {
        let conv = Conversation::new();
        let mut out = Vec::new();
        assert_eq!(FailBackend.name(), "fail");
        let (reply, _) = stream_reply(&FailBackend, &mut out, &conv).unwrap();
        assert!(reply.is_empty());
        assert!(String::from_utf8(out).unwrap().contains("boom"));
    }

    #[test]
    fn stream_reply_surfaces_label_write_error() {
        let conv = Conversation::new();
        let mut out = FlakyWriter { ok: 0 }; // even the label write fails
        assert!(stream_reply(&StubBackend::instant(), &mut out, &conv).is_err());
    }

    #[test]
    fn shell_mode_stderr_without_trailing_newline() {
        let mut a = app(b"");
        type_str(&mut a, "!printf oops >&2");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("oops"));
    }

    #[test]
    fn stream_reply_surfaces_writer_error_mid_stream() {
        let mut conv = Conversation::new();
        conv.push(Message::user("ping"));
        let mut out = FlakyWriter { ok: 1 };
        let err = stream_reply(&StubBackend::instant(), &mut out, &conv).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn run_streams_reply_then_quits() {
        let mut a = app(b"hello\r/quit\r");
        a.run().unwrap();
        let out = rendered(&a);
        assert!(out.contains("local-first AI terminal"));
        assert!(out.contains("hello"));
        assert!(out.contains("You said"));
        assert_eq!(a.conv.len(), 2);
    }

    #[test]
    fn run_propagates_output_errors() {
        let mut a = App::new(
            ScriptedInput::new(b"hi\r"),
            FlakyWriter { ok: 0 },
            Box::new(StubBackend::instant()),
            None,
        );
        assert!(a.run().is_err());
    }

    #[test]
    fn run_handles_poll_timeout_and_multiple_reads() {
        let mut a = App::new(
            MultiInput::new(&[b"hello", b"", b"\r/quit\r"]),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        );
        a.run().unwrap();
        assert_eq!(a.conv.len(), 2);
    }

    #[test]
    fn run_with_session_log_records_the_turn() {
        let dir =
            std::env::temp_dir().join(format!("zero-app-test-{}", zero_core::clock::unix_millis()));
        let (log, path) = SessionLog::create_in(&dir).unwrap();
        let mut a = App::new(
            ScriptedInput::new(b"hi\r/quit\r"),
            Vec::new(),
            Box::new(StubBackend::instant()),
            Some(log),
        );
        a.run().unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("\"role\":\"user\""));
        assert!(contents.contains("\"role\":\"assistant\""));
        assert!(contents.contains("turn_done"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn help_command_prints_help_without_streaming() {
        let mut a = app(b"/help\r/quit\r");
        a.run().unwrap();
        let out = rendered(&a);
        assert!(out.contains("Commands"));
        assert!(out.contains("reverse history search"));
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn clip_copies_last_reply_via_injected_clipboard() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let captured = Rc::new(RefCell::new(String::new()));
        let sink = captured.clone();
        let mut a = app(b"");
        a.set_clipboard(Box::new(move |s| {
            *sink.borrow_mut() = s.to_string();
            Ok(())
        }));
        // A turn produces a reply, then /clip copies it.
        type_str(&mut a, "ping");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.last_reply.contains("ping"));
        type_str(&mut a, "/clip");
        a.dispatch(Key::Enter).unwrap();
        assert_eq!(*captured.borrow(), a.last_reply);
        assert!(rendered(&a).contains("copied"));
    }

    #[test]
    fn copy_with_pipes_to_a_command() {
        // `cat` consumes stdin and exits 0 — a harmless stand-in for a clipboard.
        assert!(copy_with(&[("cat", &[])], "hello").is_ok());
    }

    #[test]
    fn copy_with_errors_when_no_tool_exists() {
        assert!(copy_with(&[("zero-no-such-binary-xyz", &[])], "x").is_err());
    }

    #[test]
    fn clip_with_nothing_says_so() {
        let mut a = app(b"");
        type_str(&mut a, "/clip");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("nothing to copy"));
    }

    #[test]
    fn clip_reports_clipboard_failure() {
        let mut a = app(b"");
        a.set_clipboard(Box::new(|_| Err(io::Error::other("no tool"))));
        type_str(&mut a, "hey");
        a.dispatch(Key::Enter).unwrap();
        type_str(&mut a, "/clip");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("clip failed"));
    }

    #[test]
    fn assistant_reply_is_markdown_rendered() {
        // A backend that emits a bold span.
        struct BoldBackend;
        impl Backend for BoldBackend {
            fn name(&self) -> &str {
                "bold"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), BackendError> {
                sink(StreamEvent::Token("**hi**".to_string()));
                sink(StreamEvent::Done(zero_core::backend::StopReason::EndTurn));
                Ok(())
            }
        }
        let mut conv = Conversation::new();
        conv.push(Message::user("x"));
        let mut out = Vec::new();
        let (reply, _) = stream_reply(&BoldBackend, &mut out, &conv).unwrap();
        assert_eq!(reply, "**hi**"); // raw markdown preserved for model/clip
        let painted = String::from_utf8(out).unwrap();
        assert!(painted.contains("\x1b[1mhi")); // displayed bold
        assert!(!painted.contains("**")); // asterisks not shown
    }

    #[test]
    fn config_command_shows_info() {
        let mut a = app(b"");
        a.set_info("qwen @ http://gx10:8000");
        type_str(&mut a, "/config");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("qwen @ http://gx10:8000"));
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn config_command_without_info_says_stub() {
        let mut a = app(b"");
        type_str(&mut a, "/config");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("stub"));
    }

    fn disc(url: &str, models: &[&str]) -> Discovered {
        Discovered {
            base_url: url.to_string(),
            models: models.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn apply_scan_lists_results_and_saves_servers() {
        let dir =
            std::env::temp_dir().join(format!("zero-scan-{}", zero_core::clock::unix_millis()));
        let spath = dir.join("servers.json");
        let mut a = app(b"");
        a.set_config(Config::default(), None, Some(spath.clone()));
        a.apply_scan(vec![
            disc("http://h:8000", &["qwen"]),
            disc("http://h:1234", &["llama"]),
        ])
        .unwrap();
        let out = rendered(&a);
        assert!(out.contains("discovered models"));
        assert!(out.contains("qwen"));
        assert!(out.contains("/connect"));
        // Persisted to the server store.
        let store = zero_core::servers::ServerStore::load(&spath).unwrap();
        assert_eq!(store.servers.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apply_scan_reports_when_empty() {
        let mut a = app(b"");
        a.apply_scan(Vec::new()).unwrap();
        assert!(rendered(&a).contains("no OpenAI-compatible servers"));
    }

    #[test]
    fn connect_index_swaps_backend_and_persists_config() {
        let dir =
            std::env::temp_dir().join(format!("zero-conn-{}", zero_core::clock::unix_millis()));
        let cpath = dir.join("config.json");
        let mut a = app(b"");
        a.set_config(Config::default(), Some(cpath.clone()), None);
        a.apply_scan(vec![disc("http://gx10:8000", &["qwen"])])
            .unwrap();
        a.connect_index(1).unwrap();
        // Backend is now the OpenAI one; its name carries the model + url.
        assert!(a.backend.name().contains("qwen"));
        assert!(a.backend.name().contains("gx10"));
        // Config persisted for next launch.
        let saved = Config::load(&cpath).unwrap();
        assert_eq!(saved.base_url.as_deref(), Some("http://gx10:8000"));
        assert_eq!(saved.model, "qwen");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn connect_index_out_of_range_is_reported() {
        let mut a = app(b"");
        a.connect_index(5).unwrap();
        assert!(rendered(&a).contains("no such entry"));
    }

    #[test]
    fn multi_model_host_offers_one_entry_per_model() {
        let mut a = app(b"");
        // One host serving two models → two connectable rows.
        a.apply_scan(vec![disc("http://h:8000", &["qwen", "llama"])])
            .unwrap();
        let targets = a.connect_targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0], ("http://h:8000".into(), "qwen".into()));
        assert_eq!(targets[1], ("http://h:8000".into(), "llama".into()));
        // Connect to the second model specifically.
        a.connect_index(2).unwrap();
        assert!(a.backend.name().contains("llama"));
    }

    #[test]
    fn model_command_switches_model_on_current_endpoint() {
        let mut a = app(b"");
        a.set_config(
            Config {
                base_url: Some("http://h:8000".into()),
                model: "qwen".into(),
                ..Config::default()
            },
            None,
            None,
        );
        a.set_model("llama-3.1-8b").unwrap();
        assert_eq!(a.config.model, "llama-3.1-8b");
        assert!(a.backend.name().contains("llama-3.1-8b"));
        // No-arg form reports the current model.
        a.out.clear();
        a.set_model("").unwrap();
        assert!(rendered(&a).contains("llama-3.1-8b"));
    }

    #[test]
    fn server_with_no_models_still_connectable() {
        let mut a = app(b"");
        a.apply_scan(vec![disc("http://h:8000", &[])]).unwrap();
        assert!(rendered(&a).contains("(no models)"));
        let targets = a.connect_targets();
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].1, ""); // empty model
    }

    #[test]
    fn servers_command_lists_saved_and_handles_empty() {
        let dir =
            std::env::temp_dir().join(format!("zero-srvlist-{}", zero_core::clock::unix_millis()));
        let spath = dir.join("servers.json");
        let mut a = app(b"");
        a.set_config(Config::default(), None, Some(spath.clone()));
        // Empty first.
        a.print_servers().unwrap();
        assert!(rendered(&a).contains("no saved servers"));
        // After a scan, it lists them.
        a.apply_scan(vec![disc("http://h:8000", &["qwen"])])
            .unwrap();
        a.out.clear();
        a.print_servers().unwrap();
        let out = rendered(&a);
        assert!(out.contains("saved servers"));
        assert!(out.contains("qwen"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bare_exit_nudges_instead_of_quitting() {
        let mut a = app(b"");
        type_str(&mut a, "exit");
        assert_eq!(a.dispatch(Key::Enter).unwrap(), Flow::Continue);
        assert!(a.ctrl_c_armed);
        assert_eq!(a.conv.len(), 0); // not sent to the model
        assert!(rendered(&a).contains("press ^C again to exit"));
        // Now ^C actually exits (armed + empty line).
        assert_eq!(a.dispatch(Key::Ctrl('c')).unwrap(), Flow::Quit);
    }

    #[test]
    fn bare_quit_is_also_nudged_case_insensitively() {
        let mut a = app(b"");
        type_str(&mut a, "QUIT");
        assert_eq!(a.dispatch(Key::Enter).unwrap(), Flow::Continue);
        assert!(a.ctrl_c_armed);
    }

    #[test]
    fn redraw_is_suppressed_during_search() {
        // The bug: a redraw after each search keystroke clobbered the search UI.
        let mut a = app(b"");
        a.dispatch(Key::Ctrl('r')).unwrap();
        a.out.clear();
        a.redraw_if_idle().unwrap();
        assert!(a.out.is_empty(), "must not draw over the search prompt");
    }

    #[test]
    fn redraw_is_suppressed_during_shell_confirm() {
        let mut a = app(b"");
        type_str(&mut a, "!rm -rf /tmp/zero-x");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.pending_shell.is_some());
        a.out.clear();
        a.redraw_if_idle().unwrap();
        assert!(a.out.is_empty(), "must not draw over the confirm prompt");
    }

    #[test]
    fn reverse_search_failed_state_renders() {
        let mut a = app(b""); // empty history
        a.dispatch(Key::Ctrl('r')).unwrap();
        a.dispatch(Key::Char('z')).unwrap(); // nothing matches
        assert!(a.search.as_ref().unwrap().idx.is_none());
        assert!(rendered(&a).contains("failed reverse-i-search"));
    }

    #[test]
    fn blank_submit_is_ignored() {
        let mut a = app(b"   \r/quit\r");
        a.run().unwrap();
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn editing_keys_mutate_the_line() {
        let mut a = app(b"");
        for c in "hello".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Home).unwrap();
        a.dispatch(Key::Right).unwrap();
        a.dispatch(Key::Delete).unwrap();
        assert_eq!(a.editor.text(), "hllo");
        a.dispatch(Key::End).unwrap();
        a.dispatch(Key::Backspace).unwrap();
        assert_eq!(a.editor.text(), "hll");
        a.dispatch(Key::Ctrl('u')).unwrap();
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn word_and_char_chords_move_cursor() {
        let mut a = app(b"");
        for c in "foo bar".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::WordLeft).unwrap();
        assert_eq!(a.editor.cursor(), 4);
        a.dispatch(Key::Ctrl('b')).unwrap();
        assert_eq!(a.editor.cursor(), 3);
        a.dispatch(Key::Ctrl('f')).unwrap();
        assert_eq!(a.editor.cursor(), 4);
        a.dispatch(Key::WordRight).unwrap();
        assert_eq!(a.editor.cursor(), 7);
    }

    #[test]
    fn shift_enter_inserts_newline_and_enter_submits() {
        let mut a = App::new(
            ScriptedInput::new(b"line1\x1b[13;2uline2\r/quit\r"),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        );
        a.run().unwrap();
        // The single submitted message spans two lines.
        assert_eq!(a.conv.len(), 2);
        assert_eq!(a.conv.messages[0].content, "line1\nline2");
    }

    #[test]
    fn ctrl_c_needs_two_presses_to_exit() {
        let mut a = app(b"");
        // Empty line: first ^C arms (Continue), second exits.
        assert_eq!(a.dispatch(Key::Ctrl('c')).unwrap(), Flow::Continue);
        assert!(a.ctrl_c_armed);
        assert_eq!(a.dispatch(Key::Ctrl('c')).unwrap(), Flow::Quit);
    }

    #[test]
    fn ctrl_c_clears_a_nonempty_line_without_exiting() {
        let mut a = app(b"");
        for c in "draft".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        assert_eq!(a.dispatch(Key::Ctrl('c')).unwrap(), Flow::Continue);
        assert!(a.editor.is_empty());
        assert!(!a.ctrl_c_armed);
    }

    #[test]
    fn other_key_disarms_ctrl_c() {
        let mut a = app(b"");
        a.dispatch(Key::Ctrl('c')).unwrap(); // arm
        a.dispatch(Key::Char('x')).unwrap(); // disarm
        assert!(!a.ctrl_c_armed);
        assert_eq!(a.dispatch(Key::Ctrl('c')).unwrap(), Flow::Continue); // re-arm, not quit
    }

    #[test]
    fn double_esc_clears_the_line() {
        let mut a = app(b"");
        for c in "junk".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Esc).unwrap();
        assert!(a.esc_pending);
        assert!(!a.editor.is_empty()); // single esc does nothing yet
        a.dispatch(Key::Esc).unwrap();
        assert!(a.editor.is_empty());
        assert!(!a.esc_pending);
    }

    #[test]
    fn single_esc_does_not_clear() {
        let mut a = app(b"");
        for c in "keep".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Esc).unwrap();
        a.dispatch(Key::Char('!')).unwrap(); // any other key cancels the esc latch
        assert!(!a.esc_pending);
        assert_eq!(a.editor.text(), "keep!");
    }

    #[test]
    fn lone_esc_press_flushed_on_timeout() {
        // Chunk 1 is a bare ESC (decoder leaves it pending), then a poll timeout
        // flushes it as Esc; chunk 2 quits. Two ESC presses clear nothing here,
        // but this proves the timeout path emits Esc without hanging.
        let mut a = App::new(
            MultiInput::new(&[b"\x1b", b"", b"/quit\r"]),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        );
        a.run().unwrap();
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn reverse_search_finds_and_accepts_history() {
        let mut a = app(b"");
        // Seed history.
        for line in ["cargo test", "git status", "cargo build"] {
            for c in line.chars() {
                a.dispatch(Key::Char(c)).unwrap();
            }
            a.dispatch(Key::Enter).unwrap();
        }
        a.dispatch(Key::Ctrl('r')).unwrap();
        assert!(a.search.is_some());
        for c in "cargo".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        // Most recent "cargo" match is "cargo build".
        let idx = a.search.as_ref().unwrap().idx.unwrap();
        assert_eq!(a.editor.history()[idx], "cargo build");
        // ^R again steps to the older "cargo test".
        a.dispatch(Key::Ctrl('r')).unwrap();
        let idx2 = a.search.as_ref().unwrap().idx.unwrap();
        assert_eq!(a.editor.history()[idx2], "cargo test");
        // Enter accepts into the line and exits search.
        a.dispatch(Key::Enter).unwrap();
        assert!(a.search.is_none());
        assert_eq!(a.editor.text(), "cargo test");
    }

    #[test]
    fn reverse_search_escape_cancels_without_changing_line() {
        let mut a = app(b"");
        for c in "deploy prod".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Enter).unwrap(); // history has "deploy prod"
        a.dispatch(Key::Ctrl('r')).unwrap();
        for c in "deploy".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Esc).unwrap(); // cancel
        assert!(a.search.is_none());
        assert!(a.editor.is_empty()); // line untouched (was cleared by submit)
    }

    #[test]
    fn reverse_search_backspace_refines_query() {
        let mut a = app(b"");
        for line in ["alpha", "beta"] {
            for c in line.chars() {
                a.dispatch(Key::Char(c)).unwrap();
            }
            a.dispatch(Key::Enter).unwrap();
        }
        a.dispatch(Key::Ctrl('r')).unwrap();
        for c in "alph".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        assert!(a.search.as_ref().unwrap().idx.is_some());
        for _ in 0..4 {
            a.dispatch(Key::Backspace).unwrap();
        }
        // Empty query → no match.
        assert!(a.search.as_ref().unwrap().idx.is_none());
    }

    #[test]
    fn ctrl_k_and_ctrl_w_kill() {
        let mut a = app(b"");
        for c in "foo bar".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Ctrl('w')).unwrap();
        assert_eq!(a.editor.text(), "foo ");
        a.dispatch(Key::Home).unwrap();
        a.dispatch(Key::Ctrl('k')).unwrap();
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn search_ignores_unmapped_keys() {
        let mut a = app(b"");
        a.dispatch(Key::Char('x')).unwrap();
        a.dispatch(Key::Enter).unwrap();
        a.dispatch(Key::Ctrl('r')).unwrap();
        a.dispatch(Key::Left).unwrap(); // no-op inside search
        assert!(a.search.is_some());
    }

    #[test]
    fn reverse_search_at_oldest_stops() {
        let mut a = app(b"");
        a.dispatch(Key::Char('o')).unwrap();
        a.dispatch(Key::Char('k')).unwrap();
        a.dispatch(Key::Enter).unwrap();
        a.dispatch(Key::Ctrl('r')).unwrap();
        a.dispatch(Key::Char('o')).unwrap();
        assert_eq!(a.search.as_ref().unwrap().idx, Some(0));
        a.dispatch(Key::Ctrl('r')).unwrap(); // nothing older than index 0
        assert_eq!(a.search.as_ref().unwrap().idx, None);
    }

    #[test]
    fn multiline_session_redraws_across_rows() {
        // Type "a", Shift+Enter, "b", Up (to row 0), then three ^C (clear, arm,
        // quit). Exercises multi-row clear + cursor move-up across loop redraws.
        let mut a = App::new(
            MultiInput::new(&[
                b"a",
                b"\x1b[13;2u",
                b"b",
                b"\x1b[A",
                b"\x03",
                b"\x03",
                b"\x03",
            ]),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        );
        a.run().unwrap();
        assert_eq!(a.conv.len(), 0);
    }

    fn type_str(a: &mut App<ScriptedInput, Vec<u8>>, s: &str) {
        for c in s.chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
    }

    #[test]
    fn shell_mode_runs_a_safe_command() {
        let mut a = app(b"");
        type_str(&mut a, "!echo zero-shell-ok");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        assert!(out.contains("zero-shell-ok"));
        assert!(out.contains("[exit 0"));
        assert!(a.pending_shell.is_none());
        assert_eq!(a.conv.len(), 0); // shell is not a model turn
    }

    #[test]
    fn shell_mode_dangerous_command_requires_confirmation() {
        let mut a = app(b"");
        type_str(&mut a, "!rm -rf /tmp/zero-does-not-exist-xyz");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.pending_shell.is_some());
        assert!(rendered(&a).contains("run anyway?"));
    }

    #[test]
    fn shell_mode_cancel_does_not_run() {
        // Create a real temp dir; cancelling must leave it intact.
        let dir = std::env::temp_dir().join(format!(
            "zero-shell-cancel-{}",
            zero_core::clock::unix_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = app(b"");
        type_str(&mut a, &format!("!rm -rf {}", dir.display()));
        a.dispatch(Key::Enter).unwrap();
        a.dispatch(Key::Char('n')).unwrap(); // decline
        assert!(a.pending_shell.is_none());
        assert!(dir.exists(), "cancelled command must not have run");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shell_mode_confirm_runs_the_command() {
        // Confirming deletes our own throwaway temp dir — real, but harmless.
        let dir = std::env::temp_dir().join(format!(
            "zero-shell-accept-{}",
            zero_core::clock::unix_millis()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(dir.exists());
        let mut a = app(b"");
        type_str(&mut a, &format!("!rm -rf {}", dir.display()));
        a.dispatch(Key::Enter).unwrap();
        a.dispatch(Key::Char('y')).unwrap(); // confirm
        assert!(a.pending_shell.is_none());
        assert!(!dir.exists(), "confirmed command should have run");
        assert!(rendered(&a).contains("[exit"));
    }

    #[test]
    fn shell_mode_prints_stdout_without_trailing_newline() {
        let mut a = app(b"");
        type_str(&mut a, "!printf zero-noeol");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        assert!(out.contains("zero-noeol"));
        assert!(out.contains("[exit 0"));
    }

    #[test]
    fn shell_mode_shows_stderr_and_nonzero_exit() {
        let mut a = app(b"");
        type_str(&mut a, "!ls /zero-definitely-missing-zzz");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        // Non-zero exit recorded; some stderr text was emitted.
        assert!(out.contains("[exit"));
        assert!(!out.contains("[exit 0"));
    }

    #[test]
    fn shell_mode_logs_the_command() {
        let dir = std::env::temp_dir().join(format!(
            "zero-shell-log-{}",
            zero_core::clock::unix_millis()
        ));
        let (log, path) = SessionLog::create_in(&dir).unwrap();
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Box::new(StubBackend::instant()),
            Some(log),
        );
        type_str(&mut a, "!echo logged");
        a.dispatch(Key::Enter).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("shell"));
        assert!(contents.contains("echo logged"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn shell_mode_empty_command_is_noop() {
        let mut a = app(b"");
        type_str(&mut a, "!");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.pending_shell.is_none());
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn ctrl_d_midline_deletes_char() {
        let mut a = app(b"");
        a.dispatch(Key::Char('x')).unwrap();
        a.dispatch(Key::Char('y')).unwrap();
        a.dispatch(Key::Home).unwrap();
        a.dispatch(Key::Ctrl('d')).unwrap();
        assert_eq!(a.editor.text(), "y");
    }

    #[test]
    fn ctrl_l_clears_screen() {
        let mut a = app(b"");
        a.dispatch(Key::Ctrl('l')).unwrap();
        assert!(rendered(&a).contains("\x1b[2J"));
    }

    #[test]
    fn unmapped_keys_are_noops() {
        let mut a = app(b"");
        assert_eq!(a.dispatch(Key::Tab).unwrap(), Flow::Continue);
        assert_eq!(a.dispatch(Key::Ctrl('z')).unwrap(), Flow::Continue);
    }

    #[test]
    fn history_keys_recall_previous_line() {
        let mut a = app(b"");
        for c in "first".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::Enter).unwrap();
        a.dispatch(Key::Up).unwrap();
        assert_eq!(a.editor.text(), "first");
        a.dispatch(Key::Down).unwrap();
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn up_down_move_between_input_lines_before_history() {
        let mut a = app(b"");
        for c in "top".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::ShiftEnter).unwrap();
        for c in "bottom".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        // Cursor on line 2; Up moves to line 1 (not history).
        a.dispatch(Key::Up).unwrap();
        let (row, _) = a.cursor_rowcol();
        assert_eq!(row, 0);
        assert_eq!(a.editor.text(), "top\nbottom"); // text unchanged
    }

    #[test]
    fn multiline_render_emits_continuation_and_cursor_moves() {
        let mut a = app(b"");
        for c in "ab".chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
        a.dispatch(Key::ShiftEnter).unwrap();
        a.dispatch(Key::Char('c')).unwrap();
        a.out.clear();
        a.redraw_input().unwrap();
        let out = rendered(&a);
        // Two rows: prompt+ab, CRLF, continuation+c.
        assert!(out.contains("\r\n"));
        assert!(out.contains("ab"));
        assert!(out.contains('c'));
    }

    #[test]
    fn redraw_moves_cursor_when_not_at_end() {
        let mut a = app(b"");
        a.dispatch(Key::Char('a')).unwrap();
        a.dispatch(Key::Char('b')).unwrap();
        a.dispatch(Key::Left).unwrap();
        a.out.clear();
        a.redraw_input().unwrap();
        assert!(rendered(&a).contains("\x1b["));
    }
}
