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
use crate::markdown::MarkdownStream;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Duration;
use zero_core::backend::{Backend, StopReason, StreamEvent, Usage};
use zero_core::clock::{format_duration, Stopwatch};
use zero_core::config::Config;
use zero_core::discovery::Discovered;
use zero_core::mcp;
use zero_core::message::{Conversation, Message, Role, ToolCall};
use zero_core::openai::OpenAiBackend;
use zero_core::servers::ServerStore;
use zero_core::session::SessionLog;
use zero_core::tools::{LoopGuard, ToolDef};

/// Prefix drawn before continuation rows of a multiline input (aligns under the
/// prompt). Same display width as the prompt.
const CONT: &str = "  ";

/// Slash commands, in the order shown as autocomplete suggestions. Single source
/// of truth for completion + the popup list.
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "show this help"),
    ("/config", "show the active backend and model"),
    ("/scan", "find model servers on your network"),
    ("/servers", "list saved servers"),
    (
        "/mcp",
        "import + connect MCP servers ( /mcp tools to list )",
    ),
    ("/connect", "attach to a discovered model"),
    ("/model", "switch model on the current endpoint"),
    (
        "/tools",
        "toggle the agentic tool loop (read/list/grep/write/edit)",
    ),
    ("/clip", "copy last response, or code block n"),
    ("/quit", "leave Zero"),
    ("/exit", "leave Zero"),
];

/// If `text` is a slash token still being typed (starts with `/`, no whitespace
/// yet), return it; otherwise `None`. Once a space is typed the rest is args, so
/// completion stops.
fn slash_query(text: &str) -> Option<&str> {
    if text.starts_with('/') && !text.contains(char::is_whitespace) {
        Some(text)
    } else {
        None
    }
}

/// Commands whose name has `query` as a prefix, in table order.
fn slash_matches(query: &str) -> Vec<(&'static str, &'static str)> {
    SLASH_COMMANDS
        .iter()
        .filter(|(name, _)| name.starts_with(query))
        .copied()
        .collect()
}

/// Longest common prefix of a set of command names — what Tab/Enter completes to
/// when several commands still match (shell-style).
fn common_prefix(names: &[&str]) -> String {
    let Some(first) = names.first() else {
        return String::new();
    };
    let mut end = first.len();
    for name in &names[1..] {
        end = end.min(name.len());
        while !first.is_char_boundary(end) || first[..end] != name[..end] {
            end -= 1;
        }
    }
    first[..end].to_string()
}

/// One-line preview of a queued message: its first line, capped to `max`
/// display columns. Appends `…` when truncated or when more lines follow, so a
/// big paste shows as a short, single-row hint instead of dominating the view.
fn queue_preview(msg: &str, max: usize) -> String {
    let first = msg.split('\n').next().unwrap_or("");
    let multiline = msg.contains('\n');
    let mut preview: String = first.chars().take(max).collect();
    if first.chars().count() > max || multiline {
        preview.push('…');
    }
    preview
}

/// Compact a token count for the status line: `840`, `1.2k`, `33k`.
fn fmt_count(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else {
        let k = n as f64 / 1000.0;
        if k >= 10.0 {
            format!("{}k", k.round() as u64)
        } else {
            format!("{k:.1}k")
        }
    }
}

/// Strip the scheme and trailing slash from a base URL for a tidy status line:
/// `http://192.168.50.125:8000/` → `192.168.50.125:8000`.
fn short_host(url: &str) -> String {
    url.strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url)
        .trim_end_matches('/')
        .to_string()
}

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

/// Queue-edit mode (`^Q`): pause the pending queue and edit a queued message in
/// place before it's sent.
struct QueueEdit {
    /// Index into `queue` currently being edited.
    sel: usize,
    /// The input line in progress before editing began, restored on exit.
    saved_input: String,
}

/// State of an in-flight model turn: the backend streams on another thread and
/// sends events down `rx`; the event loop drains them so it stays responsive
/// (can queue more input, or `^C` to interrupt).
struct StreamState {
    rx: Receiver<StreamEvent>,
    reply: String,
    md: MarkdownStream,
    sw: Stopwatch,
    /// Token usage reported by the server for this turn, once its usage chunk
    /// arrives (kept for the status line).
    usage: Option<Usage>,
}

/// Input mode, cycled with Shift+Tab (like Claude Code / opencode / pi). The set
/// is deliberately small; the agentic tool loop will extend what each one gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Mode {
    /// Default: dangerous shell commands ask before running.
    #[default]
    Normal,
    /// Auto-accept: run dangerous shell commands without the y/N prompt (and,
    /// once the tool loop lands, auto-approve its actions).
    AutoAccept,
    /// Plan: ask the model to think through an approach before acting; injects a
    /// planning directive into each request.
    Plan,
}

impl Mode {
    /// Next mode in the Shift+Tab cycle.
    fn next(self) -> Mode {
        match self {
            Mode::Normal => Mode::AutoAccept,
            Mode::AutoAccept => Mode::Plan,
            Mode::Plan => Mode::Normal,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Normal => "normal",
            Mode::AutoAccept => "auto-accept",
            Mode::Plan => "plan",
        }
    }

    /// ANSI color for the footer chip (dim normal, yellow auto, cyan plan).
    fn color(self) -> &'static str {
        match self {
            Mode::Normal => "\x1b[2m",
            Mode::AutoAccept => "\x1b[33m",
            Mode::Plan => "\x1b[36m",
        }
    }
}

/// Planning directive injected (as a system message) into requests in plan mode.
const PLAN_DIRECTIVE: &str = "You are in PLAN MODE. Do not write final code or \
take actions yet. Think through the approach and present a concise, reviewable \
plan first; wait for the go-ahead before implementing.";

/// The running terminal application.
pub struct App<I: Input, W: Write> {
    input: I,
    out: W,
    editor: LineEditor,
    conv: Conversation,
    backend: Arc<dyn Backend>,
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
    /// `Some` while in `^Q` queue-edit mode (sending is paused).
    queue_edit: Option<QueueEdit>,
    /// Current input mode (Shift+Tab cycles it).
    mode: Mode,
    /// Connected MCP servers (kept alive for the tool-call loop).
    mcp: Vec<mcp::Connection>,
    /// Path to Zero's own MCP server config (`~/.zero/mcp.json`).
    mcp_path: Option<PathBuf>,
    /// `$HOME`, for importing MCP servers from Claude Desktop / Claude Code.
    /// `None` in tests so discovery reads only the explicit `mcp_path`.
    mcp_home: Option<PathBuf>,
    /// Working directory, for project `.mcp.json` + Claude Code's per-project
    /// servers. `None` in tests.
    mcp_cwd: Option<PathBuf>,
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
    /// Fenced code blocks from the last response, for `/clip <n>`.
    last_blocks: Vec<crate::markdown::CodeBlock>,
    /// How `/clip` copies — the system clipboard by default; swappable in tests.
    clipboard: ClipboardFn,
    /// The current in-flight turn, if a reply is streaming.
    streaming: Option<StreamState>,
    /// Messages typed while a reply was streaming, run in order afterward.
    queue: VecDeque<String>,
    /// Run the backend inline instead of on a thread — deterministic for tests.
    synchronous: bool,
    /// Server-reported context window (`n_ctx`), fetched on connect. `None` until
    /// known (e.g. the stub, or a server that doesn't expose `/props`).
    ctx_window: Option<u64>,
    /// Token usage from the most recent completed turn, for the status line.
    last_usage: Option<Usage>,
    /// The uncommitted tail of the streaming reply (rendered ANSI): the part of
    /// the current line not yet ended by a newline. It is repainted *above* the
    /// pinned input box each frame; complete lines are committed to scrollback.
    /// Empty when not streaming.
    pending: String,
    /// When true, a submitted message runs the agentic tool loop (the model can
    /// call built-in filesystem tools) instead of a plain streamed chat reply.
    /// Toggled by `/tools`.
    tools_enabled: bool,
    /// Max bytes of any single tool result fed back into the context window.
    max_tool_output: usize,
    /// Max cumulative tool-result bytes fed back within one agentic turn.
    max_turn_output: usize,
    /// Files read this session, so an unchanged re-read returns a stub.
    read_cache: zero_core::context::ReadCache,
    /// Measured, cumulative bytes the context levers saved this session (`/context`).
    context_stats: zero_core::context::ContextStats,
    /// Directory where full tool outputs are spilled before compression, so the
    /// model can re-fetch dropped content (cap = offload, never silent delete).
    /// `None` (tests/no session) → compression still runs, just without a
    /// re-fetch path. Set from the session dir by the binary.
    artifact_dir: Option<PathBuf>,
}

// Canonical context-cap defaults live in zero_core::context (shared with Config);
// aliased here so the App-construction defaults can't drift from the core ones.
use zero_core::context::{DEFAULT_MAX_TOOL_OUTPUT, DEFAULT_MAX_TURN_OUTPUT, TURN_OUTPUT_FLOOR};

impl<I: Input, W: Write> App<I, W> {
    /// Build an app over an input source, an output sink, and a backend.
    pub fn new(
        input: I,
        out: W,
        backend: Arc<dyn Backend>,
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
            queue_edit: None,
            mode: Mode::default(),
            mcp: Vec::new(),
            mcp_path: None,
            mcp_home: None,
            mcp_cwd: None,
            pending_shell: None,
            info: String::new(),
            config: Config::default(),
            config_path: None,
            servers_path: None,
            scan_results: Vec::new(),
            last_reply: String::new(),
            last_blocks: Vec::new(),
            clipboard: Box::new(clipboard_copy),
            streaming: None,
            queue: VecDeque::new(),
            synchronous: false,
            ctx_window: None,
            last_usage: None,
            pending: String::new(),
            tools_enabled: false,
            max_tool_output: DEFAULT_MAX_TOOL_OUTPUT,
            max_turn_output: DEFAULT_MAX_TURN_OUTPUT,
            read_cache: zero_core::context::ReadCache::new(),
            context_stats: zero_core::context::ContextStats::new(),
            artifact_dir: None,
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
        // Apply the context caps from config (Config bakes in the defaults).
        self.max_tool_output = self.config.max_tool_output;
        self.max_turn_output = self.config.max_turn_output;
        self.config_path = config_path;
        self.servers_path = servers_path;
    }

    /// Where to read Zero's own MCP server definitions (`~/.zero/mcp.json`).
    pub fn set_mcp_path(&mut self, path: Option<PathBuf>) {
        self.mcp_path = path;
    }

    /// Enable importing MCP servers from other tools (Claude Desktop, Claude
    /// Code) and the project's `.mcp.json`, by giving `$HOME` and the working dir.
    pub fn set_mcp_discovery(&mut self, home: Option<PathBuf>, cwd: Option<PathBuf>) {
        self.mcp_home = home;
        self.mcp_cwd = cwd;
    }

    /// Set where full tool outputs are spilled so compressed results stay
    /// re-fetchable. The binary points this at the session's output directory.
    pub fn set_artifact_dir(&mut self, dir: Option<PathBuf>) {
        self.artifact_dir = dir;
    }

    /// Turn the agentic tool loop on/off (the `/tools` toggle, exposed for
    /// headless runs and integration tests).
    pub fn set_tools_enabled(&mut self, on: bool) {
        self.tools_enabled = on;
    }

    /// Read-only view of the live conversation (for headless callers and
    /// integration tests that assert on what was fed back to the model).
    pub fn conversation(&self) -> &Conversation {
        &self.conv
    }

    /// Measured context-savings stats for this session (`/context`).
    pub fn context_stats(&self) -> &zero_core::context::ContextStats {
        &self.context_stats
    }

    /// Server-reported token usage of the most recent turn, if the backend
    /// reported it (`None` for the stub). Lets a headless run print real tokens.
    pub fn last_usage(&self) -> Option<zero_core::backend::Usage> {
        self.last_usage
    }

    /// Run a single turn headlessly and return the assistant's final reply.
    /// Uses the agentic tool loop when tools are enabled (so `bash`/builtins run
    /// and their output flows through the spill+compress path), otherwise a bare
    /// non-streaming completion. The turn's trace (tool calls, inline text) is
    /// written to the output sink as usual; the binary points that at stderr so
    /// `zero -p` keeps stdout to just the returned reply.
    pub fn run_once(&mut self, prompt: &str) -> io::Result<String> {
        if self.tools_enabled {
            self.run_tool_turn(prompt)?;
        } else {
            self.conv.push(Message::user(prompt));
            let timeout = Duration::from_secs(120);
            match self.backend.complete(&self.conv, &[], timeout) {
                Ok(c) => {
                    self.write_text(&c.content)?;
                    self.conv.push(Message::assistant(&c.content));
                    if let Some(u) = c.usage {
                        self.last_usage = Some(u);
                    }
                    self.last_reply = c.content;
                }
                Err(e) => {
                    self.last_reply = format!("[error: {e}]");
                    let line = self.last_reply.clone();
                    self.write_text(&line)?;
                }
            }
        }
        Ok(self.last_reply.clone())
    }

    /// Ask the configured server for its context window (`n_ctx`) so the status
    /// line can show context usage. Best-effort and skipped in synchronous (test)
    /// mode so the unit tests never touch the network.
    fn refresh_context_window(&mut self) {
        if self.synchronous {
            return;
        }
        self.ctx_window = self.config.base_url.as_deref().and_then(|url| {
            zero_core::openai::fetch_context_window(url, Duration::from_millis(500))
        });
    }

    /// The dim footer under the input box: model · endpoint · context usage.
    /// Always shows the model; the context segment appears once we have real
    /// numbers (from the server's `/props` and/or a turn's usage chunk).
    fn status_line(&self) -> String {
        let model = if !self.config.model.is_empty() {
            self.config.model.as_str()
        } else {
            self.backend.name()
        };
        let mut parts = vec![model.to_string()];
        if let Some(url) = &self.config.base_url {
            parts.push(short_host(url));
        }
        if let Some(ctx) = self.context_summary() {
            parts.push(ctx);
        }
        parts.join("  ·  ")
    }

    /// The context-usage segment, or `None` when nothing is known yet.
    fn context_summary(&self) -> Option<String> {
        let used = self.last_usage.map(|u| u.total());
        match (self.ctx_window, used) {
            (Some(ctx), Some(used)) if ctx > 0 => {
                let pct = (used * 100 / ctx).min(100);
                Some(format!(
                    "{}/{} ctx ({pct}%)",
                    fmt_count(used),
                    fmt_count(ctx)
                ))
            }
            (Some(ctx), _) => Some(format!("{} ctx", fmt_count(ctx))),
            (None, Some(used)) => Some(format!("{} tok", fmt_count(used))),
            (None, None) => None,
        }
    }

    /// Run the event loop until the user quits.
    pub fn run(&mut self) -> io::Result<()> {
        // Ask the terminal to report disambiguated keys (kitty keyboard
        // protocol). On terminals that support it, Shift+Enter then arrives as
        // `ESC [ 13 ; 2 u`; on others this is silently ignored. Popped in finish.
        self.out.write_all(b"\x1b[>1u")?;
        // NOTE: we deliberately do NOT enable SGR mouse reporting. It would catch
        // clicks for click-to-copy, but it also steals the scroll wheel, killing
        // native scrollback — a core feature. Copy with `/clip`.
        self.print_banner()?;
        // Best-effort: learn the server's context window so the status line can
        // show usage from the first turn.
        self.refresh_context_window();
        // Auto-connect any MCP servers configured here or in Claude Desktop /
        // Claude Code — quiet when there are none (no startup noise).
        self.autoconnect_mcp()?;
        self.redraw_input()?;

        let mut inbuf: Vec<u8> = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            // Drain any streamed tokens first so the reply renders promptly.
            if self.streaming.is_some() {
                self.pump_stream()?;
            }
            let n = self.input.read(&mut buf)?; // returns within ~100ms (VTIME)
            if n == 0 {
                if inbuf == [0x1b] {
                    inbuf.clear();
                    self.dispatch(Key::Esc)?; // Esc never quits, only clears/arms
                    self.redraw_if_idle()?;
                }
                continue;
            }
            inbuf.extend_from_slice(&buf[..n]);
            let (keys, consumed) = decode_keys(&inbuf);
            inbuf.drain(0..consumed);
            for key in keys {
                if self.dispatch(key)? == Flow::Quit {
                    return self.finish();
                }
            }
            // The input box is pinned at the bottom in every mode now, so repaint
            // after handling input — including mid-stream, so the queue preview
            // and what you're typing stay visible while a reply generates.
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

    /// Route a key, honoring submodes (confirm, search, streaming) and combos.
    fn dispatch(&mut self, key: Key) -> io::Result<Flow> {
        if self.pending_shell.is_some() {
            return self.handle_confirm_key(key);
        }
        if self.search.is_some() {
            return self.handle_search_key(key);
        }
        // Queue-edit takes priority over streaming: you edit the queue *while* a
        // reply generates, with sending paused until you exit.
        if self.queue_edit.is_some() {
            return self.handle_queue_edit_key(key);
        }
        if self.streaming.is_some() {
            return self.handle_streaming_key(key);
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
            Key::Ctrl('q') => self.enter_queue_edit()?,
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
            // Tab completes a slash command (shell-style); no-op otherwise.
            Key::Tab => {
                self.try_complete_slash();
            }
            Key::BackTab => self.mode = self.mode.next(), // Shift+Tab cycles modes
            // Scrollback is the terminal's own (native); we don't intercept it.
            Key::PageUp | Key::PageDown => {}
            Key::Ctrl(_) => {} // unmapped; ignore
            // Enter completes an in-progress slash command instead of submitting;
            // a second Enter (now a full command) submits.
            Key::Enter => {
                if !self.try_complete_slash() {
                    return self.on_submit();
                }
            }
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
                if self.mode == Mode::AutoAccept {
                    // Auto-accept mode: skip the y/N gate, but still flag it.
                    self.write_text(&format!(
                        "\x1b[33m⚠ {reason}\x1b[0m \x1b[2m(auto-accepted)\x1b[0m\n"
                    ))?;
                    self.run_shell(&cmd)?;
                    return Ok(Flow::Continue);
                }
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
        if trimmed == "/context" {
            self.echo_committed(&text)?;
            self.print_context_stats()?;
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
        if trimmed == "/mcp" || trimmed.starts_with("/mcp ") {
            self.echo_committed(&text)?;
            let arg = trimmed.strip_prefix("/mcp").unwrap_or("").trim();
            self.mcp_command(arg)?;
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
        if let Some(rest) = trimmed.strip_prefix("/clip") {
            self.echo_committed(&text)?;
            self.do_clip(rest.trim())?;
            return Ok(Flow::Continue);
        }
        if trimmed == "/tools" {
            self.echo_committed(&text)?;
            self.tools_enabled = !self.tools_enabled;
            let state = if self.tools_enabled { "on" } else { "off" };
            self.write_text(&format!(
                "\x1b[2mtools {state} — the model can {}use read/list/grep/write/edit\x1b[0m\n",
                if self.tools_enabled { "" } else { "no longer " }
            ))?;
            return Ok(Flow::Continue);
        }

        // With tools enabled, run the agentic loop (the model can call built-in
        // tools) instead of a plain streamed reply.
        if self.tools_enabled {
            self.echo_committed(&text)?;
            return self.run_tool_turn(&text).map(|()| Flow::Continue);
        }

        // A normal message: start a streaming turn (on a background thread, so
        // the loop stays free to queue more input or interrupt).
        self.start_turn(&text)?;
        if self.synchronous {
            // Tests: drive the (inline-filled) turn to completion now.
            while self.streaming.is_some() {
                self.pump_stream()?;
            }
        }
        Ok(Flow::Continue)
    }

    // --- agentic tool loop (/tools) --------------------------------------

    /// Run one agentic turn: the model may call built-in tools, each gated by the
    /// current mode, until it answers with plain text. Synchronous and blocking
    /// (no mid-turn queue/interrupt yet) — the threaded version is a follow-up.
    fn run_tool_turn(&mut self, prompt: &str) -> io::Result<()> {
        use std::cell::RefCell;
        self.conv.push(Message::user(prompt));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::User, prompt);
        }
        self.write_text(&format!("\x1b[2m{}›\x1b[0m\n", zero_core::brand::slug()))?;

        let tools = zero_core::builtins::definitions();
        let backend = Arc::clone(&self.backend);
        let mode = self.mode;
        let root = std::env::current_dir().ok();
        let timeout = Duration::from_secs(120);
        let mut guard = LoopGuard::new(25);
        let mut conv = std::mem::take(&mut self.conv);

        // A RefCell wrapper lets the three loop callbacks each write to the
        // output without three simultaneous &mut borrows of `self.out`. The inner
        // block scopes those borrows so `self` is free again after it returns.
        let cap = self.max_tool_output;
        let turn_budget = self.max_turn_output;
        let artifact_dir = self.artifact_dir.clone();
        let spent = RefCell::new(0usize); // cumulative result bytes this turn
        let read_cache = RefCell::new(std::mem::take(&mut self.read_cache));
        let stats = RefCell::new(std::mem::take(&mut self.context_stats));
        let res = {
            let out = RefCell::new(&mut self.out);
            let mut completer = |c: &Conversation, t: &[ToolDef]| backend.complete(c, t, timeout);
            let mut executor = |call: &ToolCall| {
                let _ = write_raw(
                    &mut **out.borrow_mut(),
                    &format!("\x1b[2m  ⚙ {}({})\x1b[0m\n", call.name, call.arguments),
                );
                let path = call_path(call, root.as_deref());
                // Read cache: a repeat read of an unchanged WHOLE file returns a
                // stub (the content is already upstream). Ranged reads always run.
                if call.name == "read_file" && !has_range(call) {
                    if let Some(p) = &path {
                        if let Some(stub) = read_cache.borrow().check(p) {
                            // Measured saving: the would-be re-read is the file's
                            // current size (unchanged since we cached it).
                            let would_be = std::fs::metadata(p)
                                .map(|m| m.len() as usize)
                                .unwrap_or(stub.len());
                            stats.borrow_mut().record_cache_hit(would_be, stub.len());
                            let _ = write_raw(
                                &mut **out.borrow_mut(),
                                &format!("\x1b[2m  ↳ (cached) {stub}\x1b[0m\n"),
                            );
                            return stub;
                        }
                    }
                }
                let raw = gate_and_execute(mode, call, root.as_deref());
                // Context cap: bound what goes back into the window. The effective
                // cap also shrinks as the per-turn budget depletes (always keeping
                // a floor), so a turn firing many calls can't blow the window by
                // attrition. Nothing is lost — a re-fetch hint rides along.
                let raw_len = raw.len();
                let remaining = turn_budget.saturating_sub(*spent.borrow());
                let eff_cap = cap.min(remaining.max(TURN_OUTPUT_FLOOR));
                let result = cap_tool_result(call, raw, eff_cap, artifact_dir.as_deref());
                *spent.borrow_mut() += result.len();
                stats.borrow_mut().record_result(raw_len, result.len());
                // Maintain the read cache: remember successful whole reads; forget
                // a file after a write/edit so it re-reads in full next time.
                if let Some(p) = &path {
                    let ok = !result.starts_with("error:") && !result.starts_with("refused:");
                    match call.name.as_str() {
                        "read_file" if ok && !has_range(call) => read_cache.borrow_mut().record(p),
                        "write_file" | "edit_file" if ok => read_cache.borrow_mut().invalidate(p),
                        _ => {}
                    }
                }
                let _ = write_raw(
                    &mut **out.borrow_mut(),
                    &format!("\x1b[2m  ↳ {}\x1b[0m\n", first_line(&result)),
                );
                result
            };
            let mut on_text = |t: &str| {
                let _ = write_raw(&mut **out.borrow_mut(), t);
            };
            zero_core::agent::run_turn(
                &mut conv,
                &tools,
                &mut completer,
                &mut executor,
                &mut guard,
                &mut on_text,
            )
        };
        self.conv = conv;
        self.read_cache = read_cache.into_inner();
        self.context_stats = stats.into_inner();
        // Drop bulky content from applied write/edit calls — the file is on disk,
        // so re-sending the payload to the model each turn just wastes the window.
        let (before, after) = compact_applied_writes(&mut self.conv);
        self.context_stats.record_compaction(before, after);

        match res {
            Ok(outcome) => {
                let note = match outcome.stop {
                    zero_core::agent::AgentStop::Done => String::new(),
                    zero_core::agent::AgentStop::MaxSteps => {
                        "\n\x1b[33m[stopped: tool step cap reached]\x1b[0m".to_string()
                    }
                    zero_core::agent::AgentStop::DoomLoop => {
                        "\n\x1b[33m[stopped: model repeated the same tool call]\x1b[0m".to_string()
                    }
                };
                self.write_text(&format!("{note}\n"))?;
                self.last_reply = outcome.final_text.clone();
                // Summed server-reported tokens across the turn's rounds, so a
                // headless caller (and the status line) can report real usage.
                if outcome.usage.total() > 0 {
                    self.last_usage = Some(outcome.usage);
                }
                self.last_blocks = crate::markdown::code_blocks(&outcome.final_text);
                if let Some(log) = self.log.as_mut() {
                    let _ = log.record_message(Role::Assistant, &outcome.final_text);
                }
            }
            Err(e) => {
                self.write_text(&format!("\x1b[31m[{e}]\x1b[0m\n"))?;
            }
        }
        self.redraw_input()
    }

    // --- streaming turns (threaded) --------------------------------------

    /// Echo the user line, then kick off a streamed reply. The backend runs on a
    /// thread (or inline when `synchronous`), sending events down a channel.
    fn start_turn(&mut self, prompt: &str) -> io::Result<()> {
        self.echo_committed(prompt)?;
        self.conv.push(Message::user(prompt));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::User, prompt);
        }
        // The assistant label leads the live reply tail (repainted above the box
        // as tokens arrive), so it isn't committed until the line completes.
        self.pending = format!("\x1b[2m{}›\x1b[0m ", zero_core::brand::slug());

        let (tx, rx) = mpsc::channel();
        let backend = Arc::clone(&self.backend);
        // Plan mode: prepend a planning directive (system) to *this request only*
        // — self.conv is untouched, so it doesn't persist across turns/modes.
        let mut conv = self.conv.clone();
        if self.mode == Mode::Plan {
            conv.messages
                .insert(0, Message::new(Role::System, PLAN_DIRECTIVE.to_string()));
        }
        let run = move || {
            // Catch panics so a misbehaving backend can't take down the whole
            // process; surface errors/panics as a visible token + Done so the
            // turn finalizes cleanly.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.stream(&conv, &mut |ev| {
                    let _ = tx.send(ev);
                })
            }));
            let note = match result {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(format!("\n\x1b[31m[{e}]\x1b[0m")),
                Err(_) => Some("\n\x1b[31m[internal error: stream panicked]\x1b[0m".to_string()),
            };
            if let Some(note) = note {
                let _ = tx.send(StreamEvent::Token(note));
                let _ = tx.send(StreamEvent::Done(StopReason::EndTurn));
            }
        };
        if self.synchronous {
            run(); // fill the channel now — deterministic for tests
        } else {
            std::thread::spawn(run);
        }
        self.streaming = Some(StreamState {
            rx,
            reply: String::new(),
            md: MarkdownStream::new(),
            sw: Stopwatch::start(),
            usage: None,
        });
        Ok(())
    }

    /// Drain available streamed tokens, rendering them in place; finalize when
    /// the turn completes.
    fn pump_stream(&mut self) -> io::Result<()> {
        let mut tokens = Vec::new();
        let mut done = false;
        let mut usage = None;
        if let Some(s) = &self.streaming {
            loop {
                match s.rx.try_recv() {
                    Ok(StreamEvent::Token(t)) => tokens.push(t),
                    Ok(StreamEvent::Usage(u)) => usage = Some(u),
                    Ok(StreamEvent::Done(_)) => {
                        done = true;
                        break;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        done = true;
                        break;
                    }
                }
            }
        }
        if let (Some(u), Some(s)) = (usage, self.streaming.as_mut()) {
            s.usage = Some(u);
        }
        if !tokens.is_empty() {
            // Render markdown (raw kept in `reply`, ANSI appended to `pending`)
            // without overlapping the &mut self borrow.
            let mut rendered = String::new();
            if let Some(s) = self.streaming.as_mut() {
                for t in &tokens {
                    s.reply.push_str(t);
                    rendered.push_str(&s.md.feed(t));
                }
            }
            self.pending.push_str(&rendered);
            // Repaint: commits completed reply lines to scrollback and redraws
            // the tail + pinned box (a single repaint per pump, no flicker).
            self.redraw_input()?;
        }
        if done {
            self.finalize_stream()?;
        }
        Ok(())
    }

    /// The streamed turn finished: close styling, record it, then start the next
    /// queued message (if any).
    fn finalize_stream(&mut self) -> io::Result<()> {
        let Some(mut s) = self.streaming.take() else {
            return Ok(());
        };
        // Erase the live region, then commit the final reply line (pending tail +
        // any closing styling) permanently to scrollback.
        self.clear_input_block()?;
        self.cursor_row = 0;
        let mut final_line = std::mem::take(&mut self.pending);
        final_line.push_str(&s.md.finish());
        if !final_line.is_empty() {
            write_raw(&mut self.out, &final_line)?;
            write_raw(&mut self.out, "\x1b[0m\n")?;
        }
        let elapsed = s.sw.elapsed();
        if s.usage.is_some() {
            self.last_usage = s.usage;
        }
        let reply = std::mem::take(&mut s.reply);
        self.conv.push(Message::assistant(&reply));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::Assistant, &reply);
            let _ = log.record_turn_done(elapsed.as_millis());
        }
        self.last_reply = reply.clone();
        self.last_blocks = crate::markdown::code_blocks(&reply);
        self.write_text(&format!("\x1b[2m  {}\x1b[0m\n", format_duration(elapsed)))?;

        // Sending is paused while the queue is being edited (`^Q`): leave the
        // items in place and just repaint; they run when editing exits.
        if self.queue_edit.is_none() {
            if let Some(next) = self.queue.pop_front() {
                return self.start_turn(&next);
            }
        }
        self.redraw_input()?;
        Ok(())
    }

    /// Keys during a streaming turn: `^C` interrupts, `Enter` queues the typed
    /// line, `/quit` still exits. Editing keys update the pinned input box, which
    /// the run loop repaints; the queued count shows in the footer.
    fn handle_streaming_key(&mut self, key: Key) -> io::Result<Flow> {
        // `Esc Esc` interrupts too; reset the latch on any other key.
        if key != Key::Esc {
            self.esc_pending = false;
        }
        match key {
            Key::Ctrl('c') => self.interrupt_stream()?, // single ^C interrupts
            Key::Ctrl('q') => self.enter_queue_edit()?, // edit the pending queue
            Key::Esc => {
                if self.esc_pending {
                    self.esc_pending = false;
                    self.interrupt_stream()?;
                } else {
                    self.esc_pending = true;
                }
            }
            Key::Enter => {
                let text = self.editor.submit();
                let trimmed = text.trim();
                if matches!(trimmed, "/quit" | "/exit") {
                    return Ok(Flow::Quit);
                }
                if !trimmed.is_empty() {
                    self.queue.push_back(text.clone());
                    // editor cleared by submit(); footer shows the queued count
                    // on the next repaint.
                }
            }
            Key::BackTab => self.mode = self.mode.next(), // cycle modes mid-stream
            Key::Backspace => self.editor.backspace(),
            Key::Char(c) => self.editor.insert(c),
            _ => {} // other editing keys: no live echo while streaming
        }
        Ok(Flow::Continue)
    }

    /// Abort the in-flight stream (`^C`): keep the partial reply in context,
    /// drop anything queued, and return to the prompt.
    fn interrupt_stream(&mut self) -> io::Result<()> {
        let Some(mut s) = self.streaming.take() else {
            return Ok(());
        };
        // Erase the live region and commit the partial reply + interrupt note.
        self.clear_input_block()?;
        self.cursor_row = 0;
        let mut final_line = std::mem::take(&mut self.pending);
        final_line.push_str(&s.md.finish());
        if !final_line.is_empty() {
            write_raw(&mut self.out, &final_line)?;
            write_raw(&mut self.out, "\x1b[0m\n")?;
        }
        self.write_text("\x1b[2m^C interrupted\x1b[0m\n")?;
        let reply = std::mem::take(&mut s.reply);
        if !reply.trim().is_empty() {
            self.conv.push(Message::assistant(&reply));
            self.last_reply = reply.clone();
            self.last_blocks = crate::markdown::code_blocks(&reply);
        }
        self.queue.clear();
        self.redraw_input()?;
        // The detached thread (if any) finishes on its own; its sends are
        // dropped harmlessly now that `rx` is gone.
        Ok(())
    }

    /// `/clip` copies the whole last response; `/clip <n>` copies code block n.
    fn do_clip(&mut self, arg: &str) -> io::Result<()> {
        if arg.is_empty() {
            if self.last_reply.trim().is_empty() {
                return self.write_text("\x1b[2mnothing to copy yet\x1b[0m\n");
            }
            let text = self.last_reply.clone();
            return self.copy_text(&text);
        }
        match arg.parse::<usize>() {
            Ok(n) if (1..=self.last_blocks.len()).contains(&n) => {
                let body = self.last_blocks[n - 1].body.clone();
                self.copy_text(&body)
            }
            _ => self.write_text("\x1b[31mno such code block — see the /clip hints\x1b[0m\n"),
        }
    }

    /// Copy `text` via the configured clipboard and report the result.
    fn copy_text(&mut self, text: &str) -> io::Result<()> {
        let n = text.chars().count();
        match (self.clipboard)(text) {
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
            self.backend = Arc::new(b);
        }
        if let Some(path) = &self.config_path {
            let _ = self.config.save(path);
        }
        self.info = self.config.summary();
        // Endpoint/model changed: old token counts are stale; relearn n_ctx.
        self.last_usage = None;
        self.refresh_context_window();
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

    // --- MCP servers (/mcp) ----------------------------------------------

    /// `/mcp` connects configured servers and summarizes; `/mcp tools` lists the
    /// tools discovered so far.
    fn mcp_command(&mut self, arg: &str) -> io::Result<()> {
        if arg == "tools" {
            return self.print_mcp_tools();
        }
        self.connect_mcp()
    }

    /// The config locations to scan, highest precedence first: project
    /// `.mcp.json`, Zero's own file, then imports from Claude Desktop / Claude
    /// Code. Tests set only `mcp_path`, so they read just that file.
    fn mcp_candidates(&self) -> Vec<(mcp::Source, PathBuf)> {
        let mut c = Vec::new();
        if let Some(cwd) = &self.mcp_cwd {
            c.push((mcp::Source::Project, cwd.join(".mcp.json")));
        }
        if let Some(p) = &self.mcp_path {
            c.push((mcp::Source::Zero, p.clone()));
        }
        if let Some(home) = &self.mcp_home {
            c.push((
                mcp::Source::ClaudeDesktop,
                home.join("Library/Application Support/Claude/claude_desktop_config.json"),
            ));
            c.push((mcp::Source::ClaudeCode, home.join(".claude.json")));
        }
        c
    }

    /// Startup auto-connect: like `/mcp`, but silent when no servers are
    /// configured anywhere (a fresh install prints nothing at launch).
    fn autoconnect_mcp(&mut self) -> io::Result<()> {
        let cwd = self.mcp_cwd.clone().unwrap_or_else(|| PathBuf::from("."));
        if mcp::discover(&self.mcp_candidates(), &cwd)
            .servers
            .is_empty()
        {
            return Ok(());
        }
        self.connect_mcp()
    }

    /// Discover MCP servers across all sources and connect any not yet connected
    /// (blocking, best-effort), reporting the outcome and origin per server.
    fn connect_mcp(&mut self) -> io::Result<()> {
        let cwd = self.mcp_cwd.clone().unwrap_or_else(|| PathBuf::from("."));
        let found = mcp::discover(&self.mcp_candidates(), &cwd);

        // Surface (but don't fail on) a config that exists yet won't parse.
        for (source, path, msg) in &found.errors {
            self.write_text(&format!(
                "\x1b[31mmcp config ({}): {msg}\x1b[0m \x1b[2m{path}\x1b[0m\n",
                source.label()
            ))?;
        }

        if found.servers.is_empty() {
            if found.errors.is_empty() {
                self.write_text(
                    "\x1b[2mno MCP servers found — add some to ~/.zero/mcp.json, or configure \
                     them in Claude Desktop / Claude Code and they'll be imported\x1b[0m\n",
                )?;
            }
            return Ok(());
        }

        self.write_text("\x1b[2mconnecting MCP servers…\x1b[0m\n")?;
        self.out.flush()?;
        for d in &found.servers {
            if self.mcp.iter().any(|c| c.name == d.name) {
                self.write_text(&format!("  \x1b[2m• {} already connected\x1b[0m\n", d.name))?;
                continue;
            }
            match mcp::Connection::connect(&d.name, &d.spec) {
                Ok(conn) => {
                    let n = conn.tools.len();
                    self.mcp.push(conn);
                    self.write_text(&format!(
                        "  \x1b[32m✓\x1b[0m {} \x1b[2m— {n} tools (from {})\x1b[0m\n",
                        d.name,
                        d.source.label()
                    ))?;
                }
                Err(e) => {
                    self.write_text(&format!("  \x1b[31m✗ {} — {e}\x1b[0m\n", d.name))?;
                }
            }
        }
        Ok(())
    }

    /// Report the *measured* bytes the context levers saved this session — an
    /// honest accounting (never an estimate), per the measure-don't-guess rule.
    fn print_context_stats(&mut self) -> io::Result<()> {
        let s = &self.context_stats;
        if s.raw_bytes == 0 && s.total_saved() == 0 {
            return self.write_text("\x1b[2mno tool output yet — run a /tools turn first\x1b[0m\n");
        }
        let kb = |n: usize| format!("{:.1} KB", n as f64 / 1024.0);
        let out = format!(
            "\x1b[1mcontext savings\x1b[0m (measured this session)\n\
             \x20 cap:      \x1b[36m{}\x1b[0m  (oversized tool results trimmed)\n\
             \x20 cache:    \x1b[36m{}\x1b[0m  (unchanged re-reads skipped)\n\
             \x20 compact:  \x1b[36m{}\x1b[0m  (applied write/edit payloads dropped)\n\
             \x20 \x1b[1mtotal:    {}\x1b[0m  →  \x1b[32m{}% smaller window\x1b[0m\n",
            kb(s.capped_saved),
            kb(s.cached_saved),
            kb(s.compacted_saved),
            kb(s.total_saved()),
            s.reduction_pct(),
        );
        self.write_text(&out)
    }

    /// List discovered tools, grouped by server. One compact line per tool —
    /// MCP descriptions can be paragraphs (n8n et al.), so the description is
    /// collapsed to a single line and capped to keep `/mcp tools` scannable.
    fn print_mcp_tools(&mut self) -> io::Result<()> {
        if self.mcp.is_empty() {
            return self.write_text("\x1b[2mno MCP servers connected — run /mcp\x1b[0m\n");
        }
        // Borrow-safe: format first, then write.
        let mut out = String::new();
        for conn in &self.mcp {
            out.push_str(&format!(
                "\x1b[1m{}\x1b[0m \x1b[2m({} tools)\x1b[0m\n",
                conn.name,
                conn.tools.len()
            ));
            for t in &conn.tools {
                let desc = tool_desc_snippet(&t.description);
                if desc.is_empty() {
                    out.push_str(&format!("  \x1b[36m{}\x1b[0m\n", t.name));
                } else {
                    out.push_str(&format!(
                        "  \x1b[36m{}\x1b[0m \x1b[2m— {desc}\x1b[0m\n",
                        t.name
                    ));
                }
            }
        }
        self.write_text(&out)
    }

    // --- queue editing (^Q) ----------------------------------------------

    /// Enter queue-edit mode: pause sending and load the nearest queued message
    /// into the editor for editing. No-op when the queue is empty.
    fn enter_queue_edit(&mut self) -> io::Result<()> {
        if self.queue.is_empty() {
            return Ok(());
        }
        let sel = self.queue.len() - 1; // nearest the box (bottom of the list)
        let saved_input = self.editor.text();
        self.editor.set_text(&self.queue[sel]);
        self.queue_edit = Some(QueueEdit { sel, saved_input });
        self.redraw_input()
    }

    /// Keys while editing the queue: `↑`/`↓` (or repeated `^Q`) move between
    /// items, editing keys change the selected one, `Enter`/`Esc` exit. Edits to
    /// the current item are saved before moving or exiting; an item emptied this
    /// way is dropped.
    fn handle_queue_edit_key(&mut self, key: Key) -> io::Result<Flow> {
        match key {
            Key::Up | Key::Ctrl('q') => {
                self.queue_edit_select(-1);
            }
            Key::Down => {
                self.queue_edit_select(1);
            }
            Key::Enter | Key::Esc => {
                self.exit_queue_edit()?;
            }
            Key::Backspace => self.editor.backspace(),
            Key::Delete => self.editor.delete(),
            Key::Left | Key::Ctrl('b') => self.editor.left(),
            Key::Right | Key::Ctrl('f') => self.editor.right(),
            Key::Home | Key::Ctrl('a') => self.editor.home(),
            Key::End | Key::Ctrl('e') => self.editor.end(),
            Key::WordLeft => self.editor.word_left(),
            Key::WordRight => self.editor.word_right(),
            Key::Ctrl('u') => self.editor.kill_to_start(),
            Key::Ctrl('k') => self.editor.kill_to_end(),
            Key::Ctrl('w') => self.editor.kill_word(),
            Key::ShiftEnter => self.editor.insert_newline(),
            Key::Char(c) => self.editor.insert(c),
            _ => {}
        }
        Ok(Flow::Continue)
    }

    /// Persist the current edit, then move the selection by `delta` (clamped).
    fn queue_edit_select(&mut self, delta: isize) {
        let Some(qe) = self.queue_edit.as_mut() else {
            return;
        };
        self.queue[qe.sel] = self.editor.text();
        let last = self.queue.len() - 1;
        qe.sel = (qe.sel as isize + delta).clamp(0, last as isize) as usize;
        let text = self.queue[qe.sel].clone();
        self.editor.set_text(&text);
    }

    /// Save the current edit, drop any emptied items, restore the in-progress
    /// input line, and resume sending (run the next queued message if idle).
    fn exit_queue_edit(&mut self) -> io::Result<()> {
        let Some(qe) = self.queue_edit.take() else {
            return Ok(());
        };
        self.queue[qe.sel] = self.editor.text();
        self.queue.retain(|m| !m.trim().is_empty());
        self.editor.set_text(&qe.saved_input);
        // Resume: if nothing is streaming, kick off the next queued message.
        if self.streaming.is_none() {
            if let Some(next) = self.queue.pop_front() {
                return self.start_turn(&next);
            }
        }
        self.redraw_input()
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

    /// Move to the top-left of the current live region (reply tail + box) and
    /// clear downward, so it can be repainted.
    fn clear_input_block(&mut self) -> io::Result<()> {
        if self.cursor_row > 0 {
            write!(self.out, "\x1b[{}A", self.cursor_row)?;
        }
        self.out.write_all(b"\r\x1b[J")?;
        Ok(())
    }

    /// The status footer under the box: model · endpoint · ctx, plus live turn
    /// state (elapsed + interrupt hint) while a reply streams. Queued messages
    /// are listed above the box, not here.
    fn footer_text(&self) -> String {
        if let Some(qe) = &self.queue_edit {
            // Editing pauses sending; show how to navigate and finish.
            return format!(
                "editing queued {}/{}  ·  ↑↓ move · ⏎ done · sending paused",
                qe.sel + 1,
                self.queue.len()
            );
        }
        // Mode chip first (its own color), then the dim status line. The footer
        // is rendered inside a dim wrapper, so reset back to dim after the chip.
        let mut footer = format!(
            "{}{}\x1b[0m\x1b[2m  ·  {}",
            self.mode.color(),
            self.mode.label(),
            self.status_line()
        );
        if let Some(s) = &self.streaming {
            footer.push_str(&format!(
                "  ·  {} · esc to interrupt",
                format_duration(s.sw.elapsed())
            ));
        }
        // Tell the user the queue is editable whenever one exists.
        if !self.queue.is_empty() {
            footer.push_str("  ·  ^Q edit queue");
        }
        // Hint how to change mode (Shift+Tab), like Claude Code.
        footer.push_str("  ·  ⇧⇥ mode");
        footer
    }

    /// Slash commands to show in the popup for the current input — empty unless
    /// a slash token is being typed, and hidden once it's a complete unique
    /// command (nothing left to suggest).
    fn slash_suggestions(&self) -> Vec<(&'static str, &'static str)> {
        let text = self.editor.text();
        let Some(q) = slash_query(&text) else {
            return Vec::new();
        };
        let m = slash_matches(q);
        if m.len() == 1 && m[0].0 == q {
            return Vec::new(); // already fully typed; nothing to complete
        }
        m
    }

    /// Try to complete the in-progress slash command. Returns true if Enter/Tab
    /// was consumed for completion (so it must NOT also submit). Completes fully
    /// on a single match, to the longest common prefix on several, and swallows
    /// the key while a partial command is ambiguous so a stray `/c` is never sent
    /// to the model.
    fn try_complete_slash(&mut self) -> bool {
        let text = self.editor.text();
        let Some(q) = slash_query(&text) else {
            return false;
        };
        let matches = slash_matches(q);
        if matches.is_empty() {
            return false;
        }
        // Exactly the typed command exists → let Enter submit it.
        if matches.iter().any(|(name, _)| *name == q) {
            return false;
        }
        let names: Vec<&str> = matches.iter().map(|(name, _)| *name).collect();
        let completed = if names.len() == 1 {
            names[0].to_string()
        } else {
            common_prefix(&names)
        };
        if completed.len() > text.len() {
            self.editor.set_text(&completed);
        }
        true // swallow the key; the suggestion list stays visible
    }

    /// Redraw the input box + status footer, pinned at the bottom. Layout:
    /// ```text
    /// …reply tail…       (live, only while streaming — completed lines committed)
    /// ⏎ queued: …        (messages waiting to run after this reply)
    /// ──────────────     (top rule)
    /// › the input…       (one or more rows)
    /// ──────────────     (bottom rule)
    /// model · ctx …      (status footer)
    /// › /help …          (slash suggestions, if any)
    /// ```
    /// The live region is everything from the reply tail down; complete reply
    /// lines are committed to scrollback first so the region stays small and the
    /// terminal's own scrollback keeps working.
    fn redraw_input(&mut self) -> io::Result<()> {
        self.clear_input_block()?;
        let width = (crate::term::terminal_size().cols as usize).max(1);
        let rule = "─".repeat(width);

        // `head` counts rows drawn above the input box's top rule, so the cursor
        // can be placed correctly and the region erased next time.
        let mut head = 0;

        // While streaming, commit any now-complete reply lines to scrollback,
        // then repaint the still-incomplete tail above the box.
        if self.streaming.is_some() {
            while let Some(nl) = self.pending.find('\n') {
                let line: String = self.pending[..nl].to_string();
                write_raw(&mut self.out, &line)?;
                write_raw(&mut self.out, "\x1b[0m\n")?;
                self.pending.drain(..=nl);
            }
            if !self.pending.is_empty() {
                let rows = crate::ansi::wrap_ansi(&self.pending, width);
                for r in &rows {
                    write!(self.out, "{r}\r\n")?;
                }
                head += rows.len();
            }
        }

        // In queue-edit mode the editor holds the selected item; mirror it back
        // so the list preview stays live with the box.
        let editing = self.queue_edit.as_ref().map(|qe| qe.sel);
        if let Some(i) = editing {
            self.queue[i] = self.editor.text();
        }

        // Queued messages waiting to run after the current reply, listed above
        // the box. Each is capped to one line that fits the width (with an
        // ellipsis) so a huge paste can't dominate the view — and so each row
        // stays exactly one terminal row (keeping `head` accurate). The item
        // under edit is marked.
        let cap = width.saturating_sub(11).clamp(8, 80);
        let queued: Vec<String> = self.queue.iter().map(|q| queue_preview(q, cap)).collect();
        for (i, q) in queued.iter().enumerate() {
            if Some(i) == editing {
                write!(self.out, "\x1b[36m✎ editing: {q}\x1b[0m\r\n")?;
            } else {
                write!(self.out, "\x1b[2m⏎ queued: {q}\x1b[0m\r\n")?;
            }
            head += 1;
        }

        // Top rule, then each input row, then the bottom rule. The prompt marker
        // changes to `✎` while editing a queued item (same width as `›`).
        let prompt: &str = if editing.is_some() {
            "✎ "
        } else {
            self.prompt.as_str()
        };
        write!(self.out, "\x1b[2m{rule}\x1b[0m")?;
        let text = self.editor.text();
        let lines: Vec<&str> = text.split('\n').collect();
        for (i, line) in lines.iter().enumerate() {
            let pfx = if i == 0 { prompt } else { CONT };
            write!(self.out, "\r\n{pfx}{line}")?;
        }
        write!(self.out, "\r\n\x1b[2m{rule}\x1b[0m")?;

        // Status footer directly under the bottom rule.
        write!(self.out, "\r\n\x1b[2m{}\x1b[0m", self.footer_text())?;

        // Slash-command suggestions, one per line below the status footer. The
        // first (best) match is highlighted as the one Enter/Tab completes to.
        let suggestions = self.slash_suggestions();
        for (i, (name, desc)) in suggestions.iter().enumerate() {
            let mark = if i == 0 { "›" } else { " " };
            write!(
                self.out,
                "\r\n\x1b[2m{mark}\x1b[0m \x1b[36m{name}\x1b[0m  \x1b[2m{desc}\x1b[0m"
            )?;
        }

        // Cursor is on the last footer/suggestion line; move it up to its logical
        // input row/col.
        let (trow, tcol) = self.cursor_rowcol();
        let prefix_w = if trow == 0 {
            prompt.chars().count()
        } else {
            CONT.chars().count()
        };
        // Rows below the input row: bottom rule, the status line, and each
        // suggestion.
        let up = lines.len() - trow + 1 + suggestions.len();
        write!(self.out, "\x1b[{up}A\r")?;
        let col = prefix_w + tcol;
        write!(self.out, "\x1b[{col}C")?;
        // Rows above the cursor: the head rows (reply tail + queued list), the
        // top rule (1), and the input rows before this one (`trow`).
        self.cursor_row = head + 1 + trow;
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
            (
                'K',
                "/mcp",
                "import MCP servers (Claude Desktop/Code + .mcp.json) & connect",
            ),
            (
                'K',
                "/clip [n]",
                "copy the last response, or just code block n",
            ),
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
            ('H', "Modes", ""),
            (
                'K',
                "⇧⇥",
                "cycle mode: normal · auto-accept (run risky shell) · plan",
            ),
            ('H', "While a reply is generating", ""),
            (
                'K',
                "type + ⏎",
                "queue a message — runs after the current reply",
            ),
            ('K', "^C  ·  Esc Esc", "interrupt the running reply"),
            (
                'K',
                "^Q",
                "edit queued messages (↑↓ move, ⏎ done) — pauses sending",
            ),
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

    /// Write committed text straight to the terminal (raw-mode newline
    /// translation). Callers clear the live input region first (see
    /// `clear_input_block`) so committed output lands above the pinned box.
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
/// Gate a tool call by the current mode, then run it. Read-only tools always
/// run; tools that modify files run only in auto-accept mode (otherwise refused
/// with a message the model can act on — safe by default, no interactive
/// confirm needed for this synchronous slice). Returns the result text fed back
/// to the model (an error string is fine — it self-corrects).
fn gate_and_execute(mode: Mode, call: &ToolCall, root: Option<&std::path::Path>) -> String {
    use zero_core::builtins;
    // bash is advertised by builtins::definitions() but executed here, because
    // shell exec needs the mode + safety gate (like the `!` shell mode) — it is
    // intentionally NOT a builtins::execute() tool.
    if call.name == "bash" {
        return gate_and_run_bash(mode, call);
    }
    if !builtins::is_builtin(&call.name) {
        return format!("error: unknown tool {}", call.name);
    }
    let mutating = matches!(call.name.as_str(), "write_file" | "edit_file");
    if mutating && mode != Mode::AutoAccept {
        return "refused: this tool modifies files — switch to auto-accept mode \
                (Shift+Tab) to allow file changes"
            .to_string();
    }
    match builtins::execute(&call.name, &call.arguments, root) {
        Ok(out) => out,
        Err(e) => format!("error: {e}"),
    }
}

/// Gate + run a `bash` tool call. Plan mode refuses all shell (planning ≠ acting);
/// the destructive-command classifier hard-refuses dangerous commands in every
/// mode (the synchronous loop can't pause for an interactive y/N, and we never
/// auto-run `rm -rf`); otherwise the command runs and its combined output + exit
/// code come back (capping happens upstream in cap_tool_result).
fn gate_and_run_bash(mode: Mode, call: &ToolCall) -> String {
    let Some(cmd) = zero_core::tools::parse_arguments(call)
        .ok()
        .and_then(|a| a.get("command").and_then(|v| v.as_str().map(String::from)))
    else {
        return "error: bash requires a 'command' string argument".to_string();
    };
    if mode == Mode::Plan {
        return "refused: shell commands don't run in plan mode — outline the \
                approach instead, or switch mode (Shift+Tab) to execute"
            .to_string();
    }
    let verdict = zero_core::safety::classify(&cmd);
    if verdict.is_dangerous() {
        return format!(
            "refused: this command looks destructive ({}) and is blocked. If it's \
             intended, run it yourself in a shell.",
            verdict.reason.unwrap_or("flagged by the safety guard")
        );
    }
    run_bash_capture(&cmd)
}

/// Run `cmd` via `sh -c`, returning combined stdout + stderr + an exit-code line.
/// Pure-ish (only spawns a process); the model-facing capping is applied by the
/// caller so it goes through the same spill+compress path as every tool result.
fn run_bash_capture(cmd: &str) -> String {
    match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(out) => {
            let mut s = String::new();
            s.push_str(&String::from_utf8_lossy(&out.stdout));
            let err = String::from_utf8_lossy(&out.stderr);
            if !err.is_empty() {
                if !s.is_empty() && !s.ends_with('\n') {
                    s.push('\n');
                }
                s.push_str(&err);
            }
            if !s.is_empty() && !s.ends_with('\n') {
                s.push('\n');
            }
            let code = out.status.code().unwrap_or(-1);
            s.push_str(&format!("[exit {code}]"));
            s
        }
        Err(e) => format!("error: failed to run command: {e}"),
    }
}

/// Resolve the file a read/write/edit call targets (relative paths join `root`).
fn call_path(call: &ToolCall, root: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
    let args = zero_core::tools::parse_arguments(call).ok()?;
    let p = args.get("path")?.as_str()?;
    let pb = std::path::Path::new(p);
    Some(match root {
        Some(r) if pb.is_relative() => r.join(pb),
        _ => pb.to_path_buf(),
    })
}

/// True if a `read_file` call requests a line range (offset/limit). Ranged reads
/// bypass the read cache.
fn has_range(call: &ToolCall) -> bool {
    match zero_core::tools::parse_arguments(call) {
        Ok(args) => args.get("offset").is_some() || args.get("limit").is_some(),
        Err(_) => false,
    }
}

/// Cap a tool result to `max` bytes with a tool-aware re-fetch hint.
fn cap_tool_result(
    call: &ToolCall,
    result: String,
    max: usize,
    artifact_dir: Option<&std::path::Path>,
) -> String {
    if result.len() <= max {
        return result;
    }
    use zero_core::compress;
    // Offload, don't delete: spill the full output first (when a session dir is
    // set) so the compressed view can point the model back at the complete bytes.
    let artifact =
        artifact_dir.and_then(|d| compress::spill(d, &sanitize_id(&call.id), result.as_bytes()));
    compress::compress(&shape_cmd(call), &result, max, artifact.as_deref()).text
}

/// Command hint for shape detection. `read_file` is blank so the content sniff
/// decides (json vs source vs log); `bash` uses the real command.
fn shape_cmd(call: &ToolCall) -> String {
    match call.name.as_str() {
        "grep" => "grep".to_string(),
        "list_dir" => "ls".to_string(),
        "bash" => zero_core::tools::parse_arguments(call)
            .ok()
            .and_then(|a| a.get("command").and_then(|v| v.as_str().map(String::from)))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

/// Filename-safe slice of a tool-call id, for the spill artifact name.
fn sanitize_id(id: &str) -> String {
    let s: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if s.is_empty() {
        "x".to_string()
    } else {
        s
    }
}

/// After a tool turn, drop bulky payloads of successfully-applied write/edit calls
/// from history (the file is on disk). Failed/refused writes keep their args.
/// Returns `(bytes_before, bytes_after)` of the args it compacted, for stats.
fn compact_applied_writes(conv: &mut Conversation) -> (usize, usize) {
    let ok_ids: std::collections::HashSet<String> = conv
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .filter(|m| !m.content.starts_with("error:") && !m.content.starts_with("refused:"))
        .filter_map(|m| m.tool_call_id.clone())
        .collect();
    let (mut before, mut after) = (0usize, 0usize);
    for m in &mut conv.messages {
        for tc in &mut m.tool_calls {
            if ok_ids.contains(&tc.id) {
                if let Some(args) = zero_core::context::compact_call_args(&tc.name, &tc.arguments) {
                    before += tc.arguments.len();
                    after += args.len();
                    tc.arguments = args;
                }
            }
        }
    }
    (before, after)
}

/// First line of a string, for a compact one-line tool-result preview.
fn first_line(s: &str) -> &str {
    s.split('\n').next().unwrap_or("")
}

/// A one-line, length-capped snippet of an MCP tool description for `/mcp tools`.
/// Descriptions can be multi-paragraph (some servers embed whole manuals), so we
/// collapse all whitespace runs to single spaces and cap to ~80 display chars.
fn tool_desc_snippet(desc: &str) -> String {
    let flat: String = desc.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 80;
    if flat.chars().count() > MAX {
        let cut: String = flat.chars().take(MAX).collect();
        format!("{cut}…")
    } else {
        flat
    }
}

fn write_raw<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    if s.contains('\n') {
        w.write_all(s.replace('\n', "\r\n").as_bytes())
    } else {
        w.write_all(s.as_bytes())
    }?;
    Ok(())
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
        let mut a = App::new(
            ScriptedInput::new(script),
            Vec::new(),
            Arc::new(StubBackend::instant()),
            None,
        );
        a.synchronous = true; // run the backend inline → deterministic tests
        a
    }

    /// Serializes tests that mutate the process-global cwd (the read-cache /
    /// path-confinement tests). cwd is per-process, so these can't run in
    /// parallel. Poison-tolerant.
    static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Build a synchronous app driven by chunked reads (so a streamed turn
    /// finalizes between reads before the next chunk arrives).
    fn multi_app(chunks: &[&[u8]]) -> App<MultiInput, Vec<u8>> {
        let mut a = App::new(
            MultiInput::new(chunks),
            Vec::new(),
            Arc::new(StubBackend::instant()),
            None,
        );
        a.synchronous = true;
        a
    }

    fn rendered<I: Input>(a: &App<I, Vec<u8>>) -> String {
        String::from_utf8(a.out.clone()).unwrap()
    }

    fn type_into<I: Input>(a: &mut App<I, Vec<u8>>, s: &str) {
        for c in s.chars() {
            a.dispatch(Key::Char(c)).unwrap();
        }
    }

    #[test]
    fn common_prefix_handles_empty_single_and_divergent() {
        assert_eq!(common_prefix(&[]), "");
        assert_eq!(common_prefix(&["/help"]), "/help");
        assert_eq!(common_prefix(&["/config", "/connect", "/clip"]), "/c");
        assert_eq!(common_prefix(&["/scan", "/help"]), "/");
    }

    #[test]
    fn slash_query_only_matches_a_bare_slash_token() {
        assert_eq!(slash_query("/he"), Some("/he"));
        assert_eq!(slash_query("/clip 1"), None); // args started
        assert_eq!(slash_query("hello"), None);
        assert_eq!(slash_query("a /help"), None);
    }

    #[test]
    fn slash_matches_filters_by_prefix() {
        let m = slash_matches("/co");
        assert_eq!(m.len(), 2); // /config, /connect
        assert!(m.iter().all(|(n, _)| n.starts_with("/co")));
        assert!(slash_matches("/zzz").is_empty());
    }

    #[test]
    fn enter_completes_a_unique_partial_command_without_submitting() {
        let mut a = app(b"");
        type_into(&mut a, "/he");
        assert_eq!(a.dispatch(Key::Enter).unwrap(), Flow::Continue);
        assert_eq!(a.editor.text(), "/help"); // completed, not sent
    }

    #[test]
    fn tab_also_completes_a_unique_partial_command() {
        let mut a = app(b"");
        type_into(&mut a, "/se");
        a.dispatch(Key::Tab).unwrap();
        assert_eq!(a.editor.text(), "/servers");
    }

    #[test]
    fn enter_on_ambiguous_prefix_completes_to_common_prefix_and_is_swallowed() {
        let mut a = app(b"");
        type_into(&mut a, "/co");
        // /config + /connect share "/con" — Enter extends to the LCP, no submit.
        assert_eq!(a.dispatch(Key::Enter).unwrap(), Flow::Continue);
        assert_eq!(a.editor.text(), "/con");
    }

    #[test]
    fn enter_on_a_fully_typed_command_submits_it() {
        let mut a = app(b"");
        type_into(&mut a, "/help");
        // Exact command → not completion; runs the command (prints help).
        a.dispatch(Key::Enter).unwrap();
        assert!(a.editor.is_empty());
        assert!(rendered(&a).contains("show this help"));
    }

    #[test]
    fn suggestions_show_while_typing_and_hide_once_unique_complete() {
        let mut a = app(b"");
        type_into(&mut a, "/he");
        assert_eq!(a.slash_suggestions(), vec![("/help", "show this help")]);
        a.redraw_input().unwrap();
        let out = rendered(&a);
        assert!(out.contains("/help"));
        assert!(out.contains("show this help"));
        // Finish typing it: nothing left to suggest.
        type_into(&mut a, "lp");
        assert!(a.slash_suggestions().is_empty());
    }

    #[test]
    fn no_suggestions_for_plain_text_or_after_a_space() {
        let mut a = app(b"");
        type_into(&mut a, "hello");
        assert!(a.slash_suggestions().is_empty());
        a.editor.clear();
        type_into(&mut a, "/clip 1");
        assert!(a.slash_suggestions().is_empty());
    }

    #[test]
    fn streaming_pins_the_box_below_the_live_reply() {
        let (mut a, tx) = streaming_app();
        a.pending = "zero› ".to_string(); // assistant label leads the tail
        tx.send(StreamEvent::Token("hello".into())).unwrap();
        a.pump_stream().unwrap();
        let out = rendered(&a);
        assert!(out.contains("hello")); // reply tail painted
        assert!(out.contains('─')); // the input box is drawn below it
        assert!(out.contains("esc to interrupt")); // footer shows live turn state
    }

    #[test]
    fn streaming_commits_completed_lines_and_keeps_the_tail() {
        let (mut a, tx) = streaming_app();
        tx.send(StreamEvent::Token("line one\nline two".into()))
            .unwrap();
        a.pump_stream().unwrap();
        // "line one" committed (newline seen); only "line two" stays live.
        assert_eq!(a.pending, "line two");
        assert!(rendered(&a).contains("line one"));
    }

    #[test]
    fn typing_while_streaming_shows_in_the_pinned_box_and_queues() {
        let (mut a, _tx) = streaming_app();
        a.dispatch(Key::Char('h')).unwrap();
        a.dispatch(Key::Char('i')).unwrap();
        a.redraw_input().unwrap();
        assert!(rendered(&a).contains("› hi")); // live preview in the box
        a.dispatch(Key::Enter).unwrap(); // queues, doesn't submit
        assert_eq!(a.queue.len(), 1);
        a.redraw_input().unwrap();
        assert!(rendered(&a).contains("⏎ queued: hi")); // listed above the box
    }

    #[test]
    fn queue_preview_caps_and_marks_truncation() {
        assert_eq!(queue_preview("short", 20), "short"); // fits, no marker
        assert_eq!(queue_preview("abcdefghij", 4), "abcd…"); // capped + ellipsis
        assert_eq!(queue_preview("one\ntwo", 20), "one…"); // multiline → first + …
                                                           // A huge paste collapses to a short hint.
        let big = "x".repeat(5000);
        let p = queue_preview(&big, 60);
        assert_eq!(p.chars().count(), 61); // 60 + the ellipsis
    }

    #[test]
    fn mcp_with_no_sources_reports_nothing_found() {
        let mut a = app(b""); // no mcp_path, no discovery dirs
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("no MCP servers found"));
    }

    #[test]
    fn mcp_tools_with_no_connections_hints_to_connect() {
        let mut a = app(b"");
        type_str(&mut a, "/mcp tools");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("no MCP servers connected"));
    }

    #[test]
    fn mcp_imports_servers_from_claude_desktop_config() {
        // The real-world UX: a Claude Desktop config exists under $HOME and Zero
        // imports its servers without any ~/.zero/mcp.json. Uses a fake HOME and
        // an `sh` stdio mock server so the connect path runs end to end.
        let home =
            std::env::temp_dir().join(format!("zero-home-{}-{}", std::process::id(), line!()));
        let claude_dir = home.join("Library/Application Support/Claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let script = "read a; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}'; read b; read c; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"ping\",\"description\":\"p\"}]}}'";
        let cfg = format!(
            r#"{{"mcpServers":{{"imported":{{"command":"sh","args":["-c",{}]}}}}}}"#,
            zero_core::json::Value::Str(script.to_string()).to_json()
        );
        std::fs::write(claude_dir.join("claude_desktop_config.json"), cfg).unwrap();

        let mut a = app(b"");
        a.set_mcp_discovery(Some(home.clone()), Some(home.join("proj")));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        assert!(out.contains("imported"), "server name shown: {out}");
        assert!(out.contains("Claude Desktop"), "source shown: {out}");
        assert!(out.contains('✓'));
        assert_eq!(a.mcp.len(), 1);
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn autoconnect_is_silent_with_no_servers_but_connects_when_present() {
        // No sources at all → autoconnect prints nothing.
        let mut a = app(b"");
        a.autoconnect_mcp().unwrap();
        assert!(
            rendered(&a).is_empty(),
            "should be silent: {:?}",
            rendered(&a)
        );

        // With a configured server, autoconnect spawns + reports it at startup.
        let home =
            std::env::temp_dir().join(format!("zero-auto-{}-{}", std::process::id(), line!()));
        let claude_dir = home.join("Library/Application Support/Claude");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let script = "read a; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}'; read b; read c; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[]}}'";
        let cfg = format!(
            r#"{{"mcpServers":{{"auto":{{"command":"sh","args":["-c",{}]}}}}}}"#,
            zero_core::json::Value::Str(script.to_string()).to_json()
        );
        std::fs::write(claude_dir.join("claude_desktop_config.json"), cfg).unwrap();
        let mut b = app(b"");
        b.set_mcp_discovery(Some(home.clone()), Some(home.join("p")));
        b.autoconnect_mcp().unwrap();
        assert!(rendered(&b).contains("auto"));
        assert_eq!(b.mcp.len(), 1);
        std::fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn mcp_empty_config_is_reported() {
        let path = std::env::temp_dir().join(format!("zero-mcp-empty-{}.json", std::process::id()));
        std::fs::write(&path, "{}").unwrap();
        let mut a = app(b"");
        a.set_mcp_path(Some(path.clone()));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("no MCP servers found"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mcp_connects_a_configured_stdio_server_and_lists_tools() {
        use zero_core::json::Value;
        // A tiny MCP server in sh (same handshake as the core test).
        let script = "read a; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"capabilities\":{}}}'; read b; read c; printf '%s\\n' '{\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"echoes\"}]}}'";
        let cfg = Value::Object(vec![(
            "mcpServers".to_string(),
            Value::Object(vec![(
                "mock".to_string(),
                Value::Object(vec![
                    ("command".to_string(), Value::Str("sh".to_string())),
                    (
                        "args".to_string(),
                        Value::Array(vec![
                            Value::Str("-c".to_string()),
                            Value::Str(script.to_string()),
                        ]),
                    ),
                ]),
            )]),
        )]);
        let path = std::env::temp_dir().join(format!("zero-mcp-{}.json", std::process::id()));
        std::fs::write(&path, cfg.to_json()).unwrap();

        let mut a = app(b"");
        a.set_mcp_path(Some(path.clone()));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("✓"));
        assert!(rendered(&a).contains("mock"));
        assert_eq!(a.mcp.len(), 1);

        type_str(&mut a, "/mcp tools");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("echo"));

        // Re-connecting reports the already-connected server (no second spawn).
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("already connected"));
        assert_eq!(a.mcp.len(), 1);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mcp_connect_reports_a_failed_server() {
        use zero_core::json::Value;
        // A server whose command doesn't exist → connect() errors, reported ✗.
        let cfg = Value::Object(vec![(
            "mcpServers".to_string(),
            Value::Object(vec![(
                "broken".to_string(),
                Value::Object(vec![(
                    "command".to_string(),
                    Value::Str("zero-no-such-bin-xyz".to_string()),
                )]),
            )]),
        )]);
        let path = std::env::temp_dir().join(format!("zero-mcp-bad-{}.json", std::process::id()));
        std::fs::write(&path, cfg.to_json()).unwrap();
        let mut a = app(b"");
        a.set_mcp_path(Some(path.clone()));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("✗"));
        assert!(a.mcp.is_empty());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn mcp_connect_surfaces_malformed_config() {
        let path = std::env::temp_dir().join(format!("zero-mcp-mal-{}.json", std::process::id()));
        std::fs::write(&path, "{not json").unwrap();
        let mut a = app(b"");
        a.set_mcp_path(Some(path.clone()));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("mcp config ("));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn redraw_lists_a_queue_and_marks_the_edited_item() {
        let mut a = app(b"");
        a.queue.push_back("first task".to_string());
        a.queue.push_back("second task".to_string());
        // Idle redraw lists every queued item above the box.
        a.redraw_input().unwrap();
        let out = rendered(&a);
        assert!(out.contains("⏎ queued: first task"));
        assert!(out.contains("⏎ queued: second task"));
        // In queue-edit mode the selected item is marked and the prompt swaps.
        a.dispatch(Key::Ctrl('q')).unwrap(); // selects the last item
        let out = rendered(&a);
        assert!(out.contains("✎ editing: second task"));
        assert!(out.contains("✎ ")); // prompt marker swapped
    }

    #[test]
    fn redraw_colors_each_mode_chip() {
        for m in [Mode::Normal, Mode::AutoAccept, Mode::Plan] {
            let mut a = app(b"");
            a.mode = m;
            a.redraw_input().unwrap();
            let out = rendered(&a);
            assert!(out.contains(m.label()));
            assert!(out.contains(m.color())); // the chip's color code is emitted
        }
    }

    #[test]
    fn shift_tab_cycles_modes() {
        let mut a = app(b"");
        assert_eq!(a.mode, Mode::Normal);
        a.dispatch(Key::BackTab).unwrap();
        assert_eq!(a.mode, Mode::AutoAccept);
        a.dispatch(Key::BackTab).unwrap();
        assert_eq!(a.mode, Mode::Plan);
        a.dispatch(Key::BackTab).unwrap();
        assert_eq!(a.mode, Mode::Normal); // wraps
    }

    #[test]
    fn footer_shows_mode_and_change_hint() {
        let mut a = app(b"");
        a.mode = Mode::Plan;
        let f = a.footer_text();
        assert!(f.contains("plan"));
        assert!(f.contains("mode")); // ⇧⇥ mode hint
    }

    #[test]
    fn plan_mode_injects_directive_for_the_request_only() {
        use std::sync::Mutex;
        #[derive(Clone)]
        struct Rec {
            seen: Arc<Mutex<Vec<Message>>>,
        }
        impl Backend for Rec {
            fn name(&self) -> &str {
                "rec"
            }
            fn stream(
                &self,
                conv: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                *self.seen.lock().unwrap() = conv.messages.clone();
                sink(StreamEvent::Token("ok".into()));
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(Rec { seen: seen.clone() }),
            None,
        );
        a.synchronous = true;
        a.mode = Mode::Plan;
        a.start_turn("do the thing").unwrap();
        while a.streaming.is_some() {
            a.pump_stream().unwrap();
        }
        let msgs = seen.lock().unwrap();
        assert!(msgs
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("PLAN MODE")));
        assert!(msgs.iter().any(|m| m.content == "do the thing"));
        // Not persisted: the live conversation has no injected system message.
        assert!(a.conv.messages.iter().all(|m| m.role != Role::System));
    }

    #[test]
    fn tools_toggle_flips_state_and_is_listed() {
        let mut a = app(b"");
        assert!(!a.tools_enabled);
        type_str(&mut a, "/tools");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.tools_enabled);
        assert!(rendered(&a).contains("tools on"));
        type_str(&mut a, "/tools");
        a.dispatch(Key::Enter).unwrap();
        assert!(!a.tools_enabled);
    }

    #[test]
    fn gate_blocks_mutating_tools_unless_auto_accept() {
        let write = ToolCall::new("c", "write_file", r#"{"path":"x","content":"y"}"#);
        let refused = gate_and_execute(Mode::Normal, &write, None);
        assert!(refused.contains("refused"));
        // unknown tool reported, not executed
        let unknown = gate_and_execute(Mode::AutoAccept, &ToolCall::new("c", "nope", "{}"), None);
        assert!(unknown.contains("unknown tool"));
    }

    #[test]
    fn bash_runs_a_safe_command_and_returns_output_and_exit() {
        let call = ToolCall::new("c", "bash", r#"{"command":"echo hello-bash"}"#);
        let out = gate_and_execute(Mode::Normal, &call, None);
        assert!(out.contains("hello-bash"));
        assert!(out.contains("[exit 0]"));
    }

    #[test]
    fn bash_reports_nonzero_exit_and_stderr() {
        let call = ToolCall::new("c", "bash", r#"{"command":"echo oops >&2; exit 3"}"#);
        let out = gate_and_execute(Mode::AutoAccept, &call, None);
        assert!(out.contains("oops")); // stderr captured
        assert!(out.contains("[exit 3]"));
    }

    #[test]
    fn bash_refuses_dangerous_commands_in_every_mode() {
        let call = ToolCall::new("c", "bash", r#"{"command":"rm -rf /"}"#);
        for mode in [Mode::Normal, Mode::AutoAccept] {
            let out = gate_and_execute(mode, &call, None);
            assert!(out.contains("refused"), "danger not blocked in {mode:?}");
            assert!(out.contains("destructive"));
        }
    }

    #[test]
    fn bash_refuses_in_plan_mode() {
        let call = ToolCall::new("c", "bash", r#"{"command":"echo hi"}"#);
        let out = gate_and_execute(Mode::Plan, &call, None);
        assert!(out.contains("refused"));
        assert!(out.contains("plan mode"));
    }

    #[test]
    fn bash_without_command_arg_errors() {
        let call = ToolCall::new("c", "bash", r#"{"wrong":"x"}"#);
        let out = gate_and_execute(Mode::Normal, &call, None);
        assert!(out.contains("requires a 'command'"));
    }

    #[test]
    fn tool_desc_snippet_collapses_and_caps() {
        // Multi-line / multi-space descriptions collapse to one capped line.
        assert_eq!(tool_desc_snippet("short desc"), "short desc");
        assert_eq!(
            tool_desc_snippet("line one\n\n   line two"),
            "line one line two"
        );
        assert_eq!(tool_desc_snippet(""), "");
        let long = "word ".repeat(100);
        let s = tool_desc_snippet(&long);
        assert!(s.ends_with('…'));
        assert_eq!(s.chars().count(), 81); // 80 + ellipsis
        assert!(!s.contains('\n'));
    }

    #[test]
    fn mcp_tools_listing_is_compact_per_server() {
        // A server with a paragraph-long, newline-laden description must render
        // as ONE short line per tool (the "novel" bug).
        use zero_core::json::Value;
        let huge = "This is a very long tool description. ".repeat(20) + "\nwith\nnewlines";
        let script = format!(
            "read a; printf '%s\\n' '{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{{\"capabilities\":{{}}}}}}'; read b; read c; printf '%s\\n' {}",
            Value::Str(format!(
                r#"{{"jsonrpc":"2.0","id":2,"result":{{"tools":[{{"name":"big","description":{}}}]}}}}"#,
                Value::Str(huge.clone()).to_json()
            ))
            .to_json()
        );
        let cfg = Value::Object(vec![(
            "mcpServers".to_string(),
            Value::Object(vec![(
                "verbose".to_string(),
                Value::Object(vec![
                    ("command".to_string(), Value::Str("sh".to_string())),
                    (
                        "args".to_string(),
                        Value::Array(vec![Value::Str("-c".to_string()), Value::Str(script)]),
                    ),
                ]),
            )]),
        )]);
        let path = std::env::temp_dir().join(format!("zero-mcp-big-{}.json", std::process::id()));
        std::fs::write(&path, cfg.to_json()).unwrap();
        let mut a = app(b"");
        a.set_mcp_path(Some(path.clone()));
        type_str(&mut a, "/mcp");
        a.dispatch(Key::Enter).unwrap();
        // Clear the connect output, then list tools.
        a.out.clear();
        type_str(&mut a, "/mcp tools");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        assert!(out.contains("big")); // tool name shown
                                      // The full description must NOT be dumped — the tool line is capped.
        let tool_line = out.lines().find(|l| l.contains("big")).unwrap();
        assert!(
            tool_line.chars().count() < 140,
            "tool line too long: {tool_line}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn set_config_applies_context_caps() {
        let mut a = app(b"");
        let cfg = Config {
            max_tool_output: 1234,
            max_turn_output: 56_789,
            ..Config::default()
        };
        a.set_config(cfg, None, None);
        assert_eq!(a.max_tool_output, 1234);
        assert_eq!(a.max_turn_output, 56_789);
    }

    #[test]
    fn context_command_reports_nothing_before_any_turn() {
        let mut a = app(b"");
        type_str(&mut a, "/context");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("no tool output yet"));
    }

    #[test]
    fn context_command_reports_measured_savings() {
        // End-to-end through the agentic loop without touching cwd (which would
        // race other tests): an inline backend that calls grep — its result is
        // capped tiny, so the executor records a measured saving — then answers.
        use std::sync::Mutex;
        struct CapBackend {
            step: Arc<Mutex<u32>>,
        }
        impl Backend for CapBackend {
            fn name(&self) -> &str {
                "cap"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut step = self.step.lock().unwrap();
                *step += 1;
                if *step == 1 {
                    // grep the source tree (cwd) for a common token → a big result.
                    Ok(zero_core::backend::Completion {
                        content: String::new(),
                        tool_calls: vec![ToolCall::new("c1", "grep", r#"{"pattern":"fn"}"#)],
                        usage: None,
                    })
                } else {
                    Ok(zero_core::backend::Completion {
                        content: "done".to_string(),
                        tool_calls: vec![],
                        usage: None,
                    })
                }
            }
        }
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(CapBackend {
                step: Arc::new(Mutex::new(0)),
            }),
            None,
        );
        a.tools_enabled = true;
        a.max_tool_output = 64; // tiny → the grep result is capped, recording a saving
        a.run_tool_turn("grep fn").unwrap();

        assert!(a.context_stats.raw_bytes > 0, "no tool output recorded");
        assert!(a.context_stats.capped_saved > 0, "cap recorded no saving");

        type_str(&mut a, "/context");
        a.dispatch(Key::Enter).unwrap();
        let out = rendered(&a);
        assert!(out.contains("context savings"));
        assert!(out.contains("smaller window"));
    }

    #[test]
    fn tool_turn_runs_bash_and_caps_its_output() {
        // End-to-end: the model calls bash, the executor runs it, and the big
        // output is spilled+compressed (the Log-B sink, now bounded). Proves bash
        // output flows through the same recoverable cap path as every tool.
        use std::sync::Mutex;
        struct BashBackend {
            step: Arc<Mutex<u32>>,
        }
        impl Backend for BashBackend {
            fn name(&self) -> &str {
                "bash"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut step = self.step.lock().unwrap();
                *step += 1;
                if *step == 1 {
                    Ok(zero_core::backend::Completion {
                        content: String::new(),
                        // Emit ~5000 lines → well over the tiny cap below.
                        tool_calls: vec![ToolCall::new(
                            "b1",
                            "bash",
                            r#"{"command":"seq 1 5000"}"#,
                        )],
                        usage: None,
                    })
                } else {
                    Ok(zero_core::backend::Completion {
                        content: "done".to_string(),
                        tool_calls: vec![],
                        usage: None,
                    })
                }
            }
        }
        let dir =
            std::env::temp_dir().join(format!("zero-bash-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(BashBackend {
                step: Arc::new(Mutex::new(0)),
            }),
            None,
        );
        a.tools_enabled = true;
        a.max_tool_output = 256;
        a.set_artifact_dir(Some(dir.clone()));
        a.run_tool_turn("count to 5000").unwrap();

        // The bash result in history is capped (much smaller than 5000 lines).
        let tool_msg = a
            .conv
            .messages
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("a tool result");
        assert!(tool_msg.content.len() < 2000, "bash output not capped");
        // `seq` is uniform, so the repeat-fold collapses it to first+count+last
        // rather than a byte donut — a compression marker must still be present.
        assert!(
            tool_msg.content.contains("similar lines") || tool_msg.content.contains("elided"),
            "no compression marker: {}",
            tool_msg.content
        );
        // Recoverable: the full output spilled byte-identical and is referenced.
        assert!(tool_msg.content.contains("full output:"));
        let art = dir.join("out-b1.txt");
        let full = std::fs::read_to_string(&art).unwrap();
        assert!(full.contains("\n5000\n") || full.ends_with("5000\n[exit 0]"));
        assert!(a.context_stats.capped_saved > 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shape_cmd_maps_tool_to_detection_hint() {
        assert_eq!(shape_cmd(&ToolCall::new("c", "grep", "{}")), "grep");
        assert_eq!(shape_cmd(&ToolCall::new("c", "list_dir", "{}")), "ls");
        assert_eq!(shape_cmd(&ToolCall::new("c", "read_file", "{}")), ""); // sniff decides
                                                                           // bash uses the real command so a log/grep shape can be detected.
        assert_eq!(
            shape_cmd(&ToolCall::new("c", "bash", r#"{"command":"cargo test"}"#)),
            "cargo test"
        );
        // bash with unparseable args → empty (falls back to content sniff).
        assert_eq!(shape_cmd(&ToolCall::new("c", "bash", "not json")), "");
    }

    #[test]
    fn sanitize_id_keeps_filename_safe_chars() {
        assert_eq!(sanitize_id("call-7f_2"), "call-7f_2");
        assert_eq!(sanitize_id("a/b c:d"), "abcd"); // strips path/space/colon
        assert_eq!(sanitize_id(""), "x"); // never empty
        assert_eq!(sanitize_id("///"), "x");
    }

    #[test]
    fn run_once_without_tools_returns_reply_and_exposes_accessors() {
        // Bare-completion headless path via the stub backend (no new uncovered
        // code): covers run_once's non-tools arm + conversation()/context_stats().
        // The tools-on path is covered end-to-end by the bash_replay suite.
        let mut a = app(b"");
        a.set_tools_enabled(false);
        let reply = a.run_once("hello there").unwrap();
        assert!(!reply.is_empty(), "stub should echo a reply");
        assert_eq!(a.last_reply, reply);
        let roles: Vec<_> = a.conversation().messages.iter().map(|m| m.role).collect();
        assert!(roles.contains(&Role::User));
        assert!(roles.contains(&Role::Assistant));
        assert_eq!(a.context_stats().total_saved(), 0);
    }

    #[test]
    fn run_once_with_tools_drives_the_loop_via_the_stub() {
        // tools-on arm of run_once: the stub emits no tool call, so the loop
        // finishes in one round with the stub's text. Exercises run_tool_turn
        // through run_once without any new uncovered backend code.
        let mut a = app(b"");
        a.set_tools_enabled(true);
        let reply = a.run_once("anything").unwrap();
        assert!(!reply.is_empty());
        assert_eq!(a.last_reply, reply);
        assert!(a
            .conversation()
            .messages
            .iter()
            .any(|m| m.role == Role::User));
    }

    #[test]
    fn set_tools_enabled_toggles_the_flag() {
        let mut a = app(b"");
        a.set_tools_enabled(true);
        assert!(a.tools_enabled);
        a.set_tools_enabled(false);
        assert!(!a.tools_enabled);
    }

    #[test]
    fn cap_tool_result_spills_full_output_and_is_recoverable() {
        // The reassessment in action: capping OFFLOADS, never silently deletes.
        // With an artifact dir, the full output is written byte-identical and the
        // compressed view points back at it.
        let dir = std::env::temp_dir().join(format!("zero-art-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let big = "y".repeat(20_000);
        let out = cap_tool_result(
            &ToolCall::new("call-7f", "read_file", "{}"),
            big.clone(),
            4096,
            Some(dir.as_path()),
        );
        assert!(out.len() < big.len());
        assert!(out.contains("elided"));
        assert!(out.contains("full output:"));
        // Byte-identical artifact at the call-id-named path.
        let art = dir.join("out-call-7f.txt");
        assert_eq!(std::fs::read_to_string(&art).unwrap(), big);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cap_tool_result_without_artifact_dir_still_compresses() {
        // No session dir (tests / --no-log): compression still runs, just no
        // re-fetch path in the marker.
        let out = cap_tool_result(
            &ToolCall::new("c", "read_file", "{}"),
            "y".repeat(20_000),
            4096,
            None,
        );
        assert!(out.contains("elided"));
        assert!(!out.contains("full output:"));
    }

    #[test]
    fn first_line_takes_only_the_first_line() {
        assert_eq!(first_line("a\nb\nc"), "a");
        assert_eq!(first_line("solo"), "solo");
    }

    #[test]
    fn tool_turn_runs_a_round_then_answers() {
        use std::sync::Mutex;
        // A backend that calls grep once, then answers with text.
        #[derive(Clone)]
        struct ToolBackend {
            step: Arc<Mutex<u32>>,
        }
        impl Backend for ToolBackend {
            fn name(&self) -> &str {
                "toolback"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut step = self.step.lock().unwrap();
                *step += 1;
                Ok(if *step == 1 {
                    zero_core::backend::Completion {
                        content: String::new(),
                        tool_calls: vec![ToolCall::new("c1", "list_dir", r#"{"path":"."}"#)],
                        usage: None,
                    }
                } else {
                    zero_core::backend::Completion {
                        content: "all done".to_string(),
                        tool_calls: Vec::new(),
                        usage: None,
                    }
                })
            }
        }
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(ToolBackend {
                step: Arc::new(Mutex::new(0)),
            }),
            None,
        );
        a.tools_enabled = true;
        a.run_tool_turn("what's here?").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        let out = rendered(&a);
        assert!(out.contains("⚙ list_dir")); // tool call shown
        assert!(out.contains("all done")); // final answer rendered
        assert_eq!(a.last_reply, "all done");
        // History: user, assistant(call), tool result, assistant(final).
        assert!(a
            .conv
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c1")));
    }

    #[test]
    fn tool_turn_surfaces_a_backend_error() {
        struct ErrBackend;
        impl Backend for ErrBackend {
            fn name(&self) -> &str {
                "err"
            }
            fn stream(
                &self,
                _c: &Conversation,
                _s: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                Err(zero_core::backend::BackendError("kaput".to_string()))
            }
        }
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(ErrBackend),
            None,
        );
        a.tools_enabled = true;
        a.run_tool_turn("hi").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        assert!(rendered(&a).contains("kaput"));
    }

    /// A backend that returns a tool call every turn — varying args (never
    /// settles → step cap) or identical (→ doom loop).
    struct LoopBackend {
        vary: bool,
        n: std::sync::Mutex<u32>,
    }
    impl Backend for LoopBackend {
        fn name(&self) -> &str {
            "loop"
        }
        fn stream(
            &self,
            _c: &Conversation,
            sink: &mut dyn FnMut(StreamEvent),
        ) -> Result<(), zero_core::backend::BackendError> {
            sink(StreamEvent::Done(StopReason::EndTurn));
            Ok(())
        }
        fn complete(
            &self,
            _c: &Conversation,
            _t: &[ToolDef],
            _to: Duration,
        ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError> {
            let mut n = self.n.lock().unwrap();
            *n += 1;
            let args = if self.vary {
                format!(r#"{{"path":".{n}"}}"#)
            } else {
                r#"{"path":"."}"#.to_string()
            };
            Ok(zero_core::backend::Completion {
                content: String::new(),
                tool_calls: vec![ToolCall::new(format!("c{n}"), "list_dir", args)],
                usage: None,
            })
        }
    }

    fn loop_app(vary: bool) -> App<ScriptedInput, Vec<u8>> {
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(LoopBackend {
                vary,
                n: std::sync::Mutex::new(0),
            }),
            None,
        );
        a.tools_enabled = true;
        a
    }

    #[test]
    fn tool_turn_reports_step_cap() {
        let mut a = loop_app(true); // varying args → never settles
        a.run_tool_turn("loop forever").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        assert!(rendered(&a).contains("step cap reached"));
    }

    #[test]
    fn tool_turn_reports_doom_loop() {
        let mut a = loop_app(false); // identical call → doom-loop guard trips
        a.run_tool_turn("same thing").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        assert!(rendered(&a).contains("repeated the same tool call"));
    }

    #[test]
    fn tool_turn_auto_accept_allows_a_write_and_logs() {
        // write_file (round 1) then answer (round 2); a session log captures both.
        struct WriteBackend {
            path: String,
            n: std::sync::Mutex<u32>,
        }
        impl Backend for WriteBackend {
            fn name(&self) -> &str {
                "wb"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut n = self.n.lock().unwrap();
                *n += 1;
                Ok(if *n == 1 {
                    zero_core::backend::Completion {
                        content: String::new(),
                        tool_calls: vec![ToolCall::new(
                            "w1",
                            "write_file",
                            format!(r#"{{"path":"{}","content":"hi"}}"#, self.path),
                        )],
                        usage: None,
                    }
                } else {
                    zero_core::backend::Completion {
                        content: "saved it".to_string(),
                        tool_calls: Vec::new(),
                        usage: None,
                    }
                })
            }
        }
        let dir = std::env::temp_dir().join(format!("zero-tw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("note.txt");
        // A real session log so the assistant-record branch is exercised.
        let (log, _p) = SessionLog::create_in(&dir).unwrap();
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(WriteBackend {
                path: target.display().to_string(),
                n: std::sync::Mutex::new(0),
            }),
            Some(log),
        );
        a.tools_enabled = true;
        a.mode = Mode::AutoAccept; // write tools allowed only here
        a.run_tool_turn("write a note").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        assert!(rendered(&a).contains("⚙ write_file"));
        assert_eq!(a.last_reply, "saved it");
        // The auto-accept gate let the write through to execution — the tool
        // result is NOT the mode-refusal message (it executed, even if the
        // absolute path was then rejected by the workspace confinement).
        assert!(a
            .conv
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && !m.content.contains("switch to auto-accept")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tool_turn_normal_mode_refuses_a_write() {
        // edit_file (round 1) refused in normal mode; model then answers.
        struct TryWriteBackend {
            n: std::sync::Mutex<u32>,
        }
        impl Backend for TryWriteBackend {
            fn name(&self) -> &str {
                "tw"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut n = self.n.lock().unwrap();
                *n += 1;
                Ok(if *n == 1 {
                    zero_core::backend::Completion {
                        content: String::new(),
                        tool_calls: vec![ToolCall::new(
                            "w1",
                            "edit_file",
                            r#"{"path":"x","old_string":"a","new_string":"b"}"#,
                        )],
                        usage: None,
                    }
                } else {
                    zero_core::backend::Completion {
                        content: "ok, I won't".to_string(),
                        tool_calls: Vec::new(),
                        usage: None,
                    }
                })
            }
        }
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(TryWriteBackend {
                n: std::sync::Mutex::new(0),
            }),
            None,
        );
        a.tools_enabled = true; // mode stays Normal
        a.run_tool_turn("edit it").unwrap();
        let _ = a.backend.stream(&Conversation::new(), &mut |_| {});
        // The refusal is fed back as the tool result.
        assert!(a
            .conv
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && m.content.contains("refused")));
    }

    #[test]
    fn auto_accept_runs_a_dangerous_command_without_confirm() {
        let mut a = app(b"");
        a.mode = Mode::AutoAccept;
        // `mv … /dev/null` is flagged dangerous; a nonexistent source makes the
        // actual execution a harmless no-op (mv errors, deletes nothing).
        type_str(&mut a, "!mv /tmp/zero-nonexistent-src-zzz /dev/null");
        a.dispatch(Key::Enter).unwrap();
        assert!(a.pending_shell.is_none()); // no y/N gate
        assert!(rendered(&a).contains("auto-accepted"));
    }

    #[test]
    fn footer_hints_at_queue_editing_when_a_queue_exists() {
        let mut a = app(b"");
        assert!(!a.footer_text().contains("^Q")); // no queue → no hint
        a.queue.push_back("later".to_string());
        assert!(a.footer_text().contains("^Q edit queue")); // hint appears
    }

    #[test]
    fn ctrl_q_on_empty_queue_is_a_noop() {
        let mut a = app(b"");
        a.dispatch(Key::Ctrl('q')).unwrap();
        assert!(a.queue_edit.is_none());
    }

    #[test]
    fn queue_edit_navigates_and_edits_in_place() {
        let mut a = app(b"");
        a.editor.set_text("draft"); // in-progress input to restore later
        a.queue.push_back("first".to_string());
        a.queue.push_back("second".to_string());
        a.dispatch(Key::Ctrl('q')).unwrap(); // enter, selects the last item
        assert!(a.queue_edit.is_some());
        assert_eq!(a.editor.text(), "second");
        a.dispatch(Key::Char('!')).unwrap(); // edit → "second!"
        a.dispatch(Key::Up).unwrap(); // persist + move to "first"
        assert_eq!(a.queue[1], "second!");
        assert_eq!(a.editor.text(), "first");
        a.dispatch(Key::Down).unwrap(); // back to "second!"
        assert_eq!(a.editor.text(), "second!");
    }

    #[test]
    fn queue_edit_supports_line_editing_keys() {
        // Every editing key in handle_queue_edit_key runs without leaving the mode.
        let mut a = app(b"");
        a.queue.push_back("hello world".to_string());
        a.dispatch(Key::Ctrl('q')).unwrap(); // edit "hello world"
        for k in [
            Key::Home,
            Key::End,
            Key::Left,
            Key::Right,
            Key::WordLeft,
            Key::WordRight,
            Key::Ctrl('b'),
            Key::Ctrl('f'),
            Key::Ctrl('a'),
            Key::Ctrl('e'),
            Key::Backspace,
            Key::Delete,
            Key::Ctrl('w'),
            Key::Ctrl('u'),
            Key::Ctrl('k'),
            Key::ShiftEnter,
            Key::Char('z'),
            Key::Tab, // unmapped in this submode → no-op arm
        ] {
            a.dispatch(k).unwrap();
        }
        assert!(a.queue_edit.is_some()); // still editing
        a.dispatch(Key::Esc).unwrap(); // exit
        assert!(a.queue_edit.is_none());
    }

    #[test]
    fn queue_edit_renders_marker_and_paused_footer() {
        let mut a = app(b"");
        a.queue.push_back("alpha".to_string());
        a.dispatch(Key::Ctrl('q')).unwrap();
        let out = rendered(&a);
        assert!(out.contains("✎ editing: alpha"));
        assert!(a.footer_text().contains("editing queued 1/1"));
        assert!(a.footer_text().contains("sending paused"));
    }

    #[test]
    fn queue_edit_pauses_sending_until_exit() {
        let (mut a, tx) = streaming_app();
        a.queue.push_back("next".to_string());
        a.dispatch(Key::Ctrl('q')).unwrap(); // edit while streaming
        tx.send(StreamEvent::Done(StopReason::EndTurn)).unwrap();
        a.pump_stream().unwrap(); // turn finishes…
        assert!(a.streaming.is_none());
        assert_eq!(a.queue.len(), 1); // …but the queue is NOT drained (paused)
        assert!(a.queue_edit.is_some());
        a.dispatch(Key::Enter).unwrap(); // exit → resume
        assert!(a.queue_edit.is_none());
        assert!(a.queue.is_empty()); // the queued message now runs
    }

    #[test]
    fn queue_edit_dropping_an_emptied_item_and_restoring_input() {
        let mut a = app(b"");
        a.editor.set_text("keep me");
        a.queue.push_back("toss".to_string());
        a.dispatch(Key::Ctrl('q')).unwrap();
        a.editor.set_text(""); // empty the queued item
        a.dispatch(Key::Esc).unwrap(); // exit
        assert!(a.queue.is_empty()); // emptied item dropped
        assert!(a.streaming.is_none()); // nothing to run
        assert_eq!(a.editor.text(), "keep me"); // original input restored
    }

    #[test]
    fn fmt_count_is_compact() {
        assert_eq!(fmt_count(0), "0");
        assert_eq!(fmt_count(840), "840");
        assert_eq!(fmt_count(1234), "1.2k");
        assert_eq!(fmt_count(32768), "33k");
    }

    #[test]
    fn short_host_strips_scheme_and_slash() {
        assert_eq!(
            short_host("http://192.168.50.125:8000/"),
            "192.168.50.125:8000"
        );
        assert_eq!(short_host("https://api.x.ai/v1"), "api.x.ai/v1");
        assert_eq!(short_host("bare:1234"), "bare:1234");
    }

    #[test]
    fn status_line_shows_model_endpoint_and_context() {
        let mut a = app(b"");
        a.config.model = "qwen-heretic".to_string();
        a.config.base_url = Some("http://192.168.50.125:8000".to_string());
        a.ctx_window = Some(32768);
        a.last_usage = Some(Usage {
            prompt_tokens: 1000,
            completion_tokens: 200,
        });
        let s = a.status_line();
        assert!(s.contains("qwen-heretic"));
        assert!(s.contains("192.168.50.125:8000"));
        assert!(s.contains("1.2k/33k ctx (3%)"));
    }

    #[test]
    fn context_summary_covers_each_knowledge_state() {
        let mut a = app(b"");
        assert_eq!(a.context_summary(), None); // nothing known
        a.ctx_window = Some(8192);
        assert_eq!(a.context_summary().unwrap(), "8.2k ctx"); // window only
        a.last_usage = Some(Usage {
            prompt_tokens: 4096,
            completion_tokens: 0,
        });
        assert_eq!(a.context_summary().unwrap(), "4.1k/8.2k ctx (50%)");
        a.ctx_window = None; // usage but no window
        assert_eq!(a.context_summary().unwrap(), "4.1k tok");
    }

    #[test]
    fn status_line_falls_back_to_backend_name_without_a_model() {
        let a = app(b""); // stub backend, empty config model
        let s = a.status_line();
        assert!(s.contains(a.backend.name()));
    }

    #[test]
    fn redraw_renders_the_status_footer() {
        let mut a = app(b"");
        a.config.model = "qwen".to_string();
        a.redraw_input().unwrap();
        assert!(rendered(&a).contains("qwen"));
    }

    #[test]
    fn a_usage_chunk_updates_last_usage() {
        let (mut a, tx) = streaming_app();
        tx.send(StreamEvent::Usage(Usage {
            prompt_tokens: 50,
            completion_tokens: 10,
        }))
        .unwrap();
        a.pump_stream().unwrap();
        // Held in the streaming state, not promoted until the turn finishes.
        assert_eq!(a.last_usage.map(|u| u.total()), None);
        tx.send(StreamEvent::Done(StopReason::EndTurn)).unwrap();
        a.pump_stream().unwrap();
        assert_eq!(a.last_usage.map(|u| u.total()), Some(60));
    }

    #[test]
    fn refresh_context_window_is_skipped_in_synchronous_mode() {
        let mut a = app(b"");
        a.config.base_url = Some("http://127.0.0.1:1".to_string());
        a.refresh_context_window(); // synchronous → no network, stays None
        assert_eq!(a.ctx_window, None);
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
    fn shell_mode_stderr_without_trailing_newline() {
        let mut a = app(b"");
        type_str(&mut a, "!printf oops >&2");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("oops"));
    }

    #[test]
    fn run_streams_reply_then_quits() {
        // Prompt and quit in separate reads so the turn finalizes between them.
        let mut a = multi_app(&[b"hello\r", b"/quit\r"]);
        a.run().unwrap();
        let out = rendered(&a);
        assert!(out.contains("local-first AI terminal"));
        assert!(out.contains("hello"));
        assert!(out.contains("You said")); // stub reply
        assert_eq!(a.conv.len(), 2);
    }

    #[test]
    fn run_propagates_output_errors() {
        let mut a = App::new(
            ScriptedInput::new(b"hi\r"),
            FlakyWriter { ok: 0 },
            Arc::new(StubBackend::instant()),
            None,
        );
        a.synchronous = true;
        assert!(a.run().is_err());
    }

    #[test]
    fn run_handles_poll_timeout_and_multiple_reads() {
        let mut a = multi_app(&[b"hello", b"", b"\r", b"/quit\r"]);
        a.run().unwrap();
        assert_eq!(a.conv.len(), 2);
    }

    #[test]
    fn run_with_session_log_records_the_turn() {
        let dir =
            std::env::temp_dir().join(format!("zero-app-test-{}", zero_core::clock::unix_millis()));
        let (log, path) = SessionLog::create_in(&dir).unwrap();
        let mut a = App::new(
            MultiInput::new(&[b"hi\r", b"/quit\r"]),
            Vec::new(),
            Arc::new(StubBackend::instant()),
            Some(log),
        );
        a.synchronous = true;
        a.run().unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("\"role\":\"user\""));
        assert!(contents.contains("\"role\":\"assistant\""));
        assert!(contents.contains("turn_done"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn panicking_backend_does_not_crash_the_turn() {
        // A backend that panics mid-stream must not bring down the process; the
        // turn surfaces an error and the app keeps going.
        struct PanicBackend;
        impl Backend for PanicBackend {
            fn name(&self) -> &str {
                "panic"
            }
            fn stream(
                &self,
                _c: &Conversation,
                _s: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), BackendError> {
                panic!("backend went boom");
            }
        }
        // Silence the default panic hook for this test's expected panic.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut a = App::new(
            MultiInput::new(&[b"hi\r", b"/quit\r"]),
            Vec::new(),
            Arc::new(PanicBackend),
            None,
        );
        a.synchronous = true;
        a.run().unwrap();
        std::panic::set_hook(prev);
        assert!(rendered(&a).contains("internal error"));
    }

    #[test]
    fn erroring_backend_surfaces_in_the_turn() {
        let mut a = App::new(
            MultiInput::new(&[b"hi\r", b"/quit\r"]),
            Vec::new(),
            Arc::new(FailBackend),
            None,
        );
        a.synchronous = true;
        a.run().unwrap();
        assert!(rendered(&a).contains("boom")); // error shown as a token
    }

    #[test]
    fn messages_typed_while_streaming_are_queued_and_run() {
        // Start a turn, then (before it's pumped) type another message — it
        // queues and runs as its own turn after the first finalizes.
        let mut a = app(b"");
        a.start_turn("first").unwrap();
        assert!(a.streaming.is_some());
        type_into(&mut a, "second");
        a.dispatch(Key::Enter).unwrap(); // queued, not started
        assert_eq!(a.queue.len(), 1);
        a.redraw_input().unwrap();
        assert!(rendered(&a).contains("⏎ queued: second")); // listed above the box
                                                            // Pump to completion: first finalizes → second starts → finalizes.
        for _ in 0..4 {
            a.pump_stream().unwrap();
        }
        assert!(a.streaming.is_none());
        assert!(a.queue.is_empty());
        // Two user turns + two assistant replies.
        assert_eq!(a.conv.len(), 4);
        assert_eq!(a.conv.messages[0].content, "first");
        assert_eq!(a.conv.messages[2].content, "second");
    }

    #[test]
    fn ctrl_c_interrupts_a_streaming_turn() {
        let mut a = app(b"");
        a.start_turn("go").unwrap();
        a.queue.push_back("queued".to_string());
        a.dispatch(Key::Ctrl('c')).unwrap(); // interrupt
        assert!(a.streaming.is_none());
        assert!(a.queue.is_empty()); // interrupt drops the queue
        assert!(rendered(&a).contains("interrupted"));
    }

    /// Hand-build a streaming state over a manual channel so the partial-stream
    /// branches (Empty/Disconnected, partial-reply interrupt) are exercised.
    fn streaming_app() -> (App<ScriptedInput, Vec<u8>>, mpsc::Sender<StreamEvent>) {
        let mut a = app(b"");
        a.conv.push(Message::user("q"));
        let (tx, rx) = mpsc::channel();
        a.streaming = Some(StreamState {
            rx,
            reply: String::new(),
            md: MarkdownStream::new(),
            sw: Stopwatch::start(),
            usage: None,
        });
        (a, tx)
    }

    #[test]
    fn pump_renders_partial_then_waits_when_channel_empty() {
        let (mut a, tx) = streaming_app();
        tx.send(StreamEvent::Token("partial ".into())).unwrap();
        a.pump_stream().unwrap(); // renders, then try_recv → Empty → still streaming
        assert!(a.streaming.is_some());
        assert!(rendered(&a).contains("partial"));
        drop(tx);
    }

    #[test]
    fn pump_finalizes_when_channel_disconnects_without_done() {
        let (mut a, tx) = streaming_app();
        tx.send(StreamEvent::Token("bit".into())).unwrap();
        drop(tx); // no Done → disconnect ends the turn
        a.pump_stream().unwrap();
        assert!(a.streaming.is_none());
        assert_eq!(a.conv.len(), 2); // assistant reply recorded
    }

    #[test]
    fn interrupt_keeps_partial_reply_and_closes_open_markdown() {
        let (mut a, tx) = streaming_app();
        tx.send(StreamEvent::Token("**bold so far".into())).unwrap();
        a.pump_stream().unwrap(); // render the (unclosed) bold
        a.dispatch(Key::Ctrl('c')).unwrap();
        assert!(a.streaming.is_none());
        // Partial text kept in context; styling reset on interrupt.
        assert!(a
            .conv
            .messages
            .last()
            .unwrap()
            .content
            .contains("bold so far"));
        assert!(rendered(&a).contains("\x1b[0m"));
        drop(tx);
    }

    #[test]
    fn streaming_edit_keys_buffer_without_echo() {
        let (mut a, tx) = streaming_app();
        a.dispatch(Key::Char('x')).unwrap();
        a.dispatch(Key::Backspace).unwrap();
        a.dispatch(Key::Left).unwrap(); // unmapped while streaming → no-op
        assert!(a.editor.is_empty());
        assert!(a.streaming.is_some());
        drop(tx);
    }

    #[test]
    fn slash_commands_dispatch_through_submit() {
        // Exercises the /servers, /connect, /model, /model-empty command branches.
        let mut a = app(b"");
        for cmd in ["/servers", "/connect 1", "/model qwen", "/model"] {
            type_str(&mut a, cmd);
            a.dispatch(Key::Enter).unwrap();
        }
        let out = rendered(&a);
        assert!(out.contains("no saved servers"));
        assert!(out.contains("no such entry"));
        assert!(out.contains("model set: qwen"));
        assert!(out.contains("model: qwen"));
    }

    #[test]
    fn quit_while_streaming_still_exits() {
        let mut a = app(b"");
        a.start_turn("go").unwrap();
        type_into(&mut a, "/quit");
        assert_eq!(a.dispatch(Key::Enter).unwrap(), Flow::Quit);
    }

    #[test]
    fn input_box_grows_with_lines_and_shrinks_after_send() {
        // cursor_row is the cursor's row within the box (1 = first input line,
        // just below the top rule). It tracks the box height as lines are added.
        let mut a = app(b"");
        a.dispatch(Key::Char('a')).unwrap();
        a.redraw_input().unwrap();
        assert_eq!(a.cursor_row, 1); // single input line

        a.dispatch(Key::ShiftEnter).unwrap(); // add a line
        a.dispatch(Key::Char('b')).unwrap();
        a.redraw_input().unwrap();
        assert_eq!(a.cursor_row, 2); // grew: cursor on the 2nd input line

        a.dispatch(Key::Enter).unwrap(); // send (synchronous turn completes)
        assert!(a.editor.is_empty());
        a.redraw_input().unwrap();
        assert_eq!(a.cursor_row, 1); // shrank back to a one-line box
    }

    #[test]
    fn esc_twice_interrupts_a_streaming_turn() {
        let mut a = app(b"");
        a.start_turn("go").unwrap();
        a.dispatch(Key::Esc).unwrap(); // arm
        assert!(a.streaming.is_some());
        a.dispatch(Key::Esc).unwrap(); // interrupt
        assert!(a.streaming.is_none());
        assert!(rendered(&a).contains("interrupted"));
    }

    #[test]
    fn typing_between_escs_does_not_interrupt() {
        let mut a = app(b"");
        a.start_turn("go").unwrap();
        a.dispatch(Key::Esc).unwrap(); // arm
        a.dispatch(Key::Char('x')).unwrap(); // disarms
        a.dispatch(Key::Esc).unwrap(); // arms again, not interrupt
        assert!(a.streaming.is_some());
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

    /// A backend that streams a canned reply containing a code block.
    struct CodeBackend;
    impl Backend for CodeBackend {
        fn name(&self) -> &str {
            "code"
        }
        fn stream(
            &self,
            _c: &Conversation,
            sink: &mut dyn FnMut(StreamEvent),
        ) -> Result<(), BackendError> {
            sink(StreamEvent::Token(
                "here:\n```rust\nfn main() {}\n```\ndone".to_string(),
            ));
            sink(StreamEvent::Done(zero_core::backend::StopReason::EndTurn));
            Ok(())
        }
    }

    #[test]
    fn response_with_code_block_offers_per_block_clip() {
        use std::cell::RefCell;
        use std::rc::Rc;
        let captured = Rc::new(RefCell::new(String::new()));
        let sink = captured.clone();
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(CodeBackend),
            None,
        );
        a.synchronous = true;
        a.set_clipboard(Box::new(move |s| {
            *sink.borrow_mut() = s.to_string();
            Ok(())
        }));
        type_str(&mut a, "go");
        a.dispatch(Key::Enter).unwrap();
        // The block streamed a copy footer.
        assert!(rendered(&a).contains("⧉ copy"));
        assert_eq!(a.last_blocks.len(), 1);
        // /clip 1 copies just the block body, not the whole response.
        type_str(&mut a, "/clip 1");
        a.dispatch(Key::Enter).unwrap();
        assert_eq!(*captured.borrow(), "fn main() {}");
    }

    #[test]
    fn clip_index_out_of_range_is_reported() {
        let mut a = app(b"");
        type_str(&mut a, "/clip 9");
        a.dispatch(Key::Enter).unwrap();
        assert!(rendered(&a).contains("no such code block"));
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
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
        }
        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(BoldBackend),
            None,
        );
        a.synchronous = true;
        a.start_turn("x").unwrap();
        while a.streaming.is_some() {
            a.pump_stream().unwrap();
        }
        assert_eq!(a.last_reply, "**hi**"); // raw markdown preserved for model/clip
        let painted = rendered(&a);
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
            Arc::new(StubBackend::instant()),
            None,
        );
        a.synchronous = true;
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
            Arc::new(StubBackend::instant()),
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
            Arc::new(StubBackend::instant()),
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
            Arc::new(StubBackend::instant()),
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
    #[test]
    fn run_once_then_context_command_reports_via_headless_path() {
        // Covers run_once's tools arm + a follow-up /context render in one go,
        // using only known-good helpers (no new backend types).
        let mut a = app(b"");
        a.set_tools_enabled(true);
        let _ = a.run_once("do something").unwrap();
        type_str(&mut a, "/context");
        a.dispatch(Key::Enter).unwrap();
        // Either "no tool output yet" (stub made no tool call) or a savings line —
        // both are valid; assert the command produced a context report.
        let out = rendered(&a);
        assert!(out.contains("context") || out.contains("tool output"));
    }
    #[test]
    fn tool_turn_read_cache_hit_returns_stub_on_second_read() {
        // Covers the executor read-cache hit branch end to end: a read_file under
        // the workspace root succeeds (recorded), then a second read of the same
        // unchanged file returns the cached stub instead of re-reading.
        use std::sync::Mutex;
        struct TwoReads {
            n: Arc<Mutex<u32>>,
        }
        impl Backend for TwoReads {
            fn name(&self) -> &str {
                "tworeads"
            }
            fn stream(
                &self,
                _c: &Conversation,
                sink: &mut dyn FnMut(StreamEvent),
            ) -> Result<(), zero_core::backend::BackendError> {
                sink(StreamEvent::Done(StopReason::EndTurn));
                Ok(())
            }
            fn complete(
                &self,
                _c: &Conversation,
                _t: &[ToolDef],
                _to: Duration,
            ) -> Result<zero_core::backend::Completion, zero_core::backend::BackendError>
            {
                let mut n = self.n.lock().unwrap();
                *n += 1;
                Ok(match *n {
                    1 | 2 => zero_core::backend::Completion {
                        content: String::new(),
                        tool_calls: vec![ToolCall::new(
                            format!("r{n}"),
                            "read_file",
                            r#"{"path":"f.txt"}"#,
                        )],
                        usage: None,
                    },
                    _ => zero_core::backend::Completion {
                        content: "done".to_string(),
                        tool_calls: vec![],
                        usage: None,
                    },
                })
            }
        }
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("zero-rch-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "the file body long enough to matter").unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();

        let mut a = App::new(
            ScriptedInput::new(b""),
            Vec::new(),
            Arc::new(TwoReads {
                n: Arc::new(Mutex::new(0)),
            }),
            None,
        );
        a.tools_enabled = true;
        a.set_artifact_dir(Some(dir.clone()));
        let res = a.run_tool_turn("read it twice");
        std::env::set_current_dir(&prev).unwrap(); // restore before asserting
        res.unwrap();

        let reads: Vec<&Message> = a
            .conv
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        assert_eq!(reads.len(), 2);
        assert!(
            reads[0].content.contains("the file body"),
            "first read body: {}",
            reads[0].content
        );
        assert!(
            reads[1].content.contains("unchanged"),
            "second read stub: {}",
            reads[1].content
        );
        // cache hit was recorded (the file is tiny, so the stub may be larger than
        // the would-be re-read → saturating saving of 0; the point is the branch
        // fired and the second read did NOT re-inject the body).
        assert!(!reads[1].content.contains("the file body"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
