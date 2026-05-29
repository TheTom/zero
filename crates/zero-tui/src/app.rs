//! The interactive REPL: wires the terminal, key decoder, line editor, and a
//! [`Backend`] into Claude-Code-style inline rendering.
//!
//! "Inline" means output is printed to stdout in normal flow, so the terminal
//! emulator's own scrollback works exactly as users expect — we only take over
//! the *current input line*, redrawing it in place as it is edited. This is the
//! feel Zero is copying; full-screen mode (using [`crate::viewport`]) comes
//! later for the richer app surface.

use crate::editor::LineEditor;
use crate::key::{decode_keys, Key};
use crate::term::RawTerminal;
use std::io::{self, Write};
use std::time::Duration;
use zero_core::backend::{Backend, StreamEvent};
use zero_core::clock::{format_duration, Stopwatch};
use zero_core::message::{Conversation, Message, Role};
use zero_core::session::SessionLog;

/// Result of handling one key: keep looping or tear down.
#[derive(Debug, PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

/// The running terminal application.
pub struct App {
    term: RawTerminal,
    out: io::Stdout,
    editor: LineEditor,
    conv: Conversation,
    backend: Box<dyn Backend>,
    log: Option<SessionLog<std::fs::File>>,
    prompt: String,
}

impl App {
    /// Build an app over an already-raw terminal and a chosen backend.
    pub fn new(
        term: RawTerminal,
        backend: Box<dyn Backend>,
        log: Option<SessionLog<std::fs::File>>,
    ) -> Self {
        App {
            term,
            out: io::stdout(),
            editor: LineEditor::new(),
            conv: Conversation::new(),
            backend,
            log,
            prompt: "› ".to_string(),
        }
    }

    /// Run the event loop until the user quits. Restores the terminal on return
    /// (the `RawTerminal` drop guard handles raw-mode teardown).
    pub fn run(&mut self) -> io::Result<()> {
        self.print_banner()?;
        self.redraw_input()?;

        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = self.term.read(&mut buf)?;
            if n == 0 {
                continue; // poll timeout; nothing to do
            }
            pending.extend_from_slice(&buf[..n]);
            let (keys, consumed) = decode_keys(&pending);
            pending.drain(0..consumed);
            for key in keys {
                if self.handle_key(key)? == Flow::Quit {
                    self.write_text("\n")?;
                    return Ok(());
                }
            }
            self.redraw_input()?;
        }
    }

    fn handle_key(&mut self, key: Key) -> io::Result<Flow> {
        match key {
            Key::Ctrl('c') => {
                self.write_text("\n^C\n")?;
                return Ok(Flow::Quit);
            }
            Key::Ctrl('d') if self.editor.is_empty() => return Ok(Flow::Quit),
            Key::Ctrl('d') => self.editor.delete(),
            Key::Ctrl('l') => {
                // Clear screen + home, then redraw the input line.
                self.write_text("\x1b[2J\x1b[H")?;
            }
            Key::Ctrl('a') | Key::Home => self.editor.home(),
            Key::Ctrl('e') | Key::End => self.editor.end(),
            Key::Ctrl('u') => self.editor.kill_to_start(),
            Key::Ctrl('k') => self.editor.kill_to_end(),
            Key::Ctrl('w') => self.editor.kill_word(),
            Key::Ctrl(_) | Key::Esc | Key::Tab => {} // unmapped; ignore
            Key::Enter => return self.on_submit(),
            Key::Backspace => self.editor.backspace(),
            Key::Delete => self.editor.delete(),
            Key::Left => self.editor.left(),
            Key::Right => self.editor.right(),
            Key::Up => self.editor.history_prev(),
            Key::Down => self.editor.history_next(),
            Key::Char(c) => self.editor.insert(c),
        }
        Ok(Flow::Continue)
    }

    fn on_submit(&mut self) -> io::Result<Flow> {
        let text = self.editor.submit();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            // Move to a fresh input line without echoing anything.
            self.write_text("\r\x1b[K")?;
            return Ok(Flow::Continue);
        }
        if matches!(trimmed, "/quit" | "/exit") {
            return Ok(Flow::Quit);
        }
        if trimmed == "/help" {
            self.echo_committed(&text)?;
            self.print_help()?;
            return Ok(Flow::Continue);
        }

        // Echo the committed user line into normal scrollback flow.
        self.echo_committed(&text)?;
        self.conv.push(Message::user(&text));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::User, &text);
        }

        // Stream the reply. Disjoint field borrows let the sink write to stdout
        // while the backend is borrowed immutably.
        let (reply, elapsed) = stream_reply(
            self.backend.as_ref(),
            &mut self.out,
            &self.prompt,
            &self.conv,
        )?;

        self.conv.push(Message::assistant(&reply));
        if let Some(log) = self.log.as_mut() {
            let _ = log.record_message(Role::Assistant, &reply);
            let _ = log.record_turn_done(elapsed.as_millis());
        }

        // Honest, measured elapsed — dimmed, never an estimate.
        self.write_text(&format!("\n\x1b[2m  {}\x1b[0m\n", format_duration(elapsed)))?;
        Ok(Flow::Continue)
    }

    /// Print the committed input line as static scrollback (clears the live
    /// input line first, then re-emits it with the prompt).
    fn echo_committed(&mut self, text: &str) -> io::Result<()> {
        self.write_text("\r\x1b[K")?;
        let line = format!("{}{}\n", self.prompt, text);
        self.write_text(&line)
    }

    /// Redraw the in-place input line and position the cursor.
    fn redraw_input(&mut self) -> io::Result<()> {
        let text = self.editor.text();
        let line = format!("\r\x1b[K{}{}", self.prompt, text);
        self.out.write_all(line.as_bytes())?;
        // Move the cursor back to its logical position within the buffer.
        let total = text.chars().count();
        let tail = total - self.editor.cursor();
        if tail > 0 {
            write!(self.out, "\x1b[{tail}D")?;
        }
        self.out.flush()
    }

    fn print_banner(&mut self) -> io::Result<()> {
        let banner = format!(
            "\x1b[1m{}\x1b[0m — local-first AI terminal  \x1b[2m({})\x1b[0m\n\
             \x1b[2m/help for commands · ^C to quit\x1b[0m\n\n",
            zero_core::brand::name(),
            self.backend.name()
        );
        self.write_text(&banner)
    }

    fn print_help(&mut self) -> io::Result<()> {
        let help = "\
\x1b[1mcommands\x1b[0m\n\
  /help        show this\n\
  /quit /exit  leave zero\n\
\x1b[1mediting\x1b[0m\n\
  ^A/^E home/end   ^U/^K kill to start/end   ^W kill word\n\
  ↑/↓ history      ^L clear screen           ^C quit\n\n";
        self.write_text(help)
    }

    /// Write text to stdout, translating `\n` to `\r\n` for raw mode.
    fn write_text(&mut self, s: &str) -> io::Result<()> {
        write_raw(&mut self.out, s)
    }
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

/// Stream a reply from `backend`, echoing tokens live, and return the full text
/// plus the measured elapsed time. Free function so the caller can pass disjoint
/// `&mut out` and `&backend` borrows.
fn stream_reply<W: Write>(
    backend: &dyn Backend,
    out: &mut W,
    prompt: &str,
    conv: &Conversation,
) -> io::Result<(String, Duration)> {
    let _ = prompt;
    let mut reply = String::new();
    let mut io_err: Option<io::Error> = None;

    // Assistant label, dimmed to distinguish from user input.
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
            if let Err(e) = write_raw(out, &t).and_then(|()| out.flush()) {
                io_err = Some(e);
            }
        }
    });
    let elapsed = sw.elapsed();

    if let Some(e) = io_err {
        return Err(e);
    }
    if let Err(e) = stream_res {
        write_raw(out, &format!("\n\x1b[31m[{e}]\x1b[0m"))?;
    }
    Ok((reply, elapsed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zero_core::backend::StubBackend;

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
        let backend = StubBackend::instant();
        let mut out = Vec::new();
        let (reply, _elapsed) = stream_reply(&backend, &mut out, "› ", &conv).unwrap();
        assert!(reply.contains("ping"));
        let painted = String::from_utf8(out).unwrap();
        assert!(painted.contains("zero›"));
        assert!(painted.contains("ping"));
    }
}
