//! The interactive REPL: wires an input source, key decoder, line editor, and a
//! [`Backend`] into Claude-Code-style inline rendering.
//!
//! "Inline" means output is printed in normal flow, so the terminal emulator's
//! own scrollback works exactly as users expect — we only take over the *current
//! input line*, redrawing it in place as it is edited.
//!
//! [`App`] is generic over its input ([`Input`]) and output ([`Write`]) so the
//! entire loop is testable with scripted bytes and a captured buffer; the binary
//! instantiates it with a real terminal and stdout.

use crate::editor::LineEditor;
use crate::key::{decode_keys, Key};
use std::io::{self, Write};
use std::time::Duration;
use zero_core::backend::{Backend, StreamEvent};
use zero_core::clock::{format_duration, Stopwatch};
use zero_core::message::{Conversation, Message, Role};
use zero_core::session::SessionLog;

/// A source of input bytes. `read` returns 0 on a poll timeout (not EOF).
/// `RawTerminal` implements this (see `term.rs`); tests use a scripted source.
pub trait Input {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// Result of handling one key: keep looping or tear down.
#[derive(Debug, PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
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
        }
    }

    /// Run the event loop until the user quits.
    pub fn run(&mut self) -> io::Result<()> {
        self.print_banner()?;
        self.redraw_input()?;

        let mut pending: Vec<u8> = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = self.input.read(&mut buf)?;
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
            Key::Ctrl('l') => self.write_text("\x1b[2J\x1b[H")?,
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

        // Honest, measured elapsed — dimmed, never an estimate.
        self.write_text(&format!("\n\x1b[2m  {}\x1b[0m\n", format_duration(elapsed)))?;
        Ok(Flow::Continue)
    }

    /// Print the committed input line as static scrollback.
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
  /quit /exit  leave\n\
\x1b[1mediting\x1b[0m\n\
  ^A/^E home/end   ^U/^K kill to start/end   ^W kill word\n\
  ↑/↓ history      ^L clear screen           ^C quit\n\n";
        self.write_text(help)
    }

    /// Write text, translating `\n` to `\r\n` for raw mode.
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

    /// Input that yields a sequence of chunks (one per read), then 0 forever.
    /// An empty chunk simulates a poll timeout.
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
    fn run_handles_poll_timeout_and_multiple_reads() {
        // Chunk 1 types a line, chunk 2 is an empty poll timeout (n==0), chunk 3
        // quits. Exercises the post-batch redraw and the timeout-continue path.
        let mut a = App::new(
            MultiInput::new(&[b"hello", b"", b"\r/quit\r"]),
            Vec::new(),
            Box::new(StubBackend::instant()),
            None,
        );
        a.run().unwrap();
        assert_eq!(a.conv.len(), 2); // the "hello" turn streamed
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
    fn run_streams_reply_then_quits() {
        let mut a = app(b"hello\r/quit\r");
        a.run().unwrap();
        let out = rendered(&a);
        assert!(out.contains("local-first AI terminal")); // banner
        assert!(out.contains("hello")); // echoed input
        assert!(out.contains("You said")); // stub reply
                                           // conversation captured both turns
        assert_eq!(a.conv.len(), 2);
    }

    #[test]
    fn run_quits_on_ctrl_c() {
        let mut a = app(b"\x03");
        a.run().unwrap();
        assert!(rendered(&a).contains("^C"));
    }

    #[test]
    fn run_quits_on_ctrl_d_when_empty() {
        let mut a = app(&[0x04]);
        a.run().unwrap();
        // No reply streamed; just banner + input line.
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn help_command_prints_help_without_streaming() {
        let mut a = app(b"/help\r/quit\r");
        a.run().unwrap();
        let out = rendered(&a);
        assert!(out.contains("commands"));
        assert!(out.contains("kill word"));
        assert_eq!(a.conv.len(), 0); // /help is not a model turn
    }

    #[test]
    fn blank_submit_is_ignored() {
        let mut a = app(b"   \r/quit\r");
        a.run().unwrap();
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn exit_alias_quits() {
        let mut a = app(b"/exit\r");
        a.run().unwrap();
        assert_eq!(a.conv.len(), 0);
    }

    #[test]
    fn editing_keys_mutate_the_line() {
        let mut a = app(b"");
        for c in "hello".chars() {
            a.handle_key(Key::Char(c)).unwrap();
        }
        a.handle_key(Key::Home).unwrap();
        a.handle_key(Key::Right).unwrap();
        a.handle_key(Key::Delete).unwrap();
        assert_eq!(a.editor.text(), "hllo");
        a.handle_key(Key::End).unwrap();
        a.handle_key(Key::Backspace).unwrap();
        assert_eq!(a.editor.text(), "hll");
        a.handle_key(Key::Ctrl('u')).unwrap();
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn ctrl_chords_map_to_editor_ops() {
        let mut a = app(b"");
        for c in "foo bar".chars() {
            a.handle_key(Key::Char(c)).unwrap();
        }
        a.handle_key(Key::Ctrl('w')).unwrap(); // kill word
        assert_eq!(a.editor.text(), "foo ");
        a.handle_key(Key::Ctrl('e')).unwrap(); // end
        assert_eq!(a.editor.cursor(), 4);
        a.handle_key(Key::Ctrl('a')).unwrap(); // home
        assert_eq!(a.editor.cursor(), 0);
        a.handle_key(Key::Ctrl('k')).unwrap(); // kill from start to end
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn ctrl_d_midline_deletes_char() {
        let mut a = app(b"");
        a.handle_key(Key::Char('x')).unwrap();
        a.handle_key(Key::Char('y')).unwrap();
        a.handle_key(Key::Home).unwrap();
        a.handle_key(Key::Ctrl('d')).unwrap(); // delete at cursor, not quit
        assert_eq!(a.editor.text(), "y");
    }

    #[test]
    fn ctrl_l_clears_screen() {
        let mut a = app(b"");
        a.handle_key(Key::Ctrl('l')).unwrap();
        assert!(rendered(&a).contains("\x1b[2J"));
    }

    #[test]
    fn unmapped_keys_are_noops() {
        let mut a = app(b"");
        assert_eq!(a.handle_key(Key::Tab).unwrap(), Flow::Continue);
        assert_eq!(a.handle_key(Key::Esc).unwrap(), Flow::Continue);
        assert_eq!(a.handle_key(Key::Ctrl('z')).unwrap(), Flow::Continue);
    }

    #[test]
    fn history_keys_recall_previous_line() {
        let mut a = app(b"");
        for c in "first".chars() {
            a.handle_key(Key::Char(c)).unwrap();
        }
        a.handle_key(Key::Enter).unwrap(); // submits + streams
        a.handle_key(Key::Up).unwrap(); // recall "first"
        assert_eq!(a.editor.text(), "first");
        a.handle_key(Key::Down).unwrap(); // back to empty draft
        assert_eq!(a.editor.text(), "");
    }

    #[test]
    fn stream_reply_surfaces_writer_error_mid_stream() {
        let mut conv = Conversation::new();
        conv.push(Message::user("ping"));
        // Allow the label write, fail on the first token.
        let mut out = FlakyWriter { ok: 1 };
        let err = stream_reply(&StubBackend::instant(), &mut out, &conv).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn run_propagates_output_errors() {
        // ok: 0 → the very first banner write fails, run() returns Err.
        let mut a = App::new(
            ScriptedInput::new(b"hi\r"),
            FlakyWriter { ok: 0 },
            Box::new(StubBackend::instant()),
            None,
        );
        assert!(a.run().is_err());
    }

    #[test]
    fn redraw_moves_cursor_when_not_at_end() {
        let mut a = app(b"");
        a.handle_key(Key::Char('a')).unwrap();
        a.handle_key(Key::Char('b')).unwrap();
        a.handle_key(Key::Left).unwrap();
        a.out.clear();
        a.redraw_input().unwrap();
        // Cursor moved back one column.
        assert!(rendered(&a).contains("\x1b[1D"));
    }
}
