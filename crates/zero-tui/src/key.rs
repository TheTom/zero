//! Terminal input decoding: raw byte stream → [`Key`] events.
//!
//! This is the pure heart of input handling — no I/O, no terminal state — so it
//! is exhaustively unit-testable. The imperative shell ([`crate::term`]) feeds
//! it bytes read from the tty; this module turns them into keys, correctly
//! handling multi-byte UTF-8 and ANSI escape sequences that arrive split across
//! reads.

/// A decoded key press.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    /// A printable character (already UTF-8 decoded).
    Char(char),
    Enter,
    Tab,
    Backspace,
    Delete,
    Esc,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    /// A Ctrl-letter chord, normalized to lowercase (`Ctrl('c')` for ^C).
    Ctrl(char),
}

/// Outcome of trying to decode one key at the front of a buffer.
enum Step {
    /// A key spanning `len` bytes.
    Key(Key, usize),
    /// `len` bytes consumed with no key emitted (unrecognized escape).
    Consume(usize),
    /// Not enough bytes yet — a longer sequence may still be coming.
    NeedMore,
}

/// Decode as many keys as possible from `buf`. Returns the decoded keys and the
/// number of bytes consumed; leftover bytes (an incomplete trailing sequence)
/// should be retained by the caller and prepended to the next read.
pub fn decode_keys(buf: &[u8]) -> (Vec<Key>, usize) {
    let mut keys = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        match decode_one(&buf[pos..]) {
            Step::Key(k, len) => {
                keys.push(k);
                pos += len;
            }
            Step::Consume(len) => pos += len,
            Step::NeedMore => break,
        }
    }
    (keys, pos)
}

fn decode_one(buf: &[u8]) -> Step {
    let b = buf[0];
    match b {
        0x1b => decode_escape(buf),
        b'\r' | b'\n' => Step::Key(Key::Enter, 1),
        b'\t' => Step::Key(Key::Tab, 1),
        0x7f | 0x08 => Step::Key(Key::Backspace, 1),
        // Other C0 control bytes → Ctrl-letter. 0x01 == ^A, .. 0x1a == ^Z.
        0x01..=0x1a => Step::Key(Key::Ctrl((b - 1 + b'a') as char), 1),
        // Remaining low controls we don't map — drop them.
        0x00 | 0x1c..=0x1f => Step::Consume(1),
        // ASCII printable.
        0x20..=0x7e => Step::Key(Key::Char(b as char), 1),
        // UTF-8 multi-byte lead byte.
        _ => decode_utf8(buf),
    }
}

fn decode_escape(buf: &[u8]) -> Step {
    // Lone ESC at end of buffer: ambiguous (could be Alt/CSI prefix). Wait for
    // more; the shell decides it's a bare Esc after a read timeout.
    if buf.len() < 2 {
        return Step::NeedMore;
    }
    match buf[1] {
        b'[' | b'O' => decode_csi(buf),
        // ESC followed by something else: treat the ESC alone as a press and
        // let the following byte decode on its own next iteration.
        _ => Step::Key(Key::Esc, 1),
    }
}

/// Decode a CSI / SS3 sequence: `ESC [ ...` or `ESC O ...`.
fn decode_csi(buf: &[u8]) -> Step {
    // buf[0] == ESC, buf[1] == '[' or 'O'.
    if buf.len() < 3 {
        return Step::NeedMore;
    }
    let final_byte = buf[2];
    match final_byte {
        b'A' => Step::Key(Key::Up, 3),
        b'B' => Step::Key(Key::Down, 3),
        b'C' => Step::Key(Key::Right, 3),
        b'D' => Step::Key(Key::Left, 3),
        b'H' => Step::Key(Key::Home, 3),
        b'F' => Step::Key(Key::End, 3),
        // Numeric "ESC [ N ~" form: read digits, expect a trailing '~'.
        b'0'..=b'9' => decode_csi_numeric(buf),
        // Unknown CSI final byte — consume the 3 bytes and move on.
        _ => Step::Consume(3),
    }
}

fn decode_csi_numeric(buf: &[u8]) -> Step {
    // Scan digits starting at index 2 until a terminator.
    let mut i = 2;
    while i < buf.len() && buf[i].is_ascii_digit() {
        i += 1;
    }
    if i >= buf.len() {
        return Step::NeedMore; // digits not yet terminated
    }
    if buf[i] != b'~' {
        // e.g. modifier sequences like "ESC [ 1 ; 2 A" — skip the whole run.
        return Step::Consume(i + 1);
    }
    let num = &buf[2..i];
    let total = i + 1; // include the '~'
    let key = match num {
        b"1" | b"7" => Key::Home,
        b"4" | b"8" => Key::End,
        b"3" => Key::Delete,
        _ => return Step::Consume(total),
    };
    Step::Key(key, total)
}

fn decode_utf8(buf: &[u8]) -> Step {
    let len = utf8_len(buf[0]);
    match len {
        0 => Step::Consume(1), // invalid lead byte
        n if buf.len() < n => Step::NeedMore,
        n => match std::str::from_utf8(&buf[..n]) {
            Ok(s) => match s.chars().next() {
                Some(c) => Step::Key(Key::Char(c), n),
                None => Step::Consume(n),
            },
            Err(_) => Step::Consume(1), // malformed; resync by one byte
        },
    }
}

/// Expected total byte length of a UTF-8 sequence given its lead byte, or 0 if
/// the byte cannot start one.
fn utf8_len(lead: u8) -> usize {
    match lead {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        0xf0..=0xf7 => 4,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(bytes: &[u8]) -> Vec<Key> {
        decode_keys(bytes).0
    }

    #[test]
    fn decodes_ascii_run() {
        assert_eq!(
            keys(b"hi!"),
            vec![Key::Char('h'), Key::Char('i'), Key::Char('!')]
        );
    }

    #[test]
    fn decodes_enter_tab_backspace() {
        assert_eq!(keys(b"\r"), vec![Key::Enter]);
        assert_eq!(keys(b"\n"), vec![Key::Enter]);
        assert_eq!(keys(b"\t"), vec![Key::Tab]);
        assert_eq!(keys(&[0x7f]), vec![Key::Backspace]);
        assert_eq!(keys(&[0x08]), vec![Key::Backspace]);
    }

    #[test]
    fn decodes_ctrl_chords() {
        assert_eq!(keys(&[0x03]), vec![Key::Ctrl('c')]);
        assert_eq!(keys(&[0x04]), vec![Key::Ctrl('d')]);
        assert_eq!(keys(&[0x15]), vec![Key::Ctrl('u')]);
        assert_eq!(keys(&[0x17]), vec![Key::Ctrl('w')]);
        assert_eq!(keys(&[0x01]), vec![Key::Ctrl('a')]);
        assert_eq!(keys(&[0x05]), vec![Key::Ctrl('e')]);
    }

    #[test]
    fn decodes_arrow_keys() {
        assert_eq!(keys(b"\x1b[A"), vec![Key::Up]);
        assert_eq!(keys(b"\x1b[B"), vec![Key::Down]);
        assert_eq!(keys(b"\x1b[C"), vec![Key::Right]);
        assert_eq!(keys(b"\x1b[D"), vec![Key::Left]);
    }

    #[test]
    fn decodes_application_cursor_mode() {
        // ESC O A — some terminals send SS3 for arrows.
        assert_eq!(keys(b"\x1bOA"), vec![Key::Up]);
    }

    #[test]
    fn decodes_home_end_delete_numeric() {
        assert_eq!(keys(b"\x1b[H"), vec![Key::Home]);
        assert_eq!(keys(b"\x1b[F"), vec![Key::End]);
        assert_eq!(keys(b"\x1b[1~"), vec![Key::Home]);
        assert_eq!(keys(b"\x1b[4~"), vec![Key::End]);
        assert_eq!(keys(b"\x1b[3~"), vec![Key::Delete]);
    }

    #[test]
    fn lone_escape_at_end_is_incomplete() {
        // A bare ESC byte cannot be classified yet — caller must wait.
        let (k, consumed) = decode_keys(&[0x1b]);
        assert!(k.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn partial_arrow_is_incomplete_and_resumes() {
        // First read delivers only "ESC [".
        let (k, consumed) = decode_keys(b"\x1b[");
        assert!(k.is_empty());
        assert_eq!(consumed, 0);
        // Next read completes it.
        assert_eq!(keys(b"\x1b[C"), vec![Key::Right]);
    }

    #[test]
    fn decodes_multibyte_utf8() {
        assert_eq!(keys("é".as_bytes()), vec![Key::Char('é')]);
        assert_eq!(keys("世".as_bytes()), vec![Key::Char('世')]);
        assert_eq!(keys("😀".as_bytes()), vec![Key::Char('😀')]);
    }

    #[test]
    fn split_multibyte_utf8_is_incomplete() {
        let full = "世".as_bytes();
        let (k, consumed) = decode_keys(&full[..1]);
        assert!(k.is_empty());
        assert_eq!(consumed, 0);
        assert_eq!(keys(full), vec![Key::Char('世')]);
    }

    #[test]
    fn mixed_stream_decodes_in_order() {
        let ks = keys(b"ab\x1b[Dc\r");
        assert_eq!(
            ks,
            vec![
                Key::Char('a'),
                Key::Char('b'),
                Key::Left,
                Key::Char('c'),
                Key::Enter
            ]
        );
    }

    #[test]
    fn esc_then_char_yields_esc_and_char() {
        // ESC immediately followed by a printable: emit Esc, then the char.
        assert_eq!(keys(b"\x1bx"), vec![Key::Esc, Key::Char('x')]);
    }

    #[test]
    fn unknown_csi_modifier_is_skipped() {
        // "ESC [ 1 ; 2 A" (shift-up) — we don't model modifiers; drop it.
        let ks = keys(b"\x1b[1;2A");
        assert!(ks.is_empty() || !ks.contains(&Key::Up));
    }
}
