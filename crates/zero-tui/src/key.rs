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
    /// Word-wise cursor moves (Ctrl/Alt + Left/Right).
    WordLeft,
    WordRight,
    /// Shift/Alt+Enter — insert a newline instead of submitting. Only sent by
    /// terminals that distinguish it (CSI-u, or meta+Enter as `ESC CR`).
    ShiftEnter,
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
        // Return sends CR in raw mode → submit. Ctrl+J sends LF → insert a
        // newline (a universal multiline key that works in every terminal,
        // since Shift+Enter is only distinguishable on some).
        b'\r' => Step::Key(Key::Enter, 1),
        b'\n' => Step::Key(Key::ShiftEnter, 1),
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
        // Meta/Alt + Enter (`ESC CR`/`ESC LF`) → insert a newline.
        b'\r' | b'\n' => Step::Key(Key::ShiftEnter, 2),
        // ESC followed by something else: treat the ESC alone as a press and
        // let the following byte decode on its own next iteration.
        _ => Step::Key(Key::Esc, 1),
    }
}

/// Decode a CSI / SS3 sequence: `ESC [ …` or `ESC O …`.
///
/// Handles the modern parameterized form `ESC [ <p1> ; <p2> <final>` where `p2`
/// is a modifier (2=Shift, 3=Alt, 5=Ctrl), plus the `~`-terminated numeric form
/// and the CSI-u form `ESC [ <code> ; <mod> u`.
fn decode_csi(buf: &[u8]) -> Step {
    // buf[0] == ESC, buf[1] == '[' or 'O'.
    let mut i = 2;
    // Collect up to two numeric params separated by ';'.
    let mut params = [0u32; 2];
    let mut nparams = 0usize;
    while i < buf.len() {
        let c = buf[i];
        if c.is_ascii_digit() {
            if nparams == 0 {
                nparams = 1;
            }
            let slot = nparams - 1;
            if slot < params.len() {
                params[slot] = params[slot].saturating_mul(10) + u32::from(c - b'0');
            }
            i += 1;
        } else if c == b';' {
            nparams = (nparams + 1).min(params.len() + 1);
            i += 1;
        } else {
            break;
        }
    }
    if i >= buf.len() {
        return Step::NeedMore; // final byte not yet arrived
    }
    let final_byte = buf[i];
    let total = i + 1;
    let modifier = params.get(1).copied().unwrap_or(0);
    let is_word = matches!(modifier, 3 | 5 | 7); // Alt / Ctrl (7 = Ctrl+Alt)

    let key = match final_byte {
        b'A' => Key::Up,
        b'B' => Key::Down,
        b'C' if is_word => Key::WordRight,
        b'D' if is_word => Key::WordLeft,
        b'C' => Key::Right,
        b'D' => Key::Left,
        b'H' => Key::Home,
        b'F' => Key::End,
        b'u' => {
            // CSI-u (kitty keyboard protocol): param0 = codepoint, param1 = mods.
            // With the protocol enabled, Shift+Enter arrives here as `13;2u`.
            return match params[0] {
                // Enter: plain → submit; with any modifier (Shift) → newline.
                13 => Step::Key(
                    if modifier >= 2 {
                        Key::ShiftEnter
                    } else {
                        Key::Enter
                    },
                    total,
                ),
                27 => Step::Key(Key::Esc, total),
                9 => Step::Key(Key::Tab, total),
                127 => Step::Key(Key::Backspace, total),
                _ => Step::Consume(total),
            };
        }
        b'~' => match params[0] {
            1 | 7 => Key::Home,
            4 | 8 => Key::End,
            3 => Key::Delete,
            _ => return Step::Consume(total),
        },
        // Unknown final byte; drop the whole sequence.
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
        assert_eq!(keys(b"\r"), vec![Key::Enter]); // Return = CR = submit
        assert_eq!(keys(b"\t"), vec![Key::Tab]);
        assert_eq!(keys(&[0x7f]), vec![Key::Backspace]);
        assert_eq!(keys(&[0x08]), vec![Key::Backspace]);
    }

    #[test]
    fn ctrl_j_lf_inserts_newline() {
        // Ctrl+J / LF → newline, the universal multiline key.
        assert_eq!(keys(b"\n"), vec![Key::ShiftEnter]);
        assert_eq!(keys(&[0x0a]), vec![Key::ShiftEnter]);
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
    fn invalid_lead_byte_is_consumed_without_a_key() {
        // 0x80 is a stray UTF-8 continuation byte — not a valid lead.
        let (k, consumed) = decode_keys(&[0x80]);
        assert!(k.is_empty());
        assert_eq!(consumed, 1);
    }

    #[test]
    fn malformed_multibyte_resyncs_by_one_byte() {
        // 0xE0 claims a 3-byte sequence but the continuations are invalid lead
        // bytes too, so every byte is dropped without producing a key.
        let (k, consumed) = decode_keys(&[0xE0, 0x80, 0x80]);
        assert!(k.is_empty());
        assert_eq!(consumed, 3);
    }

    #[test]
    fn unmapped_low_controls_are_dropped() {
        // NUL and FS are consumed silently (no Key).
        assert!(keys(&[0x00]).is_empty());
        assert!(keys(&[0x1c]).is_empty());
    }

    #[test]
    fn unknown_csi_final_byte_is_consumed() {
        // ESC [ Z (back-tab) — not modeled; consumed, no key.
        let (k, consumed) = decode_keys(b"\x1b[Z");
        assert!(k.is_empty());
        assert_eq!(consumed, 3);
    }

    #[test]
    fn unterminated_csi_numeric_waits_for_more() {
        // "ESC [ 1" with no terminator yet → incomplete.
        let (k, consumed) = decode_keys(b"\x1b[1");
        assert!(k.is_empty());
        assert_eq!(consumed, 0);
    }

    #[test]
    fn shift_modifier_on_arrow_falls_back_to_plain_arrow() {
        // "ESC [ 1 ; 2 A" (shift-up): we don't model shift on arrows → Up.
        assert_eq!(keys(b"\x1b[1;2A"), vec![Key::Up]);
    }

    #[test]
    fn ctrl_and_alt_arrows_are_word_moves() {
        assert_eq!(keys(b"\x1b[1;5D"), vec![Key::WordLeft]); // ctrl+left
        assert_eq!(keys(b"\x1b[1;5C"), vec![Key::WordRight]); // ctrl+right
        assert_eq!(keys(b"\x1b[1;3D"), vec![Key::WordLeft]); // alt+left
        assert_eq!(keys(b"\x1b[1;3C"), vec![Key::WordRight]); // alt+right
    }

    #[test]
    fn shift_enter_via_csi_u() {
        assert_eq!(keys(b"\x1b[13;2u"), vec![Key::ShiftEnter]);
    }

    #[test]
    fn csi_u_legacy_keys_when_kitty_protocol_enabled() {
        // With the kitty protocol on, these report as CSI-u.
        assert_eq!(keys(b"\x1b[13u"), vec![Key::Enter]); // plain Enter → submit
        assert_eq!(keys(b"\x1b[27u"), vec![Key::Esc]);
        assert_eq!(keys(b"\x1b[9u"), vec![Key::Tab]);
        assert_eq!(keys(b"\x1b[127u"), vec![Key::Backspace]);
    }

    #[test]
    fn meta_enter_inserts_newline() {
        assert_eq!(keys(b"\x1b\r"), vec![Key::ShiftEnter]);
        assert_eq!(keys(b"\x1b\n"), vec![Key::ShiftEnter]);
    }

    #[test]
    fn ctrl_r_decodes_as_chord() {
        assert_eq!(keys(&[0x12]), vec![Key::Ctrl('r')]);
    }

    #[test]
    fn unknown_csi_u_codepoint_is_dropped() {
        // CSI-u for some other key we don't map.
        let (k, consumed) = decode_keys(b"\x1b[99;2u");
        assert!(k.is_empty());
        assert_eq!(consumed, 7);
    }

    #[test]
    fn unmapped_numeric_csi_is_dropped() {
        // ESC [ 5 ~ (PageUp) — we don't map it; consumed, no key.
        let (k, consumed) = decode_keys(b"\x1b[5~");
        assert!(k.is_empty());
        assert_eq!(consumed, 4);
    }

    #[test]
    fn partial_modifier_sequence_waits_for_more() {
        let (k, consumed) = decode_keys(b"\x1b[1;");
        assert!(k.is_empty());
        assert_eq!(consumed, 0);
    }
}
