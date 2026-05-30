//! ANSI-aware text measurement and wrapping for the full-screen renderer.
//!
//! The inline renderer never needed this — it let the terminal wrap. The
//! full-screen compositor lays out every row itself, so it must know the
//! *display* width of styled text (escape sequences occupy zero columns) and
//! wrap without splitting an escape sequence or counting it as a column.
//!
//! Scope: SGR / CSI sequences (`ESC [ … final`) and two-byte `ESC x` sequences,
//! which is everything this app emits. Wide CJK and combining marks still count
//! as one column — the same known limitation as [`crate::viewport`].

/// A display cell: any zero-width escape sequences that precede a visible
/// character, glued to that character so wrapping never separates them.
struct Cell {
    prefix: String,
    ch: char,
}

/// Split a line into (cells, trailing-escapes). Each cell is one visible column
/// with its leading escape codes; trailing escapes after the last visible char
/// are returned separately so they can ride along on the final row.
fn cells(line: &str) -> (Vec<Cell>, String) {
    let mut out = Vec::new();
    let mut pending = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            pending.push(c);
            // CSI: ESC [ ... final byte in 0x40..=0x7e.
            if chars.peek() == Some(&'[') {
                pending.push(chars.next().expect("just peeked '['"));
                for p in chars.by_ref() {
                    pending.push(p);
                    if ('\x40'..='\x7e').contains(&p) {
                        break;
                    }
                }
            } else if let Some(p) = chars.next() {
                // Two-byte escape (e.g. ESC 7).
                pending.push(p);
            }
        } else {
            out.push(Cell {
                prefix: std::mem::take(&mut pending),
                ch: c,
            });
        }
    }
    (out, pending)
}

/// Number of display columns a styled string occupies (escapes are zero-width).
pub fn display_width(s: &str) -> usize {
    cells(s).0.len()
}

fn join(cells: &[Cell]) -> String {
    let mut s = String::new();
    for c in cells {
        s.push_str(&c.prefix);
        s.push(c.ch);
    }
    s
}

/// Wrap one logical line to `width` display columns, keeping escape sequences
/// intact. Soft-breaks at the last space within a row, else hard-breaks a long
/// word — matching [`crate::viewport::wrap_line`] on plain text. An empty line
/// yields a single (possibly escape-only) row so blank lines still occupy space.
pub fn wrap_ansi(line: &str, width: usize) -> Vec<String> {
    let (cells, trailing) = cells(line);
    if width == 0 || cells.is_empty() {
        // Nothing to wrap: one row carrying any escape-only / empty content.
        let mut row = join(&cells);
        row.push_str(&trailing);
        return vec![row];
    }
    let mut rows: Vec<String> = Vec::new();
    let mut start = 0;
    while start < cells.len() {
        let hard_end = (start + width).min(cells.len());
        if hard_end == cells.len() {
            rows.push(join(&cells[start..hard_end]));
            break;
        }
        if cells[hard_end].ch == ' ' {
            rows.push(join(&cells[start..hard_end]));
            start = hard_end + 1; // drop the boundary space
            continue;
        }
        if let Some(sp) = (start + 1..hard_end).rev().find(|&i| cells[i].ch == ' ') {
            rows.push(join(&cells[start..sp]));
            start = sp + 1;
            continue;
        }
        rows.push(join(&cells[start..hard_end]));
        start = hard_end;
    }
    // Trailing escapes ride on the last row.
    if !trailing.is_empty() {
        if let Some(last) = rows.last_mut() {
            last.push_str(&trailing);
        } else {
            rows.push(trailing);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOLD: &str = "\x1b[1m";
    const RESET: &str = "\x1b[0m";

    #[test]
    fn display_width_ignores_escapes() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width(&format!("{BOLD}abc{RESET}")), 3);
        assert_eq!(display_width(""), 0);
        // Escape-only string is zero columns.
        assert_eq!(display_width(BOLD), 0);
    }

    #[test]
    fn display_width_handles_two_byte_escape() {
        // ESC 7 (save cursor) is zero-width.
        assert_eq!(display_width("\x1b7ab"), 2);
    }

    #[test]
    fn wrap_plain_matches_simple_wrap() {
        assert_eq!(
            wrap_ansi("hello world foo", 11),
            vec!["hello world".to_string(), "foo".to_string()]
        );
    }

    #[test]
    fn wrap_keeps_style_with_its_char() {
        // Bold spans the wrap; the code stays glued to the 'w'.
        let line = format!("hello {BOLD}world{RESET} foo");
        let rows = wrap_ansi(&line, 11);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].ends_with(&format!("{BOLD}world{RESET}")) || rows[0].contains(BOLD));
        // Display width of the first row is within the limit (escapes don't count).
        assert!(display_width(&rows[0]) <= 11);
    }

    #[test]
    fn wrap_does_not_count_escapes_toward_width() {
        // 5 visible chars wrapped at 5 → a single row, despite the escapes.
        let line = format!("{BOLD}a{RESET}{BOLD}b{RESET}cde");
        assert_eq!(wrap_ansi(&line, 5).len(), 1);
    }

    #[test]
    fn wrap_hard_breaks_long_word() {
        assert_eq!(
            wrap_ansi("abcdefghij", 4),
            vec!["abcd".to_string(), "efgh".to_string(), "ij".to_string()]
        );
    }

    #[test]
    fn empty_line_yields_one_row() {
        assert_eq!(wrap_ansi("", 10), vec![String::new()]);
    }

    #[test]
    fn escape_only_line_is_one_row() {
        assert_eq!(wrap_ansi(BOLD, 10), vec![BOLD.to_string()]);
    }

    #[test]
    fn width_zero_returns_unwrapped() {
        assert_eq!(wrap_ansi("abc", 0), vec!["abc".to_string()]);
    }

    #[test]
    fn trailing_escape_rides_last_row() {
        let line = format!("abcdef{RESET}");
        let rows = wrap_ansi(&line, 3);
        assert_eq!(rows.len(), 2);
        assert!(rows[1].ends_with(RESET));
    }

    // --- property / fuzz (std-only, deterministic) -----------------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Random line mixing visible chars, real SGR escapes, and multibyte UTF-8 —
    /// the inputs most likely to break width counting or split an escape.
    fn random_line(rng: &mut Rng) -> String {
        const POOL: &[&str] = &[
            "a", "Z", "0", " ", "中", "😀", "é", "\x1b[1m", "\x1b[0m", "\x1b[36m", "\x1b7",
            "\x1b[2m",
        ];
        let n = rng.below(20);
        (0..n)
            .map(|_| POOL[rng.below(POOL.len() as u64) as usize])
            .collect()
    }

    #[test]
    fn property_wrap_preserves_nonspace_chars_and_bounds_width() {
        // Invariants for every random (line, width):
        //   1. no row exceeds `width` display columns;
        //   2. concatenating the rows' NON-SPACE visible chars reproduces the
        //      original non-space visible chars exactly, in order.
        // Spaces are excluded because soft-wrap intentionally drops the single
        // space at a break boundary (same contract as viewport::wrap_line).
        let mut rng = Rng(0xA11C_E0F0_1234_5678);
        for _ in 0..5000 {
            let line = random_line(&mut rng);
            let width = (rng.below(12) + 1) as usize;
            let rows = wrap_ansi(&line, width);
            assert!(!rows.is_empty());
            for r in &rows {
                assert!(
                    display_width(r) <= width,
                    "row {r:?} width {} > {width}",
                    display_width(r)
                );
            }
            let visible: String = rows
                .iter()
                .flat_map(|r| visible_chars(r))
                .filter(|c| *c != ' ')
                .collect();
            let original: String = visible_chars(&line)
                .into_iter()
                .filter(|c| *c != ' ')
                .collect();
            assert_eq!(visible, original, "lost chars wrapping {line:?} @ {width}");
        }
    }

    /// Strip escape sequences, returning just the visible characters.
    fn visible_chars(s: &str) -> Vec<char> {
        let mut out = Vec::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for p in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&p) {
                            break;
                        }
                    }
                } else {
                    chars.next();
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn fuzz_wrap_never_panics_on_arbitrary_input() {
        let mut rng = Rng(0xBEEF_0000_FACE_1111);
        for _ in 0..20_000 {
            let len = rng.below(30) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            let s = String::from_utf8_lossy(&bytes);
            let width = rng.below(16) as usize; // includes 0 (the unwrapped path)
            let _ = wrap_ansi(&s, width); // must not panic
            let _ = display_width(&s);
        }
    }
}
