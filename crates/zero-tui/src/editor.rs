//! A single-line input editor with readline-style editing and history.
//!
//! Pure model: it owns a character buffer and a cursor and exposes operations
//! that mutate them. No rendering, no I/O — the shell renders `buffer()` with
//! the cursor at `cursor()`. Char-indexed throughout so multi-byte input and
//! cursor math stay correct.

/// An editable input line plus command history.
#[derive(Debug, Default)]
pub struct LineEditor {
    buf: Vec<char>,
    /// Cursor position as a char index in `0..=buf.len()`.
    cursor: usize,
    history: Vec<String>,
    /// `None` while editing the live line; `Some(i)` while browsing history.
    hist_pos: Option<usize>,
    /// The live draft, stashed when the user steps into history.
    stash: Vec<char>,
}

impl LineEditor {
    pub fn new() -> Self {
        LineEditor::default()
    }

    /// Current line contents.
    pub fn text(&self) -> String {
        self.buf.iter().collect()
    }

    /// Cursor position as a char index.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Insert a character at the cursor.
    pub fn insert(&mut self, c: char) {
        self.buf.insert(self.cursor, c);
        self.cursor += 1;
    }

    /// Insert a string at the cursor (a bracketed paste). Carriage returns are
    /// normalized to newlines (`\r\n` and lone `\r` → `\n`) so pasted text neither
    /// submits nor leaves stray CRs; other control characters (except `\t`) are
    /// dropped so a pasted escape/NUL can't corrupt the buffer or the render.
    pub fn insert_str(&mut self, s: &str) {
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\r' => {
                    self.insert('\n');
                    if chars.peek() == Some(&'\n') {
                        chars.next(); // collapse a CRLF pair into the one newline
                    }
                }
                '\n' | '\t' => self.insert(c),
                c if !c.is_control() => self.insert(c),
                _ => {} // drop other controls (ESC, NUL, …)
            }
        }
    }

    /// Delete the character before the cursor (Backspace).
    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.buf.remove(self.cursor);
        }
    }

    /// Delete the character at the cursor (Delete / ^D mid-line).
    pub fn delete(&mut self) {
        if self.cursor < self.buf.len() {
            self.buf.remove(self.cursor);
        }
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub fn right(&mut self) {
        if self.cursor < self.buf.len() {
            self.cursor += 1;
        }
    }

    /// Insert a newline at the cursor (Shift/Alt+Enter) — enables multiline.
    pub fn insert_newline(&mut self) {
        self.insert('\n');
    }

    /// True if the buffer spans more than one line.
    pub fn is_multiline(&self) -> bool {
        self.buf.contains(&'\n')
    }

    /// Start index of the line the cursor is on (just after the previous `\n`).
    fn line_start(&self) -> usize {
        self.buf[..self.cursor]
            .iter()
            .rposition(|&c| c == '\n')
            .map_or(0, |i| i + 1)
    }

    /// End index of the current line (index of the next `\n`, or buffer end).
    fn line_end(&self) -> usize {
        self.buf[self.cursor..]
            .iter()
            .position(|&c| c == '\n')
            .map_or(self.buf.len(), |i| self.cursor + i)
    }

    /// Move to the start of the current line (Home / ^A).
    pub fn home(&mut self) {
        self.cursor = self.line_start();
    }

    /// Move to the end of the current line (End / ^E).
    pub fn end(&mut self) {
        self.cursor = self.line_end();
    }

    /// Move the cursor left by one word (over whitespace, then over the word).
    pub fn word_left(&mut self) {
        let is_ws = |c: char| c.is_whitespace();
        while self.cursor > 0 && is_ws(self.buf[self.cursor - 1]) {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !is_ws(self.buf[self.cursor - 1]) {
            self.cursor -= 1;
        }
    }

    /// Move the cursor right by one word (over the word, then over whitespace).
    pub fn word_right(&mut self) {
        let n = self.buf.len();
        let is_ws = |c: char| c.is_whitespace();
        while self.cursor < n && !is_ws(self.buf[self.cursor]) {
            self.cursor += 1;
        }
        while self.cursor < n && is_ws(self.buf[self.cursor]) {
            self.cursor += 1;
        }
    }

    /// Move up one visual line, keeping the column. Returns false if already on
    /// the first line (so the caller can fall back to history recall).
    pub fn line_up(&mut self) -> bool {
        let start = self.line_start();
        if start == 0 {
            return false;
        }
        let col = self.cursor - start;
        let prev_start = self.buf[..start - 1]
            .iter()
            .rposition(|&c| c == '\n')
            .map_or(0, |i| i + 1);
        let prev_end = start - 1; // the '\n' position
        self.cursor = (prev_start + col).min(prev_end);
        true
    }

    /// Move down one visual line, keeping the column. Returns false if already on
    /// the last line (so the caller can fall back to history).
    pub fn line_down(&mut self) -> bool {
        let end = self.line_end();
        if end >= self.buf.len() {
            return false;
        }
        let col = self.cursor - self.line_start();
        let next_start = end + 1;
        let next_end = self.buf[next_start..]
            .iter()
            .position(|&c| c == '\n')
            .map_or(self.buf.len(), |i| next_start + i);
        self.cursor = (next_start + col).min(next_end);
        true
    }

    /// Replace the whole buffer (used to accept a history-search match).
    pub fn set_text(&mut self, text: &str) {
        self.buf = text.chars().collect();
        self.cursor = self.buf.len();
        self.hist_pos = None;
    }

    /// Kill from the cursor to the end of the current line (^K).
    pub fn kill_to_end(&mut self) {
        let end = self.line_end();
        self.buf.drain(self.cursor..end);
    }

    /// Kill from the start of the current line to the cursor (^U).
    pub fn kill_to_start(&mut self) {
        let start = self.line_start();
        self.buf.drain(start..self.cursor);
        self.cursor = start;
    }

    /// Kill the word before the cursor (^W): trailing whitespace, then the word.
    pub fn kill_word(&mut self) {
        let is_ws = |c: char| c.is_whitespace();
        let mut start = self.cursor;
        while start > 0 && is_ws(self.buf[start - 1]) {
            start -= 1;
        }
        while start > 0 && !is_ws(self.buf[start - 1]) {
            start -= 1;
        }
        self.buf.drain(start..self.cursor);
        self.cursor = start;
    }

    /// Clear the line entirely.
    pub fn clear(&mut self) {
        self.buf.clear();
        self.cursor = 0;
        self.hist_pos = None;
    }

    /// Commit the current line: returns its text, pushes non-empty distinct
    /// lines to history, and resets to an empty live line.
    pub fn submit(&mut self) -> String {
        let text: String = self.buf.iter().collect();
        if !text.trim().is_empty() && self.history.last().map(String::as_str) != Some(text.as_str())
        {
            self.history.push(text.clone());
        }
        self.clear();
        text
    }

    /// Step back into older history (Up).
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        match self.hist_pos {
            None => {
                // Entering history: stash the live draft.
                self.stash = std::mem::take(&mut self.buf);
                self.hist_pos = Some(self.history.len() - 1);
            }
            Some(0) => return, // already at oldest
            Some(i) => self.hist_pos = Some(i - 1),
        }
        self.load_history_entry();
    }

    /// Step forward toward the live draft (Down).
    pub fn history_next(&mut self) {
        match self.hist_pos {
            None => {}
            Some(i) if i + 1 < self.history.len() => {
                self.hist_pos = Some(i + 1);
                self.load_history_entry();
            }
            Some(_) => {
                // Past the newest entry → restore the stashed live draft.
                self.hist_pos = None;
                self.buf = std::mem::take(&mut self.stash);
                self.cursor = self.buf.len();
            }
        }
    }

    fn load_history_entry(&mut self) {
        if let Some(i) = self.hist_pos {
            self.buf = self.history[i].chars().collect();
            self.cursor = self.buf.len();
        }
    }

    /// Read-only view of history (for tests / persistence later).
    pub fn history(&self) -> &[String] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(s: &str) -> LineEditor {
        let mut e = LineEditor::new();
        for c in s.chars() {
            e.insert(c);
        }
        e
    }

    #[test]
    fn inserts_and_reads_text() {
        let e = typed("hello");
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), 5);
    }

    #[test]
    fn insert_str_pastes_multiline_normalizing_crlf() {
        let mut e = LineEditor::new();
        // CRLF, lone CR, and LF all become a single '\n'; tabs survive.
        e.insert_str("line1\r\nline2\rline3\n\tindented");
        assert_eq!(e.text(), "line1\nline2\nline3\n\tindented");
        assert_eq!(e.cursor(), e.text().chars().count());
    }

    #[test]
    fn insert_str_drops_stray_control_chars() {
        let mut e = LineEditor::new();
        // A pasted ESC and NUL must not land in the buffer; printables/newlines do.
        e.insert_str("a\x1b[Db\0c\n");
        assert_eq!(e.text(), "a[Dbc\n"); // ESC byte dropped, its '[D' remain as text
        assert!(!e.text().contains('\u{1b}') && !e.text().contains('\0'));
    }

    #[test]
    fn insert_str_at_cursor_keeps_surrounding_text() {
        let mut e = typed("ac");
        e.left(); // between a and c
        e.insert_str("XY");
        assert_eq!(e.text(), "aXYc");
        assert_eq!(e.cursor(), 3);
    }

    #[test]
    fn backspace_removes_before_cursor() {
        let mut e = typed("abc");
        e.backspace();
        assert_eq!(e.text(), "ab");
        assert_eq!(e.cursor(), 2);
    }

    #[test]
    fn cursor_movement_and_midline_insert() {
        let mut e = typed("ac");
        e.left(); // between a and c
        e.insert('b');
        assert_eq!(e.text(), "abc");
        assert_eq!(e.cursor(), 2);
    }

    #[test]
    fn home_end_bound_the_line() {
        let mut e = typed("abc");
        e.home();
        assert_eq!(e.cursor(), 0);
        e.left(); // can't go below 0
        assert_eq!(e.cursor(), 0);
        e.end();
        assert_eq!(e.cursor(), 3);
        e.right(); // can't exceed len
        assert_eq!(e.cursor(), 3);
    }

    #[test]
    fn delete_removes_at_cursor() {
        let mut e = typed("abc");
        e.home();
        e.delete();
        assert_eq!(e.text(), "bc");
        assert_eq!(e.cursor(), 0);
    }

    #[test]
    fn kill_to_end_and_start() {
        let mut e = typed("hello world");
        e.home();
        e.right();
        e.right();
        e.right();
        e.right();
        e.right(); // cursor after "hello"
        e.kill_to_end();
        assert_eq!(e.text(), "hello");
        e.kill_to_start();
        assert_eq!(e.text(), "");
    }

    #[test]
    fn kill_word_eats_trailing_space_and_word() {
        let mut e = typed("foo bar ");
        e.kill_word();
        assert_eq!(e.text(), "foo ");
        e.kill_word();
        assert_eq!(e.text(), "");
    }

    #[test]
    fn submit_returns_text_and_clears() {
        let mut e = typed("run tests");
        let out = e.submit();
        assert_eq!(out, "run tests");
        assert!(e.is_empty());
        assert_eq!(e.history(), &["run tests".to_string()]);
    }

    #[test]
    fn submit_skips_blank_and_consecutive_dupes() {
        let mut e = typed("   ");
        e.submit();
        assert!(e.history().is_empty());

        for _ in 0..2 {
            for c in "same".chars() {
                e.insert(c);
            }
            e.submit();
        }
        assert_eq!(e.history().len(), 1);
    }

    #[test]
    fn history_prev_next_navigates() {
        let mut e = LineEditor::new();
        for line in ["first", "second"] {
            for c in line.chars() {
                e.insert(c);
            }
            e.submit();
        }
        // Type a live draft, then browse history.
        e.insert('x');
        e.history_prev();
        assert_eq!(e.text(), "second");
        e.history_prev();
        assert_eq!(e.text(), "first");
        e.history_prev(); // clamp at oldest
        assert_eq!(e.text(), "first");
        e.history_next();
        assert_eq!(e.text(), "second");
        e.history_next(); // back to the stashed live draft
        assert_eq!(e.text(), "x");
    }

    #[test]
    fn history_navigation_on_empty_history_is_noop() {
        let mut e = LineEditor::new();
        e.history_prev(); // nothing to recall
        e.history_next(); // not browsing history
        assert!(e.is_empty());
        assert!(e.history().is_empty());
    }

    #[test]
    fn clear_resets_buffer_and_history_browsing() {
        let mut e = typed("draft");
        e.clear();
        assert!(e.is_empty());
        assert_eq!(e.cursor(), 0);
    }

    #[test]
    fn newline_makes_multiline_and_home_end_are_line_local() {
        let mut e = typed("ab");
        e.insert_newline();
        for c in "cd".chars() {
            e.insert(c);
        }
        assert_eq!(e.text(), "ab\ncd");
        assert!(e.is_multiline());
        // Cursor on the second line; Home/End stay on that line.
        e.home();
        assert_eq!(e.cursor(), 3); // start of "cd"
        e.end();
        assert_eq!(e.cursor(), 5); // end of "cd"
    }

    #[test]
    fn line_up_down_keep_column_and_report_edges() {
        let mut e = LineEditor::new();
        for c in "abc\nde\nfghi".chars() {
            if c == '\n' {
                e.insert_newline();
            } else {
                e.insert(c);
            }
        }
        // Cursor at end of "fghi" (col 4 on last line).
        assert!(!e.line_down(), "already on last line");
        assert!(e.line_up()); // to "de" — clamps to its end (col 2)
        assert_eq!(e.cursor(), 6); // index of 'd'(4) 'e'(5) end=6
        assert!(e.line_up()); // to "abc"
        assert!(!e.line_up(), "already on first line");
        e.line_down();
        assert!(e.is_multiline());
    }

    #[test]
    fn word_moves_skip_words_and_whitespace() {
        let mut e = typed("foo bar baz");
        e.word_left();
        assert_eq!(e.cursor(), 8); // start of "baz"
        e.word_left();
        assert_eq!(e.cursor(), 4); // start of "bar"
        e.home();
        e.word_right();
        assert_eq!(e.cursor(), 4); // past "foo " to start of "bar"
    }

    #[test]
    fn kill_to_end_is_line_local_in_multiline() {
        let mut e = LineEditor::new();
        for c in "abc\ndef".chars() {
            if c == '\n' {
                e.insert_newline();
            } else {
                e.insert(c);
            }
        }
        e.home(); // start of "def"
        e.kill_to_end();
        assert_eq!(e.text(), "abc\n");
    }

    #[test]
    fn set_text_replaces_buffer_and_parks_cursor_at_end() {
        let mut e = typed("old");
        e.set_text("recalled command");
        assert_eq!(e.text(), "recalled command");
        assert_eq!(e.cursor(), 16);
    }

    #[test]
    fn multibyte_editing_is_char_correct() {
        let mut e = typed("héllo");
        assert_eq!(e.cursor(), 5);
        e.backspace();
        assert_eq!(e.text(), "héll");
        e.home();
        e.right();
        e.insert('X');
        assert_eq!(e.text(), "hXéll");
    }
}
