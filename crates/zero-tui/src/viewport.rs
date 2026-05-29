//! Scrollback buffer + word wrapping — the pure model behind the output pane.
//!
//! Width is measured in `char`s (one column per scalar). That is correct for
//! ASCII and most Latin text; wide CJK and zero-width combining marks are not
//! yet accounted for. TODO: switch to a display-width function when we add a
//! proper renderer — tracked as a known limitation, not a silent assumption.

/// Wrap one logical line to `width` columns. Breaks at the last space within a
/// row when possible (soft wrap), otherwise hard-breaks a long word. An empty
/// input yields a single empty row so blank lines still occupy space.
pub fn wrap_line(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let hard_end = (start + width).min(chars.len());
        if hard_end == chars.len() {
            // Everything left fits on one row.
            rows.push(chars[start..hard_end].iter().collect());
            break;
        }
        if chars[hard_end] == ' ' {
            // The width boundary lands exactly on a space: clean break, and
            // drop the space so it doesn't lead the next row.
            rows.push(chars[start..hard_end].iter().collect());
            start = hard_end + 1;
            continue;
        }
        // We'd split a word at the boundary — back up to the last space.
        if let Some(sp) = (start + 1..hard_end).rev().find(|&i| chars[i] == ' ') {
            rows.push(chars[start..sp].iter().collect());
            start = sp + 1; // drop the break space
            continue;
        }
        // No space to break on: hard-break the long word.
        rows.push(chars[start..hard_end].iter().collect());
        start = hard_end;
    }
    rows
}

/// An append-only log of output lines with a scroll position. The frontend asks
/// for [`Scrollback::view`] each frame to get exactly the rows to paint.
#[derive(Debug, Default)]
pub struct Scrollback {
    lines: Vec<String>,
    /// Rows scrolled *up* from the bottom. 0 means pinned to the latest output.
    scroll: usize,
}

impl Scrollback {
    pub fn new() -> Self {
        Scrollback::default()
    }

    /// Append one complete logical line.
    pub fn push_line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    /// Append streamed text to the current last line, starting new lines on
    /// embedded `\n`. Used to grow an assistant reply token-by-token.
    pub fn append(&mut self, text: &str) {
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        let mut parts = text.split('\n');
        if let Some(first) = parts.next() {
            self.lines.last_mut().unwrap().push_str(first);
        }
        for part in parts {
            self.lines.push(part.to_string());
        }
    }

    /// Number of logical lines stored.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    pub fn scroll_offset(&self) -> usize {
        self.scroll
    }

    /// All wrapped rows at the given width (used for scroll math and tests).
    fn all_rows(&self, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        for line in &self.lines {
            rows.extend(wrap_line(line, width));
        }
        rows
    }

    /// The rows to display: a bottom-anchored window of `height` rows into the
    /// wrapped content, shifted up by the current scroll offset.
    pub fn view(&self, width: usize, height: usize) -> Vec<String> {
        if height == 0 {
            return Vec::new();
        }
        let rows = self.all_rows(width);
        let total = rows.len();
        if total <= height {
            return rows;
        }
        // Clamp scroll so we never page past the top.
        let max_scroll = total - height;
        let scroll = self.scroll.min(max_scroll);
        let end = total - scroll;
        let start = end - height;
        rows[start..end].to_vec()
    }

    /// Scroll toward older output by `n` rows (clamped lazily at render time).
    pub fn scroll_up(&mut self, n: usize) {
        self.scroll = self.scroll.saturating_add(n);
    }

    /// Scroll toward newer output by `n` rows.
    pub fn scroll_down(&mut self, n: usize) {
        self.scroll = self.scroll.saturating_sub(n);
    }

    /// Pin back to the latest output.
    pub fn scroll_to_bottom(&mut self) {
        self.scroll = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_empty_yields_one_blank_row() {
        assert_eq!(wrap_line("", 10), vec![String::new()]);
    }

    #[test]
    fn wrap_short_line_is_unchanged() {
        assert_eq!(wrap_line("hello", 10), vec!["hello".to_string()]);
    }

    #[test]
    fn wrap_soft_breaks_at_space() {
        assert_eq!(
            wrap_line("hello world foo", 11),
            vec!["hello world".to_string(), "foo".to_string()]
        );
    }

    #[test]
    fn wrap_hard_breaks_long_word() {
        assert_eq!(
            wrap_line("abcdefghij", 4),
            vec!["abcd".to_string(), "efgh".to_string(), "ij".to_string()]
        );
    }

    #[test]
    fn wrap_reassembles_to_original_chars() {
        let text = "the quick brown fox jumped";
        let joined: String = wrap_line(text, 7).join(" ");
        // Every original word survives wrapping.
        for word in text.split(' ') {
            assert!(joined.contains(word), "missing {word} in {joined}");
        }
    }

    #[test]
    fn append_grows_last_line_and_splits_newlines() {
        let mut sb = Scrollback::new();
        sb.push_line("assistant: ");
        sb.append("hello");
        sb.append(" world");
        assert_eq!(sb.len(), 1);
        sb.append("\nsecond line");
        assert_eq!(sb.len(), 2);
        assert_eq!(
            sb.view(80, 10),
            vec!["assistant: hello world", "second line"]
        );
    }

    #[test]
    fn view_is_bottom_anchored() {
        let mut sb = Scrollback::new();
        for i in 0..10 {
            sb.push_line(format!("line {i}"));
        }
        let v = sb.view(80, 3);
        assert_eq!(v, vec!["line 7", "line 8", "line 9"]);
    }

    #[test]
    fn view_returns_all_when_content_fits() {
        let mut sb = Scrollback::new();
        sb.push_line("a");
        sb.push_line("b");
        assert_eq!(sb.view(80, 10), vec!["a", "b"]);
    }

    #[test]
    fn scrolling_up_shows_older_rows_and_clamps() {
        let mut sb = Scrollback::new();
        for i in 0..10 {
            sb.push_line(format!("line {i}"));
        }
        sb.scroll_up(2);
        assert_eq!(sb.view(80, 3), vec!["line 5", "line 6", "line 7"]);
        sb.scroll_up(1000); // clamps at top
        assert_eq!(sb.view(80, 3), vec!["line 0", "line 1", "line 2"]);
        sb.scroll_to_bottom();
        assert_eq!(sb.view(80, 3), vec!["line 7", "line 8", "line 9"]);
    }

    #[test]
    fn wrapping_counts_toward_scroll_rows() {
        let mut sb = Scrollback::new();
        // One logical line that wraps into 3 rows at width 5.
        sb.push_line("aaaaabbbbbccccc");
        sb.push_line("tail");
        let v = sb.view(5, 2);
        assert_eq!(v, vec!["ccccc", "tail"]);
    }
}
