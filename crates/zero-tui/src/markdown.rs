//! A tiny streaming Markdown → ANSI renderer for assistant output.
//!
//! Tokens arrive a few characters at a time, so `**bold**` is frequently split
//! across chunks. [`MarkdownStream`] is a stateful filter: feed it each chunk,
//! write what it returns, and call [`MarkdownStream::finish`] at the end. It
//! handles the inline subset that matters for chat: `**bold**`, `*italic*`,
//! `` `code` ``, and `#` headings. It is pure (no I/O), so it is fully tested.

/// SGR codes.
const BOLD_ON: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const ITALIC_ON: &str = "\x1b[3m";
const ITALIC_OFF: &str = "\x1b[23m";
const CODE_ON: &str = "\x1b[36m"; // cyan reads as inline code
const CODE_OFF: &str = "\x1b[39m";
const RESET: &str = "\x1b[0m";

/// Incremental Markdown renderer. One per assistant response.
#[derive(Debug, Default)]
pub struct MarkdownStream {
    bold: bool,
    italic: bool,
    code: bool,
    /// We've seen one `*` and are waiting to see if a second makes it `**`.
    pending_star: bool,
    /// Cursor is at the start of a line (for heading detection).
    at_line_start: bool,
    /// Currently bolding a `#` heading line.
    heading: bool,
    /// Still consuming the leading `#`/space run of a heading.
    heading_marker: bool,
}

impl MarkdownStream {
    pub fn new() -> Self {
        MarkdownStream {
            at_line_start: true,
            ..Default::default()
        }
    }

    /// Feed a chunk; returns the ANSI-rendered text to write.
    pub fn feed(&mut self, chunk: &str) -> String {
        let mut out = String::with_capacity(chunk.len() + 8);
        for ch in chunk.chars() {
            // A lone `*` followed by a non-`*` was italic, not the start of `**`.
            if self.pending_star && ch != '*' {
                self.toggle_italic(&mut out);
                self.pending_star = false;
            }
            match ch {
                '\n' => {
                    if self.heading {
                        out.push_str(BOLD_OFF);
                        self.heading = false;
                    }
                    out.push('\n');
                    self.at_line_start = true;
                    self.heading_marker = false;
                }
                '#' if self.at_line_start => {
                    if !self.heading {
                        out.push_str(BOLD_ON);
                        self.heading = true;
                    }
                    self.heading_marker = true; // suppress the marker glyph
                }
                ' ' if self.heading_marker => {
                    self.heading_marker = false; // swallow one space after `#`
                    self.at_line_start = false;
                }
                '*' => {
                    if self.pending_star {
                        self.toggle_bold(&mut out);
                        self.pending_star = false;
                    } else {
                        self.pending_star = true;
                    }
                    self.at_line_start = false;
                    self.heading_marker = false;
                }
                '`' => {
                    self.toggle_code(&mut out);
                    self.at_line_start = false;
                    self.heading_marker = false;
                }
                other => {
                    out.push(other);
                    self.at_line_start = false;
                    self.heading_marker = false;
                }
            }
        }
        out
    }

    /// Flush trailing state at end of the response and clear any active styling.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if self.pending_star {
            out.push('*'); // a trailing lone `*` was literal
            self.pending_star = false;
        }
        if self.bold || self.italic || self.code || self.heading {
            out.push_str(RESET);
            self.bold = false;
            self.italic = false;
            self.code = false;
            self.heading = false;
        }
        out
    }

    fn toggle_bold(&mut self, out: &mut String) {
        self.bold = !self.bold;
        out.push_str(if self.bold { BOLD_ON } else { BOLD_OFF });
    }

    fn toggle_italic(&mut self, out: &mut String) {
        self.italic = !self.italic;
        out.push_str(if self.italic { ITALIC_ON } else { ITALIC_OFF });
    }

    fn toggle_code(&mut self, out: &mut String) {
        self.code = !self.code;
        out.push_str(if self.code { CODE_ON } else { CODE_OFF });
    }
}

/// Convenience: render a whole string in one shot.
pub fn render(text: &str) -> String {
    let mut md = MarkdownStream::new();
    let mut out = md.feed(text);
    out.push_str(&md.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bold_in_one_chunk() {
        assert_eq!(render("**3**"), format!("{BOLD_ON}3{BOLD_OFF}"));
    }

    #[test]
    fn bold_split_across_chunks() {
        let mut md = MarkdownStream::new();
        let mut out = md.feed("**");
        out.push_str(&md.feed("3"));
        out.push_str(&md.feed("**"));
        out.push_str(&md.finish());
        assert_eq!(out, format!("{BOLD_ON}3{BOLD_OFF}"));
    }

    #[test]
    fn italic_then_text() {
        // Closing `*` is followed by a space, so italic closes cleanly.
        assert_eq!(render("*hi* x"), format!("{ITALIC_ON}hi{ITALIC_OFF} x"));
    }

    #[test]
    fn inline_code() {
        assert_eq!(render("`x`"), format!("{CODE_ON}x{CODE_OFF}"));
    }

    #[test]
    fn heading_is_bolded_and_marker_hidden() {
        assert_eq!(
            render("# Title\nbody"),
            format!("{BOLD_ON}Title{BOLD_OFF}\nbody")
        );
    }

    #[test]
    fn multi_hash_heading_marker_hidden() {
        assert_eq!(render("### Big\n"), format!("{BOLD_ON}Big{BOLD_OFF}\n"));
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(render("just words, no markup"), "just words, no markup");
    }

    #[test]
    fn dashes_are_left_alone() {
        // The strawberry breakdown: dashes must not be treated as markup.
        assert_eq!(render("s-t-a-w"), "s-t-a-w");
    }

    #[test]
    fn trailing_lone_star_is_literal() {
        // No style was open, so no reset is appended.
        assert_eq!(render("5 *"), "5 *");
    }

    #[test]
    fn unclosed_bold_is_reset_at_end() {
        let out = render("**oops");
        assert!(out.starts_with(BOLD_ON));
        assert!(out.ends_with(RESET));
    }

    #[test]
    fn finish_is_empty_when_nothing_open() {
        let mut md = MarkdownStream::new();
        md.feed("plain");
        assert_eq!(md.finish(), "");
    }
}
