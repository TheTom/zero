//! A tiny streaming Markdown → ANSI renderer for assistant output, plus a
//! fenced-code-block extractor for per-section copy (`/clip <n>`).
//!
//! Tokens arrive a few characters at a time, so markup like `**bold**` or a
//! ```` ``` ```` fence is frequently split across chunks. [`MarkdownStream`] is a
//! stateful filter: feed it each chunk, write what it returns, and call
//! [`MarkdownStream::finish`] at the end. It handles the inline subset that
//! matters for chat — `**bold**`, `*italic*`, `` `code` ``, `#` headings — and
//! fenced code blocks (rendered dim, markers hidden). Pure, so it is fully
//! tested.

const BOLD_ON: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const ITALIC_ON: &str = "\x1b[3m";
const ITALIC_OFF: &str = "\x1b[23m";
const CODE_ON: &str = "\x1b[36m"; // cyan inline code
const CODE_OFF: &str = "\x1b[39m";
const DIM_ON: &str = "\x1b[2m"; // fenced code block
const DIM_OFF: &str = "\x1b[22m";
const RESET: &str = "\x1b[0m";

/// Incremental Markdown renderer. One per assistant response.
#[derive(Debug, Default)]
pub struct MarkdownStream {
    bold: bool,
    italic: bool,
    code: bool,
    in_fence: bool,
    /// Saw one `*`, waiting to see if a second makes it `**`.
    pending_star: bool,
    /// Count of consecutive backticks since line start (fence detection).
    pending_backticks: u8,
    /// True until a non-backtick char appears on the current line.
    line_only_backticks: bool,
    /// Cursor is at the start of a line.
    at_line_start: bool,
    /// Swallow the rest of the line (trailing text after a closing fence).
    suppress_eol: bool,
    /// Accumulating a fence's language tag before emitting the block header.
    collecting_lang: bool,
    lang_buf: String,
    heading: bool,
    heading_marker: bool,
}

impl MarkdownStream {
    pub fn new() -> Self {
        MarkdownStream {
            at_line_start: true,
            line_only_backticks: true,
            ..Default::default()
        }
    }

    /// Feed a chunk; returns the ANSI-rendered text to write.
    pub fn feed(&mut self, chunk: &str) -> String {
        let mut out = String::with_capacity(chunk.len() + 8);
        for ch in chunk.chars() {
            // Collect (and suppress) a fence's language tag. The `/clip` marker
            // is emitted as a FOOTER when the block closes, not here.
            if self.collecting_lang {
                if ch == '\n' {
                    self.collecting_lang = false;
                    self.at_line_start = true;
                    self.line_only_backticks = true;
                } else {
                    self.lang_buf.push(ch);
                }
                continue;
            }
            if self.suppress_eol {
                if ch == '\n' {
                    self.newline(&mut out);
                }
                continue;
            }
            if ch == '\n' {
                self.resolve_pending_backticks(&mut out);
                self.newline(&mut out);
                continue;
            }
            // Fence delimiter: three backticks starting a line.
            if ch == '`' && self.at_line_start && self.line_only_backticks {
                self.pending_backticks += 1;
                if self.pending_backticks == 3 {
                    self.pending_backticks = 0;
                    self.line_only_backticks = false;
                    if self.in_fence {
                        self.in_fence = false;
                        // Footer below the block; its line ends with the close
                        // fence's own newline (suppress_eol).
                        self.emit_block_footer(&mut out);
                        self.suppress_eol = true;
                    } else {
                        self.in_fence = true;
                        self.collecting_lang = true;
                        self.lang_buf.clear();
                    }
                }
                continue;
            }
            // Fewer than three leading backticks → not a fence; resolve them.
            if self.pending_backticks > 0 {
                self.resolve_pending_backticks(&mut out);
            }
            self.line_only_backticks = false;

            if self.in_fence {
                out.push(ch); // raw inside a code block (no inline parsing)
                self.at_line_start = false;
                continue;
            }
            self.inline(ch, &mut out);
        }
        out
    }

    /// Inline (non-fenced) character handling.
    fn inline(&mut self, ch: char, out: &mut String) {
        if self.pending_star && ch != '*' {
            self.toggle_italic(out);
            self.pending_star = false;
        }
        match ch {
            '#' if self.at_line_start => {
                if !self.heading {
                    out.push_str(BOLD_ON);
                    self.heading = true;
                }
                self.heading_marker = true;
            }
            ' ' if self.heading_marker => {
                self.heading_marker = false;
                self.at_line_start = false;
            }
            '*' => {
                if self.pending_star {
                    self.toggle_bold(out);
                    self.pending_star = false;
                } else {
                    self.pending_star = true;
                }
                self.at_line_start = false;
                self.heading_marker = false;
            }
            '`' => {
                self.toggle_code(out);
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

    /// Emit the dim footer below a closed code block, carrying its `/clip`
    /// index. No trailing newline — the close fence's own newline ends the line.
    fn emit_block_footer(&mut self, out: &mut String) {
        let lang = self.lang_buf.trim();
        let label = if lang.is_empty() {
            "── ⧉ copy ──".to_string()
        } else {
            format!("── {lang} · ⧉ copy ──")
        };
        out.push_str(&format!("{DIM_ON}{label}{DIM_OFF}"));
    }

    fn newline(&mut self, out: &mut String) {
        if self.heading {
            out.push_str(BOLD_OFF);
            self.heading = false;
        }
        out.push('\n');
        self.at_line_start = true;
        self.line_only_backticks = true;
        self.heading_marker = false;
        self.suppress_eol = false;
    }

    /// Emit 1–2 leading backticks that turned out not to be a fence.
    fn resolve_pending_backticks(&mut self, out: &mut String) {
        let n = std::mem::take(&mut self.pending_backticks);
        for _ in 0..n {
            if self.in_fence {
                out.push('`');
            } else {
                self.toggle_code(out);
            }
        }
    }

    /// Flush trailing state at end of the response and clear active styling.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        self.resolve_pending_backticks(&mut out);
        if self.pending_star {
            out.push('*');
            self.pending_star = false;
        }
        if self.bold || self.italic || self.code || self.heading || self.in_fence {
            out.push_str(RESET);
            self.bold = false;
            self.italic = false;
            self.code = false;
            self.heading = false;
            self.in_fence = false;
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

/// A fenced code block extracted from a response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeBlock {
    pub lang: String,
    pub body: String,
}

/// Extract fenced (```` ``` ````) code blocks from raw markdown. An unterminated
/// final fence still yields its accumulated body.
pub fn code_blocks(text: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut in_fence = false;
    let mut lang = String::new();
    let mut body: Vec<&str> = Vec::new();
    for line in text.split('\n') {
        if line.trim_start().starts_with("```") {
            if in_fence {
                blocks.push(CodeBlock {
                    lang: std::mem::take(&mut lang),
                    body: body.join("\n"),
                });
                body.clear();
                in_fence = false;
            } else {
                in_fence = true;
                lang = line.trim_start().trim_start_matches('`').trim().to_string();
            }
        } else if in_fence {
            body.push(line);
        }
    }
    if in_fence && !body.is_empty() {
        blocks.push(CodeBlock {
            lang,
            body: body.join("\n"),
        });
    }
    blocks
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
        assert_eq!(render("*hi* x"), format!("{ITALIC_ON}hi{ITALIC_OFF} x"));
    }

    #[test]
    fn inline_code() {
        assert_eq!(render("`x`"), format!("{CODE_ON}x{CODE_OFF}"));
    }

    #[test]
    fn inline_code_at_line_start() {
        // A single leading backtick is inline code, not a fence.
        assert_eq!(render("`y`"), format!("{CODE_ON}y{CODE_OFF}"));
    }

    #[test]
    fn heading_is_bolded_and_marker_hidden() {
        assert_eq!(
            render("# Title\nbody"),
            format!("{BOLD_ON}Title{BOLD_OFF}\nbody")
        );
    }

    #[test]
    fn plain_text_and_dashes_unchanged() {
        assert_eq!(render("s-t-a-w, plain"), "s-t-a-w, plain");
    }

    #[test]
    fn fenced_block_shows_copy_footer_and_hides_fences() {
        let out = render("```rust\nlet x = 1;\n```\n");
        // The ``` markers are gone; a copy footer with the lang appears.
        assert!(!out.contains("```"));
        assert!(out.contains("⧉ copy"));
        assert!(out.contains("rust")); // lang shown in the footer
        assert!(out.contains("let x = 1;")); // body rendered
    }

    #[test]
    fn two_blocks_each_get_a_copy_footer() {
        let out = render("```\na\n```\ntext\n```\nb\n```\n");
        assert_eq!(out.matches("⧉ copy").count(), 2);
    }

    #[test]
    fn fence_disables_inline_parsing_inside() {
        // `*` and `_` inside a code fence stay literal.
        let out = render("```\na * b ** c\n```\n");
        assert!(out.contains("a * b ** c"));
    }

    #[test]
    fn trailing_lone_star_is_literal() {
        assert_eq!(render("5 *"), "5 *");
    }

    #[test]
    fn unclosed_bold_is_reset_at_end() {
        let out = render("**oops");
        assert!(out.starts_with(BOLD_ON));
        assert!(out.ends_with(RESET));
    }

    #[test]
    fn code_blocks_extracts_fences_with_lang() {
        let text = "intro\n```rust\nfn main() {}\n```\nmid\n```\nplain\n```\n";
        let blocks = code_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].lang, "rust");
        assert_eq!(blocks[0].body, "fn main() {}");
        assert_eq!(blocks[1].lang, "");
        assert_eq!(blocks[1].body, "plain");
    }

    #[test]
    fn code_blocks_handles_unterminated_fence() {
        let blocks = code_blocks("```py\nx = 1\nstill going");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].body, "x = 1\nstill going");
    }

    #[test]
    fn code_blocks_none_when_no_fences() {
        assert!(code_blocks("just prose, `inline` only").is_empty());
    }
}
