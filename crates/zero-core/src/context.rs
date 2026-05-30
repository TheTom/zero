//! Context-efficiency primitives for the agentic tool loop.
//!
//! Local models have small windows (8–32K), and measurement of real agentic
//! transcripts shows ~95% of a turn's context is raw tool I/O — not reasoning —
//! with a tiny long tail of giant tool results carrying most of the bytes. So
//! bounding tool output is the single biggest lever on how much useful work fits
//! in a window. These helpers are pure + std-only so every saving is unit-proven.
//!
//! Design rule: **cap, never lose.** A capped result always keeps the
//! highest-signal parts (head + tail) and carries an explicit marker telling the
//! model how to retrieve the dropped span — nothing is destroyed, only deferred.

/// Default per-result context cap (bytes) ≈ ~1K tokens — generous for a 32K
/// window, safe for an 8K one. Tuned from transcript analysis where a 2.6% long
/// tail of results carried ~70% of all tool-output bytes.
pub const DEFAULT_MAX_TOOL_OUTPUT: usize = 4096;
/// Default cumulative per-turn tool-output budget (bytes) ≈ 6K tokens.
pub const DEFAULT_MAX_TURN_OUTPUT: usize = 24_000;
/// Minimum bytes any single result keeps even when the turn budget is spent
/// (the "always keep at least one" rule — a starved result is useless).
pub const TURN_OUTPUT_FLOOR: usize = 256;

/// Estimate token count from text via the standard chars/4 heuristic. Exact
/// counts need the model's tokenizer; chars/4 is the industry approximation and
/// is all we need to *budget* (we only compare against a budget, never bill).
pub fn estimate_tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

/// Cap `text` to roughly `max_bytes`, keeping the **head and tail** (the ends
/// carry the most signal: a file's start, and where output was heading / the
/// error at the end) and collapsing the middle to a one-line marker. `hint` is a
/// short re-fetch instruction appended to the marker so the model knows how to
/// recover the dropped span (e.g. `"read_file with offset 120"`). Returns `text`
/// unchanged when already within budget.
///
/// Char-boundary safe. A `max_bytes` too small to hold the marker still returns
/// the marker (we never panic or slice mid-codepoint).
pub fn cap_middle(text: &str, max_bytes: usize, hint: &str) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let dropped_marker = |dropped: usize| -> String {
        if hint.is_empty() {
            format!("\n… [{dropped} bytes elided] …\n")
        } else {
            format!("\n… [{dropped} bytes elided — {hint}] …\n")
        }
    };
    // Reserve room for the marker; split the rest head/tail (~60/40 — the head of
    // a file/listing is usually more useful than the tail).
    let marker_len = dropped_marker(text.len()).len();
    if max_bytes <= marker_len {
        // No room for real content — just say how much was dropped.
        return dropped_marker(text.len());
    }
    let budget = max_bytes - marker_len;
    let head_budget = budget * 6 / 10;
    let tail_budget = budget - head_budget;

    let head_end = floor_boundary(text, head_budget);
    let tail_start = ceil_boundary(text, text.len() - tail_budget);
    // If head and tail would overlap (tiny budget), fall back to head-only.
    if tail_start <= head_end {
        let end = floor_boundary(text, budget);
        let dropped = text.len() - end;
        return format!("{}{}", &text[..end], dropped_marker(dropped));
    }
    let dropped = tail_start - head_end;
    format!(
        "{}{}{}",
        &text[..head_end],
        dropped_marker(dropped),
        &text[tail_start..]
    )
}

/// Largest char boundary `<= i`.
fn floor_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest char boundary `>= i`.
fn ceil_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Remembers which files a tool loop has already read this session, so a repeat
/// read of an unchanged file returns a tiny stub instead of re-injecting the
/// whole content. Measured transcripts show the same file read 5–7× — pure
/// duplication in the window. "Cap, don't lose": the content is already upstream
/// in the conversation, so the stub points back to it.
///
/// Change is detected by `(modified-time, length)` — cheap, no hashing, and the
/// loop always reflects the working tree (an edited file re-reads in full).
#[derive(Debug, Default)]
pub struct ReadCache {
    seen: std::collections::HashMap<std::path::PathBuf, FileStamp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    /// Modified time as whole seconds since the epoch (0 if unavailable).
    mtime: u64,
    len: u64,
}

impl ReadCache {
    pub fn new() -> Self {
        ReadCache::default()
    }

    /// If `path` was read before and is unchanged since, return a stub to feed
    /// back instead of the file contents. `None` means "read it" (new, changed,
    /// or un-stattable).
    pub fn check(&self, path: &std::path::Path) -> Option<String> {
        let now = stamp(path)?;
        match self.seen.get(path) {
            Some(prev) if *prev == now => Some(format!(
                "[unchanged since your earlier read of {} ({} bytes) — \
                 reuse that result, or read a line range for more]",
                path.display(),
                now.len
            )),
            _ => None,
        }
    }

    /// Record `path`'s current stamp after a successful read.
    pub fn record(&mut self, path: &std::path::Path) {
        if let Some(s) = stamp(path) {
            self.seen.insert(path.to_path_buf(), s);
        }
    }

    /// Forget a path (e.g. after a write/edit) so it re-reads in full next time.
    pub fn invalidate(&mut self, path: &std::path::Path) {
        self.seen.remove(path);
    }
}

/// Current `(mtime, len)` for a file, or `None` if it can't be stat'd.
fn stamp(path: &std::path::Path) -> Option<FileStamp> {
    let md = std::fs::metadata(path).ok()?;
    if !md.is_file() {
        return None;
    }
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(FileStamp {
        mtime,
        len: md.len(),
    })
}

/// Compact a successfully-applied write/edit tool call's arguments for history:
/// keep `path` but replace the bulky content fields (`content`, `old_string`,
/// `new_string`) with a short `[N bytes applied]` placeholder. Once a file is
/// written, echoing its full content back into the window every subsequent turn
/// is dead weight — measured as ~7% of transcript bytes living in tool-call args.
/// "Cap, don't lose": the file is on disk; the model can `read_file` to see it.
///
/// Returns the compacted JSON, or `None` when there's nothing large enough to
/// compact (idempotent: the short placeholder is never re-compacted).
pub fn compact_call_args(name: &str, arguments: &str) -> Option<String> {
    /// Don't bother compacting payloads smaller than this.
    const MIN: usize = 200;
    let fields: &[&str] = match name {
        "write_file" => &["content"],
        "edit_file" => &["old_string", "new_string"],
        _ => return None,
    };
    let crate::json::Value::Object(obj) = crate::json::Value::parse(arguments).ok()? else {
        return None;
    };
    let mut changed = false;
    let compacted: Vec<(String, crate::json::Value)> = obj
        .into_iter()
        .map(|(k, v)| {
            if fields.contains(&k.as_str()) {
                if let Some(s) = v.as_str() {
                    if s.len() >= MIN {
                        changed = true;
                        return (
                            k,
                            crate::json::Value::Str(format!(
                                "[{} bytes applied — read_file to view]",
                                s.len()
                            )),
                        );
                    }
                }
            }
            (k, v)
        })
        .collect();
    changed.then(|| crate::json::Value::Object(compacted).to_json())
}

/// Compact the bulky args of every SUCCESSFULLY-applied write/edit call in a
/// conversation, in place. A write whose tool result is present and not an
/// error/refusal has landed on disk, so its inline `content`/`old_string`/
/// `new_string` payload is dead weight that's otherwise re-sent every round.
/// Replaces each with a `[N bytes applied — read_file to view]` stub (the model
/// still knows the file exists). Returns `(bytes_before, bytes_after)` of the
/// args it compacted, for measured stats.
///
/// Safe to call EVERY round mid-turn (idempotent: the stub is never re-compacted),
/// which is the point — it stops a long agentic turn from re-sending a just-written
/// file's full body on each subsequent model call. Pure conversation surgery.
pub fn compact_applied_writes(conv: &mut crate::message::Conversation) -> (usize, usize) {
    use crate::message::Role;
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
                if let Some(args) = compact_call_args(&tc.name, &tc.arguments) {
                    before += tc.arguments.len();
                    after += args.len();
                    tc.arguments = args;
                }
            }
        }
    }
    (before, after)
}

/// Cumulative, **measured** (never estimated) record of how many bytes the
/// context levers saved this session — for an honest `/context` report. Each
/// counter is the difference between what a tool produced and what actually
/// reached the model's window, attributed to the lever responsible.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ContextStats {
    /// Raw bytes tool executions produced, before any capping (cache hits, which
    /// never run the tool, are not counted here).
    pub raw_bytes: usize,
    /// Bytes actually fed back to the model after capping.
    pub kept_bytes: usize,
    /// Bytes dropped by per-result + per-turn capping.
    pub capped_saved: usize,
    /// Bytes avoided by the read cache (an unchanged re-read returns a stub
    /// instead of the would-be file content).
    pub cached_saved: usize,
    /// Bytes removed by compacting applied write/edit args out of history.
    pub compacted_saved: usize,
}

impl ContextStats {
    pub fn new() -> Self {
        ContextStats::default()
    }

    /// Record a tool result that ran: `raw` produced, `kept` fed back. The
    /// difference (if any) is attributed to capping.
    pub fn record_result(&mut self, raw: usize, kept: usize) {
        self.raw_bytes += raw;
        self.kept_bytes += kept;
        self.capped_saved += raw.saturating_sub(kept);
    }

    /// Record a read-cache hit: the tool did NOT run; `would_be` is the size of
    /// the content that would have been re-injected, `stub` the size returned.
    pub fn record_cache_hit(&mut self, would_be: usize, stub: usize) {
        self.cached_saved += would_be.saturating_sub(stub);
        self.kept_bytes += stub;
    }

    /// Record a history compaction of write/edit args: `before` → `after` bytes.
    pub fn record_compaction(&mut self, before: usize, after: usize) {
        self.compacted_saved += before.saturating_sub(after);
    }

    /// Total bytes saved across all levers.
    pub fn total_saved(&self) -> usize {
        self.capped_saved + self.cached_saved + self.compacted_saved
    }

    /// Reduction as a percentage of what *would* have entered the window without
    /// the levers (capped+cached baseline). 0 when nothing has flowed yet.
    pub fn reduction_pct(&self) -> u32 {
        let would_be = self.kept_bytes + self.capped_saved + self.cached_saved;
        if would_be == 0 {
            return 0;
        }
        ((self.capped_saved + self.cached_saved) as u64 * 100 / would_be as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_tokens_is_chars_over_four_rounded_up() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2); // 5/4 → 2
        assert_eq!(estimate_tokens("😀"), 1); // chars, not bytes
    }

    #[test]
    fn cap_leaves_short_text_untouched() {
        let s = "small output";
        assert_eq!(cap_middle(s, 1000, "hint"), s);
    }

    #[test]
    fn cap_keeps_head_and_tail_and_marks_the_drop() {
        let text: String = (0..2000).map(|i| (b'a' + (i % 26) as u8) as char).collect();
        let out = cap_middle(&text, 200, "read_file offset 100");
        assert!(out.len() < 400, "capped len {}", out.len());
        assert!(out.len() < text.len());
        assert!(out.starts_with(&text[..50]));
        assert!(out.ends_with(&text[text.len() - 50..]));
        assert!(out.contains("elided"));
        assert!(out.contains("read_file offset 100"));
    }

    #[test]
    fn cap_reduces_tokens_for_a_giant_result() {
        let big = "x".repeat(40_000);
        let before = estimate_tokens(&big);
        let after = estimate_tokens(&cap_middle(&big, 4096, "narrow your query"));
        assert!(after < before / 5, "before {before} after {after}");
    }

    #[test]
    fn cap_is_char_boundary_safe_on_multibyte() {
        let text = "é".repeat(5000); // 2 bytes each
        let out = cap_middle(&text, 300, "");
        assert!(out.len() < text.len());
        assert!(out.contains("elided"));
    }

    #[test]
    fn cap_with_budget_below_marker_returns_just_the_marker() {
        let text = "abcdefghijklmnopqrstuvwxyz".repeat(10);
        let out = cap_middle(&text, 5, "x");
        assert!(out.contains("elided"));
        assert!(out.len() < text.len());
    }

    #[test]
    fn cap_tiny_budget_falls_back_to_head_only() {
        let text = "abcdefghijklmnopqrstuvwxyz0123456789".repeat(20);
        let out = cap_middle(&text, 90, "more");
        assert!(out.contains("elided"));
        assert!(out.starts_with('a'));
        assert!(out.len() < text.len());
    }

    #[test]
    fn read_cache_skips_unchanged_and_reads_changed() {
        let dir = std::env::temp_dir().join(format!("zero-rc-{}-{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("a.txt");
        std::fs::write(&f, "hello").unwrap();

        let mut c = ReadCache::new();
        assert!(c.check(&f).is_none());
        c.record(&f);
        let stub = c.check(&f).expect("stub for unchanged file");
        assert!(stub.contains("unchanged"));
        assert!(stub.contains("5 bytes"));

        std::fs::write(&f, "hello world, now longer").unwrap();
        assert!(c.check(&f).is_none(), "changed file must re-read");

        c.record(&f);
        assert!(c.check(&f).is_some());
        c.invalidate(&f);
        assert!(c.check(&f).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_cache_ignores_missing_and_dirs() {
        let mut c = ReadCache::new();
        assert!(c
            .check(std::path::Path::new("/no/such/zero-file"))
            .is_none());
        c.record(std::path::Path::new("/no/such/zero-file")); // no-op
        assert!(c.check(std::env::temp_dir().as_path()).is_none()); // a dir
    }

    #[test]
    fn context_stats_measures_each_lever() {
        let mut s = ContextStats::new();
        // A 1000-byte result capped to 200 → 800 saved by capping.
        s.record_result(1000, 200);
        assert_eq!(s.raw_bytes, 1000);
        assert_eq!(s.kept_bytes, 200);
        assert_eq!(s.capped_saved, 800);
        // A cache hit: 5000 would-be content, 80-byte stub → 4920 saved.
        s.record_cache_hit(5000, 80);
        assert_eq!(s.cached_saved, 4920);
        assert_eq!(s.kept_bytes, 280);
        // A compaction: 5000-byte args → 40 → 4960 saved.
        s.record_compaction(5000, 40);
        assert_eq!(s.compacted_saved, 4960);
        assert_eq!(s.total_saved(), 800 + 4920 + 4960);
        // Reduction = (capped+cached) / (kept + capped + cached).
        // = 5720 / (280 + 5720) = 5720/6000 = 95%.
        assert_eq!(s.reduction_pct(), 95);
    }

    #[test]
    fn context_stats_empty_is_zero_not_a_panic() {
        let s = ContextStats::new();
        assert_eq!(s.total_saved(), 0);
        assert_eq!(s.reduction_pct(), 0); // no divide-by-zero
    }

    #[test]
    fn compact_call_args_drops_big_payloads_keeps_path() {
        let big = "z".repeat(5000);
        let args = format!(r#"{{"path":"f.rs","content":"{big}"}}"#);
        let out = compact_call_args("write_file", &args).expect("should compact");
        assert!(out.contains("f.rs"));
        assert!(out.contains("5000 bytes applied"));
        assert!(!out.contains(&big));
        assert!(out.len() < args.len() / 5);

        let ea = format!(r#"{{"path":"f.rs","old_string":"{big}","new_string":"{big}"}}"#);
        let eo = compact_call_args("edit_file", &ea).unwrap();
        assert!(!eo.contains(&big));
        assert!(eo.contains("f.rs"));
    }

    #[test]
    fn compact_call_args_is_idempotent_and_skips_small_and_other_tools() {
        assert!(compact_call_args("write_file", r#"{"path":"f","content":"hi"}"#).is_none());
        assert!(compact_call_args("read_file", r#"{"path":"f"}"#).is_none());
        let big = "z".repeat(5000);
        let args = format!(r#"{{"path":"f","content":"{big}"}}"#);
        let once = compact_call_args("write_file", &args).unwrap();
        assert!(compact_call_args("write_file", &once).is_none());
    }
}
