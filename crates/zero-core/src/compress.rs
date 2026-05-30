//! Content-aware, **recoverable** tool-output compression.
//!
//! The lesson from every coding-agent post-mortem (Claude Code, OpenHands, Manus,
//! the JetBrains summarization study) is the same: silent lossy truncation is the
//! #1 cause of agent self-sabotage — it drops the error in the middle of a log or
//! the match deep in a grep, with no way to get it back. So the rule here is:
//!
//! > **Cap = keep the high-signal representation + offload the rest to a file +
//! > emit a loud, re-fetchable marker. Never silently delete bytes.**
//!
//! [`compress`] is pure (no I/O) so it's fully unit- and replay-testable: it takes
//! the raw output and an optional `artifact` path (where the caller spilled the
//! full bytes) and returns the model-facing view. [`spill`] does the filesystem
//! side. The executor wires them: spill first, then compress with the path so the
//! marker can point the model back at the full output via `read_file`.
//!
//! Which bytes are "high signal" depends on the output's shape (a grep's value is
//! its `file:line` refs, a build log's is the first error + tail), so we detect
//! the shape and route to a per-shape compressor. Everything is deterministic and
//! std-only.

use std::path::{Path, PathBuf};

/// The kind of tool output, which decides how to compress it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputShape {
    /// `grep`/`rg` results: `path:line:body` — keep refs, drop bodies.
    Grep,
    /// A unified diff: keep `--stat` + hunk headers, drop unchanged context.
    Diff,
    /// Build/test logs: keep the first error, severity lines, and the tail.
    Log,
    /// Directory listings: depth + fan-out caps.
    Dir,
    /// JSON: minify (v1); structural summarize (v2).
    Json,
    /// Binary blob: never inline; just report size.
    Binary,
    /// Unknown line/stream output: head + tail donut.
    Generic,
}

impl OutputShape {
    pub fn label(self) -> &'static str {
        match self {
            OutputShape::Grep => "grep",
            OutputShape::Diff => "diff",
            OutputShape::Log => "log",
            OutputShape::Dir => "dir",
            OutputShape::Json => "json",
            OutputShape::Binary => "binary",
            OutputShape::Generic => "generic",
        }
    }
}

/// The result of compressing one tool output.
#[derive(Debug, Clone)]
pub struct Compressed {
    /// The model-facing view (high-signal + marker).
    pub text: String,
    /// Detected shape.
    pub shape: OutputShape,
    /// Bytes of the original output.
    pub raw_bytes: usize,
    /// Bytes of the view actually fed to the model.
    pub kept_bytes: usize,
}

impl Compressed {
    /// Bytes saved (>=0).
    pub fn saved(&self) -> usize {
        self.raw_bytes.saturating_sub(self.kept_bytes)
    }
}

/// git's heuristic: a NUL byte in the first 8000 bytes ⇒ binary.
pub fn is_binary(raw: &[u8]) -> bool {
    raw.iter().take(8000).any(|&b| b == 0)
}

/// Detect the output shape from the command that produced it (primary signal —
/// we ran it, we know) and a sniff of the first bytes (confirms / handles
/// unknown commands).
pub fn detect(cmd: &str, raw: &[u8]) -> OutputShape {
    if is_binary(raw) {
        return OutputShape::Binary;
    }
    let c = cmd.trim_start();
    // Command-verb signal first.
    let verb_shape = if starts_any(c, &["grep", "rg ", "rg\t", "ag ", "git grep"]) {
        Some(OutputShape::Grep)
    } else if c.contains("diff") {
        Some(OutputShape::Diff)
    } else if starts_any(
        c,
        &[
            "cargo",
            "make",
            "npm test",
            "pnpm test",
            "yarn test",
            "pytest",
            "go test",
            "go build",
        ],
    ) || c.contains("test")
    {
        Some(OutputShape::Log)
    } else if starts_any(c, &["ls", "tree", "find"]) {
        Some(OutputShape::Dir)
    } else {
        None
    };
    if let Some(s) = verb_shape {
        return s;
    }
    // First-bytes sniff for unknown commands.
    let head: String = String::from_utf8_lossy(&raw[..raw.len().min(SNIFF)]).into_owned();
    let first = head.trim_start().chars().next();
    if head.starts_with("diff --git") {
        return OutputShape::Diff;
    }
    if matches!(first, Some('{') | Some('[')) {
        return OutputShape::Json;
    }
    // grep shape: a majority of the first lines look like `path:NN:...`.
    let lines: Vec<&str> = head.lines().take(20).collect();
    if !lines.is_empty() {
        let grepish = lines.iter().filter(|l| looks_like_grep_line(l)).count();
        if grepish * 10 >= lines.len() * 7 {
            return OutputShape::Grep;
        }
    }
    // log shape: severity keyword in the head sniff.
    if has_severity(&head) {
        return OutputShape::Log;
    }
    // Tier 2 — deep-needle escalation. The head sniff only saw the first ~2 KB;
    // a log-shaped output from an unrecognized verb (`./build.sh`, a custom
    // runner) can carry its first error far past that. Only worth scanning the
    // remainder when the output is big enough that the donut would actually drop
    // the middle. Overlap the boundary by the longest token so a severity word
    // split across the 2 KB cut isn't missed by both passes.
    if raw.len() > SNIFF {
        let from = SNIFF.saturating_sub(MAX_SEVERITY_LEN);
        if contains_severity_bytes(&raw[from..]) {
            return OutputShape::Log;
        }
    }
    OutputShape::Generic
}

/// Bytes of the head sniff window used by [`detect`].
const SNIFF: usize = 2048;
/// Longest [`SEVERITY`] token ("assertion failed") — the boundary overlap.
const MAX_SEVERITY_LEN: usize = 16;

fn starts_any(s: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| s.starts_with(p))
}

/// `path:123: body` or `path:123:45: body`.
fn looks_like_grep_line(l: &str) -> bool {
    let mut it = l.splitn(3, ':');
    let (Some(p), Some(n)) = (it.next(), it.next()) else {
        return false;
    };
    !p.is_empty() && !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
}

const SEVERITY: &[&str] = &[
    "error:",
    "error[",
    "warning:",
    "FAILED",
    "FAIL ",
    "panicked",
    "panic:",
    "Exception",
    "Traceback",
    "assertion failed",
    "AssertionError",
];

fn has_severity(s: &str) -> bool {
    SEVERITY.iter().any(|k| s.contains(k))
}

/// True if any severity token occurs anywhere in `raw`. Byte-level, no alloc,
/// single pass with a first-byte gate so the common no-match case stays ~O(n).
/// Backs the deep-needle escalation in [`detect`]: a log-shaped output from an
/// unrecognized verb can carry its first `error:` far past the 2 KB head sniff,
/// and the generic donut would drop exactly that line. Worst case O(n·k) for k
/// short needles, sub-ms even on a multi-MB log.
fn contains_severity_bytes(raw: &[u8]) -> bool {
    for i in 0..raw.len() {
        let b = raw[i];
        for k in SEVERITY {
            let needle = k.as_bytes();
            if needle[0] == b && raw[i..].starts_with(needle) {
                return true;
            }
        }
    }
    false
}

fn is_severity_line(l: &str) -> bool {
    SEVERITY.iter().any(|k| l.contains(k))
}

/// Compress `raw` to roughly `budget` bytes, shape-aware. `cmd` is the command
/// that produced it (drives shape detection). `artifact`, if set, is the path
/// where the full output was spilled — embedded in the marker so the model can
/// re-fetch the dropped content. Pure: does no I/O.
///
/// Output already within budget is returned verbatim (no marker).
pub fn compress(cmd: &str, raw: &str, budget: usize, artifact: Option<&Path>) -> Compressed {
    let raw_bytes = raw.len();
    let shape = detect(cmd, raw.as_bytes());

    // Binary is special: never inline regardless of budget.
    if shape == OutputShape::Binary {
        let text = format!(
            "[binary output, {raw_bytes} bytes]{}",
            artifact_hint(artifact)
        );
        let kept = text.len();
        return Compressed {
            text,
            shape,
            raw_bytes,
            kept_bytes: kept,
        };
    }
    if raw_bytes <= budget {
        return Compressed {
            text: raw.to_string(),
            shape,
            raw_bytes,
            kept_bytes: raw_bytes,
        };
    }

    let text = match shape {
        OutputShape::Log => compress_log(raw, budget, artifact),
        OutputShape::Grep => compress_grep(raw, budget, artifact),
        // Diff / Dir / Json get the generic donut in v1; dedicated handlers are v2.
        _ => compress_generic(raw, budget, artifact),
    };
    let kept_bytes = text.len();
    Compressed {
        text,
        shape,
        raw_bytes,
        kept_bytes,
    }
}

/// "  (full output: <path>)" or "" when no artifact.
fn artifact_hint(artifact: Option<&Path>) -> String {
    match artifact {
        Some(p) => format!("  (full output: {})", p.display()),
        None => String::new(),
    }
}

/// Build/test log: keep a small head (toolchain banner), every severity line
/// (the *first error* is the cause — never drop it), and the tail (summary/exit),
/// collapsing the low-signal middle. Recoverable via the artifact.
fn compress_log(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    let lines: Vec<&str> = raw.lines().collect();
    let n = lines.len();
    const HEAD: usize = 8;
    const TAIL: usize = 30;

    let mut keep_head: Vec<usize> = (0..HEAD.min(n)).collect();
    let mut keep_tail: Vec<usize> = (n.saturating_sub(TAIL)..n).collect();
    // Severity lines anywhere in the middle (cap to avoid a log that's all errors),
    // PLUS a few trailing context lines per error — compilers put the location
    // (`--> file:line`) and the offending code snippet on the lines *after* the
    // `error:` line, so keeping only the error line loses where it is.
    const CONTEXT_AFTER: usize = 3;
    let mid = HEAD..n.saturating_sub(TAIL);
    let mut sev: Vec<usize> = Vec::new();
    let mut errs = 0;
    for i in mid.clone() {
        if is_severity_line(lines[i]) && errs < 60 {
            errs += 1;
            for j in i..=(i + CONTEXT_AFTER).min(n.saturating_sub(TAIL).saturating_sub(1)) {
                sev.push(j);
            }
        }
    }

    let mut keep = Vec::new();
    keep.append(&mut keep_head);
    keep.append(&mut sev);
    keep.append(&mut keep_tail);
    keep.sort_unstable();
    keep.dedup();

    let kept_lines = keep.len();
    let mut out = String::new();
    let mut prev: Option<usize> = None;
    for &i in &keep {
        if let Some(p) = prev {
            if i > p + 1 {
                out.push_str(&format!("… [{} lines elided] …\n", i - p - 1));
            }
        }
        out.push_str(lines[i]);
        out.push('\n');
        prev = Some(i);
    }
    let _ = budget; // log compressor is line-driven, not byte-driven
    out.push_str(&format!(
        "[log compressed: {kept_lines}/{n} lines kept — first error + severity + tail.{}]\n",
        artifact_hint(artifact)
    ));
    out
}

/// grep/search: keep every `path:line` ref (the high-signal, re-fetchable part),
/// truncate each match body, and when there are many matches collapse to a
/// per-file count. Recoverable via the artifact.
fn compress_grep(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    const BODY_CAP: usize = 120;
    // Branch on MATCH COUNT, not byte size: the refs are the high-signal part we
    // always want to keep, so up to MAX_REFS matches keep every `path:line` ref
    // (just trimming the bulky bodies). Only beyond that do we collapse to
    // per-file counts. (Byte-branching here wrongly collapsed a handful of
    // long-bodied matches whose refs we could easily have kept.)
    const MAX_REFS: usize = 40;
    let lines: Vec<&str> = raw.lines().collect();
    let total = lines.len();

    // Few matches: keep all refs, trim bodies.
    if total <= MAX_REFS {
        let mut out = String::new();
        for l in &lines {
            out.push_str(&trim_grep_body(l, BODY_CAP));
            out.push('\n');
        }
        out.push_str(&format!(
            "[grep compressed: {total} matches, bodies trimmed.{}]\n",
            artifact_hint(artifact)
        ));
        let _ = budget;
        return out;
    }

    // Many matches: collapse to per-file counts (grep -c form).
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, (usize, Vec<String>)> =
        std::collections::HashMap::new();
    for l in &lines {
        let path = l.split(':').next().unwrap_or(l).to_string();
        let e = counts.entry(path.clone()).or_insert_with(|| {
            order.push(path.clone());
            (0, Vec::new())
        });
        e.0 += 1;
        if let Some(num) = l.split(':').nth(1) {
            if e.1.len() < 6 {
                e.1.push(num.to_string());
            }
        }
    }
    let nfiles = order.len();
    let mut out = String::new();
    for path in &order {
        let (c, nums) = &counts[path];
        let more = if *c > nums.len() {
            format!(",…+{}", c - nums.len())
        } else {
            String::new()
        };
        out.push_str(&format!(
            "{path}: {c} matches (lines {}{more})\n",
            nums.join(",")
        ));
    }
    out.push_str(&format!(
        "[grep compressed: {total} matches across {nfiles} files — counts only. \
         Re-run with -l for filenames, or scope the path.{}]\n",
        artifact_hint(artifact)
    ));
    out
}

fn trim_grep_body(line: &str, cap: usize) -> String {
    // keep `path:line:` (and optional `:col:`) prefix, trim the body
    let mut it = line.splitn(3, ':');
    match (it.next(), it.next(), it.next()) {
        (Some(p), Some(n), Some(body)) if n.bytes().all(|b| b.is_ascii_digit()) => {
            let body = body.trim_start();
            if body.chars().count() > cap {
                let cut: String = body.chars().take(cap).collect();
                format!("{p}:{n}: {cut}…")
            } else {
                format!("{p}:{n}: {body}")
            }
        }
        _ => line.to_string(),
    }
}

/// Generic / unknown line output: head + tail "donut", middle elided, on char
/// boundaries. Recoverable via the artifact. Tail-weighted slightly (errors and
/// exit status live at the end).
fn compress_generic(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    let hint = artifact_hint(artifact);
    // Marker overhead reservation.
    let marker_room = 80 + hint.len();
    if budget <= marker_room {
        return format!("[output {} bytes elided.{hint}]\n", raw.len());
    }
    let body = budget - marker_room;
    let head_budget = body * 55 / 100;
    let tail_budget = body - head_budget;

    let head_end = floor_boundary(raw, head_budget);
    let tail_start = ceil_boundary(raw, raw.len() - tail_budget);
    if tail_start <= head_end {
        let end = floor_boundary(raw, body);
        return format!(
            "{}\n… [{} bytes elided.{hint}] …\n",
            &raw[..end],
            raw.len() - end
        );
    }
    format!(
        "{}\n… [{} bytes elided.{hint}] …\n{}",
        &raw[..head_end],
        tail_start - head_end,
        &raw[tail_start..]
    )
}

fn floor_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Spill the full output to a per-session artifact file and return its path, so a
/// compressed view can point the model back at the complete bytes. `dir` is the
/// session's artifact directory (created if missing); `id` a short unique tag.
/// Best-effort: on any I/O error returns `None` (compression still works, just
/// without a re-fetch path).
pub fn spill(dir: &Path, id: &str, raw: &[u8]) -> Option<PathBuf> {
    std::fs::create_dir_all(dir).ok()?;
    let path = dir.join(format!("out-{id}.txt"));
    std::fs::write(&path, raw).ok()?;
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn art() -> Option<&'static Path> {
        Some(Path::new("/tmp/zero/out-ab.txt"))
    }

    #[test]
    fn is_binary_detects_nul() {
        assert!(is_binary(b"abc\0def"));
        assert!(!is_binary(b"plain text\nwith lines"));
    }

    #[test]
    fn detect_uses_command_verb() {
        assert_eq!(
            detect("grep -rn foo .", b"src/a.rs:1: foo"),
            OutputShape::Grep
        );
        assert_eq!(detect("cargo build", b"   Compiling x"), OutputShape::Log);
        assert_eq!(detect("git diff", b"diff --git a b"), OutputShape::Diff);
        assert_eq!(detect("ls -R", b"a\nb\n"), OutputShape::Dir);
    }

    #[test]
    fn detect_sniffs_unknown_commands() {
        assert_eq!(
            detect("./mytool", b"diff --git a/x b/x\n"),
            OutputShape::Diff
        );
        assert_eq!(detect("./mytool", b"{\"a\":1}"), OutputShape::Json);
        assert_eq!(
            detect("./mytool", b"src/a.rs:12: hit\nsrc/b.rs:4: hit\n"),
            OutputShape::Grep
        );
        assert_eq!(detect("./mytool", b"boom error: nope"), OutputShape::Log);
        assert_eq!(
            detect("./mytool", b"just some prose here"),
            OutputShape::Generic
        );
    }

    #[test]
    fn detect_finds_severity_deep_past_the_head_sniff() {
        // GAP 1: a log-shaped output from an UNRECOGNIZED verb whose first error
        // lands far past the 2 KB head sniff must still classify as Log — else
        // the generic donut drops the one line that matters. Regression test for
        // the live `./build.sh`-style miss.
        let mut raw = vec![b'x'; 5000];
        raw.extend_from_slice(b"\nerror: deep failure at the end\n");
        assert_eq!(detect("./build.sh", &raw), OutputShape::Log);
    }

    #[test]
    fn deep_severity_scan_skipped_when_output_is_small() {
        // No escalation for sub-sniff output: a short shapeless blob with no
        // head needle stays Generic (the donut keeps it whole anyway).
        assert_eq!(
            detect("./build.sh", b"all good, nothing here"),
            OutputShape::Generic
        );
    }

    #[test]
    fn severity_token_straddling_the_sniff_boundary_is_still_found() {
        // The escalation overlaps the boundary by MAX_SEVERITY_LEN, so a token
        // cut in half at byte 2048 is caught. Place "error:" to span the cut.
        let mut raw = vec![b'x'; SNIFF - 3];
        raw.extend_from_slice(b"error: split across the 2KB boundary\n");
        raw.extend_from_slice(&vec![b'y'; 1000]);
        assert_eq!(detect("./build.sh", &raw), OutputShape::Log);
    }

    #[test]
    fn max_severity_len_covers_the_longest_token() {
        // Invariant the boundary-overlap relies on.
        let longest = SEVERITY.iter().map(|s| s.len()).max().unwrap();
        assert!(
            MAX_SEVERITY_LEN >= longest,
            "overlap {MAX_SEVERITY_LEN} < longest token {longest}"
        );
    }

    #[test]
    fn small_output_passes_through_untouched() {
        let c = compress("echo hi", "hello world", 4096, art());
        assert_eq!(c.text, "hello world");
        assert_eq!(c.saved(), 0);
    }

    #[test]
    fn binary_is_never_inlined() {
        let raw = String::from_utf8_lossy(&[0u8; 5000]).into_owned();
        let c = compress("cat x.bin", &raw, 4096, art());
        assert_eq!(c.shape, OutputShape::Binary);
        assert!(c.text.contains("binary output"));
        assert!(c.text.contains("/tmp/zero/out-ab.txt"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn log_keeps_first_error_and_tail_and_is_recoverable() {
        // Build log: banner, a wall of progress spam, an error in the MIDDLE,
        // more spam, then a summary tail. Naive head/tail would drop the error.
        let mut s = String::new();
        for i in 0..5 {
            s.push_str(&format!("   Compiling crate{i} v0.1\n"));
        }
        for i in 0..400 {
            s.push_str(&format!("   progress line {i}\n"));
        }
        s.push_str("error[E0308]: mismatched types in the middle\n");
        for i in 0..400 {
            s.push_str(&format!("   more progress {i}\n"));
        }
        s.push_str("test result: FAILED. 1 failed\n");
        let c = compress("cargo test", &s, 2000, art());
        assert_eq!(c.shape, OutputShape::Log);
        // The mid-log error survives — the whole point.
        assert!(c
            .text
            .contains("error[E0308]: mismatched types in the middle"));
        // The tail summary survives.
        assert!(c.text.contains("test result: FAILED"));
        // Big reduction.
        assert!(
            c.kept_bytes < c.raw_bytes / 3,
            "kept {} raw {}",
            c.kept_bytes,
            c.raw_bytes
        );
        // Loud + recoverable.
        assert!(c.text.contains("log compressed"));
        assert!(c.text.contains("/tmp/zero/out-ab.txt"));
    }

    #[test]
    fn grep_many_matches_collapses_to_counts_keeping_refs() {
        // 150 matches across 3 files — the Log-B offender shape.
        let mut s = String::new();
        for i in 0..50 {
            s.push_str(&format!(
                "src/a.rs:{}: some matching body text here\n",
                i + 1
            ));
        }
        for i in 0..60 {
            s.push_str(&format!("src/b.rs:{}: another matching body line\n", i + 1));
        }
        for i in 0..40 {
            s.push_str(&format!("docs/c.md:{}: doc match body\n", i + 1));
        }
        let c = compress("grep -rn match .", &s, 400, art());
        assert_eq!(c.shape, OutputShape::Grep);
        // File refs preserved (high signal), counts shown.
        assert!(c.text.contains("src/a.rs: 50 matches"));
        assert!(c.text.contains("src/b.rs: 60 matches"));
        assert!(c.text.contains("docs/c.md: 40 matches"));
        // Bodies dropped → big reduction.
        assert!(c.kept_bytes < c.raw_bytes / 3);
        assert!(c.text.contains("/tmp/zero/out-ab.txt"));
    }

    #[test]
    fn grep_few_matches_keeps_refs_trims_bodies() {
        let mut s = String::new();
        for i in 0..10 {
            s.push_str(&format!("src/a.rs:{}: {}\n", i + 1, "x".repeat(500)));
        }
        let c = compress("grep -rn x .", &s, 300, art());
        assert_eq!(c.shape, OutputShape::Grep);
        // All 10 refs kept.
        for i in 0..10 {
            assert!(c.text.contains(&format!("src/a.rs:{}:", i + 1)));
        }
        // Bodies trimmed with ellipsis.
        assert!(c.text.contains('…'));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn generic_donut_keeps_head_and_tail_char_safe() {
        let s = "é".repeat(5000); // 2-byte chars: must not split a codepoint
        let c = compress("./unknown", &s, 400, art());
        assert_eq!(c.shape, OutputShape::Generic);
        assert!(c.text.starts_with('é'));
        assert!(c.text.contains("elided"));
        assert!(c.text.contains("/tmp/zero/out-ab.txt"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn compress_is_deterministic() {
        let s = format!("grep body\n{}", "src/x.rs:1: hit\n".repeat(300));
        let a = compress("grep x .", &s, 500, art());
        let b = compress("grep x .", &s, 500, art());
        assert_eq!(a.text, b.text);
    }

    #[test]
    fn spill_writes_full_bytes_and_roundtrips() {
        let dir =
            std::env::temp_dir().join(format!("zero-spill-{}-{}", std::process::id(), line!()));
        let raw = b"the full output\nevery byte\n";
        let p = spill(&dir, "deadbeef", raw).expect("spill ok");
        assert_eq!(std::fs::read(&p).unwrap(), raw);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn marker_without_artifact_is_clean() {
        let s = "x".repeat(5000);
        let c = compress("./unknown", &s, 300, None);
        assert!(!c.text.contains("full output:"));
        assert!(c.text.contains("elided"));
    }

    #[test]
    fn every_shape_has_a_label() {
        for (shape, want) in [
            (OutputShape::Grep, "grep"),
            (OutputShape::Diff, "diff"),
            (OutputShape::Log, "log"),
            (OutputShape::Dir, "dir"),
            (OutputShape::Json, "json"),
            (OutputShape::Binary, "binary"),
            (OutputShape::Generic, "generic"),
        ] {
            assert_eq!(shape.label(), want);
        }
    }

    #[test]
    fn diff_dir_json_use_generic_donut_in_v1() {
        // v1: these shapes are detected but routed through the generic donut.
        let big_json = format!("[{}]", "\"x\",".repeat(3000));
        let cj = compress("./tool", &big_json, 400, art());
        assert_eq!(cj.shape, OutputShape::Json);
        assert!(cj.text.contains("elided"));
        assert!(cj.kept_bytes < cj.raw_bytes);

        let big_diff = format!("diff --git a b\n{}", "+added line\n".repeat(2000));
        let cd = compress("git diff", &big_diff, 400, art());
        assert_eq!(cd.shape, OutputShape::Diff);
        assert!(cd.text.contains("elided"));

        let big_dir = "afile\n".repeat(3000);
        let cdir = compress("ls -R", &big_dir, 400, art());
        assert_eq!(cdir.shape, OutputShape::Dir);
        assert!(cdir.text.contains("elided"));
    }

    #[test]
    fn generic_budget_below_marker_returns_just_the_marker() {
        let s = "z".repeat(5000);
        // budget smaller than the marker reservation → marker-only path.
        let c = compress("./unknown", &s, 10, art());
        assert!(c.text.contains("elided"));
        assert!(c.text.contains("5000 bytes"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn generic_tiny_budget_falls_back_to_head_only() {
        // Budget big enough for the marker but not head+tail → head-only branch.
        let s = "abcdefghij".repeat(500);
        let c = compress("./unknown", &s, 120, art());
        assert!(c.text.starts_with('a'));
        assert!(c.text.contains("elided"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn generic_donut_keeps_both_head_and_tail_when_budget_allows() {
        // ASCII (clean boundaries), budget large enough that head and tail don't
        // overlap → exercises the both-kept branch (not the head-only fallback).
        let s = "a".repeat(5000);
        let c = compress("./x", &s, 2000, art());
        assert_eq!(c.shape, OutputShape::Generic);
        assert!(c.text.starts_with("aaaa"));
        assert!(c.text.ends_with("aaaa\n") || c.text.ends_with("aaaa"));
        assert!(c.text.contains("elided"));
        // Both ends present means the elision marker sits in the middle.
        let mid = c.text.find("elided").unwrap();
        assert!(mid > 100 && mid < c.text.len() - 100);
    }

    #[test]
    fn grep_few_matches_short_bodies_are_not_ellipsized() {
        // ≤40 matches with short bodies → kept verbatim (no trailing …).
        let mut s = String::new();
        for i in 0..5 {
            s.push_str(&format!("src/a.rs:{}: short body\n", i + 1));
        }
        // Force the compressor (over budget) without crossing MAX_REFS.
        let c = compress("grep x .", &s, 10, art());
        assert_eq!(c.shape, OutputShape::Grep);
        assert!(c.text.contains("src/a.rs:1: short body"));
        assert!(!c.text.contains('…')); // short bodies kept whole
    }

    #[test]
    fn grep_counts_no_overflow_marker_when_six_or_fewer_per_file() {
        // >40 total (count branch) but a file with ≤6 matches shows its lines
        // with no ",…+N" overflow suffix.
        let mut s = String::new();
        for i in 0..41 {
            s.push_str(&format!("big.rs:{}: body\n", i + 1));
        }
        for i in 0..3 {
            s.push_str(&format!("small.rs:{}: body\n", i + 1));
        }
        let c = compress("grep x .", &s, 200, art());
        assert!(c.text.contains("small.rs: 3 matches (lines 1,2,3)"));
        assert!(!c.text.contains("small.rs: 3 matches (lines 1,2,3,…"));
    }

    #[test]
    fn grep_counts_show_overflow_marker_past_six_lines() {
        // A file with >6 matches exercises the ",…+N" overflow in the count form.
        let mut s = String::new();
        for i in 0..50 {
            s.push_str(&format!("only.rs:{}: body\n", i + 1));
        }
        // pad past MAX_REFS with a second file
        for i in 0..50 {
            s.push_str(&format!("two.rs:{}: body\n", i + 1));
        }
        let c = compress("grep -rn x .", &s, 200, art());
        assert_eq!(c.shape, OutputShape::Grep);
        assert!(c.text.contains("only.rs: 50 matches"));
        assert!(c.text.contains(",…+44")); // 6 shown, 44 more
    }
}
