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
    /// A unified diff: keep file + hunk headers and every +/- change line, drop the
    /// unchanged context; fold near-identical change runs (bulk inserts).
    Diff,
    /// Build/test logs: keep the first error, severity lines, and the tail.
    Log,
    /// Directory listings: generic donut + repeat-fold (no dedicated handler yet).
    Dir,
    /// JSON: lossless whitespace minify (keeps it valid); donut the minified bytes
    /// only if still over budget.
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
        OutputShape::Json => compress_json(raw, budget, artifact),
        OutputShape::Diff => compress_diff(raw, budget, artifact),
        // Dir falls through to the generic donut, which now folds repeated runs.
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

/// A `read_file` result that's over budget. Unlike command output, a file read is
/// something the model deliberately asked to *see in full* — so we must NOT apply
/// the lossy content-shape compressors here. A source file that merely mentions
/// `error:` would be misdetected as a log and severity-filtered; a head+tail donut
/// of code drops the middle functions while *looking* complete — a real edit-time
/// hazard. Instead:
///   - if it's valid JSON, minify (lossless, full file, still valid), else
///   - keep a faithful line-aligned PREFIX + a loud marker that nudges a ranged
///     re-read (`offset`/`limit`) and names the artifact.
pub fn compress_file_read(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    // Lossless path: a JSON document comes back whole and valid, just minified.
    if !is_jsonl(raw) && matches!(detect("", raw.as_bytes()), OutputShape::Json) {
        if let Some(min) = json_minify(raw) {
            if min.len() <= budget {
                return min;
            }
        }
    }
    let total_lines = raw.lines().count();
    let hint = artifact_hint(artifact);
    let marker_room = 110 + hint.len();
    if budget <= marker_room {
        return format!(
            "[file is {total_lines} lines — too large to inline. \
             read_file with offset/limit to view a range.{hint}]\n"
        );
    }
    let body = budget - marker_room;
    let mut kept = String::new();
    let mut kept_lines = 0usize;
    for line in raw.lines() {
        if kept.len() + line.len() + 1 > body {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
        kept_lines += 1;
    }
    format!(
        "{kept}[file truncated: showing lines 1-{kept_lines} of {total_lines}. \
         read_file with offset/limit to see more.{hint}]\n"
    )
}

/// JSON: minify losslessly (strip insignificant whitespace — ~20-40% free and,
/// unlike a byte-donut, keeps the JSON *valid* for the model to parse). A donut on
/// JSON would splice head+tail into syntactic garbage, so we never donut raw JSON:
/// if the minified form still exceeds budget we donut the *minified* bytes (the
/// view is then truncated JSON, flagged + recoverable via the artifact).
fn compress_json(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    // JSONL / NDJSON (one JSON value per line) must stay line-oriented: minifying
    // would strip the inter-record newlines and concatenate the objects. Route it
    // to the generic (fold + donut) path, which preserves record boundaries.
    if is_jsonl(raw) {
        return compress_generic(raw, budget, artifact);
    }
    match json_minify(raw) {
        // Lossless win that fits — the ideal case, no marker needed.
        Some(min) if min.len() <= budget => min,
        // Minified but still over budget: donut the smaller minified bytes.
        Some(min) => {
            let inner = compress_generic(&min, budget, artifact);
            format!(
                "[json minified to {} bytes, then truncated]\n{inner}",
                min.len()
            )
        }
        // Not valid JSON (e.g. unterminated string) — treat as generic text.
        None => compress_generic(raw, budget, artifact),
    }
}

/// Heuristic: ≥2 non-blank lines and a majority begin a JSON value *at column 0*
/// ⇒ JSONL/NDJSON, not one pretty-printed document. The column-0 check is what
/// separates the two: JSONL records are unindented complete values, whereas a
/// pretty-printed array's inner `{…}` objects are INDENTED (so they don't count)
/// — without it, a pretty array-of-objects would be misread as JSONL.
fn is_jsonl(raw: &str) -> bool {
    let lines: Vec<&str> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(20)
        .collect();
    if lines.len() < 2 {
        return false;
    }
    let starters = lines
        .iter()
        .filter(|l| l.starts_with('{') || l.starts_with('['))
        .count();
    starters * 10 >= lines.len() * 8 // ≥80% of lines begin a JSON value at col 0
}

/// Strip whitespace that sits *outside* string literals. Byte-exact for the kept
/// content (preserves big integers, key order, escapes — everything a parse →
/// re-serialize round-trip would risk). Returns `None` on an unterminated string
/// (malformed input) so the caller can fall back to the generic donut.
fn json_minify(raw: &str) -> Option<String> {
    let mut out = String::with_capacity(raw.len());
    let mut in_str = false;
    let mut esc = false;
    for ch in raw.chars() {
        if in_str {
            out.push(ch);
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
        } else if ch == '"' {
            in_str = true;
            out.push(ch);
        } else if !ch.is_whitespace() {
            out.push(ch);
        }
        // insignificant whitespace outside strings is dropped
    }
    if in_str {
        return None; // unterminated string ⇒ not well-formed JSON
    }
    Some(out)
}

/// Unified diff: the signal is *what changed and where* — keep every file header
/// (`diff `, `index `, `--- `, `+++ `), every hunk header (`@@ … @@`), and every
/// added/removed line; drop the unchanged context lines (leading space) that pad
/// most diffs. Recoverable via the artifact. Line-driven like the log compressor.
fn compress_diff(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    let mut out = String::new();
    let mut elided = 0usize;
    let mut kept = 0usize;
    let flush = |out: &mut String, elided: &mut usize| {
        if *elided > 0 {
            out.push_str(&format!("… [{} context lines elided] …\n", *elided));
            *elided = 0;
        }
    };
    for line in raw.lines() {
        let b = line.as_bytes();
        let is_header = line.starts_with("diff ")
            || line.starts_with("index ")
            || line.starts_with("--- ")
            || line.starts_with("+++ ")
            || line.starts_with("@@");
        // A change line is +/- but NOT the +++/--- file headers (already caught).
        let is_change = matches!(b.first(), Some(b'+') | Some(b'-'));
        if is_header || is_change {
            flush(&mut out, &mut elided);
            out.push_str(line);
            out.push('\n');
            kept += 1;
        } else {
            elided += 1; // unchanged context (leading space) or blank
        }
    }
    flush(&mut out, &mut elided);
    // Fold runs of near-identical header/change lines (bulk inserts, generated
    // code, repeated edits) so a diff that's *all* changes can't blow the budget.
    // Distinct changes keep their own skeleton and never fold. Recoverable.
    let folded = fold_repeats(&out);
    let _ = budget; // diff compressor is structure-driven, not byte-driven
    format!(
        "{folded}[diff compressed: {kept} header/change lines kept, context dropped.{}]\n",
        artifact_hint(artifact)
    )
}

/// FNV-1a hash of a line's *skeleton*: every maximal run of ASCII digits collapses
/// to a single `#`, so `Compiling foo v0.1` and `Compiling bar v0.2` — or `1` and
/// `50000` — share a hash. Computed without allocating the skeleton string.
fn skeleton_hash(line: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let byte = if b[i].is_ascii_digit() {
            while i + 1 < b.len() && b[i + 1].is_ascii_digit() {
                i += 1;
            }
            b'#'
        } else {
            b[i]
        };
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    h
}

/// Collapse runs of adjacent same-skeleton lines (progress spam, numeric ranges,
/// repeated "Compiling x" lines) to first + last + a count, so the high-signal
/// boundaries survive while the redundant middle goes. Keeps both ends because the
/// answer is often at a boundary (e.g. the *last* number of a `seq`). Runs shorter
/// than REPEAT_MIN are left untouched. Lossless-of-signal; full bytes in artifact.
fn fold_repeats(raw: &str) -> String {
    const REPEAT_MIN: usize = 4;
    let lines: Vec<&str> = raw.lines().collect();
    let mut out = String::with_capacity(raw.len());
    let mut folded_any = false;
    let mut i = 0;
    while i < lines.len() {
        let h = skeleton_hash(lines[i]);
        let mut j = i + 1;
        while j < lines.len() && skeleton_hash(lines[j]) == h {
            j += 1;
        }
        let run = j - i;
        if run >= REPEAT_MIN {
            // first line, count of the dropped middle, last line — all verbatim.
            folded_any = true;
            out.push_str(lines[i]);
            out.push('\n');
            out.push_str(&format!("… {} similar lines …\n", run - 2));
            out.push_str(lines[j - 1]);
            out.push('\n');
        } else {
            for &l in &lines[i..j] {
                out.push_str(l);
                out.push('\n');
            }
        }
        i = j;
    }
    // No run folded → return the input untouched so byte counts and trailing-
    // newline shape are preserved exactly (the caller relies on len comparison).
    if !folded_any {
        return raw.to_string();
    }
    out
}

/// Generic / unknown line output: fold repeated runs first (kills uniform spam
/// like a 50k-line `seq` or progress bars), then a head + tail "donut" over what
/// remains, middle elided on char boundaries. Recoverable via the artifact.
/// Tail-weighted slightly (errors and exit status live at the end).
fn compress_generic(raw: &str, budget: usize, artifact: Option<&Path>) -> String {
    // Fold first: a uniform dump collapses to a few lines and may now fit budget
    // outright, turning a lossy donut into a folded view that keeps every run's
    // boundaries. Still byte-lossy (the middle of each run is gone), so when it
    // fits we append the artifact pointer rather than returning silently.
    let folded = fold_repeats(raw);
    if folded.len() < raw.len() && folded.len() <= budget {
        return format!(
            "{folded}[folded repeated runs.{}]\n",
            artifact_hint(artifact)
        );
    }
    let raw = folded.as_str();
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
    fn json_minify_is_lossless_outside_strings() {
        // Whitespace between tokens goes; whitespace INSIDE strings stays.
        let src = "{\n  \"a\" : 1,\n  \"b\": \"keep  me\"\n}";
        let min = json_minify(src).unwrap();
        assert_eq!(min, "{\"a\":1,\"b\":\"keep  me\"}");
    }

    #[test]
    fn json_minify_preserves_large_integers_exactly() {
        // A parse→f64→serialize round-trip would corrupt this; byte minify must not.
        let src = "{ \"id\": 1780157881126000007 }";
        let min = json_minify(src).unwrap();
        assert!(
            min.contains("1780157881126000007"),
            "big int corrupted: {min}"
        );
    }

    #[test]
    fn json_minify_rejects_unterminated_string() {
        assert!(json_minify("{\"a\": \"oops").is_none());
    }

    #[test]
    fn json_escaped_quote_does_not_end_string() {
        let src = "{\"a\":\"he said \\\"hi\\\"  x\"}";
        // the double-spaced run is inside the string → preserved
        assert_eq!(json_minify(src).unwrap(), src.replace("\n", ""));
    }

    #[test]
    fn compress_json_minifies_and_stays_valid_when_it_fits() {
        // budget between the spaced (38 B) and minified (22 B) sizes → minify path,
        // result fits, no donut, no marker. (Under-budget input passes verbatim by
        // the global rule; minify is the over-budget JSON handler.)
        let spaced = "{ \"k\" : 1 ,  \"list\" : [ 1 , 2 , 3 ] }";
        let c = compress("./tool", spaced, 30, art());
        assert_eq!(c.shape, OutputShape::Json);
        assert_eq!(c.text, "{\"k\":1,\"list\":[1,2,3]}");
    }

    #[test]
    fn compress_json_donuts_minified_bytes_when_still_too_big() {
        // Minify alone can't get under budget → donut the minified form, flagged.
        let big = format!("[{}]", "\"x\" , ".repeat(3000));
        let c = compress("./tool", &big, 400, art());
        assert_eq!(c.shape, OutputShape::Json);
        assert!(c.text.contains("json minified"));
        assert!(c.text.contains("elided"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn file_read_keeps_a_faithful_prefix_not_a_lossy_shape_view() {
        // A source file that mentions "error:" must NOT be log-filtered into a view
        // that drops middle functions; read_file gets a clean prefix + ranged nudge.
        let mut src = String::from("use std::io;\nfn top() {}\n");
        for i in 0..400 {
            src.push_str(&format!("fn middle_{i}() {{ /* error: not really */ }}\n"));
        }
        src.push_str("fn bottom_unique_marker() {}\n");
        let out = compress_file_read(&src, 512, art());
        // faithful prefix: the first lines are present verbatim, in order.
        assert!(
            out.starts_with("use std::io;\nfn top() {}\n"),
            "prefix not faithful: {out}"
        );
        // it's a prefix, so the bottom is NOT shown (but recoverable + nudged).
        assert!(!out.contains("fn bottom_unique_marker"));
        assert!(
            out.contains("read_file with offset/limit"),
            "no ranged-read nudge"
        );
        assert!(
            out.contains("of 403"),
            "should name total line count: {}",
            &out[out.len().saturating_sub(120)..]
        );
        assert!(out.contains("/tmp/zero/out-ab.txt"), "no artifact pointer");
    }

    #[test]
    fn file_read_of_json_comes_back_whole_and_minified() {
        // A JSON file read is losslessly minified (full content, still valid) rather
        // than prefixed — the model gets the entire document, fewer bytes.
        let json = "{\n  \"a\": 1,\n  \"b\": [1, 2, 3],\n  \"c\": \"keep\"\n}";
        let out = compress_file_read(json, 60, art());
        assert_eq!(out, "{\"a\":1,\"b\":[1,2,3],\"c\":\"keep\"}");
    }

    #[test]
    fn file_read_tiny_budget_names_the_file_size() {
        let big = "x\n".repeat(5000);
        let out = compress_file_read(&big, 20, art());
        assert!(out.contains("5000 lines"));
        assert!(out.contains("offset/limit"));
    }

    #[test]
    fn jsonl_is_not_minified_into_one_line() {
        // NDJSON: each line a value. Minify would concatenate them; the guard must
        // route it to the line-oriented path so record boundaries survive.
        let jsonl: String = (0..50)
            .map(|i| format!("{{\"i\": {i}, \"v\": \"x\"}}\n"))
            .collect();
        assert!(is_jsonl(&jsonl));
        let c = compress("cat events.jsonl", &jsonl, 200, art());
        assert_eq!(c.shape, OutputShape::Json);
        // boundaries preserved → there are still multiple lines / a fold marker,
        // never a single concatenated blob.
        assert!(
            c.text.contains('\n'),
            "JSONL collapsed to one line: {}",
            c.text
        );
    }

    #[test]
    fn pretty_array_of_objects_is_not_jsonl() {
        // Inner objects are INDENTED, so the column-0 check must not flag this as
        // JSONL — it's one document and should minify (regression for the trim bug).
        let pretty = "[\n  { \"a\": 1 },\n  { \"a\": 2 },\n  { \"a\": 3 }\n]";
        assert!(!is_jsonl(pretty), "pretty array misread as JSONL");
        let c = compress("cat arr.json", pretty, 25, art());
        assert_eq!(c.text, "[{\"a\":1},{\"a\":2},{\"a\":3}]");
    }

    #[test]
    fn single_multiline_json_object_is_still_minified() {
        // One pretty-printed object spanning many lines is NOT JSONL — it minifies.
        let pretty = "{\n  \"a\": 1,\n  \"b\": 2,\n  \"c\": [1, 2, 3]\n}";
        assert!(!is_jsonl(pretty));
        // budget between minified (25 B) and pretty (~40 B) → clean minified result.
        let c = compress("cat x.json", pretty, 30, art());
        assert_eq!(c.text, "{\"a\":1,\"b\":2,\"c\":[1,2,3]}");
    }

    #[test]
    fn compress_diff_keeps_headers_and_changes_drops_context() {
        let mut d =
            String::from("diff --git a/x b/x\nindex 111..222\n--- a/x\n+++ b/x\n@@ -1,5 +1,5 @@\n");
        d.push_str(" unchanged context\n".repeat(500).as_str());
        d.push_str("-removed line\n+added line\n");
        let c = compress("git diff", &d, 400, art());
        assert_eq!(c.shape, OutputShape::Diff);
        assert!(c.text.contains("@@ -1,5 +1,5 @@"), "hunk header dropped");
        assert!(c.text.contains("-removed line") && c.text.contains("+added line"));
        assert!(c.text.contains("+++ b/x") && c.text.contains("--- a/x"));
        assert!(
            !c.text.contains("unchanged context"),
            "context should be dropped"
        );
        assert!(c.text.contains("context lines elided"));
        assert!(c.kept_bytes < c.raw_bytes);
    }

    #[test]
    fn repeat_fold_collapses_a_numeric_range_keeping_both_ends() {
        // The seq-1-50000 case: folds to first + count + last, far under budget.
        let mut s: String = (1..=50_000).map(|i| format!("{i}\n")).collect();
        s.push_str("[exit 0]\n");
        let c = compress("seq 1 50000", &s, 4096, art());
        assert!(
            c.text.contains("\n1\n") || c.text.starts_with("1\n"),
            "first kept"
        );
        assert!(c.text.contains("50000"), "last value of the run kept");
        assert!(c.text.contains("[exit 0]"), "trailing distinct line kept");
        assert!(c.text.contains("similar lines"), "fold marker present");
        assert!(
            c.kept_bytes < 500,
            "should fold to well under 500 B, got {}",
            c.kept_bytes
        );
        assert!(c.text.contains("folded repeated runs"));
    }

    #[test]
    fn repeat_fold_leaves_short_runs_untouched() {
        // Fewer than REPEAT_MIN identical-skeleton lines must not fold.
        let s = "1\n2\n3\nalpha\nbeta\n";
        let folded = fold_repeats(s);
        assert_eq!(folded, s, "short run wrongly folded");
    }

    #[test]
    fn skeleton_hash_masks_digit_runs() {
        assert_eq!(
            skeleton_hash("Compiling foo v0.1"),
            skeleton_hash("Compiling foo v9.9")
        );
        assert_eq!(skeleton_hash("1"), skeleton_hash("50000"));
        assert_ne!(skeleton_hash("alpha"), skeleton_hash("beta"));
    }

    #[test]
    fn dir_output_donuts_when_lines_dont_fold() {
        // Dir has no dedicated handler. Use letters-only base-26 names (no digits,
        // so no shared skeleton) → adjacent lines differ → no fold → generic donut.
        let big_dir: String = (0..3000)
            .map(|i| {
                let mut s = String::new();
                let mut n = i;
                loop {
                    s.push((b'a' + (n % 26) as u8) as char);
                    n /= 26;
                    if n == 0 {
                        break;
                    }
                }
                s.push('\n');
                s
            })
            .collect();
        let cdir = compress("ls -R", &big_dir, 400, art());
        assert_eq!(cdir.shape, OutputShape::Dir);
        assert!(
            cdir.text.contains("elided"),
            "expected a donut on non-folding dir output"
        );
        assert!(cdir.kept_bytes < cdir.raw_bytes);
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
