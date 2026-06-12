//! Built-in tools the model can call: the filesystem + search primitives every
//! coding agent needs. Each is defined ([`ToolDef`]) and executed here, so the
//! capability lives in the engine (terminal and a future app share it).
//!
//! Execution is deliberately *gated upstream*: the frontend decides — per its
//! mode (normal confirms, auto-accept skips) and the [`crate::safety`] classifier
//! — whether to call [`execute`]. This module does the work and returns a string
//! result (or an error string the caller feeds back as a tool error). Shell
//! execution is intentionally NOT here; it needs the mode/safety gate and lives
//! in the frontend.
//!
//! Outputs are truncated (a researched pitfall: unbounded file/grep output blows
//! the context window) with an explicit "truncated" marker so the model knows to
//! narrow its query.

use crate::json::Value;
use crate::tools::{parse_arguments, ToolDef};
use std::fmt::Write as _;
use std::path::Path;

/// Max matching lines `grep` returns.
const MAX_GREP_HITS: usize = 100;
/// Max entries `list_dir` returns.
const MAX_DIR_ENTRIES: usize = 200;

/// The filesystem/search tools, as definitions to advertise to the model.
pub fn definitions() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "read_file",
            "Read a UTF-8 text file. Optionally pass a line range (offset/limit) \
             to fetch only a span — e.g. after grep points you at file:line, read \
             that span instead of the whole file to save context.",
            schema(&[
                ("path", "string", "Path to the file to read.", true),
                (
                    "offset",
                    "integer",
                    "1-based first line to read (optional; default 1).",
                    false,
                ),
                (
                    "limit",
                    "integer",
                    "Max number of lines to read from offset (optional).",
                    false,
                ),
            ]),
        ),
        ToolDef::new(
            "list_dir",
            "List the entries (files and subdirectories) of a directory.",
            schema(&[("path", "string", "Directory path to list.", true)]),
        ),
        ToolDef::new(
            "grep",
            "Search files under a directory for lines containing a substring.",
            schema(&[
                ("pattern", "string", "Substring to search for.", true),
                (
                    "path",
                    "string",
                    "Directory to search (default '.').",
                    false,
                ),
            ]),
        ),
        ToolDef::new(
            "write_file",
            "Create or overwrite a file with the given contents.",
            schema(&[
                ("path", "string", "Path to write.", true),
                ("content", "string", "Full file contents.", true),
            ]),
        ),
        ToolDef::new(
            "edit_file",
            "Replace the first occurrence of old_string with new_string in a file. \
             old_string must match exactly and be unique enough to be unambiguous.",
            schema(&[
                ("path", "string", "Path to edit.", true),
                ("old_string", "string", "Exact text to replace.", true),
                ("new_string", "string", "Replacement text.", true),
            ]),
        ),
        ToolDef::new(
            "bash",
            // The steering lives in the description (belt; the harness output cap
            // is the actual guarantee): nudge toward scoped, low-output commands so
            // a local model's small window isn't blown by an unbounded dump.
            "Run a shell command via `sh -c` and return its combined output + exit \
             code. Prefer scoped, low-output commands: pass paths to grep/ls rather \
             than searching the whole tree, use `head`/`grep -c`/`-l` and `--json` \
             where available, and read a file with read_file (offset/limit) instead \
             of `cat`. Huge output is capped (with the full result saved to a file \
             you can re-read), so narrow the command rather than relying on that.",
            schema(&[("command", "string", "The shell command to run.", true)]),
        ),
    ]
}

/// Run a built-in tool by name with its raw [`ToolCall`]-style arguments string.
/// Returns the result text, or an `Err(message)` the caller feeds back as a tool
/// error (the model then retries). `root`, if set, confines path access.
pub fn execute(name: &str, arguments: &str, root: Option<&Path>) -> Result<String, String> {
    let args = parse_arguments(&crate::message::ToolCall::new("", name, arguments))?;
    match name {
        "read_file" => read_file(&args, root),
        "list_dir" => list_dir(&args, root),
        "grep" => grep(&args, root),
        "write_file" => write_file(&args, root),
        "edit_file" => edit_file(&args, root),
        other => Err(format!("unknown tool: {other}")),
    }
}

/// True if `name` is a built-in tool.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "read_file" | "list_dir" | "grep" | "write_file" | "edit_file"
    )
}

// --- individual tools ---------------------------------------------------

fn read_file(args: &Value, root: Option<&Path>) -> Result<String, String> {
    let path = resolve(req_str(args, "path")?, root)?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read failed: {e}"))?;
    // Optional line range (1-based `offset`, `limit` lines) — lets the model fetch
    // the precise span a grep pointer named instead of the whole file. This is
    // the second half of the two-stage "search returns pointers, read fetches the
    // span" pattern that keeps the context window small.
    let offset = opt_u64(args, "offset");
    let limit = opt_u64(args, "limit");
    if offset.is_some() || limit.is_some() {
        let lines: Vec<&str> = text.lines().collect();
        let start = offset
            .map(|o| (o.max(1) - 1) as usize)
            .unwrap_or(0)
            .min(lines.len());
        let count = limit.map(|l| l as usize).unwrap_or(lines.len() - start);
        let end = start.saturating_add(count).min(lines.len());
        return Ok(lines[start..end].join("\n"));
    }
    // Return the file whole. Bounding for the context window is the agent loop's
    // single, recoverable job (cap_tool_result → compress_file_read): it spills the
    // COMPLETE bytes and shows a faithful prefix + offset/limit nudge. Truncating
    // here too would only corrupt that — a partial spill and an understated line
    // count — and saves no memory (read_to_string already loaded the whole file).
    Ok(text)
}

fn list_dir(args: &Value, root: Option<&Path>) -> Result<String, String> {
    let path = resolve(req_str(args, "path")?, root)?;
    let mut entries: Vec<String> = Vec::new();
    let rd = std::fs::read_dir(&path).map_err(|e| format!("list failed: {e}"))?;
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let suffix = match entry.file_type() {
            Ok(t) if t.is_dir() => "/",
            _ => "",
        };
        entries.push(format!("{name}{suffix}"));
    }
    entries.sort();
    let total = entries.len();
    entries.truncate(MAX_DIR_ENTRIES);
    let mut out = entries.join("\n");
    if total > MAX_DIR_ENTRIES {
        let _ = write!(
            out,
            "\n… ({} more entries omitted)",
            total - MAX_DIR_ENTRIES
        );
    }
    Ok(out)
}

fn grep(args: &Value, root: Option<&Path>) -> Result<String, String> {
    let pattern = req_str(args, "pattern")?;
    let dir = resolve(opt_str(args, "path").unwrap_or("."), root)?;
    let mut hits: Vec<String> = Vec::new();
    let mut truncated = false;
    walk(&dir, &mut |file| {
        if hits.len() >= MAX_GREP_HITS {
            truncated = true;
            return;
        }
        if let Ok(text) = std::fs::read_to_string(file) {
            for (n, line) in text.lines().enumerate() {
                if line.contains(pattern) {
                    // Cap each preview so one minified/huge line can't bloat the
                    // pointer list — the model reads the span via read_file if it
                    // needs more than this snippet.
                    let preview = grep_preview(line.trim());
                    // Pointer paths are root-relative: leaner per-hit, and still
                    // valid read_file input (which resolves against the same root).
                    let shown = root.and_then(|r| file.strip_prefix(r).ok()).unwrap_or(file);
                    hits.push(format!("{}:{}: {}", shown.display(), n + 1, preview));
                    if hits.len() >= MAX_GREP_HITS {
                        truncated = true;
                        break;
                    }
                }
            }
        }
    });
    if hits.is_empty() {
        return Ok(format!("no matches for {pattern:?}"));
    }
    let mut out = hits.join("\n");
    if truncated {
        let _ = write!(
            out,
            "\n… (capped at {MAX_GREP_HITS} matches; narrow your pattern)"
        );
    }
    Ok(out)
}

fn write_file(args: &Value, root: Option<&Path>) -> Result<String, String> {
    let path = resolve(req_str(args, "path")?, root)?;
    let content = req_str(args, "content")?;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&path, content).map_err(|e| format!("write failed: {e}"))?;
    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        path.display()
    ))
}

fn edit_file(args: &Value, root: Option<&Path>) -> Result<String, String> {
    let path = resolve(req_str(args, "path")?, root)?;
    let old = req_str(args, "old_string")?;
    let new = req_str(args, "new_string")?;
    if old == new {
        return Err("old_string and new_string are identical (no-op)".to_string());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("read failed: {e}"))?;
    let count = text.matches(old).count();
    if count == 0 {
        return Err("old_string not found in file".to_string());
    }
    if count > 1 {
        return Err(format!(
            "old_string is ambiguous: {count} occurrences — make it more specific"
        ));
    }
    let edited = text.replacen(old, new, 1);
    std::fs::write(&path, &edited).map_err(|e| format!("write failed: {e}"))?;
    Ok(format!("edited {}", path.display()))
}

// --- helpers ------------------------------------------------------------

/// Build a JSON-Schema object for a tool's parameters from (name, type, desc,
/// required) tuples.
fn schema(params: &[(&str, &str, &str, bool)]) -> Value {
    let props = params
        .iter()
        .map(|(name, ty, desc, _)| {
            (
                name.to_string(),
                Value::Object(vec![
                    ("type".to_string(), Value::Str(ty.to_string())),
                    ("description".to_string(), Value::Str(desc.to_string())),
                ]),
            )
        })
        .collect();
    let required: Vec<Value> = params
        .iter()
        .filter(|(_, _, _, req)| *req)
        .map(|(name, ..)| Value::Str(name.to_string()))
        .collect();
    Value::Object(vec![
        ("type".to_string(), Value::Str("object".to_string())),
        ("properties".to_string(), Value::Object(props)),
        ("required".to_string(), Value::Array(required)),
    ])
}

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required argument: {key}"))
}

fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

/// Trim a grep match line to a single short preview (≤160 chars, char-safe) so a
/// minified or generated line doesn't blow up the pointer list.
fn grep_preview(line: &str) -> String {
    const MAX: usize = 160;
    if line.chars().count() <= MAX {
        return line.to_string();
    }
    let cut: String = line.chars().take(MAX).collect();
    format!("{cut}…")
}

/// Optional non-negative integer argument (JSON numbers are f64).
fn opt_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|n| *n >= 0.0)
        .map(|n| n as u64)
}

/// Resolve a tool-supplied path against an optional confinement `root`. An
/// absolute path is rejected when a root is set; `..` that escapes the root is
/// rejected. Keeps a model from reading outside the workspace.
fn resolve(path: &str, root: Option<&Path>) -> Result<std::path::PathBuf, String> {
    match root {
        None => Ok(std::path::PathBuf::from(path)),
        Some(root) => {
            let p = Path::new(path);
            if p.is_absolute() {
                return Err("absolute paths are not allowed".to_string());
            }
            // Reject any `..` component — strict against textual escapes.
            if p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err("path escapes the workspace root".to_string());
            }
            let joined = root.join(p);
            // Symlink guard: the `..` check stops textual escapes, but a symlink
            // *inside* the root can still point outside it (e.g. a prior `ln -s /
            // esc` then `read_file esc/etc/passwd`). Canonicalize the existing
            // portion of the target and require it to stay within the canonical
            // root. A not-yet-created write target is checked via its nearest
            // existing ancestor (the directory the write lands in).
            if let Ok(canon_root) = root.canonicalize() {
                let mut probe: &Path = joined.as_path();
                let existing = loop {
                    match probe.canonicalize() {
                        Ok(c) => break Some(c),
                        Err(_) => match probe.parent() {
                            Some(par) => probe = par,
                            None => break None,
                        },
                    }
                };
                if let Some(existing) = existing {
                    if !existing.starts_with(&canon_root) {
                        return Err("path escapes the workspace root".to_string());
                    }
                }
            }
            Ok(joined)
        }
    }
}

/// Recursively visit files under `dir`, calling `f` for each regular file.
/// Skips hidden entries and common heavy dirs to keep grep cheap and relevant.
fn walk(dir: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() => walk(&path, f),
            Ok(t) if t.is_file() => f(&path),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use std::sync::atomic::{AtomicU64, Ordering};
    static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

    fn tmp() -> PathBuf {
        // Unique temp dir per call, race-free across parallel tests: a global
        // atomic counter + process id, and create_dir (which errors if the dir
        // already exists) in a retry loop — no TOCTOU `exists()` check.
        let base = std::env::temp_dir();
        loop {
            let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
            let dir = base.join(format!("zero-builtins-{}-{seq}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return dir,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("tmp dir create failed: {e}"),
            }
        }
    }

    fn args(json: &str) -> &str {
        json
    }

    #[test]
    fn definitions_cover_the_builtin_set() {
        let names: Vec<_> = definitions().iter().map(|d| d.name.clone()).collect();
        for n in ["read_file", "list_dir", "grep", "write_file", "edit_file"] {
            assert!(names.contains(&n.to_string()), "missing {n}");
            assert!(is_builtin(n));
        }
        assert!(!is_builtin("run_shell"));
        // bash is ADVERTISED (the model can call it) but is NOT a builtins::execute
        // tool — it's gated + run in the frontend (needs the safety/mode gate).
        assert!(names.contains(&"bash".to_string()), "bash not advertised");
        assert!(!is_builtin("bash"));
        // calling it through execute() is therefore an explicit error.
        assert!(execute("bash", r#"{"command":"echo hi"}"#, None)
            .unwrap_err()
            .contains("unknown tool"));
    }

    #[test]
    fn write_then_read_roundtrips() {
        let dir = tmp();
        execute(
            "write_file",
            r#"{"path":"a.txt","content":"hello world"}"#,
            Some(&dir),
        )
        .unwrap();
        let out = execute("read_file", r#"{"path":"a.txt"}"#, Some(&dir)).unwrap();
        assert_eq!(out, "hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_dir_sorts_and_marks_directories() {
        let dir = tmp();
        std::fs::write(dir.join("b.txt"), "x").unwrap();
        std::fs::create_dir(dir.join("sub")).unwrap();
        let out = execute("list_dir", r#"{"path":"."}"#, Some(&dir)).unwrap();
        assert_eq!(out, "b.txt\nsub/");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_finds_matches_with_line_numbers() {
        let dir = tmp();
        std::fs::write(dir.join("c.txt"), "alpha\nbeta needle\ngamma").unwrap();
        let out = execute("grep", r#"{"pattern":"needle","path":"."}"#, Some(&dir)).unwrap();
        assert!(out.contains("c.txt"));
        assert!(out.contains(":2:"));
        assert!(out.contains("beta needle"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_file_line_range_returns_only_the_span() {
        let dir = tmp();
        let body = (1..=100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("f.txt"), &body).unwrap();

        // offset+limit fetches exactly that span — much smaller than the whole.
        let span = execute(
            "read_file",
            r#"{"path":"f.txt","offset":10,"limit":3}"#,
            Some(&dir),
        )
        .unwrap();
        assert_eq!(span, "line 10\nline 11\nline 12");
        let whole = execute("read_file", r#"{"path":"f.txt"}"#, Some(&dir)).unwrap();
        assert!(span.len() < whole.len() / 5, "span not much smaller"); // proven saving

        // offset alone reads to EOF from that line.
        let tail = execute("read_file", r#"{"path":"f.txt","offset":99}"#, Some(&dir)).unwrap();
        assert_eq!(tail, "line 99\nline 100");

        // offset past EOF is empty, not an error (graceful).
        let past = execute("read_file", r#"{"path":"f.txt","offset":9999}"#, Some(&dir)).unwrap();
        assert_eq!(past, "");

        // No range → whole file unchanged (backward compatible).
        assert_eq!(whole, body);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_file_offset_defaults_and_clamps() {
        let dir = tmp();
        std::fs::write(dir.join("g.txt"), "a\nb\nc\nd").unwrap();
        // offset 0 is treated as 1 (1-based); limit beyond EOF clamps.
        let r = execute(
            "read_file",
            r#"{"path":"g.txt","offset":0,"limit":99}"#,
            Some(&dir),
        )
        .unwrap();
        assert_eq!(r, "a\nb\nc\nd");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_preview_is_capped_for_huge_lines() {
        let dir = tmp();
        let huge = format!("needle {}", "x".repeat(5000));
        std::fs::write(dir.join("min.js"), huge).unwrap();
        let out = execute("grep", r#"{"pattern":"needle"}"#, Some(&dir)).unwrap();
        // The pointer line is short despite the 5KB source line.
        let hit = out.lines().find(|l| l.contains("min.js")).unwrap();
        assert!(
            hit.chars().count() < 220,
            "preview not capped: {} chars",
            hit.chars().count()
        );
        assert!(hit.contains('…'));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_stage_grep_then_read_span() {
        // The differentiator workflow end to end: grep returns path:line pointers;
        // the model reads only the pointed-at span, never the whole file.
        let dir = tmp();
        let mut lines: Vec<String> = (1..=200).map(|i| format!("filler {i}")).collect();
        lines[120] = "the TARGET line".to_string(); // 0-based 120 → line 121
        std::fs::write(dir.join("big.rs"), lines.join("\n")).unwrap();

        let hits = execute("grep", r#"{"pattern":"TARGET","path":"."}"#, Some(&dir)).unwrap();
        assert!(hits.contains(":121:"));
        // Read a tight window around the hit instead of the 200-line file.
        let span = execute(
            "read_file",
            r#"{"path":"big.rs","offset":120,"limit":3}"#,
            Some(&dir),
        )
        .unwrap();
        assert!(span.contains("the TARGET line"));
        assert!(span.lines().count() == 3); // only the window
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_reports_no_matches() {
        let dir = tmp();
        std::fs::write(dir.join("c.txt"), "nothing here").unwrap();
        let out = execute("grep", r#"{"pattern":"zzz","path":"."}"#, Some(&dir)).unwrap();
        assert!(out.contains("no matches"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_skips_hidden_and_heavy_dirs() {
        let dir = tmp();
        std::fs::create_dir(dir.join("target")).unwrap();
        std::fs::write(dir.join("target").join("x.txt"), "needle").unwrap();
        std::fs::write(dir.join("ok.txt"), "needle").unwrap();
        let out = execute("grep", r#"{"pattern":"needle"}"#, Some(&dir)).unwrap();
        assert!(out.contains("ok.txt"));
        assert!(!out.contains("target"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edit_replaces_unique_string() {
        let dir = tmp();
        std::fs::write(dir.join("d.txt"), "foo bar baz").unwrap();
        execute(
            "edit_file",
            r#"{"path":"d.txt","old_string":"bar","new_string":"QUX"}"#,
            Some(&dir),
        )
        .unwrap();
        let out = std::fs::read_to_string(dir.join("d.txt")).unwrap();
        assert_eq!(out, "foo QUX baz");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn edit_rejects_ambiguous_and_missing_and_noop() {
        let dir = tmp();
        std::fs::write(dir.join("e.txt"), "x x x").unwrap();
        let amb = execute(
            "edit_file",
            r#"{"path":"e.txt","old_string":"x","new_string":"y"}"#,
            Some(&dir),
        );
        assert!(amb.unwrap_err().contains("ambiguous"));
        let miss = execute(
            "edit_file",
            r#"{"path":"e.txt","old_string":"zzz","new_string":"y"}"#,
            Some(&dir),
        );
        assert!(miss.unwrap_err().contains("not found"));
        let noop = execute(
            "edit_file",
            r#"{"path":"e.txt","old_string":"x","new_string":"x"}"#,
            Some(&dir),
        );
        assert!(noop.unwrap_err().contains("no-op"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn root_confinement_rejects_escape_and_absolute() {
        let dir = tmp();
        let esc = execute("read_file", r#"{"path":"../secret"}"#, Some(&dir));
        assert!(esc.unwrap_err().contains("escapes"));
        let abs = execute("read_file", r#"{"path":"/etc/hosts"}"#, Some(&dir));
        assert!(abs.unwrap_err().contains("absolute"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn root_confinement_rejects_symlink_escape() {
        // A symlink INSIDE the root pointing outside it must not become a read/write
        // escape hatch — the textual `..` check can't catch this; canonicalization
        // does. (Repro of the audited hole: `ln -s / esc` then read `esc/...`.)
        let dir = tmp();
        let outside = tmp();
        std::fs::write(outside.join("secret"), "classified").unwrap();
        std::os::unix::fs::symlink(&outside, dir.join("esc")).unwrap();

        // Read through the symlink → rejected.
        let r = execute("read_file", r#"{"path":"esc/secret"}"#, Some(&dir));
        assert!(
            r.unwrap_err().contains("escapes"),
            "symlink read escape not blocked"
        );
        // Write through the symlink (target dir exists via the link) → rejected.
        let w = execute(
            "write_file",
            r#"{"path":"esc/pwned","content":"x"}"#,
            Some(&dir),
        );
        assert!(
            w.unwrap_err().contains("escapes"),
            "symlink write escape not blocked"
        );
        // A normal in-root write still works (guard is not over-broad).
        let ok = execute(
            "write_file",
            r#"{"path":"sub/normal.txt","content":"hi"}"#,
            Some(&dir),
        );
        assert!(ok.is_ok(), "in-root write wrongly blocked: {ok:?}");

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&outside).ok();
    }

    #[test]
    fn missing_required_argument_errors() {
        let dir = tmp();
        let e = execute("read_file", args("{}"), Some(&dir));
        assert!(e.unwrap_err().contains("missing required argument: path"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_arguments_error_not_panic() {
        let e = execute("read_file", "{bad json", Some(Path::new(".")));
        assert!(e.unwrap_err().contains("invalid tool arguments"));
    }

    #[test]
    fn unknown_tool_is_reported() {
        let e = execute("frobnicate", "{}", None);
        assert!(e.unwrap_err().contains("unknown tool"));
    }

    #[test]
    fn read_returns_full_content_capping_is_the_loop_layer() {
        // read_file no longer self-truncates: it returns the file WHOLE so the
        // agent loop's single recoverable cap (compress_file_read) can spill the
        // complete bytes and report an accurate line count. (Pre-truncating here
        // corrupted both and saved no memory — read_to_string already loaded it.)
        let dir = tmp();
        let big = "a".repeat(50_000);
        std::fs::write(dir.join("big.txt"), &big).unwrap();
        let out = execute("read_file", r#"{"path":"big.txt"}"#, Some(&dir)).unwrap();
        assert_eq!(out, big, "read_file must return the whole file, uncapped");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn list_dir_truncates_and_marks_overflow() {
        let dir = tmp();
        for i in 0..(MAX_DIR_ENTRIES + 5) {
            std::fs::write(dir.join(format!("f{i:04}.txt")), "x").unwrap();
        }
        let out = execute("list_dir", r#"{"path":"."}"#, Some(&dir)).unwrap();
        assert!(out.contains("more entries omitted"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_caps_at_hit_limit() {
        let dir = tmp();
        // One file, more matching lines than the cap.
        let body = (0..(MAX_GREP_HITS + 20))
            .map(|i| format!("needle line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("many.txt"), body).unwrap();
        let out = execute("grep", r#"{"pattern":"needle"}"#, Some(&dir)).unwrap();
        assert!(out.contains("capped at"));
        assert_eq!(
            out.lines().filter(|l| l.contains("needle")).count(),
            MAX_GREP_HITS
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_caps_across_multiple_files() {
        let dir = tmp();
        // Fill the cap in the first file, then a second file's hits are skipped.
        let body = (0..MAX_GREP_HITS)
            .map(|i| format!("needle {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("a_full.txt"), body).unwrap();
        std::fs::write(dir.join("z_more.txt"), "needle extra").unwrap();
        let out = execute("grep", r#"{"pattern":"needle"}"#, Some(&dir)).unwrap();
        assert!(out.contains("capped at"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn grep_on_missing_dir_is_empty_not_error() {
        let dir = tmp();
        // walk() over a non-existent subdir just yields nothing.
        let out = execute("grep", r#"{"pattern":"x","path":"nope"}"#, Some(&dir)).unwrap();
        assert!(out.contains("no matches"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_missing_file_errors() {
        let dir = tmp();
        let e = execute("read_file", r#"{"path":"ghost.txt"}"#, Some(&dir));
        assert!(e.unwrap_err().contains("read failed"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn no_root_allows_absolute_paths() {
        // Without a root, absolute paths are permitted (the frontend decides).
        let dir = tmp();
        let f = dir.join("abs.txt");
        std::fs::write(&f, "ok").unwrap();
        let out = execute(
            "read_file",
            &format!(r#"{{"path":"{}"}}"#, f.display()),
            None,
        )
        .unwrap();
        assert_eq!(out, "ok");
        std::fs::remove_dir_all(&dir).ok();
    }
}
