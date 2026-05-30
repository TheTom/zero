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

/// Max bytes of a single tool result before truncation.
const MAX_OUTPUT: usize = 16_000;
/// Max matching lines `grep` returns.
const MAX_GREP_HITS: usize = 100;
/// Max entries `list_dir` returns.
const MAX_DIR_ENTRIES: usize = 200;

/// The filesystem/search tools, as definitions to advertise to the model.
pub fn definitions() -> Vec<ToolDef> {
    vec![
        ToolDef::new(
            "read_file",
            "Read a UTF-8 text file and return its contents.",
            schema(&[("path", "string", "Path to the file to read.", true)]),
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
    Ok(truncate(&text))
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
                    hits.push(format!("{}:{}: {}", file.display(), n + 1, line.trim()));
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
            // Reject any `..` component — simple and strict (no symlink games).
            if p.components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err("path escapes the workspace root".to_string());
            }
            Ok(root.join(p))
        }
    }
}

/// Truncate a result to [`MAX_OUTPUT`] bytes (on a char boundary) with a marker.
fn truncate(text: &str) -> String {
    if text.len() <= MAX_OUTPUT {
        return text.to_string();
    }
    let mut end = MAX_OUTPUT;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n… (output truncated at {MAX_OUTPUT} bytes; read a smaller range or grep)",
        &text[..end]
    )
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
    fn read_truncates_huge_files() {
        let dir = tmp();
        let big = "a".repeat(MAX_OUTPUT + 500);
        std::fs::write(dir.join("big.txt"), &big).unwrap();
        let out = execute("read_file", r#"{"path":"big.txt"}"#, Some(&dir)).unwrap();
        assert!(out.contains("truncated"));
        assert!(out.len() < big.len());
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
    fn truncate_respects_char_boundary() {
        // A multi-byte char straddling MAX_OUTPUT must not split mid-codepoint.
        let dir = tmp();
        let body = "é".repeat(MAX_OUTPUT); // 2 bytes each → well over the cap
        std::fs::write(dir.join("u.txt"), &body).unwrap();
        let out = execute("read_file", r#"{"path":"u.txt"}"#, Some(&dir)).unwrap();
        assert!(out.contains("truncated")); // and didn't panic on a boundary
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
