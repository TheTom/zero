// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Loop gates: the harness's "did it actually work?" check. A gate names a
//! command (run by the frontend through the [`crate::tools`] executor seam), an
//! **extractor** that pulls a value out of that command's output, and a **pass
//! expression** the extracted value must satisfy. The model never gets to claim a
//! win — the harness runs the gate and compares the number itself.
//!
//! Everything here is **pure**: `evaluate(gate, stdout, exit_code)` does no I/O,
//! so the whole win/lose decision is deterministic and unit-tested without a
//! process. (PRD: Loop Builder → "Gates are measured, not asserted".)

use crate::json::Value;

/// How to pull the value a gate checks out of its command's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Extractor {
    /// The process exit code, as a number (`parse = "exit"`).
    Exit,
    /// A dotted/indexed path into the stdout parsed as JSON
    /// (`parse = "json:.cosine_mean"` or `json:.results[0].score`).
    Json(String),
    /// The raw stdout text, trimmed (the default when no `parse` is given).
    Text,
}

impl Extractor {
    /// Parse a `parse = "…"` value into an [`Extractor`]. `exit` → [`Exit`],
    /// `json:<path>` → [`Json`], anything else (incl. empty) → [`Text`].
    ///
    /// [`Exit`]: Extractor::Exit
    /// [`Json`]: Extractor::Json
    pub fn parse(spec: &str) -> Extractor {
        let s = spec.trim();
        if s == "exit" {
            Extractor::Exit
        } else if let Some(path) = s.strip_prefix("json:") {
            Extractor::Json(path.trim().to_string())
        } else {
            Extractor::Text
        }
    }
}

/// What kind of gate this is. A `Command` gate parses a command's output; a
/// `Rubric` gate is judged in a fresh context (handled by the frontend, not here)
/// — `evaluate` only scores `Command` gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    Command,
    Rubric,
}

/// One gate: the command to run, how to read its result, and the bar it must
/// clear. `run` is executed by the frontend (through the `Executor` seam); this
/// module only scores the output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gate {
    pub name: String,
    pub kind: GateKind,
    /// The shell command the harness runs (empty for a rubric gate).
    pub run: String,
    /// How to extract the checked value from the command's stdout.
    pub extractor: Extractor,
    /// The pass expression the extracted value must satisfy (see [`evaluate`]).
    pub pass: String,
    /// For a rubric gate: the criteria file. Ignored by [`evaluate`].
    pub rubric: Option<String>,
}

impl Gate {
    /// A command gate from its parts.
    pub fn command(name: &str, run: &str, parse: &str, pass: &str) -> Gate {
        Gate {
            name: name.to_string(),
            kind: GateKind::Command,
            run: run.to_string(),
            extractor: Extractor::parse(parse),
            pass: pass.to_string(),
            rubric: None,
        }
    }
}

/// The result of scoring a gate — what the harness records and the model cites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateOutcome {
    pub name: String,
    pub passed: bool,
    /// The extracted value, rendered as text (for the ledger + win citations).
    pub actual: String,
    /// A human-readable reason (`"0.9942 >= 0.9987 → false"`, `"bad json"`, …).
    pub detail: String,
}

/// Score a **command** gate against the output of its run. Pure and total — never
/// panics, never does I/O. A rubric gate (judged elsewhere) returns a non-passing
/// outcome explaining it must be judged in a fresh context.
///
/// Pass expression grammar (the operand is trimmed):
/// - numeric: `>= N`, `> N`, `<= N`, `< N`, `== N`, `!= N` — compares the
///   extracted value parsed as a number; a non-numeric value fails closed.
/// - text: `contains <s>`, `matches <s>` (substring), `== <s>` / `!= <s>` when
///   the operand isn't a number — compares the extracted text.
/// - bare `true` / `false` — checks a boolean-ish extracted value.
pub fn evaluate(gate: &Gate, stdout: &str, exit_code: i32) -> GateOutcome {
    if gate.kind == GateKind::Rubric {
        return GateOutcome {
            name: gate.name.clone(),
            passed: false,
            actual: String::new(),
            detail: "rubric gate — judge in a fresh context".to_string(),
        };
    }
    let (actual, num) = extract(&gate.extractor, stdout, exit_code);
    let (passed, detail) = check_pass(&gate.pass, &actual, num);
    GateOutcome {
        name: gate.name.clone(),
        passed,
        actual,
        detail,
    }
}

/// Extract the checked value as `(text, maybe_number)`.
fn extract(extractor: &Extractor, stdout: &str, exit_code: i32) -> (String, Option<f64>) {
    match extractor {
        Extractor::Exit => (exit_code.to_string(), Some(exit_code as f64)),
        Extractor::Text => {
            let t = stdout.trim().to_string();
            let n = t.parse::<f64>().ok();
            (t, n)
        }
        Extractor::Json(path) => match Value::parse(stdout.trim()) {
            Ok(v) => match navigate(&v, path) {
                Some(found) => render_value(found),
                None => (format!("<no json path {path}>"), None),
            },
            Err(_) => ("<bad json>".to_string(), None),
        },
    }
}

/// Render a JSON value to `(text, maybe_number)` for comparison.
fn render_value(v: &Value) -> (String, Option<f64>) {
    match v {
        Value::Num(n) => (trim_num(*n), Some(*n)),
        Value::Str(s) => (s.clone(), s.parse::<f64>().ok()),
        Value::Bool(b) => (b.to_string(), None),
        Value::Null => ("null".to_string(), None),
        other => (other.to_json(), None),
    }
}

/// Format a number without a trailing `.0` for integers (cleaner citations).
fn trim_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

/// Walk a dotted/indexed JSON path (`.a.b[0].c`, leading `.` optional).
fn navigate<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = root;
    for seg in path_segments(path) {
        cur = match seg {
            Seg::Key(k) => cur.get(&k)?,
            Seg::Index(i) => cur.as_array()?.get(i)?,
        };
    }
    Some(cur)
}

enum Seg {
    Key(String),
    Index(usize),
}

/// Split `.a.b[0].c` (or `a.b[0].c`) into key/index segments.
fn path_segments(path: &str) -> Vec<Seg> {
    let mut segs = Vec::new();
    for part in path.split('.') {
        if part.is_empty() {
            continue;
        }
        // A part may be `key`, `key[0]`, or `[0]`.
        let mut rest = part;
        if let Some(name_end) = rest.find('[') {
            let (name, brackets) = rest.split_at(name_end);
            if !name.is_empty() {
                segs.push(Seg::Key(name.to_string()));
            }
            rest = brackets;
            // Consume one or more `[i]` groups.
            while let Some(close) = rest.find(']') {
                let idx = &rest[1..close];
                if let Ok(i) = idx.trim().parse::<usize>() {
                    segs.push(Seg::Index(i));
                }
                rest = &rest[close + 1..];
                if !rest.starts_with('[') {
                    break;
                }
            }
        } else {
            segs.push(Seg::Key(rest.to_string()));
        }
    }
    segs
}

/// Evaluate a pass expression against the extracted value, returning
/// `(passed, detail)`.
fn check_pass(pass: &str, actual: &str, num: Option<f64>) -> (bool, String) {
    let p = pass.trim();

    // Two-char numeric ops first so `>=` isn't read as `>`.
    for (op, f) in [
        (">=", cmp_ge as fn(f64, f64) -> bool),
        ("<=", cmp_le),
        ("==", cmp_eq),
        ("!=", cmp_ne),
    ] {
        if let Some(rhs) = p.strip_prefix(op) {
            return numeric_or_text(op, rhs.trim(), actual, num, f);
        }
    }
    for (op, f) in [(">", cmp_gt as fn(f64, f64) -> bool), ("<", cmp_lt)] {
        if let Some(rhs) = p.strip_prefix(op) {
            return numeric_or_text(op, rhs.trim(), actual, num, f);
        }
    }
    if let Some(s) = p
        .strip_prefix("contains ")
        .or_else(|| p.strip_prefix("matches "))
    {
        let needle = unquote(s.trim());
        let ok = actual.contains(&needle);
        return (ok, format!("{actual:?} contains {needle:?} → {ok}"));
    }
    if p == "true" || p == "false" {
        let want = p == "true";
        let ok = actual == p || (num == Some(1.0) && want) || (num == Some(0.0) && !want);
        return (ok, format!("{actual:?} is {p} → {ok}"));
    }
    (false, format!("unrecognized pass expression {pass:?}"))
}

/// Apply a numeric op when both sides are numbers; otherwise fall back to text
/// equality for `==`/`!=` (so `pass = "== ok"` works on a string extractor).
fn numeric_or_text(
    op: &str,
    rhs: &str,
    actual: &str,
    num: Option<f64>,
    f: fn(f64, f64) -> bool,
) -> (bool, String) {
    if let (Some(a), Ok(b)) = (num, rhs.parse::<f64>()) {
        let ok = f(a, b);
        return (ok, format!("{a} {op} {b} → {ok}"));
    }
    // Text comparison only makes sense for equality ops.
    match op {
        "==" => {
            let want = unquote(rhs);
            let ok = actual == want;
            (ok, format!("{actual:?} == {want:?} → {ok}"))
        }
        "!=" => {
            let want = unquote(rhs);
            let ok = actual != want;
            (ok, format!("{actual:?} != {want:?} → {ok}"))
        }
        _ => (false, format!("{actual:?} is not numeric for {op} {rhs:?}")),
    }
}

fn cmp_ge(a: f64, b: f64) -> bool {
    a >= b
}
fn cmp_le(a: f64, b: f64) -> bool {
    a <= b
}
fn cmp_gt(a: f64, b: f64) -> bool {
    a > b
}
fn cmp_lt(a: f64, b: f64) -> bool {
    a < b
}
fn cmp_eq(a: f64, b: f64) -> bool {
    a == b
}
fn cmp_ne(a: f64, b: f64) -> bool {
    a != b
}

/// Strip matching surrounding single or double quotes, if present.
fn unquote(s: &str) -> String {
    let s = s.trim();
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let (f, l) = (bytes[0], bytes[bytes.len() - 1]);
        if (f == b'"' && l == b'"') || (f == b'\'' && l == b'\'') {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractor_parse_classifies() {
        assert_eq!(Extractor::parse("exit"), Extractor::Exit);
        assert_eq!(
            Extractor::parse("json:.cosine_mean"),
            Extractor::Json(".cosine_mean".to_string())
        );
        assert_eq!(Extractor::parse(""), Extractor::Text);
        assert_eq!(Extractor::parse("whatever"), Extractor::Text);
    }

    #[test]
    fn json_numeric_gate_passes_and_fails() {
        let g = Gate::command("quality", "run.sh", "json:.cosine_mean", ">= 0.9987");
        let pass = evaluate(&g, r#"{"cosine_mean": 0.9990}"#, 0);
        assert!(pass.passed, "{}", pass.detail);
        assert_eq!(pass.actual, "0.999");
        let fail = evaluate(&g, r#"{"cosine_mean": 0.9942}"#, 0);
        assert!(!fail.passed);
        assert!(fail.detail.contains("0.9987"));
    }

    #[test]
    fn json_nested_and_indexed_path() {
        let g = Gate::command("g", "x", "json:.results[1].score", "> 10");
        let out = evaluate(&g, r#"{"results":[{"score":5},{"score":42}]}"#, 0);
        assert!(out.passed);
        assert_eq!(out.actual, "42");
    }

    #[test]
    fn exit_code_gate() {
        let g = Gate::command("build", "make", "exit", "== 0");
        assert!(evaluate(&g, "", 0).passed);
        assert!(!evaluate(&g, "", 1).passed);
        let nonzero = Gate::command("fails", "x", "exit", "!= 0");
        assert!(evaluate(&nonzero, "", 2).passed);
    }

    #[test]
    fn text_contains_and_equality() {
        let g = Gate::command("smoke", "x", "", "contains PASSED");
        assert!(evaluate(&g, "tests: PASSED (42)", 0).passed);
        assert!(!evaluate(&g, "tests: FAILED", 0).passed);
        let eq = Gate::command("eq", "x", "", "== ok");
        assert!(evaluate(&eq, "  ok  ", 0).passed); // trimmed
        assert!(!evaluate(&eq, "nope", 0).passed);
    }

    #[test]
    fn all_numeric_operators() {
        let mk = |pass: &str| Gate::command("g", "x", "", pass);
        assert!(evaluate(&mk("> 5"), "6", 0).passed);
        assert!(!evaluate(&mk("> 5"), "5", 0).passed);
        assert!(evaluate(&mk("< 5"), "4", 0).passed);
        assert!(evaluate(&mk(">= 5"), "5", 0).passed);
        assert!(evaluate(&mk("<= 5"), "5", 0).passed);
        assert!(evaluate(&mk("== 5"), "5", 0).passed);
        assert!(evaluate(&mk("!= 5"), "6", 0).passed);
    }

    #[test]
    fn non_numeric_actual_fails_a_numeric_gate_closed() {
        let g = Gate::command("g", "x", "", ">= 0.99");
        let out = evaluate(&g, "not a number", 0);
        assert!(!out.passed);
        assert!(out.detail.contains("not numeric"));
    }

    #[test]
    fn bad_json_and_missing_path_fail_gracefully() {
        let g = Gate::command("g", "x", "json:.a", ">= 1");
        assert!(!evaluate(&g, "{not json", 0).passed);
        assert!(evaluate(&g, "{not json", 0).actual.contains("bad json"));
        let missing = evaluate(&g, r#"{"b":2}"#, 0);
        assert!(!missing.passed);
        assert!(missing.actual.contains("no json path"));
    }

    #[test]
    fn rubric_gate_is_not_scored_here() {
        let g = Gate {
            name: "writeup".to_string(),
            kind: GateKind::Rubric,
            run: String::new(),
            extractor: Extractor::Text,
            pass: String::new(),
            rubric: Some("criteria.md".to_string()),
        };
        let out = evaluate(&g, "anything", 0);
        assert!(!out.passed);
        assert!(out.detail.contains("fresh context"));
    }

    #[test]
    fn unrecognized_pass_is_a_clear_fail() {
        let g = Gate::command("g", "x", "", "definitely-not-an-op");
        let out = evaluate(&g, "5", 0);
        assert!(!out.passed);
        assert!(out.detail.contains("unrecognized"));
    }

    #[test]
    fn quotes_are_stripped_in_text_ops() {
        let g = Gate::command("g", "x", "", r#"contains "hello world""#);
        assert!(evaluate(&g, "say hello world now", 0).passed);
        let eq = Gate::command("g", "x", "", "== 'done'");
        assert!(evaluate(&eq, "done", 0).passed);
    }

    #[test]
    fn boolean_pass() {
        let g = Gate::command("g", "x", "json:.ok", "true");
        assert!(evaluate(&g, r#"{"ok": true}"#, 0).passed);
        assert!(!evaluate(&g, r#"{"ok": false}"#, 0).passed);
    }

    #[test]
    fn integers_render_without_trailing_zero() {
        let g = Gate::command("g", "x", "json:.n", ">= 0");
        assert_eq!(evaluate(&g, r#"{"n": 6401}"#, 0).actual, "6401");
    }
}
