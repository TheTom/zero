//! The agentic tool-call layer: tool definitions, robust parsing of a model's
//! tool calls (structured OR text-fallback), and the request-building + result
//! plumbing for the loop. Pure and std-only; the actual execution of a tool and
//! the network round-trip live in the frontend / backend, so everything here is
//! unit-tested without a model or a filesystem.
//!
//! Design choices are driven by hard-won pitfalls (see the research note):
//!  * **Detect a tool turn by the PRESENCE of tool calls, never by
//!    `finish_reason`** — local servers report `stop` even when calling tools.
//!  * **Content-fallback parser**: quantized models (Zero targets a quantized
//!    qwen) frequently emit calls as ```json fences or `<tool_call>` XML in the
//!    text instead of the structured field. We parse both.
//!  * **`arguments` stays a string** end to end (see [`crate::message::ToolCall`]).
//!  * **Step cap + doom-loop guard** bound runaway loops.

use crate::json::Value;
use crate::message::ToolCall;

/// A tool the model may call. `parameters` is a JSON Schema object describing the
/// arguments (mirrors OpenAI's `function.parameters` / MCP's `inputSchema`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolDef {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        ToolDef {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }

    /// The OpenAI `tools[]` entry: `{"type":"function","function":{...}}`.
    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("type".to_string(), Value::Str("function".to_string())),
            (
                "function".to_string(),
                Value::Object(vec![
                    ("name".to_string(), Value::Str(self.name.clone())),
                    (
                        "description".to_string(),
                        Value::Str(self.description.clone()),
                    ),
                    ("parameters".to_string(), self.parameters.clone()),
                ]),
            ),
        ])
    }
}

/// Build the `tools` array for a request body from a set of definitions.
pub fn tools_value(defs: &[ToolDef]) -> Value {
    Value::Array(defs.iter().map(ToolDef::to_value).collect())
}

/// Parse the assistant's tool calls from a (non-streamed) chat completion choice.
///
/// Tries the structured `message.tool_calls` field first; if that is absent or
/// empty, falls back to scanning `message.content` for ```json fences and
/// `<tool_call>…</tool_call>` blocks (what quantized/local models emit). IDs are
/// synthesized when the server omits them so results can always be correlated.
pub fn parse_tool_calls(message: &Value) -> Vec<ToolCall> {
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        let parsed = parse_structured(calls);
        if !parsed.is_empty() {
            return parsed;
        }
    }
    // Fallback: the call is hidden in the text content.
    if let Some(content) = message.get("content").and_then(Value::as_str) {
        return parse_from_text(content);
    }
    Vec::new()
}

/// Parse the structured `tool_calls` array. Synthesizes an id if missing (vLLM
/// streaming omits it) so every call can be answered with a `tool_call_id`.
fn parse_structured(calls: &[Value]) -> Vec<ToolCall> {
    let mut out = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let func = c.get("function");
        let name = func
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue; // a call with no name is unusable
        }
        let id = c
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{i}"));
        // `arguments` may be a JSON string (spec) or, on some local servers, an
        // object — normalize to a string either way.
        let arguments = match func.and_then(|f| f.get("arguments")) {
            Some(Value::Str(s)) => s.clone(),
            Some(other) => other.to_json(),
            None => "{}".to_string(),
        };
        out.push(ToolCall::new(id, name, arguments));
    }
    out
}

/// Scan free text for tool calls a model emitted instead of using the structured
/// field. Handles `<tool_call>{json}</tool_call>` (Hermes/Qwen) and a single
/// ```json fenced object with a `name`/`arguments` shape.
fn parse_from_text(content: &str) -> Vec<ToolCall> {
    let mut out = Vec::new();

    // 1) <tool_call> … </tool_call> blocks (possibly several).
    let mut rest = content;
    while let Some(start) = rest.find("<tool_call>") {
        let after = &rest[start + "<tool_call>".len()..];
        let Some(end) = after.find("</tool_call>") else {
            break;
        };
        let inner = after[..end].trim();
        if let Some(call) = tool_call_from_json(inner, out.len()) {
            out.push(call);
        }
        rest = &after[end + "</tool_call>".len()..];
    }
    if !out.is_empty() {
        return out;
    }

    // 2) A ```json … ``` fenced block that looks like a call.
    if let Some(body) = fenced_json(content) {
        if let Some(call) = tool_call_from_json(&body, 0) {
            out.push(call);
        }
    }
    out
}

/// Extract the body of the first ```json (or bare ```) fenced block.
fn fenced_json(content: &str) -> Option<String> {
    let start = content.find("```")?;
    let after = &content[start + 3..];
    // Skip an optional language tag up to the newline.
    let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
    let body = &after[body_start..];
    let end = body.find("```")?;
    Some(body[..end].trim().to_string())
}

/// Turn a JSON object `{"name":..,"arguments":..}` (arguments may be an object or
/// a string) into a [`ToolCall`], synthesizing an id from `index`.
fn tool_call_from_json(text: &str, index: usize) -> Option<ToolCall> {
    let v = Value::parse(text).ok()?;
    let name = v.get("name").and_then(Value::as_str)?.to_string();
    if name.is_empty() {
        return None;
    }
    // Accept `arguments` (OpenAI) or `parameters`/`args` (model variants).
    let args = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .or_else(|| v.get("args"));
    let arguments = match args {
        Some(Value::Str(s)) => s.clone(),
        Some(other) => other.to_json(),
        None => "{}".to_string(),
    };
    Some(ToolCall::new(format!("call_{index}"), name, arguments))
}

/// Parse a tool call's `arguments` string into a JSON value. Returns `Err` with a
/// human message on malformed JSON — the caller feeds that back as a tool error
/// so the model can retry, rather than crashing or poisoning history.
pub fn parse_arguments(call: &ToolCall) -> Result<Value, String> {
    let s = call.arguments.trim();
    if s.is_empty() {
        return Ok(Value::Object(vec![]));
    }
    Value::parse(s).map_err(|e| format!("invalid tool arguments JSON: {e}"))
}

/// A guard against runaway tool loops: a hard step cap plus detection of the same
/// (name, arguments) call repeating, which is the most common local-model
/// failure (it omits a param and never "succeeds").
#[derive(Debug, Clone)]
pub struct LoopGuard {
    max_steps: usize,
    step: usize,
    /// The last few (name, args) signatures, newest last.
    recent: Vec<String>,
    repeat_limit: usize,
}

impl LoopGuard {
    pub fn new(max_steps: usize) -> Self {
        LoopGuard {
            max_steps,
            step: 0,
            recent: Vec::new(),
            repeat_limit: 3,
        }
    }

    pub fn steps_taken(&self) -> usize {
        self.step
    }

    /// Record one completed step. Returns `false` when the step cap is reached.
    pub fn record_step(&mut self) -> bool {
        self.step += 1;
        self.step < self.max_steps
    }

    /// True when `calls` repeat the immediately-preceding identical batch
    /// `repeat_limit` times in a row — i.e. the model is stuck in a doom loop.
    pub fn is_doom_loop(&mut self, calls: &[ToolCall]) -> bool {
        let sig = calls
            .iter()
            .map(|c| format!("{}:{}", c.name, c.arguments))
            .collect::<Vec<_>>()
            .join("|");
        self.recent.push(sig.clone());
        if self.recent.len() > self.repeat_limit {
            self.recent.remove(0);
        }
        self.recent.len() == self.repeat_limit && self.recent.iter().all(|s| *s == sig)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(pairs: Vec<(&str, Value)>) -> Value {
        Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }
    fn s(v: &str) -> Value {
        Value::Str(v.to_string())
    }

    #[test]
    fn tool_def_serializes_to_openai_shape() {
        let def = ToolDef::new("read_file", "Read a file", obj(vec![("type", s("object"))]));
        let v = def.to_value();
        assert_eq!(v.get("type").and_then(Value::as_str), Some("function"));
        let f = v.get("function").unwrap();
        assert_eq!(f.get("name").and_then(Value::as_str), Some("read_file"));
        assert!(f.get("parameters").is_some());
    }

    #[test]
    fn parses_structured_tool_calls() {
        let msg = obj(vec![(
            "tool_calls",
            Value::Array(vec![obj(vec![
                ("id", s("call_abc")),
                (
                    "function",
                    obj(vec![("name", s("grep")), ("arguments", s(r#"{"q":"x"}"#))]),
                ),
            ])]),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments, r#"{"q":"x"}"#);
    }

    #[test]
    fn structured_missing_id_is_synthesized() {
        let msg = obj(vec![(
            "tool_calls",
            Value::Array(vec![obj(vec![(
                "function",
                obj(vec![("name", s("ls")), ("arguments", s("{}"))]),
            )])]),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls[0].id, "call_0");
    }

    #[test]
    fn structured_object_arguments_normalized_to_string() {
        // Some local servers return `arguments` as an object, not a string.
        let msg = obj(vec![(
            "tool_calls",
            Value::Array(vec![obj(vec![
                ("id", s("c1")),
                (
                    "function",
                    obj(vec![
                        ("name", s("ls")),
                        ("arguments", obj(vec![("path", s("/tmp"))])),
                    ]),
                ),
            ])]),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls[0].arguments, r#"{"path":"/tmp"}"#);
    }

    #[test]
    fn structured_call_without_name_is_skipped() {
        let msg = obj(vec![(
            "tool_calls",
            Value::Array(vec![obj(vec![("id", s("c1"))])]),
        )]);
        assert!(parse_tool_calls(&msg).is_empty());
    }

    #[test]
    fn fallback_parses_tool_call_xml_in_content() {
        let msg = obj(vec![(
            "content",
            s(
                r#"sure, let me look. <tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>"#,
            ),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments, r#"{"path":"a.txt"}"#);
    }

    #[test]
    fn fallback_parses_multiple_tool_call_blocks() {
        let msg = obj(vec![(
            "content",
            s(
                r#"<tool_call>{"name":"ls","arguments":{}}</tool_call><tool_call>{"name":"pwd","arguments":{}}</tool_call>"#,
            ),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].name, "pwd");
        assert_eq!(calls[1].id, "call_1");
    }

    #[test]
    fn fallback_parses_fenced_json() {
        let msg = obj(vec![(
            "content",
            s("Here:\n```json\n{\"name\":\"grep\",\"arguments\":{\"q\":\"foo\"}}\n```\n"),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments, r#"{"q":"foo"}"#);
    }

    #[test]
    fn fallback_accepts_parameters_alias() {
        let msg = obj(vec![(
            "content",
            s(r#"<tool_call>{"name":"ls","parameters":{"path":"."}}</tool_call>"#),
        )]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls[0].arguments, r#"{"path":"."}"#);
    }

    #[test]
    fn structured_field_wins_over_content() {
        let msg = obj(vec![
            (
                "tool_calls",
                Value::Array(vec![obj(vec![
                    ("id", s("real")),
                    (
                        "function",
                        obj(vec![("name", s("ls")), ("arguments", s("{}"))]),
                    ),
                ])]),
            ),
            ("content", s("<tool_call>{\"name\":\"nope\"}</tool_call>")),
        ]);
        let calls = parse_tool_calls(&msg);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
    }

    #[test]
    fn plain_text_yields_no_calls() {
        let msg = obj(vec![("content", s("just a normal answer, no tools"))]);
        assert!(parse_tool_calls(&msg).is_empty());
    }

    #[test]
    fn parse_arguments_handles_empty_and_valid_and_invalid() {
        assert_eq!(
            parse_arguments(&ToolCall::new("c", "t", "")).unwrap(),
            Value::Object(vec![])
        );
        let v = parse_arguments(&ToolCall::new("c", "t", r#"{"a":1}"#)).unwrap();
        assert_eq!(v.get("a").and_then(Value::as_f64), Some(1.0));
        assert!(parse_arguments(&ToolCall::new("c", "t", "{bad")).is_err());
    }

    #[test]
    fn loop_guard_caps_steps() {
        let mut g = LoopGuard::new(3);
        assert!(g.record_step()); // 1
        assert!(g.record_step()); // 2
        assert!(!g.record_step()); // 3 → cap reached
        assert_eq!(g.steps_taken(), 3);
    }

    #[test]
    fn loop_guard_detects_doom_loop() {
        let mut g = LoopGuard::new(100);
        let calls = vec![ToolCall::new("c", "ls", "{}")];
        assert!(!g.is_doom_loop(&calls)); // 1st
        assert!(!g.is_doom_loop(&calls)); // 2nd
        assert!(g.is_doom_loop(&calls)); // 3rd identical → doom
    }

    #[test]
    fn loop_guard_resets_on_different_call() {
        let mut g = LoopGuard::new(100);
        let a = vec![ToolCall::new("c", "ls", "{}")];
        let b = vec![ToolCall::new("c", "pwd", "{}")];
        g.is_doom_loop(&a);
        g.is_doom_loop(&a);
        assert!(!g.is_doom_loop(&b)); // different breaks the streak
    }

    #[test]
    fn tools_value_wraps_all_defs() {
        let defs = vec![
            ToolDef::new("a", "", obj(vec![])),
            ToolDef::new("b", "", obj(vec![])),
        ];
        assert_eq!(tools_value(&defs).as_array().map(<[_]>::len), Some(2));
    }

    // --- fuzz: tool-call parsing on untrusted model output ---------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn fuzz_parse_tool_calls_never_panics() {
        // parse_tool_calls runs on whatever a (possibly broken, possibly hostile)
        // model emits. It must never panic, and every returned call must carry a
        // non-empty name + id (so it can always be answered with a tool_call_id).
        let mut rng = Rng(0x700F_0005_0000_0001);
        // Fragments biased toward the structures the parser cares about.
        const FRAG: &[&str] = &[
            "<tool_call>",
            "</tool_call>",
            "{",
            "}",
            "[",
            "]",
            "\"name\"",
            ":",
            ",",
            "\"arguments\"",
            "```json",
            "```",
            "ls",
            "\\",
            "\"",
            "{}",
            "null",
            "中😀",
        ];
        for _ in 0..20_000 {
            let n = rng.below(12);
            let content: String = (0..n)
                .map(|_| FRAG[rng.below(FRAG.len() as u64) as usize])
                .collect();
            let msg = Value::Object(vec![("content".to_string(), Value::Str(content))]);
            for call in parse_tool_calls(&msg) {
                assert!(!call.name.is_empty(), "tool call with empty name");
                assert!(!call.id.is_empty(), "tool call with empty id");
            }
        }
    }

    #[test]
    fn fuzz_parse_arguments_never_panics() {
        // Arguments come from the model verbatim — parse must yield Ok or Err,
        // never panic, for arbitrary bytes.
        let mut rng = Rng(0xABCD_0011_2233_4455);
        for _ in 0..20_000 {
            let len = rng.below(40) as usize;
            let bytes: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            let args = String::from_utf8_lossy(&bytes).into_owned();
            let call = ToolCall::new("c", "t", args);
            let _ = parse_arguments(&call); // must not panic
        }
    }

    #[test]
    fn loop_guard_doom_detection_is_order_independent() {
        // Property: K identical batches in a row trip the guard regardless of the
        // call's exact name/args; a different batch always resets the streak.
        let mut rng = Rng(0x9999_1111_2222_3333);
        for _ in 0..2000 {
            let name = format!("t{}", rng.below(5));
            let args = format!("{{\"x\":{}}}", rng.below(5));
            let calls = vec![ToolCall::new("c", &name, &args)];
            let mut g = LoopGuard::new(1000);
            assert!(!g.is_doom_loop(&calls));
            assert!(!g.is_doom_loop(&calls));
            assert!(g.is_doom_loop(&calls)); // 3rd identical → doom
        }
    }
}
