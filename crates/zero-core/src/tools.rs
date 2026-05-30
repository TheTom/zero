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
//!  * **Progress-based [`LoopGuard`]** ends runaways via stuck detection + a soft
//!    nudge (not a step cap), so legitimately long tasks run free.

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

/// What the loop guard advises after a round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopVerdict {
    /// Keep going — the turn is making progress.
    Continue,
    /// The model appears stuck (repeating work without progress). Inject this
    /// guidance as a message and give it ONE more round to converge.
    Nudge(String),
    /// Stuck — discard the polluted history and RESTART the loop from a clean
    /// context (the original task + a concrete progress summary the caller builds).
    /// Mechanistically the strongest fix: a fresh context doesn't contain the
    /// repetition attractor the way a nudged one does. The caller rebuilds `conv`.
    Reset,
    /// Still stuck after recovery (or a catastrophe backstop tripped). End the turn.
    Stop(String),
}

/// What the guard does the FIRST time it detects a stuck pattern. Lets the
/// escalation policy be A/B-ablated (stop vs nudge vs reset) without touching the
/// detection logic. (A repeat of the same stuck pattern always escalates to Stop.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recovery {
    /// Stop immediately on first detection. Simplest; abandons the task.
    StopOnly,
    /// Inject a corrective message, give one more round (the current default).
    Nudge,
    /// Discard history and restart from task + progress summary. Caller rebuilds.
    Reset,
}

/// A guard against runaway tool loops driven by **lack of progress, not step
/// count** — because a step cap punishes legitimately long tasks and only bounds
/// waste instead of preventing it (a 25-round wander still burns 25 rounds of
/// growing context before the cap fires). Instead it watches a sliding window of
/// `(action, result)` signatures and flags *non-progress*:
///
/// * the same `(call, result)` pair recurs — you did the same thing, got the same
///   thing back: definitionally no progress;
/// * the same action recurs `repeat_limit` times in the window (even if not
///   consecutive — catches read→edit→read→edit churn on one file);
/// * a short A→B→A→B / A→B→C→A→B→C action cycle.
///
/// On detection it returns [`LoopVerdict::Nudge`] (soft: tell the model to stop
/// exploring and finalize). It is FORGIVING: if the model then makes progress the
/// nudge is considered to have worked and the turn runs free again — we only
/// [`LoopVerdict::Stop`] when it gets stuck *the same way* (same action signature)
/// a second time. A different stuck pattern earns its own fresh nudge. A high
/// `backstop` round count is the last resort against a pathological loop that keeps
/// emitting novel-looking calls — not the working limit, just a catastrophe ceiling.
#[derive(Debug, Clone)]
pub struct LoopGuard {
    /// Window of `(name_sig, action_sig, result_sig)` for recent rounds, newest
    /// last. `name_sig` = the tool NAMES only (ignoring args) so "9 writes to the
    /// same file with different content" is caught as churn; `action_sig` = name +
    /// args (exact repeat); `result_sig` = the (capped) outputs.
    window: Vec<(String, String, String)>,
    window_cap: usize,
    repeat_limit: usize,
    /// How many times one tool NAME may appear in the window before it's churn.
    name_limit: usize,
    rounds: usize,
    backstop: usize,
    /// The stuck-signature we last NUDGED about, if any. Set on a nudge; cleared
    /// only after `recovery_rounds` CONSECUTIVE clean rounds (sustained recovery,
    /// not a single varied round). We STOP if the same signature gets stuck again
    /// before recovering — "stuck the same way twice".
    nudged_sig: Option<String>,
    /// Consecutive clean (progress) rounds since the last stuck signal.
    clean_streak: usize,
    /// Clean rounds required to consider a nudge "recovered" and clear the strike.
    recovery_rounds: usize,
    /// Total nudges issued this turn. A model that keeps tripping DIFFERENT stuck
    /// patterns (oscillating just enough to dodge the same-way check) still isn't
    /// converging — after `max_nudges` we stop regardless.
    nudge_count: usize,
    max_nudges: usize,
    /// The message injected on a nudge. `{tool}` and `{count}` are filled with the
    /// repeated tool name and how many times it appears in the window (so the nudge
    /// can name what's stuck). Overridable so the wording can be A/B-ablated.
    nudge_template: String,
    /// What to do on FIRST detection (ablatable: stop / nudge / reset).
    recovery: Recovery,
    /// Resets used this turn, and the cap. A fresh context that ALSO wanders the
    /// same way would otherwise reset forever — after `max_resets` we Stop.
    reset_count: usize,
    max_resets: usize,
}

/// The default nudge: firm diagnosis + an explicit positive redirect (the
/// best-evidence combo — naked prohibition backfires on small models). `{tool}`
/// names the repeated tool; `{count}` how many times it ran.
pub const DEFAULT_NUDGE: &str = "You appear to be repeating tool calls without making \
progress. Stop exploring: finalize your work now, verify it once, and give your best \
answer. If the task is already done, just answer.";

impl LoopGuard {
    /// `backstop` is the catastrophe ceiling (rounds) — set generously; stuck
    /// detection is what should normally end a runaway, not this.
    pub fn new(backstop: usize) -> Self {
        LoopGuard {
            window: Vec::new(),
            window_cap: 12,
            repeat_limit: 3,
            // Trigger SOONER. Evidence (Huang'24 self-correction is weak; our own
            // 30-run nudge ablation: wording had no effect, ~100% of nudged runs
            // still hard-stopped) says the nudge rarely rescues, so detecting churn
            // late just burns tokens before the inevitable stop. OpenHands fires at
            // 3-6 repeats; match that. name_limit 5→4 catches "polish-forever" at
            // ~rewrite 4 (was 9) without false-positiving on a healthy agent that
            // rotates through its ~6 tools (balanced use ≈2-3 of each per 12-window;
            // churn means ONE tool dominating >1/3 of recent activity). max_nudges
            // 3→1 = one nudge (a free rider on the stop, in case it helps) then stop.
            name_limit: 4,
            rounds: 0,
            backstop: backstop.max(1),
            nudged_sig: None,
            clean_streak: 0,
            recovery_rounds: 2,
            nudge_count: 0,
            max_nudges: 1,
            nudge_template: DEFAULT_NUDGE.to_string(),
            recovery: Recovery::Nudge,
            reset_count: 0,
            max_resets: 2,
        }
    }

    /// Set the first-detection recovery policy (stop / nudge / reset). Ablatable.
    pub fn with_recovery(mut self, r: Recovery) -> Self {
        self.recovery = r;
        self
    }

    /// Clear the stuck-detection window. The caller invokes this after acting on a
    /// [`LoopVerdict::Reset`] (rebuilding the conversation) so the fresh loop starts
    /// with no repetition history priming another immediate trip.
    pub fn reset_window(&mut self) {
        self.window.clear();
        self.nudged_sig = None;
        self.clean_streak = 0;
    }

    /// Raise the nudge ceiling — how many times the guard will nudge before it
    /// stops outright. Production ships with 1 (research: nudges rarely rescue, so
    /// don't burn rounds re-nudging); tests of the *forgiveness* semantics set it
    /// higher to exercise nudge→recover→nudge paths in isolation.
    #[cfg(test)]
    fn with_max_nudges(mut self, n: usize) -> Self {
        self.max_nudges = n;
        self
    }

    /// Override the nudge message (for A/B ablation of tone/specificity). Empty
    /// input keeps the default. `{tool}`/`{count}` placeholders are filled per-fire.
    pub fn with_nudge(mut self, template: &str) -> Self {
        if !template.trim().is_empty() {
            self.nudge_template = template.to_string();
        }
        self
    }

    pub fn steps_taken(&self) -> usize {
        self.rounds
    }

    /// Signature of the tool NAMES in a batch (ignoring args). `name_limit`
    /// repeats of this in the window = churn (e.g. 9 write_file with different
    /// content — "polishing forever").
    fn name_sig(calls: &[ToolCall]) -> String {
        calls
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join("|")
    }

    /// A stable signature for a batch of calls (name + arguments).
    fn action_sig(calls: &[ToolCall]) -> String {
        calls
            .iter()
            .map(|c| format!("{}:{}", c.name, c.arguments))
            .collect::<Vec<_>>()
            .join("|")
    }

    /// A normalized signature for a batch of results — capped so minor tails don't
    /// defeat equality, but enough to tell "same output" from "different output".
    fn result_sig(results: &[String]) -> String {
        let joined = results.join("|");
        let cap = 200.min(joined.len());
        let mut end = cap;
        while end > 0 && !joined.is_char_boundary(end) {
            end -= 1;
        }
        joined[..end].to_string()
    }

    /// Detect a repeating A→B→A→B (period 2) or A→B→C→A→B→C (period 3) cycle at
    /// the tail of the action sequence.
    fn has_cycle(actions: &[&str]) -> bool {
        for period in [2usize, 3] {
            if actions.len() >= period * 2 {
                let n = actions.len();
                let tail = &actions[n - period * 2..];
                if tail[..period] == tail[period..] && tail[0] != tail[1.min(period - 1)] {
                    // require the cycle to have >1 distinct element (A→A→A is the
                    // repeat case, handled separately; this is the alternation case)
                    if tail.iter().collect::<std::collections::HashSet<_>>().len() > 1 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Record one completed round (the calls the model made and the results it got)
    /// and advise whether to continue, nudge, or stop. Progress-based, not step-count.
    pub fn record_round(&mut self, calls: &[ToolCall], results: &[String]) -> LoopVerdict {
        self.rounds += 1;
        let nsig = Self::name_sig(calls);
        let asig = Self::action_sig(calls);
        let rsig = Self::result_sig(results);
        self.window.push((nsig.clone(), asig.clone(), rsig.clone()));
        if self.window.len() > self.window_cap {
            self.window.remove(0);
        }

        // Signal 1: same (action,result) pair recurs → did the same thing, got the
        // same thing back. The strongest no-progress signal.
        let pair_repeats = self
            .window
            .iter()
            .filter(|(_, a, r)| *a == asig && *r == rsig)
            .count();
        // Signal 2: same action (name+args) recurs repeat_limit times.
        let action_repeats = self.window.iter().filter(|(_, a, _)| *a == asig).count();
        // Signal 3: same tool NAME (ignoring args) recurs name_limit times — catches
        // "polish forever": 9 write_file with different content, same path. This is
        // the productive-looking churn the action signal misses.
        let name_repeats = self.window.iter().filter(|(n, _, _)| *n == nsig).count();
        // Signal 4: short alternating cycle of action types.
        let actions: Vec<&str> = self.window.iter().map(|(_, a, _)| a.as_str()).collect();
        let cycling = Self::has_cycle(&actions);

        let stuck = pair_repeats >= 2
            || action_repeats >= self.repeat_limit
            || name_repeats >= self.name_limit
            || cycling;
        // Key the "stuck the same way twice" strike on the KIND of stuckness: exact
        // action repeats key on asig; name-churn (different args each time) keys on
        // the tool-name signature so the second strike can actually match.
        let stuck_key = if pair_repeats >= 2 || action_repeats >= self.repeat_limit {
            asig.clone()
        } else {
            format!("names:{nsig}")
        };

        if self.rounds >= self.backstop {
            return LoopVerdict::Stop(format!(
                "catastrophe backstop: {} rounds without finishing",
                self.rounds
            ));
        }
        if stuck {
            self.clean_streak = 0;
            return self.on_stuck(stuck_key, calls, name_repeats.max(action_repeats));
        }
        // Progress this round. Require SUSTAINED recovery (recovery_rounds clean in
        // a row) before clearing the strike — a single varied round between sticks
        // is oscillation, not recovery, and must not reset the "same way twice" check.
        self.clean_streak += 1;
        if self.clean_streak >= self.recovery_rounds {
            self.nudged_sig = None;
        }
        LoopVerdict::Continue
    }

    /// Decide the verdict on a detected stuck pattern, per the `recovery` policy.
    /// `stuck_key` identifies the KIND of stuckness (so a repeat the same way can
    /// escalate); `calls`/`count` fill the nudge template.
    fn on_stuck(&mut self, stuck_key: String, calls: &[ToolCall], count: usize) -> LoopVerdict {
        // A repeat of the SAME stuck pattern always escalates to Stop, whatever the
        // policy — we already tried to recover from this exact thing and it recurred.
        let same_way = self.nudged_sig.as_deref() == Some(stuck_key.as_str());
        match self.recovery {
            Recovery::StopOnly => LoopVerdict::Stop("stuck — no progress".into()),
            Recovery::Reset => {
                if same_way || self.reset_count >= self.max_resets {
                    return LoopVerdict::Stop(
                        "still stuck after a fresh-context reset — not converging".into(),
                    );
                }
                self.nudged_sig = Some(stuck_key);
                self.reset_count += 1;
                LoopVerdict::Reset
            }
            Recovery::Nudge => {
                // Stop if same-way-again, or we've spent the nudge budget.
                if same_way || self.nudge_count >= self.max_nudges {
                    let why = if same_way {
                        "stuck the same way after a nudge — repeating tool calls without converging"
                    } else {
                        "kept thrashing across several nudges without finishing"
                    };
                    return LoopVerdict::Stop(why.into());
                }
                self.nudged_sig = Some(stuck_key);
                self.nudge_count += 1;
                let tool = calls
                    .first()
                    .map(|c| c.name.as_str())
                    .unwrap_or("that tool");
                let msg = self
                    .nudge_template
                    .replace("{tool}", tool)
                    .replace("{count}", &count.to_string());
                LoopVerdict::Nudge(msg)
            }
        }
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

    // progress through distinct, productive rounds (VARIED tools, distinct results)
    // never trips the guard — many rounds are fine as long as work is advancing.
    #[test]
    fn loop_guard_lets_real_progress_run() {
        let mut g = LoopGuard::new(100);
        let tools = ["read_file", "edit_file", "bash", "grep", "write_file"];
        for i in 0..20 {
            let t = tools[i % tools.len()]; // rotate tools → no single-name churn
            let calls = vec![ToolCall::new("c", t, format!("{{\"n\":{i}}}"))];
            let res = vec![format!("did {t} step {i}")];
            assert_eq!(g.record_round(&calls, &res), LoopVerdict::Continue);
        }
        assert_eq!(g.steps_taken(), 20); // 20 rounds, no false stop — step count is NOT the limit
    }

    // same (action,result) pair recurring → nudge; stuck the SAME way again → stop.
    #[test]
    fn loop_guard_nudges_then_stops_on_repeated_pair() {
        let mut g = LoopGuard::new(100);
        let calls = vec![ToolCall::new("c", "read_file", "{\"path\":\"x\"}")];
        let res = vec!["same content".to_string()];
        assert_eq!(g.record_round(&calls, &res), LoopVerdict::Continue); // 1st
        assert!(matches!(
            g.record_round(&calls, &res),
            LoopVerdict::Nudge(_)
        )); // 2nd: pair repeats
        assert!(matches!(g.record_round(&calls, &res), LoopVerdict::Stop(_))); // same way again → stop
    }

    // Recovery::Reset → first detection RESETS (clean-context restart); a fresh
    // DIFFERENT stuck pattern resets again up to max_resets, then Stops.
    #[test]
    fn loop_guard_reset_policy_resets_then_stops() {
        let mut g = LoopGuard::new(100).with_recovery(Recovery::Reset); // max_resets=2
                                                                        // identical action 3× → action_repeats hits repeat_limit(3) → reset #1.
        let a = vec![ToolCall::new(
            "c",
            "write_file",
            "{\"path\":\"g.py\",\"content\":\"v1\"}",
        )];
        assert_eq!(g.record_round(&a, &["w1".into()]), LoopVerdict::Continue);
        assert_eq!(g.record_round(&a, &["w2".into()]), LoopVerdict::Continue);
        assert_eq!(g.record_round(&a, &["w3".into()]), LoopVerdict::Reset);
        g.reset_window(); // caller clears after acting on Reset
                          // a DIFFERENT churn after reset → reset #2
        let b = vec![ToolCall::new("c", "bash", "{\"command\":\"x\"}")];
        for i in 0..4 {
            let v = g.record_round(&b, &[format!("r{i}")]);
            if v == LoopVerdict::Reset {
                g.reset_window();
                break;
            }
        }
        // a THIRD churn → max_resets exhausted → Stop
        let c = vec![ToolCall::new("c", "grep", "{\"q\":\"y\"}")];
        let mut stopped = false;
        for i in 0..6 {
            if matches!(g.record_round(&c, &[format!("g{i}")]), LoopVerdict::Stop(_)) {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "after max_resets the reset policy must Stop");
    }

    // Recovery::StopOnly → stop on the very first detection (no nudge, no reset).
    #[test]
    fn loop_guard_stop_only_policy_stops_immediately() {
        let mut g = LoopGuard::new(100).with_recovery(Recovery::StopOnly);
        let a = vec![ToolCall::new("c", "read_file", "{\"path\":\"x\"}")];
        let same = vec!["same".to_string()];
        assert_eq!(g.record_round(&a, &same), LoopVerdict::Continue);
        assert!(matches!(g.record_round(&a, &same), LoopVerdict::Stop(_))); // first stuck → stop
    }

    // FORGIVING: nudge, then the model makes progress → it runs free again; a later
    // unrelated stuck pattern earns a FRESH nudge, not an immediate stop.
    #[test]
    fn loop_guard_lets_a_recovered_turn_keep_running() {
        // exercise the forgiveness mechanism in isolation (prod ceiling is 1).
        let mut g = LoopGuard::new(100).with_max_nudges(5);
        let stuck_a = vec![ToolCall::new("c", "read_file", "{\"path\":\"x\"}")];
        let same = vec!["same".to_string()];
        assert_eq!(g.record_round(&stuck_a, &same), LoopVerdict::Continue);
        assert!(matches!(
            g.record_round(&stuck_a, &same),
            LoopVerdict::Nudge(_)
        )); // stuck on A
            // model recovers: distinct, productive rounds with VARIED tools (a run
            // of one tool would itself be churn now — recovery means real progress).
        let recover = ["edit_file", "bash", "grep", "read_file"];
        for (i, t) in recover.iter().enumerate() {
            let c = vec![ToolCall::new("c", *t, format!("{{\"n\":{i}}}"))];
            assert_eq!(
                g.record_round(&c, &[format!("did {t} {i}")]),
                LoopVerdict::Continue
            );
        }
        // now a DIFFERENT stuck pattern (B) → fresh nudge, NOT a stop
        let stuck_b = vec![ToolCall::new("c", "bash", "{\"command\":\"ls\"}")];
        let outb = vec!["files".to_string()];
        assert_eq!(g.record_round(&stuck_b, &outb), LoopVerdict::Continue);
        assert!(
            matches!(g.record_round(&stuck_b, &outb), LoopVerdict::Nudge(_)),
            "a fresh stuck pattern after recovery should nudge, not stop"
        );
    }

    // read→edit→read→edit churn on one file (non-identical calls) is caught by the
    // action-repeat signal — the gap the old exact-match guard missed (the 177k run).
    #[test]
    fn loop_guard_catches_alternating_wander() {
        let mut g = LoopGuard::new(100);
        let read = vec![ToolCall::new("c", "read_file", "{\"path\":\"g.py\"}")];
        let edit = vec![ToolCall::new("c", "edit_file", "{\"path\":\"g.py\"}")];
        // distinct results each round so the PAIR signal doesn't fire — only the
        // action-cycle / action-repeat signal should catch this.
        let mut verdict = LoopVerdict::Continue;
        for i in 0..6 {
            let calls = if i % 2 == 0 { &read } else { &edit };
            verdict = g.record_round(calls, &[format!("result {i}")]);
            if verdict != LoopVerdict::Continue {
                break;
            }
        }
        assert!(
            matches!(verdict, LoopVerdict::Nudge(_)),
            "alternating read/edit churn should be flagged as stuck"
        );
    }

    // "polish forever": repeated write_file with DIFFERENT content each time (so
    // action_sig differs every round) is caught by the tool-NAME frequency signal —
    // the real 53k-token failure the action-only signal missed.
    #[test]
    fn loop_guard_catches_same_tool_different_args_churn() {
        let mut g = LoopGuard::new(100);
        let mut verdict = LoopVerdict::Continue;
        for i in 0..6 {
            // every write distinct content + distinct result → only the name signal
            // (5× write_file in the window) can catch it.
            let calls = vec![ToolCall::new(
                "c",
                "write_file",
                format!("{{\"content\":\"v{i}\"}}"),
            )];
            verdict = g.record_round(&calls, &[format!("wrote v{i}")]);
            if verdict != LoopVerdict::Continue {
                break;
            }
        }
        assert!(
            matches!(verdict, LoopVerdict::Nudge(_)),
            "repeated write_file with different content should be flagged as churn"
        );
    }

    // OSCILLATION: a model that trips a stuck pattern, varies for ONE round, trips
    // again (different pattern), repeatedly — dodging the "same way twice" check —
    // still isn't converging. The max_nudges ceiling stops it. (The real 51k run.)
    #[test]
    fn loop_guard_stops_oscillating_thrash() {
        let mut g = LoopGuard::new(100);
        // Alternate: 3 writes (name-churn → nudge), 1 varied round, repeat. The
        // single varied round can't satisfy recovery_rounds(2), so strikes persist
        // and the nudge ceiling trips.
        let mut stopped = false;
        for i in 0..40 {
            let calls = if i % 4 == 3 {
                vec![ToolCall::new("c", "grep", format!("{{\"q\":{i}}}"))] // the "varied" round
            } else {
                vec![ToolCall::new("c", "write_file", format!("{{\"c\":{i}}}"))]
                // churn
            };
            if matches!(
                g.record_round(&calls, &[format!("r{i}")]),
                LoopVerdict::Stop(_)
            ) {
                stopped = true;
                break;
            }
        }
        assert!(
            stopped,
            "oscillating thrash should eventually Stop, not nudge forever"
        );
        assert!(
            g.steps_taken() < 40,
            "should stop well before any high backstop"
        );
    }

    // the catastrophe backstop still stops a loop that keeps emitting novel calls.
    #[test]
    fn loop_guard_backstop_is_last_resort() {
        let mut g = LoopGuard::new(5);
        let mut last = LoopVerdict::Continue;
        for i in 0..10 {
            // every call distinct (novel args) and distinct results → no stuck
            // signal; only the backstop can end it.
            let calls = vec![ToolCall::new(
                "c",
                format!("tool{i}"),
                format!("{{\"k\":{i}}}"),
            )];
            last = g.record_round(&calls, &[format!("ok {i}")]);
            if matches!(last, LoopVerdict::Stop(_)) {
                break;
            }
        }
        assert!(matches!(last, LoopVerdict::Stop(_)));
        assert!(g.steps_taken() <= 5);
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
    fn loop_guard_repeat_detection_is_content_independent() {
        // Property: a repeated identical (action,result) pair trips a nudge then a
        // same-way stop regardless of the call's exact name/args/result content.
        let mut rng = Rng(0x9999_1111_2222_3333);
        for _ in 0..2000 {
            let name = format!("t{}", rng.below(5));
            let args = format!("{{\"x\":{}}}", rng.below(5));
            let calls = vec![ToolCall::new("c", &name, &args)];
            let res = vec![format!("r{}", rng.below(5))];
            let mut g = LoopGuard::new(1000);
            assert_eq!(g.record_round(&calls, &res), LoopVerdict::Continue);
            assert!(matches!(
                g.record_round(&calls, &res),
                LoopVerdict::Nudge(_)
            ));
            assert!(matches!(g.record_round(&calls, &res), LoopVerdict::Stop(_)));
        }
    }
}
