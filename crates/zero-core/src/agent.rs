//! The agentic tool-call loop driver.
//!
//! Pure orchestration: it owns the send → detect-calls → execute → feed-back →
//! repeat cycle, but takes the model call and the tool execution as closures, so
//! it's fully unit-tested without a network or a filesystem. The frontend wires
//! the real [`crate::openai::OpenAiBackend::complete`] and a mode/safety-gated
//! executor into it.
//!
//! Embodies the researched invariants:
//!  * loop continues on PRESENCE of tool calls, never on `finish_reason`;
//!  * every assistant turn carrying calls is pushed to history verbatim, each
//!    answered by exactly one `tool` message with the matching id;
//!  * bounded by [`crate::tools::LoopGuard`] (step cap + doom-loop), so a
//!    misbehaving local model can't run forever.

use crate::backend::{BackendError, Completion};
use crate::message::{Conversation, Message, ToolCall};
use crate::tools::{LoopGuard, ToolDef};

/// What one finished turn produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnOutcome {
    /// The assistant's final text (the answer once it stopped calling tools).
    pub final_text: String,
    /// Why the loop stopped.
    pub stop: AgentStop,
    /// Number of tool-call rounds executed.
    pub rounds: usize,
    /// Server-reported token usage summed across every model round of this turn
    /// (so an agentic turn that took 3 round-trips reports all 3). Zero when the
    /// backend reported no usage (e.g. the stub).
    pub usage: crate::backend::Usage,
    /// `(bytes_before, bytes_after)` of write/edit args compacted DURING the turn
    /// (applied-write payloads stubbed each round so a long turn doesn't re-send a
    /// just-written file's body every model call). For the measured `/context` stat.
    pub compacted: (usize, usize),
}

/// Why the agent loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStop {
    /// The model produced a text answer with no further tool calls.
    Done,
    /// The step cap was hit (final text may be partial).
    MaxSteps,
    /// The same call repeated — broke out to avoid a doom loop.
    DoomLoop,
}

/// One step's interaction with the model, supplied by the caller.
pub trait Completer {
    fn complete(
        &mut self,
        conv: &Conversation,
        tools: &[ToolDef],
    ) -> Result<Completion, BackendError>;
}

impl<F> Completer for F
where
    F: FnMut(&Conversation, &[ToolDef]) -> Result<Completion, BackendError>,
{
    fn complete(
        &mut self,
        conv: &Conversation,
        tools: &[ToolDef],
    ) -> Result<Completion, BackendError> {
        self(conv, tools)
    }
}

/// Execute one tool call and return the result text fed back to the model. The
/// caller has already applied the mode/safety gate; returning an error string is
/// fine — it goes back as a tool result so the model can self-correct.
pub trait Executor {
    fn execute(&mut self, call: &ToolCall) -> String;
}

impl<F> Executor for F
where
    F: FnMut(&ToolCall) -> String,
{
    fn execute(&mut self, call: &ToolCall) -> String {
        self(call)
    }
}

/// Run the tool loop until the model answers with no tool calls (or a guard
/// trips). `conv` is mutated in place with the full transcript (assistant turns,
/// tool results) so the caller keeps the real history. `on_text` is invoked with
/// each assistant text chunk for display.
pub fn run_turn(
    conv: &mut Conversation,
    tools: &[ToolDef],
    completer: &mut dyn Completer,
    executor: &mut dyn Executor,
    guard: &mut LoopGuard,
    on_text: &mut dyn FnMut(&str),
) -> Result<TurnOutcome, BackendError> {
    let mut rounds = 0;
    let mut usage = crate::backend::Usage::default();
    let mut compacted = (0usize, 0usize);
    loop {
        let completion = completer.complete(conv, tools)?;
        // Accumulate server-reported tokens across every round of the turn.
        if let Some(u) = completion.usage {
            usage.prompt_tokens += u.prompt_tokens;
            usage.completion_tokens += u.completion_tokens;
        }

        // No tool calls → this is the final answer. Record it and finish.
        if completion.tool_calls.is_empty() {
            if !completion.content.is_empty() {
                on_text(&completion.content);
            }
            conv.push(Message::assistant(&completion.content));
            return Ok(TurnOutcome {
                final_text: completion.content,
                stop: AgentStop::Done,
                rounds,
                usage,
                compacted,
            });
        }

        // Surface any text the model emitted alongside its calls.
        if !completion.content.is_empty() {
            on_text(&completion.content);
        }

        // Doom-loop guard: the same calls repeating means we're stuck.
        if guard.is_doom_loop(&completion.tool_calls) {
            conv.push(Message::assistant_tool_calls(
                &completion.content,
                completion.tool_calls.clone(),
            ));
            // Answer each call so history stays well-formed before bailing.
            for call in &completion.tool_calls {
                conv.push(Message::tool_result(
                    &call.id,
                    "[aborted: repeated identical tool call]",
                ));
            }
            return Ok(TurnOutcome {
                final_text: completion.content,
                stop: AgentStop::DoomLoop,
                rounds,
                usage,
                compacted,
            });
        }

        // Record the assistant's tool-call turn verbatim, then execute each call
        // and append exactly one tool result per id (well-formed history).
        conv.push(Message::assistant_tool_calls(
            &completion.content,
            completion.tool_calls.clone(),
        ));
        for call in &completion.tool_calls {
            let result = executor.execute(call);
            conv.push(Message::tool_result(&call.id, result));
        }
        rounds += 1;

        // Mid-turn compaction: now that this round's writes have landed (their
        // results are in `conv`), stub their bulky applied payloads so the next
        // model round doesn't re-send a just-written file's full body. Idempotent,
        // so re-running it each round only ever touches newly-applied writes.
        let (b, a) = crate::context::compact_applied_writes(conv);
        compacted.0 += b;
        compacted.1 += a;

        // Step cap: stop before another model round.
        if !guard.record_step() {
            return Ok(TurnOutcome {
                final_text: completion.content,
                stop: AgentStop::MaxSteps,
                rounds,
                usage,
                compacted,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    fn text_completion(s: &str) -> Completion {
        Completion {
            content: s.to_string(),
            tool_calls: Vec::new(),
            usage: None,
        }
    }
    fn call_completion(name: &str) -> Completion {
        Completion {
            content: String::new(),
            tool_calls: vec![ToolCall::new("c1", name, "{}")],
            usage: None,
        }
    }

    #[test]
    fn finishes_immediately_on_text_answer() {
        let mut conv = Conversation::new();
        conv.push(Message::user("hi"));
        let mut completer = |_: &Conversation, _: &[ToolDef]| Ok(text_completion("hello"));
        let mut exec = |_: &ToolCall| String::new();
        let mut guard = LoopGuard::new(10);
        let mut seen = String::new();
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |t| seen.push_str(t),
        )
        .unwrap();
        assert_eq!(out.stop, AgentStop::Done);
        assert_eq!(out.rounds, 0);
        assert_eq!(out.final_text, "hello");
        assert_eq!(seen, "hello");
        assert_eq!(conv.messages.last().unwrap().role, Role::Assistant);
    }

    #[test]
    fn executes_one_round_then_answers() {
        let mut conv = Conversation::new();
        conv.push(Message::user("read it"));
        // First call returns a tool call; second returns text.
        let mut step = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            step += 1;
            Ok(if step == 1 {
                call_completion("read_file")
            } else {
                text_completion("the file says hi")
            })
        };
        let mut exec = |c: &ToolCall| format!("ran {}", c.name);
        let mut guard = LoopGuard::new(10);
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(out.stop, AgentStop::Done);
        assert_eq!(out.rounds, 1);
        // History: user, assistant(tool_calls), tool result, assistant(final).
        assert_eq!(conv.messages.len(), 4);
        assert_eq!(conv.messages[1].role, Role::Assistant);
        assert_eq!(conv.messages[1].tool_calls.len(), 1);
        assert_eq!(conv.messages[2].role, Role::Tool);
        assert_eq!(conv.messages[2].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(conv.messages[2].content, "ran read_file");
        assert_eq!(conv.messages[3].content, "the file says hi");
    }

    #[test]
    fn applied_write_payload_is_compacted_mid_turn() {
        // The agentic-build finding: a write_file's full content must NOT keep
        // riding along in history every subsequent round. After round 1 writes a
        // big file (and its result confirms success), run_turn stubs the payload so
        // round 2+ don't re-send the body. Verified in the final conversation +
        // the measured `compacted` counters.
        let big = "x".repeat(5000);
        let args = format!("{{\"path\":\"game.py\",\"content\":\"{big}\"}}");
        let mut step = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            step += 1;
            Ok(if step == 1 {
                Completion {
                    content: String::new(),
                    tool_calls: vec![ToolCall::new("w1", "write_file", &args)],
                    usage: None,
                }
            } else {
                text_completion("done")
            })
        };
        let mut exec = |_: &ToolCall| "{\"bytes_written\":5000}".to_string(); // success
        let mut guard = LoopGuard::new(10);
        let mut conv = Conversation::new();
        conv.push(Message::user("write the game"));
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(out.stop, AgentStop::Done);
        // The write_file call in history no longer carries the 5KB body.
        let wf = conv
            .messages
            .iter()
            .flat_map(|m| &m.tool_calls)
            .find(|tc| tc.name == "write_file")
            .expect("the write_file call");
        assert!(
            !wf.arguments.contains(&big),
            "full content still in history"
        );
        assert!(wf.arguments.contains("bytes applied"), "no compaction stub");
        // The saving was measured and rides back on the outcome.
        assert!(out.compacted.0 > out.compacted.1, "compaction not recorded");
        assert!(
            out.compacted.0 >= 5000,
            "should have compacted the 5KB body"
        );
    }

    #[test]
    fn failed_write_payload_is_kept_for_retry() {
        // A write whose result is an error keeps its args — the model needs them
        // to retry. Compaction only ever touches APPLIED writes.
        let big = "y".repeat(3000);
        let args = format!("{{\"path\":\"x.py\",\"content\":\"{big}\"}}");
        let mut step = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            step += 1;
            Ok(if step == 1 {
                Completion {
                    content: String::new(),
                    tool_calls: vec![ToolCall::new("w1", "write_file", &args)],
                    usage: None,
                }
            } else {
                text_completion("gave up")
            })
        };
        let mut exec = |_: &ToolCall| "error: permission denied".to_string();
        let mut guard = LoopGuard::new(10);
        let mut conv = Conversation::new();
        conv.push(Message::user("write it"));
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        let wf = conv
            .messages
            .iter()
            .flat_map(|m| &m.tool_calls)
            .find(|tc| tc.name == "write_file")
            .unwrap();
        assert!(
            wf.arguments.contains(&big),
            "failed write must keep its payload"
        );
        assert_eq!(out.compacted, (0, 0), "nothing should have been compacted");
    }

    #[test]
    fn every_call_gets_exactly_one_result() {
        let mut conv = Conversation::new();
        let mut step = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            step += 1;
            Ok(if step == 1 {
                Completion {
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall::new("a", "ls", "{}"),
                        ToolCall::new("b", "pwd", "{}"),
                    ],
                    usage: None,
                }
            } else {
                text_completion("done")
            })
        };
        let mut exec = |c: &ToolCall| format!("out:{}", c.id);
        let mut guard = LoopGuard::new(10);
        run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        let tool_msgs: Vec<_> = conv
            .messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .collect();
        assert_eq!(tool_msgs.len(), 2);
        assert_eq!(tool_msgs[0].tool_call_id.as_deref(), Some("a"));
        assert_eq!(tool_msgs[1].tool_call_id.as_deref(), Some("b"));
    }

    #[test]
    fn step_cap_stops_a_runaway() {
        let mut conv = Conversation::new();
        // Always returns a (varying) call so it never naturally stops.
        let mut n = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            n += 1;
            Ok(Completion {
                content: String::new(),
                tool_calls: vec![ToolCall::new(
                    format!("c{n}"),
                    "ls",
                    format!("{{\"n\":{n}}}"),
                )],
                usage: None,
            })
        };
        let mut exec = |_: &ToolCall| "ok".to_string();
        let mut guard = LoopGuard::new(3);
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(out.stop, AgentStop::MaxSteps);
        assert_eq!(out.rounds, 3);
    }

    #[test]
    fn doom_loop_is_broken() {
        let mut conv = Conversation::new();
        // Identical call every time → doom loop on the 3rd.
        let mut completer = |_: &Conversation, _: &[ToolDef]| Ok(call_completion("ls"));
        let mut exec = |_: &ToolCall| "same".to_string();
        let mut guard = LoopGuard::new(100);
        let out = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(out.stop, AgentStop::DoomLoop);
        // History still well-formed: last assistant has calls, each answered.
        let last_tool = conv
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Tool)
            .unwrap();
        assert!(last_tool.content.contains("aborted"));
    }

    #[test]
    fn backend_error_propagates() {
        let mut conv = Conversation::new();
        let mut completer = |_: &Conversation, _: &[ToolDef]| Err(BackendError("boom".to_string()));
        let mut exec = |_: &ToolCall| String::new();
        let mut guard = LoopGuard::new(10);
        let err = run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |_| {},
        )
        .unwrap_err();
        assert_eq!(err.0, "boom");
    }

    #[test]
    fn text_alongside_calls_is_surfaced() {
        let mut conv = Conversation::new();
        let mut step = 0;
        let mut completer = |_: &Conversation, _: &[ToolDef]| {
            step += 1;
            Ok(if step == 1 {
                Completion {
                    content: "let me check".to_string(),
                    tool_calls: vec![ToolCall::new("c1", "ls", "{}")],
                    usage: None,
                }
            } else {
                text_completion("done")
            })
        };
        let mut exec = |_: &ToolCall| "ok".to_string();
        let mut guard = LoopGuard::new(10);
        let mut seen = String::new();
        run_turn(
            &mut conv,
            &[],
            &mut completer,
            &mut exec,
            &mut guard,
            &mut |t| {
                seen.push_str(t);
                seen.push('|');
            },
        )
        .unwrap();
        assert_eq!(seen, "let me check|done|");
    }

    // --- property: the loop always terminates with well-formed history ----

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
    fn property_loop_terminates_and_history_is_well_formed() {
        // For a backend that emits random completions (text / 1-3 tool calls,
        // sometimes repeating), run_turn must ALWAYS:
        //   1. terminate (bounded by the guard) — the test itself hanging = fail,
        //   2. leave history where every assistant turn carrying tool_calls is
        //      followed by exactly one `tool` message per call id, in order.
        // This is the invariant the whole feature relies on for valid requests.
        for seed in 0..400u64 {
            let rng = std::cell::RefCell::new(Rng(0x1000 + seed));
            let mut conv = Conversation::new();
            conv.push(Message::user("go"));
            let mut completer = |_: &Conversation, _: &[ToolDef]| {
                let mut r = rng.borrow_mut();
                if r.below(3) == 0 {
                    return Ok(Completion {
                        content: "final".to_string(),
                        tool_calls: vec![],
                        usage: None,
                    });
                }
                let n = r.below(3) + 1;
                let calls = (0..n)
                    .map(|i| ToolCall::new(format!("c{i}"), format!("t{}", r.below(2)), "{}"))
                    .collect();
                Ok(Completion {
                    content: String::new(),
                    tool_calls: calls,
                    usage: None,
                })
            };
            let mut exec = |_: &ToolCall| "ok".to_string();
            let mut guard = LoopGuard::new(15);
            let out = run_turn(
                &mut conv,
                &[],
                &mut completer,
                &mut exec,
                &mut guard,
                &mut |_| {},
            )
            .unwrap();
            // Some terminal outcome was reached (no infinite loop).
            assert!(matches!(
                out.stop,
                AgentStop::Done | AgentStop::MaxSteps | AgentStop::DoomLoop
            ));
            // Well-formedness: each assistant message with tool_calls is followed
            // by exactly one Tool message per call id, in the same order.
            let msgs = &conv.messages;
            for (i, m) in msgs.iter().enumerate() {
                if m.role == Role::Assistant && !m.tool_calls.is_empty() {
                    let results = &msgs[i + 1..i + 1 + m.tool_calls.len()];
                    assert_eq!(
                        results.len(),
                        m.tool_calls.len(),
                        "seed {seed}: missing tool results"
                    );
                    for (call, res) in m.tool_calls.iter().zip(results) {
                        assert_eq!(res.role, Role::Tool);
                        assert_eq!(res.tool_call_id.as_deref(), Some(call.id.as_str()));
                    }
                }
            }
        }
    }
}
