// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! One wake of a loop, orchestrated by the harness. A wake is a **fresh context**
//! assembled from disk (spec + rules + recent state), one model call, then the
//! harness — not the model — runs the gates and banks the evidence. The model
//! never claims a win; it cites gate results the harness measured.
//!
//! [`run_wake`] is the integration point over three seams so it's deterministic in
//! tests: the [`crate::backend::Backend`] (the model), a [`GateRunner`] (runs a
//! gate's command — `sh -c` in production, a fake in tests), and a
//! [`crate::loop_store::LoopStore`] (disk). The pure next-step decision stays in
//! [`crate::loop_runner`]; this module does the I/O and records the tick.

use crate::backend::Backend;
use crate::gate::{self, GateKind};
use crate::loop_config::LoopConfig;
use crate::loop_ledger::{GateRecord, TickRow};
use crate::loop_store::{LoopStore, StateRow};
use crate::message::{Conversation, Message};
use std::io;
use std::time::Duration;

/// How many recent state rows ride in the wake prompt (the capped "state tail").
pub const STATE_TAIL: usize = 6;
const WAKE_TIMEOUT: Duration = Duration::from_secs(180);

/// The model-facing contract reminder appended to every wake prompt.
const CONTRACT_REMINDER: &str = "\
End your reply with a state row the harness will bank:\n\
  STATE: <one or two lines of working notes — what you tried, what you found>\n\
  NEXT ACTION: <the single next step>\n\
Only write `LOOP DONE` if the measured bar is met — the harness runs the gates and \
will reject an unproven claim.";

/// Runs a gate's command and returns its `(combined_output, exit_code)`. The
/// frontend wires `sh -c`; tests pass a fake. (Kept a trait so a closure works.)
pub trait GateRunner {
    fn run(&mut self, command: &str) -> (String, i32);
}

impl<F: FnMut(&str) -> (String, i32)> GateRunner for F {
    fn run(&mut self, command: &str) -> (String, i32) {
        self(command)
    }
}

/// The result of one wake — what the caller (scheduler) reasons over next.
#[derive(Debug, Clone)]
pub struct WakeOutcome {
    pub wake: u64,
    /// The model's raw reply.
    pub reply: String,
    /// Did the model claim the loop is done this wake?
    pub done_claimed: bool,
    /// Did **every** command gate pass (and was there at least one)? This is the
    /// exit-gate signal for a done-claim — an unmeasured loop can never be "done".
    pub gates_all_passed: bool,
    /// The tick row the harness banked.
    pub tick: TickRow,
}

/// Assemble the wake prompt: spec (the mission), distilled rules, the recent state
/// tail, and the contract reminder — in that order (rules are the cheapest,
/// highest-value tokens, so they ride every wake). Pure and unit-tested.
pub fn assemble_wake_prompt(
    spec: &str,
    rules: &str,
    state_tail: &[StateRow],
    config: &LoopConfig,
) -> String {
    let mut p = String::new();
    p.push_str("# MISSION (spec)\n");
    p.push_str(spec.trim());
    if !rules.trim().is_empty() {
        p.push_str("\n\n# RULES (verified — always apply)\n");
        p.push_str(rules.trim());
    }
    if let Some(bar) = &config.bar {
        p.push_str(&format!("\n\n# BAR (the only target)\n{}", bar.value));
    }
    if !state_tail.is_empty() {
        p.push_str("\n\n# RECENT STATE (your prior wakes)\n");
        for r in state_tail {
            p.push_str(&format!(
                "- wake {}: {} → NEXT: {}\n",
                r.wake,
                first_line(&r.body),
                r.next_action
            ));
        }
    }
    p.push_str("\n\n# THIS WAKE\nRun one iteration toward the mission. ");
    p.push_str(CONTRACT_REMINDER);
    p
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("").trim()
}

/// Run one wake: assemble the prompt, call the model, run the command gates, bank a
/// state row + a tick row. `wake` is this wake's 1-based number, `now_ms` the
/// wall-clock for the tick. The exit-gate verdict and the next-step decision are
/// the caller's job (via [`crate::loop_runner::decide`]).
pub fn run_wake(
    store: &LoopStore,
    backend: &dyn Backend,
    gates: &mut dyn GateRunner,
    wake: u64,
    now_ms: u64,
) -> io::Result<WakeOutcome> {
    let config = store.config()?;
    let sw = crate::clock::Stopwatch::start();

    // 1. Fresh context from disk.
    let prompt = assemble_wake_prompt(
        &store.spec(),
        &store.rules(),
        &store.state_tail(STATE_TAIL),
        &config,
    );
    let mut conv = Conversation::new();
    conv.push(Message::user(prompt));

    // 2. One model call.
    let completion = backend
        .complete(&conv, &[], WAKE_TIMEOUT)
        .map_err(|e| io::Error::other(format!("backend: {e}")))?;
    let reply = completion.content;
    let tokens = completion.usage.map(|u| u.total()).unwrap_or(0);

    // 3. The harness runs the command gates (rubric gates are judged elsewhere).
    let mut gate_records = Vec::new();
    for g in &config.gates {
        if g.kind == GateKind::Command && !g.run.trim().is_empty() {
            let (out, code) = gates.run(&g.run);
            let o = gate::evaluate(g, &out, code);
            gate_records.push(GateRecord {
                name: o.name,
                passed: o.passed,
                actual: o.actual,
            });
        }
    }
    let gates_all_passed = !gate_records.is_empty() && gate_records.iter().all(|g| g.passed);

    // 4. Extract the model's state row + done-claim from its reply.
    let parsed = parse_wake_reply(&reply);
    let state_written = parsed.next_action.is_some();
    if state_written || !parsed.body.trim().is_empty() {
        let _ = store.append_state(&StateRow {
            wake,
            body: parsed.body.clone(),
            next_action: parsed.next_action.clone().unwrap_or_default(),
        });
    }

    // 5. Bank the tick.
    let tick = TickRow {
        ts_ms: now_ms,
        wake,
        tokens,
        elapsed_ms: sw.elapsed().as_millis() as u64,
        gates: gate_records,
        state_written,
        done_claimed: parsed.done_claimed,
        note: String::new(),
    };
    store.ledger()?.append(tick.clone())?;

    Ok(WakeOutcome {
        wake,
        reply,
        done_claimed: parsed.done_claimed,
        gates_all_passed,
        tick,
    })
}

/// What the harness reads back out of a wake's reply.
struct ParsedReply {
    body: String,
    next_action: Option<String>,
    done_claimed: bool,
}

/// Extract the state row (STATE / NEXT ACTION) and the done-claim (`LOOP DONE`)
/// from a model reply. Tolerant of casing and of the model omitting the markers
/// (in which case `next_action` is `None` → the harness records a missed state
/// write and pauses, rather than barrelling on).
fn parse_wake_reply(reply: &str) -> ParsedReply {
    let done_claimed = reply
        .lines()
        .any(|l| l.trim().to_ascii_uppercase().starts_with("LOOP DONE"));

    let mut next_action = None;
    let mut body_lines = Vec::new();
    for line in reply.lines() {
        let t = line.trim();
        let upper = t.to_ascii_uppercase();
        if let Some(rest) = upper.strip_prefix("NEXT ACTION:") {
            // Preserve original casing of the value.
            let val = t[t.len() - rest.len()..].trim();
            next_action = Some(val.to_string());
        } else if upper.starts_with("STATE:") {
            body_lines.push(t["STATE:".len()..].trim());
        } else if !upper.starts_with("LOOP DONE") {
            body_lines.push(t);
        }
    }
    let body = body_lines
        .iter()
        .filter(|l| !l.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    ParsedReply {
        body,
        next_action,
        done_claimed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendError, Completion, StopReason, StreamEvent, Usage};
    use crate::loop_runner::{decide, Action, Event, StopReason as RunStop, TickInput};
    use std::path::PathBuf;
    use std::sync::Mutex;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "zero-looprun-{}-{}-{tag}",
            std::process::id(),
            crate::clock::unix_millis()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A backend that returns one canned reply per wake.
    struct ScriptBackend {
        replies: Mutex<Vec<String>>,
    }
    impl ScriptBackend {
        fn new(replies: &[&str]) -> Self {
            ScriptBackend {
                replies: Mutex::new(replies.iter().rev().map(|s| s.to_string()).collect()),
            }
        }
    }
    impl Backend for ScriptBackend {
        fn name(&self) -> &str {
            "script"
        }
        fn stream(
            &self,
            _c: &Conversation,
            s: &mut dyn FnMut(StreamEvent),
        ) -> Result<(), BackendError> {
            s(StreamEvent::Done(StopReason::EndTurn));
            Ok(())
        }
        fn complete(
            &self,
            _c: &Conversation,
            _t: &[crate::tools::ToolDef],
            _to: Duration,
        ) -> Result<Completion, BackendError> {
            let content = self.replies.lock().unwrap().pop().unwrap_or_default();
            Ok(Completion {
                content,
                tool_calls: vec![],
                usage: Some(Usage {
                    prompt_tokens: 100,
                    completion_tokens: 20,
                }),
            })
        }
    }

    const TOML: &str =
        "[bar]\nvalue = \"cosine >= 0.99\"\nversion = 1\n[[gate]]\nname=\"quality\"\nrun=\"measure\"\nparse=\"json:.cosine\"\npass=\">= 0.99\"\n";

    #[test]
    fn assemble_prompt_orders_spec_rules_bar_state() {
        let cfg = LoopConfig::parse(TOML).unwrap();
        let tail = vec![StateRow {
            wake: 1,
            body: "tried scalar tuning".into(),
            next_action: "fuse qkv".into(),
        }];
        let p = assemble_wake_prompt("the mission", "always measure", &tail, &cfg);
        let i_spec = p.find("the mission").unwrap();
        let i_rules = p.find("always measure").unwrap();
        let i_bar = p.find("cosine >= 0.99").unwrap();
        let i_state = p.find("fuse qkv").unwrap();
        assert!(i_spec < i_rules && i_rules < i_bar && i_bar < i_state);
        assert!(p.contains("NEXT ACTION:"));
    }

    #[test]
    fn a_winning_wake_records_a_passing_gate_and_stops() {
        let root = tmp("win");
        let store = LoopStore::at(&root, "perf");
        store.create("push cosine to 0.99", TOML, "").unwrap();
        let backend =
            ScriptBackend::new(&["STATE: fused qkv, looks good\nNEXT ACTION: verify\nLOOP DONE"]);
        // The harness gate measures a passing cosine.
        let mut runner = |_cmd: &str| (r#"{"cosine": 0.995}"#.to_string(), 0);
        let out = run_wake(&store, &backend, &mut runner, 1, 1000).unwrap();

        assert!(out.done_claimed);
        assert!(out.gates_all_passed, "gate should pass at 0.995");
        assert_eq!(out.tick.gates[0].actual, "0.995");
        // The done-claim with a passing exit gate → GoalMet.
        let cfg = store.config().unwrap();
        let summary = store.ledger().unwrap().summary();
        let action = decide(&TickInput {
            config: &cfg,
            summary: &summary,
            now_ms: 1000,
            deadline_ms: None,
            paused: false,
            event: Event::DoneClaim {
                exit_gate_passed: out.gates_all_passed,
            },
        });
        assert_eq!(action, Action::Stop(RunStop::GoalMet));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_false_done_claim_does_not_stop_the_loop() {
        // The model claims done but the measured gate FAILS — the harness must not
        // let it quit (the #1 long-run failure: premature stops on unmeasured wins).
        let root = tmp("false");
        let store = LoopStore::at(&root, "perf");
        store.create("push cosine to 0.99", TOML, "").unwrap();
        let backend =
            ScriptBackend::new(&["STATE: i think it's done\nNEXT ACTION: stop\nLOOP DONE"]);
        let mut runner = |_cmd: &str| (r#"{"cosine": 0.91}"#.to_string(), 0);
        let out = run_wake(&store, &backend, &mut runner, 1, 1000).unwrap();
        assert!(out.done_claimed);
        assert!(!out.gates_all_passed, "0.91 < 0.99 must fail the gate");

        let cfg = store.config().unwrap();
        let summary = store.ledger().unwrap().summary();
        let action = decide(&TickInput {
            config: &cfg,
            summary: &summary,
            now_ms: 1000,
            deadline_ms: None,
            paused: false,
            event: Event::DoneClaim {
                exit_gate_passed: out.gates_all_passed,
            },
        });
        // Not a stop — a revitalize carrying the unmet criterion.
        assert!(matches!(action, Action::Revitalize(_)), "got {action:?}");
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_wake_with_no_state_row_is_recorded_as_a_missed_write() {
        let root = tmp("nostate");
        let store = LoopStore::at(&root, "x");
        store.create("mission", TOML, "").unwrap();
        // The model ignores the contract — no NEXT ACTION.
        let backend = ScriptBackend::new(&["just rambling, no state row here"]);
        let mut runner = |_cmd: &str| ("{}".to_string(), 0);
        let out = run_wake(&store, &backend, &mut runner, 1, 1000).unwrap();
        assert!(
            !out.tick.state_written,
            "no NEXT ACTION ⇒ missed state write"
        );

        // The next schedule decision pauses-and-flags rather than waking again.
        let cfg = store.config().unwrap();
        let summary = store.ledger().unwrap().summary();
        assert!(matches!(
            decide(&TickInput {
                config: &cfg,
                summary: &summary,
                now_ms: 1000,
                deadline_ms: None,
                paused: false,
                event: Event::Schedule,
            }),
            Action::Pause(crate::loop_runner::PauseReason::MissedStateWrite)
        ));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn wake_banks_state_and_tick_to_disk() {
        let root = tmp("bank");
        let store = LoopStore::at(&root, "x");
        store.create("mission", TOML, "").unwrap();
        let backend = ScriptBackend::new(&["STATE: did a thing\nNEXT ACTION: do the next thing"]);
        let mut runner = |_cmd: &str| (r#"{"cosine": 0.5}"#.to_string(), 0);
        run_wake(&store, &backend, &mut runner, 1, 1000).unwrap();
        // State row landed.
        let rows = store.state_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].next_action, "do the next thing");
        assert!(rows[0].body.contains("did a thing"));
        // Tick landed with measured tokens.
        let led = store.ledger().unwrap();
        assert_eq!(led.rows().len(), 1);
        assert_eq!(led.rows()[0].tokens, 120);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn parse_reply_is_case_insensitive_and_tolerant() {
        let p = parse_wake_reply("blah\nnext action: go left\nloop done");
        assert!(p.done_claimed);
        assert_eq!(p.next_action.as_deref(), Some("go left"));
        let none = parse_wake_reply("no markers at all");
        assert!(!none.done_claimed);
        assert!(none.next_action.is_none());
    }
}
