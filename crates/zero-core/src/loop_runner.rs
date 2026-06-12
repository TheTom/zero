// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! The loop's decision core: a **pure** state machine `decide(input) -> Action`
//! that the scheduler and runner obey. All of the PRD's liveness contracts live
//! here as deterministic logic over (config, ledger summary, clock) — no I/O, no
//! model, no background runtime — so the nasty paths (deadline crossed, budget
//! gone, a wake that skipped its state write, the model quitting early) are
//! unit-tested against synthetic ledgers.
//!
//! Two decision points (PRD: *Loop liveness*):
//! - [`Event::Schedule`] — "should the next wake fire?" Checks pause, deadline,
//!   budget, and the per-wake state-write contract.
//! - [`Event::DoneClaim`] — the model claimed done; the harness ran the exit
//!   gate. A win stops; a false stop **revitalizes** (re-inject spec + the unmet
//!   criterion); a *repeated* false stop **escalates to the operator** rather than
//!   nudging forever. "I am done" is a claim the harness verifies, never an action.

use crate::loop_config::{LoopConfig, OnExhaust};
use crate::loop_ledger::LedgerSummary;

/// After this many consecutive done-claims, stop nudging and escalate to the
/// operator — a model that quits N times on the same evidence either is right or
/// needs information only the human has (PRD: operator signal beats autonomous
/// escalation). Tunable; the structural stop is what matters, not the exact N.
pub const REPEAT_STOP_ESCALATE: u64 = 3;

/// What prompted a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// The scheduler is deciding whether to fire the next wake.
    Schedule,
    /// A wake ended with the model claiming the loop is done; the harness ran the
    /// target/exit gate and reports whether it passed.
    DoneClaim { exit_gate_passed: bool },
}

/// Everything the decision needs — all of it pure data.
#[derive(Debug, Clone)]
pub struct TickInput<'a> {
    pub config: &'a LoopConfig,
    pub summary: &'a LedgerSummary,
    /// Current wall-clock in unix millis.
    pub now_ms: u64,
    /// The deadline resolved to unix millis (the caller parses the config's
    /// RFC3339 string; the pure machine only compares). `None` = no deadline.
    pub deadline_ms: Option<u64>,
    /// The operator paused this loop — overrides everything but is itself just a
    /// `Pause` outcome (the runner won't fire while paused).
    pub paused: bool,
    pub event: Event,
}

/// The action the runner must take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Fire the next wake.
    Wake,
    /// Stop firing, keep all state, surface the reason to the operator.
    Pause(PauseReason),
    /// End the loop gracefully (write a final row, tear down children, disarm).
    Stop(StopReason),
    /// Re-run after a false stop: re-inject the spec verbatim plus the specific
    /// unmet criterion. Carries the operator/model-facing message.
    Revitalize(String),
    /// A repeated false stop: ask the operator instead of nudging again.
    EscalateToHuman(String),
}

/// Why a loop paused (resumable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauseReason {
    /// The operator paused it.
    Operator,
    /// A wake ended without banking its required state row — the contract failed,
    /// so the loop pauses and flags rather than barrelling on with lost state.
    MissedStateWrite,
    /// A measured budget (wakes or tokens) was exhausted.
    BudgetExhausted,
}

/// Why a loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The absolute deadline passed.
    DeadlineReached,
    /// The exit gate passed — the goal is met.
    GoalMet,
    /// A budget was exhausted and `on_exhaust = "stop"`.
    BudgetExhausted,
}

/// The pure decision. Deterministic over its inputs; never panics, never does I/O.
pub fn decide(input: &TickInput) -> Action {
    match input.event {
        Event::Schedule => decide_schedule(input),
        Event::DoneClaim { exit_gate_passed } => decide_done(input, exit_gate_passed),
    }
}

/// "Should the next wake fire?" — pause/deadline/budget/contract checks, in
/// priority order (operator pause first, hard deadline next, then budget, then the
/// per-wake state-write contract).
fn decide_schedule(input: &TickInput) -> Action {
    if input.paused {
        return Action::Pause(PauseReason::Operator);
    }
    if let Some(dl) = input.deadline_ms {
        if input.now_ms >= dl {
            return Action::Stop(StopReason::DeadlineReached);
        }
    }
    if budget_exhausted(input) {
        return match input.config.budget.on_exhaust {
            OnExhaust::Stop => Action::Stop(StopReason::BudgetExhausted),
            OnExhaust::Pause => Action::Pause(PauseReason::BudgetExhausted),
        };
    }
    // Contract: a wake that didn't bank its state row pauses the loop (and flags)
    // rather than waking again with lost working state.
    if input.config.contract.require_state_append && !input.summary.last_state_written {
        return Action::Pause(PauseReason::MissedStateWrite);
    }
    Action::Wake
}

/// A measured budget is spent.
fn budget_exhausted(input: &TickInput) -> bool {
    let b = &input.config.budget;
    let s = input.summary;
    matches!(b.max_wakes, Some(m) if s.wakes >= m)
        || matches!(b.max_tokens, Some(m) if s.tokens_spent >= m)
}

/// Handle a done-claim: a passed exit gate stops; a false stop revitalizes, then
/// escalates once it repeats.
fn decide_done(input: &TickInput, exit_gate_passed: bool) -> Action {
    if exit_gate_passed {
        return Action::Stop(StopReason::GoalMet);
    }
    // This claim plus the trailing run already on the ledger.
    let total_claims = input.summary.consecutive_done_claims + 1;
    if total_claims >= REPEAT_STOP_ESCALATE {
        return Action::EscalateToHuman(escalation_message(input, total_claims));
    }
    Action::Revitalize(revitalize_message(input))
}

/// "exit gate not satisfied: <failing gates>; bar: <value>" — the specific unmet
/// criterion, so revitalization re-injects exactly what's missing (not a vague
/// "keep going").
fn revitalize_message(input: &TickInput) -> String {
    let mut msg = String::from("exit gate not satisfied");
    let failing: Vec<String> = input
        .summary
        .last_gates
        .iter()
        .filter(|g| !g.passed)
        .map(|g| format!("{} = {}", g.name, g.actual))
        .collect();
    if !failing.is_empty() {
        msg.push_str(": ");
        msg.push_str(&failing.join("; "));
    }
    if let Some(bar) = &input.config.bar {
        msg.push_str(&format!(" (bar: {})", bar.value));
    }
    msg
}

fn escalation_message(input: &TickInput, claims: u64) -> String {
    format!(
        "loop claimed done {claims}× with the same evidence{}; asking the operator \
         instead of nudging again",
        input
            .config
            .bar
            .as_ref()
            .map(|b| format!(" against bar {}", b.value))
            .unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loop_config::{Budget, Contract, LoopConfig};
    use crate::loop_ledger::{GateRecord, LedgerSummary};

    fn cfg() -> LoopConfig {
        LoopConfig::default()
    }

    fn input<'a>(
        config: &'a LoopConfig,
        summary: &'a LedgerSummary,
        event: Event,
    ) -> TickInput<'a> {
        TickInput {
            config,
            summary,
            now_ms: 1_000_000,
            deadline_ms: None,
            paused: false,
            event,
        }
    }

    #[test]
    fn fires_a_wake_under_normal_conditions() {
        let c = cfg();
        let s = LedgerSummary {
            wakes: 5,
            last_state_written: true,
            ..Default::default()
        };
        assert_eq!(decide(&input(&c, &s, Event::Schedule)), Action::Wake);
    }

    #[test]
    fn operator_pause_overrides_a_ready_wake() {
        let c = cfg();
        let s = LedgerSummary {
            last_state_written: true,
            ..Default::default()
        };
        let mut i = input(&c, &s, Event::Schedule);
        i.paused = true;
        assert_eq!(decide(&i), Action::Pause(PauseReason::Operator));
    }

    #[test]
    fn deadline_stops_the_loop() {
        let c = cfg();
        let s = LedgerSummary {
            last_state_written: true,
            ..Default::default()
        };
        let mut i = input(&c, &s, Event::Schedule);
        i.now_ms = 2_000;
        i.deadline_ms = Some(1_000); // already passed
        assert_eq!(decide(&i), Action::Stop(StopReason::DeadlineReached));
        // Not yet reached → still wakes.
        i.deadline_ms = Some(5_000);
        assert_eq!(decide(&i), Action::Wake);
    }

    #[test]
    fn budget_exhaustion_pauses_or_stops_per_config() {
        let mut c = cfg();
        c.budget = Budget {
            max_wakes: Some(10),
            max_tokens: None,
            on_exhaust: OnExhaust::Pause,
        };
        let s = LedgerSummary {
            wakes: 10,
            last_state_written: true,
            ..Default::default()
        };
        assert_eq!(
            decide(&input(&c, &s, Event::Schedule)),
            Action::Pause(PauseReason::BudgetExhausted)
        );
        c.budget.on_exhaust = OnExhaust::Stop;
        assert_eq!(
            decide(&input(&c, &s, Event::Schedule)),
            Action::Stop(StopReason::BudgetExhausted)
        );
    }

    #[test]
    fn token_budget_also_exhausts() {
        let mut c = cfg();
        c.budget.max_tokens = Some(1_000);
        let s = LedgerSummary {
            tokens_spent: 1_200,
            last_state_written: true,
            ..Default::default()
        };
        assert!(matches!(
            decide(&input(&c, &s, Event::Schedule)),
            Action::Pause(PauseReason::BudgetExhausted)
        ));
    }

    #[test]
    fn missed_state_write_pauses_and_flags() {
        let c = cfg(); // require_state_append defaults on
        let s = LedgerSummary {
            wakes: 3,
            last_state_written: false,
            ..Default::default()
        };
        assert_eq!(
            decide(&input(&c, &s, Event::Schedule)),
            Action::Pause(PauseReason::MissedStateWrite)
        );
    }

    #[test]
    fn missed_state_write_is_allowed_when_contract_is_off() {
        let mut c = cfg();
        c.contract = Contract {
            require_state_append: false,
            ..Default::default()
        };
        let s = LedgerSummary {
            last_state_written: false,
            ..Default::default()
        };
        assert_eq!(decide(&input(&c, &s, Event::Schedule)), Action::Wake);
    }

    #[test]
    fn deadline_takes_priority_over_budget_and_contract() {
        let mut c = cfg();
        c.budget.max_wakes = Some(1);
        let s = LedgerSummary {
            wakes: 99,
            last_state_written: false,
            ..Default::default()
        };
        let mut i = input(&c, &s, Event::Schedule);
        i.now_ms = 10;
        i.deadline_ms = Some(5);
        // Deadline wins over the budget/contract failures.
        assert_eq!(decide(&i), Action::Stop(StopReason::DeadlineReached));
    }

    #[test]
    fn done_claim_with_passing_exit_gate_stops_as_goal_met() {
        let c = cfg();
        let s = LedgerSummary::default();
        let i = input(
            &c,
            &s,
            Event::DoneClaim {
                exit_gate_passed: true,
            },
        );
        assert_eq!(decide(&i), Action::Stop(StopReason::GoalMet));
    }

    #[test]
    fn first_false_stop_revitalizes_with_the_unmet_criterion() {
        let mut c = cfg();
        c.bar = Some(crate::loop_config::Bar {
            value: "cosine 0.9987".into(),
            version: 1,
            conditions: vec![],
            remeasure: None,
        });
        let s = LedgerSummary {
            consecutive_done_claims: 0,
            last_gates: vec![GateRecord {
                name: "quality".into(),
                passed: false,
                actual: "0.943".into(),
            }],
            ..Default::default()
        };
        match decide(&input(
            &c,
            &s,
            Event::DoneClaim {
                exit_gate_passed: false,
            },
        )) {
            Action::Revitalize(msg) => {
                assert!(msg.contains("quality = 0.943"), "msg: {msg}");
                assert!(msg.contains("cosine 0.9987"), "bar missing: {msg}");
            }
            other => panic!("expected revitalize, got {other:?}"),
        }
    }

    #[test]
    fn repeated_false_stop_escalates_to_the_operator() {
        let c = cfg();
        // Two trailing claims already + this one = 3 → escalate.
        let s = LedgerSummary {
            consecutive_done_claims: REPEAT_STOP_ESCALATE - 1,
            ..Default::default()
        };
        match decide(&input(
            &c,
            &s,
            Event::DoneClaim {
                exit_gate_passed: false,
            },
        )) {
            Action::EscalateToHuman(msg) => assert!(msg.contains("operator")),
            other => panic!("expected escalation, got {other:?}"),
        }
        // One fewer trailing claim → still just revitalizes.
        let s2 = LedgerSummary {
            consecutive_done_claims: REPEAT_STOP_ESCALATE - 2,
            ..Default::default()
        };
        assert!(matches!(
            decide(&input(
                &c,
                &s2,
                Event::DoneClaim {
                    exit_gate_passed: false
                }
            )),
            Action::Revitalize(_)
        ));
    }

    #[test]
    fn passing_exit_gate_stops_even_after_repeated_claims() {
        // A real win ends the loop regardless of prior false stops.
        let c = cfg();
        let s = LedgerSummary {
            consecutive_done_claims: 9,
            ..Default::default()
        };
        assert_eq!(
            decide(&input(
                &c,
                &s,
                Event::DoneClaim {
                    exit_gate_passed: true
                }
            )),
            Action::Stop(StopReason::GoalMet)
        );
    }
}
