// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! `ticks.jsonl` — the harness-written loop ledger: one JSON row per wake (real
//! timestamp, measured tokens + elapsed, gate results, whether the wake banked a
//! state row, whether it claimed done). The pure [`crate::loop_runner`] state
//! machine decides the next action from a [`LedgerSummary`] derived here.
//!
//! Append-only and crash-safe without a DB: each row is a full line written then
//! `sync_data`'d; a torn trailing line (power loss mid-append) is detected and
//! skipped on load. JSON is serialized through the std-only [`crate::json`].
//! ("RAM is a cache of the JSONL" — the in-memory rows rebuild from disk on open.)

use crate::clock::unix_millis;
use crate::json::Value;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

/// One gate's result as recorded in a tick row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateRecord {
    pub name: String,
    pub passed: bool,
    /// The measured value, for citation (e.g. `"0.9942"`).
    pub actual: String,
}

/// One wake's record.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TickRow {
    pub ts_ms: u64,
    /// Sequential wake number (1-based).
    pub wake: u64,
    /// Tokens spent this wake (measured, server-reported).
    pub tokens: u64,
    pub elapsed_ms: u64,
    pub gates: Vec<GateRecord>,
    /// Did the wake bank a state row before ending? (contract enforcement)
    pub state_written: bool,
    /// Did the model claim the loop is done this wake?
    pub done_claimed: bool,
    /// The NEXT ACTION the wake banked (empty if none) — used to detect a loop
    /// that is *present but not progressing* (the same next step every wake).
    pub next_action: String,
    /// Freeform note (revitalized / escalated / compacted / backend error …).
    pub note: String,
}

impl TickRow {
    fn to_value(&self) -> Value {
        let gates = self
            .gates
            .iter()
            .map(|g| {
                Value::Object(vec![
                    ("name".into(), Value::Str(g.name.clone())),
                    ("passed".into(), Value::Bool(g.passed)),
                    ("actual".into(), Value::Str(g.actual.clone())),
                ])
            })
            .collect();
        Value::Object(vec![
            ("ts_ms".into(), Value::Num(self.ts_ms as f64)),
            ("wake".into(), Value::Num(self.wake as f64)),
            ("tokens".into(), Value::Num(self.tokens as f64)),
            ("elapsed_ms".into(), Value::Num(self.elapsed_ms as f64)),
            ("gates".into(), Value::Array(gates)),
            ("state_written".into(), Value::Bool(self.state_written)),
            ("done_claimed".into(), Value::Bool(self.done_claimed)),
            ("next_action".into(), Value::Str(self.next_action.clone())),
            ("note".into(), Value::Str(self.note.clone())),
        ])
    }

    fn from_value(v: &Value) -> Option<TickRow> {
        let num = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0) as u64;
        let boolean = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
        let gates = v
            .get("gates")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|g| {
                        Some(GateRecord {
                            name: g.get("name")?.as_str()?.to_string(),
                            passed: g.get("passed").and_then(Value::as_bool).unwrap_or(false),
                            actual: g
                                .get("actual")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Some(TickRow {
            ts_ms: num("ts_ms"),
            wake: num("wake"),
            tokens: num("tokens"),
            elapsed_ms: num("elapsed_ms"),
            gates,
            state_written: boolean("state_written"),
            done_claimed: boolean("done_claimed"),
            next_action: v
                .get("next_action")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            note: v
                .get("note")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        })
    }

    /// Stamp `ts_ms` with the real clock if unset (0). Used by `append`.
    fn stamped(mut self) -> Self {
        if self.ts_ms == 0 {
            self.ts_ms = unix_millis() as u64;
        }
        self
    }
}

/// The append-only tick ledger for one loop. Holds the in-memory rows (a cache of
/// the on-disk JSONL, rebuilt on [`open`](Ledger::open)).
pub struct Ledger {
    path: PathBuf,
    rows: Vec<TickRow>,
}

impl Ledger {
    /// Open (or create) the ledger at `path`, loading existing rows. A torn
    /// trailing line from a crashed append is skipped, not fatal.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Ledger> {
        let path = path.into();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let rows = match fs::read_to_string(&path) {
            Ok(text) => parse_rows(&text),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e),
        };
        Ok(Ledger { path, rows })
    }

    /// Append one wake's row: write the full line + `sync_data` (so a crash leaves
    /// either the whole row or none), then cache it in memory. Stamps the
    /// timestamp from the real clock when the caller left it 0.
    pub fn append(&mut self, row: TickRow) -> io::Result<()> {
        let row = row.stamped();
        let mut line = row.to_value().to_json();
        line.push('\n');
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        f.write_all(line.as_bytes())?;
        f.sync_data()?;
        self.rows.push(row);
        Ok(())
    }

    /// The recorded rows, oldest first.
    pub fn rows(&self) -> &[TickRow] {
        &self.rows
    }

    /// Derive the summary the state machine reasons over.
    pub fn summary(&self) -> LedgerSummary {
        summarize(&self.rows)
    }
}

/// What the pure state machine needs to decide the next action — derivable from
/// the ledger with no I/O (so the decision logic is unit-tested with no files).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LedgerSummary {
    /// Number of wakes recorded.
    pub wakes: u64,
    /// Total measured tokens across all wakes.
    pub tokens_spent: u64,
    /// Did the most recent wake bank a state row? (`true` when there are no rows.)
    pub last_state_written: bool,
    /// Trailing run of consecutive done-claims (drives repeated-stop escalation).
    pub consecutive_done_claims: u64,
    /// Trailing run of wakes that banked the **same** non-empty NEXT ACTION —
    /// "present but not progressing". Drives a no-progress pause.
    pub consecutive_repeat_next_action: u64,
    /// The most recent wake's gate results (empty if none).
    pub last_gates: Vec<GateRecord>,
}

impl LedgerSummary {
    /// All of the most recent wake's gates passed (and there was at least one).
    pub fn last_gates_all_passed(&self) -> bool {
        !self.last_gates.is_empty() && self.last_gates.iter().all(|g| g.passed)
    }
}

/// Compute a [`LedgerSummary`] from rows. Public for the runner's tests.
pub fn summarize(rows: &[TickRow]) -> LedgerSummary {
    let tokens_spent = rows.iter().map(|r| r.tokens).sum();
    let last_state_written = rows.last().map(|r| r.state_written).unwrap_or(true);
    let consecutive_done_claims = rows.iter().rev().take_while(|r| r.done_claimed).count() as u64;
    // Trailing run of identical, non-empty next actions = present but not progressing.
    let consecutive_repeat_next_action = match rows.last() {
        Some(last) if !last.next_action.trim().is_empty() => rows
            .iter()
            .rev()
            .take_while(|r| r.next_action == last.next_action)
            .count() as u64,
        _ => 0,
    };
    let last_gates = rows.last().map(|r| r.gates.clone()).unwrap_or_default();
    LedgerSummary {
        wakes: rows.len() as u64,
        tokens_spent,
        last_state_written,
        consecutive_done_claims,
        consecutive_repeat_next_action,
        last_gates,
    }
}

/// Parse JSONL rows, skipping a blank or torn (unparseable) trailing/any line.
fn parse_rows(text: &str) -> Vec<TickRow> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| Value::parse(l).ok().and_then(|v| TickRow::from_value(&v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "zero-ledger-{}-{}-{tag}",
            std::process::id(),
            unix_millis()
        ))
    }

    fn row(wake: u64, tokens: u64) -> TickRow {
        TickRow {
            wake,
            tokens,
            elapsed_ms: 100,
            state_written: true,
            ..Default::default()
        }
    }

    #[test]
    fn append_then_reopen_roundtrips() {
        let dir = tmp("rt");
        let path = dir.join("ticks.jsonl");
        {
            let mut l = Ledger::open(&path).unwrap();
            let mut r = row(1, 500);
            r.gates = vec![GateRecord {
                name: "quality".into(),
                passed: false,
                actual: "0.99".into(),
            }];
            l.append(r).unwrap();
            l.append(row(2, 700)).unwrap();
            assert_eq!(l.rows().len(), 2);
        }
        // Reopen reads the rows back, byte-roundtripped.
        let l = Ledger::open(&path).unwrap();
        assert_eq!(l.rows().len(), 2);
        assert_eq!(l.rows()[0].wake, 1);
        assert_eq!(l.rows()[0].gates[0].actual, "0.99");
        assert_eq!(l.rows()[1].tokens, 700);
        assert!(l.rows()[0].ts_ms > 1_577_836_800_000); // real stamp
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn torn_trailing_line_is_skipped_on_load() {
        let dir = tmp("torn");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ticks.jsonl");
        // Two good rows + a half-written third (crash mid-append).
        let good = row(1, 100).to_value().to_json();
        let good2 = row(2, 200).to_value().to_json();
        fs::write(&path, format!("{good}\n{good2}\n{{\"ts_ms\":3,\"wa")).unwrap();
        let l = Ledger::open(&path).unwrap();
        assert_eq!(l.rows().len(), 2, "torn line must be dropped");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn summary_tallies_tokens_wakes_and_state() {
        let rows = vec![row(1, 100), row(2, 250), row(3, 50)];
        let s = summarize(&rows);
        assert_eq!(s.wakes, 3);
        assert_eq!(s.tokens_spent, 400);
        assert!(s.last_state_written);
        assert_eq!(s.consecutive_done_claims, 0);
    }

    #[test]
    fn summary_counts_repeated_next_action_as_no_progress() {
        let mut rows = vec![row(1, 10), row(2, 10), row(3, 10), row(4, 10)];
        rows[0].next_action = "different".into();
        for r in &mut rows[1..] {
            r.next_action = "instrument the qkv bucket".into(); // same 3 wakes running
        }
        let s = summarize(&rows);
        assert_eq!(s.consecutive_repeat_next_action, 3);
        // An empty next action doesn't count as a stall.
        let mut blank = vec![row(1, 10)];
        blank[0].next_action = String::new();
        assert_eq!(summarize(&blank).consecutive_repeat_next_action, 0);
    }

    #[test]
    fn summary_counts_trailing_done_claims_only() {
        let mut rows = vec![row(1, 10), row(2, 10), row(3, 10), row(4, 10)];
        rows[0].done_claimed = true; // an old, isolated claim — not part of the run
        rows[2].done_claimed = true;
        rows[3].done_claimed = true;
        let s = summarize(&rows);
        assert_eq!(s.consecutive_done_claims, 2, "only the trailing run counts");
    }

    #[test]
    fn empty_ledger_summary_defaults() {
        let s = summarize(&[]);
        assert_eq!(s.wakes, 0);
        assert_eq!(s.tokens_spent, 0);
        assert!(s.last_state_written, "no rows ⇒ not a missed write");
        assert!(!s.last_gates_all_passed());
    }

    #[test]
    fn missing_state_write_is_visible_in_summary() {
        let mut r = row(1, 100);
        r.state_written = false;
        let s = summarize(std::slice::from_ref(&r));
        assert!(!s.last_state_written);
    }

    #[test]
    fn last_gates_all_passed_helper() {
        let mut r = row(1, 100);
        r.gates = vec![
            GateRecord {
                name: "a".into(),
                passed: true,
                actual: "1".into(),
            },
            GateRecord {
                name: "b".into(),
                passed: true,
                actual: "ok".into(),
            },
        ];
        assert!(summarize(std::slice::from_ref(&r)).last_gates_all_passed());
        r.gates[1].passed = false;
        assert!(!summarize(std::slice::from_ref(&r)).last_gates_all_passed());
    }
}
