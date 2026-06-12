// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! `loop.toml` — the machine half of a loop: schedule, the measured bar, the
//! per-wake contract, gates, budget, authority bounds, and an optional launch
//! trigger. This is the part the harness *executes and never delegates*.
//!
//! TOML is not in `std`, and the PRD's "zero runtime dependencies" rule forbids a
//! crate, so this is a **small hand-rolled subset reader** — exactly the cases the
//! loop config uses: `[tables]`, `[[arrays-of-tables]]`, `key = value` where a
//! value is a string, integer, bool, array-of-strings, or single-line inline
//! table. `#` comments and blank lines are skipped. Parsing is total — a malformed
//! file returns `Err(reason)`, never a panic.

use crate::gate::Gate;
use std::time::Duration;

/// A parsed `loop.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LoopConfig {
    pub schedule: Schedule,
    pub bar: Option<Bar>,
    pub contract: Contract,
    pub gates: Vec<Gate>,
    pub budget: Budget,
    pub authority: Authority,
    pub trigger: Option<Trigger>,
}

/// `[schedule]` — when a *running* loop wakes (distinct from a launch trigger).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Schedule {
    /// Fallback wake interval, e.g. `"30m"` (see [`parse_duration`]).
    pub heartbeat: Option<String>,
    /// Event-driven wake conditions (`["job:done"]`); v1 approximates via heartbeat.
    pub wake_on: Vec<String>,
    /// Absolute deadline (RFC3339-ish string), harness-enforced.
    pub deadline: Option<String>,
}

/// `[bar]` — the target as a measured, versioned artifact (never a bare number).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bar {
    pub value: String,
    pub version: u32,
    /// The conditions the bar was measured under (`seq`, `clocks`, …).
    pub conditions: Vec<(String, String)>,
    pub remeasure: Option<String>,
}

/// `[contract]` — enforced per wake. Defaults are all-on (the safe contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contract {
    pub inject_spec: bool,
    pub require_state_append: bool,
    pub require_next_action: bool,
}

impl Default for Contract {
    fn default() -> Self {
        Contract {
            inject_spec: true,
            require_state_append: true,
            require_next_action: true,
        }
    }
}

/// What to do when a budget is exhausted — never a silent stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnExhaust {
    #[default]
    Pause,
    Stop,
}

/// `[budget]` — measured limits (tokens are real, not estimated).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Budget {
    pub max_wakes: Option<u64>,
    pub max_tokens: Option<u64>,
    pub on_exhaust: OnExhaust,
}

/// `[authority]` — concrete bounds on what a 3 a.m. run may do.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Authority {
    /// Surfaces the loop may push to (e.g. `github.com/org/repo#20`).
    pub allow_push: Vec<String>,
    /// Per-host restart caps (`{ spark = 3, train = 3 }`).
    pub max_restarts: Vec<(String, u32)>,
}

/// What to do when a scheduled fire was missed (zero was down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnMiss {
    #[default]
    Skip,
    RunNow,
    Ask,
}

/// What to do when a prior run is still active at fire time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Overlap {
    #[default]
    Skip,
    Queue,
    Parallel,
}

/// `[trigger]` — when the loop *launches* (distinct from `[schedule]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trigger {
    pub when: String,
    pub on_miss: OnMiss,
    pub overlap: Overlap,
}

impl LoopConfig {
    /// Parse `loop.toml` text. Total: returns `Err(reason)` on malformed input.
    pub fn parse(text: &str) -> Result<LoopConfig, String> {
        let doc = Toml::parse(text)?;
        let mut cfg = LoopConfig::default();

        if let Some(t) = doc.table("schedule") {
            cfg.schedule = Schedule {
                heartbeat: t.str("heartbeat"),
                wake_on: t.arr("wake_on"),
                deadline: t.str("deadline"),
            };
        }
        if let Some(t) = doc.table("bar") {
            cfg.bar = Some(Bar {
                value: t.str("value").unwrap_or_default(),
                version: t.int("version").unwrap_or(1).max(0) as u32,
                conditions: t.inline("conditions"),
                remeasure: t.str("remeasure"),
            });
        }
        if let Some(t) = doc.table("contract") {
            let d = Contract::default();
            cfg.contract = Contract {
                inject_spec: t.bool("inject_spec").unwrap_or(d.inject_spec),
                require_state_append: t
                    .bool("require_state_append")
                    .unwrap_or(d.require_state_append),
                require_next_action: t
                    .bool("require_next_action")
                    .unwrap_or(d.require_next_action),
            };
        }
        if let Some(t) = doc.table("budget") {
            cfg.budget = Budget {
                max_wakes: t.int("max_wakes").map(|n| n.max(0) as u64),
                max_tokens: t.int("max_tokens").map(|n| n.max(0) as u64),
                on_exhaust: match t.str("on_exhaust").as_deref() {
                    Some("stop") => OnExhaust::Stop,
                    _ => OnExhaust::Pause,
                },
            };
        }
        if let Some(t) = doc.table("authority") {
            cfg.authority = Authority {
                allow_push: t.arr("allow_push"),
                max_restarts: t
                    .inline("max_restarts")
                    .into_iter()
                    .filter_map(|(k, v)| v.parse::<u32>().ok().map(|n| (k, n)))
                    .collect(),
            };
        }
        if let Some(t) = doc.table("trigger") {
            cfg.trigger = Some(Trigger {
                when: t.str("when").unwrap_or_default(),
                on_miss: match t.str("on_miss").as_deref() {
                    Some("run-now") => OnMiss::RunNow,
                    Some("ask") => OnMiss::Ask,
                    _ => OnMiss::Skip,
                },
                overlap: match t.str("overlap").as_deref() {
                    Some("queue") => Overlap::Queue,
                    Some("parallel") => Overlap::Parallel,
                    _ => Overlap::Skip,
                },
            });
        }
        for t in doc.array_of("gate") {
            let kind = t.str("kind");
            if kind.as_deref() == Some("rubric") {
                cfg.gates.push(Gate {
                    name: t.str("name").unwrap_or_default(),
                    kind: crate::gate::GateKind::Rubric,
                    run: t.str("run").unwrap_or_default(),
                    extractor: crate::gate::Extractor::Text,
                    pass: t.str("pass").unwrap_or_default(),
                    rubric: t.str("rubric"),
                });
            } else {
                cfg.gates.push(Gate::command(
                    &t.str("name").unwrap_or_default(),
                    &t.str("run").unwrap_or_default(),
                    &t.str("parse").unwrap_or_default(),
                    &t.str("pass").unwrap_or_default(),
                ));
            }
        }
        Ok(cfg)
    }
}

/// Parse a short duration: `<n><unit>` where unit ∈ `s|m|h|d`, or `<n>` (seconds).
/// Returns `None` for anything else. (`"30m"` → 1800s, `"6h"` → 21600s.)
pub fn parse_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (num, unit) = match s.char_indices().find(|(_, c)| c.is_ascii_alphabetic()) {
        Some((i, _)) => (&s[..i], &s[i..]),
        None => (s, "s"),
    };
    let n: u64 = num.trim().parse().ok()?;
    let secs = match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86400,
        _ => return None,
    };
    Some(Duration::from_secs(secs))
}

// --- the minimal TOML reader ------------------------------------------------

/// One TOML value, restricted to what `loop.toml` uses.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tv {
    Str(String),
    Int(i64),
    Bool(bool),
    Arr(Vec<String>),
    Inline(Vec<(String, String)>),
}

/// A parsed document: named single tables plus named arrays-of-tables.
struct Toml {
    singles: Vec<(String, Table)>,
    arrays: Vec<(String, Vec<Table>)>,
}

/// A table is an ordered list of key→value (order preserved for determinism).
#[derive(Default)]
struct Table {
    kv: Vec<(String, Tv)>,
}

impl Table {
    fn get(&self, key: &str) -> Option<&Tv> {
        self.kv.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn str(&self, key: &str) -> Option<String> {
        match self.get(key) {
            Some(Tv::Str(s)) => Some(s.clone()),
            Some(Tv::Int(n)) => Some(n.to_string()),
            Some(Tv::Bool(b)) => Some(b.to_string()),
            _ => None,
        }
    }
    fn int(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(Tv::Int(n)) => Some(*n),
            Some(Tv::Str(s)) => s.parse().ok(),
            _ => None,
        }
    }
    fn bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Tv::Bool(b)) => Some(*b),
            _ => None,
        }
    }
    fn arr(&self, key: &str) -> Vec<String> {
        match self.get(key) {
            Some(Tv::Arr(a)) => a.clone(),
            _ => Vec::new(),
        }
    }
    fn inline(&self, key: &str) -> Vec<(String, String)> {
        match self.get(key) {
            Some(Tv::Inline(t)) => t.clone(),
            _ => Vec::new(),
        }
    }
}

impl Toml {
    fn table(&self, name: &str) -> Option<&Table> {
        self.singles.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
    fn array_of(&self, name: &str) -> &[Table] {
        self.arrays
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
            .unwrap_or(&[])
    }

    fn parse(text: &str) -> Result<Toml, String> {
        let mut singles: Vec<(String, Table)> = Vec::new();
        let mut arrays: Vec<(String, Vec<Table>)> = Vec::new();
        // `Cur` points at the table the next key=value lands in.
        enum Cur {
            None,
            Single(usize),
            Array(usize), // index into `arrays`; appends to the last entry
        }
        let mut cur = Cur::None;

        for (lineno, raw) in text.lines().enumerate() {
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if let Some(inner) = line.strip_prefix("[[").and_then(|s| s.strip_suffix("]]")) {
                let name = inner.trim().to_string();
                let idx = match arrays.iter().position(|(n, _)| *n == name) {
                    Some(i) => i,
                    None => {
                        arrays.push((name, Vec::new()));
                        arrays.len() - 1
                    }
                };
                arrays[idx].1.push(Table::default());
                cur = Cur::Array(idx);
            } else if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                let name = inner.trim().to_string();
                singles.push((name, Table::default()));
                cur = Cur::Single(singles.len() - 1);
            } else if let Some(eq) = line.find('=') {
                let key = line[..eq].trim().to_string();
                let val = parse_value(line[eq + 1..].trim())
                    .ok_or_else(|| format!("line {}: bad value for {key:?}", lineno + 1))?;
                let table = match cur {
                    Cur::Single(i) => &mut singles[i].1,
                    Cur::Array(i) => arrays[i]
                        .1
                        .last_mut()
                        .ok_or_else(|| format!("line {}: key before [[table]]", lineno + 1))?,
                    Cur::None => {
                        return Err(format!("line {}: key {key:?} before any table", lineno + 1))
                    }
                };
                table.kv.push((key, val));
            } else {
                return Err(format!("line {}: not a header or key=value", lineno + 1));
            }
        }
        Ok(Toml { singles, arrays })
    }
}

/// Drop a trailing `# comment` not inside a string.
fn strip_comment(line: &str) -> &str {
    let mut in_str = false;
    let bytes = line.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' | b'\'' => in_str = !in_str,
            b'#' if !in_str => return &line[..i],
            _ => {}
        }
    }
    line
}

/// Parse a single TOML value (string / int / bool / array / inline table).
fn parse_value(s: &str) -> Option<Tv> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(inner) = quoted(s) {
        return Some(Tv::Str(inner));
    }
    if s == "true" {
        return Some(Tv::Bool(true));
    }
    if s == "false" {
        return Some(Tv::Bool(false));
    }
    if let Some(inner) = s.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let items = split_top(inner)
            .into_iter()
            .filter(|x| !x.trim().is_empty())
            .map(|x| quoted(x.trim()).unwrap_or_else(|| x.trim().to_string()))
            .collect();
        return Some(Tv::Arr(items));
    }
    if let Some(inner) = s.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        let mut pairs = Vec::new();
        for part in split_top(inner) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let eq = part.find('=')?;
            let k = part[..eq].trim().to_string();
            let v = part[eq + 1..].trim();
            let vs = quoted(v).unwrap_or_else(|| v.to_string());
            pairs.push((k, vs));
        }
        return Some(Tv::Inline(pairs));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(Tv::Int(n));
    }
    // A bare token (unquoted string) — accept it as a string for leniency.
    Some(Tv::Str(s.to_string()))
}

/// If `s` is a `"…"` or `'…'` quoted string, return its inner text.
fn quoted(s: &str) -> Option<String> {
    let b = s.as_bytes();
    if b.len() >= 2 {
        let (f, l) = (b[0], b[b.len() - 1]);
        if (f == b'"' && l == b'"') || (f == b'\'' && l == b'\'') {
            return Some(s[1..s.len() - 1].to_string());
        }
    }
    None
}

/// Split on top-level commas (not inside quotes / nested brackets/braces).
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' | b'\'' => in_str = !in_str,
            b'[' | b'{' if !in_str => depth += 1,
            b']' | b'}' if !in_str => depth -= 1,
            b',' if !in_str && depth == 0 => {
                out.push(s[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::GateKind;

    const SAMPLE: &str = r#"
        # the loop's machine half
        [schedule]
        heartbeat = "30m"
        wake_on   = ["job:done"]
        deadline  = "2026-06-13T11:00:00-05:00"

        [bar]
        value     = "6762 tok/s @ cosine 0.9987"  # measured
        version   = 3
        conditions = { seq = 6401, clocks = "locked-2392", harness = "nemo_band" }
        remeasure = "ssh spark python3 ~/nemo_band.py --engine vllm-nvfp4"

        [contract]
        require_next_action = false

        [[gate]]
        name   = "quality"
        run    = "ssh spark python3 ~/nemo_band.py --engine ours"
        parse  = "json:.cosine_mean"
        pass   = ">= 0.9987"

        [[gate]]
        name   = "writeup"
        kind   = "rubric"
        rubric = "criteria.md"

        [budget]
        max_wakes   = 200
        max_tokens  = 5000000
        on_exhaust  = "pause"

        [authority]
        allow_push  = ["github.com/org/repo#20"]
        max_restarts = { spark = 3, train = 3 }

        [trigger]
        when    = "daily 08:00"
        on_miss = "skip"
        overlap = "skip"
    "#;

    #[test]
    fn parses_the_full_prd_sample() {
        let c = LoopConfig::parse(SAMPLE).unwrap();
        assert_eq!(c.schedule.heartbeat.as_deref(), Some("30m"));
        assert_eq!(c.schedule.wake_on, vec!["job:done"]);
        assert!(c.schedule.deadline.unwrap().starts_with("2026-06-13"));

        let bar = c.bar.unwrap();
        assert_eq!(bar.version, 3);
        assert!(bar.value.contains("6762"));
        assert_eq!(
            bar.conditions,
            vec![
                ("seq".to_string(), "6401".to_string()),
                ("clocks".to_string(), "locked-2392".to_string()),
                ("harness".to_string(), "nemo_band".to_string()),
            ]
        );

        // Contract: defaults on, one explicit override.
        assert!(c.contract.inject_spec);
        assert!(c.contract.require_state_append);
        assert!(!c.contract.require_next_action);

        assert_eq!(c.gates.len(), 2);
        assert_eq!(c.gates[0].name, "quality");
        assert_eq!(c.gates[0].pass, ">= 0.9987");
        assert_eq!(c.gates[1].kind, GateKind::Rubric);
        assert_eq!(c.gates[1].rubric.as_deref(), Some("criteria.md"));

        assert_eq!(c.budget.max_wakes, Some(200));
        assert_eq!(c.budget.max_tokens, Some(5_000_000));
        assert_eq!(c.budget.on_exhaust, OnExhaust::Pause);

        assert_eq!(c.authority.allow_push, vec!["github.com/org/repo#20"]);
        assert_eq!(
            c.authority.max_restarts,
            vec![("spark".to_string(), 3), ("train".to_string(), 3)]
        );

        let trig = c.trigger.unwrap();
        assert_eq!(trig.when, "daily 08:00");
        assert_eq!(trig.on_miss, OnMiss::Skip);
        assert_eq!(trig.overlap, Overlap::Skip);
    }

    #[test]
    fn empty_config_is_all_defaults() {
        let c = LoopConfig::parse("").unwrap();
        assert!(c.contract.inject_spec); // default-on
        assert!(c.gates.is_empty());
        assert!(c.bar.is_none());
        assert_eq!(c.budget.on_exhaust, OnExhaust::Pause);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let c = LoopConfig::parse("# just a comment\n\n   \n[budget]\nmax_wakes = 5\n").unwrap();
        assert_eq!(c.budget.max_wakes, Some(5));
    }

    #[test]
    fn malformed_input_errors_not_panics() {
        // A key before any table header.
        assert!(LoopConfig::parse("foo = 1").is_err());
        // A line that's neither header nor key=value.
        assert!(LoopConfig::parse("[budget]\njust some words").is_err());
    }

    #[test]
    fn on_exhaust_and_trigger_variants() {
        let c = LoopConfig::parse(
            "[budget]\non_exhaust = \"stop\"\n[trigger]\nwhen=\"every 6h\"\non_miss=\"run-now\"\noverlap=\"parallel\"\n",
        )
        .unwrap();
        assert_eq!(c.budget.on_exhaust, OnExhaust::Stop);
        let t = c.trigger.unwrap();
        assert_eq!(t.on_miss, OnMiss::RunNow);
        assert_eq!(t.overlap, Overlap::Parallel);
    }

    #[test]
    fn parse_duration_units() {
        assert_eq!(parse_duration("30m"), Some(Duration::from_secs(1800)));
        assert_eq!(parse_duration("6h"), Some(Duration::from_secs(21600)));
        assert_eq!(parse_duration("45s"), Some(Duration::from_secs(45)));
        assert_eq!(parse_duration("2d"), Some(Duration::from_secs(172800)));
        assert_eq!(parse_duration("90"), Some(Duration::from_secs(90))); // bare = seconds
        assert_eq!(parse_duration("nonsense"), None);
        assert_eq!(parse_duration(""), None);
    }

    #[test]
    fn comment_inside_a_string_is_kept() {
        let c = LoopConfig::parse("[bar]\nvalue = \"a # b\"\nversion = 1\n").unwrap();
        assert_eq!(c.bar.unwrap().value, "a # b");
    }

    #[test]
    fn array_of_tables_groups_independently() {
        let c = LoopConfig::parse(
            "[[gate]]\nname=\"a\"\npass=\"exit == 0\"\n[[gate]]\nname=\"b\"\npass=\"> 1\"\n",
        )
        .unwrap();
        assert_eq!(c.gates.len(), 2);
        assert_eq!(c.gates[0].name, "a");
        assert_eq!(c.gates[1].name, "b");
    }
}
