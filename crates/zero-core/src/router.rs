// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! Pure routing for the fleet orchestrator: given the request context and the
//! [`FleetConfig`], decide which tier answers — and, on a tier failure, in what
//! order to try the rest.
//!
//! Everything here is a pure function over text + config, so it is exhaustively
//! table- and property-tested with no network or model in the loop (the same
//! discipline as `gate.rs` and the loop `LoopGuard`). Routing never calls a model
//! to decide routing — that would be an estimate, and it would be slow. The
//! signals are cheap and observable: the editor **mode**, the **prompt shape**, a
//! **manual** pin, and whether the **prior** turn failed.

use crate::fleet_config::{FleetConfig, RoutingMode};

/// The editor mode, as the router cares about it. Mirrors the TUI's modes but
/// lives in the core so routing is testable without the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModeHint {
    #[default]
    Normal,
    AutoAccept,
    Plan,
}

impl ModeHint {
    /// Compact encoding for an atomic signal (see `orchestrator::FleetSignal`).
    pub fn as_u8(self) -> u8 {
        match self {
            ModeHint::Normal => 0,
            ModeHint::AutoAccept => 1,
            ModeHint::Plan => 2,
        }
    }
    /// Decode; any unknown byte is [`ModeHint::Normal`].
    pub fn from_u8(b: u8) -> ModeHint {
        match b {
            1 => ModeHint::AutoAccept,
            2 => ModeHint::Plan,
            _ => ModeHint::Normal,
        }
    }
}

/// What the prompt looks like it needs. Derived purely from text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    /// Planning / architecture / hard debugging / long prompts → strongest tier.
    Plan,
    /// Short lookups / status / simple questions → cheapest tier.
    Simple,
    /// Everything else → the baseline tier.
    Normal,
}

/// The inputs to a routing decision.
#[derive(Debug, Clone, Copy)]
pub struct RouteInput<'a> {
    /// The user's latest message (the thing being routed).
    pub prompt: &'a str,
    /// The assistant's previous turn (`""` if none). When the user's latest
    /// message is a bare continuation ("do it" / "yes"), the real task lives here —
    /// the long plan or question the model just produced — so routing reads it
    /// instead of the affirmation. See [`route`].
    pub prev_assistant: &'a str,
    /// The current editor mode.
    pub mode: ModeHint,
    /// A manual tier pin (`/model deep`), if the user set one. `Some("auto")` or
    /// an unknown key is treated as "no pin".
    pub manual: Option<&'a str>,
    /// Whether the previous turn exhausted its tiers with an error → bias up.
    pub prior_failed: bool,
}

/// Whether a user message is a bare go-ahead ("do it", "yes", "ok go") rather than
/// a substantive request. Such a message carries no routing signal of its own — the
/// work is whatever the assistant just proposed — so routing falls back to the
/// previous assistant turn. Long messages always carry their own intent.
pub fn is_continuation(s: &str) -> bool {
    let t = s
        .trim()
        .trim_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_ascii_lowercase();
    if t.is_empty() || t.len() > 40 {
        return false;
    }
    const AFFIRM: &[&str] = &[
        "do it",
        "do that",
        "do so",
        "yes",
        "yep",
        "yeah",
        "y",
        "ok",
        "okay",
        "k",
        "sure",
        "go",
        "go ahead",
        "go for it",
        "proceed",
        "continue",
        "carry on",
        "make it so",
        "ship it",
        "lgtm",
        "approved",
        "sounds good",
        "please do",
        "yes please",
        "run it",
        "execute",
        "perfect",
    ];
    AFFIRM.iter().any(|a| {
        t.starts_with(a)
            && t[a.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric())
    })
}

/// How sure the heuristic is. A `Low` verdict marks a genuinely ambiguous request
/// (a medium imperative with no strong signal) — the point at which an optional
/// model-based classifier layer would earn its keep. A `High` verdict has a clear
/// signal and routes without one. Mirrors the "fast pass, escalate to a thinking
/// pass only when uncertain" shape Claude Code's command classifier uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    High,
    Low,
}

/// A heuristic classification: the read intent plus how sure we are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classification {
    pub intent: Intent,
    pub confidence: Confidence,
}

/// Planning / deep-reasoning verbs → the strong tier.
const PLAN_KW: &[&str] = &[
    "plan",
    "architect",
    "design ",
    "refactor",
    "investigat",
    "root cause",
    "strateg",
    "step by step",
    "approach",
    "rewrite",
    "redesign",
    "trade-off",
    "tradeoff",
];
/// Debugging / failure signals — a stack trace or "why is this broken" almost
/// always wants the strong model. These are the cheapest high-value signal.
const DEBUG_KW: &[&str] = &[
    "debug",
    "why does",
    "why is",
    "error:",
    "panic",
    "exception",
    "traceback",
    "stack trace",
    "stacktrace",
    "segfault",
    "segmentation fault",
    "deadlock",
    "race condition",
    "doesn't work",
    "not working",
    "broken",
];
/// Short-lookup openers → the fast tier.
const SIMPLE_STARTS: &[&str] = &[
    "what", "list", "show", "status", "read ", "ls", "cat ", "is ", "are ", "how many", "where ",
    "when ", "which ", "print", "echo",
];

/// Classify a prompt by shape — cheap, pure, never a model call. Returns the read
/// [`Intent`] and a [`Confidence`].
///
/// - **Plan / High**: a planning verb, a debugging/error signal, a multi-step
///   request, or a long prompt (> 800 chars).
/// - **Simple / High**: short (< 80 chars), no code fence, ends in `?` or opens
///   with a lookup word (`what`, `list`, `show`, …).
/// - **Normal**: no strong signal — `High` only when trivially short, else `Low`
///   (a medium imperative we're genuinely unsure about).
pub fn classify(prompt: &str) -> Classification {
    let p = prompt.trim();
    if p.is_empty() {
        return Classification {
            intent: Intent::Normal,
            confidence: Confidence::High,
        };
    }
    let lower = p.to_ascii_lowercase();
    let has = |kws: &[&str]| kws.iter().any(|k| lower.contains(k));

    if p.len() > 800 || has(PLAN_KW) || has(DEBUG_KW) || is_multistep(p, &lower) {
        return Classification {
            intent: Intent::Plan,
            confidence: Confidence::High,
        };
    }
    if p.len() < 80
        && !p.contains("```")
        && (p.ends_with('?') || SIMPLE_STARTS.iter().any(|s| lower.starts_with(s)))
    {
        return Classification {
            intent: Intent::Simple,
            confidence: Confidence::High,
        };
    }
    // No strong signal: a very short imperative is trivially the baseline; anything
    // longer is genuinely ambiguous (where a model classifier, if enabled, helps).
    let confidence = if p.len() < 24 {
        Confidence::High
    } else {
        Confidence::Low
    };
    Classification {
        intent: Intent::Normal,
        confidence,
    }
}

/// A multi-step request (a numbered list, "first … then …", or "step by step") is
/// planning-grade.
fn is_multistep(p: &str, lower: &str) -> bool {
    if lower.contains("step by step") || (lower.contains("first") && lower.contains("then")) {
        return true;
    }
    // Two or more lines that open with an ordinal like `1.` or `2)`.
    p.lines()
        .filter(|l| {
            let mut c = l.trim_start().chars();
            matches!(c.next(), Some(d) if d.is_ascii_digit())
                && matches!(c.next(), Some('.') | Some(')'))
        })
        .count()
        >= 2
}

/// A routing decision: which tier, and how sure the heuristic was. A `Low`
/// confidence is the cue for the orchestrator's optional model classifier to weigh
/// in; with the classifier off it's purely informational.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteDecision {
    pub tier: String,
    pub confidence: Confidence,
}

/// Decide which tier should answer, with a confidence. The returned key is always a
/// tier that exists in `cfg` (a stale pin is clamped to the baseline), unless `cfg`
/// has no tiers at all (then `""`). Resolution order: **manual pin → manual mode →
/// (continuation → prior turn) mode/intent → escalate-on-prior-failure**. Explicit
/// signals (a pin, plan mode, a deliberate escalation) are always `High` confidence;
/// only a heuristic read of free text can be `Low`.
pub fn route_decision(input: &RouteInput, cfg: &FleetConfig) -> RouteDecision {
    let high = |tier: String| RouteDecision {
        tier,
        confidence: Confidence::High,
    };
    // 1. A valid manual pin wins outright (the operator stays in control).
    if let Some(m) = input.manual {
        let m = m.trim();
        if !m.is_empty() && m != "auto" && cfg.tier(m).is_some() {
            return high(m.to_string());
        }
    }
    // 2. Manual mode with no usable pin → the strongest tier (a stable default).
    if cfg.routing == RoutingMode::Manual {
        return high(cfg.strongest().unwrap_or_default().to_string());
    }

    // 3. Auto: decide what text the intent is read from. A bare "do it" carries no
    //    signal — the task is whatever the model just proposed — so route on the
    //    previous assistant turn instead of the affirmation.
    let continuation = is_continuation(input.prompt) && !input.prev_assistant.trim().is_empty();
    let text = if continuation {
        input.prev_assistant
    } else {
        input.prompt
    };
    let (mut intent, mut confidence) = if matches!(input.mode, ModeHint::Plan) {
        (Intent::Plan, Confidence::High)
    } else {
        let c = classify(text);
        (c.intent, c.confidence)
    };
    // Executing a just-proposed plan is never a "fast tier" job: floor a
    // continuation at the baseline, and treat a long proposal as planning-grade.
    if continuation {
        if input.prev_assistant.len() > 600 {
            intent = Intent::Plan;
            confidence = Confidence::High;
        } else if intent == Intent::Simple {
            intent = Intent::Normal;
        }
    }
    let mut key = match intent {
        Intent::Plan => cfg.plan_pin.as_deref().or_else(|| cfg.strongest()),
        Intent::Simple => cfg.simple_pin.as_deref().or_else(|| cfg.weakest()),
        Intent::Normal => cfg.baseline_key(),
    }
    .unwrap_or_default()
    .to_string();

    // A pin that names a since-removed tier clamps to the baseline.
    if cfg.tier(&key).is_none() {
        key = cfg.baseline_key().unwrap_or_default().to_string();
    }
    // 4. Escalate one rung toward the strongest after a failure — a deliberate
    //    move on evidence, not a guess.
    if input.prior_failed {
        key = escalate(cfg, &key);
        confidence = Confidence::High;
    }
    RouteDecision {
        tier: key,
        confidence,
    }
}

/// The routed tier key (see [`route_decision`] for the confidence too).
pub fn route(input: &RouteInput, cfg: &FleetConfig) -> String {
    route_decision(input, cfg).tier
}

/// Move one rung toward the strongest tier along the fallback ladder.
fn escalate(cfg: &FleetConfig, key: &str) -> String {
    match cfg.fallback.iter().position(|k| k == key) {
        Some(i) if i > 0 => cfg.fallback[i - 1].clone(),
        Some(_) => key.to_string(), // already strongest
        None => cfg.strongest().unwrap_or(key).to_string(),
    }
}

/// The order to try tiers when the routed one errors: the routed tier first, then
/// the configured fallback ladder, then any remaining declared tiers. Deduplicated
/// and limited to tiers that actually exist.
pub fn fallback_order(cfg: &FleetConfig, start: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let add = |out: &mut Vec<String>, k: &str| {
        if cfg.tier(k).is_some() && !out.iter().any(|x| x == k) {
            out.push(k.to_string());
        }
    };
    add(&mut out, start);
    for k in &cfg.fallback {
        add(&mut out, k);
    }
    for t in &cfg.tiers {
        add(&mut out, &t.key);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fleet_config::FleetConfig;

    fn cfg() -> FleetConfig {
        FleetConfig::parse(
            r#"
            enabled  = true
            baseline = "balanced"
            [[tier]]
            key = "deep"
            where = "http://d:1"
            [[tier]]
            key = "balanced"
            where = "http://b:1"
            [[tier]]
            key = "fast"
            where = "http://f:1"
            [routing]
            plan_mode = "deep"
            simple_queries = "fast"
            fallback = ["deep", "balanced", "fast"]
        "#,
        )
        .unwrap()
    }

    fn input<'a>(
        prompt: &'a str,
        mode: ModeHint,
        manual: Option<&'a str>,
        failed: bool,
    ) -> RouteInput<'a> {
        RouteInput {
            prompt,
            prev_assistant: "",
            mode,
            manual,
            prior_failed: failed,
        }
    }

    /// A routing input with a preceding assistant turn (for continuation tests).
    fn after<'a>(prev_assistant: &'a str, prompt: &'a str) -> RouteInput<'a> {
        RouteInput {
            prompt,
            prev_assistant,
            mode: ModeHint::Normal,
            manual: None,
            prior_failed: false,
        }
    }

    #[test]
    fn mode_u8_roundtrips() {
        for m in [ModeHint::Normal, ModeHint::AutoAccept, ModeHint::Plan] {
            assert_eq!(ModeHint::from_u8(m.as_u8()), m);
        }
        assert_eq!(ModeHint::from_u8(99), ModeHint::Normal);
    }

    #[test]
    fn classify_table() {
        let cases = [
            ("plan the migration", Intent::Plan),
            ("refactor this module", Intent::Plan),
            ("why does this crash?", Intent::Plan), // debug signal beats simple
            ("rewrite the scheduler", Intent::Plan),
            ("compare these two approaches and pick one", Intent::Plan), // "approach"
            ("the test panics with index out of bounds", Intent::Plan),  // debug signal
            ("here's the error: thread 'main' panicked", Intent::Plan),  // "error:"/"panic"
            ("what is the capital?", Intent::Simple),
            ("list the files", Intent::Simple),
            ("status", Intent::Simple),
            (
                "fix the typo on line 4 of the readme please",
                Intent::Normal,
            ),
            ("", Intent::Normal),
            ("   ", Intent::Normal),
        ];
        for (p, want) in cases {
            assert_eq!(classify(p).intent, want, "prompt {p:?}");
        }
    }

    #[test]
    fn multistep_request_is_plan() {
        let numbered = "do these:\n1. read the file\n2. patch it\n3. run tests";
        assert_eq!(classify(numbered).intent, Intent::Plan);
        assert_eq!(
            classify("first set up the config then wire the handler").intent,
            Intent::Plan
        );
    }

    #[test]
    fn confidence_high_on_strong_signal_low_on_ambiguous() {
        // Strong signals → High.
        assert_eq!(classify("refactor the parser").confidence, Confidence::High);
        assert_eq!(classify("list the files").confidence, Confidence::High);
        assert_eq!(classify("ok").confidence, Confidence::High); // trivially short
                                                                 // A medium imperative with no strong signal → Low (the model-classifier cue).
        assert_eq!(
            classify("update the handler to use the new field name everywhere").confidence,
            Confidence::Low
        );
    }

    #[test]
    fn long_prompt_is_plan() {
        let long = "a".repeat(900);
        assert_eq!(classify(&long).intent, Intent::Plan);
    }

    #[test]
    fn code_fence_short_prompt_is_not_simple() {
        // Short but contains a code fence → Normal, not Simple.
        assert_eq!(classify("fix ```x```?").intent, Intent::Normal);
    }

    #[test]
    fn manual_pin_wins() {
        let c = cfg();
        assert_eq!(
            route(&input("plan it", ModeHint::Plan, Some("fast"), false), &c),
            "fast"
        );
    }

    #[test]
    fn manual_auto_or_unknown_is_ignored() {
        let c = cfg();
        // "auto" → fall through to auto routing (plan mode → deep).
        assert_eq!(
            route(&input("hi", ModeHint::Plan, Some("auto"), false), &c),
            "deep"
        );
        // unknown tier → ignored, auto routing (normal → baseline).
        assert_eq!(
            route(
                &input("do a thing", ModeHint::Normal, Some("nope"), false),
                &c
            ),
            "balanced"
        );
    }

    #[test]
    fn plan_mode_routes_to_deep() {
        let c = cfg();
        assert_eq!(
            route(&input("anything", ModeHint::Plan, None, false), &c),
            "deep"
        );
    }

    #[test]
    fn simple_prompt_routes_to_fast() {
        let c = cfg();
        assert_eq!(
            route(&input("list files", ModeHint::Normal, None, false), &c),
            "fast"
        );
    }

    #[test]
    fn normal_prompt_routes_to_baseline() {
        let c = cfg();
        assert_eq!(
            route(
                &input(
                    "fix the bug in the parser for me thanks",
                    ModeHint::Normal,
                    None,
                    false
                ),
                &c
            ),
            "balanced"
        );
    }

    #[test]
    fn prior_failure_escalates_one_rung() {
        let c = cfg();
        // baseline (balanced) → escalate → deep.
        assert_eq!(
            route(
                &input(
                    "fix the bug in the parser for me thanks",
                    ModeHint::Normal,
                    None,
                    true
                ),
                &c
            ),
            "deep"
        );
        // already strongest (plan→deep) stays deep.
        assert_eq!(
            route(&input("plan it", ModeHint::Plan, None, true), &c),
            "deep"
        );
        // fast → escalate → balanced.
        assert_eq!(
            route(&input("list files", ModeHint::Normal, None, true), &c),
            "balanced"
        );
    }

    #[test]
    fn manual_mode_with_no_pin_uses_strongest() {
        let c = FleetConfig::parse(
            "enabled=true\nrouting=manual\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\n[[tier]]\nkey=\"b\"\nwhere=\"http://b:1\"",
        )
        .unwrap();
        assert_eq!(route(&input("x", ModeHint::Normal, None, false), &c), "a");
        // a manual pin still wins in manual mode
        assert_eq!(
            route(&input("x", ModeHint::Normal, Some("b"), false), &c),
            "b"
        );
    }

    #[test]
    fn stale_pin_clamps_to_baseline() {
        // plan_pin points at a tier that doesn't exist → clamp to baseline.
        let c = FleetConfig::parse(
            "enabled=true\nbaseline=\"b\"\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\n[[tier]]\nkey=\"b\"\nwhere=\"http://b:1\"\n[routing]\nplan_mode=\"ghost\"",
        )
        .unwrap();
        assert_eq!(route(&input("x", ModeHint::Plan, None, false), &c), "b");
    }

    #[test]
    fn fallback_order_starts_with_routed_then_ladder() {
        let c = cfg();
        assert_eq!(
            fallback_order(&c, "balanced"),
            vec!["balanced", "deep", "fast"]
        );
        assert_eq!(fallback_order(&c, "deep"), vec!["deep", "balanced", "fast"]);
    }

    #[test]
    fn fallback_order_includes_tiers_missing_from_ladder() {
        // A tier not in the fallback list still gets appended.
        let c = FleetConfig::parse(
            "enabled=true\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"\n[[tier]]\nkey=\"b\"\nwhere=\"http://b:1\"\n[routing]\nfallback=[\"a\"]",
        )
        .unwrap();
        assert_eq!(fallback_order(&c, "a"), vec!["a", "b"]);
    }

    #[test]
    fn fallback_order_ignores_unknown_start() {
        let c = cfg();
        // unknown start contributes nothing but the ladder still fills in.
        assert_eq!(
            fallback_order(&c, "ghost"),
            vec!["deep", "balanced", "fast"]
        );
    }

    #[test]
    fn continuation_predicate() {
        for s in [
            "do it",
            "Do it.",
            "yes",
            "yes!",
            "ok go ahead",
            "sure, proceed",
            "ship it",
            "y",
        ] {
            assert!(is_continuation(s), "{s:?} should be a continuation");
        }
        for s in [
            "yesterday's run failed", // "yes" prefix but a real word
            "do it but first explain the tradeoffs between the two designs in detail",
            "implement the parser", // a real instruction
            "",
        ] {
            assert!(!is_continuation(s), "{s:?} should NOT be a continuation");
        }
    }

    #[test]
    fn bare_do_it_routes_on_the_prior_assistant_turn() {
        let c = cfg();
        // The model just laid out a plan (planning keywords); "do it" → deep, not
        // the baseline a 5-char prompt would otherwise get.
        let plan = "Here is the plan: refactor the parser, add a streaming decoder, \
                    then update the tests. Shall I proceed?";
        assert_eq!(route(&after(plan, "do it"), &c), "deep");
    }

    #[test]
    fn continuation_after_long_keywordless_proposal_routes_deep() {
        let c = cfg();
        // A long proposal with NO planning keyword: the length-of-prior-turn rule
        // (> 600 chars) is what makes "do it" planning-grade, exercising that branch.
        let long =
            "i would change the way the tokens are counted and the way the results come back. "
                .repeat(8);
        assert!(long.len() > 600 && classify(&long).intent == Intent::Normal);
        assert_eq!(route(&after(&long, "do it"), &c), "deep");
    }

    #[test]
    fn continuation_after_a_short_proposal_floors_at_baseline_not_fast() {
        let c = cfg();
        // A short proposal that classify() alone would read as Simple — but a "do
        // it" must not drop to the fast tier.
        assert_eq!(
            route(&after("want me to list the files?", "yes"), &c),
            "balanced"
        );
    }

    #[test]
    fn continuation_with_no_prior_turn_uses_the_prompt() {
        let c = cfg();
        // First message of a session is "do it" with nothing before → fall back to
        // classifying the prompt itself (short → baseline path, here Normal).
        assert_eq!(route(&after("", "do it"), &c), "balanced");
    }

    #[test]
    fn substantive_message_ignores_the_prior_turn() {
        let c = cfg();
        // Not a continuation → route on the user's own (simple) prompt.
        assert_eq!(
            route(
                &after(
                    "a long plan the model wrote with refactor steps",
                    "list files"
                ),
                &c
            ),
            "fast"
        );
    }

    #[test]
    fn route_decision_confidence_reflects_signal_strength() {
        let c = cfg();
        // Explicit + strong signals are High confidence.
        let strong = [
            input("x", ModeHint::Plan, None, false), // plan mode
            input("refactor it", ModeHint::Normal, None, false), // strong keyword
            input("x", ModeHint::Normal, Some("deep"), false), // manual pin
        ];
        for i in strong {
            assert_eq!(route_decision(&i, &c).confidence, Confidence::High);
        }
        // A medium imperative with no strong signal → Low (the model-classifier cue).
        let amb = route_decision(
            &input(
                "update the handler to use the new field name everywhere",
                ModeHint::Normal,
                None,
                false,
            ),
            &c,
        );
        assert_eq!(amb.confidence, Confidence::Low);
        assert_eq!(amb.tier, "balanced");
        // A deliberate escalation is evidence, not a guess → High.
        assert_eq!(
            route_decision(
                &input(
                    "update the handler to use the new field",
                    ModeHint::Normal,
                    None,
                    true
                ),
                &c
            )
            .confidence,
            Confidence::High
        );
    }

    // --- property tests (hand-rolled SplitMix64, seeds 0..400) ----------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn idx(&mut self, len: usize) -> usize {
            (self.next() as usize) % len
        }
    }

    #[test]
    fn prop_route_always_returns_an_existing_tier() {
        let c = cfg();
        let prompts = [
            "plan it",
            "list files",
            "fix the parser bug now",
            "",
            "status",
        ];
        let modes = [ModeHint::Normal, ModeHint::AutoAccept, ModeHint::Plan];
        let manuals: [Option<&str>; 5] = [
            None,
            Some("auto"),
            Some("ghost"),
            Some("fast"),
            Some("deep"),
        ];
        for seed in 0u64..400 {
            let mut r = Rng(seed);
            let inp = input(
                prompts[r.idx(prompts.len())],
                modes[r.idx(modes.len())],
                manuals[r.idx(manuals.len())],
                r.next() & 1 == 0,
            );
            let key = route(&inp, &c);
            assert!(
                c.tier(&key).is_some(),
                "seed {seed}: routed to missing tier {key:?}"
            );

            // fallback_order: starts with the routed tier and covers every tier.
            let order = fallback_order(&c, &key);
            assert_eq!(
                order.first().map(String::as_str),
                Some(key.as_str()),
                "seed {seed}"
            );
            assert_eq!(
                order.len(),
                c.tiers.len(),
                "seed {seed}: order must cover all tiers once"
            );
            for t in &c.tiers {
                assert!(
                    order.contains(&t.key),
                    "seed {seed}: order missing {}",
                    t.key
                );
            }
        }
    }
}
