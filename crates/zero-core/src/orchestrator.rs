// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! The fleet orchestrator: one [`Backend`] that fronts several tier-backends and
//! routes each request to the right one. Because it *is* a `Backend`, it drops in
//! wherever a single backend goes — the agent loop, tool calling, the streaming
//! render path — without any UI change (north star #2, "Terminal ≡ App").
//!
//! Routing is engine-agnostic: Zero already re-sends the full conversation every
//! turn (OpenAI-compatible chat is stateless), so sending it to a *different* tier
//! is correct on any server; only the server's cache warmth differs. The harder
//! cache-preservation work (KV slot save/restore on a shared GPU) is a separate,
//! optional layer and is not part of this core.
//!
//! On a tier error the orchestrator walks the [`fallback_order`]; on a fully
//! failed turn it records that so the *next* turn routes one rung stronger.

use crate::backend::{Backend, BackendError, Completion, StreamEvent};
use crate::fleet_config::{FleetConfig, TierConfig};
use crate::message::{Conversation, Role};
use crate::router::{self, ModeHint, RouteInput};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

/// Live routing signals the frontend updates and the orchestrator reads at route
/// time. Shared via `Arc`, mutated from the UI thread, read on the stream thread —
/// hence atomics + a small mutex. The `Backend` trait only sees the conversation,
/// so the mode and any manual pin travel through here.
#[derive(Debug)]
pub struct FleetSignal {
    mode: AtomicU8,
    manual: Mutex<Option<String>>,
}

impl Default for FleetSignal {
    fn default() -> Self {
        FleetSignal {
            mode: AtomicU8::new(ModeHint::Normal.as_u8()),
            manual: Mutex::new(None),
        }
    }
}

impl FleetSignal {
    pub fn new() -> FleetSignal {
        FleetSignal::default()
    }

    /// Set the current editor mode (call when the mode cycles).
    pub fn set_mode(&self, mode: ModeHint) {
        self.mode.store(mode.as_u8(), Ordering::Relaxed);
    }

    /// The current editor mode.
    pub fn mode(&self) -> ModeHint {
        ModeHint::from_u8(self.mode.load(Ordering::Relaxed))
    }

    /// Set or clear the manual tier pin (`/model deep` / `/model auto`).
    pub fn set_manual(&self, pin: Option<String>) {
        *self.lock_manual() = pin;
    }

    /// The current manual pin, if any.
    pub fn manual(&self) -> Option<String> {
        self.lock_manual().clone()
    }

    fn lock_manual(&self) -> MutexGuard<'_, Option<String>> {
        // A poisoned lock just means a panicked writer; the data is still valid.
        self.manual.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// One tier in the running orchestrator: its key + the backend that fills it.
struct Tier {
    key: String,
    backend: Arc<dyn Backend>,
}

/// A [`Backend`] that routes across tiers (see module docs).
pub struct OrchestratorBackend {
    tiers: Vec<Tier>,
    config: FleetConfig,
    signal: Arc<FleetSignal>,
    name: String,
    /// The tier chosen for the most recent request (for `/fleet` and the status
    /// bar).
    active: Mutex<String>,
    /// Whether the last completed request exhausted every tier with an error.
    last_failed: AtomicBool,
}

impl OrchestratorBackend {
    /// Build from a [`FleetConfig`]. `build` turns a tier into a concrete backend
    /// (the binary passes a closure over `OpenAiBackend::from_config`); tests pass
    /// a stub. Returns `None` if no tier produced a backend.
    pub fn new(
        config: FleetConfig,
        build: impl Fn(&TierConfig) -> Option<Arc<dyn Backend>>,
        signal: Arc<FleetSignal>,
    ) -> Option<OrchestratorBackend> {
        let mut tiers = Vec::new();
        for tc in &config.tiers {
            if let Some(backend) = build(tc) {
                tiers.push(Tier {
                    key: tc.key.clone(),
                    backend,
                });
            }
        }
        if tiers.is_empty() {
            return None;
        }
        let keys = tiers
            .iter()
            .map(|t| t.key.as_str())
            .collect::<Vec<_>>()
            .join("/");
        let name = format!("fleet ({} tiers: {keys})", tiers.len());
        let active = tiers[0].key.clone();
        Some(OrchestratorBackend {
            tiers,
            config,
            signal,
            name,
            active: Mutex::new(active),
            last_failed: AtomicBool::new(false),
        })
    }

    /// A handle to the shared signal, so the frontend can push mode / pin updates.
    pub fn signal(&self) -> Arc<FleetSignal> {
        Arc::clone(&self.signal)
    }

    /// The tier key chosen for the most recent request.
    pub fn active_tier(&self) -> String {
        self.active
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_active(&self, key: &str) {
        let mut g = self.active.lock().unwrap_or_else(|e| e.into_inner());
        if *g != key {
            *g = key.to_string();
        }
    }

    /// The tier indices to try, in order, for this conversation.
    fn order_indices(&self, conv: &Conversation) -> Vec<usize> {
        let prompt = last_user(conv);
        let prev_assistant = last_assistant(conv);
        let manual = self.signal.manual();
        let input = RouteInput {
            prompt: &prompt,
            prev_assistant: &prev_assistant,
            mode: self.signal.mode(),
            manual: manual.as_deref(),
            prior_failed: self.last_failed.load(Ordering::Relaxed),
        };
        let decision = router::route_decision(&input, &self.config);
        if std::env::var_os("ZERO_FLEET_DEBUG").is_some() {
            // Opt-in routing trace (set ZERO_FLEET_DEBUG=1) — makes delegation
            // visible on stderr without a model call or the TUI status chip.
            eprintln!(
                "zero: fleet routed -> {} [confidence {:?}]",
                decision.tier, decision.confidence
            );
        }
        router::fallback_order(&self.config, &decision.tier)
            .iter()
            .filter_map(|k| self.tiers.iter().position(|t| &t.key == k))
            .collect()
    }
}

/// The text of the last user message — the thing routing reads.
fn last_user(conv: &Conversation) -> String {
    last_of(conv, Role::User)
}

/// The text of the last assistant message — read when the latest user message is a
/// bare continuation, so "do it" routes on what the model just proposed.
fn last_assistant(conv: &Conversation) -> String {
    last_of(conv, Role::Assistant)
}

fn last_of(conv: &Conversation, role: Role) -> String {
    conv.messages
        .iter()
        .rev()
        .find(|m| m.role == role)
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

impl Backend for OrchestratorBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn stream(
        &self,
        conv: &Conversation,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<(), BackendError> {
        let order = self.order_indices(conv);
        if order.is_empty() {
            return Err(BackendError("fleet: no tiers available".to_string()));
        }
        let mut last_err: Option<BackendError> = None;
        for idx in order {
            let tier = &self.tiers[idx];
            self.set_active(&tier.key);
            let mut emitted = false;
            let res = tier.backend.stream(conv, &mut |ev| {
                if matches!(ev, StreamEvent::Token(_)) {
                    emitted = true;
                }
                sink(ev);
            });
            match res {
                Ok(()) => {
                    self.last_failed.store(false, Ordering::Relaxed);
                    return Ok(());
                }
                Err(e) => {
                    // If the tier already streamed text, falling back would
                    // double up the output — surface the error instead.
                    if emitted {
                        self.last_failed.store(true, Ordering::Relaxed);
                        return Err(e);
                    }
                    last_err = Some(e);
                }
            }
        }
        self.last_failed.store(true, Ordering::Relaxed);
        Err(last_err.unwrap_or_else(|| BackendError("fleet: all tiers failed".to_string())))
    }

    fn complete(
        &self,
        conv: &Conversation,
        tools: &[crate::tools::ToolDef],
        timeout: Duration,
    ) -> Result<Completion, BackendError> {
        let order = self.order_indices(conv);
        if order.is_empty() {
            return Err(BackendError("fleet: no tiers available".to_string()));
        }
        let mut last_err: Option<BackendError> = None;
        for idx in order {
            self.set_active(&self.tiers[idx].key);
            match self.tiers[idx].backend.complete(conv, tools, timeout) {
                Ok(c) => {
                    self.last_failed.store(false, Ordering::Relaxed);
                    return Ok(c);
                }
                Err(e) => last_err = Some(e),
            }
        }
        self.last_failed.store(true, Ordering::Relaxed);
        Err(last_err.unwrap_or_else(|| BackendError("fleet: all tiers failed".to_string())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StopReason;
    use crate::message::Message;
    use std::collections::HashSet;

    /// A scriptable backend: records each call, fails when its key is in the
    /// shared `failing` set, and (optionally) emits a token before failing.
    struct Mock {
        name: String,
        calls: Arc<Mutex<Vec<String>>>,
        failing: Arc<Mutex<HashSet<String>>>,
        emit_before_fail: bool,
    }

    impl Mock {
        fn fails(&self) -> bool {
            self.failing.lock().unwrap().contains(&self.name)
        }
    }

    impl Backend for Mock {
        fn name(&self) -> &str {
            &self.name
        }
        fn stream(
            &self,
            _conv: &Conversation,
            sink: &mut dyn FnMut(StreamEvent),
        ) -> Result<(), BackendError> {
            self.calls.lock().unwrap().push(self.name.clone());
            if self.fails() {
                if self.emit_before_fail {
                    sink(StreamEvent::Token("partial".to_string()));
                }
                return Err(BackendError(format!("{} down", self.name)));
            }
            sink(StreamEvent::Token(format!("hi from {}", self.name)));
            sink(StreamEvent::Done(StopReason::EndTurn));
            Ok(())
        }
        fn complete(
            &self,
            _conv: &Conversation,
            _tools: &[crate::tools::ToolDef],
            _timeout: Duration,
        ) -> Result<Completion, BackendError> {
            self.calls.lock().unwrap().push(self.name.clone());
            if self.fails() {
                return Err(BackendError(format!("{} down", self.name)));
            }
            Ok(Completion {
                content: format!("done by {}", self.name),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    struct Harness {
        orch: OrchestratorBackend,
        calls: Arc<Mutex<Vec<String>>>,
        failing: Arc<Mutex<HashSet<String>>>,
    }

    impl Harness {
        fn new(emit_before_fail: bool) -> Harness {
            let calls = Arc::new(Mutex::new(Vec::new()));
            let failing = Arc::new(Mutex::new(HashSet::new()));
            let cfg = FleetConfig::parse(
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
            .unwrap();
            let c = calls.clone();
            let f = failing.clone();
            let orch = OrchestratorBackend::new(
                cfg,
                move |tc| {
                    Some(Arc::new(Mock {
                        name: tc.key.clone(),
                        calls: c.clone(),
                        failing: f.clone(),
                        emit_before_fail,
                    }) as Arc<dyn Backend>)
                },
                Arc::new(FleetSignal::new()),
            )
            .unwrap();
            Harness {
                orch,
                calls,
                failing,
            }
        }

        fn fail(&self, keys: &[&str]) {
            let mut g = self.failing.lock().unwrap();
            for k in keys {
                g.insert((*k).to_string());
            }
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    fn user(text: &str) -> Conversation {
        let mut c = Conversation::new();
        c.push(Message::user(text));
        c
    }

    fn stream_text(
        orch: &OrchestratorBackend,
        conv: &Conversation,
    ) -> Result<String, BackendError> {
        let mut out = String::new();
        orch.stream(conv, &mut |ev| {
            if let StreamEvent::Token(t) = ev {
                out.push_str(&t);
            }
        })?;
        Ok(out)
    }

    #[test]
    fn name_describes_the_fleet() {
        let h = Harness::new(false);
        assert!(h.orch.name().contains("fleet"));
        assert!(h.orch.name().contains("deep"));
    }

    #[test]
    fn new_returns_none_when_nothing_builds() {
        let cfg =
            FleetConfig::parse("enabled=true\n[[tier]]\nkey=\"a\"\nwhere=\"http://a:1\"").unwrap();
        let got = OrchestratorBackend::new(cfg, |_| None, Arc::new(FleetSignal::new()));
        assert!(got.is_none());
    }

    #[test]
    fn routes_simple_prompt_to_fast() {
        let h = Harness::new(false);
        let out = stream_text(&h.orch, &user("list the files")).unwrap();
        assert_eq!(h.orch.active_tier(), "fast");
        assert!(out.contains("fast"));
        assert_eq!(h.calls(), vec!["fast"]);
    }

    #[test]
    fn routes_plan_mode_to_deep() {
        let h = Harness::new(false);
        h.orch.signal().set_mode(ModeHint::Plan);
        let out = h
            .orch
            .complete(&user("anything"), &[], Duration::from_secs(1))
            .unwrap();
        assert_eq!(h.orch.active_tier(), "deep");
        assert!(out.content.contains("deep"));
    }

    #[test]
    fn manual_pin_overrides_routing() {
        let h = Harness::new(false);
        h.orch.signal().set_mode(ModeHint::Plan); // would route deep…
        h.orch.signal().set_manual(Some("fast".to_string())); // …but pin wins
        h.orch
            .complete(&user("plan it"), &[], Duration::from_secs(1))
            .unwrap();
        assert_eq!(h.orch.active_tier(), "fast");
        // clearing the pin restores auto routing
        h.orch.signal().set_manual(None);
        h.orch
            .complete(&user("plan it"), &[], Duration::from_secs(1))
            .unwrap();
        assert_eq!(h.orch.active_tier(), "deep");
    }

    #[test]
    fn complete_falls_back_through_the_chain() {
        let h = Harness::new(false);
        h.fail(&["balanced", "deep"]); // baseline + escalation both down
        let out = h
            .orch
            .complete(
                &user("do a normal task here please"),
                &[],
                Duration::from_secs(1),
            )
            .unwrap();
        assert!(out.content.contains("fast"));
        // order: baseline balanced → deep → fast
        assert_eq!(h.calls(), vec!["balanced", "deep", "fast"]);
        assert_eq!(h.orch.active_tier(), "fast");
    }

    #[test]
    fn complete_errors_when_all_tiers_fail() {
        let h = Harness::new(false);
        h.fail(&["deep", "balanced", "fast"]);
        let err = h
            .orch
            .complete(&user("list files"), &[], Duration::from_secs(1))
            .unwrap_err();
        assert!(err.0.contains("down") || err.0.contains("all tiers"));
    }

    #[test]
    fn stream_falls_back_when_no_tokens_emitted() {
        let h = Harness::new(false); // failing tiers emit nothing
        h.fail(&["fast"]);
        let out = stream_text(&h.orch, &user("list files")).unwrap();
        // fast failed cleanly → fall back along ladder to deep
        assert!(out.contains("deep"));
        assert_eq!(h.calls(), vec!["fast", "deep"]);
    }

    #[test]
    fn stream_does_not_fall_back_after_partial_output() {
        let h = Harness::new(true); // a failing tier emits a token first
        h.fail(&["fast"]);
        let err = stream_text(&h.orch, &user("list files")).unwrap_err();
        assert!(err.0.contains("fast down"));
        assert_eq!(h.calls(), vec!["fast"]); // did NOT try the next tier
    }

    #[test]
    fn prior_failure_escalates_the_next_turn() {
        let h = Harness::new(false);
        // First turn: a normal prompt routes to balanced, but everything fails →
        // records the failure.
        h.fail(&["deep", "balanced", "fast"]);
        let _ = h.orch.complete(
            &user("a normal task to do now"),
            &[],
            Duration::from_secs(1),
        );
        // Recover the fleet; next normal turn should START one rung up (deep),
        // because the prior turn failed.
        {
            h.failing.lock().unwrap().clear();
        }
        h.orch
            .complete(
                &user("a normal task to do now"),
                &[],
                Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(h.orch.active_tier(), "deep");
    }

    #[test]
    fn bare_do_it_routes_on_the_models_long_proposal() {
        let h = Harness::new(false);
        let plan = "Here's the plan: refactor the parser, add a streaming decoder, \
                    thread the new error type through every call site, then update \
                    the tests and golden fixtures — a lot of files, ordering matters. \
                    Proceed?";
        let mut conv = Conversation::new();
        conv.push(Message::user("can you fix the parser?"));
        conv.push(Message::assistant(plan));
        conv.push(Message::user("do it"));
        h.orch.complete(&conv, &[], Duration::from_secs(1)).unwrap();
        // Without the continuation rule a 5-char "do it" would route to baseline;
        // reading the long proposal routes it to deep.
        assert_eq!(h.orch.active_tier(), "deep");
    }

    #[test]
    fn debug_env_logs_route_without_breaking() {
        let h = Harness::new(false);
        std::env::set_var("ZERO_FLEET_DEBUG", "1");
        let out = h
            .orch
            .complete(&user("list files"), &[], Duration::from_secs(1));
        std::env::remove_var("ZERO_FLEET_DEBUG");
        assert!(out.is_ok());
        assert_eq!(h.orch.active_tier(), "fast");
    }

    #[test]
    fn empty_conversation_still_routes() {
        let h = Harness::new(false);
        // No user message → baseline; must not panic.
        let out = h
            .orch
            .complete(&Conversation::new(), &[], Duration::from_secs(1))
            .unwrap();
        assert!(out.content.contains("balanced"));
    }

    #[test]
    fn signal_mode_and_manual_roundtrip() {
        let sig = FleetSignal::new();
        assert_eq!(sig.mode(), ModeHint::Normal);
        sig.set_mode(ModeHint::AutoAccept);
        assert_eq!(sig.mode(), ModeHint::AutoAccept);
        assert!(sig.manual().is_none());
        sig.set_manual(Some("deep".to_string()));
        assert_eq!(sig.manual().as_deref(), Some("deep"));
        sig.set_manual(None);
        assert!(sig.manual().is_none());
    }
}
