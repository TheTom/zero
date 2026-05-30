//! Live-model integration: drive the REAL model through the headless harness and
//! verify it can use the `bash` tool and relay deterministic output.
//!
//! This is the "call the model directly through the harness and verify output"
//! half of the bash test plan. A real LLM is not deterministic, so this can't be
//! a gate test — it is `#[ignore]`d AND env-gated, and skips gracefully when the
//! model isn't reachable. Run it on demand against gx10 (or any OpenAI-compatible
//! server):
//!
//!   ZERO_LIVE_MODEL=1 cargo test -p zero-tui --test bash_live -- --ignored --nocapture
//!
//! Config resolution: ZERO_URL / ZERO_MODEL env vars override; otherwise the
//! standard ~/.zero/config.json is used.

use std::sync::Arc;
use zero_core::backend::Backend;
use zero_core::openai::OpenAiBackend;
use zero_core::Config;
use zero_tui::{App, Input};

struct EofInput;
impl Input for EofInput {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

/// Resolve a live config from env vars, then ~/.zero/config.json. Returns None if
/// no backend can be determined (test then skips).
fn live_config() -> Option<Config> {
    let mut cfg = std::env::var_os("HOME")
        .map(|h| {
            std::path::PathBuf::from(h)
                .join(zero_core::brand::dot_dir())
                .join("config.json")
        })
        .and_then(|p| Config::load(p).ok())
        .unwrap_or_default();
    if let Ok(url) = std::env::var("ZERO_URL") {
        cfg.base_url = Some(url);
    }
    if let Ok(model) = std::env::var("ZERO_MODEL") {
        cfg.model = model;
    }
    cfg.has_backend().then_some(cfg)
}

#[test]
#[ignore = "hits a real model; run with ZERO_LIVE_MODEL=1 -- --ignored"]
fn live_model_runs_bash_through_the_harness() {
    if std::env::var("ZERO_LIVE_MODEL").is_err() {
        eprintln!("skipping: set ZERO_LIVE_MODEL=1 to run the live test");
        return;
    }
    let Some(cfg) = live_config() else {
        eprintln!("skipping: no backend configured (ZERO_URL / ~/.zero/config.json)");
        return;
    };
    let Some(backend) = OpenAiBackend::from_config(&cfg) else {
        eprintln!("skipping: backend could not be built from config");
        return;
    };
    eprintln!(
        "live: {} @ {}",
        cfg.model,
        cfg.base_url.as_deref().unwrap_or("?")
    );

    let dir = std::env::temp_dir().join(format!("zero-bashlive-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let backend: Arc<dyn Backend> = Arc::new(backend);
    let mut app = App::new(EofInput, Vec::new(), backend, None);
    app.set_config(cfg, None, None);
    app.set_artifact_dir(Some(dir.clone()));
    app.set_tools_enabled(true);

    // A unique sentinel so we know the bash tool actually ran (not the model
    // hallucinating the answer). Ask for exactly that echo and the output back.
    let sentinel = "zero-live-probe-4242";
    let prompt = format!(
        "Use the bash tool to run exactly this command: echo {sentinel}\n\
         Then tell me, in your reply, the exact output the command produced."
    );
    let reply = app.run_once(&prompt).expect("run_once");
    eprintln!("\n--- model reply ---\n{reply}\n-------------------");

    // The bash tool ran and its result entered the conversation.
    let ran_bash = app
        .conversation()
        .messages
        .iter()
        .any(|m| m.content.contains(sentinel));
    assert!(
        ran_bash,
        "the sentinel never appeared in the conversation — bash tool likely didn't run"
    );
    // And the model relayed it.
    assert!(
        reply.contains(sentinel),
        "model reply did not contain the command output"
    );
    std::fs::remove_dir_all(&dir).ok();
}
