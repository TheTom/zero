//! Deterministic bash replay suite — drives the **real** agentic executor through
//! the public headless API ([`App::run_once`]) with a scripted backend that emits
//! bash tool calls, then asserts the no-useful-information-lost contract on what
//! actually came back.
//!
//! Why this exists: a real session log showed shell output was 88.5% of all
//! tool-result bytes (broad `grep`, `gh pr diff`, build logs). Phase 1-4 added
//! recoverable content-aware compression; this proves it end-to-end on
//! shell-shaped output, exercising gate → run (`sh -c`) → spill → compress as one
//! pipeline. Commands are chosen to be deterministic (`seq`, `echo`, `printf`,
//! `grep` on a fixture we write), so the assertions are stable across runs.
//!
//! For the live-model variant (the real model driving bash through the harness),
//! see `bash_live.rs`, which is `#[ignore]`d and env-gated.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use zero_core::backend::{Backend, BackendError, Completion, StreamEvent};
use zero_core::message::{Conversation, Role, ToolCall};
use zero_core::tools::ToolDef;
use zero_tui::{App, Input};

/// A no-op input: always at EOF. Headless runs never read keys.
struct EofInput;
impl Input for EofInput {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

/// A backend that replays a fixed list of bash commands as tool calls (one per
/// round), then answers with plain text. Deterministic: no model, no network.
struct ReplayBackend {
    /// Each entry: (tool_call_id, command). Issued one per `complete` round.
    cmds: Vec<(String, String)>,
    round: Mutex<usize>,
}

impl ReplayBackend {
    fn new(cmds: &[(&str, &str)]) -> Self {
        ReplayBackend {
            cmds: cmds
                .iter()
                .map(|(id, c)| (id.to_string(), c.to_string()))
                .collect(),
            round: Mutex::new(0),
        }
    }
}

impl Backend for ReplayBackend {
    fn name(&self) -> &str {
        "replay"
    }
    fn stream(
        &self,
        _c: &Conversation,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<(), BackendError> {
        sink(StreamEvent::Done(zero_core::backend::StopReason::EndTurn));
        Ok(())
    }
    fn complete(
        &self,
        _c: &Conversation,
        _t: &[ToolDef],
        _to: Duration,
    ) -> Result<Completion, BackendError> {
        let mut r = self.round.lock().unwrap();
        let i = *r;
        *r += 1;
        if i < self.cmds.len() {
            let (id, cmd) = &self.cmds[i];
            let args = zero_core::json::Value::Object(vec![(
                "command".to_string(),
                zero_core::json::Value::Str(cmd.clone()),
            )])
            .to_json();
            Ok(Completion {
                content: String::new(),
                tool_calls: vec![ToolCall::new(id.as_str(), "bash", args)],
            })
        } else {
            Ok(Completion {
                content: "done".to_string(),
                tool_calls: vec![],
            })
        }
    }
}

/// Build + run a headless turn that replays `cmds`, with the per-result cap set
/// to `cap` and a fresh artifact dir. Returns the finished App for inspection.
fn run_replay(cmds: &[(&str, &str)], cap: usize, art: &PathBuf) -> App<EofInput, Vec<u8>> {
    std::fs::create_dir_all(art).unwrap();
    let backend: Arc<dyn Backend> = Arc::new(ReplayBackend::new(cmds));
    let mut app = App::new(EofInput, Vec::new(), backend, None);
    let cfg = zero_core::Config {
        max_tool_output: cap,
        max_turn_output: 1_000_000, // don't let the per-turn budget interfere here
        ..zero_core::Config::default()
    };
    app.set_config(cfg, None, None);
    app.set_artifact_dir(Some(art.clone()));
    app.set_tools_enabled(true);
    app.run_once("go").unwrap();
    app
}

/// The tool-result messages fed back to the model, in order.
fn tool_results(app: &App<EofInput, Vec<u8>>) -> Vec<String> {
    app.conversation()
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .map(|m| m.content.clone())
        .collect()
}

fn unique_dir(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("zero-bashreplay-{}-{tag}", std::process::id()))
}

#[test]
fn small_output_passes_through_uncompressed() {
    let dir = unique_dir("small");
    let app = run_replay(&[("s1", "echo hello-world")], 4096, &dir);
    let r = &tool_results(&app)[0];
    assert!(r.contains("hello-world"), "missing output: {r}");
    assert!(r.contains("[exit 0]"), "missing exit code: {r}");
    // Small output is NOT compressed.
    assert!(!r.contains("elided"), "should not be compressed: {r}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn large_output_is_capped_but_fully_recoverable() {
    let dir = unique_dir("large");
    // seq 1 5000 is ~23KB of deterministic output.
    let app = run_replay(&[("b1", "seq 1 5000")], 512, &dir);
    let r = &tool_results(&app)[0];
    // Capped in the model's view.
    assert!(r.len() < 2000, "not capped: {} bytes", r.len());
    assert!(
        r.contains("elided") || r.contains("compressed"),
        "no cap marker: {r}"
    );
    // Recoverable: the full output spilled byte-identical and is referenced.
    assert!(r.contains("full output:"), "no re-fetch path: {r}");
    let art = dir.join("out-b1.txt");
    let full = std::fs::read_to_string(&art).expect("artifact spilled");
    // The full output has both ends of the range + the exit line — nothing lost.
    assert!(full.contains("\n1\n") || full.starts_with("1\n"));
    assert!(full.contains("5000"));
    assert!(full.contains("[exit 0]"));
    assert!(
        full.len() > r.len() * 3,
        "artifact should dwarf the capped view"
    );
    // The saving was measured.
    assert!(app.context_stats().capped_saved > 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn grep_output_keeps_file_line_refs() {
    let dir = unique_dir("grep");
    std::fs::create_dir_all(&dir).unwrap();
    // Deterministic fixture: 60 lines, "needle" on a known subset.
    let fixture = dir.join("fixture.txt");
    let mut body = String::new();
    for i in 1..=60 {
        if i % 10 == 0 {
            body.push_str(&format!("line {i}: needle here\n"));
        } else {
            body.push_str(&format!("line {i}: filler\n"));
        }
    }
    std::fs::write(&fixture, body).unwrap();
    let cmd = format!("grep -n needle {}", fixture.display());
    let app = run_replay(&[("g1", &cmd)], 4096, &dir);
    let r = &tool_results(&app)[0];
    // Every match's line number survives (10,20,30,40,50,60).
    for n in [10, 20, 30, 40, 50, 60] {
        assert!(r.contains(&format!("{n}:")), "lost line {n} ref: {r}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stderr_and_nonzero_exit_are_captured() {
    let dir = unique_dir("stderr");
    let app = run_replay(&[("e1", "echo to-stderr >&2; exit 7")], 4096, &dir);
    let r = &tool_results(&app)[0];
    assert!(r.contains("to-stderr"), "stderr not captured: {r}");
    assert!(r.contains("[exit 7]"), "nonzero exit not reported: {r}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dangerous_command_is_refused_and_never_executed() {
    let dir = unique_dir("danger");
    std::fs::create_dir_all(&dir).unwrap();
    // Proof of non-execution: the chain would create a sentinel BEFORE the rm.
    // The safety classifier flags the whole chain (rm -rf /), so it's refused and
    // NOTHING runs — the sentinel must not exist afterward.
    let sentinel = dir.join("should-not-exist");
    let cmd = format!("touch {} && rm -rf /", sentinel.display());
    let app = run_replay(&[("d1", &cmd)], 4096, &dir);
    let r = &tool_results(&app)[0];
    assert!(r.contains("refused"), "danger not refused: {r}");
    assert!(r.contains("destructive"), "no destructive reason: {r}");
    assert!(
        !sentinel.exists(),
        "REFUSED COMMAND STILL RAN — sentinel created"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn replays_a_multi_command_session_bounding_each_result() {
    // Mimics the real Log-B shape: several shell calls in one turn (a big seq, a
    // grep on a fixture, a small echo). Each result is independently bounded and
    // the high-signal content of each survives.
    let dir = unique_dir("multi");
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = dir.join("src.txt");
    let mut body = String::new();
    for i in 1..=40 {
        body.push_str(&format!("fn item_{i}() {{}} // TARGET\n"));
    }
    std::fs::write(&fixture, body).unwrap();

    let grep_cmd = format!("grep -n TARGET {}", fixture.display());
    let cmds: &[(&str, &str)] = &[
        ("m1", "seq 1 3000"),
        ("m2", &grep_cmd),
        ("m3", "echo final-step-ok"),
    ];
    let app = run_replay(cmds, 400, &dir);
    let results = tool_results(&app);
    assert_eq!(results.len(), 3, "expected 3 tool results");

    // 1) big seq capped + recoverable.
    assert!(results[0].contains("elided") || results[0].contains("compressed"));
    assert!(results[0].contains("full output:"));
    assert!(dir.join("out-m1.txt").exists());

    // 2) grep refs survive (a sample of the 40 matches).
    for n in [1, 20, 40] {
        assert!(results[1].contains(&format!("{n}:")), "lost grep ref {n}");
    }

    // 3) small echo passes through verbatim.
    assert!(results[2].contains("final-step-ok"));
    assert!(!results[2].contains("elided"));

    // Cumulative savings were measured across the turn.
    assert!(app.context_stats().capped_saved > 0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn compression_through_the_executor_is_deterministic() {
    // Same command sequence twice → byte-identical model-facing results.
    let d1 = unique_dir("det1");
    let d2 = unique_dir("det2");
    let cmds: &[(&str, &str)] = &[("x1", "seq 1 4000")];
    let a = run_replay(cmds, 300, &d1);
    let b = run_replay(cmds, 300, &d2);
    // Strip the artifact path (it embeds the dir, which differs) before comparing.
    let strip = |s: &str| -> String {
        s.lines()
            .map(|l| {
                if let Some(i) = l.find("full output:") {
                    l[..i].to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&tool_results(&a)[0]), strip(&tool_results(&b)[0]));
    std::fs::remove_dir_all(&d1).ok();
    std::fs::remove_dir_all(&d2).ok();
}
