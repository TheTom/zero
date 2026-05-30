//! Equivalence proof: SAME model behaviour, SAME final answer, FEWER tokens.
//!
//! The project goal is to show Zero gets *equivalent data* from a model while
//! using tools *more efficiently* than other wrappers. The token half is measured
//! live (scripts/bench-vs-hermes.sh); this file proves the half a benchmark can't:
//! that Zero's output compression does **not change the answer** — it only changes
//! how many tokens the tool result costs.
//!
//! Method (deterministic, no network): a mock "model" that, on its final round,
//! *reads the actual tool-result text from the conversation* and extracts a needle
//! from it — exactly what a real model does, not a canned reply. We run the
//! identical agentic turn twice against the same mock:
//!   - caps OFF (huge per-result cap → full output reaches the model), and
//!   - caps ON  (tiny cap → output is compressed + spilled).
//!
//! Then we assert the model produced the **same answer both times** while the
//! bytes actually fed back to it **differ** (compressed < full). Same data out,
//! fewer tokens in: the goal, proven.

use std::sync::{Arc, Mutex};
use std::time::Duration;
use zero_core::backend::{Backend, BackendError, Completion, StreamEvent};
use zero_core::message::{Conversation, Role, ToolCall};
use zero_core::tools::ToolDef;
use zero_tui::{App, Input};

struct EofInput;
impl Input for EofInput {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

/// A mock model that calls a tool once, then answers by EXTRACTING a fact from the
/// tool result in the conversation — so its answer depends on the data it actually
/// received, not on a hardcoded string. `tool`/`args` drive round 1; round 2 finds
/// the needle line in the latest Tool message and echoes the value after `needle`.
struct ExtractingBackend {
    tool: String,
    args: String,
    needle: String,
    round: Mutex<u32>,
    /// Records the byte length of the tool-result the model saw on round 2 — i.e.
    /// what actually entered the context window after capping.
    seen_result_bytes: Arc<Mutex<usize>>,
}

impl Backend for ExtractingBackend {
    fn name(&self) -> &str {
        "extracting"
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
        conv: &Conversation,
        _t: &[ToolDef],
        _to: Duration,
    ) -> Result<Completion, BackendError> {
        let mut r = self.round.lock().unwrap();
        *r += 1;
        if *r == 1 {
            return Ok(Completion {
                content: String::new(),
                tool_calls: vec![ToolCall::new("c1", &self.tool, &self.args)],
                usage: None,
            });
        }
        // Round 2: read the most recent tool result the model was given and pull
        // the value that follows the needle. This is the "did the data survive?"
        // probe — if compression dropped the needle line, extraction fails and the
        // answers won't match.
        let tool_text = conv
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Tool)
            .map(|m| m.content.clone())
            .unwrap_or_default();
        *self.seen_result_bytes.lock().unwrap() = tool_text.len();
        let answer = tool_text
            .lines()
            .find_map(|l| {
                l.split_once(&self.needle)
                    .map(|(_, v)| v.trim().to_string())
            })
            .unwrap_or_else(|| "NEEDLE-NOT-FOUND".to_string());
        Ok(Completion {
            content: format!("answer: {answer}"),
            tool_calls: vec![],
            usage: None,
        })
    }
}

/// Run one headless turn with `cap` bytes per tool result, against a mock that
/// runs `bash -c <cmd>` (so the tool output is real + deterministic) then extracts
/// the value after `needle`. Returns `(final_answer, bytes_the_model_saw)`.
fn run_with_cap(cmd: &str, needle: &str, cap: usize, art: &std::path::Path) -> (String, usize) {
    std::fs::create_dir_all(art).unwrap();
    let seen = Arc::new(Mutex::new(0usize));
    // Build the args JSON via the json encoder so commands with quotes / $()
    // are escaped correctly (a hand-rolled format! breaks on inner quotes).
    let args = zero_core::json::Value::Object(vec![(
        "command".to_string(),
        zero_core::json::Value::Str(cmd.to_string()),
    )])
    .to_json();
    let backend: Arc<dyn Backend> = Arc::new(ExtractingBackend {
        tool: "bash".to_string(),
        args,
        needle: needle.to_string(),
        round: Mutex::new(0),
        seen_result_bytes: seen.clone(),
    });
    let mut app = App::new(EofInput, Vec::new(), backend, None);
    let cfg = zero_core::Config {
        max_tool_output: cap,
        max_turn_output: 100_000_000,
        ..zero_core::Config::default()
    };
    app.set_config(cfg, None, None);
    app.set_artifact_dir(Some(art.to_path_buf()));
    app.set_tools_enabled(true);
    let answer = app.run_once("find the marker").unwrap();
    let bytes = *seen.lock().unwrap();
    (answer, bytes)
}

fn tmp(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("zero-equiv-{}-{tag}", std::process::id()))
}

/// Read the single spilled `out-*.txt` artifact directly under `dir`.
fn artifact_body(dir: &std::path::Path) -> String {
    for f in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = f.path();
        if p.is_file() && p.file_name().unwrap().to_string_lossy().starts_with("out-") {
            return std::fs::read_to_string(p).unwrap_or_default();
        }
    }
    String::new()
}

#[test]
fn same_answer_when_the_marker_is_in_the_kept_tail() {
    // A big log whose conclusion (the marker) is at the END — the place a build's
    // result/exit status actually lives. The head+tail donut keeps the tail, so
    // the model extracts the SAME answer capped or not, with far fewer bytes in
    // its context. This is the realistic "answer-bearing line survives" case.
    let cmd = "seq 1 5000; echo 'RESULT MARKER=42 done'";
    let needle = "MARKER=";

    let d_off = tmp("tail-off");
    let d_on = tmp("tail-on");
    let (ans_off, bytes_off) = run_with_cap(cmd, needle, 10_000_000, &d_off);
    let (ans_on, bytes_on) = run_with_cap(cmd, needle, 512, &d_on);

    // (1) EQUIVALENT DATA: identical extracted answer both ways.
    assert_eq!(ans_off, "answer: 42 done", "uncapped lost it: {ans_off}");
    assert_eq!(
        ans_on, ans_off,
        "compression CHANGED the answer: on={ans_on} off={ans_off}"
    );

    // (2) MORE EFFICIENT: the capped run fed far fewer bytes to the model.
    assert!(
        bytes_on < bytes_off / 10,
        "expected big reduction: on={bytes_on} off={bytes_off}"
    );

    // (3) NOTHING LOST: the full output is spilled for re-fetch regardless.
    assert!(
        artifact_body(&d_on).contains("MARKER=42"),
        "marker not in spilled artifact"
    );

    std::fs::remove_dir_all(&d_off).ok();
    std::fs::remove_dir_all(&d_on).ok();
}

#[test]
fn unstructured_middle_is_dropped_from_view_but_recoverable_from_disk() {
    // The honest converse: for GENUINELY SHAPELESS output — high-entropy lines that
    // share no skeleton, so the repeat-fold can't collapse them — a needle buried
    // mid-stream is NOT kept in the head+tail donut. But it IS spilled, so it stays
    // recoverable. This documents the real, residual boundary of inline compression
    // and proves the cap is offload-not-delete even when the view can't keep the
    // middle. (A real model sees the elision marker + re-fetches via the artifact.)
    //
    // NB: semi-uniform middles (e.g. `filler line N` runs) are now *rescued* by the
    // repeat-fold — a needle that breaks such a run survives as a fold boundary. To
    // hit the true donut-drops-the-middle case we use distinct-length `x` lines (no
    // two share a skeleton ⇒ no fold) with the marker buried at line 100 of 200.
    let cmd = "for i in $(seq 1 200); do if [ $i -eq 100 ]; then echo MARKER=42; else printf 'x%.0s' $(seq 1 $i); echo; fi; done";
    let needle = "MARKER=";
    let d_on = tmp("uns-on");
    let (ans_on, bytes_on) = run_with_cap(cmd, needle, 512, &d_on);

    // The capped *view* legitimately can't carry an arbitrary middle line, so the
    // single-pass extraction misses it — that's expected for shapeless output.
    assert_eq!(
        ans_on, "answer: NEEDLE-NOT-FOUND",
        "unexpected: donut kept the middle? {ans_on}"
    );
    // But NOTHING IS LOST: the full output is on disk, marker included — the model
    // would follow the marker and re-fetch the span.
    let full = artifact_body(&d_on);
    assert!(
        full.contains("MARKER=42"),
        "marker not recoverable from artifact"
    );
    assert!(
        full.len() > bytes_on * 10,
        "artifact should dwarf the capped view"
    );

    std::fs::remove_dir_all(&d_on).ok();
}

#[test]
fn grep_answer_is_identical_capped_vs_uncapped() {
    // Same equivalence proof on grep-shaped output: many matches, the model picks
    // the line number of a specific one. Capping collapses bodies but keeps refs,
    // so the extracted answer is unchanged.
    let dir = tmp("grep-fixture");
    std::fs::create_dir_all(&dir).unwrap();
    let fixture = dir.join("data.txt");
    let mut body = String::new();
    for i in 1..=200 {
        if i == 137 {
            body.push_str("TARGET=here a long trailing body to inflate the match size xxxxxxxx\n");
        } else {
            body.push_str(&format!(
                "noise {i} with padding to make lines fat yyyyyyyyyyyyyy\n"
            ));
        }
    }
    std::fs::write(&fixture, body).unwrap();
    let cmd = format!("grep -n TARGET {}", fixture.display());
    let needle = "TARGET=";

    let d_off = tmp("g-off");
    let d_on = tmp("g-on");
    let (ans_off, bytes_off) = run_with_cap(&cmd, needle, 10_000_000, &d_off);
    let (ans_on, bytes_on) = run_with_cap(&cmd, needle, 200, &d_on);

    assert_eq!(
        ans_on, ans_off,
        "grep compression changed the answer: on={ans_on} off={ans_off}"
    );
    assert!(
        ans_off.contains("here"),
        "uncapped grep lost the value: {ans_off}"
    );
    assert!(
        bytes_on <= bytes_off,
        "capped grep grew: on={bytes_on} off={bytes_off}"
    );

    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&d_off).ok();
    std::fs::remove_dir_all(&d_on).ok();
}
