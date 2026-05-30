//! Replay harness: run tool outputs shaped like the ones in real session logs
//! through the compressor and assert the before/after contract.
//!
//! Motivation: two analyzed Claude Code sessions showed very different token
//! sinks — one image/re-read heavy, one where **shell output was 88.5% of all
//! tool-result bytes** (broad `grep -rn`, full `gh pr diff` dumps). This test
//! reconstructs those exact shapes (sized to the real byte counts) and proves
//! that compression:
//!   1. reduces tokens by at least a shape-appropriate threshold,
//!   2. preserves the high-signal content (the error line, the file:line ref),
//!   3. is recoverable (the marker names the spill artifact), and
//!   4. is deterministic.
//!
//! Run `cargo test -p zero-core --test replay_compress -- --nocapture` to see the
//! before/after table.

use std::path::Path;
use zero_core::compress::{compress, OutputShape};
use zero_core::context::estimate_tokens;

/// One reconstructed fixture: a command + its (shaped) raw output, plus what the
/// compressed view must still contain to count as non-lossy.
struct Fixture {
    name: &'static str,
    cmd: &'static str,
    raw: String,
    /// Substrings that MUST survive compression (the high-signal content).
    must_keep: Vec<String>,
    /// Minimum % token reduction expected for this shape at the test budget.
    min_reduction_pct: u32,
    expect_shape: OutputShape,
}

/// A broad recursive grep — the real Log-B offender was 11,334 bytes / ~140 hits.
fn grep_dump() -> Fixture {
    let mut raw = String::new();
    // ~140 matches across a handful of files, long-ish bodies (the bulk).
    for i in 0..70 {
        raw.push_str(&format!(
            "turbo/src/quant.rs:{}: let q = orthogonal_project(x, sqrt(d)); // hadamard rotate\n",
            i * 7 + 3
        ));
    }
    for i in 0..50 {
        raw.push_str(&format!(
            "turbo/src/rotate.rs:{}: // QR decomposition feeds the orthogonal basis here\n",
            i * 3 + 1
        ));
    }
    for i in 0..20 {
        raw.push_str(&format!(
            "docs/turbo4.md:{}: the sqrt(d) scaling keeps the projection norm-preserving\n",
            i + 1
        ));
    }
    Fixture {
        name: "grep -rn orthogonal (140 hits, ~11KB)",
        cmd: "grep -rn -i orthogonal turbo",
        raw,
        // Every file must still be findable, and a specific deep ref must survive.
        must_keep: vec![
            "turbo/src/quant.rs".into(),
            "turbo/src/rotate.rs".into(),
            "docs/turbo4.md".into(),
        ],
        min_reduction_pct: 60,
        expect_shape: OutputShape::Grep,
    }
}

/// A failing `cargo test` log: banner, progress spam, an error in the MIDDLE,
/// more spam, then the summary tail. Naive head/tail would drop the error.
fn build_log() -> Fixture {
    let mut raw = String::new();
    raw.push_str("   Compiling turbo v0.4.0\n");
    for i in 0..300 {
        raw.push_str(&format!("   Compiling dep{i} v1.0.0\n"));
    }
    raw.push_str("error[E0308]: mismatched types: expected f32, found f64\n");
    raw.push_str("   --> turbo/src/quant.rs:142:18\n");
    for i in 0..300 {
        raw.push_str(&format!("warning: unused variable progress_{i}\n"));
    }
    raw.push_str("test result: FAILED. 1 failed; 87 passed\n");
    Fixture {
        name: "cargo test (fail, error mid-log)",
        cmd: "cargo test",
        raw,
        must_keep: vec![
            "error[E0308]: mismatched types".into(), // the cause — never drop it
            "turbo/src/quant.rs:142".into(),         // its location
            "test result: FAILED".into(),            // the tail summary
        ],
        min_reduction_pct: 70,
        expect_shape: OutputShape::Log,
    }
}

/// `gh pr diff` — real was ~4,657 bytes. Diff is generic-donut in v1, so the
/// reduction is modest and we assert it's at least detected + recoverable.
fn pr_diff() -> Fixture {
    let mut raw = String::from("diff --git a/turbo/src/quant.rs b/turbo/src/quant.rs\n");
    raw.push_str("@@ -10,7 +10,9 @@ fn quantize(x: &[f32]) -> Vec<u8>\n");
    for i in 0..200 {
        raw.push_str(&format!("+    let scaled_{i} = x[{i}] * sqrt_d;\n"));
    }
    Fixture {
        name: "gh pr diff (~4.6KB)",
        cmd: "gh pr diff 93",
        raw,
        must_keep: vec!["diff --git".into()],
        min_reduction_pct: 10,
        expect_shape: OutputShape::Diff,
    }
}

fn fixtures() -> Vec<Fixture> {
    vec![grep_dump(), build_log(), pr_diff()]
}

#[test]
fn replay_real_log_shapes_through_compression() {
    // Tight budget mimicking a local-model setting where compression actually fires.
    const BUDGET: usize = 2048;
    let artifact = Path::new("/tmp/zero/out-replay.txt"); // pretend-spilled path

    eprintln!(
        "\n{:<40} {:>8} {:>8} {:>7}  {:<8} {:>6} {:>6}",
        "fixture", "raw_tok", "kept_tok", "saved%", "shape", "signal", "detrm"
    );
    eprintln!("{}", "-".repeat(92));

    for f in fixtures() {
        let c = compress(f.cmd, &f.raw, BUDGET, Some(artifact));

        let raw_tok = estimate_tokens(&f.raw);
        let kept_tok = estimate_tokens(&c.text);
        let reduction = if raw_tok == 0 {
            0
        } else {
            ((raw_tok - kept_tok.min(raw_tok)) * 100 / raw_tok) as u32
        };

        // (2) high-signal content preserved.
        let signal_ok = f.must_keep.iter().all(|k| c.text.contains(k.as_str()));
        // (4) determinism: same input → same output.
        let c2 = compress(f.cmd, &f.raw, BUDGET, Some(artifact));
        let deterministic = c.text == c2.text;

        eprintln!(
            "{:<40} {:>8} {:>8} {:>6}%  {:<8} {:>6} {:>6}",
            f.name,
            raw_tok,
            kept_tok,
            reduction,
            c.shape.label(),
            if signal_ok { "ok" } else { "LOST" },
            if deterministic { "ok" } else { "VARY" },
        );

        // Assertions (the gate).
        assert_eq!(c.shape, f.expect_shape, "{}: wrong shape", f.name);
        assert!(
            reduction >= f.min_reduction_pct,
            "{}: reduction {reduction}% < expected {}%",
            f.name,
            f.min_reduction_pct
        );
        assert!(signal_ok, "{}: high-signal content was lost", f.name);
        assert!(deterministic, "{}: compression not deterministic", f.name);
        // (3) recoverable: the marker points back at the full output.
        assert!(
            c.text.contains("out-replay.txt"),
            "{}: no re-fetch path in marker",
            f.name
        );
        // Sanity: the view is genuinely smaller than the raw.
        assert!(
            c.kept_bytes < c.raw_bytes,
            "{}: not actually smaller",
            f.name
        );
    }
    eprintln!();
}
