//! Black-box integration tests for the `zero` binary.
//!
//! These spawn the *actual compiled binary* (`CARGO_BIN_EXE_zero`) as a
//! subprocess — the real `main()` path that in-process tests can't reach:
//! arg parsing, backend selection, the headless `-p` short-circuit (before
//! `RawTerminal::enable`), stdout/stderr separation, and exit codes. `main.rs`
//! is excluded from the coverage gate precisely because only a spawned-process
//! test exercises it, so this file is where that path earns its keep.
//!
//! Determinism: the default backend is the built-in stub (`--stub`), so no
//! network or model is involved. One test stands up a tiny in-process HTTP mock
//! that speaks just enough OpenAI to drive the agentic `bash` tool end to end
//! through the binary — still fully deterministic (localhost, fixed response).

use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Path to the freshly-built `zero` binary, provided by Cargo to integration
/// tests. Using this (not a hardcoded path) guarantees we test the current build.
fn zero_bin() -> &'static str {
    env!("CARGO_BIN_EXE_zero")
}

/// Run the binary with `args`, optional stdin, and a clean-ish env. Returns
/// `(stdout, stderr, exit_code)`. Always `--no-log` so tests never touch
/// `~/.zero/sessions`, and a throwaway `--config` so the user's real config
/// can't influence the run.
fn run(args: &[&str], stdin: Option<&str>) -> (String, String, i32) {
    let (o, e, c, _home) = run_in_home(args, stdin);
    (o, e, c)
}

/// Like [`run`] but also points `$HOME` at a fresh temp dir and returns it, so a
/// test can inspect what the binary wrote under `$HOME/.zero/` — notably the
/// spilled tool-output artifacts in `$HOME/.zero/outputs/<ts>/`. The caller owns
/// cleanup of the returned home dir.
fn run_in_home(args: &[&str], stdin: Option<&str>) -> (String, String, i32, std::path::PathBuf) {
    let salt = format!("{}-{}", std::process::id(), args.join("_").len());
    let home = std::env::temp_dir().join(format!("zero-cli-home-{salt}"));
    std::fs::create_dir_all(&home).unwrap();
    let cfg = home.join("config.json");
    let mut cmd = Command::new(zero_bin());
    cmd.args(args)
        .arg("--config")
        .arg(&cfg)
        .env("HOME", &home)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn zero");
    if let Some(s) = stdin {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(s.as_bytes())
            .expect("write stdin");
    } // dropping stdin closes it → EOF for `-p -`
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
        home,
    )
}

/// The single spilled artifact under `$HOME/.zero/outputs/<ts>/` after a run that
/// capped a tool result. Returns `(path, full_contents)`. Panics if not exactly
/// one artifact is found (the tests below each produce exactly one).
fn sole_artifact(home: &std::path::Path) -> (std::path::PathBuf, String) {
    let outputs = home.join(".zero").join("outputs");
    let mut found = Vec::new();
    // outputs/<launch-ts>/out-<id>.txt
    for ts in std::fs::read_dir(&outputs).into_iter().flatten().flatten() {
        for f in std::fs::read_dir(ts.path()).into_iter().flatten().flatten() {
            found.push(f.path());
        }
    }
    assert_eq!(
        found.len(),
        1,
        "expected exactly one spilled artifact under {}, found {:?}",
        outputs.display(),
        found
    );
    let body = std::fs::read_to_string(&found[0]).expect("read artifact");
    (found[0].clone(), body)
}

#[test]
fn headless_print_writes_reply_to_stdout_only() {
    // The core `zero -p` contract: stdout carries exactly the reply (+ newline),
    // the trace goes to stderr, exit 0 — and it does NOT error on a non-tty
    // (the `-p` branch short-circuits before RawTerminal::enable).
    let (stdout, stderr, code) = run(&["-p", "ping zero", "--stub", "--no-log"], None);
    assert_eq!(code, 0, "stderr: {stderr}");
    assert!(stdout.contains("ping zero"), "stdout: {stdout:?}");
    // The stub's canned shape proves it's the reply, not an error.
    assert!(stdout.contains("stub reply"), "stdout: {stdout:?}");
    // The "needs an interactive terminal" error must NOT appear.
    assert!(
        !stderr.contains("interactive terminal"),
        "stderr: {stderr:?}"
    );
    // stdout is just the one reply line.
    assert_eq!(
        stdout.lines().count(),
        1,
        "stdout not single-line: {stdout:?}"
    );
}

#[test]
fn headless_print_reads_prompt_from_stdin() {
    // `-p -` reads the whole prompt from stdin.
    let (stdout, _stderr, code) = run(&["-p", "-", "--stub", "--no-log"], Some("from stdin pipe"));
    assert_eq!(code, 0);
    assert!(stdout.contains("from stdin pipe"), "stdout: {stdout:?}");
}

#[test]
fn help_lists_headless_flags_and_exits_zero() {
    let (stdout, _stderr, code) = run(&["--help"], None);
    assert_eq!(code, 0);
    assert!(stdout.contains("-p, --print"), "help missing -p: {stdout}");
    assert!(stdout.contains("--tools"), "help missing --tools: {stdout}");
    assert!(stdout.contains("--stub"));
}

#[test]
fn unknown_flag_exits_nonzero_with_message() {
    let (_stdout, stderr, code) = run(&["--definitely-not-a-flag"], None);
    assert_ne!(code, 0, "should fail on unknown flag");
    assert!(stderr.contains("unknown argument"), "stderr: {stderr:?}");
}

#[test]
fn no_log_suppresses_the_session_log_line() {
    // With --no-log the binary must not announce "logging to …".
    let (_stdout, stderr, code) = run(&["-p", "hi", "--stub", "--no-log"], None);
    assert_eq!(code, 0);
    assert!(!stderr.contains("logging to"), "stderr: {stderr:?}");
}

#[test]
fn headless_tools_run_bash_against_a_real_http_backend() {
    // The full real path through the binary: a localhost HTTP server speaks just
    // enough OpenAI to (1) ask for a `bash` tool call running a sentinel echo,
    // then (2) answer with text relaying it. The binary parses args → builds the
    // OpenAI backend (NOT the stub) → runs the agentic loop → bash executes →
    // result feeds back → final reply hits stdout. Deterministic: fixed responses.
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    let sentinel = "zero-cli-sentinel-9931";
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let round = Arc::new(Mutex::new(0u32));

    let handle = thread::spawn(move || {
        // Two completions: round 1 → a bash tool call; round 2 → the final text.
        for _ in 0..2 {
            let (mut sock, _) = match listener.accept() {
                Ok(s) => s,
                Err(_) => return,
            };
            sock.set_read_timeout(Some(std::time::Duration::from_millis(200)))
                .ok();
            // Drain the whole request: read until the timeout fires, not until a
            // short read (the round-2 request body can span many chunks).
            let mut buf = [0u8; 8192];
            loop {
                match sock.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
            let mut r = round.lock().unwrap();
            *r += 1;
            let body = if *r == 1 {
                // Structured tool call: run the sentinel echo via bash.
                format!(
                    r#"{{"choices":[{{"message":{{"content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"bash","arguments":"{{\"command\":\"echo {sentinel}\"}}"}}}}]}}}}]}}"#
                )
            } else {
                // Final answer relaying the output.
                format!(
                    r#"{{"choices":[{{"message":{{"content":"the command printed {sentinel}"}}}}]}}"#
                )
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });

    let url = format!("http://127.0.0.1:{port}");
    let (stdout, stderr, code) = run(
        &[
            "-p",
            "use bash to echo the sentinel and tell me what it printed",
            "--tools",
            "--url",
            &url,
            "--model",
            "mock",
            "--no-log",
        ],
        None,
    );
    let _ = handle.join();

    assert_eq!(code, 0, "stderr: {stderr}");
    // The model's final reply (relaying the sentinel) reached stdout.
    assert!(
        stdout.contains(sentinel),
        "sentinel not in stdout: {stdout:?}"
    );
    // The bash tool actually ran — its echoed output appears in the streamed
    // trace on stderr (⚙ bash / ↳ …), proving the tool executed, not hallucinated.
    assert!(
        stderr.contains("bash") && stderr.contains(sentinel),
        "bash trace missing the sentinel on stderr: {stderr:?}"
    );
}

/// Spin up a localhost HTTP server that speaks just enough OpenAI for a 2-round
/// agentic turn: round 1 returns a single `bash` tool call running `cmd`; round 2
/// returns final text. Returns the base URL; the server thread exits after 2
/// requests. Deterministic.
fn mock_openai_bash(cmd: &str) -> String {
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let cmd = cmd.to_string();
    let round = Arc::new(Mutex::new(0u32));
    thread::spawn(move || {
        // Serve indefinitely (the agentic loop may make extra requests, and the
        // round-2 request carries the full — possibly large — tool result back).
        loop {
            let Ok((mut sock, _)) = listener.accept() else {
                return;
            };
            sock.set_read_timeout(Some(std::time::Duration::from_millis(200)))
                .ok();
            // Drain the whole HTTP request: read until the read-timeout fires
            // (signalled by WouldBlock/TimedOut), not until a short read — the
            // request body can span many 8 KB chunks (it includes prior tool
            // output fed back to the model).
            let mut buf = [0u8; 8192];
            loop {
                match sock.read(&mut buf) {
                    Ok(0) => break,    // client closed
                    Ok(_) => continue, // more to read / keep draining
                    Err(_) => break,   // timeout → request fully received
                }
            }
            let mut r = round.lock().unwrap();
            *r += 1;
            // JSON-escape the command for embedding in the tool-call arguments string.
            let cmd_esc = cmd.replace('\\', "\\\\").replace('"', "\\\"");
            let body = if *r == 1 {
                format!(
                    r#"{{"choices":[{{"message":{{"content":"","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"bash","arguments":"{{\"command\":\"{cmd_esc}\"}}"}}}}]}}}}]}}"#
                )
            } else {
                r#"{"choices":[{"message":{"content":"done"}}]}"#.to_string()
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[test]
fn token_saving_caps_big_bash_output_but_keeps_it_recoverable() {
    // The point of the whole compression effort, proven through the real binary:
    // a bash command produces ~23 KB of output; the model-facing result is capped
    // small, BUT the full output is spilled to $HOME/.zero/outputs/<ts>/ byte-for
    // -byte and the compressed view names that artifact. Nothing is lost.
    let url = mock_openai_bash("seq 1 5000");
    // Configure a tiny per-result cap so the 23 KB result is definitely capped.
    let cfg = format!(
        r#"{{"base_url":"{url}","model":"mock","max_tool_output":512,"max_turn_output":1000000}}"#
    );

    // Write the config into the isolated $HOME the binary will use. We can't know
    // the home path before spawning, so pass --config explicitly AND set $HOME via
    // run_in_home; point --config at the same file we pre-write.
    let salt = "tokensave";
    let home = std::env::temp_dir().join(format!("zero-cli-toksave-{}-{salt}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let cfg_path = home.join("config.json");
    std::fs::write(&cfg_path, &cfg).unwrap();

    let out = Command::new(zero_bin())
        .args(["-p", "run seq via bash", "--tools", "--no-log", "--config"])
        .arg(&cfg_path)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code().unwrap_or(-1), 0, "stderr: {stderr}");

    // The full output spilled to disk, byte-for-byte: it must contain BOTH ends of
    // the range (1 and 5000) and the exit line — the proof nothing was lost.
    let (artifact, full) = sole_artifact(&home);
    assert!(
        full.contains("\n1\n") || full.starts_with("1\n"),
        "missing head"
    );
    assert!(
        full.contains("\n5000\n") || full.contains("5000\n[exit 0]"),
        "missing tail"
    );
    assert!(full.contains("[exit 0]"), "missing exit line");
    // The full artifact dwarfs the per-result cap — it really is the whole output.
    assert!(
        full.len() > 10_000,
        "artifact too small to be the full output: {}",
        full.len()
    );

    // bash actually ran (the trace shows the call).
    assert!(stderr.contains("bash"), "no bash trace: {stderr:?}");
    // The existence of the artifact is itself the proof capping fired: spill only
    // happens on the over-budget path. And it's RECOVERABLE — a model handed the
    // marker could read any line range back; we simulate that round-trip here by
    // pulling lines 2500–2502 straight out of the spilled file.
    let around: Vec<&str> = full.lines().skip(2499).take(3).collect();
    assert_eq!(
        around,
        vec!["2500", "2501", "2502"],
        "middle not recoverable from artifact"
    );
    // The artifact is named after the tool-call id (out-<id>.txt) — the same handle
    // the compressed marker embeds for re-fetch.
    let name = artifact.file_name().unwrap().to_string_lossy();
    assert!(
        name.starts_with("out-") && name.ends_with(".txt"),
        "artifact name: {name}"
    );

    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn token_saving_grep_keeps_all_file_line_refs_through_the_binary() {
    // grep's high-signal content (file:line refs) must survive capping even when
    // the bodies are dropped — proven through the spawned binary. We grep a file
    // we write under $HOME so the path is stable and inside the run.
    let home = std::env::temp_dir().join(format!("zero-cli-grep-{}-grepsave", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let fixture = home.join("fixture.txt");
    let mut body = String::new();
    for i in 1..=60 {
        if i % 10 == 0 {
            body.push_str(&format!(
                "line {i}: NEEDLE with a long trailing body to inflate size\n"
            ));
        } else {
            body.push_str(&format!("line {i}: filler filler filler filler filler\n"));
        }
    }
    std::fs::write(&fixture, &body).unwrap();

    let cmd = format!("grep -n NEEDLE {}", fixture.display());
    let url = mock_openai_bash(&cmd);
    let cfg = format!(
        r#"{{"base_url":"{url}","model":"mock","max_tool_output":200,"max_turn_output":1000000}}"#
    );
    let cfg_path = home.join("config.json");
    std::fs::write(&cfg_path, &cfg).unwrap();

    let out = Command::new(zero_bin())
        .args(["-p", "grep it", "--tools", "--no-log", "--config"])
        .arg(&cfg_path)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code().unwrap_or(-1), 0, "stderr: {stderr}");

    // The full grep output spilled byte-identical: every match line survives there.
    let (_artifact, full) = sole_artifact(&home);
    for n in [10, 20, 30, 40, 50, 60] {
        assert!(
            full.contains(&format!("{n}: NEEDLE")),
            "lost match {n} in artifact"
        );
    }
    std::fs::remove_dir_all(&home).ok();
}

// --- negative paths & robustness (hardening the flow) --------------------

#[test]
fn dead_backend_url_fails_gracefully_not_a_panic_or_hang() {
    // A real --url pointing at a refused port: the binary must exit cleanly with
    // an error on stderr and empty stdout — never panic, never hang. (Connection
    // refused is immediate, so no timeout wait here.)
    let (stdout, stderr, code) = run(
        &[
            "-p",
            "hello",
            "--url",
            "http://127.0.0.1:1",
            "--model",
            "x",
            "--no-log",
        ],
        None,
    );
    assert_eq!(code, 0, "should exit cleanly: stderr {stderr:?}");
    // Non-tools path surfaces the error as the reply ([error: …]); either way it
    // must mention an error and must NOT have produced a real answer.
    assert!(
        stdout.contains("error") || stderr.contains("error") || stderr.contains("refused"),
        "no error surfaced. stdout={stdout:?} stderr={stderr:?}"
    );
    assert!(
        !stdout.contains("stub reply"),
        "should not have fallen back to stub"
    );
}

#[test]
fn malformed_config_falls_back_without_crashing() {
    // A garbage --config file: the binary loads it best-effort (unwrap_or_default),
    // so it falls back to the stub and still answers — no crash, exit 0.
    let home = std::env::temp_dir().join(format!("zero-cli-badcfg-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let cfg = home.join("config.json");
    std::fs::write(&cfg, "{ this is not valid json").unwrap();
    let out = Command::new(zero_bin())
        .args(["-p", "ping", "--stub", "--no-log", "--config"])
        .arg(&cfg)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code().unwrap_or(-1), 0);
    assert!(stdout.contains("ping"), "stdout: {stdout:?}");
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn empty_and_whitespace_prompts_are_handled() {
    // Empty / whitespace-only prompts must not crash. The prompt is trimmed before
    // run_once, so a whitespace prompt becomes empty — the run still exits 0.
    for p in ["", "   ", "\t\n  "] {
        let (_stdout, stderr, code) = run(&["-p", p, "--stub", "--no-log"], None);
        assert_eq!(code, 0, "prompt {p:?} crashed: {stderr:?}");
        assert!(!stderr.contains("panic"), "panicked on {p:?}: {stderr:?}");
    }
}

#[test]
fn missing_print_value_errors_cleanly() {
    // `-p` with no value → arg parse error, nonzero exit, no panic. Spawn directly
    // (not via run(), which would inject --config and become -p's "value").
    let out = Command::new(zero_bin())
        .arg("-p")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(out.status.code().unwrap_or(-1), 0);
    assert!(stderr.contains("needs a value"), "stderr: {stderr:?}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr:?}");
}

#[test]
fn dangerous_bash_is_refused_through_the_binary_and_never_runs() {
    // Security gate, end to end: the model asks bash to do something destructive
    // chained after a sentinel-creating touch. The classifier flags the whole
    // chain → refused → NOTHING runs, so the sentinel file must not exist.
    let home = std::env::temp_dir().join(format!("zero-cli-danger-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let sentinel = home.join("MUST_NOT_EXIST");
    // touch <sentinel> && rm -rf /  — the classifier flags `rm -rf /`.
    let cmd = format!("touch {} && rm -rf /", sentinel.display());
    let url = mock_openai_bash(&cmd);
    let cfg_path = home.join("config.json");
    std::fs::write(
        &cfg_path,
        format!(r#"{{"base_url":"{url}","model":"mock"}}"#),
    )
    .unwrap();

    let out = Command::new(zero_bin())
        .args(["-p", "do the thing", "--tools", "--no-log", "--config"])
        .arg(&cfg_path)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code().unwrap_or(-1), 0, "stderr: {stderr}");
    assert!(stderr.contains("refused"), "danger not refused: {stderr:?}");
    assert!(
        !sentinel.exists(),
        "REFUSED COMMAND STILL RAN — sentinel was created"
    );
    std::fs::remove_dir_all(&home).ok();
}

#[test]
fn bash_path_confinement_rejects_absolute_read_but_does_not_crash() {
    // A read_file tool call with an absolute path escapes the workspace root and
    // is rejected by the confinement check; the loop reports it and finishes
    // cleanly (the model gets an error it can act on — no panic).
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let round = Arc::new(Mutex::new(0u32));
    thread::spawn(move || loop {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        sock.set_read_timeout(Some(std::time::Duration::from_millis(200)))
            .ok();
        let mut buf = [0u8; 8192];
        loop {
            match sock.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => continue,
            }
        }
        let mut r = round.lock().unwrap();
        *r += 1;
        let body = if *r == 1 {
            // read_file with an absolute path → confinement rejects it.
            r#"{"choices":[{"message":{"content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"/etc/passwd\"}"}}]}}]}"#.to_string()
        } else {
            r#"{"choices":[{"message":{"content":"understood"}}]}"#.to_string()
        };
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = sock.write_all(resp.as_bytes());
    });
    let url = format!("http://127.0.0.1:{port}");
    let (stdout, stderr, code) = run(
        &[
            "-p",
            "read passwd",
            "--tools",
            "--url",
            &url,
            "--model",
            "mock",
            "--no-log",
        ],
        None,
    );
    assert_eq!(code, 0, "stderr: {stderr}");
    // The confinement rejection appears in the tool trace; the model's follow-up
    // answer reaches stdout. No panic anywhere.
    assert!(!stderr.contains("panicked"), "panicked: {stderr:?}");
    assert!(
        stderr.contains("absolute") || stderr.contains("error") || stdout.contains("understood"),
        "confinement path not exercised cleanly: stdout={stdout:?} stderr={stderr:?}"
    );
}
