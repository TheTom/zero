// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

//! `zero` — entry point for the local-first AI terminal.
//!
//! Loads `~/.zero/config.json` (creating an example on first run), lets CLI
//! flags override it, and builds an OpenAI-compatible backend when a `base_url`
//! is configured — otherwise the stub. The App is identical either way.

use std::process::ExitCode;
use std::time::Duration;
use zero_core::backend::{Backend, StubBackend};
use zero_core::config::Config;
use zero_core::openai::OpenAiBackend;
use zero_core::session::SessionLog;
use zero_tui::{App, Input, RawTerminal};

/// A no-op input source for headless runs: always at EOF (never reads keys).
struct EofInput;
impl Input for EofInput {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}

/// Read all of stdin as the prompt (for `zero -p -`).
fn read_stdin() -> std::io::Result<String> {
    use std::io::Read;
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s)
}

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("zero: {e}");
            return ExitCode::FAILURE;
        }
    };
    if args.help {
        print_usage();
        return ExitCode::SUCCESS;
    }

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zero: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> std::io::Result<()> {
    // `zero rules <init|add|status>` — headless project-rules authoring/inspection,
    // operating on the current directory. No backend/config needed.
    if let Some((sub, text)) = &args.rules {
        return do_rules(sub, text.as_deref(), args.rules_global);
    }
    // `zero logs` — print where this project's logs + spilled artifacts live, so
    // they're never hidden. No backend needed.
    if args.logs {
        return do_logs();
    }

    let cfg_path = args
        .config_path
        .clone()
        .or_else(config_path)
        .ok_or_else(|| std::io::Error::other("could not determine config path ($HOME unset)"))?;

    // First run: drop an example config so there's something to edit.
    if !cfg_path.exists() {
        let _ = Config::default().save(&cfg_path);
        eprintln!("zero: wrote example config to {}", cfg_path.display());
    }

    let mut cfg = Config::load(&cfg_path).unwrap_or_default();
    args.apply_to(&mut cfg);

    // Pick the backend: stub if forced or no URL, otherwise OpenAI-compatible.
    let backend: std::sync::Arc<dyn Backend> = if args.stub || !cfg.has_backend() {
        if args.instant {
            std::sync::Arc::new(StubBackend::instant())
        } else {
            std::sync::Arc::new(StubBackend::paced(Duration::from_millis(18)))
        }
    } else {
        match OpenAiBackend::from_config(&cfg) {
            Some(b) => std::sync::Arc::new(b),
            None => std::sync::Arc::new(StubBackend::paced(Duration::from_millis(18))),
        }
    };

    let (log, log_path) = if args.no_log {
        (None, None)
    } else {
        open_log(backend.name())
    };

    // Headless one-shot (`-p`/`--print`): no raw terminal. The turn's trace goes
    // to stderr so stdout carries only the final reply (`zero -p "…" > out.txt`).
    // `-p -` reads the prompt from stdin.
    if let Some(p) = &args.print {
        let prompt = if p == "-" { read_stdin()? } else { p.clone() };
        let mut app = App::new(EofInput, std::io::stderr(), backend, log);
        app.set_config(cfg.clone(), Some(cfg_path.clone()), servers_path());
        app.set_log_path(log_path);
        app.set_artifact_dir(outputs_dir());
        app.set_tools_enabled(args.tools);
        app.set_auto_accept(args.accept_edits);
        let reply = app.run_once(prompt.trim())?;
        // Real, server-reported token usage for the turn (summed across agentic
        // rounds) — to stderr so stdout stays just the reply. Machine-greppable
        // for the Zero-vs-Hermes benchmark; absent for the stub backend.
        if let Some(u) = app.last_usage() {
            eprintln!(
                "[usage: prompt={} completion={} total={}]",
                u.prompt_tokens,
                u.completion_tokens,
                u.total()
            );
        }
        println!("{reply}");
        return Ok(());
    }

    let term = RawTerminal::enable().map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("{e} — zero needs an interactive terminal (run it directly in a shell)"),
        )
    })?;
    let mut app = App::new(term, std::io::stdout(), backend, log);
    app.set_config(cfg.clone(), Some(cfg_path.clone()), servers_path());
    app.set_log_path(log_path);
    app.set_info(format!("{}\nconfig: {}", cfg.summary(), cfg_path.display()));
    // MCP server definitions live next to the config (~/.zero/mcp.json), and are
    // also imported from Claude Desktop / Claude Code + the project's .mcp.json.
    app.set_mcp_path(cfg_path.parent().map(|d| d.join("mcp.json")));
    app.set_mcp_discovery(home(), std::env::current_dir().ok());
    // Where full tool outputs spill so compressed results stay re-fetchable.
    app.set_artifact_dir(outputs_dir());
    app.run()
}

/// `zero rules <init|add|status>` — project-rules authoring on the current dir.
fn do_rules(sub: &str, text: Option<&str>, global: bool) -> std::io::Result<()> {
    use zero_core::rules;
    let cwd = std::env::current_dir()?;
    let home = home();

    match sub {
        "init" => print!("{}", rules::apply_init(&cwd, home.as_deref(), global)?),
        "add" => {
            let text = text.ok_or_else(|| std::io::Error::other("rules add needs text"))?;
            println!("{}", rules::apply_add(&cwd, home.as_deref(), global, text)?);
        }
        "status" => {
            let reg = rules::load(&cwd, home.as_deref());
            println!(
                "rules: {} enforced · {} soft source(s) · {} warning(s)",
                reg.rules.len(),
                reg.soft.len(),
                reg.warnings.len()
            );
            for r in &reg.rules {
                println!(
                    "  · {} [{}] [{}/{}] {}",
                    r.id,
                    reg.source_of(&r.id).label(),
                    r.on.label(),
                    r.action.label(),
                    r.mat
                );
            }
        }
        "doctor" => {
            let reg = rules::load(&cwd, home.as_deref());
            let issues = rules::doctor(&reg);
            if issues.is_empty() {
                println!("rules doctor: no issues");
            } else {
                println!("rules doctor: {} issue(s)", issues.len());
                for i in issues {
                    println!("  ! {i}");
                }
            }
        }
        other => {
            return Err(std::io::Error::other(format!(
                "unknown rules subcommand '{other}' (init|add|status|doctor)"
            )));
        }
    }
    Ok(())
}

/// `zero logs` — tell the user exactly where their logs and spilled tool-output
/// artifacts live (and the most recent transcript), so nothing is hidden.
fn do_logs() -> std::io::Result<()> {
    match session_dir() {
        Some(dir) => {
            println!("session logs: {}", dir.display());
            // The most recent transcript in this project's log dir, if any.
            if let Some(latest) = latest_transcript(&dir) {
                println!("  latest:    {}", latest.display());
            }
            if let Some(d) = std::env::var_os("ZERO_SESSION_DIR") {
                println!(
                    "  (location set by ZERO_SESSION_DIR={})",
                    d.to_string_lossy()
                );
            }
        }
        None => println!("session logs: unavailable (no HOME)"),
    }
    if let Some(out) = home().map(|h| h.join(zero_core::brand::dot_dir()).join("outputs")) {
        println!(
            "tool artifacts: {} (spilled full tool outputs, per launch)",
            out.display()
        );
    }
    Ok(())
}

/// Newest `*.jsonl` transcript under `dir`, by filename (names embed a unix time).
fn latest_transcript(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut newest: Option<std::path::PathBuf> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "jsonl")
            && newest
                .as_ref()
                .is_none_or(|n| p.file_name() > n.file_name())
        {
            newest = Some(p);
        }
    }
    newest
}

/// Open a session transcript, logging a hint to stderr. Never fatal. Returns the
/// log and the path it opened so the frontend can show it on `/logs`.
fn open_log(
    backend_name: &str,
) -> (
    Option<SessionLog<std::fs::File>>,
    Option<std::path::PathBuf>,
) {
    let Some(dir) = session_dir() else {
        return (None, None);
    };
    match SessionLog::create_in(&dir) {
        Ok((mut log, path)) => {
            // Self-describing transcript: what model and which working directory.
            let _ = log.record_meta("backend", backend_name);
            if let Ok(cwd) = std::env::current_dir() {
                let _ = log.record_meta("cwd", &cwd.display().to_string());
            }
            eprintln!("zero: logging to {}", path.display());
            (Some(log), Some(path))
        }
        Err(e) => {
            eprintln!("zero: session log disabled ({e})");
            (None, None)
        }
    }
}

fn home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

fn config_path() -> Option<std::path::PathBuf> {
    home().map(|h| h.join(zero_core::brand::dot_dir()).join("config.json"))
}

/// Where session transcripts live. Honors `ZERO_SESSION_DIR` (let the user put
/// logs wherever they want — no hidden location), else `~/.{slug}/sessions`, and
/// nests a **per-project** subdirectory (sanitized cwd) so "the logs for *this*
/// repo" are findable instead of one flat pile across every project.
fn session_dir() -> Option<std::path::PathBuf> {
    let base = match std::env::var_os("ZERO_SESSION_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => home()?.join(zero_core::brand::dot_dir()).join("sessions"),
    };
    Some(match std::env::current_dir().ok() {
        Some(cwd) => base.join(project_slug(&cwd)),
        None => base,
    })
}

/// Turn a working-directory path into a single safe directory name, e.g.
/// `/Users/tom/dev/zero` → `Users-tom-dev-zero` (mirrors Claude Code's scheme).
fn project_slug(cwd: &std::path::Path) -> String {
    let s: String = cwd
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let trimmed = s.trim_matches('-');
    if trimmed.is_empty() {
        "root".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Per-launch directory for spilled tool-output artifacts (the re-fetch path for
/// compressed results). Timestamped so launches don't collide.
fn outputs_dir() -> Option<std::path::PathBuf> {
    home().map(|h| {
        h.join(zero_core::brand::dot_dir())
            .join("outputs")
            .join(zero_core::clock::unix_millis().to_string())
    })
}

fn servers_path() -> Option<std::path::PathBuf> {
    home().map(|h| h.join(zero_core::brand::dot_dir()).join("servers.json"))
}

/// Dependency-free argument parsing with valued flags.
#[derive(Debug, Default, PartialEq, Eq)]
struct Args {
    help: bool,
    instant: bool,
    no_log: bool,
    stub: bool,
    url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    config_path: Option<std::path::PathBuf>,
    /// Headless one-shot prompt (`-p`/`--print`). `-` reads the prompt from stdin.
    print: Option<String>,
    /// Enable the agentic tool loop in a headless run (`--tools`).
    tools: bool,
    /// Auto-accept write/edit in a headless run (`--accept-edits`). Without it a
    /// headless `--tools` run is stuck in Normal mode, which refuses every write.
    accept_edits: bool,
    /// `rules <sub> [text]` headless subcommand: `(sub, text)` — init|add|status|doctor.
    rules: Option<(String, Option<String>)>,
    /// `--global`: target `~/.{slug}/` for `rules init|add` instead of the cwd.
    rules_global: bool,
    /// `logs` headless subcommand: print where this project's logs + artifacts live.
    logs: bool,
}

impl Args {
    fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Self, String> {
        let mut out = Args::default();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            let mut take = |flag: &str| -> Result<String, String> {
                it.next().ok_or_else(|| format!("{flag} needs a value"))
            };
            match a.as_str() {
                "-h" | "--help" => out.help = true,
                "--instant" => out.instant = true,
                "--no-log" => out.no_log = true,
                "--stub" => out.stub = true,
                "--tools" => out.tools = true,
                "--accept-edits" => out.accept_edits = true,
                "--global" => out.rules_global = true,
                "logs" => out.logs = true,
                "rules" => {
                    let sub = it
                        .next()
                        .ok_or("rules needs a subcommand: init|add|status")?;
                    let text = if sub == "add" {
                        // `--global` may appear before the text (matches the TUI's
                        // `/rules add --global <text>`); consume it as the flag, not
                        // as the rule body, so the text never becomes "--global".
                        let mut next = it.next().ok_or("rules add needs text (quote it)")?;
                        if next == "--global" {
                            out.rules_global = true;
                            next = it.next().ok_or("rules add needs text (quote it)")?;
                        }
                        Some(next)
                    } else {
                        None
                    };
                    out.rules = Some((sub, text));
                }
                "-p" | "--print" => out.print = Some(take("-p")?),
                "--url" => out.url = Some(take("--url")?),
                "--model" => out.model = Some(take("--model")?),
                "--api-key" => out.api_key = Some(take("--api-key")?),
                "--config" => out.config_path = Some(take("--config")?.into()),
                other => return Err(format!("unknown argument: {other}")),
            }
        }
        Ok(out)
    }

    /// Apply CLI overrides onto a loaded config.
    fn apply_to(&self, cfg: &mut Config) {
        if let Some(u) = &self.url {
            cfg.base_url = Some(u.clone());
        }
        if let Some(m) = &self.model {
            cfg.model = m.clone();
        }
        if let Some(k) = &self.api_key {
            cfg.api_key = Some(k.clone());
        }
    }
}

fn print_usage() {
    println!(
        "zero — local-first AI terminal\n\n\
         usage: zero [options]\n\n\
         options:\n\
         \x20 --url <url>      OpenAI-compatible base URL (e.g. http://host:8000)\n\
         \x20 --model <name>   model name to request\n\
         \x20 --api-key <key>  bearer token (local servers usually need none)\n\
         \x20 --config <path>  use a specific config file\n\
         \x20 -p, --print <s>  headless: run one prompt, print the reply, exit ('-' = stdin)\n\
         \x20 --tools          enable the agentic tool loop in a headless run\n\
         \x20 --accept-edits   auto-accept write/edit in a headless --tools run\n\
         \x20 --stub           force the built-in stub backend\n\
         \x20 --instant        stub streams with no pacing delay\n\
         \x20 --no-log         do not write a session transcript\n\
         \x20 -h, --help       show this help\n\n\
         config: ~/.zero/config.json (created on first run)\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valued_flags() {
        let a =
            Args::parse(["--url", "http://h:1", "--model", "qwen", "--no-log"].map(String::from))
                .unwrap();
        assert_eq!(a.url.as_deref(), Some("http://h:1"));
        assert_eq!(a.model.as_deref(), Some("qwen"));
        assert!(a.no_log);
    }

    #[test]
    fn parses_headless_print_and_tools() {
        let a = Args::parse(["-p", "do the thing", "--tools"].map(String::from)).unwrap();
        assert_eq!(a.print.as_deref(), Some("do the thing"));
        assert!(a.tools);
        // long form + stdin sentinel
        let b = Args::parse(["--print", "-"].map(String::from)).unwrap();
        assert_eq!(b.print.as_deref(), Some("-"));
        assert!(!b.tools);
        assert!(!b.accept_edits);
    }

    #[test]
    fn parses_accept_edits() {
        let a = Args::parse(["-p", "go", "--tools", "--accept-edits"].map(String::from)).unwrap();
        assert!(a.tools && a.accept_edits);
    }

    #[test]
    fn print_without_value_errors() {
        assert!(Args::parse(["-p".to_string()]).is_err());
    }

    #[test]
    fn missing_flag_value_errors() {
        assert!(Args::parse(["--url".to_string()]).is_err());
    }

    #[test]
    fn unknown_flag_errors() {
        assert!(Args::parse(["--wat".to_string()]).is_err());
    }

    #[test]
    fn help_flag_recognized() {
        assert!(Args::parse(["-h".to_string()]).unwrap().help);
        assert!(Args::parse(["--help".to_string()]).unwrap().help);
    }

    #[test]
    fn overrides_apply_onto_config() {
        let a = Args::parse(["--url", "http://x:2", "--model", "m"].map(String::from)).unwrap();
        let mut cfg = Config::default();
        a.apply_to(&mut cfg);
        assert_eq!(cfg.base_url.as_deref(), Some("http://x:2"));
        assert_eq!(cfg.model, "m");
    }

    #[test]
    fn config_and_session_paths_under_dotdir() {
        if let Some(p) = config_path() {
            assert!(p.ends_with("config.json"));
        }
        if let Some(d) = session_dir() {
            // Now nests a per-project subdir under `sessions/`.
            assert!(d.to_string_lossy().contains("sessions"));
            assert!(d.parent().is_some_and(|p| p.ends_with("sessions")));
        }
    }

    #[test]
    fn project_slug_sanitizes_paths() {
        assert_eq!(
            project_slug(std::path::Path::new("/Users/tom/dev/zero")),
            "Users-tom-dev-zero"
        );
        assert_eq!(project_slug(std::path::Path::new("/")), "root");
        assert_eq!(project_slug(std::path::Path::new("/a/b_c.d")), "a-b-c-d");
    }

    #[test]
    fn session_dir_honors_env_override() {
        // Saving + restoring the env var keeps this test isolated.
        let prev = std::env::var_os("ZERO_SESSION_DIR");
        std::env::set_var("ZERO_SESSION_DIR", "/tmp/zero-logs-test");
        let d = session_dir().unwrap();
        assert!(d.to_string_lossy().starts_with("/tmp/zero-logs-test"));
        match prev {
            Some(v) => std::env::set_var("ZERO_SESSION_DIR", v),
            None => std::env::remove_var("ZERO_SESSION_DIR"),
        }
    }
}
