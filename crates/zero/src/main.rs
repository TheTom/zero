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

    let log = if args.no_log {
        None
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

/// Open a session transcript, logging a hint to stderr. Never fatal.
fn open_log(backend_name: &str) -> Option<SessionLog<std::fs::File>> {
    let dir = session_dir()?;
    match SessionLog::create_in(&dir) {
        Ok((mut log, path)) => {
            let _ = log.record_meta("backend", backend_name);
            eprintln!("zero: logging to {}", path.display());
            Some(log)
        }
        Err(e) => {
            eprintln!("zero: session log disabled ({e})");
            None
        }
    }
}

fn home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

fn config_path() -> Option<std::path::PathBuf> {
    home().map(|h| h.join(zero_core::brand::dot_dir()).join("config.json"))
}

fn session_dir() -> Option<std::path::PathBuf> {
    home().map(|h| h.join(zero_core::brand::dot_dir()).join("sessions"))
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
                "rules" => {
                    let sub = it
                        .next()
                        .ok_or("rules needs a subcommand: init|add|status")?;
                    let text = if sub == "add" {
                        Some(it.next().ok_or("rules add needs text (quote it)")?)
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
            assert!(d.ends_with("sessions"));
        }
    }
}
