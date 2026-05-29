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
use zero_tui::{App, RawTerminal};

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
    let backend: Box<dyn Backend> = if args.stub || !cfg.has_backend() {
        if args.instant {
            Box::new(StubBackend::instant())
        } else {
            Box::new(StubBackend::paced(Duration::from_millis(18)))
        }
    } else {
        match OpenAiBackend::from_config(&cfg) {
            Some(b) => Box::new(b),
            None => Box::new(StubBackend::paced(Duration::from_millis(18))),
        }
    };

    let log = if args.no_log {
        None
    } else {
        open_log(backend.name())
    };

    let term = RawTerminal::enable().map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("{e} — zero needs an interactive terminal (run it directly in a shell)"),
        )
    })?;
    let mut app = App::new(term, std::io::stdout(), backend, log);
    app.set_info(format!("{}\nconfig: {}", cfg.summary(), cfg_path.display()));
    app.run()
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
