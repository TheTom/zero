//! `zero` — entry point for the local-first AI terminal.
//!
//! MVP slice: the TUI-first vertical. It runs the Claude-Code-style inline REPL
//! against a [`StubBackend`] so the *feel* is real (streaming, in-place editing,
//! history, honest elapsed times) while the OpenAI-compatible HTTP backend lands
//! next. Swap the backend below and nothing in the UI changes — that is the
//! whole point of the `zero-core::Backend` seam.

use std::process::ExitCode;
use std::time::Duration;
use zero_core::backend::{Backend, StubBackend};
use zero_core::session::SessionLog;
use zero_tui::{App, RawTerminal};

fn main() -> ExitCode {
    let args = Args::parse(std::env::args().skip(1));
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
    // Backend selection. Today: a stub. Tomorrow: OpenAI-compatible HTTP to the
    // local qwen box. The App is identical either way.
    let backend: Box<dyn Backend> = if args.instant {
        Box::new(StubBackend::instant())
    } else {
        Box::new(StubBackend::paced(Duration::from_millis(18)))
    };

    // Open a session transcript unless disabled.
    let log = if args.no_log {
        None
    } else {
        match session_dir() {
            Some(dir) => match SessionLog::create_in(&dir) {
                Ok((mut log, path)) => {
                    let _ = log.record_meta("backend", backend.name());
                    eprintln!("zero: logging to {}", path.display());
                    Some(log)
                }
                Err(e) => {
                    eprintln!("zero: could not open session log ({e}); continuing without it");
                    None
                }
            },
            None => None,
        }
    };

    // Enter raw mode (RAII-restored on drop) and run.
    let term = RawTerminal::enable().map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!(
                "{e} — zero needs an interactive terminal (try running it directly in a shell)"
            ),
        )
    })?;
    let mut app = App::new(term, std::io::stdout(), backend, log);
    app.run()
}

/// `$HOME/<dot-dir>/sessions`, or `None` if `$HOME` is unset. The dot-dir is
/// derived from the product slug (see `zero_core::brand`) so a rename moves it.
fn session_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|home| {
        std::path::Path::new(&home)
            .join(zero_core::brand::dot_dir())
            .join("sessions")
    })
}

/// Minimal, dependency-free argument parsing.
#[derive(Debug, Default, PartialEq, Eq)]
struct Args {
    help: bool,
    instant: bool,
    no_log: bool,
}

impl Args {
    fn parse<I: IntoIterator<Item = String>>(args: I) -> Self {
        let mut out = Args::default();
        for a in args {
            match a.as_str() {
                "-h" | "--help" => out.help = true,
                "--instant" => out.instant = true,
                "--no-log" => out.no_log = true,
                _ => {} // ignore unknown flags for now
            }
        }
        out
    }
}

fn print_usage() {
    println!(
        "zero — local-first AI terminal\n\n\
         usage: zero [options]\n\n\
         options:\n\
         \x20 --instant   stream stub replies with no pacing delay\n\
         \x20 --no-log    do not write a session transcript\n\
         \x20 -h, --help  show this help\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flags() {
        let a = Args::parse(["--instant", "--no-log"].map(String::from));
        assert!(a.instant);
        assert!(a.no_log);
        assert!(!a.help);
    }

    #[test]
    fn help_flag_recognized() {
        assert!(Args::parse(["-h".to_string()]).help);
        assert!(Args::parse(["--help".to_string()]).help);
    }

    #[test]
    fn unknown_flags_ignored() {
        assert_eq!(Args::parse(["--wat".to_string()]), Args::default());
    }

    #[test]
    fn session_dir_under_home() {
        if let Some(dir) = session_dir() {
            assert!(dir.ends_with("sessions"));
        }
    }
}
