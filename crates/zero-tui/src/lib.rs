//! `zero-tui` — Zero's terminal frontend, dependency-free.
//!
//! Layered as functional-core / imperative-shell:
//! - pure, fully-tested models — [`key`] (byte→key decoding), [`editor`]
//!   (line editing + history), [`viewport`] (scrollback + wrapping);
//! - a thin `unsafe` shell — [`term`] (raw mode + size via libc symbols we
//!   declare ourselves, no `libc` crate);
//! - the wiring — [`app`], the Claude-Code-style inline REPL.

pub mod app;
pub mod editor;
pub mod key;
pub mod markdown;
pub mod term;
pub mod viewport;

pub use app::App;
pub use editor::LineEditor;
pub use key::{decode_keys, Key};
pub use term::{terminal_size, RawTerminal, Size};
pub use viewport::{wrap_line, Scrollback};
