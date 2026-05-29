//! `zero-core` — the dependency-free engine behind Zero.
//!
//! Zero is a local-first AI coding harness. The *core* holds everything that is
//! independent of how you talk to it (terminal or app): chat [`message`] types,
//! the model [`backend`] abstraction, an honest [`clock`], std-only [`json`],
//! and append-only [`session`] logging.
//!
//! Design rule, enforced by review: **zero runtime dependencies.** Everything
//! here is `std` only. See the workspace `Cargo.toml`.
//!
//! Frontends (the `zero` binary today, an app later) are thin shells over this
//! core — anything the terminal can do, the app can do, because the capability
//! lives here, not in the UI.

pub mod backend;
pub mod brand;
pub mod clock;
pub mod json;
pub mod message;
pub mod session;

pub use backend::{Backend, BackendError, StopReason, StreamEvent, StubBackend};
pub use clock::{format_duration, Stopwatch};
pub use json::Value;
pub use message::{Conversation, Message, Role};
pub use session::SessionLog;
