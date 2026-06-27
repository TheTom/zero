// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

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

pub mod agent;
pub mod backend;
pub mod brand;
pub mod builtins;
pub mod clock;
pub mod compress;
pub mod config;
pub mod context;
pub mod discovery;
pub mod fleet_config;
pub mod gate;
pub mod history;
pub mod http;
pub mod json;
pub mod loop_config;
pub mod loop_ledger;
pub mod loop_run;
pub mod loop_runner;
pub mod loop_store;
pub mod mcp;
pub mod message;
pub mod openai;
pub mod orchestrator;
pub mod recall;
pub mod router;
pub mod rules;
pub mod safety;
pub mod sched;
pub mod servers;
pub mod session;
pub mod tools;
pub mod wins;

pub use backend::{Backend, BackendError, StopReason, StreamEvent, StubBackend};
pub use clock::{format_duration, Stopwatch};
pub use config::Config;
pub use fleet_config::{FleetConfig, RoutingMode, TierConfig};
pub use json::Value;
pub use message::{Conversation, Message, Role};
pub use openai::OpenAiBackend;
pub use orchestrator::{FleetSignal, OrchestratorBackend};
pub use router::{Classification, Confidence, Intent, ModeHint, RouteDecision, RouteInput};
pub use session::SessionLog;
