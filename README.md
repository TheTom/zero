# Zero

A local-first, **zero-dependency** AI coding harness in Rust — built to feel like
Claude Code's terminal, designed first for local LLMs, and architected so the
terminal and a future app share one engine.

> Status: **MVP slice 1 — the terminal feel.** Inline streaming REPL with
> readline-style editing, history, honest elapsed timing, and JSONL session
> logging, running against a stub backend. The OpenAI-compatible HTTP client
> (for the qwen box) is the next slice — see [Roadmap](#roadmap).

## Why "zero"

- **Zero runtime dependencies.** No crates. `std` only — hand-rolled JSON, a
  raw-mode terminal built on the handful of libc `termios`/`ioctl` symbols we
  declare ourselves, our own input decoder and line editor. The runtime
  `Cargo.toml`s have empty `[dependencies]`, and they stay that way.
- **Local-first.** Points at your own model (e.g. qwen on the asus gx10) over an
  OpenAI-compatible endpoint. No cloud required.

## North stars (the design bets)

These are why Zero exists, not just nice-to-haves:

1. **Claude-Code terminal feel.** Output flows into the terminal's *native*
   scrollback; only the current input line is taken over and redrawn in place.
   No rigid full-screen takeover.
2. **Terminal ≡ app.** Every capability lives in `zero-core`, the dependency-free
   engine. Frontends (this terminal, a future app) are thin shells. Anything the
   terminal can do, the app can do, for free.
3. **Honest timestamps.** Zero **measures, never estimates.** `clock.rs` only
   reports real monotonic elapsed time. No "this will take a week" — it took an
   hour, and we'll show the hour.
4. **Rich, replayable logging.** Every turn is appended to a JSONL transcript
   with a real wall-clock timestamp. (Modeled on what Claude logs well.)
5. **Compaction done right.** *Not yet built.* The hard problem we want to spend
   real time on — most agents compact badly and resurface stale context. Slated
   as its own slice once the agentic loop exists.
6. **Instructions that are actually followed.** The `CLAUDE.md`-equivalent must
   be *reliably obeyed*, not ignored after a few turns. Open research problem —
   directions sketched in [docs/north-stars.md](docs/north-stars.md#6-instructions-that-are-actually-followed).

The full reasoning behind each is in **[docs/north-stars.md](docs/north-stars.md)**.

## Architecture

Functional-core / imperative-shell. The pure cores are exhaustively unit-tested;
the only `unsafe` is the thin terminal FFI shell.

```
crates/
  zero-core/   # the engine — std only, no UI, no I/O assumptions
    json.rs       hand-rolled JSON value + parser + serializer
    message.rs    Role / Message / Conversation (OpenAI-compatible shape)
    backend.rs    Backend trait + StreamEvent + StubBackend
    clock.rs      Stopwatch + honest duration formatting
    session.rs    append-only JSONL transcript logging
  zero-tui/    # the terminal frontend — std only
    key.rs        bytes → Key decoder (UTF-8 + ANSI escapes)   [pure]
    editor.rs     readline-style line editor + history          [pure]
    viewport.rs   scrollback + word wrap                        [pure]
    term.rs       raw mode + window size via libc symbols       [unsafe shell]
    app.rs        the inline REPL event loop                     [wiring]
  zero/        # the binary — arg parsing, backend selection, wiring
```

The seam that makes north star #2 work is `zero_core::Backend`: the terminal
talks only to that trait, so swapping `StubBackend` for the real HTTP client
changes nothing in the UI.

## Build & run

```bash
cargo build --release
./target/release/zero            # interactive (needs a real terminal)
./target/release/zero --instant  # no streaming pacing delay
./target/release/zero --no-log   # don't write a session transcript
./target/release/zero --help
```

Sessions are logged to `~/.zero/sessions/zero-<unixtime>.jsonl`.

### Keys

`^A`/`^E` home/end · `^U`/`^K` kill to start/end · `^W` kill word ·
`↑`/`↓` history · `^L` clear screen · `^C` / `/quit` exit.

## Tests & quality gate

```bash
cargo test --workspace      # 73 tests, all pure-core logic covered
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

Every crate sets `clippy::all = "deny"`. The pure cores (JSON, key decode, line
editor, wrapping, clock, session log) carry the coverage; the FFI shell is kept
minimal precisely because it's the hard-to-test part.

> **Coverage tooling** (llvm source-based via `cargo`'s `-C instrument-coverage`)
> is a follow-up — it needs `llvm-tools`, which is a *toolchain* component, not a
> crate dependency, so it doesn't violate the zero-deps rule. Tracked below.

### Toolchain note

Pinned to Rust **1.93.1** via `rust-toolchain.toml`. The machine's `stable`
toolchain currently has a **corrupted `rust-std` component** (the std libs are
missing from `~/.rustup/toolchains/stable-*/lib/rustlib/`), which breaks every
build under `stable`. Repair with a full reinstall when convenient:

```bash
rustup toolchain uninstall stable && rustup toolchain install stable
```

Then bump the pin (or remove `rust-toolchain.toml`).

## Renaming the project

The name "Zero" is **not final** and is cheap to change:

- **Display name / config dir / prompt label** all derive from
  `crates/zero-core/src/brand.rs`. Edit `DEFAULT_NAME` and `DEFAULT_SLUG` (or set
  the `ZERO_NAME` / `ZERO_SLUG` env vars at runtime — no recompile). The session
  dir `~/.zero/` follows the slug automatically.
- **Crate names** (`zero-core`, `zero-tui`, `zero`) are mechanical to rename:
  rename the dirs, update `name =` in each `Cargo.toml`, the workspace `members`,
  and the inter-crate `path` deps. A sed pass over `zero-core`/`zero-tui` plus the
  binary handles it.

Nothing else hardcodes the name.

## Roadmap

- [x] **Slice 1 — terminal feel.** Inline REPL, editing, history, streaming,
      honest timing, JSONL logging. *(this MVP)*
- [ ] **Slice 2 — real brain.** OpenAI-compatible HTTP client (std `TcpStream`,
      SSE streaming, JSON we already have) → point at the qwen box. Backend trait
      already in place.
- [ ] **Slice 3 — agentic loop.** Tool-calling (read/write/edit/bash/ls), the
      tool trait, and the model↔tools turn loop.
- [ ] **Slice 4 — compaction.** The north-star research problem.
- [ ] **Slice 5 — app parity.** A second frontend over the same `zero-core`.
- [ ] Coverage reporting in CI; full-screen mode using `viewport.rs`.

## License

MIT.
