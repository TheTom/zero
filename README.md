# Zero

A local-first AI coding terminal in Rust with **zero runtime dependencies**.
Built to feel like Claude Code's terminal, designed first for local LLMs.

> The name "Zero" isn't final — it's a one-file swap. See [Renaming](#renaming).

## Repo map

Flat by design — three crates, one file per concern, no deep nesting.

```
zero/
├── Cargo.toml              workspace + release profile
├── rust-toolchain.toml     pinned to 1.93.1
├── README.md
├── scripts/coverage.sh     coverage gate (>=98%, enforced)
└── crates/
    ├── zero-core/          the engine — std only, no UI, no I/O assumptions
    │   ├── brand.rs          product name (one-file rename)
    │   ├── json.rs           hand-rolled JSON: parse + serialize
    │   ├── message.rs        Role / Message / Conversation
    │   ├── backend.rs        Backend trait + StubBackend
    │   ├── clock.rs          honest elapsed timing (measure, never estimate)
    │   └── session.rs        append-only JSONL transcript log
    ├── zero-tui/           the terminal frontend — std only
    │   ├── key.rs            bytes → keys (UTF-8 + ANSI)      [pure]
    │   ├── editor.rs         line editor + history           [pure]
    │   ├── viewport.rs       scrollback + word wrap           [pure]
    │   ├── term.rs           raw mode via libc FFI            [unsafe shell]
    │   └── app.rs            the inline REPL loop
    └── zero/               the binary — args, wiring
        └── main.rs
```

**Pure cores carry the logic and the tests; the only `unsafe` is `term.rs`.**
The seam that keeps the UI swappable is `zero_core::Backend` — the terminal
talks only to that trait, so the real model drops in without UI changes.

## Run

```bash
cargo build --release
./target/release/zero            # interactive (needs a real terminal)
./target/release/zero --instant  # no streaming delay
./target/release/zero --no-log   # no session transcript
```

Sessions log to `~/.zero/sessions/zero-<unixtime>.jsonl`.
Keys: `^A`/`^E` home/end · `^U`/`^K` kill · `^W` kill word · `↑`/`↓` history ·
`^L` clear · `^C` / `/quit` exit.

## Test & coverage

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
./scripts/coverage.sh                  # enforces >=98% (fails the build below)
```

**Coverage is held at ≥98%** (lines, functions, regions) and enforced by
`scripts/coverage.sh`. The only excluded files are `term.rs` (libc FFI — can't
run without a real tty) and `main.rs` (process bootstrap); all engine and TUI
logic is covered. `cargo-llvm-cov` is a dev tool, not a crate dependency.

### Pre-push gate

`scripts/hooks/pre-push` blocks a push unless: (1) no AI / co-author attribution
in any pushed commit, (2) tests pass, (3) coverage stays ≥98%. Enable it once
per clone:

```bash
git config core.hooksPath scripts/hooks
```

## Renaming

- **Display name / config dir / prompt label** come from
  `crates/zero-core/src/brand.rs` — edit `DEFAULT_NAME` / `DEFAULT_SLUG`, or set
  `ZERO_NAME` / `ZERO_SLUG` env vars (no recompile).
- **Crate names** rename mechanically: dirs + `name =` in each `Cargo.toml` +
  workspace `members` + `path` deps.

## Toolchain note

Pinned to Rust **1.93.1** because the machine's `stable` toolchain has a
corrupted `rust-std` (missing std libs → "can't find crate for std"). Repair:
`rustup toolchain uninstall stable && rustup toolchain install stable`, then
bump the pin.

## License

MIT.
