# Zero

A local-first AI coding terminal in Rust with **zero runtime dependencies**.
Built to feel like Claude Code's terminal, designed first for local LLMs.

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

### Keys

| Keys | Action |
|------|--------|
| `^A` / `^E`, `Home` / `End` | start / end of line |
| `^B` / `^F` | back / forward one char |
| `⌥←` / `⌥→`, `^←` / `^→` | back / forward one word |
| `^W` | delete word back |
| `^U` / `^K` | kill to start / end of line |
| `^L` | clear screen |
| `^J` | insert newline — multiline input (works in every terminal) |
| `Shift+Enter` / `⌥+Enter` | insert newline (on terminals that send a distinct code) |
| `Enter` | submit |
| `↑` / `↓` | move between input lines, else recall history |
| `^R` | reverse history search (type to match, `^R` for older, `Enter` accept, `Esc` cancel) |
| `Esc Esc` | clear the line |
| `^C` | clear the line; on an empty line, `^C` again to exit |
| `^D` | exit on an empty line |

### Shell mode & the safety guard

Prefix a line with `!` to run it as a shell command inline (`!cargo test`,
`!git status`). Output, exit code, and measured time print in place.

Every command — `!` shell now, agent tool calls later — passes through a
**destructive-command guard** (`zero-core::safety`) first. It's a hard, in-code
classifier, not a soft prompt rule: catastrophic commands (`rm -rf` on a
critical path, `git reset --hard`, `dd of=/dev/…`, `sudo`, fork bombs, …) are
flagged and require an explicit `y/N` confirmation before they run. The lesson
from other harnesses is that prompt-level rules don't stop `rm -rf ~`; a gate at
the execution boundary does.

> **Multiline / Shift+Enter:** terminals only send a distinct Shift+Enter when
> they support the CSI-u/kitty keyboard protocol (kitty, WezTerm, recent iTerm2).
> Everywhere else it reads as plain Enter — so **`^J` is the reliable newline
> key** (Return sends CR = submit, `^J` sends LF = newline). `⌥+Enter` also works
> on terminals that send `ESC CR`. Word-wise moves (`⌥`/`^` + arrows) likewise
> depend on the terminal sending the modifier sequence.

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

## Toolchain note

Pinned to Rust **1.93.1** because the machine's `stable` toolchain has a
corrupted `rust-std` (missing std libs → "can't find crate for std"). Repair:
`rustup toolchain uninstall stable && rustup toolchain install stable`, then
bump the pin.

## License

MIT.
