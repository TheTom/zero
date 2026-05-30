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
    │   ├── ansi.rs           display-width-aware wrapping      [pure]
    │   ├── term.rs           raw mode via libc FFI            [unsafe shell]
    │   └── app.rs            the REPL loop + bottom-pinned box
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
./target/release/zero --stub     # force the built-in echo backend
./target/release/zero --no-log   # no session transcript
```

### Rendering

Inline, with a **bottom-pinned input box** (Claude-Code-style). Output prints in
normal flow so your terminal's own scrollback keeps working — but the input box
+ status footer stay parked at the bottom *the whole time*, including while a
reply streams. The trick is a small live region: completed reply lines are
committed to scrollback as they finish, and only the unfinished tail + the box
are repainted in place each frame (no alt-screen, no lost scrollback).

## Connecting a model

Zero talks to any OpenAI-compatible server (llama.cpp, vLLM, Ollama's shim, …).
With no backend configured it runs a stub that echoes your input.

Config lives at `~/.zero/config.json` (created on first run):

```json
{
  "base_url": "http://gx10-a309.local:8000",
  "model": "Qwen3.6-35B-A3B-...-Q5_K_M.gguf",
  "api_key": "",
  "temperature": null,
  "system_prompt": "",
  "max_tool_output": 4096,
  "max_turn_output": 24000
}
```

`max_tool_output` / `max_turn_output` tune the [context caps](#context-efficiency)
(bytes) — raise them for a big-window model, lower them for an 8K one. Omit either
to use the default shown.

CLI flags override the file:

```bash
zero --url http://gx10-a309.local:8000 --model qwen-heretic
zero --api-key sk-...        # bearer token, if your server needs one
zero --config ./other.json   # use a specific config file
```

`/config` inside the app shows the active backend and model. Plain `http://`
only (Zero is local-first; no TLS).

### Auto-discovery

Don't know the URL? Let Zero find it:

```
/scan            scan this device + the local network for model servers
/connect <n>     attach to a discovered model (swaps the live backend)
/model <name>    switch model on the current endpoint
/servers         list servers found before
```

`/scan` probes **loopback** (servers running on this device, like LM Studio or
Ollama) *and* the local `/24` on common LLM ports (8000, 8080, 11434, 1234, …),
reads each server's `/v1/models`, and lists every model it found — a host serving
several models shows up as several pick choices:

```
discovered models
  1) qwen-heretic     http://192.168.50.125:8000
  2) llama-3.1-8b     http://127.0.0.1:1234
  3) mistral          http://127.0.0.1:1234
use /connect <n> to attach
```

`/connect <n>` attaches immediately and saves the choice to `config.json`
(next launch auto-connects). `/model <name>` switches the model on the current
endpoint. Discovered servers are remembered in `~/.zero/servers.json`;
re-scanning refreshes their model lists and drops a saved model if the box now
serves a different one.

> Cloud endpoints (`https://`) aren't supported yet — Zero is local-first and
> currently speaks plain `http` only. TLS is a later addition.

Sessions log to `~/.zero/sessions/zero-<unixtime>.jsonl`.

### Status line

A dim footer under the input box always shows what you're talking to and how
full the context is:

```
qwen-heretic  ·  192.168.50.125:8000  ·  1.2k/33k ctx (4%)
```

The context window (`n_ctx`) is read from the server's `/props` endpoint on
connect; per-turn token usage comes from the server's own usage report (via
`stream_options.include_usage`) — never an estimate. Until the server reports
numbers, the segment shows what's known (just the window, or nothing for the
stub).

### While a reply is generating

The model streams on a background thread and the input box stays pinned at the
bottom, so the prompt is live the whole time:

- **Type ahead / queue** — keep typing; the pinned box previews the line. Each
  `Enter` **queues** it — queued messages are listed just above the box
  (`⏎ queued: …`) and run in order once the current reply finishes. Doesn't
  interrupt.
- **`^Q` — edit the queue** — jump up into the queued messages and edit them in
  place before they're sent. `↑`/`↓` (or repeated `^Q`) move between items, edit
  the selected one inline, `Enter`/`Esc` to finish. **Sending is paused** while
  you edit (the current reply keeps streaming); empty an item to drop it.
- **`^C` or `Esc Esc`** — interrupt the in-flight reply (keeps the partial text
  in context, clears the queue), e.g. to redirect it.

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
| `Tab` | complete the slash command you're typing |
| `Enter` | submit — or complete an in-progress slash command (`/he`→`/help`) |
| `↑` / `↓` | move between input lines, else recall history |
| `^R` | reverse history search (type to match, `^R` for older, `Enter` accept, `Esc` cancel) |
| `^Q` | edit queued messages before they send (↑↓ move, `Enter`/`Esc` done) |
| `⇧⇥` (Shift+Tab) | cycle input mode (normal → auto-accept → plan) |
| `Esc Esc` | clear the line |
| `^C` | clear the line; on an empty line, `^C` again to exit |
| `^D` | exit on an empty line |

### Modes

`Shift+Tab` cycles the input mode (shown in the status footer, Claude-Code style):

- **normal** — default; dangerous shell commands ask before running.
- **auto-accept** — run flagged shell commands without the `y/N` prompt (it will
  also auto-approve the agentic tool loop once that lands).
- **plan** — injects a planning directive into each request so the model lays
  out an approach for review before acting (the live conversation isn't mutated;
  it's added to the request only).

### Agentic tools

`/tools` toggles the agentic tool loop. With it on, a submitted message runs a
non-streaming loop: the model can call built-in tools — `read_file`, `list_dir`,
`grep`, `write_file`, `edit_file` — and Zero feeds each result back until the
model answers in plain text. Tool calls and results show inline (`⚙ name(args)`
/ `↳ result`).

Gating follows the mode (Shift+Tab): read-only tools always run; **file-modifying
tools (`write_file`/`edit_file`) run only in auto-accept mode** — in normal mode
they're refused with a message the model can act on. Paths are confined to the
working directory. The loop is bounded (step cap + doom-loop guard) so a local
model can't run away.

> Non-streaming on purpose: local servers' *streaming* tool-call parsers are
> buggy (calls split/lost across chunks), so the loop reads each turn whole. Zero
> also recovers tool calls a quantized model emits as `<tool_call>`/```json text.

#### Context efficiency

Local models have small windows (8–32K), and measuring real agentic transcripts
shows **~95% of a turn's context is raw tool I/O, not reasoning** — a tiny long
tail of giant tool results carries most of the bytes, the same file gets read
5–7×, and write payloads sit in history forever. Zero attacks that directly. The
rule throughout is **cap, don't lose** — every drop is re-fetchable (file on
disk, line range, or already upstream in the conversation):

1. **Per-result cap** — any single tool result over `max_tool_output` (default
   4 KB) is collapsed to head + tail with an `… [N bytes elided — <hint>] …`
   marker naming how to re-fetch the rest. Recovers the bulk of the long tail.
2. **Per-turn budget** — cumulative tool output within one turn is bounded
   (default 24 KB); the cap shrinks as the budget depletes, always keeping a
   256-byte floor per result, so a turn firing many calls can't blow the window
   by attrition.
3. **Read cache** — a repeat read of a file unchanged (by mtime + length) since
   you last read it returns a one-line stub instead of the content; invalidated
   on `write_file`/`edit_file` so an edited file re-reads in full.
4. **Two-stage search** — `grep` returns `path:line` pointers (each preview
   capped), and `read_file` takes an optional `offset`/`limit` line range so the
   model fetches just the span a pointer named, not the whole file.
5. **Write compaction** — once a `write_file`/`edit_file` succeeds, its bulky
   content is stripped from the tool-call args in history (the file is on disk);
   a refused/failed write keeps its args so the model can retry.

All of it is pure, std-only (`zero_core::context`), and unit-proven — each lever
has a test asserting both the byte saving and that the full content is still
reachable. The caps are tunable per model via `max_tool_output` / `max_turn_output`
in `config.json`.

**`/context`** reports the *measured* (never estimated) bytes saved this session,
broken down by lever:

```
context savings (measured this session)
  cap:      36.2 KB  (oversized tool results trimmed)
  cache:    18.0 KB  (unchanged re-reads skipped)
  compact:   4.9 KB  (applied write/edit payloads dropped)
  total:    59.1 KB  →  71% smaller window
```

### MCP servers

Zero speaks the [Model Context Protocol](https://modelcontextprotocol.io) over
the stdio transport (zero-dep: a subprocess + JSON-RPC over its pipes, one
message per line).

**You don't have to redeclare servers you already use, or even run a command.**
Zero auto-connects configured MCP servers at startup, importing them from the
tools where they already live, in this precedence order (`/mcp` re-runs it):

1. `./.mcp.json` — the project's own servers (highest precedence)
2. `~/.zero/mcp.json` — Zero's own file
3. **Claude Desktop** — `~/Library/Application Support/Claude/claude_desktop_config.json`
4. **Claude Code** — `~/.claude.json` (global + the current project's servers)

Same name in two sources → the higher-precedence one wins. To add a server just
for Zero, drop it in `~/.zero/mcp.json` (Claude-compatible shape):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
      "env": {}
    }
  }
}
```

Servers connect automatically on launch (silent if none are configured). The
commands re-run or inspect that:

```
/mcp          re-discover from all sources + connect them (shows origin)
/mcp tools    list every discovered tool (name · server · description)
```

HTTP/SSE servers (a `url` instead of a `command`) are skipped — Zero is
stdio-only for now.

> **Discovery only for now.** `/mcp` connects and lists the tools a server
> exposes, but the model can't *call* them yet — that needs the agentic
> tool-call loop, which is the next slice. MCP tools will plug into it then.

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

> **Multiline / Shift+Enter:** Zero enables the kitty keyboard protocol on
> startup, so Shift+Enter works on terminals that support it (kitty, WezTerm,
> Ghostty, recent iTerm2). On terminals that don't, **`^J` is the universal
> newline key** (Return sends CR = submit, `^J` sends LF = newline). Word-wise
> moves (`⌥`/`^` + arrows) likewise depend on the terminal sending the sequence.

### Output rendering & clipboard

Assistant output is rendered as inline Markdown on the fly — `**bold**`,
`*italic*`, `` `code` ``, `#` headings, and fenced code blocks become real
terminal styling (the raw text is kept for the model and for copying).

Copy to the system clipboard (`pbcopy` / `wl-copy` / `xclip`):

- `/clip <n>` — copy code block *n* (blocks render a `── rust · ⧉ copy ──`
  footer marking the target). `/clip` copies the whole last response.

## Test & coverage

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
./scripts/coverage.sh                  # enforces >=98% (fails the build below)
```

**Line coverage is held at ≥98%** (currently ~99%) and enforced by
`scripts/coverage.sh`. The only excluded files are `term.rs` (libc FFI — can't
run without a real tty) and `main.rs` (process bootstrap); all engine and TUI
logic is covered, including the HTTP/SSE client (tested against an in-process
localhost mock). `cargo-llvm-cov` is a dev tool, not a crate dependency.

### Pre-push gate

`scripts/hooks/pre-push` blocks a push unless: (1) no AI / co-author attribution
in any pushed commit, (2) tests pass, (3) line coverage stays ≥98%. Enable it
once per clone:

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
