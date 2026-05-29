# North Stars

The bets that justify building Zero instead of using an existing harness. Each
is a *direction*, not a checkbox — we'll know we're winning when these feel
obviously better than the alternatives.

---

## 1. Claude-Code terminal feel

The terminal should behave like Claude Code's: output flows into the terminal's
**native scrollback**, only the current input line is owned and redrawn in
place. No rigid full-screen takeover, no fighting the user's scroll. The other
harnesses feel stiff; this is the thing to copy.

**Status:** done in slice 1 (`zero-tui/app.rs`, inline render).

## 2. Terminal ≡ App

Every capability lives in `zero-core` (dependency-free engine). Frontends are
thin shells over the `Backend` trait and friends. **Anything the terminal can
do, the app can do**, because the capability never lives in the UI.

**Status:** seam established (`zero_core::Backend`). Second frontend is slice 5.

## 3. Honest timestamps

> "no one keeps track of timestamps… they don't give it to the agent and time
> estimates drive me insane. 'we spent a week doing this' — no, we spent an hour."

Zero **measures, never estimates.** `clock.rs` only reports real monotonic
elapsed time, and only after the fact. There is no code path that predicts a
duration. Every turn shows the actual wall time it took.

**Status:** done in slice 1 (`zero-core/clock.rs`, dimmed elapsed per turn).

## 4. Rich, replayable logging

Claude logs well; copy that. Every turn appends to a JSONL transcript with a
**real wall-clock timestamp**. Logs should be enough to replay or audit a
session, never a lossy summary.

**Status:** baseline in slice 1 (`zero-core/session.rs`). Expand with tool calls
and token usage as those land.

## 5. Compaction done right

The hard problem, and the one worth the most effort. Most agents compact badly:
they summarize, lose the thread, and then **resurface stale context from way
earlier** in the previous window as the thing to continue. We want compaction
that preserves the *live* working set and recent intent, not a random-feeling
jump backwards.

**Status:** deliberately deferred — it needs the agentic loop (slice 3) to exist
first so there's real context shape to compact. This is its own slice (4) and
gets real design time, not a quick heuristic.

## 6. Instructions that are actually followed

> "the equivalent CLAUDE.md… holy shit does it piss me off. It's never followed.
> We need to make sure it's always followed."

The project-instruction file (Zero's equivalent of `CLAUDE.md`) must be
**reliably obeyed**, not politely ignored after a few turns. This is an open
research problem; the failure modes we're targeting:

- **Decay over turns.** Instructions injected once at the top of context lose
  influence as the conversation grows and as compaction (#5) rewrites history.
- **Soft phrasing.** "Please try to…" reads as optional. Rules need to be
  unambiguous and machine-checkable where possible.
- **No enforcement.** Nothing verifies the model actually complied, so
  violations pass silently.

Directions to explore (none committed yet):

1. **Re-inject every turn, not once.** Keep the rules in a high-salience system
   position on *every* request, and make compaction treat them as pinned/never
   summarizable. Cheap, and probably the single biggest win.
2. **Hard vs soft rules.** Split the instruction file into rules that can be
   *enforced in code* (e.g. "never run `python`, use `python3`" → a tool-call
   pre-check that rewrites/blocks) vs. style guidance that stays advisory. The
   enforceable ones don't rely on the model remembering.
3. **Post-turn adherence check.** A lightweight pass (possibly a cheap local
   model call, or pure rules) that flags when output violated a stated rule, and
   surfaces or auto-corrects it — closing the "no enforcement" gap.
4. **Salience over volume.** A short, sharp, well-placed rule set beats a long
   wall of markdown the model skims. Tooling to keep the file tight.

This is a north star, not a slice yet — but it shapes how we build the prompt
assembly and compaction layers, so it's captured now so we don't paint
ourselves into a corner. Tracked alongside #5 because compaction is where
instruction salience most often dies.

---

## Non-negotiable constraint: zero dependencies

Cuts across all of the above. The runtime has **no Rust crate dependencies** —
`std` only. It's a constraint *and* a feature: full control over the prompt
assembly, the streaming path, and the compaction logic, with nothing we can't
read and change. Toolchain components (clippy, llvm coverage) are fine; crates
are not.
