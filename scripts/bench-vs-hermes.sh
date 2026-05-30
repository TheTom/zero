#!/usr/bin/env bash
# Compare Zero vs Hermes on identical prompts over N iterations against the SAME
# model endpoint (gx10), to measure whether Zero's tool-output compression saves
# tokens/latency on agentic tasks. On-demand benchmark — NOT a gate test (a real
# model is non-deterministic), so it lives in scripts/, never in `cargo test`.
#
# Honest-measurement note: Zero's headless path currently uses a non-streaming
# completion and does not emit per-run token usage, so apples-to-apples *token*
# parity is not yet wired here (see docs/design/followups.md — "headless token
# reporting" prereq). What IS measured today, per tool, per iteration:
#   - wall-clock latency (the shell times it; Zero also measures internally),
#   - bytes of final stdout (a proxy for response verbosity),
#   - exit status (did the run succeed),
#   - for Zero: the /context byte-savings the compression achieved.
# When the token-usage prereq lands, drop the real prompt/completion totals into
# the table below and the comparison becomes exact.
#
# Usage:
#   scripts/bench-vs-hermes.sh [N] [prompt...]
#   N         iterations per tool (default 3)
#   prompt    the task (default: a multi-step shell task that triggers tools)
#
# Requires: a built ./target/release/zero, `hermes` on PATH, and a configured
# gx10 endpoint in ~/.zero/config.json (Zero) and Hermes's own config. Both must
# point at the SAME model for the comparison to be meaningful.
set -uo pipefail
cd "$(dirname "$0")/.."

N="${1:-3}"
shift || true
PROMPT="${*:-List the Rust files under crates/zero-core/src, then tell me which one is largest. Use tools.}"

ZERO_BIN="./target/release/zero"
[ -x "$ZERO_BIN" ] || { echo "build first: cargo build --release" >&2; exit 1; }
command -v hermes >/dev/null || { echo "hermes not on PATH" >&2; exit 1; }

run_dir="$(mktemp -d)"
echo "bench: N=$N  prompt=$(printf '%q' "$PROMPT")"
echo "artifacts: $run_dir"
printf '%-8s %-4s %8s %10s %6s\n' "tool" "it" "ms" "out_bytes" "exit"
printf '%s\n' "------------------------------------------------"

bench_one() {
  local tool="$1" it="$2"; shift 2
  local out="$run_dir/${tool}-${it}.out"
  local start end ms code
  start=$(python3 -c 'import time;print(int(time.time()*1000))')
  "$@" >"$out" 2>"$run_dir/${tool}-${it}.err"
  code=$?
  end=$(python3 -c 'import time;print(int(time.time()*1000))')
  ms=$((end - start))
  printf '%-8s %-4s %8s %10s %6s\n' "$tool" "$it" "$ms" "$(wc -c <"$out" | tr -d ' ')" "$code"
}

for it in $(seq 1 "$N"); do
  # Zero: headless, tools on, real backend from ~/.zero/config.json.
  bench_one zero "$it" "$ZERO_BIN" -p "$PROMPT" --tools
  # Hermes: headless one-shot (-z), default toolset.
  bench_one hermes "$it" hermes -z "$PROMPT" --yolo
done

echo
echo "Zero /context savings (last run): run \`zero\` interactively + /context, or"
echo "parse the run's ~/.zero/outputs/<ts>/ artifacts. Token parity: pending the"
echo "headless-usage prereq (docs/design/followups.md)."
echo "Raw outputs + stderr traces saved under: $run_dir"
