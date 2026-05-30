#!/usr/bin/env bash
# Compare Zero vs Hermes on identical prompts over N iterations against the SAME
# model endpoint (gx10), to measure whether Zero's tool-output compression saves
# tokens on agentic tasks. On-demand benchmark — NOT a gate test (a real model is
# non-deterministic), so it lives in scripts/, never in `cargo test`.
#
# Token measurement: Zero prints `[usage: prompt=N completion=N total=N]` to
# stderr on a headless run — server-reported, summed across agentic rounds, never
# estimated. This script parses that for Zero's real total. Hermes token
# accounting is best-effort: we grep its output for a usage/token line; if it
# emits none in `-z` mode the Hermes total shows `?` (latency + bytes still
# compared). Per tool / iteration we record: latency, stdout bytes, exit, tokens.
#
# Usage:
#   scripts/bench-vs-hermes.sh [N] [prompt...]
#   N         iterations per tool (default 5 — enough to average out nondeterminism)
#   prompt    the task (default: a multi-step shell task that triggers tools)
#
# Requires: a built ./target/release/zero, `hermes` on PATH, and the SAME model
# configured for both (Zero: ~/.zero/config.json; Hermes: its own config).
set -uo pipefail
cd "$(dirname "$0")/.."

N="${1:-5}"
shift || true
PROMPT="${*:-List the Rust files under crates/zero-core/src, then tell me which one is largest. Use tools.}"

ZERO_BIN="./target/release/zero"
[ -x "$ZERO_BIN" ] || { echo "build first: cargo build --release" >&2; exit 1; }
command -v hermes >/dev/null || { echo "hermes not on PATH" >&2; exit 1; }

run_dir="$(mktemp -d)"
echo "bench: N=$N  prompt=$(printf '%q' "$PROMPT")"
echo "artifacts: $run_dir"
[ -n "${ZERO_BENCH_EXPECT:-}" ] && echo "correctness check: stdout must match /${ZERO_BENCH_EXPECT}/"
printf '%-8s %-4s %8s %10s %8s %6s %4s\n' "tool" "it" "ms" "out_bytes" "tokens" "exit" "ok"
printf '%s\n' "------------------------------------------------------------"

# Pull a total-token count out of a tool's combined stdout+stderr. Zero emits the
# canonical `[usage: … total=N]`; for Hermes we fall back to the first
# "total*tokens*N" / "tokens: N" shaped line. Echoes the number, or "?" if none.
extract_tokens() {
  local f="$1"
  # Zero's exact line first.
  local n
  n=$(grep -oE '\[usage:[^]]*total=[0-9]+\]' "$f" 2>/dev/null | grep -oE 'total=[0-9]+' | grep -oE '[0-9]+' | tail -1)
  if [ -n "${n:-}" ]; then echo "$n"; return; fi
  # Generic fallback for Hermes-style output.
  n=$(grep -oiE 'total[^0-9]{0,12}([0-9]{2,})[^0-9]*tokens|tokens[^0-9]{0,4}([0-9]{2,})' "$f" 2>/dev/null \
        | grep -oE '[0-9]{2,}' | head -1)
  echo "${n:-?}"
}

# accumulators (bash arithmetic; "?" tokens skip the sum)
# Optional correctness check: if ZERO_BENCH_EXPECT is set to a regex naming the
# fact both wrappers should surface (e.g. the known-largest filename), each run's
# stdout is matched against it. This measures the GOAL's *equivalent data* half
# alongside the token half — one run proves both: same answer, fewer tokens.
EXPECT="${ZERO_BENCH_EXPECT:-}"

declare -A SUM_MS SUM_TOK CNT_TOK OK_HITS
for t in zero hermes; do SUM_MS[$t]=0; SUM_TOK[$t]=0; CNT_TOK[$t]=0; OK_HITS[$t]=0; done

bench_one() {
  local tool="$1" it="$2"; shift 2
  local out="$run_dir/${tool}-${it}.out" err="$run_dir/${tool}-${it}.err"
  local start end ms code tok ok
  start=$(python3 -c 'import time;print(int(time.time()*1000))')
  "$@" >"$out" 2>"$err"
  code=$?
  end=$(python3 -c 'import time;print(int(time.time()*1000))')
  ms=$((end - start))
  tok=$(extract_tokens "$err"); [ "$tok" = "?" ] && tok=$(extract_tokens "$out")
  ok="-"
  if [ -n "$EXPECT" ]; then
    if grep -qiE "$EXPECT" "$out"; then ok="Y"; OK_HITS[$tool]=$(( OK_HITS[$tool] + 1 )); else ok="N"; fi
  fi
  printf '%-8s %-4s %8s %10s %8s %6s %4s\n' "$tool" "$it" "$ms" "$(wc -c <"$out" | tr -d ' ')" "$tok" "$code" "$ok"
  SUM_MS[$tool]=$(( SUM_MS[$tool] + ms ))
  if [[ "$tok" =~ ^[0-9]+$ ]]; then
    SUM_TOK[$tool]=$(( SUM_TOK[$tool] + tok )); CNT_TOK[$tool]=$(( CNT_TOK[$tool] + 1 ))
  fi
}

for it in $(seq 1 "$N"); do
  bench_one zero   "$it" "$ZERO_BIN" -p "$PROMPT" --tools
  bench_one hermes "$it" hermes -z "$PROMPT" --yolo
done

echo
echo "== averages =="
for t in zero hermes; do
  avg_ms=$(( SUM_MS[$t] / N ))
  if [ "${CNT_TOK[$t]}" -gt 0 ]; then
    avg_tok=$(( SUM_TOK[$t] / CNT_TOK[$t] ))
    printf '%-8s avg_ms=%-8s avg_tokens=%-8s (n=%s with usage)\n' "$t" "$avg_ms" "$avg_tok" "${CNT_TOK[$t]}"
  else
    printf '%-8s avg_ms=%-8s avg_tokens=? (no usage line found)\n' "$t" "$avg_ms"
  fi
done
if [ "${CNT_TOK[zero]}" -gt 0 ] && [ "${CNT_TOK[hermes]}" -gt 0 ]; then
  zt=$(( SUM_TOK[zero] / CNT_TOK[zero] )); ht=$(( SUM_TOK[hermes] / CNT_TOK[hermes] ))
  if [ "$ht" -gt 0 ]; then
    echo "Zero uses $(( zt * 100 / ht ))% of Hermes's tokens on this task (lower = Zero saves)."
  fi
fi

# The GOAL verdict: equivalent data (both got the right answer) AND fewer tokens.
if [ -n "$EXPECT" ]; then
  echo
  echo "== equivalent-data check (matched /$EXPECT/) =="
  printf '%-8s %s/%s runs correct\n' "zero" "${OK_HITS[zero]}" "$N"
  printf '%-8s %s/%s runs correct\n' "hermes" "${OK_HITS[hermes]}" "$N"
  if [ "${OK_HITS[zero]}" -eq "$N" ] && [ "${OK_HITS[hermes]}" -eq "$N" ]; then
    echo "GOAL: both wrappers produced equivalent data on every run."
    [ "${CNT_TOK[zero]:-0}" -gt 0 ] && [ "${CNT_TOK[hermes]:-0}" -gt 0 ] && \
      echo "      → same answers, compare the token column above for the efficiency win."
  else
    echo "NOTE: not every run matched — inspect the mismatching outputs in $run_dir"
    echo "      (model nondeterminism, or a real difference between wrappers)."
  fi
fi
echo
echo "Raw outputs + stderr traces under: $run_dir"
echo "Tip: set ZERO_BENCH_EXPECT='<regex of the expected fact>' to check equivalence."
