#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

# Nudge ablation: when the guard fires, which WORDING makes the model converge
# fastest? Two axes — tone (soft / firm-negative+redirect) × specificity (generic /
# names-the-stuck-action) — so 4 variants, plus the shipped default as a control.
# Same gx10 model, a wander-prone task, N runs each. We can't force a nudge (the
# model wanders nondeterministically), so we run many times and report, per variant:
#   runs, %runs-that-nudged, avg total tokens, avg tool calls, %runs-that-STOPPED
#   (stopped = nudge failed to rescue → bad), and avg tokens AMONG nudged runs.
# Lower tokens + fewer stops among nudged runs = the nudge that actually rescues.
#
# Usage: scripts/ablate-nudge.sh [REPS]   (default 6)
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-6}"
GX="${ABLATE_URL:-http://192.168.50.125:8000}"
ZERO_ABS="$(cd "$(dirname ./target/release/zero)" && pwd)/zero"
[ -x "$ZERO_ABS" ] || { echo "build first: cargo build --release" >&2; exit 1; }
W="$(mktemp -d)"; PORT=8073
echo "nudge ablation: reps=$REPS  artifacts=$W"

# {tool}/{count} are filled by the guard at fire time.
N_SOFT_GEN='You appear to be repeating tool calls without making progress. Stop exploring: finalize your work now, verify it once, and give your best answer.'
N_SOFT_SPEC='You have called {tool} {count} times without finishing. Consider finalizing: verify your work once and give your best answer.'
N_FIRM_GEN='STOP. You are looping and wasting effort with no progress. Do NOT keep calling tools. Instead: run your solution once to confirm it works, then give your final answer now.'
N_FIRM_SPEC='STOP. You have called {tool} {count} times and are not converging. Do NOT call {tool} again. Instead: run the solution once to confirm it works, then give your final answer now.'

variant_text() {
  case "$1" in
    default)   echo "" ;;   # empty → guard uses its built-in DEFAULT_NUDGE
    soft_gen)  echo "$N_SOFT_GEN" ;;
    soft_spec) echo "$N_SOFT_SPEC" ;;
    firm_gen)  echo "$N_FIRM_GEN" ;;
    firm_spec) echo "$N_FIRM_SPEC" ;;
  esac
}

# A task that reliably tempts wandering: vague + "make it really good" + verify loop.
TASK='Build a polished terminal Tic-Tac-Toe in Python as tictactoe.py in the current directory: 2-player, input validation, win/draw detection, a clean board UI. Make it really solid. Run it with piped moves to verify, and fix anything that is not perfect before finishing. One file.'

results="$W/results.tsv"
printf 'variant\trep\ttotal\tcalls\tnudged\tstopped\n' > "$results"

for variant in default soft_gen soft_spec firm_gen firm_spec; do
  vt="$(variant_text "$variant")"
  for rep in $(seq 1 "$REPS"); do
    run="$W/$variant-$rep"; mkdir -p "$run"
    : > "$W/tok.log"
    UPSTREAM="$GX" TOKEN_LOG="$W/tok.log" python3 scripts/token-proxy.py "$PORT" >/dev/null 2>&1 &
    px=$!
    until curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 0.3; done
    ( cd "$run" && HOME="$W/home" ZERO_NUDGE="$vt" timeout 500 "$ZERO_ABS" -p "$TASK" \
        --tools --accept-edits --url "http://127.0.0.1:$PORT" --no-log >zero.out 2>zero.err )
    kill $px 2>/dev/null
    total=$(awk '{p+=$2;c+=$3} END{print int(p+c)}' "$W/tok.log")
    calls=$(grep -c '⚙' "$run/zero.err" 2>/dev/null || echo 0)
    nudged=$(grep -c 'guard:' "$run/zero.err" 2>/dev/null || echo 0)
    # 'guard: stopping' = the nudge FAILED to rescue (had to hard-stop)
    stopped=$(grep -c 'guard: stopping' "$run/zero.err" 2>/dev/null || echo 0)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$variant" "$rep" "${total:-0}" "$calls" "$nudged" "$stopped" >> "$results"
    printf '  %-10s r%s  total=%-7s calls=%-3s nudges=%-2s stop=%s\n' "$variant" "$rep" "${total:-0}" "$calls" "$nudged" "$stopped"
  done
done

echo; echo "== per-variant summary =="
python3 - "$results" <<'PY'
import sys, csv
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
v=defaultdict(lambda: {"tot":[], "calls":[], "nudged":0, "stopped":0, "n":0, "nud_tot":[]})
for r in rows:
    d=v[r['variant']]; d["n"]+=1
    d["tot"].append(int(r['total'])); d["calls"].append(int(r['calls']))
    was_nudged = int(r['nudged'])>0
    if was_nudged: d["nudged"]+=1; d["nud_tot"].append(int(r['total']))
    if int(r['stopped'])>0: d["stopped"]+=1
def avg(x): return sum(x)//len(x) if x else 0
print(f"{'variant':10} {'n':>2} {'avg_tok':>8} {'avg_calls':>9} {'%nudged':>7} {'%stopped':>8} {'avg_tok|nudged':>14}")
for name in ("default","soft_gen","soft_spec","firm_gen","firm_spec"):
    d=v.get(name)
    if not d or d["n"]==0: continue
    print(f"{name:10} {d['n']:>2} {avg(d['tot']):>8} {avg(d['calls']):>9} "
          f"{100*d['nudged']//d['n']:>6}% {100*d['stopped']//d['n']:>7}% {avg(d['nud_tot']):>14}")
print("\nBest nudge = lowest 'avg_tok|nudged' with lowest %stopped (rescued the wander cheaply).")
print(f"raw: {sys.argv[1]}")
PY
echo "artifacts: $W"
