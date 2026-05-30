#!/usr/bin/env bash
# Recovery-policy ablation: when the guard detects a stuck loop, what works best —
# STOP (abandon), NUDGE (inject guidance, current default), or RESET (discard the
# polluted history and restart from task + concrete progress summary)?
#
# Hypothesis (from loop-recovery research): a nudge can't break a degeneration loop
# because the repetition is still in context; a RESET to a clean context is the
# mechanistically-correct fix — and since the wander usually happens AFTER the work
# is basically done, reset+continue should convert hard-stop FAILURES into completed
# tasks at lower total tokens.
#
# Same gx10 model, the wander-prone task, N reps each. Per arm we measure:
#   %completed (turn ended Done, not guard-stopped), avg total tokens, avg quality
#   (judge.py --single on the produced file), avg tool calls.
# Best policy = highest %completed × quality at the lowest tokens.
#
# Usage: scripts/ablate-recovery.sh [REPS]   (default 6)
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-6}"
GX="${ABLATE_URL:-http://192.168.50.125:8000}"
ZERO_ABS="$(cd "$(dirname ./target/release/zero)" && pwd)/zero"
[ -x "$ZERO_ABS" ] || { echo "build first: cargo build --release" >&2; exit 1; }
W="$(mktemp -d)"; PORT=8070
echo "recovery ablation: reps=$REPS  artifacts=$W"

# Wander-prone task (induces the polish-loop): vague "make it solid / fix anything".
TASK='Build a polished terminal Tic-Tac-Toe in Python as tictactoe.py in the current directory: 2-player, input validation, win/draw detection, a clean board UI. Make it really solid. Run it with piped moves to verify, and fix anything that is not perfect before finishing. One file.'
echo "$TASK" > "$W/prompt.txt"

results="$W/results.tsv"
printf 'policy\trep\ttotal\tcalls\tcompleted\tquality\n' > "$results"

for policy in stop nudge reset; do
  for rep in $(seq 1 "$REPS"); do
    run="$W/$policy-$rep"; mkdir -p "$run"
    : > "$W/tok.log"
    UPSTREAM="$GX" TOKEN_LOG="$W/tok.log" python3 scripts/token-proxy.py "$PORT" >/dev/null 2>&1 &
    px=$!
    until curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 0.3; done
    ( cd "$run" && HOME="$W/home" ZERO_RECOVERY="$policy" timeout 600 "$ZERO_ABS" -p "$TASK" \
        --tools --accept-edits --url "http://127.0.0.1:$PORT" --no-log >zero.out 2>zero.err )
    kill $px 2>/dev/null
    total=$(awk '{p+=$2;c+=$3} END{print int(p+c)}' "$W/tok.log")
    calls=$(grep -c '⚙' "$run/zero.err" 2>/dev/null || echo 0)
    # completed = the guard did NOT stop it (no "guard: stopping" marker)
    completed=1; grep -q 'guard: stopping' "$run/zero.err" 2>/dev/null && completed=0
    q="NA"
    if [ -f "$run/tictactoe.py" ]; then
      q=$(JUDGE_URL="$GX" python3 scripts/judge.py --single --prompt-file "$W/prompt.txt" \
            --a "$run/tictactoe.py" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["total"])' 2>/dev/null || echo NA)
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$policy" "$rep" "${total:-0}" "$calls" "$completed" "$q" >> "$results"
    printf '  %-6s r%s  total=%-7s calls=%-3s completed=%s quality=%s\n' "$policy" "$rep" "${total:-0}" "$calls" "$completed" "$q"
  done
done

echo; echo "== per-policy summary =="
python3 - "$results" <<'PY'
import sys, csv
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
p=defaultdict(lambda: {"tot":[], "q":[], "calls":[], "done":0, "n":0})
for r in rows:
    d=p[r['policy']]; d["n"]+=1
    d["tot"].append(int(r['total'])); d["calls"].append(int(r['calls']))
    if r['completed']=="1": d["done"]+=1
    if r['quality']!="NA": d["q"].append(float(r['quality']))
def avg(x): return sum(x)//len(x) if x else 0
def favg(x): return round(sum(x)/len(x),1) if x else float('nan')
print(f"{'policy':6} {'n':>2} {'%completed':>10} {'avg_tok':>8} {'avg_calls':>9} {'avg_quality':>11}")
for pol in ("stop","nudge","reset"):
    d=p.get(pol)
    if not d or d["n"]==0: continue
    print(f"{pol:6} {d['n']:>2} {100*d['done']//d['n']:>9}% {avg(d['tot']):>8} {avg(d['calls']):>9} {favg(d['q']):>11}")
print("\nBest = highest %completed AND quality at lowest tokens. Reset should lift")
print("%completed (rescues hard-stops) without a quality drop.")
print(f"raw: {sys.argv[1]}")
PY
echo "artifacts: $W"
