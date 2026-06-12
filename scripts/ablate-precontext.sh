#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

# Pre-context mimicry test: does feeding ZERO'S model the relevant parts of
# HERMES'S 21.6KB system prompt move code quality? Hypothesis from the teardown:
# Hermes's prompt is 65% skills-catalog + 30% memory/persona/CLI-format — almost
# no generic coding guidance. The one transferable, plausibly-quality-driving
# piece is the 818-char USER PROFILE (personalized "values correctness, thorough,
# no skipping steps, production app"). This isolates that.
#
# Arms (system_prompt fed via --config; NOT baked into Zero's code):
#   Z       Zero's shipped default (control)
#   Z+PROF  Zero default + Hermes's verbatim user-profile steering
#   Z+CLI   Zero default + Hermes's CLI-agent operating framing
#   Z+BOTH  Zero default + both
#   HERMES  Hermes's FULL 21.6KB prompt verbatim (the upper-bound reference)
#
# Same gx10 model, judged blind by scripts/judge.py --single (calibrated). N reps.
# Reuses the verbatim Hermes pieces in /tmp/hsys2/ (profile.txt, cli.txt, system.txt).
# Run AFTER the nudge ablation finishes (one live experiment at a time).
#
# Usage: scripts/ablate-precontext.sh [REPS]   (default 3)
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-3}"
GX="${ABLATE_URL:-http://192.168.50.125:8000}"
ZERO_ABS="$(cd "$(dirname ./target/release/zero)" && pwd)/zero"
[ -x "$ZERO_ABS" ] || { echo "build first: cargo build --release" >&2; exit 1; }
HSYS="${HSYS_DIR:-/tmp/hermes-precontext}"
for f in profile.txt cli.txt system.txt; do
  [ -s "$HSYS/$f" ] || { echo "missing $HSYS/$f — capture Hermes's prompt first" >&2; exit 1; }
done
W="$(mktemp -d)"; PORT=8071
echo "pre-context test: reps=$REPS  hermes-pieces=$HSYS  artifacts=$W"

ZDEF='You are Zero, a terminal coding assistant. Prefer tools over guessing: read and search the real files before you answer or edit, and make minimal, correct changes. Be concise. Avoid destructive shell commands unless asked. Tool output may be capped with a marker — when you need the omitted part, re-fetch it (read_file with offset/limit, or read the named artifact path) rather than assuming it.'
PROF="$(cat "$HSYS/profile.txt")"
CLI="$(cat "$HSYS/cli.txt")"
HFULL="$(cat "$HSYS/system.txt")"

arm_prompt() {
  case "$1" in
    Z)      printf '%s' "$ZDEF" ;;
    Z+PROF) printf '%s\n\n%s' "$ZDEF" "$PROF" ;;
    Z+CLI)  printf '%s\n\n%s' "$ZDEF" "$CLI" ;;
    Z+BOTH) printf '%s\n\n%s\n\n%s' "$ZDEF" "$CLI" "$PROF" ;;
    HERMES) printf '%s' "$HFULL" ;;
  esac
}

TASK='Create a simple terminal Tic-Tac-Toe game in Python in the current directory. Write tictactoe.py with a playable 2-player game, then run it once piped with moves to verify it doesn'\''t crash. Keep it one file.'
echo "$TASK" > "$W/prompt.txt"

arm_config() {
  arm_prompt "$1" | python3 -c '
import json,sys
json.dump({"base_url":"'"$GX"'","model":"local","api_key":"","temperature":None,"system_prompt":sys.stdin.read()}, open(sys.argv[1],"w"))
' "$2"
}

results="$W/results.tsv"
printf 'arm\trep\tsys_tok\ttotal_tok\tquality\n' > "$results"

for arm in Z Z+PROF Z+CLI Z+BOTH HERMES; do
  cfg="$W/cfg-$arm.json"; arm_config "$arm" "$cfg"
  systok=$(arm_prompt "$arm" | python3 -c 'import sys;print(len(sys.stdin.read())//4)')
  for rep in $(seq 1 "$REPS"); do
    run="$W/$arm-$rep"; mkdir -p "$run"
    : > "$W/tok.log"
    UPSTREAM="$GX" TOKEN_LOG="$W/tok.log" python3 scripts/token-proxy.py "$PORT" >/dev/null 2>&1 &
    px=$!
    until curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 0.3; done
    ( cd "$run" && HOME="$W/home" timeout 500 "$ZERO_ABS" -p "$TASK" \
        --tools --accept-edits --url "http://127.0.0.1:$PORT" --config "$cfg" --no-log \
        >zero.out 2>zero.err )
    kill $px 2>/dev/null
    total=$(awk '{p+=$2;c+=$3} END{print int(p+c)}' "$W/tok.log")
    q="NA"
    if [ -f "$run/tictactoe.py" ]; then
      q=$(JUDGE_URL="$GX" python3 scripts/judge.py --single --prompt-file "$W/prompt.txt" \
            --a "$run/tictactoe.py" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["total"])' 2>/dev/null || echo NA)
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$rep" "$systok" "${total:-0}" "$q" >> "$results"
    printf '  %-7s r%s  sys=%-5s total=%-7s quality=%s\n' "$arm" "$rep" "$systok" "${total:-0}" "$q"
  done
done

echo; echo "== per-arm means =="
python3 - "$results" <<'PY'
import sys, csv
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
a=defaultdict(lambda: {"q":[], "tot":[], "sys":None})
for r in rows:
    d=a[r['arm']]; d["sys"]=r['sys_tok']
    if r['quality']!="NA": d["q"].append(float(r['quality']))
    if r['total_tok'].isdigit(): d["tot"].append(int(r['total_tok']))
print(f"{'arm':7} {'sys_tok':>7} {'avg_total':>9} {'avg_quality':>11} {'n':>3}")
for arm in ("Z","Z+PROF","Z+CLI","Z+BOTH","HERMES"):
    d=a.get(arm)
    if not d: continue
    q=sum(d["q"])/len(d["q"]) if d["q"] else float('nan')
    t=sum(d["tot"])//len(d["tot"]) if d["tot"] else 0
    print(f"{arm:7} {d['sys']:>7} {t:>9} {q:>11.1f} {len(d['q']):>3}")
print("\nQ: does any Hermes pre-context piece raise Zero's quality above the Z control,")
print("and is the token cost worth it? (HERMES = full-prompt upper bound.)")
print(f"raw: {sys.argv[1]}")
PY
echo "artifacts: $W"
