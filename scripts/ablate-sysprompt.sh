#!/usr/bin/env bash
# System-prompt ablation: which minimal prompt delta recovers code QUALITY, and
# what does it cost in tokens? One independent variable (the system prompt),
# additive arms, same gx10 model, blind same-model judge (scripts/judge.py).
#
# Arms (additive):
#   V1  current shipped default (lean: concise + tool-first + re-fetch)
#   V2  V1 + quality clause (exact spec/IO mode, validate + edge cases, run-to-verify)
#   V3  V2 + standards clause (correctness/completeness over brevity, production-grade)
# Ceiling: Hermes's full prompt is the reference (run bench-vs-hermes separately).
#
# Per (arm × task × rep) we capture: quality (0-50, judge --single) + system-prompt
# token cost + total turn tokens (via token-proxy). Output: a table + per-arm means.
#
# Usage: scripts/ablate-sysprompt.sh [REPS]   (default 2)
set -uo pipefail
cd "$(dirname "$0")/.."

REPS="${1:-2}"
GX="${ABLATE_URL:-http://192.168.50.125:8000}"
ZERO="./target/release/zero"
[ -x "$ZERO" ] || { echo "build first: cargo build --release" >&2; exit 1; }

W="$(mktemp -d)"; PORT=8077
ZERO_ABS="$(cd "$(dirname "$ZERO")" && pwd)/$(basename "$ZERO")"
REPO="$(pwd)"
echo "ablation: reps=$REPS  model=gx10  artifacts=$W"

# ── Arm system prompts ────────────────────────────────────────────────────────
# V1 = current shipped lean default.
# V2 = V1 + my hand-written quality clause (spec/IO mode, validate, run-to-verify).
# V3 = V2 + "standards/production-grade" — the clause RESEARCH flagged as the
#      unproven/redundant bit (persona-adjacent steering); kept as a control.
# V4 = V1 + the EVIDENCE-BACKED clause from the literature ablation: general
#      solution / no hard-coding (the #1 impact-per-token lever), full requirement
#      coverage, edge cases, plan-briefly, and self-verify (Reflexion +11pp). ~30 tok.
V1='You are Zero, a terminal coding assistant. Prefer tools over guessing: read and search the real files before you answer or edit, and make minimal, correct changes. Be concise. Avoid destructive shell commands unless asked. Tool output may be capped with a marker — when you need the omitted part, re-fetch it (read_file with offset/limit, or read the named artifact path) rather than assuming it.'
QUALITY=' Satisfy every stated requirement exactly — if the task names an input mode (piped stdin, interactive, args), support that mode. Validate inputs and handle edge cases. Before claiming done, actually run the code and confirm it works.'
STANDARDS=' Favor correctness and completeness over brevity; write production-grade code, not a sketch.'
EVIDENCE=' Implement a correct, general solution that works for all valid inputs — never hard-code or special-case just to pass the examples. Implement every stated requirement fully, including the named input mode; handle edge cases and validate inputs. Briefly plan before coding. Before finishing, run the code and confirm it meets every requirement.'

# arm -> system prompt
arm_prompt() {
  case "$1" in
    V1) printf '%s' "$V1" ;;
    V2) printf '%s%s' "$V1" "$QUALITY" ;;
    V3) printf '%s%s%s' "$V1" "$QUALITY" "$STANDARDS" ;;
    V4) printf '%s%s' "$V1" "$EVIDENCE" ;;
  esac
}

# ── Tasks (spec-rich so quality differences show) ─────────────────────────────
T1='Create a simple terminal Tic-Tac-Toe game in Python in the current directory. Write tictactoe.py with a playable 2-player game, then run it once piped with moves to verify it doesn'\''t crash. Keep it one file.'
T2='Create a command-line todo app in Python in the current directory as todo.py. It must support: add a task, list tasks, mark a task done, and delete a task — taking the action and arguments from the command line (e.g. python todo.py add "buy milk"). Persist tasks to a JSON file. Then run it to add two tasks and list them, to verify it works. One file.'
declare_task() { case "$1" in T1) printf '%s' "$T1";; T2) printf '%s' "$T2";; esac; }
task_file() { case "$1" in T1) echo tictactoe.py;; T2) echo todo.py;; esac; }

# write a config file for an arm (JSON, escaping the prompt safely via python3)
arm_config() {
  local arm="$1" cfg="$2"
  arm_prompt "$arm" | python3 -c '
import json,sys
sp=sys.stdin.read()
json.dump({"base_url":"'"$GX"'","model":"local","api_key":"","temperature":None,"system_prompt":sp}, open(sys.argv[1],"w"))
' "$cfg"
}

# ── Run one (arm × task × rep): returns via files in $W ───────────────────────
results="$W/results.tsv"
printf 'arm\ttask\trep\tsys_tok\ttotal_tok\tquality\n' > "$results"

for arm in V1 V2 V3 V4; do
  cfg="$W/cfg-$arm.json"; arm_config "$arm" "$cfg"
  systok=$(arm_prompt "$arm" | python3 -c 'import sys;print(len(sys.stdin.read())//4)')  # ~4 chars/tok
  for task in T1 T2; do
    tf=$(task_file "$task")
    reps_here=$REPS; [ "$task" = "T2" ] && reps_here=1   # T2 generalization check: 1 rep
    for rep in $(seq 1 "$reps_here"); do
      run="$W/$arm-$task-$rep"; mkdir -p "$run"
      : > "$W/tok.log"
      UPSTREAM="$GX" TOKEN_LOG="$W/tok.log" python3 scripts/token-proxy.py "$PORT" >/dev/null 2>&1 &
      px=$!
      until curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 0.3; done
      ( cd "$run" && HOME="$W/home" timeout 500 "$ZERO_ABS" -p "$(declare_task "$task")" \
          --tools --accept-edits --url "http://127.0.0.1:$PORT" --config "$cfg" --no-log \
          >zero.reply 2>zero.err )
      kill $px 2>/dev/null
      total=$(awk '{p+=$2;c+=$3} END{print int((p+c))}' "$W/tok.log")
      # judge the produced file (absolute, single-artifact)
      q="NA"
      if [ -f "$run/$tf" ]; then
        printf '%s' "$(declare_task "$task")" > "$run/prompt.txt"
        q=$(JUDGE_URL="$GX" python3 scripts/judge.py --single --prompt-file "$run/prompt.txt" \
              --a "$run/$tf" 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["total"])' 2>/dev/null || echo "NA")
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$task" "$rep" "$systok" "${total:-0}" "$q" >> "$results"
      printf '  %-3s %-3s r%s  sys=%-4s total=%-7s quality=%s\n' "$arm" "$task" "$rep" "$systok" "${total:-0}" "$q"
    done
  done
done

echo; echo "== per-arm means (quality /50, tokens) =="
python3 - "$results" <<'PY'
import sys, csv
from collections import defaultdict
rows=list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
arms=defaultdict(lambda: {"q":[], "tot":[], "sys":None})
for r in rows:
    a=arms[r['arm']]; a["sys"]=r['sys_tok']
    if r['quality']!="NA": a["q"].append(float(r['quality']))
    if r['total_tok'].isdigit(): a["tot"].append(int(r['total_tok']))
print(f"{'arm':4} {'sys_tok':>7} {'avg_total':>9} {'avg_quality':>11} {'n':>3}")
for arm in ("V1","V2","V3","V4"):
    a=arms[arm]
    q=sum(a["q"])/len(a["q"]) if a["q"] else float('nan')
    t=sum(a["tot"])//len(a["tot"]) if a["tot"] else 0
    print(f"{arm:4} {a['sys']:>7} {t:>9} {q:>11.1f} {len(a['q']):>3}")
print("\nDecision: ship the lowest-token arm whose quality matches Hermes's ceiling (43.0).")
print(f"raw rows: {sys.argv[1]}")
PY
echo "artifacts kept under: $W"
