#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

# Compare Zero vs Hermes on identical prompts over N iterations against the SAME
# model endpoint (gx10), to measure whether Zero's tool-output compression saves
# tokens on agentic tasks. On-demand benchmark — NOT a gate test (a real model is
# non-deterministic), so it lives in scripts/, never in `cargo test`.
#
# TOKENS ARE HALF THE STORY. "6% of the tokens" is meaningless if the output is
# worse. For build/deliverable tasks, pair this with scripts/judge.py — a blind,
# position-debiased, same-model rubric judge — to capture QUALITY too. Only claim a
# win when judge.py says quality holds AND this says tokens dropped. (For the simple
# fact-retrieval default task here, ZERO_BENCH_EXPECT covers the quality half.)
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

# ── Optional proxy mode (ZERO_BENCH_PROXY=1) ──────────────────────────────────
# Hermes reports no token usage of its own, so a same-model token race needs a
# neutral meter. scripts/token-proxy.py sits in front of gx10 and tees each
# response's server-reported usage to a log — identical accounting for whichever
# wrapper made the call (validated: it matches Zero's own [usage:] exactly). In
# proxy mode we route BOTH wrappers through it and read the per-run log delta,
# so the token column becomes apples-to-apples and measured, never estimated.
#
# Routing: Zero takes --url (no config touch). Hermes has no config-path flag, so
# we back up ~/.hermes/config.yaml, rewrite its base_url to the proxy, and a trap
# ALWAYS restores it on exit (even on Ctrl-C / error) so the user's config is safe.
PROXY_MODE=0
PROXY_LOG="$run_dir/token-proxy.log"
ZERO_URL_ARGS=()
if [ "${ZERO_BENCH_PROXY:-}" = "1" ]; then
  PROXY_MODE=1
  PROXY_PORT="${ZERO_BENCH_PROXY_PORT:-8099}"
  # Upstream = Zero's configured base_url (the real gx10), else a sane default.
  UPSTREAM="${ZERO_UPSTREAM:-$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.zero/config.json"))).get("base_url",""))' 2>/dev/null)}"
  UPSTREAM="${UPSTREAM:-http://192.168.50.125:8000}"
  HCFG="$HOME/.hermes/config.yaml"
  HCFG_BAK="$run_dir/hermes-config.yaml.bak"
  echo "proxy mode: 127.0.0.1:$PROXY_PORT → $UPSTREAM  (server-measured tokens for BOTH wrappers)"
  UPSTREAM="$UPSTREAM" TOKEN_LOG="$PROXY_LOG" python3 scripts/token-proxy.py "$PROXY_PORT" &
  PROXY_PID=$!
  # Restore-everything trap: kill proxy, restore hermes config if we changed it.
  cleanup() { kill "$PROXY_PID" 2>/dev/null; [ -f "$HCFG_BAK" ] && cp "$HCFG_BAK" "$HCFG"; }
  trap cleanup EXIT INT TERM
  # Wait for the proxy to bind before any wrapper runs.
  until curl -s -m 3 "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; do sleep 0.3; done
  # Point Zero at the proxy (no config mutation).
  ZERO_URL_ARGS=(--url "http://127.0.0.1:$PROXY_PORT")
  # Point Hermes at the proxy via a backed-up, trap-restored config edit.
  if [ -f "$HCFG" ]; then
    cp "$HCFG" "$HCFG_BAK"
    python3 - "$HCFG" "http://127.0.0.1:$PROXY_PORT/v1" <<'PY'
import re, sys
path, proxy = sys.argv[1], sys.argv[2]
src = open(path).read()
# Rewrite only the first top-level base_url (the active local backend).
out = re.sub(r'(base_url:\s*)\S+', rf'\1{proxy}', src, count=1)
open(path, "w").write(out)
PY
  else
    echo "WARN: $HCFG not found — Hermes won't be proxied (its tokens stay '?')." >&2
  fi
fi

# Sum prompt+completion tokens logged by the proxy from line $1+1 to EOF.
proxy_tokens_since() {
  [ -f "$PROXY_LOG" ] || { echo 0; return; }
  awk -v start="$1" 'NR>start{p+=$2; c+=$3} END{print p+c+0}' "$PROXY_LOG"
}
printf '%-8s %-4s %8s %10s %8s %6s %4s\n' "tool" "it" "ms" "out_bytes" "tokens" "exit" "ok"
printf '%s\n' "------------------------------------------------------------"

# Pull a total-token count out of a tool's combined stdout+stderr. Zero emits the
# canonical `[usage: … total=N]`; Claude Code (--output-format json) carries an
# `usage` object we sum input+output from; Hermes falls back to a "tokens: N"
# shaped line. Echoes the number, or "?" if none.
extract_tokens() {
  local f="$1" tool="${2:-}"
  local n
  # Zero's exact line first.
  n=$(grep -oE '\[usage:[^]]*total=[0-9]+\]' "$f" 2>/dev/null | grep -oE 'total=[0-9]+' | grep -oE '[0-9]+' | tail -1)
  if [ -n "${n:-}" ]; then echo "$n"; return; fi
  # Claude Code JSON: input_tokens + output_tokens (+cache) from the usage object.
  if [ "$tool" = "claude" ]; then
    n=$(python3 - "$f" <<'PY' 2>/dev/null
import sys, json
try:
    d = json.load(open(sys.argv[1]))
    u = d.get("usage", {})
    tot = (u.get("input_tokens", 0) + u.get("output_tokens", 0)
           + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0))
    print(tot if tot else "")
except Exception:
    print("")
PY
)
    [ -n "${n:-}" ] && { echo "$n"; return; }
  fi
  # Generic fallback (Hermes-style).
  n=$(grep -oiE 'total[^0-9]{0,12}([0-9]{2,})[^0-9]*tokens|tokens[^0-9]{0,4}([0-9]{2,})' "$f" 2>/dev/null \
        | grep -oE '[0-9]{2,}' | head -1)
  echo "${n:-?}"
}

# Claude Code's reply text lives in the JSON `result` field (not raw stdout), so
# the correctness check must read THAT. Echoes the result string, or the file.
claude_result_text() {
  python3 - "$1" <<'PY' 2>/dev/null || cat "$1"
import sys, json
try:
    print(json.load(open(sys.argv[1])).get("result", ""))
except Exception:
    sys.exit(1)
PY
}

# accumulators (bash arithmetic; "?" tokens skip the sum)
# Optional correctness check: if ZERO_BENCH_EXPECT is set to a regex naming the
# fact both wrappers should surface (e.g. the known-largest filename), each run's
# stdout is matched against it. This measures the GOAL's *equivalent data* half
# alongside the token half — one run proves both: same answer, fewer tokens.
EXPECT="${ZERO_BENCH_EXPECT:-}"

# Claude Code is an OPTIONAL third column. It is a DIFFERENT MODEL (cloud Claude),
# not the gx10 endpoint — so it is a quality/reference baseline, NOT part of the
# same-model Zero-vs-Hermes token race. It also costs real money per run. Opt in
# with ZERO_BENCH_CLAUDE=1 and `claude` on PATH.
TOOLS=(zero hermes)
if [ "${ZERO_BENCH_CLAUDE:-}" = "1" ] && command -v claude >/dev/null; then
  TOOLS+=(claude)
  echo "claude: ENABLED — DIFFERENT-MODEL reference baseline (cloud, not gx10); costs \$."
elif [ "${ZERO_BENCH_CLAUDE:-}" = "1" ]; then
  echo "claude: requested but not on PATH — skipping." >&2
fi

# Per-tool accumulators. macOS ships bash 3.2 (no associative arrays), so we use
# indirect scalars `SUM_MS_zero` etc. — keeps the script runnable out-of-the-box
# with no Homebrew bash. aget reads, ainc adds (both via ${!var} indirection).
aget() { local v="$1_$2"; echo "${!v:-0}"; }
ainc() { local v="$1_$2"; eval "$v=$(( ${!v:-0} + $3 ))"; }
for t in "${TOOLS[@]}"; do eval "SUM_MS_$t=0 SUM_TOK_$t=0 CNT_TOK_$t=0 OK_HITS_$t=0"; done

bench_one() {
  local tool="$1" it="$2"; shift 2
  local out="$run_dir/${tool}-${it}.out" err="$run_dir/${tool}-${it}.err"
  local start end ms code tok ok logbase
  # In proxy mode, the server-measured token total for this run is the proxy-log
  # delta — authoritative and identical across wrappers. Record the log length
  # before the call so we sum only this run's upstream responses.
  logbase=0
  [ "$PROXY_MODE" = "1" ] && [ -f "$PROXY_LOG" ] && logbase=$(wc -l <"$PROXY_LOG" | tr -d ' ')
  start=$(python3 -c 'import time;print(int(time.time()*1000))')
  "$@" >"$out" 2>"$err"
  code=$?
  end=$(python3 -c 'import time;print(int(time.time()*1000))')
  ms=$((end - start))
  if [ "$PROXY_MODE" = "1" ]; then
    tok=$(proxy_tokens_since "$logbase"); [ "$tok" = "0" ] && tok="?"
  else
    tok=$(extract_tokens "$err" "$tool"); [ "$tok" = "?" ] && tok=$(extract_tokens "$out" "$tool")
  fi
  ok="-"
  if [ -n "$EXPECT" ]; then
    # Claude's reply is the JSON `result` field; everyone else's is raw stdout.
    local hay="$out"
    if [ "$tool" = "claude" ]; then claude_result_text "$out" >"$out.txt"; hay="$out.txt"; fi
    if grep -qiE "$EXPECT" "$hay"; then ok="Y"; ainc OK_HITS "$tool" 1; else ok="N"; fi
  fi
  printf '%-8s %-4s %8s %10s %8s %6s %4s\n' "$tool" "$it" "$ms" "$(wc -c <"$out" | tr -d ' ')" "$tok" "$code" "$ok"
  ainc SUM_MS "$tool" "$ms"
  if [[ "$tok" =~ ^[0-9]+$ ]]; then
    ainc SUM_TOK "$tool" "$tok"; ainc CNT_TOK "$tool" 1
  fi
}

for it in $(seq 1 "$N"); do
  bench_one zero   "$it" "$ZERO_BIN" -p "$PROMPT" --tools ${ZERO_URL_ARGS[@]+"${ZERO_URL_ARGS[@]}"}
  bench_one hermes "$it" hermes -z "$PROMPT" --yolo
  for t in "${TOOLS[@]}"; do
    [ "$t" = "claude" ] && bench_one claude "$it" claude -p "$PROMPT" --output-format json
  done
done

echo
echo "== averages =="
for t in "${TOOLS[@]}"; do
  avg_ms=$(( $(aget SUM_MS "$t") / N ))
  cnt=$(aget CNT_TOK "$t")
  if [ "$cnt" -gt 0 ]; then
    avg_tok=$(( $(aget SUM_TOK "$t") / cnt ))
    printf '%-8s avg_ms=%-8s avg_tokens=%-8s (n=%s with usage)\n' "$t" "$avg_ms" "$avg_tok" "$cnt"
  else
    printf '%-8s avg_ms=%-8s avg_tokens=? (no usage line found)\n' "$t" "$avg_ms"
  fi
done
# Same-model race: Zero vs Hermes both hit gx10, so this ratio is a pure
# wrapper-efficiency number — the /goal.
if [ "$(aget CNT_TOK zero)" -gt 0 ] && [ "$(aget CNT_TOK hermes)" -gt 0 ]; then
  zt=$(( $(aget SUM_TOK zero) / $(aget CNT_TOK zero) ))
  ht=$(( $(aget SUM_TOK hermes) / $(aget CNT_TOK hermes) ))
  if [ "$ht" -gt 0 ]; then
    echo "Zero uses $(( zt * 100 / ht ))% of Hermes's tokens on this task (lower = Zero saves)."
  fi
fi
# Claude Code is a DIFFERENT model (cloud), so its token count is NOT comparable
# to the gx10 numbers — print it as a labeled reference, never as a ratio.
if [[ " ${TOOLS[*]} " == *" claude "* ]] && [ "$(aget CNT_TOK claude)" -gt 0 ]; then
  ct=$(( $(aget SUM_TOK claude) / $(aget CNT_TOK claude) ))
  echo "claude=$ct avg tokens — REFERENCE ONLY (cloud Claude, different model; not a same-model comparison vs gx10)."
fi

# The GOAL verdict: equivalent data (both got the right answer) AND fewer tokens.
if [ -n "$EXPECT" ]; then
  echo
  echo "== equivalent-data check (matched /$EXPECT/) =="
  printf '%-8s %s/%s runs correct\n' "zero" "$(aget OK_HITS zero)" "$N"
  printf '%-8s %s/%s runs correct\n' "hermes" "$(aget OK_HITS hermes)" "$N"
  [[ " ${TOOLS[*]} " == *" claude "* ]] && \
    printf '%-8s %s/%s runs correct (reference, different model)\n' "claude" "$(aget OK_HITS claude)" "$N"
  if [ "$(aget OK_HITS zero)" -eq "$N" ] && [ "$(aget OK_HITS hermes)" -eq "$N" ]; then
    echo "GOAL: both wrappers produced equivalent data on every run."
    [ "$(aget CNT_TOK zero)" -gt 0 ] && [ "$(aget CNT_TOK hermes)" -gt 0 ] && \
      echo "      → same answers, compare the token column above for the efficiency win."
  else
    echo "NOTE: not every run matched — inspect the mismatching outputs in $run_dir"
    echo "      (model nondeterminism, or a real difference between wrappers)."
  fi
fi
echo
echo "Raw outputs + stderr traces under: $run_dir"
echo "Tip: set ZERO_BENCH_EXPECT='<regex of the expected fact>' to check equivalence."
