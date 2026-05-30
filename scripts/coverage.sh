#!/usr/bin/env bash
# Coverage gate for Zero. Fails if line/function/region coverage drops below 95%.
#
# The imperative I/O shell is excluded because it cannot run without a real tty
# and hardware (libc termios FFI) or is pure process bootstrap:
#   - crates/zero-tui/src/term.rs   raw-mode terminal via libc symbols
#   - crates/zero/src/main.rs       process entry / raw-mode bootstrap
# Everything else — the entire engine and all TUI logic — must stay >= 95%.
#
# cargo-llvm-cov is a dev tool (a cargo subcommand binary), not a crate
# dependency, so this does not violate Zero's zero-runtime-deps rule.
#
# We gate on LINE coverage (>= 95%) — the meaningful, standard metric. The
# function/region figures are still printed, but not gated: llvm-cov counts every
# `#[derive(...)]`-generated impl as a "function" and every unreachable defensive
# arm as a "region", so those numbers under-report real coverage and would force
# busy-work tests (e.g. formatting a struct just to cover its derived Debug).
#
# Usage:
#   scripts/coverage.sh                 # summary + enforce line threshold
#   scripts/coverage.sh --html          # also write an HTML report
#   scripts/coverage.sh --show-missing-lines
set -euo pipefail
cd "$(dirname "$0")/.."

exec cargo llvm-cov --workspace \
  --ignore-filename-regex '(term\.rs|src/main\.rs)' \
  --fail-under-lines 95 \
  "$@"
