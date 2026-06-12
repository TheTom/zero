#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Zero Contributors

"""Blind, position-debiased LLM judge for comparing two deliverables on quality.

The point: token efficiency is meaningless without a quality comparison. This
scores two artifacts (e.g. Zero's vs Hermes's output for the same task) against a
rubric, using the SAME model — so "fewer tokens" can only be claimed a win when
quality holds. Measured, repeatable, not eyeballed-once.

Debiasing (LLM judges have a strong position/verbosity bias):
  * Run each pairing BOTH ways (A=left,B=right then A=right,B=left) and average,
    so a constant "prefer the first one" bias cancels out.
  * The judge never sees which tool produced which artifact (labelled X / Y).
  * Rubric forces per-criterion scores, not a vibe, and demands the spec be the
    yardstick (does it do what was ASKED), not raw length.

Usage:
  judge.py --prompt-file P --a fileA --b fileB [--a-name zero --b-name hermes]
  judge.py --prompt "..."   --a fileA --b fileB
Env:
  JUDGE_URL   (default http://192.168.50.125:8000)   JUDGE_MODEL (optional)
Output: JSON to stdout — per-criterion scores for each side (averaged over both
orderings), totals, and a verdict. Exit 0 always (it's a measurement, not a gate).
"""

import argparse
import json
import os
import sys
import urllib.request

URL = os.environ.get("JUDGE_URL", "http://192.168.50.125:8000").rstrip("/")
MODEL = os.environ.get("JUDGE_MODEL", "local")

RUBRIC = [
    ("correctness", "Does it actually work and do what was asked? Bugs, crashes, dead code count against."),
    ("spec_fit", "Does it satisfy the SPECIFIC requirements in the prompt (every stated constraint)?"),
    ("completeness", "Are the expected features present, or is it a thin/partial take?"),
    ("robustness", "Input validation, edge cases, error handling."),
    ("clarity", "Readable, well-structured, maintainable — independent of length."),
]

SYS = (
    "You are a strict, fair code reviewer. Score two candidate solutions, X and Y, "
    "for the SAME task. Judge ONLY quality against the task's requirements — never "
    "reward length or verbosity for its own right. A shorter solution that fully "
    "meets the spec beats a longer one that doesn't. Return ONLY JSON."
)


def chat(messages, max_tokens=1200):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


def extract_json(text):
    """Pull the first {...} JSON object out of a model reply."""
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON in judge reply")
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[i:j + 1])
    raise ValueError("unbalanced JSON in judge reply")


def judge_single(prompt, artifact):
    """Score ONE deliverable 0-10 per criterion against the task. For ablations
    where each arm is scored absolutely (temp=0 → deterministic). Returns {crit:score}."""
    rubric_lines = "\n".join(f"  - {k}: {desc}" for k, desc in RUBRIC)
    user = f"""TASK:
{prompt}

--- CANDIDATE ---
{artifact}

Score the candidate 0-10 on every criterion (be strict; the spec is the yardstick,
not length):
{rubric_lines}

Return ONLY this JSON (integers 0-10):
{{{', '.join(f'"{k}": 0' for k, _ in RUBRIC)}, "note": "one sentence"}}"""
    reply = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}])
    return extract_json(reply)


def judge_once(prompt, left, right):
    """One scoring pass: X=left, Y=right. Returns {'X':{crit:score}, 'Y':{...}}."""
    rubric_lines = "\n".join(f"  - {k}: {desc}" for k, desc in RUBRIC)
    keys = ", ".join(k for k, _ in RUBRIC)
    user = f"""TASK GIVEN TO BOTH:
{prompt}

--- CANDIDATE X ---
{left}

--- CANDIDATE Y ---
{right}

Score each candidate 0-10 on every criterion:
{rubric_lines}

Return ONLY this JSON (integers 0-10):
{{"X": {{{', '.join(f'"{k}": 0' for k, _ in RUBRIC)}}},
  "Y": {{{', '.join(f'"{k}": 0' for k, _ in RUBRIC)}}},
  "note": "one sentence on the key difference"}}
Criteria keys: {keys}."""
    reply = chat([{"role": "system", "content": SYS}, {"role": "user", "content": user}])
    return extract_json(reply)


def avg_scores(a_first, b_first, a_label, b_label):
    """Combine the two orderings into per-side averaged criterion scores.

    Pass 1: X=a_label, Y=b_label.  Pass 2: X=b_label, Y=a_label (positions swapped).
    So a_label's scores = average(pass1['X'], pass2['Y']).
    """
    out = {a_label: {}, b_label: {}}
    for k, _ in RUBRIC:
        out[a_label][k] = round((a_first["X"].get(k, 0) + b_first["Y"].get(k, 0)) / 2, 1)
        out[b_label][k] = round((a_first["Y"].get(k, 0) + b_first["X"].get(k, 0)) / 2, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--a", required=True, help="path to deliverable A (or the only one with --single)")
    ap.add_argument("--b", help="path to deliverable B (omit with --single)")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--single", action="store_true", help="score just --a absolutely (ablation mode)")
    args = ap.parse_args()

    prompt = args.prompt or open(args.prompt_file).read()
    a = open(args.a).read()

    # Single-artifact absolute scoring (for ablations).
    if args.single:
        crit = judge_single(prompt, a)
        total = round(sum(v for k, v in crit.items() if k != "note"), 1)
        print(json.dumps({
            "scores": {k: v for k, v in crit.items() if k != "note"},
            "total": total,
            "max_total": len(RUBRIC) * 10,
            "note": crit.get("note", ""),
        }, indent=2))
        return

    b = open(args.b).read()

    # Two passes, positions swapped, to cancel the judge's position bias.
    pass1 = judge_once(prompt, a, b)          # X=a, Y=b
    pass2 = judge_once(prompt, b, a)          # X=b, Y=a
    scores = avg_scores(pass1, pass2, args.a_name, args.b_name)

    totals = {name: round(sum(crit.values()), 1) for name, crit in scores.items()}
    maxtot = len(RUBRIC) * 10
    winner = max(totals, key=totals.get)
    margin = abs(totals[args.a_name] - totals[args.b_name])
    verdict = "tie" if margin < 2 else winner

    result = {
        "scores": scores,
        "totals": totals,
        "max_total": maxtot,
        "winner": verdict,
        "margin": round(margin, 1),
        "notes": {"a_first": pass1.get("note", ""), "b_first": pass2.get("note", "")},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
