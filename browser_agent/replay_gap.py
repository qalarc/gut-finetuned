#!/usr/bin/env python
"""replay_gap.py — quantify THE GAP between the cloud teacher (Jev) and our local
fine-tuned Laya, decision by decision, on real captured browser flows.

For every row in datasets/browser-ops.jsonl (each = a decision Jev actually made while
driving a browser: state, question heads, Jev's soft answer), replay the SAME state +
questions against the local endpoint (serve.py :8798) and compare:

  - argmax agreement per question head (did local pick the same operation/target?)
  - probability of the teacher's choice under the local model (how much mass it gives)
  - Jev's own confidence in that choice (the target the local model must reach)

Usage: laya-venv/bin/python browser_agent/replay_gap.py [--url http://127.0.0.1:8798] [--source-filter trace]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "browser-ops.jsonl"


def call_local(url, state, questions, domain="browser-ops"):
    body = {"state": state, "questions": questions}
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Laya-Domain": domain},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def soft(answer) -> dict:
    """Normalize a stored teacher answer to {label: p}."""
    if isinstance(answer, dict) and "probabilities" in answer:
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    if isinstance(answer, dict):
        for k in ("choice", "noul", "score"):
            if k in answer:
                return {str(answer[k]): 1.0}
    if isinstance(answer, str):
        return {answer: 1.0}
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        v = float(answer)
        return {"true": v, "false": 1.0 - v}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8798")
    ap.add_argument(
        "--source-filter", default=None, help="only rows whose source contains this"
    )
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in DATASET.open() if l.strip()]
    if a.source_filter:
        rows = [r for r in rows if a.source_filter in r.get("source", "")]
    if a.limit:
        rows = rows[: a.limit]
    if not rows:
        sys.exit("no rows matched")

    per_head = defaultdict(lambda: {"n": 0, "agree": 0, "mass": [], "teacher_conf": []})
    flow_results = []
    for i, r in enumerate(rows):
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
        answers = r.get("answers") or {}
        for head, teacher_ans in answers.items():
            q = (r.get("questions") or {}).get(head)
            if not q:
                continue
            t = soft(teacher_ans)
            if not t:
                continue
            t_label = max(t, key=t.get)
            try:
                resp = call_local(a.url, state, {head: q})
            except Exception as e:
                print(f"row {i} head {head}: local call failed: {e}", file=sys.stderr)
                continue
            l_ans = (resp.get("answers") or {}).get(head) or {}
            l = soft(l_ans)
            l_label = max(l, key=l.get) if l else None
            h = per_head[head]
            h["n"] += 1
            h["agree"] += int(l_label == t_label)
            h["mass"].append(float(l.get(t_label, 0.0)))
            h["teacher_conf"].append(t[t_label])
        flow_results.append((i, r.get("source", "?")))

    print(f"replayed {len(rows)} decisions against {a.url}\n")
    print(
        f"{'head':<16} {'n':>4} {'agree':>7} {'mass@teacher':>13} {'teacher_conf':>13}"
    )
    total_n = total_agree = 0
    for head, h in sorted(per_head.items()):
        mass = statistics.mean(h["mass"]) if h["mass"] else 0
        tc = statistics.mean(h["teacher_conf"]) if h["teacher_conf"] else 0
        print(
            f"{head:<16} {h['n']:>4} {h['agree'] / h['n']:>6.1%} {mass:>13.3f} {tc:>13.3f}"
        )
        total_n += h["n"]
        total_agree += h["agree"]
    print(
        f"\nOVERALL: {total_agree}/{total_n} = {total_agree / max(1, total_n):.1%} argmax agreement"
    )

    src = Counter(s for _, s in flow_results)
    print("\nby source:", dict(src))


if __name__ == "__main__":
    main()
