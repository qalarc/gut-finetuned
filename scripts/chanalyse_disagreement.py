#!/usr/bin/env python
"""chanalyse_disagreement.py — three-way investigation of the chanalyse topic split.

The chanalyse-topics r1 (16k GLM-labeled threads) came back with agreement=0.42 vs its
own teacher but the BEST calibration of any run (ECE 0.078). Hypotheses:
  H1: the model systematically mispredicts specific topics (confusion),
  H2: GLM's labels and Jev genuinely disagree on messy threads (teacher disagreement),
  H3: the model found its own (wrong) mode.

Method: reproduce the exact held-out split (train.py, seed 13), sample N threads, get
  - the fine-tuned model's topic (checkpoints/kaggle/chanalyse-topics-ckpt),
  - Jev zero-shot's topic (cloud, soft),
  - GLM's stored label,
then emit: 3-way agreement rates, per-topic confusion (GLM→model), and the most
interesting class: cases where MODEL+JEV agree against GLM (likely GLM label errors).

Usage: laya-venv/bin/python scripts/chanalyse_disagreement.py [--n 150]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_shadow_label import jev_call, to_soft_answers  # noqa: E402
from monitor_shadow import _load_key_env  # noqa: E402

SRC = ROOT / "datasets" / "chanalyse-topics.jsonl"
CKPT = ROOT / "checkpoints" / "kaggle" / "chanalyse-topics-ckpt"
OUT = ROOT / "benchmarks" / f"chanalyse-disagreement-{time.strftime('%Y-%m-%d')}.json"


def heldout_rows():
    """Reproduce train.py's exact 90/10 row split (seed 13)."""
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    rng = random.Random(13)
    rng.shuffle(rows)
    return rows[: max(1, int(len(rows) * 0.1))]


def model_topic(state, question):
    from laya import Agent

    global _agent
    try:
        _agent
    except NameError:
        _agent = Agent(str(CKPT))
    res = _agent.predict(state, {"topic": question})
    a = res.get("answers", {}).get("topic", {})
    probs = a.get("probabilities") or {}
    return (max(probs, key=probs.get) if probs else a.get("choice")), a


def jev_topic(state, question):
    resp = jev_call(state, {"topic": question})
    soft = to_soft_answers({"topic": question}, resp.get("answers", {})).get("topic")
    if isinstance(soft, dict) and "probabilities" in soft:
        p = soft["probabilities"]
        return max(p, key=p.get), p
    if isinstance(soft, str):
        return soft, {soft: 1.0}
    return None, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--skip-model", action="store_true")
    a = ap.parse_args()
    _load_key_env()

    held = heldout_rows()
    rng = random.Random(99)
    rng.shuffle(held)
    sample = held[: a.n]
    print(f"held-out={len(held)}, sampling {len(sample)}")

    records = []
    for i, r in enumerate(sample):
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
        q = r["questions"]["topic"]
        glm = r["answers"].get("topic")
        if isinstance(glm, dict):
            glm = glm.get("choice") or max(
                (glm.get("probabilities") or {k: 0}),
                key=lambda k: glm["probabilities"][k] if "probabilities" in glm else 0,
            )
        mt = None if a.skip_model else None
        try:
            mt, mt_probs = (None, {}) if a.skip_model else model_topic(state, q)
        except Exception as e:
            print(f"{i}: model failed {e}", file=sys.stderr)
        try:
            jt, jt_probs = jev_topic(state, q)
        except Exception as e:
            print(f"{i}: jev failed {e}", file=sys.stderr)
            continue
        records.append(
            {
                "i": i,
                "glm": glm,
                "model": mt,
                "jev": jt,
                "model_probs": mt_probs if isinstance(mt_probs, dict) else {},
                "jev_probs": jt_probs,
                "state_excerpt": state[:180],
            }
        )
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sample)}", flush=True)

    # ---- analysis
    n = len(records)
    mm = sum(1 for r in records if r["model"] == r["glm"])
    jm = sum(1 for r in records if r["jev"] == r["glm"])
    mj = sum(1 for r in records if r["model"] == r["jev"])
    both_against_glm = [
        r
        for r in records
        if r["model"] == r["jev"] and r["glm"] not in (None, r["model"])
    ]
    glm_right_cases = [
        r for r in records if r["model"] == r["jev"] and r["glm"] == r["model"]
    ]

    confusion = defaultdict(Counter)  # glm -> model predictions
    for r in records:
        confusion[r["glm"]][r["model"]] += 1
    jev_glm_diff = Counter()
    for r in records:
        if r["jev"] != r["glm"]:
            jev_glm_diff[(r["glm"], r["jev"])] += 1

    summary = {
        "n": n,
        "model_vs_glm": round(mm / n, 4),
        "jev_vs_glm": round(jm / n, 4),
        "model_vs_jev": round(mj / n, 4),
        "model_jev_agree_against_glm": len(both_against_glm),
        "all_three_agree": len(glm_right_cases),
        "top_glm_to_model_confusions": {
            f"{g} -> {m}": c
            for (g, m), c in Counter(
                (r["glm"], r["model"]) for r in records if r["model"] != r["glm"]
            ).most_common(12)
        },
        "top_glm_to_jev_disagreements": {
            f"{g} -> {j}": c for (g, j), c in jev_glm_diff.most_common(12)
        },
        "suspected_glm_label_errors": [
            {"glm": r["glm"], "both_said": r["model"], "excerpt": r["state_excerpt"]}
            for r in both_against_glm[:15]
        ],
    }
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=2)[:2200])
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
