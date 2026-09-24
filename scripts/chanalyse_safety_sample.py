#!/usr/bin/env python
"""chanalyse_safety_sample.py — extend the chanalyse domain with a SAFETY head.

Samples real threads from datasets/chanalyse-topics.jsonl (16k, GLM topic labels) and
asks Jev for soft labels on TWO heads: topic (re-check) + safety (safe/edgy/offensive).
Output: datasets/chanalyse-safety.jsonl — same state text, both heads, soft answers.
These rows fine-tune the chanalyse model for moderation + topic grading.

Usage: laya-venv/bin/python scripts/chanalyse_safety_sample.py [--n 300]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_shadow_label import jev_call, to_soft_answers  # noqa: E402
from monitor_shadow import _load_key_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "chanalyse-topics.jsonl"
OUT = ROOT / "datasets" / "chanalyse-safety.jsonl"

QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "Topic classification for this thread. Choose the closest topic.",
        "criteria": {
            "bitcoin": "Bitcoin / BTC",
            "ethereum": "Ethereum / ETH",
            "altcoins": "Altcoins (general)",
            "defi": "DeFi / decentralized finance",
            "nfts": "NFTs",
            "mining": "Mining / hardware",
            "markets": "Markets / trading discussion",
            "politics": "Politics / culture war",
            "other": "Other / off-topic",
        },
    },
    "safety": {
        "type": "choice",
        "instructions": "Content safety rating for a general-audience feed.",
        "criteria": {
            "safe": "fine for a general feed",
            "edgy": "crude or adult-flavoured language but tolerable with a filter",
            "offensive": "slurs, threats, illegal content or targeted harassment",
        },
    },
}


def label_one(args):
    idx, row = args
    state = row["state"]
    try:
        resp = jev_call(state, QUESTIONS)
    except Exception:
        return None
    answers = to_soft_answers(QUESTIONS, resp.get("answers", {}))
    if "topic" not in answers or "safety" not in answers:
        return None
    return {
        "state": state,
        "questions": QUESTIONS,
        "answers": answers,
        "source": "chanalyse-jev-safety",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"src_row": idx, "glm_topic": row.get("answers", {}).get("topic")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    _load_key_env()

    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    rng = random.Random(7)
    sample = [(i, r) for i, r in enumerate(rows)]
    rng.shuffle(sample)
    sample = sample[: a.n]

    done = 0
    with OUT.open("a") as f:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed({ex.submit(label_one, s): s for s in sample}):
                row = fut.result()
                if row:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done += 1
                    if done % 50 == 0:
                        print(f"{done}/{a.n}", flush=True)
    print(f"DONE: {done} rows -> {OUT}")


if __name__ == "__main__":
    main()
