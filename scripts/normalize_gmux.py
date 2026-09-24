#!/usr/bin/env python
"""normalize_gmux.py — repair datasets/gmux-routing.jsonl to the CANONICAL router shape.

Root cause (found via the gmux jev-graded audit, 2026-09-23): 605/606 rows were
Jev-shadowed with a 3-option risk question (readonly/local/destructive) and NO
needs_reasoning/context_size questions, while crates/gmux-router serves a 4-option risk
(readonly/local-writes/destructive/external-side-effects) + needs_reasoning (noul) +
context_size (score). Training on one shape and serving another = guaranteed calibration
failure (the r1 gate: ECE 0.368, crash 109/120).

This script:
  1. rewrites every row's questions to the canonical router criteria (identical strings),
  2. remaps old labels/probability keys: local→local-writes, external→external-side-effects,
  3. backfills the missing needs_reasoning + context_size answers via Jev (one call/row,
     soft labels), keeping existing complexity/risk answers,
  4. writes datasets/gmux-routing.jsonl in place (backup: gmux-routing.jsonl.bak-<ts>).

Usage: laya-venv/bin/python scripts/normalize_gmux.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_shadow_label import jev_call, to_soft_answers  # noqa: E402
from monitor_shadow import _load_key_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "gmux-routing.jsonl"

CANON = {
    "complexity": {
        "type": "choice",
        "instructions": "Task complexity",
        "criteria": {
            "trivial": "one-liner, mechanical",
            "simple": "single-file mechanical change",
            "moderate": "multi-file change with reasoning",
            "complex": "architectural / cross-module redesign",
        },
    },
    "risk": {
        "type": "choice",
        "instructions": "Operation risk",
        "criteria": {
            "readonly": "no writes at all",
            "local-writes": "repo writes only, reversible",
            "destructive": "history rewrites, data loss, irreversible in-place changes",
            "external-side-effects": "touches infra, prod, external services or people",
        },
    },
    "needs_reasoning": {
        "type": "noul",
        "instructions": "The task requires multi-step reasoning, not just a verdict",
    },
    "context_size": {
        "type": "score",
        "instructions": "Context size needed",
        "criteria": ["small", "medium", "large"],
    },
}
REMAP = {"local": "local-writes", "external": "external-side-effects"}
FILL_QUESTIONS = {
    "needs_reasoning": CANON["needs_reasoning"],
    "context_size": CANON["context_size"],
}


def remap_answer(ans):
    if isinstance(ans, str):
        return REMAP.get(ans, ans)
    if isinstance(ans, dict) and "probabilities" in ans:
        return {
            "probabilities": {
                REMAP.get(k, k): v for k, v in ans["probabilities"].items()
            }
        }
    if isinstance(ans, dict):
        out = {}
        for k, v in ans.items():
            kk = REMAP.get(k, k)
            out[kk] = REMAP.get(v, v) if isinstance(v, str) else v
        return out
    return ans


def fill_one(row):
    """Jev-backfill the two missing heads for one row. Returns answers-dict or None."""
    resp = jev_call(row["state"], FILL_QUESTIONS)
    got = to_soft_answers(FILL_QUESTIONS, resp.get("answers", {}))
    return got or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    _load_key_env()

    rows = [json.loads(l) for l in DS.open() if l.strip()]
    backup = DS.with_suffix(f".jsonl.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy(DS, backup)
    print(f"{len(rows)} rows; backup → {backup}")

    # pass 1: canonicalize questions + remap existing answers
    need_fill = []
    for r in rows:
        r["questions"] = {k: dict(CANON[k]) for k in CANON}  # same order as the router
        r["answers"] = {k: remap_answer(v) for k, v in r.get("answers", {}).items()}
        if "needs_reasoning" not in r["answers"] or "context_size" not in r["answers"]:
            need_fill.append(r)
    print(f"rows needing backfill: {len(need_fill)}")

    # pass 2: Jev backfill (threadpool)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fill_one, r): r for r in need_fill}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                got = fut.result()
                if got:
                    r["answers"].update(got)
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  fill failed: {e}", file=sys.stderr)
            if i % 100 == 0:
                print(f"  backfill {i}/{len(need_fill)} (ok={done} fail={failed})")

    # pass 3: rows we couldn't fill lose the unfilled questions (stay self-consistent)
    dropped = 0
    for r in rows:
        for q in list(r["questions"]):
            if q not in r["answers"]:
                del r["questions"][q]
                dropped += 1
    with DS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"DONE: {len(rows)} rows canonical (backfilled {done}, failed {failed}, dropped-q {dropped})"
    )


if __name__ == "__main__":
    main()
