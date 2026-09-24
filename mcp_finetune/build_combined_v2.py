#!/usr/bin/env python3
"""build_combined_v2.py — fold the soft-labelled rfai-intent-v2 dataset into
the next combined fine-tuning corpus.

Reads:
  datasets/combined-all.jsonl      (current 19,921-row corpus, contains 477
                                    hard-labelled rfai-intent rows from v1)
  datasets/rfai-intent-v2.jsonl    (721 rows: archive one-hot + SOFT GLM
                                    teacher labels + synth-verified)

Writes:
  datasets/combined-all-v2.jsonl   (v1 rfai-intent rows REPLACED by v2 rows;
                                    rfai-feed untouched; everything else
                                    byte-identical)

Run AFTER the current combined-all training finishes — never against the
dataset a live trainer is reading (the trainer loads it once at start, so
writing a NEW file keeps things safe regardless).

    python3 build_combined_v2.py [--stats-only]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "combined-all.jsonl"
V2 = ROOT / "datasets" / "rfai-intent-v2.jsonl"
OUT = ROOT / "datasets" / "combined-all-v2.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()

    if not V2.exists():
        sys.exit(f"missing {V2} — run export_rfai.py --v2 first")

    v2_rows = [json.loads(l) for l in V2.open()]
    for r in v2_rows:
        r["source_domain"] = "rfai-intent-v2"
        r.setdefault("meta", {})["supersedes"] = "rfai-intent"

    kept, replaced, out_rows = 0, 0, []
    domains = collections.Counter()
    for line in SRC.open():
        d = json.loads(line)
        if d.get("source_domain") == "rfai-intent":
            replaced += 1
            continue  # v1 rows dropped; v2 appended below
        kept += 1
        domains[d.get("source_domain", "?")] += 1
        out_rows.append(line.rstrip("\n"))

    for r in v2_rows:
        domains["rfai-intent-v2"] += 1
        out_rows.append(json.dumps(r, ensure_ascii=False))

    print(
        f"kept (non-rfai-intent): {kept} | dropped v1 rfai-intent: {replaced} | added v2: {len(v2_rows)}"
    )
    print("per-domain:", dict(domains))
    if a.stats_only:
        return 0

    with OUT.open("w") as f:
        f.write("\n".join(out_rows) + "\n")
    print(f"wrote {len(out_rows)} rows -> {OUT}")
    print(
        "\nnext: train with --dataset datasets/combined-all-v2.jsonl "
        "(and a FRESH LAYA_PROGRESS_FILE + new --out checkpoint dir)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
