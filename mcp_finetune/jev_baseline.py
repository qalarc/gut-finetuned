#!/usr/bin/env python
"""jev_baseline.py — run Jev (TypeSafe cloud decision engine) over the RFAI
intent held-out split, for comparison with Laya zero-shot / Laya fine-tuned /
GLM labels.

Uses the SAME row-level split as train.py (random.Random(13) shuffle, first
10% held out) on datasets/rfai-intent.jsonl, so results are directly
comparable with the fine-tune gate eval and the handover-curve measurements.

Per row asks two questions in one call (the Jev batching pattern):
  intent — choice over the 9 canonical classes
  urgent — noul true/false (interrupt-worthiness)

Writes results to datasets/jev-baseline-results.jsonl + prints the analysis:
agreement vs teacher labels, per-class breakdown, confidence-vs-correctness,
handover curve (offload % and precision at each threshold), latency stats.

Usage:
  python3 jev_baseline.py [--dataset ../datasets/rfai-intent.jsonl] [--limit 47]
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "https://api.typesafe.ai/v1/systemone"

CRITERIA = {
    "emergency": "mayday, pan pan, distress, fire, accident, injury, man down",
    "convoy": "trucking coordination, ETAs, road trains, b-doubles",
    "navigation": "position reports, headings, waypoints, ETAs, ATC-style reports",
    "check-in": "announcing presence, radio checks, calling CQ",
    "technical": "signal quality, QRM, repeater issues, equipment",
    "weather": "conditions, storms, forecasts, hazards",
    "social": "casual chat, ragchew",
    "data": "digital bursts, tones, CTCSS, DTMF",
    "unknown": "unintelligible, no content, noise, broadcast/music",
}

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Classify the intent of this radio transmission.",
        "criteria": CRITERIA,
    },
    "urgent": {
        "type": "noul",
        "instructions": "True if an operator monitoring this channel should be "
        "interrupted for this transmission (distress, safety, "
        "convoy-critical); false otherwise.",
    },
}


def api_key() -> str:
    k = os.environ.get("TYPESAFE_API_KEY", "")
    if not k:
        f = Path.home() / ".secrets" / "typesafe.env"
        for line in f.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                k = line.split("=", 1)[1].strip()
    return k


def ask_jev(state: str) -> dict:
    body = json.dumps(
        {"state": state, "model": "jev-latest", "questions": QUESTIONS}
    ).encode()
    req = urllib.request.Request(ENDPOINT, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {api_key()}")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read())
    out["_latency_s"] = round(time.time() - t0, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "rfai-intent.jsonl"))
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap rows (default: full held-out split)",
    )
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.dataset)]
    rng = random.Random(13)
    rng.shuffle(rows)
    held = rows[: max(1, int(len(rows) * 0.1))]
    if a.limit:
        held = held[: a.limit]

    print(f"running Jev over {len(held)} held-out rows ...")
    results = []
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(ask_jev, r["state"]): r for r in held}
        for i, fut in enumerate(cf.as_completed(futs)):
            r = futs[fut]
            try:
                out = fut.result()
                ans = out.get("answers", {})
                results.append(
                    {
                        "state": r["state"][:160],
                        "want": r["answers"]["intent"],
                        "want_urgent": r["answers"]["urgent"],
                        "got": ans.get("intent", {}).get("choice"),
                        "conf": ans.get("intent", {}).get("confidence"),
                        "probs": ans.get("intent", {}).get("probabilities"),
                        "got_urgent": ans.get("urgent", {}).get("noul"),
                        "urgent_conf": ans.get("urgent", {}).get("confidence"),
                        "latency": out.get("_latency_s"),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "state": r["state"][:160],
                        "want": r["answers"]["intent"],
                        "error": str(e)[:120],
                    }
                )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(held)}", flush=True)

    outp = ROOT / "datasets" / "jev-baseline-results.jsonl"
    with outp.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r.get("got")]
    err = [r for r in results if r.get("error")]
    lat = sorted(r["latency"] for r in ok if r.get("latency"))
    print(f"\nresults: {len(ok)} ok, {len(err)} errors -> {outp}")
    if lat:
        print(
            f"latency: p50={lat[len(lat) // 2]:.2f}s p90={lat[int(len(lat) * 0.9)]:.2f}s"
        )

    agree = sum(1 for r in ok if r["got"] == r["want"])
    print(
        f"\nAGREEMENT with teacher (GLM/keyword labels): {agree}/{len(ok)} = {agree / len(ok):.1%}"
    )

    # confidence vs correctness
    bins = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        b = min(int((r.get("conf") or 0) * 5), 4)
        bins[b][0] += 1
        bins[b][1] += r["got"] == r["want"]
    print("confidence calibration (bucket: correct/total):")
    for b in sorted(bins):
        n, c = bins[b]
        print(f"  {b / 5:.1f}-{(b + 1) / 5:.1f}: {c}/{n}")

    # handover curve
    print("\nhandover curve (Jev handles >= threshold, rest escalate to GLM):")
    print(f"{'thresh':>6} | {'jev handles':>11} | {'errors kept':>11}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        handled = [r for r in ok if (r.get("conf") or 0) >= t]
        errk = sum(1 for r in handled if r["got"] != r["want"])
        pct = 100 * len(handled) / len(ok) if ok else 0
        print(f"{t:>6.2f} | {pct:>10.0f}% | {errk:>3}/{len(handled)}")

    # urgency correlation
    if ok and all("want_urgent" in r and r.get("got_urgent") is not None for r in ok):
        try:
            pairs = [(float(r["want_urgent"]), float(r["got_urgent"])) for r in ok]
            ma = sum(p[0] for p in pairs) / len(pairs)
            mb = sum(p[1] for p in pairs) / len(pairs)
            num = sum((p[0] - ma) * (p[1] - mb) for p in pairs)
            den = (
                sum((p[0] - ma) ** 2 for p in pairs)
                * sum((p[1] - mb) ** 2 for p in pairs)
            ) ** 0.5
            print(
                f"\nurgency correlation (teacher vs Jev): r={num / den if den else 0:.2f}"
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
