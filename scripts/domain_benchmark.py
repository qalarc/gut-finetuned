#!/usr/bin/env python
"""domain_benchmark.py — prove it with real figures: replay each domain's captured
decisions against a Laya endpoint (local serve.py) and measure agreement with the
teacher labels we trained on.

For every benchmarked domain this script:
  1. samples rows from datasets/<domain>.jsonl (seeded shuffle — reproducible),
  2. sends the SAME state + questions to the endpoint (X-Laya-Domain: <domain>),
  3. compares argmax(label distribution) against the teacher's stored answer,
  4. writes benchmarks/<date>-<domain>.json + prints the table.

These are OUR tests on OUR data: run before fine-tuning (baseline = the gap), and after
every registered checkpoint (the improvement). Figures quoted anywhere must come from
here or from checkpoints/*/eval.json — nowhere else.

Usage:
  laya-venv/bin/python scripts/domain_benchmark.py --all --n 100
  laya-venv/bin/python scripts/domain_benchmark.py --domain browser-ops --n 150
  laya-venv/bin/python scripts/domain_benchmark.py --domain chanalyse-topics --n 200 --endpoint http://127.0.0.1:8798
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

DOMAINS = {
    # domain: (dataset, heads to test, sample cap)
    "osint-grading": ("osint-grading.jsonl", ["tier"], 150),
    "seo-grading": ("seo-grading.jsonl", None, 150),  # all heads
    "browser-ops": ("browser-ops.jsonl", ["operation", "visit_target"], 250),
    "chanalyse-topics": ("chanalyse-topics.jsonl", ["topic"], 200),
    "monitor-triage": ("monitor-triage.jsonl", None, 150),
    "compliance-marks": ("compliance-marks.jsonl", None, 120),
    "gmux-routing": ("gmux-routing.jsonl", None, 150),
}


def soft(answer) -> dict:
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


def call(endpoint: str, domain: str, state, questions):
    body = {"state": state, "questions": questions}
    headers = {"Content-Type": "application/json", "X-Laya-Domain": domain}
    if "typesafe.ai" in endpoint:
        body["model"] = "jev-1.13.0"
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key:
            env = Path.home() / ".secrets/typesafe.env"
            if env.is_file():
                for line in env.read_text().splitlines():
                    if line.startswith("TYPESAFE_API_KEY="):
                        key = line.split("=", 1)[1].strip()
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run_domain(endpoint, domain, ds_file, heads, n, seed=42):
    rows = [json.loads(l) for l in (ROOT / "datasets" / ds_file).open() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    per_head = defaultdict(lambda: {"n": 0, "agree": 0, "mass": [], "lat": []})
    lat_all = []
    errors = 0
    for r in rows:
        state = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
        answers = r.get("answers") or {}
        test_heads = heads or list((r.get("questions") or {}).keys())
        for head in test_heads:
            q = (r.get("questions") or {}).get(head)
            t = soft(answers.get(head))
            if not q or not t:
                continue
            try:
                t0 = time.time()
                resp = call(endpoint, domain, state, {head: q})
                lat_all.append((time.time() - t0) * 1000)
            except Exception:
                errors += 1
                continue
            got = soft((resp.get("answers") or {}).get(head) or {})
            if not got:
                continue
            t_label = max(t, key=t.get)
            l_label = max(got, key=got.get)
            h = per_head[head]
            h["n"] += 1
            h["agree"] += int(l_label == t_label)
            h["mass"].append(float(got.get(t_label, 0.0)))
    out = {
        "domain": domain,
        "endpoint": endpoint,
        "n_rows": len(rows),
        "errors": errors,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "heads": {},
    }
    print(f"\n== {domain} ({len(rows)} rows vs {endpoint}) ==")
    print(f"{'head':<16}{'n':>6}{'agreement':>11}{'mass@teacher':>14}{'avg ms':>9}")
    for head, h in sorted(per_head.items()):
        agree = h["agree"] / h["n"] if h["n"] else 0
        mass = statistics.mean(h["mass"]) if h["mass"] else 0
        lat = statistics.mean([l for l in lat_all]) if lat_all else 0
        print(f"{head:<16}{h['n']:>6}{agree:>10.1%} {mass:>13.3f} {lat:>8.0f}")
        out["heads"][head] = {
            "n": h["n"],
            "agreement": round(agree, 4),
            "mass_on_teacher": round(mass, 4),
            "avg_ms": round(lat, 1),
        }
    overall_n = sum(h["n"] for h in per_head.values())
    overall_a = sum(h["agree"] for h in per_head.values())
    out["overall_agreement"] = round(overall_a / overall_n, 4) if overall_n else 0.0
    out["overall_n"] = overall_n
    print(f"  OVERALL: {out['overall_agreement']:.1%} over {overall_n} decisions")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--endpoint", default=os.environ.get("LAYA_URL", "http://127.0.0.1:8798")
    )
    ap.add_argument("--domain", choices=sorted(DOMAINS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    env = Path.home() / ".secrets/typesafe.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

    targets = list(DOMAINS) if a.all else ([a.domain] if a.domain else [])
    if not targets:
        sys.exit("choose --domain <name> or --all")
    BENCH.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d")
    results = []
    for domain in targets:
        ds_file, heads, cap = DOMAINS[domain]
        ds_path = ROOT / "datasets" / ds_file
        if not ds_path.is_file():
            print(f"skip {domain}: no dataset")
            continue
        res = run_domain(a.endpoint, domain, ds_file, heads, min(a.n, cap))
        results.append(res)
        (BENCH / f"{stamp}-{domain}.json").write_text(json.dumps(res, indent=2))
    print(f"\nsaved {len(results)} benchmark files -> benchmarks/{stamp}-*.json")


if __name__ == "__main__":
    main()
