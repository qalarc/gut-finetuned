#!/usr/bin/env python
"""geo_live_collect.py — collect REAL GEO-audit states graded by Jev (protocol §1).

Runs qalarc.ai's geo_pipeline.check_site() live against every property, renders each
result into the exact geo-audit-triage state format used by the synthetic generator,
and asks Jev for priority+fixnow. Source="jev-live" rows appended to the dataset.

This gives the fine-tune REAL site states (not templates) — the audit triage model
learns from actual qalarc GEO findings. Run after every audit or weekly.

Usage: laya-venv/bin/python scripts/geo_live_collect.py [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEO_PIPELINE = Path.home() / "projects/qalarc.ai/scripts/geo_pipeline.py"
DEFAULT_OUT = ROOT / "datasets" / "geo-audit-triage.jsonl"
API = "https://api.typesafe.ai/v1/systemone"

Q_PRIORITY = {
    "type": "choice",
    "instructions": "Fix priority for this GEO audit finding",
    "criteria": {
        "p0": "blocks indexing or domain dead - fix infrastructure before any SEO",
        "p1": "high value - costs citations or visibility right now",
        "p2": "worthwhile but not urgent",
        "skip": "cosmetic or no measurable effect",
    },
}
Q_FIXNOW = {"type": "noul", "instructions": "Fix in the current deploy window"}


def load_pipeline():
    spec = importlib.util.spec_from_file_location("geo_pipeline", GEO_PIPELINE)
    assert spec is not None and spec.loader is not None, "cannot load geo_pipeline"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa: PLC0415 - deliberate dynamic import
    return mod


def render_state(domain: str, weight: int, r: dict) -> str:
    """Same shape as gen_geo_audit() in jev_shadow_label.py, from LIVE check_site data."""
    if r.get("issues"):
        iss = "; ".join(r["issues"])
    else:
        iss = (
            f"all checks clean - llms.txt {r.get('llms', 200)}, "
            f"{r.get('ai_bots_explicit', 0)} AI bots declared, "
            f"JSON-LD valid ({r.get('ld_ok', 0)} blocks), title '{r.get('title', '')[:40]}'"
        )
    return f"GEO audit {domain} (weight {weight}): {iss}."


def jev(state: str, key: str) -> dict | None:
    body = json.dumps(
        {
            "model": "jev-1.13.0",
            "state": state,
            "questions": {"priority": Q_PRIORITY, "fixnow": Q_FIXNOW},
        }
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001 - retry then give up
            if attempt == 2:
                return None
            time.sleep(2**attempt)


def _key() -> str:
    k = os.environ.get("TYPESAFE_API_KEY")
    if k:
        return k
    for line in (Path.home() / ".secrets/typesafe.env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() == "TYPESAFE_API_KEY":
                return v.strip().strip('"')
    sys.exit("TYPESAFE_API_KEY missing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    key = _key()
    gp = load_pipeline()

    out = Path(a.out)
    rows = 0
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    with concurrent_pool() as ex:  # sequential is fine: 9 sites, keep simple
        for domain, weight in gp.SITES:
            try:
                r = gp.check_site(domain, weight)
            except Exception as e:  # noqa: BLE001
                r = {
                    "domain": domain,
                    "weight": weight,
                    "issues": [f"audit error: {e}"],
                }
            state = render_state(domain, weight, r)
            ans = jev(state, key)
            if not ans or "answers" not in ans:
                print(f"[skip] {domain}: jev failed", file=sys.stderr)
                continue
            row = {
                "state": state,
                "questions": {"priority": Q_PRIORITY, "fixnow": Q_FIXNOW},
                "answers": ans["answers"],
                "source": "jev-live",
                "ts": ts,
            }
            with out.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            prio = ans["answers"].get("priority")
            print(
                f"[{rows}] {domain}: priority={prio} fixnow={ans['answers'].get('fixnow')}"
            )
    print(f"DONE: +{rows} live rows -> {out}")


def concurrent_pool():
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    return ThreadPoolExecutor(max_workers=1)


if __name__ == "__main__":
    main()
