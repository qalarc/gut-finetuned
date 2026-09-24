#!/usr/bin/env python
"""gmux_collect.py — ingest gmux-router DecisionLog JSONL into datasets/gmux-routing.jsonl.

The router (crates/gmux-router in gmux_v7) logs every RoutingDecision as a JSONL line:
  {ts, source, state:{task,files_touched,context}, questions:{...}, answers:{q:[label,conf]},
   tier, confidence, reason, elapsed_ms}

This converts each line to the protocol format (notes/FINETUNING_PROTOCOL.md §0):
  {"state": json-string, "questions": {...}, "answers": {q: label-or-float}, "source": "gmux-router:<backend>", "ts": ...}
Dedupes against rows already present.

Usage: laya-venv/bin/python scripts/gmux_collect.py [--log ~/.local/share/gmux/routing_decisions.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = Path.home() / ".local/share/gmux/routing_decisions.jsonl"


def to_protocol_row(line: dict) -> dict | None:
    qs = line.get("questions") or {}
    if not qs:
        return None
    answers = {}
    for qname, cell in (line.get("answers") or {}).items():
        # router logs answers as [label, confidence] arrays
        if isinstance(cell, (list, tuple)) and cell:
            v = cell[0]
        elif isinstance(cell, dict):
            v = cell.get("choice", cell.get("noul", cell.get("score")))
        else:
            v = cell
        if v is None:
            continue
        qtype = (qs.get(qname) or {}).get("type")
        if qtype == "noul":
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
        answers[qname] = v
    if not answers:
        return None
    return {
        "state": json.dumps(line["state"], ensure_ascii=False)
        if isinstance(line.get("state"), (dict, list))
        else line.get("state", ""),
        "questions": qs,
        "answers": answers,
        "source": f"gmux-router:{line.get('source', '?')}",
        "ts": line.get("ts", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--dataset", default=str(ROOT / "datasets" / "gmux-routing.jsonl"))
    a = ap.parse_args()

    log_path, ds_path = Path(a.log), Path(a.dataset)
    if not log_path.is_file():
        raise SystemExit(f"no router log at {log_path}")

    seen = set()
    if ds_path.is_file():
        for l in ds_path.open():
            try:
                r = json.loads(l)
                seen.add(
                    json.dumps(
                        [r.get("state"), sorted((r.get("questions") or {}).keys())],
                        default=str,
                    )
                )
            except Exception:
                pass

    added = skipped = bad = 0
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with ds_path.open("a") as out:
        for l in log_path.open():
            l = l.strip()
            if not l:
                continue
            try:
                row = to_protocol_row(json.loads(l))
            except Exception:
                bad += 1
                continue
            if not row:
                bad += 1
                continue
            key = json.dumps(
                [row["state"], sorted(row["questions"].keys())], default=str
            )
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    print(
        f"gmux_collect: +{added} rows (skipped {skipped} dups, {bad} bad) -> {ds_path}"
    )


if __name__ == "__main__":
    main()
