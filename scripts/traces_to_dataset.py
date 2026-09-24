#!/usr/bin/env python
"""traces_to_dataset.py — convert jev-ultrafast traces → datasets/browser-ops.jsonl.

jev-ultrafast saves runner state as `state.json` (agent.snapshot() + verification fields).
Every model call is one entry in `decisions[]`, and each entry embeds the EXACT SystemOne
request it made (`request` = {state, questions, model}) plus all probability heads — so
conversion is lossless:

  state     = json.dumps(entry["request"]["state"])          # page + element table + goal
  questions = entry["request"]["questions"]                   # operation + *_target heads
  answers   = the EXECUTED operation + its target head only (soft distributions);
              unused target heads are NOT trained on (they were never acted on)
  source    = "trace"

Verification (LAYA_BRIDGE.md Phase A: "verified outcomes only"): a run's rows are kept
only if the run finished DONE and the runner's verification passed — state["status"]=="done"
plus summary.json verified=true, or state["verification"] truthy when present.

Usage:
  laya-venv/bin/python scripts/traces_to_dataset.py TRACES_DIR [MORE_DIRS...]
  laya-venv/bin/python scripts/traces_to_dataset.py ~/.../jev-ultrafast/artifacts --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets" / "browser-ops.jsonl"


def run_verified(d: Path) -> tuple[bool, str]:
    """Did this trace run finish with a verified DONE?"""
    state_file = d / "state.json"
    if not state_file.is_file():
        return False, "no state.json"
    try:
        st = json.loads(state_file.read_text())
    except Exception as e:
        return False, f"bad state.json: {e}"
    status = st.get("status")
    if status == "blocked":
        return False, "run BLOCKED"
    verified = None
    summary = d / "summary.json"
    if summary.is_file():
        try:
            verified = bool(json.loads(summary.read_text()).get("verified"))
        except Exception:
            pass
    if verified is None and "verification" in st:
        verified = bool(st.get("verification"))
    if status != "done":
        return False, f"status={status}"
    if verified is False:
        return False, "verification failed"
    return True, ("verified" if verified else "done (no explicit verification field)")


def rows_from_run(d: Path):
    st = json.loads((d / "state.json").read_text())
    goal = st.get("goal", "")
    ok, why = run_verified(d)
    ts = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime((d / "state.json").stat().st_mtime)
    )
    out = []
    for dec in st.get("decisions", []):
        req = dec.get("request") or {}
        questions = req.get("questions") or {}
        if not questions:
            continue
        op = dec.get("operation")
        if not op:
            continue
        answers = {}
        op_ans = (dec.get("raw_answers") or {}).get("operation")
        if isinstance(op_ans, dict) and op_ans.get("probabilities"):
            answers["operation"] = {
                "probabilities": {
                    k: round(float(v), 4) for k, v in op_ans["probabilities"].items()
                }
            }
        # only the target head whose operation was executed
        tgt_key = op.lower() + "_target"
        tgt_probs = dec.get("target_probabilities") or {}
        if dec.get("target") and tgt_key in questions and tgt_probs:
            answers[tgt_key] = {
                "probabilities": {
                    str(k): round(float(v), 4) for k, v in tgt_probs.items()
                }
            }
        if not answers:
            continue
        row = {
            "state": json.dumps(
                {"goal": goal, **(req.get("state") or {})}, ensure_ascii=False
            ),
            "questions": questions,
            "answers": answers,
            "source": "trace",
            "ts": ts,
            "meta": {
                "run": str(d),
                "choice": dec.get("choice"),
                "run_ok": ok,
                "verify": why,
            },
        }
        h = hashlib.sha1(
            (row["state"] + json.dumps(answers, sort_keys=True)).encode()
        ).hexdigest()
        row["meta"]["hash"] = h
        out.append(row)
    return out, ok, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dirs",
        nargs="+",
        help="trace dirs (each containing state.json) or parent dirs to scan",
    )
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument(
        "--include-unverified",
        action="store_true",
        help="also keep rows from runs that did not verify (kept, but tagged in meta)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs = []
    for p in map(Path, a.dirs):
        if (p / "state.json").is_file():
            runs.append(p)
        else:
            runs.extend(
                sorted(
                    d
                    for d in p.rglob("state.json")
                    if (d.parent / "state.json").is_file()
                )
            )
    runs = sorted({r if (r / "state.json").is_file() else r.parent for r in runs})

    seen = set()
    if Path(a.dataset).is_file():
        for l in Path(a.dataset).open():
            try:
                seen.add(json.loads(l).get("meta", {}).get("hash", ""))
            except Exception:
                pass

    added = kept_runs = 0
    rows_out = []
    for d in runs:
        try:
            rows, ok, why = rows_from_run(d)
        except Exception as e:
            print(f"skip {d}: {e}")
            continue
        if not ok and not a.include_unverified:
            print(f"skip {d}: {why}")
            continue
        kept_runs += 1
        for r in rows:
            if r["meta"]["hash"] in seen:
                continue
            seen.add(r["meta"]["hash"])
            rows_out.append(r)

    print(f"{len(runs)} runs scanned, {kept_runs} kept, {len(rows_out)} new rows")
    if a.dry_run:
        for r in rows_out[:3]:
            print(json.dumps(r)[:220], "…")
        return
    if rows_out:
        Path(a.dataset).parent.mkdir(parents=True, exist_ok=True)
        with Path(a.dataset).open("a") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"+{len(rows_out)} -> {a.dataset}")


if __name__ == "__main__":
    main()
