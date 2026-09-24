#!/usr/bin/env python
"""monitor_shadow.py — Jev-shadow on LIVE monitor transitions (protocol §1 strategy 1).

Watches the concierge hub's event feed (SQLite, opened READ-ONLY — the monitor service
itself is untouched) and labels every new monitor transition with Jev: severity, wake,
cause. Soft distributions appended to datasets/monitor-triage.jsonl (source:
jev-shadow-live). This is the flywheel: the monitor runs anyway; every alert it emits
becomes a training row for the local model that will eventually shadow it.

Run modes:
  --once          poll once and exit (for a systemd timer / cron)
  --loop [SEC]    long-running poller (default 300s)
  --backfill N    label the N most recent monitor events first (dedup applies)

Install (optional, user-level):
  systemd user timer calling: laya-venv/bin/python scripts/monitor_shadow.py --once
  (unit files provided in scripts/systemd/ — enable manually; never auto-enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_shadow_label import jev_call, to_soft_answers  # noqa: E402


def _load_key_env():
    env = Path.home() / ".secrets/typesafe.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


_load_key_env()

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "monitor-triage.jsonl"
HUB_DB = Path(
    os.environ.get("CH_HUB_DB", Path.home() / ".local/share/concierge-hub/hub.db")
)
STATE_F = Path(
    os.environ.get(
        "MONITOR_SHADOW_STATE",
        Path.home() / ".local/share/laya-integrations/monitor_shadow_state.json",
    )
)

Q_SEVERITY = {
    "type": "choice",
    "instructions": "Severity",
    "criteria": {
        "low": "cosmetic",
        "medium": "degradation",
        "high": "outage",
        "critical": "emergency",
    },
}
Q_WAKE = {"type": "noul", "instructions": "Wake the owner at 3am"}
Q_CAUSE = {
    "type": "choice",
    "instructions": "Likely cause",
    "criteria": {
        "origin_down": "app server 5xx / process dead",
        "network": "connectivity, DNS or routing",
        "tls": "certificate problem",
        "upstream": "third-party dependency slow or failing",
    },
}


def load_state():
    try:
        return json.loads(STATE_F.read_text())
    except Exception:
        return {"last_id": 0}


def save_state(st):
    STATE_F.parent.mkdir(parents=True, exist_ok=True)
    STATE_F.write_text(json.dumps(st))


def fetch_new_events(last_id: int, limit: int = 50):
    conn = sqlite3.connect(f"file:{HUB_DB}?mode=ro", uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT id, ts, severity, message FROM events WHERE source='monitor' "
            "AND id > ? ORDER BY id ASC LIMIT ?",
            (last_id, limit),
        ).fetchall()
    finally:
        conn.close()


def fetch_backfill(n: int):
    conn = sqlite3.connect(f"file:{HUB_DB}?mode=ro", uri=True, timeout=3)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT id, ts, severity, message FROM events WHERE source='monitor' "
            "ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()[::-1]
    finally:
        conn.close()


def label_event(ev) -> dict | None:
    """Monitor transition → Jev soft labels. Recoveries get a slimmer question set."""
    msg = ev["message"].strip()
    sev = ev["severity"]
    is_recovery = "UP" in msg or "recovered" in msg.lower()
    questions = {"severity": Q_SEVERITY}
    if not is_recovery:
        questions["wake"] = Q_WAKE
        questions["cause"] = Q_CAUSE
    state = f"[monitor:{sev}] {msg}"
    try:
        resp = jev_call(state, questions)
    except Exception as e:
        print(f"  jev failed for event {ev['id']}: {e}", file=sys.stderr)
        return None
    answers = to_soft_answers(questions, resp.get("answers", {}))
    if not answers:
        return None
    return {
        "state": state,
        "questions": questions,
        "answers": answers,
        "source": "jev-shadow-live",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ev["ts"])),
    }


def seen_keys() -> set:
    """Dedupe against rows already in the dataset (same state string)."""
    out = set()
    if DATASET.is_file():
        for l in DATASET.open():
            try:
                out.add(json.loads(l)["state"])
            except Exception:
                pass
    return out


def process(events) -> int:
    seen = seen_keys()
    added = 0
    for ev in events:
        row = label_event(ev)
        if not row:
            continue
        if row["state"] in seen:
            print(f"  dedupe: event {ev['id']} already labeled")
            continue
        seen.add(row["state"])
        with DATASET.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        added += 1
        print(f"  + event {ev['id']} [{ev['severity']}] {ev['message'][:60]}")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", type=int, nargs="?", const=300, metavar="SEC")
    ap.add_argument("--backfill", type=int, default=0, metavar="N")
    a = ap.parse_args()

    if not HUB_DB.is_file():
        sys.exit(f"hub db not found: {HUB_DB}")
    st = load_state()

    if a.backfill:
        events = fetch_backfill(a.backfill)
        if events:
            st["last_id"] = max(st.get("last_id", 0), events[-1]["id"])
        print(f"backfill: {len(events)} events -> +{process(events)} rows")
        save_state(st)
        return

    while True:
        events = fetch_new_events(st.get("last_id", 0))
        if events:
            added = process(events)
            st["last_id"] = events[-1]["id"]
            save_state(st)
            print(
                f"[{time.strftime('%H:%M:%S')}] {len(events)} new events, +{added} labeled rows"
            )
        elif a.once:
            print("no new events")
        if a.once or not a.loop:
            return
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
