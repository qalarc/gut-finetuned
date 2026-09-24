#!/usr/bin/env python
"""export_library_marks.py — export the Doof.ing library catalog into the
compliance-marks dataset (the "library ratings / compliance marks" training area).

Sources:
  1. catalogData.entries parsed from github_repos/Doof.ing/library/browse.html
     (text, category, interpretation, votes)
  2. OPTIONAL: the visitor's localStorage dump — `libraryVotes` ({id: true}). Export it
     from the browser console on browse.html:
       copy(JSON.stringify(localStorage.getItem('libraryVotes')))
     and save to /tmp/libraryVotes.json — those become keep-labels (voted = keep).

Each real entry becomes one dataset row; Jev supplies the soft teacher labels and the
community votes stay in the state (future outcome labels, protocol §1.2).

Usage: laya-venv/bin/python scripts/export_library_marks.py [--votes /tmp/libraryVotes.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_shadow_label import jev_call, to_soft_answers  # noqa: E402
from monitor_shadow import _load_key_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BROWSE_HTML = Path.home() / "projects/github_repos/Doof.ing/library/browse.html"
DATASET = ROOT / "datasets" / "compliance-marks.jsonl"

QUESTIONS = {
    "keep": {
        "type": "choice",
        "instructions": "Library compliance decision",
        "criteria": {
            "keep": "fits the library's purpose and standards",
            "review": "needs a human look",
            "remove": "violates scope/spam",
        },
    },
    "canonical": {
        "type": "noul",
        "instructions": "This entry is core canon of the library",
    },
    "on_topic": {
        "type": "choice",
        "instructions": "Theme check",
        "criteria": {
            "on-theme": "matches the library's theme",
            "tangential": "loosely related",
            "off-theme": "unrelated or spam",
        },
    },
}


def parse_catalog() -> list[dict]:
    """Extract the entries JSON array from the inline catalogData in browse.html."""
    html = BROWSE_HTML.read_text()
    m = re.search(r'"entries"\s*:\s*(\[.*?\])\s*,\s*"', html, re.S)
    if not m:
        m = re.search(r'"entries"\s*:\s*(\[[^\]]*\])', html, re.S)
    if not m:
        sys.exit(f"could not locate entries in {BROWSE_HTML}")
    # The array is valid JSON but may carry trailing commas before the next key —
    # sanitize: find the balanced bracket span.
    depth, end = 0, 0
    for i, ch in enumerate(m.group(1)):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    raw = m.group(1)[:end]
    raw = re.sub(r",\s*([\]}])", r"\1", raw)  # trailing commas
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--votes", default=None, help="optional libraryVotes localStorage dump"
    )
    a = ap.parse_args()
    _load_key_env()

    entries = parse_catalog()
    print(f"catalog entries: {len(entries)}")
    votes = {}
    if a.votes and Path(a.votes).is_file():
        votes = json.loads(Path(a.votes).read_text())
        print(f"localStorage votes loaded: {len(votes)}")

    added = 0
    seen = set()
    if DATASET.is_file():
        for l in DATASET.open():
            try:
                seen.add(json.loads(l).get("meta", {}).get("entry_id"))
            except Exception:
                pass
    with DATASET.open("a") as f:
        for e in entries:
            if e["id"] in seen:
                continue
            state = (
                f"library entry: '{e['text']}' [{e['category']}] "
                f"interpretation: {e['interpretation']} community votes: {e['votes']}."
                + (
                    " VISITOR_VOTED=yes"
                    if votes.get(str(e["id"])) or votes.get(e["id"])
                    else ""
                )
            )
            try:
                resp = jev_call(state, QUESTIONS)
            except Exception as ex:
                print(f"  entry {e['id']}: jev failed ({ex})", file=sys.stderr)
                continue
            answers = to_soft_answers(QUESTIONS, resp.get("answers", {}))
            if not answers:
                continue
            row = {
                "state": state,
                "questions": QUESTIONS,
                "answers": answers,
                "source": "doof-library",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "meta": {"entry_id": e["id"], "votes": e["votes"]},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            added += 1
    print(f"DONE: +{added} catalog rows -> {DATASET}")


if __name__ == "__main__":
    main()
