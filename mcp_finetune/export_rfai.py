#!/usr/bin/env python
"""export_rfai.py — build the rfai-intent fine-tuning dataset from the RFAI archive.

Reads RFAI's production SQLite archive (rfai.sqlite) and emits protocol-format
JSONL for train.py:  datasets/rfai-intent.jsonl

    {"state": transcript,
     "questions": {"intent": {choice, 9 classes}, "urgent": {noul}},
     "answers":  {"intent": <label>, "urgent": <urgency float>},
     "meta": {...}}

Sources of labels:
  1. rows already classified in the archive (GLM/keyword hybrid, columns
     intent + urgency)                       -> source "archive"
  2. --teacher-glm N: sample N unclassified rows that DO have transcripts and
     label them with the local GLM 4.7 Flash (abliterated) classifier RFAI
     already uses (db/intent_llm.classify_intent) -> source "glm-teacher"

Class mapping (canonical 9-class schema = RFAI production prompt):
  'transmitted' rows are skipped (own TX, not traffic);
  'aviation'    maps to 'navigation' (position/ATC reports);
  NULL intent with a transcript is only used via the GLM teacher path.

Usage:
  $LV export_rfai.py --db ~/projects/hack_RF/RFAI/db/data/rfai.sqlite
  $LV export_rfai.py --db ... --teacher-glm 80     # adds GLM-teacher rows
  $LV export_rfai.py --db ... --stats              # just show label distribution
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "rfai-intent.jsonl"

# the canonical production schema (mirrors RFAI db/intent_llm.py prompt rules)
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

MIN_TRANSCRIPT = 40  # chars — below this a transcript carries no intent signal

import urllib.request  # noqa: E402

OLLAMA = "http://localhost:11434"

_SOFT_PROMPT = """You are a radio intelligence analyst. Read this {band} transmission and rate it.

Transcript: "{transcript}"

Respond with ONLY valid JSON, no markdown:
{{
  "intent": "<one of: emergency convoy navigation check-in technical weather social data unknown>",
  "probabilities": {{"emergency": 0.0, "convoy": 0.0, "navigation": 0.0, "check-in": 0.0, "technical": 0.0, "weather": 0.0, "social": 0.0, "data": 0.0, "unknown": 0.0}},
  "urgency": <0.0-1.0>,
  "confidence": <0.0-1.0>
}}

The probabilities object must cover all nine classes and sum to ~1.0 —
spread mass over every plausible class instead of one-hot."""


def glm_soft(transcript: str, band: str, timeout: int = 75) -> dict | None:
    """Soft teacher label: GLM 4.7 Flash (abliterated) via local Ollama."""
    payload = json.dumps({
        "model": "glm-4.7-flash-abl:alexei-v2",
        "prompt": _SOFT_PROMPT.format(band=band or "radio", transcript=transcript.strip()),
        "stream": False, "format": "json",
        "options": {"temperature": 0.25, "num_predict": 220},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.load(r).get("response", "{}")
        d = json.loads(raw)
        probs = {k: max(0.0, float(v)) for k, v in d["probabilities"].items() if k in CRITERIA}
        if len(probs) != len(CRITERIA):
            return None
        total = sum(probs.values())
        if total <= 0:
            return None
        probs = {k: round(v / total, 4) for k, v in probs.items()}
        return {"probabilities": probs, "urgency": max(0.0, min(1.0, float(d.get("urgency", 0.0))))}
    except Exception:
        return None


SOFT_CACHE = OUT.parent / "rfai-soft-cache.json"


def _cache_load() -> dict:
    import hashlib
    if SOFT_CACHE.exists():
        try:
            return json.load(SOFT_CACHE.open())
        except Exception:
            return {}
    return {}


def _cache_key(transcript: str, band: str) -> str:
    import hashlib
    return hashlib.sha1(f"{band}|{transcript.strip()}".encode()).hexdigest()


def glm_soft_cached(transcript: str, band: str) -> dict | None:
    """glm_soft with a JSON sidecar cache keyed by (band, transcript)."""
    cache = _cache_load()
    k = _cache_key(transcript, band)
    if k in cache:
        return cache[k]
    res = glm_soft(transcript, band)
    if res is not None:
        cache[k] = res
        tmp = SOFT_CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(SOFT_CACHE)
    return res


def synth_records(n: int, seed: int) -> list[dict]:
    """Balanced synthetic corpus from RFAI's synth_data.py (intent-labeled by construction).

    generate_corpus() is intent-weighted per band, so we over-generate once and
    then pick balanced per-intent examples.
    """
    sys.path.insert(0, "/home/fivelidz/projects/hack_RF/RFAI/db")
    import synth_data  # noqa: E402
    pool = synth_data.generate_corpus(count=max(n * 6, 300), seed=seed)
    by_intent: dict[str, list[dict]] = {}
    for r in pool:
        i = r.get("intent")
        if i in CRITERIA and r.get("transcript"):
            by_intent.setdefault(i, []).append(r)
    per = max(1, n // len(CRITERIA))
    out = []
    for intent, recs in by_intent.items():
        for r in recs[:per]:
            r["_want"] = intent
            out.append(r)
    return out


def export_v2(db: Path, synth_n: int, seed: int) -> int:
    """v2: archive one-hot rows + ALL teacher candidates with SOFT GLM labels +
    balanced GLM-verified synthetic rows. Writes rfai-intent-v2.jsonl."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows_out, skipped = [], collections.Counter()

    def add(transcript, band, answers, source, txid):
        if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT:
            skipped["too-short"] += 1
            return False
        rows_out.append({
            "state": transcript.strip(),
            "questions": QUESTIONS,
            "answers": answers,
            "meta": {"source": source, "band": band, "tx": txid},
        })
        return True

    for txid, transcript, band, intent, urgency in c.execute(
        "SELECT id, transcript, band, intent, urgency FROM transmissions "
        "WHERE intent IS NOT NULL"
    ):
        if intent == "transmitted":
            skipped["transmitted"] += 1
            continue
        if intent == "aviation":
            intent = "navigation"
        if intent not in CRITERIA:
            skipped["no-label"] += 1
            continue
        add(transcript, band,
            {"intent": intent, "urgent": max(0.0, min(1.0, float(urgency or 0.0)))},
            "archive", txid)
    n_archive = len(rows_out)

    cand = c.execute(
        "SELECT id, transcript, band FROM transmissions "
        "WHERE intent IS NULL AND transcript IS NOT NULL AND LENGTH(transcript) >= ? "
        "ORDER BY id", (MIN_TRANSCRIPT,)).fetchall()
    print(f"soft-labelling {len(cand)} teacher candidates with GLM...")
    kept = 0
    for i, (txid, transcript, band) in enumerate(cand):
        soft = glm_soft_cached(transcript, band)
        if soft:
            add(transcript, band,
                {"intent": {"probabilities": soft["probabilities"]},
                 "urgent": soft["urgency"]},
                "glm-teacher-soft", txid)
            kept += 1
        else:
            skipped["soft-fail"] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cand)} (kept {kept})", flush=True)

    if synth_n:
        print(f"synthesising + verifying ~{synth_n} rows...")
        recs = synth_records(synth_n, seed)
        kept = 0
        for i, r in enumerate(recs):
            soft = glm_soft_cached(r["transcript"], r.get("band", ""))
            if soft and max(soft["probabilities"], key=soft["probabilities"].get) == r["_want"]:
                add(r["transcript"], r.get("band", ""),
                    {"intent": {"probabilities": soft["probabilities"]},
                     "urgent": soft["urgency"]},
                    "synth-glm-verified", None)
                kept += 1
            else:
                skipped["synth-reject"] += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(recs)} (kept {kept})", flush=True)

    out = OUT.parent / "rfai-intent-v2.jsonl"
    with out.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def argmax_intent(r):
        a = r["answers"]["intent"]
        return a if isinstance(a, str) else max(a["probabilities"], key=a["probabilities"].get)
    dist = collections.Counter(argmax_intent(r) for r in rows_out)
    print(f"\nwrote {len(rows_out)} rows -> {out}")
    print(f"  archive: {n_archive} | teacher+synth: {len(rows_out) - n_archive} | skipped: {dict(skipped)}")
    for k, v in dist.most_common():
        print(f"    {k:12s} {v}")
    return 0


def row_state(transcript: str, band: str | None) -> str:
    return transcript.strip()


def export(db: Path, teacher_n: int, seed: int) -> int:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows_out: list[dict] = []
    skipped = collections.Counter()

    def add(transcript, band, intent, urgency, source, txid):
        if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT:
            skipped["too-short"] += 1
            return
        if intent == "transmitted":  # own TX, not traffic
            skipped["transmitted"] += 1
            return
        if intent == "aviation":  # ATC/position reports -> canonical schema
            intent = "navigation"
        if intent is None or intent not in CRITERIA:
            skipped["no-label"] += 1
            return
        rows_out.append(
            {
                "state": row_state(transcript, band),
                "questions": QUESTIONS,
                "answers": {
                    "intent": intent,
                    "urgent": max(0.0, min(1.0, float(urgency or 0.0))),
                },
                "meta": {"source": source, "band": band, "tx": txid},
            }
        )

    # 1. already-labeled archive rows
    for txid, transcript, band, intent, urgency in c.execute(
        "SELECT id, transcript, band, intent, urgency FROM transmissions "
        "WHERE intent IS NOT NULL"
    ):
        add(transcript, band, intent, urgency, "archive", txid)
    n_archive = len(rows_out)

    # 2. GLM teacher rows (unclassified but with real transcripts)
    if teacher_n:
        sys.path.insert(0, str(Path(db).parent.parent))  # RFAI/db for intent_llm
        from intent_llm import classify_intent  # noqa: E402

        cand = c.execute(
            "SELECT id, transcript, band FROM transmissions "
            "WHERE intent IS NULL AND transcript IS NOT NULL "
            "AND LENGTH(transcript) >= ? ORDER BY id",
            (MIN_TRANSCRIPT,),
        ).fetchall()
        rng = random.Random(seed)
        # keep all non-FM candidates (rare, valuable); sample the FM mass
        non_fm = [r for r in cand if "FM" not in (r[2] or "")]
        fm = [r for r in cand if "FM" in (r[2] or "")]
        take = non_fm + rng.sample(fm, min(max(0, teacher_n - len(non_fm)), len(fm)))
        print(f"teacher labelling {len(take)} rows with GLM ({len(non_fm)} non-FM)...")
        for i, (txid, transcript, band) in enumerate(take):
            res = classify_intent(transcript, band=band or "")
            add(
                transcript,
                band,
                res.get("intent"),
                res.get("urgency"),
                "glm-teacher",
                txid,
            )
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(take)}")

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = collections.Counter(r["answers"]["intent"] for r in rows_out)
    print(f"\nwrote {len(rows_out)} rows -> {OUT}")
    print(
        f"  archive labels: {n_archive} | glm-teacher: {len(rows_out) - n_archive} | skipped: {dict(skipped)}"
    )
    print("  class distribution:")
    for k, v in dist.most_common():
        print(f"    {k:12s} {v}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to rfai.sqlite")
    ap.add_argument(
        "--teacher-glm",
        type=int,
        default=0,
        help="label N unclassified-but-transcribed rows with local GLM",
    )
    ap.add_argument("--v2", action="store_true",
                    help="v2 mode: all teacher candidates with SOFT GLM labels + synthetic augmentation")
    ap.add_argument("--synth", type=int, default=300, help="[v2] synthetic rows to generate+verify")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.v2:
        sys.exit(export_v2(Path(args.db), args.synth, args.seed))
    sys.exit(export(Path(args.db), args.teacher_glm, args.seed))
