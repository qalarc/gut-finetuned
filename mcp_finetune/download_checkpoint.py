#!/usr/bin/env python
"""download_checkpoint.py — pick and download Laya checkpoints, with descriptions.

The three upstream checkpoints are OPTIONAL downloads (~0.65-2.3 GB each). This
tool exists so a human (or agent) can see exactly what each one is before
pulling it:

    python download_checkpoint.py --list            # what each is + cached state
    python download_checkpoint.py --pick english    # just the English checkpoint
    python download_checkpoint.py --pick multilingual
    python download_checkpoint.py --pick typed-decisions
    python download_checkpoint.py --pick all        # everything (full Router)

Run inside the laya venv:
    LV=~/projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python
    $LV download_checkpoint.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_REPO = "convaiinnovations/laya"
CACHE_GLOB = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya"

CHECKPOINTS = {
    "english": {
        "subfolder": None,
        "name": "laya (English)",
        "size": "~2.3 GB",
        "params": "421M (ModernBERT-large)",
        "context": "512 tokens",
        "what": "The default English checkpoint. Best pick for English-only "
        "traffic — radio transcripts, support triage, monitor alerts. "
        "This is the checkpoint benchmarked at 33 ms/question on a T4 "
        "and the one the RFAI rfai-intent fine-tune builds on.",
        "pick_when": "Your states are English and context fits in 512 tokens "
        "(virtually all radio transmissions do).",
    },
    "multilingual": {
        "subfolder": "multilingual",
        "name": "laya-multilingual",
        "size": "~0.65 GB",
        "params": "322M (mmBERT-base)",
        "context": "1024 tokens",
        "what": "Covers 100+ languages, roughly 2x faster than English, and "
        "the only checkpoint that works non-English (45/51 MASSIVE "
        "languages clear 3x random vs 23/51 for English). Slightly "
        "weaker on pure-English suites.",
        "pick_when": "Traffic mixes languages (multilingual crews, VF "
        "maritime, border comms) or you want the smallest "
        "footprint with the longest base context.",
    },
    "typed-decisions": {
        "subfolder": "typed-decisions",
        "name": "laya-typed-decisions",
        "size": "~0.81 GB",
        "params": "421M (ModernBERT-large)",
        "context": "1024 tokens",
        "what": "The workflow-tuned large checkpoint: trained on the six "
        "typed-decision application themes (triage, routing, guard, "
        "moderation, email, shortlist) with 1024-token context — the "
        "longest of the large pair. Same speed class as English.",
        "pick_when": "Longer states (full log threads, email, multi-doc) or "
        "multi-question workflows where the extra training "
        "distribution helps zero-shot.",
    },
}


def cached_subfolders() -> set[str]:
    """Which checkpoints already have weights in the local HF cache."""
    have = set()
    snaps = CACHE_GLOB / "snapshots"
    if not snaps.is_dir():
        return have
    blobs = (
        list((CACHE_GLOB / "blobs").glob("*"))
        if (CACHE_GLOB / "blobs").is_dir()
        else []
    )
    if not blobs:
        return have
    have.add("english")  # root files present
    for sub in ("multilingual", "typed-decisions"):
        if any(s.is_dir() and (s / sub).exists() for s in snaps.iterdir()):
            have.add(sub)
    return have


def show_list() -> None:
    have = cached_subfolders()
    print("Laya checkpoints (all optional downloads, Apache-2.0, local):\n")
    total = 0.0
    for key, c in CHECKPOINTS.items():
        mark = "[cached]" if key in have else "[not downloaded]"
        print(
            f"  {key:16s} {mark:16s} {c['size']:9s} {c['params']}, ctx {c['context']}"
        )
        print(f"    what:  {c['what']}")
        print(f"    when:  {c['pick_when']}\n")
    print("  Pick with: --pick english|multilingual|typed-decisions|all")
    if not have:
        print("\n  Nothing downloaded yet. The English checkpoint is the right")
        print("  first pull for RFAI (~1.7 GB).")


def download(pick: str) -> int:
    from huggingface_hub import snapshot_download

    keys = list(CHECKPOINTS) if pick == "all" else [pick]
    if pick != "all" and pick not in CHECKPOINTS:
        print(f"unknown pick {pick!r} — run --list", file=sys.stderr)
        return 2
    for key in keys:
        c = CHECKPOINTS[key]
        patterns = [
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        ]
        if c["subfolder"]:
            patterns = [f"{c['subfolder']}/{p}" for p in patterns]
        print(f"downloading {c['name']} ({c['size']}) ...")
        p = snapshot_download(HF_REPO, allow_patterns=patterns, token=None)
        print(f"  -> {p}")
    print("\nDone. Verify with: python download_checkpoint.py --list")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list", action="store_true", help="show descriptions + cached state"
    )
    ap.add_argument(
        "--pick", default=None, help="english|multilingual|typed-decisions|all"
    )
    a = ap.parse_args()
    if a.list or not a.pick:
        show_list()
        sys.exit(0)
    sys.exit(download(a.pick))
