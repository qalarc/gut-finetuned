#!/usr/bin/env python
"""keys.py — Typesafe (Jev) API key resolution for all qalarc decision-stack components.

Resolution order (first hit wins):
  1. $TYPESAFE_API_KEY environment variable
  2. $TYPESAFE_KEY_FILE (path to a file containing the raw key)
  3. ~/.secrets/typesafe.env  (TYPESAFE_API_KEY=... line)
  4. ./.typesafe_key          (project-local file — for multi-user / per-user setups)

Multi-user note: every component (jev MCP, laya serve.py, osint grading, triage sidecar,
jev-seo) imports this resolver, so a user can run the whole stack with their own Jev key
by either exporting TYPESAFE_API_KEY or dropping their key into one of the file locations.
Share the key file, never the key itself.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_key() -> tuple[str, str]:
    """Returns (key, source_description)."""
    k = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if k:
        return k, "env TYPESAFE_API_KEY"

    kf = os.environ.get("TYPESAFE_KEY_FILE", "").strip()
    if kf:
        p = Path(kf).expanduser()
        if p.is_file():
            return p.read_text().strip(), f"key file {kf}"

    for loc in (
        Path.home() / ".secrets" / "typesafe.env",
        Path.cwd() / ".typesafe_key",
    ):
        if loc.is_file():
            for line in loc.read_text().splitlines():
                line = line.strip()
                if line.startswith("TYPESAFE_API_KEY="):
                    return line.split("=", 1)[1].strip(), f"file {loc}"
            # raw-key file (single line, no KEY= prefix)
            first = next((l for l in loc.read_text().splitlines() if l.strip()), "")
            if first and not first.startswith("#"):
                return first.strip(), f"file {loc}"
    return "", "none"


def get_key() -> str:
    return resolve_key()[0]


if __name__ == "__main__":
    k, src = resolve_key()
    print(f"source: {src}")
    print(f"key: {k[:12]}...{k[-6:]}" if k else "key: NOT FOUND")
