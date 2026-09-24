"""playwright_capture.py — Stage-1 instrumentation for the browser-ops fine-tune
(the Playwright plan: record (state, action, outcome) triples from REAL automation).

Wrap any Playwright / patchright Page to emit one protocol row per action:

    from playwright_capture import capture_page
    page = capture_page(page, source="eye_of_web")   # instead of plain Page
    ... existing automation unchanged ...

Every click/fill/goto/select/check then appends to datasets/browser-ops.jsonl:
    state     = compact page snapshot (url, title, visible-element summary)
    questions = operation (CLICK/TYPE_TEXT/SELECT/NAVIGATE/WAIT) + <op>_target heads
    answers   = the chosen operation + target, as hard labels (the script's own choice
                IS the teacher — it's working automation code)
    source    = "playwright:<your-tag>"

Privacy: states contain page structure, not form values — fills are recorded as
TYPE_TEXT without the typed content. PII-sensitive scripts (rego flows) should pass
redact=True (default) and avoid capture on authenticated personal pages.

The rows train the same RLCD loop (mcp_finetune/train.py) and serve through :8798 with
X-Laya-Domain: browser-ops — the System-1 for browser agents (action selection,
completion detection; System-2 stays with the LLM for novel situations).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "browser-ops.jsonl"

OPERATION_LABELS = {
    "click": "CLICK",
    "fill": "TYPE_TEXT",
    "type": "TYPE_TEXT",
    "select_option": "SELECT",
    "goto": "NAVIGATE",
    "wait_for_selector": "WAIT",
    "check": "CLICK",
    "press": "CLICK",
}


def _element_summary(page, limit: int = 40) -> list[dict]:
    """Compact visible-element table (role + label + index) — mirrors jev-ultrafast's
    element-table shape so both data sources train one model the same way."""
    try:
        return page.evaluate(
            """(limit) => {
              const out = [];
              const nodes = document.querySelectorAll('a,button,input,select,textarea,[role=button]');
              for (const n of nodes) {
                if (out.length >= limit) break;
                const r = n.getBoundingClientRect();
                if (!r.width && !r.height) continue;
                out.push({
                  role: n.tagName.toLowerCase(),
                  label: (n.innerText || n.getAttribute('aria-label') || n.getAttribute('placeholder') || n.value || '').trim().slice(0, 80),
                });
              }
              return out.map((e, i) => ({index: String(i + 1), ...e}));
            }""",
            limit,
        )
    except Exception:
        return []


def _snapshot(page) -> dict:
    try:
        title = page.title()
        url = page.url
    except Exception:
        title, url = "", ""
    return {
        "page": {"url": url, "title": title, "text": ""},
        "elements": _element_summary(page),
    }


def _log_row(source: str, page, op: str, target: str) -> None:
    row = {
        "state": json.dumps({"goal": source, **_snapshot(page)}, ensure_ascii=False),
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": source, "rules": "next browser operation"},
                "criteria": {
                    "CLICK": "click an element",
                    "TYPE_TEXT": "enter text",
                    "SELECT": "select a value",
                    "NAVIGATE": "go to a URL",
                    "WAIT": "wait for content",
                },
            },
            "operation_target": {
                "type": "choice",
                "instructions": "which element",
                "criteria": {
                    str(i + 1): e["label"] or e["role"]
                    for i, e in enumerate(_element_summary(page))
                },
            },
        },
        "answers": {"operation": op, "operation_target": target},
        "source": f"playwright:{source}",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    with DATASET.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def capture_page(page, source: str = "unnamed", redact: bool = True):
    """Return a proxy that mirrors the Page and logs actions to browser-ops.jsonl."""
    import sys

    class _Captured:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def _captured(self, op, target_arg):
            try:
                target = str(target_arg)
                if redact and op == "TYPE_TEXT":
                    target = "<redacted>"
                _log_row(source, self._inner, op, target[:60])
            except Exception:
                pass  # capture must never break the automation
            return getattr(self._inner, op)

        def click(self, *a, **k):
            return self._captured("CLICK", a[0] if a else "?")(*a, **k)

        def fill(self, *a, **k):
            return self._captured("TYPE_TEXT", a[0] if a else "?")(*a, **k)

        def select_option(self, *a, **k):
            return self._captured("SELECT", a[0] if a else "?")(*a, **k)

        def goto(self, *a, **k):
            return self._captured("NAVIGATE", a[0] if a else "?")(*a, **k)

        def check(self, *a, **k):
            return self._captured("CLICK", a[0] if a else "?")(*a, **k)

        def press(self, *a, **k):
            return self._captured("CLICK", a[0] if a else "?")(*a, **k)

    return _Captured(page)
