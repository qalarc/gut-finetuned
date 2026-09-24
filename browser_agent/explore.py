#!/usr/bin/env python
"""explore.py — autonomous site explorer: the browser-ops flywheel AND the target
runtime for the fine-tuned Laya swap.

Loop: playwright drives chromium; the frontier grows from real hrefs; every cycle the
DECIDER (Jev cloud today — Laya at :8798 once browser-ops passes its gate) picks the next
link from the page's link table via the SAME operation/target typed questions the
browser-ops model trains on. Each cycle appends one protocol row to
datasets/browser-ops.jsonl (source: playwright-explore) — exploring IS data collection.

Safety: same-origin links only, never touches logout/delete/pay/submit/download, never
fills forms, respects a per-domain page cap.

Usage (chromium CDP optional — playwright launches its own):
  .browser-venv/bin/python explore.py --seed https://tradez.au --pages 25
  LAYA_URL=http://127.0.0.1:8798/v1/systemone explore.py --seed ...   # Phase B swap
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "browser-ops.jsonl"
BAD_HREF = re.compile(
    r"logout|signout|delete|remove|pay|checkout|submit|download|mailto:|tel:|#", re.I
)

OPERATION_Q = {
    "type": "choice",
    "instructions": {"rules": "pick the next exploration operation"},
    "criteria": {
        "VISIT": "navigate to the chosen link",
        "DONE": "no link worth visiting",
    },
}


def decide(url: str, state, questions, domain: str | None = None):
    """Typed decision via the configured backend: LAYA_URL (local Laya, Phase B) when set,
    otherwise the Jev cloud API (Phase A). Same wire protocol on both ends.
    Retries with backoff on 403/429/5xx (burst limits)."""
    headers = {"Content-Type": "application/json"}
    body = {"state": state, "questions": questions}
    if domain:
        headers["X-Laya-Domain"] = domain
    if "typesafe.ai" in url:
        body["model"] = "jev-1.13.0"
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key and Path.home().joinpath(".secrets/typesafe.env").is_file():
            for line in (
                Path.home().joinpath(".secrets/typesafe.env").read_text().splitlines()
            ):
                if line.startswith("TYPESAFE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    delay = 2.0
    for attempt in range(4):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST", headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()), (time.time() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("decide: retries exhausted")


def log_row(
    site, url, links, chosen, probs, local_ms=None, jev_heads=False, op_probs=None
):
    """Save one decision as a training row — with the SAME question heads the decider
    used (schema drift across rows is what broke gmux-routing's first calibration)."""
    tgt = "click_target" if jev_heads else "visit_target"
    op_opts = (
        {"CLICK": "visit the chosen link", "DONE": "no link worth visiting"}
        if jev_heads
        else {"VISIT": "visit the chosen link", "DONE": "no link worth visiting"}
    )
    tgt_opts = (
        {str(i + 1): {"element": f"[{i + 1}] {t}"} for i, t in enumerate(links)}
        if jev_heads
        else {str(i + 1): t for i, t in enumerate(links)}
    )
    op_ans = (
        {k: round(float(v), 4) for k, v in (op_probs or {}).items()}
        if jev_heads and op_probs
        else ({"CLICK": 1.0, "DONE": 0.0} if jev_heads else {"VISIT": 1.0, "DONE": 0.0})
    )
    row = {
        "state": json.dumps(
            {
                "goal": f"explore {site}: pick the most informative unvisited internal link",
                "page": {"url": url, "title": "", "text": ""},
                "elements": [
                    {"index": str(i + 1), "label": t} for i, t in enumerate(links)
                ],
            },
            ensure_ascii=False,
        ),
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": f"explore {site}"},
                "criteria": op_opts,
            },
            tgt: {
                "type": "choice",
                "instructions": "which link to visit next",
                "criteria": tgt_opts,
            },
        },
        "answers": {
            "operation": {"probabilities": op_ans},
            tgt: {
                "probabilities": {
                    k: round(float(v), 4) for k, v in (probs or {}).items()
                }
                if probs
                else chosen
            },
        },
        "source": "playwright-explore",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": {"from": url, "to": chosen},
    }
    if local_ms:
        row["meta"]["local_ms"] = local_ms
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    with DATASET.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def harvest_links(page, base_url, visited) -> list[tuple[str, str]]:
    """Same-origin hrefs as (abs_url, label), filtered for safety."""
    raw = page.evaluate(
        """() => [...document.querySelectorAll('a[href]')].map(a => [a.href, (a.innerText||'').trim().slice(0,70)]).filter(([h,t]) => h && t)"""
    )
    out, seen = [], set()
    for href, label in raw:
        if BAD_HREF.search(href or "") or BAD_HREF.search(label or ""):
            continue
        absu = urljoin(base_url, href.split("#")[0]).rstrip("/")
        p = urlparse(absu)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc != urlparse(base_url).netloc:  # same-origin only
            continue
        if absu in visited or absu in seen:
            continue
        seen.add(absu)
        out.append((absu, label))
    return out


def explore(
    seed: str, max_pages: int, dry: bool = False, on_event=None, should_stop=None
):
    """Walk a site frontier. Every hop: harvest links → DECIDER picks → navigate → log row.
    on_event(dict) fires per step (for the console UI); should_stop() aborts cleanly."""
    from playwright.sync_api import sync_playwright

    laya = os.environ.get("LAYA_URL")
    endpoint = laya if laya else "https://api.typesafe.ai/v1/systemone"
    decider_name = f"laya-local" if laya else "jev-cloud"
    origin = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"
    visited = set()
    frontier = [(seed.rstrip("/"), "(seed)")]
    site = urlparse(seed).netloc
    pages_done = 0

    def emit(ev):
        if on_event:
            ev["decider"] = decider_name
            on_event(ev)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            executable_path=os.environ.get("BH_CHROME_PATH", "/usr/bin/chromium"),
            args=["--no-sandbox"],
        )
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        while frontier and pages_done < max_pages:
            if should_stop and should_stop():
                break
            url, via = frontier.pop(0)
            if url in visited:
                continue
            try:
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(600)
            except Exception as e:
                print(f"  skip {url}: {e}")
                visited.add(url)
                continue
            pages_done += 1
            visited.add(url)
            links = harvest_links(page, url, visited)
            title = page.title()
            print(
                f"[{pages_done}/{max_pages}] {url} ({title[:40]!r}) links={len(links)} via={via[:30]}"
            )
            emit(
                {
                    "type": "page",
                    "n": pages_done,
                    "max": max_pages,
                    "url": url,
                    "title": title,
                    "links": len(links),
                    "via": via,
                }
            )
            if not links or dry:
                continue
            # DECISION: which link is most informative? (Jev today, Laya when gated)
            labels = [t for _, t in links[:30]]
            jev_heads = os.environ.get("BROWSER_HEADS", "") == "jev" or bool(laya)
            if jev_heads:
                # jev-ultrafast native heads — what cklxx/laya-browser (and our v2
                # fine-tune) are trained on: CLICK/DONE + click_target over elements
                qs = {
                    "operation": {
                        "type": "choice",
                        "instructions": {"goal": f"explore {site}"},
                        "criteria": {
                            "CLICK": "visit the chosen link",
                            "DONE": "no link worth visiting",
                        },
                    },
                    "click_target": {
                        "type": "choice",
                        "instructions": "which link to visit next",
                        "criteria": {
                            str(i + 1): {"element": f"[{i + 1}] {t}"}
                            for i, t in enumerate(labels)
                        },
                    },
                }
            else:
                qs = {
                    "operation": OPERATION_Q,
                    "visit_target": {
                        "type": "choice",
                        "instructions": "which link to visit next",
                        "criteria": {str(i + 1): t for i, t in enumerate(labels)},
                    },
                }
            state = {
                "goal": f"explore {site}",
                "page": {"url": url, "title": title},
                "links": [{"i": str(i + 1), "text": t} for i, t in enumerate(labels)],
            }
            try:
                resp, ms = decide(
                    endpoint, state, qs, domain=("browser-ops" if laya else None)
                )
            except Exception as e:
                print(f"  decide failed: {e}")
                continue
            tgt_head = "click_target" if jev_heads else "visit_target"
            tgt = resp["answers"].get(tgt_head, {})
            probs = tgt.get("probabilities") or {}
            choice = tgt.get("choice") or (max(probs, key=probs.get) if probs else None)
            if not choice or choice not in {str(i + 1) for i in range(len(labels))}:
                continue
            if resp["answers"].get("operation", {}).get("choice") == "DONE":
                emit(
                    {
                        "type": "decision",
                        "from": url,
                        "to": "(done)",
                        "label": "DONE",
                        "top": [("DONE", 1.0)],
                        "ms": round(ms),
                        "row": False,
                    }
                )
                break
            next_url = links[int(choice) - 1][0]
            next_label = links[int(choice) - 1][1]
            op_ans = resp["answers"].get("operation", {})
            log_row(
                site,
                url,
                labels,
                {choice: 1.0},
                probs,
                jev_heads=jev_heads,
                op_probs=op_ans.get("probabilities"),
            )
            top3 = (
                sorted(probs.items(), key=lambda kv: -kv[1])[:3]
                if probs
                else [(choice, 1.0)]
            )
            emit(
                {
                    "type": "decision",
                    "from": url,
                    "to": next_url,
                    "label": next_label,
                    "top": top3,
                    "ms": round(ms),
                    "row": True,
                }
            )
            frontier.append((next_url, next_label))
            # breadth: also queue a few unvisited links so the walk covers the site,
            # while the DECIDER still steers which page is visited next
            frontier.extend(links[:3])
        browser.close()
    emit(
        {
            "type": "done",
            "pages": pages_done,
            "visited": len(visited),
            "frontier": len(frontier),
        }
    )
    print(f"done: {pages_done} pages, visited={len(visited)}, frontier={len(frontier)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--dry", action="store_true", help="map only, no decisions/rows")
    a = ap.parse_args()
    explore(a.seed, a.pages, a.dry)


if __name__ == "__main__":
    main()
