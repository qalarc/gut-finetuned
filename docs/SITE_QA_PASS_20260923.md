# Site QA pass — Jev browser agent, first pass (2026-09-23)

**Method:** the jev-ultrafast browser agent (Jev cloud as brain, headless chromium via
browser-harness CDP) ran 6 navigation/functionality tasks across the qalarc site
portfolio — a functional review that doubles as browser-ops Phase-A training capture
(`capture_traces.py` → `traces/` → `scripts/traces_to_dataset.py` →
`datasets/browser-ops.jsonl`).

## Verdicts per site

| Site | Result | Evidence |
|---|---|---|
| **tradez.au** | ✅ FULL PASS | 12.9s, 3 clicks: header nav → Terms → samples page → back home via logo. All targets resolved. |
| **qalarc.com** | ✅ nav works (task budget-blocked) | 15/15 nav clicks loaded real pages (Products, Focus, Work with us, AI-OS, About, Projects, Blog, blog post /blog/_pipeline/protocol). Hit the 15-action demo budget before reaching the contact link — no site defect. |
| **qalarc.com/blog** | ⚠️ verify render | "All Posts (44)" click resolved to `/blog/#` and the observed body text was EMPTY — client-side-rendered index likely invisible to quick snapshots. Needs a human eye / slower snapshot. |
| **endispute.com.au** | ⚠️ verify render | "Services" click → `/#services` with EMPTY observed body — same suspicion (JS-only section or slow hydrate). Retries hit env screenshot-timeouts (chromium under training CPU load), not a confirmed site fault. |
| **goetica.io** | 🔴 **DNS DEAD** | Agent saw chrome DNS error; independent server probe: `Name or service not known`. Apex does not resolve. |
| **www.qalarc.ai / qalarc.ai** | 🔴 **DNS DEAD** | Same probe failure for BOTH apex and www. |

## Actions needed (owner)

1. **goetica.io + qalarc.ai** — check registrar/DNS. Neither resolves from this server;
   the concierge monitor does not currently cover these hostnames (no alerts fired).
   Recommend adding them to the monitor targets so this can't recur silently.
2. **qalarc.com/blog + endispute services** — confirm SSR/prerender so crawlers and
   agent snapshots see content (also an SEO/GEO issue: AI engines snapshot like we do).

## Environment notes (for repeatability)

- Chromium must run with CDP before the agent starts:
  `chromium --headless=new --remote-debugging-port=9222 --user-data-dir=$HOME/.config/chromium --no-first-run about:blank`
  (browser-harness detects the browser via the profile's SingletonLock + probes 9222/9223).
- Run it while training is idle: screenshots time out at 5s under heavy CPU load
  (two retries failed purely on this).
- Rerun: `browser_agent/capture_traces.py` (tasks in `tasks.jsonl`; retry set in
  `tasks_retry.jsonl`).

## Training yield

First verified run converted: +4 rows in `datasets/browser-ops.jsonl` (source `trace`).
The loop is proven end-to-end (agent → trace → converter → dataset); volume needs
regular passes (each pass ≈ 20-80 rows when the environment isn't contended) plus the
Playwright instrumentation (`browser_agent/playwright_capture.py`) on real automation.
