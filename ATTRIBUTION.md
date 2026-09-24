# Attribution

This project stands on open work from several teams. Keep these credits accurate anywhere
this project is shown (README, video, LinkedIn, talks).

## Laya — the local decision engine
- **Repo:** https://github.com/NandhaKishorM/laya — Nandha Kishor M / **Convai Innovations**
- **License:** Apache-2.0 · **Weights:** https://huggingface.co/convaiinnovations/laya
- Non-autoregressive System 1 decision engine (choice/score/noul), RLCD fine-tuning method,
  Router, and benchmark suite — all their work. We contribute: the MCP fine-tuning protocol,
  dataset tooling, Jev-compatible local endpoint, and integration glue.

## Jev / System One — the cloud decision engine + paradigm
- **Vendor:** **TypeSafe AI** (typesafe.ai) — Jev `jev-1.13.0` via api.typesafe.ai.
- The typed-decisions API design (state + questions → probabilities + confidence) and the
  "atomic questions composed in code" doctrine are theirs; our local endpoint speaks their
  wire protocol for drop-in compatibility.

## jev-ultrafast — the browser agent
- **Repo:** https://github.com/browser-use/jev-ultrafast — **browser-use** team × TypeSafe.
- **License:** MIT. The bridge design in `browser_agent/` targets their agent; their repo's
  trace format feeds the browser-ops dataset.

## qalarc contributions (this repo)
- `mcp_finetune/` — MCP-based fine-tuning protocol server (collect/train/calibrate/eval/
  register), the **RLCD training runner** (`train.py`, adapted from the Laya notebook to a
  single-device production loop), and the Jev-compatible local serving endpoint
  (`serve.py`, audit-logged, preload-warmable)
- `notes/` — protocol + integration maps · `datasets/` schema · bridge design
- **Collection flywheels** (`scripts/`): Jev-shadow bulk labeler, live monitor-shadow
  watcher, OSINT scan-hit grading, browser-trace converter, gmux decision-log collector
- **`gmux-router` crate** (in gmux_v7) — typed-decision task router with three
  wire-identical backends (Jev cloud / Laya local / offline heuristic) — qalarc original
- **Kaggle 2×T4 kernel pipeline** (`kaggle/`, `scripts/make_kaggle_kernel.py`) — qalarc
- Domain datasets and fine-tuned checkpoints remain qalarc IP
- **Browser-ops fine-tune program** (Playwright/patchright instrumentation →
  `(state, action)` capture → browser-ops domain) — qalarc original; target roles:
  action selection, workflow routing, error recovery, compliance gate, completion detection

Cite as: "Built on Laya (Convai Innovations, Apache-2.0) and the Jev/System One API
(TypeSafe AI); browser loop via browser-use/jev-ultrafast (MIT). Fine-tuning protocol and
integrations by qalarc."
