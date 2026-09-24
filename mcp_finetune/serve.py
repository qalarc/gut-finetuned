#!/usr/bin/env python
"""serve.py — LOCAL Jev-API-compatible endpoint backed by Laya.

Speaks TypeSafe's wire protocol (POST /v1/systemone, same request/response schema),
so anything configured for Jev (gmux_v7 router, jev-ultrafast, our own integrations)
switches to local by changing ONE base URL. Domain routing: optional ?domain= or
X-Laya-Domain header picks the fine-tuned checkpoint; default serves registry/base.

Run: laya-venv/bin/python serve.py [--port 17620]

Binds 127.0.0.1 only. Endpoints:
  GET  /health         — {"ok": true, "loaded": [...], "registered": {...}}
  POST /v1/systemone   — {state, model?, questions} + X-Laya-Domain header
  POST /predict        — TAF scheduler alias: {domain|checkpoint, state,
                         questions} → {answers: {direction: {choice,
                         probabilities}}, checkpoint, ...}
The /predict default port matches taf_scheduler's TAF_LAYA_BASE default
(http://127.0.0.1:17620 — LAYA_KELLY_PLAN.md).
Test: curl -s localhost:8798/v1/systemone -d '{"state":"...","questions":{...}}'
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import threading

from fastapi import FastAPI, Header
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "checkpoints" / "registry.json"

# no CORS middleware on purpose: localhost service; same-origin policy blocks
# drive-by browser calls, local processes are unaffected (review 2026-09-22)
app = FastAPI(title="laya-systemone", version="1.0.1")

_agents: dict = {}
_load_lock = threading.Lock()  # review 2026-09-22: prevents double model-load race
_registry_mtime: float | None = None
_stats = {"calls": 0, "total_ms": 0.0}


def _registry_changed() -> bool:
    """True if checkpoints/registry.json changed since we last loaded an agent —
    lets finetune_register invalidate serve.py's cache without a restart."""
    global _registry_mtime
    m = REGISTRY.stat().st_mtime if REGISTRY.is_file() else 0.0
    if _registry_mtime is None:
        _registry_mtime = m
        return False
    return m != _registry_mtime


def _agent(domain: str):
    with _load_lock:
        return _agent_locked(domain)


def _agent_locked(domain: str):
    global _registry_mtime
    key = domain or "base"
    if key in _agents and not _registry_changed():
        return _agents[key]
    _registry_mtime = REGISTRY.stat().st_mtime if REGISTRY.is_file() else 0.0
    import laya

    entry = None
    if domain and REGISTRY.is_file():
        reg = json.loads(REGISTRY.read_text())
        entry = reg.get(domain)
    if entry and Path(entry["path"]).is_dir():
        a = laya.load(entry["path"])
    else:
        a = laya.load("convaiinnovations/laya")
    _agents[key] = a
    return a


class Question(BaseModel):
    type: str
    instructions: str
    criteria: dict | list | None = None


class SystemOneRequest(BaseModel):
    state: str | dict
    model: str | None = None
    questions: dict[str, Question]


@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "laya-local",
        "loaded": list(_agents),
        "registered": json.loads(REGISTRY.read_text()) if REGISTRY.is_file() else {},
        "calls": _stats["calls"],
        "avg_ms": round(_stats["total_ms"] / max(1, _stats["calls"]), 1),
    }


@app.post("/v1/systemone")
def system_one(req: SystemOneRequest, x_laya_domain: str | None = Header(default=None)):
    t0 = time.time()
    a = _agent(x_laya_domain or "")
    res = a.predict(
        req.state if isinstance(req.state, str) else json.dumps(req.state),
        {k: v.model_dump(exclude_none=True) for k, v in req.questions.items()},
    )
    elapsed = (time.time() - t0) * 1000
    _stats["calls"] += 1
    _stats["total_ms"] += elapsed
    # Audit log (review 2026-09-22 rec): every served call → JSONL. These are candidate
    # training rows once a human/teacher confirms them; traces/*.jsonl is gitignored.
    try:
        with (ROOT / "traces" / "serve_calls.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "domain": x_laya_domain or "base",
                        "state": req.state,
                        "questions": {
                            k: v.model_dump(exclude_none=True)
                            for k, v in req.questions.items()
                        },
                        "answers": res.get("answers", {}),
                        "local_ms": round(elapsed, 1),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass
    out = {
        "model": f"laya-local-{x_laya_domain or 'base'}",
        "answers": res.get("answers", {}),
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "local_ms": round(elapsed, 1),
        },
    }
    return out


class PredictRequest(BaseModel):
    """Body used by the TAF scheduler's job_laya_daily (LAYA_KELLY_PLAN.md).

    `domain` selects the registered checkpoint; `checkpoint` is an alias for
    the same thing (registry keys double as checkpoint ids).
    """

    domain: str | None = None
    checkpoint: str | None = None
    state: str | dict
    questions: dict[str, Question]


@app.post("/predict")
def predict_taf(req: PredictRequest):
    """TAF-facing alias of /v1/systemone.

    Same inference path, body and response shaped for the Rust scheduler:
    {answers: {direction: {choice, probabilities, ...}}, ...}. The domain is
    taken from the body (checkpoint id == registry domain key).
    """
    domain = req.domain or req.checkpoint or ""
    inner = SystemOneRequest(
        state=req.state,
        model=req.checkpoint,
        questions=req.questions,
    )
    out = system_one(inner, x_laya_domain=domain)
    out["checkpoint"] = domain
    return out


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    # Default 17620: the port taf_scheduler's job_laya_daily calls
    # (TAF_LAYA_BASE default http://127.0.0.1:17620).
    ap.add_argument("--port", type=int, default=17620)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    # LAYA_PRELOAD=1 (review 2026-09-22 rec): warm the base agent at startup so the
    # first real call doesn't pay the ~20-25s model load (MCP client timeouts).
    if os.environ.get("LAYA_PRELOAD"):
        import threading

        threading.Thread(target=lambda: _agent(""), daemon=True).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
