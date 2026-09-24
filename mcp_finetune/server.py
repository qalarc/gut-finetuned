#!/usr/bin/env python
"""Laya fine-tuning protocol — MCP server (stdio).

Agent-drivable loop: collect decision datasets → fine-tune Laya locally → calibrate →
evaluate → register the checkpoint behind laya_predict. Protocol: notes/FINETUNING_PROTOCOL.md

Tools:
  laya_status()                          checkpoints, device, registry, dataset sizes
  laya_predict(state, questions, domain) local inference (registered ckpt if any)
  dataset_add(domain, state, questions, answers, source)   append one labeled decision
  dataset_stats(domain)                  size, class balance, sources
  finetune_start(domain, epochs)         launch train.py in background (subprocess)
  finetune_status()                      running job progress / last result
  finetune_eval(domain)                  score latest checkpoint on held-out (via train.py)
  finetune_register(domain, path)        put checkpoint behind laya_predict

Run: ~/projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python server.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
CKPTS = ROOT / "checkpoints"
CKPTS.mkdir(parents=True, exist_ok=True)
DATASETS.mkdir(parents=True, exist_ok=True)
REGISTRY = CKPTS / "registry.json"
PROGRESS = ROOT / "mcp_finetune" / "train_progress.json"

# train.py needs the laya venv (torch/transformers/laya). Resolve once: env override >
# the known runner venv > whatever python is running this server (usually the venv itself).
_DEFAULT_VENV = (
    Path.home()
    / "projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python"
)
RUNNER_PY = (
    Path(os.environ["LAYA_RUNNER_PY"])
    if os.environ.get("LAYA_RUNNER_PY")
    else (_DEFAULT_VENV if _DEFAULT_VENV.exists() else Path(sys.executable))
)

mcp = MCPServer(
    name="laya-finetune",
    instructions=(
        "Laya local decision engine + fine-tuning protocol. Use laya_predict for local "
        "typed decisions (choice/score/noul) once a domain is registered; collect training "
        "data with dataset_add whenever a decision (yours or Jev's) proves good/bad; "
        "fine-tune per-domain once datasets pass ~1k rows (protocol in notes/). Never "
        "register an uncalibrated checkpoint (eval gate: teacher-agreement >=0.85, ECE <=0.10)."
    ),
)

_agents: dict = {}


def _load_registry() -> dict:
    if REGISTRY.is_file():
        return json.loads(REGISTRY.read_text())
    return {}


def _get_agent(domain: str | None):
    """Lazy-load laya. Registered domain checkpoint > base english."""
    import laya  # heavy: only on first predict

    key = domain or "base"
    if key in _agents:
        return _agents[key]
    reg = _load_registry()
    entry = reg.get(domain) if domain else None
    if entry and Path(entry["path"]).is_dir():
        agent = laya.load(entry["path"])
    else:
        agent = laya.load("convaiinnovations/laya")
    _agents[key] = agent
    return agent


@mcp.tool()
def laya_status() -> dict:
    """Local Laya status: loaded checkpoints, dataset sizes, registry, last training job."""
    reg = _load_registry()
    ds = {p.stem: sum(1 for _ in p.open()) for p in DATASETS.glob("*.jsonl")}
    job = json.loads(PROGRESS.read_text()) if PROGRESS.is_file() else None
    return {
        "loaded": list(_agents),
        "registered": reg,
        "datasets": ds,
        "last_job": job,
        "runner_venv": "typesafe_jev/laya-venv",
    }


@mcp.tool()
def laya_predict(state: str, questions: dict, domain: str | None = None) -> dict:
    """Local typed decision. questions = {name: {type, instructions, criteria?}}.
    Serves the fine-tuned checkpoint registered for `domain` if present, else base English."""
    t0 = time.time()
    agent = _get_agent(domain)
    res = agent.predict(state, questions)
    res["local_ms"] = round((time.time() - t0) * 1000, 1)
    res["domain"] = domain or "base"
    return res


@mcp.tool()
def dataset_add(
    domain: str, state: str, questions: dict, answers: dict, source: str = "agent"
) -> dict:
    """Append one labeled decision to datasets/<domain>.jsonl (the fine-tuning contract —
    see notes/FINETUNING_PROTOCOL.md §0). Log whenever a decision proves good or bad."""
    rec = {
        "state": state,
        "questions": questions,
        "answers": answers,
        "source": source,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = DATASETS / f"{domain}.jsonl"
    n = sum(1 for _ in path.open()) if path.is_file() else 0
    if n >= 100000:
        return {"error": f"dataset {domain} at cap (100k rows) — split or archive"}
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True, "domain": domain, "rows": sum(1 for _ in path.open())}


def _ans_value(ans):
    """Extract the label value from an answer cell. Answers may be: a plain string
    (choice/score), a float (noul 0-1), or a dict ({choice|score|noul: v} or
    {probabilities: {...}} soft label → argmax)."""
    if isinstance(ans, str):
        return ans
    if isinstance(ans, bool):
        return str(ans).lower()
    if isinstance(ans, (int, float)):
        return round(float(ans), 2)
    if isinstance(ans, dict):
        for k in ("choice", "score", "noul"):
            if ans.get(k) is not None:
                v = ans[k]
                return round(float(v), 2) if isinstance(v, (int, float)) else v
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and probs:
            return max(probs, key=lambda k: probs[k])
    return None


@mcp.tool()
def dataset_stats(domain: str) -> dict:
    """Row count + per-question answer balance + source mix for a dataset."""
    path = DATASETS / f"{domain}.jsonl"
    if not path.is_file():
        return {"error": f"no dataset {domain}"}
    rows = [json.loads(l) for l in path.open()]
    balance: dict = {}
    sources: dict = {}
    for r in rows:
        sources[r.get("source", "?")] = sources.get(r.get("source", "?"), 0) + 1
        for qname, ans in (r.get("answers") or {}).items():
            v = _ans_value(ans)
            if v is not None:
                balance.setdefault(qname, {}).setdefault(str(v), 0)
                balance[qname][str(v)] += 1
    return {
        "rows": len(rows),
        "answer_balance": balance,
        "sources": sources,
        "ready_for_finetune": len(rows) >= 1000,
    }


@mcp.tool()
def finetune_start(domain: str, epochs: int = 4, base: str = "english") -> dict:
    """Launch train.py for a domain dataset (background subprocess). Progress via finetune_status."""
    if PROGRESS.is_file():
        j0 = json.loads(PROGRESS.read_text())
        try:
            alive = subprocess.run(["ps", "-p", str(j0["pid"])], capture_output=True)
            if j0.get("pid") and j0["pid"] in alive.stdout.decode():
                return {"error": f"a job is already running (pid {j0['pid']}, domain {j0.get('domain')})"}
        except Exception:
            pass
    ds = DATASETS / f"{domain}.jsonl"
    if not ds.is_file():
        return {"error": f"no dataset {domain} — collect with dataset_add first"}
    out = CKPTS / f"{domain}-{time.strftime('%Y%m%d-%H%M%S')}"
    cmd = [
        str(RUNNER_PY),
        str(Path(__file__).parent / "train.py"),
        "--dataset",
        str(ds),
        "--base",
        base,
        "--epochs",
        str(epochs),
        "--out",
        str(out),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=(ROOT / "mcp_finetune" / "train.log").open("w"),
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    PROGRESS.write_text(
        json.dumps(
            {
                "domain": domain,
                "pid": proc.pid,
                "started": time.time(),
                "status": "running",
                "out": str(out),
            }
        )
    )
    return {
        "ok": True,
        "pid": proc.pid,
        "out": str(out),
        "runner_py": str(RUNNER_PY),
        "note": "train.py validates, splits, then runs the RLCD loop + calibration (protocol §2-3)",
    }


@mcp.tool()
def finetune_status() -> dict:
    """Current/last training job: running? progress rows, log tail."""
    if not PROGRESS.is_file():
        return {"status": "no job yet"}
    j = json.loads(PROGRESS.read_text())
    try:
        p = subprocess.run(["ps", "-p", str(j["pid"])], capture_output=True)
        j["running"] = j["pid"] in p.stdout.decode()
    except Exception:
        pass
    log = ROOT / "mcp_finetune" / "train.log"
    if log.is_file():
        j["log_tail"] = log.read_text()[-500:]
    return j


@mcp.tool()
def finetune_eval(domain: str) -> dict:
    """Evaluate the latest checkpoint for a domain on its held-out split (delegates to
    train.py --eval). Gate: teacher-agreement >=0.85, ECE <=0.10 before register."""
    ck = sorted(CKPTS.glob(f"{domain}-*"), key=lambda p: p.name)
    if not ck:
        return {"error": "no checkpoint — finetune_start first"}
    r = subprocess.run(
        [
            str(RUNNER_PY),
            str(Path(__file__).parent / "train.py"),
            "--eval",
            "--dataset",
            str(DATASETS / f"{domain}.jsonl"),
            "--ckpt",
            str(ck[-1]),
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    return {
        "checkpoint": str(ck[-1]),
        "output": r.stdout[-800:],
        "stderr": r.stderr[-300:],
    }


@mcp.tool()
def finetune_register(domain: str, path: str) -> dict:
    """Register a fine-tuned+calibrated checkpoint for a domain (served by laya_predict).
    Only after finetune_eval passes the gate."""
    p = Path(path)
    if not p.is_dir():
        return {"error": f"{path} is not a checkpoint dir"}
    reg = _load_registry()
    reg[domain] = {"path": str(p), "registered": time.strftime("%Y-%m-%d %H:%M"),
                   "gate": "NOT VERIFIED — run finetune_eval first (protocol: agreement>=0.85, ECE<=0.10)"}
    REGISTRY.write_text(json.dumps(reg, indent=2))
    _agents.pop(domain, None)
    return {"ok": True, "domain": domain, "served_by": "laya_predict"}


if __name__ == "__main__":
    import asyncio

    asyncio.run(mcp.run_stdio_async())
