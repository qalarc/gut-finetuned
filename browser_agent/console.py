#!/usr/bin/env python
"""console.py — the Browser Ops Console: a working interface for autonomous browser
navigation, powered by the typed-decision stack.

  laya-venv/bin/python browser_agent/console.py [--port 8799]

Open http://127.0.0.1:8799 → enter a seed URL + page budget → watch every decision
stream in: the link the decider picked, its top probabilities, latency, and WHO decided
(jev-cloud today; the fine-tuned Laya tomorrow via the LAYA_URL field — the same
wire protocol, one field swap). Every decision is a browser-ops training row.

Layout: FastAPI + a single embedded page. The explorer runs in a worker thread and
pushes events into an in-memory feed.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore as explorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()
_feed: list[dict] = []  # every event, indexed
_state = {
    "running": False,
    "stop": False,
    "seed": None,
    "started": None,
    "decider": None,
}

app = FastAPI(title="Browser Ops Console")

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Browser Ops Console</title>
<style>
*{box-sizing:border-box;margin:0}
body{background:#0a0e14;color:#e6e9ef;font-family:'Segoe UI',system-ui,sans-serif;padding:26px;max-width:1100px;margin:0 auto}
h1{font-size:21px;letter-spacing:.04em;color:#5b8cff;margin-bottom:4px}
.sub{color:#5b6478;font-size:13px;margin-bottom:18px}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
input{background:#111722;border:1px solid #1d2635;color:#e6e9ef;border-radius:8px;padding:10px 12px;font-size:14px}
#url{flex:1;min-width:260px}#pages{width:80px}#laya{width:300px}
button{background:#5b8cff;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-weight:700;cursor:pointer;font-size:14px}
button.stop{background:#ff6b6b}button:disabled{opacity:.4;cursor:default}
.stat{display:inline-block;margin-right:22px;color:#8a93a6;font-size:13px}
.stat b{color:#e6e9ef;font-size:17px;font-family:monospace}
.decision{background:#111722;border:1px solid #1d2635;border-left:3px solid #5b8cff;border-radius:8px;padding:10px 14px;margin-bottom:8px;animation:in .25s ease}
.decision.laya{border-left-color:#3ecf8e}
@keyframes in{from{opacity:0;transform:translateY(-4px)}to{opacity:1}}
.decision .to{color:#5b8cff;font-weight:600;text-decoration:none}
.decision .meta{color:#5b6478;font-size:12px;margin-top:3px}
.badge{float:right;font-size:11px;padding:2px 10px;border-radius:999px;background:rgba(91,140,255,.15);color:#5b8cff;font-weight:700}
.badge.laya{background:rgba(62,207,142,.15);color:#3ecf8e}
.prob{display:inline-block;margin-left:8px;font-size:12px;color:#8a93a6;font-family:monospace}
.top{color:#3ecf8e}
.pagebar{background:#111722;border:1px solid #1d2635;border-radius:8px;padding:7px 14px;margin-bottom:10px;color:#8a93a6;font-size:13px}
.pagebar b{color:#e6e9ef}
</style></head><body>
<h1>BROWSER OPS CONSOLE</h1>
<div class="sub">autonomous site navigation · every hop is a typed decision (and a training row) · decider: jev-cloud ⇄ laya-local by one field</div>
<div class="row">
  <input id="url" placeholder="seed URL, e.g. https://tradez.au" value="https://tradez.au">
  <input id="pages" type="number" value="15" min="1" max="60" title="page budget">
  <input id="laya" placeholder="LAYA_URL — leave empty for Jev cloud (e.g. http://127.0.0.1:8798/v1/systemone)">
  <button id="go" onclick="start()">Explore</button>
  <button id="stopb" class="stop" onclick="stop()" disabled>Stop</button>
</div>
<div>
  <span class="stat">pages <b id="np">0</b></span>
  <span class="stat">decisions <b id="nd">0</b></span>
  <span class="stat">decider <b id="dc">—</b></span>
  <span class="stat">avg decision <b id="ms">—</b></span>
</div>
<div id="feed" style="margin-top:16px"></div>
<script>
let idx = 0, timer = null;
async function post(path, body){ return (await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body||{})})).json(); }
async function start(){
  const r = await post('/api/explore', {url: document.getElementById('url').value,
    pages: parseInt(document.getElementById('pages').value)||15,
    laya_url: document.getElementById('laya').value || null});
  if(!r.ok){ alert(r.error); return; }
  document.getElementById('go').disabled = true; document.getElementById('stopb').disabled = false;
  idx = 0; timer = setInterval(poll, 900);
}
async function stop(){ await post('/api/stop'); }
async function poll(){
  const r = await (await fetch('/api/feed?since=' + idx)).json();
  const f = document.getElementById('feed');
  for(const ev of r.events){
    idx = ev.i + 1;
    if(ev.type === 'page'){
      f.insertAdjacentHTML('afterbegin', `<div class="pagebar">page <b>${ev.n}/${ev.max}</b> · <b>${esc(ev.title||ev.url)}</b> · ${ev.links} links harvested ${ev.via? '· via <b>'+esc(ev.via)+'</b>':''}</div>`);
      document.getElementById('np').textContent = ev.n;
    } else if(ev.type === 'decision'){
      const probs = ev.top.map(([k,v],i)=>`<span class="prob ${i===0?'top':''}">${esc(k)} ${(v*100).toFixed(0)}%</span>`).join('');
      f.insertAdjacentHTML('afterbegin', `<div class="decision ${ev.decider.startsWith('laya')?'laya':''}"><span class="badge ${ev.decider.startsWith('laya')?'laya':''}">${ev.decider}</span>
        → <a class="to" href="${esc(ev.to)}" target="_blank">${esc(ev.label)}</a>${probs}
        <div class="meta">from ${esc(ev.from)} · decided in ${ev.ms} ms · saved as training row</div></div>`);
      const nd = document.getElementById('nd'); nd.textContent = parseInt(nd.textContent)+1;
      document.getElementById('ms').textContent = ev.ms + ' ms';
      document.getElementById('dc').textContent = ev.decider;
    } else if(ev.type === 'done' || ev.type === 'stopped'){
      f.insertAdjacentHTML('afterbegin', `<div class="pagebar"><b>${ev.type === 'done' ? 'walk complete' : 'stopped'}</b> — ${ev.pages ?? ev.n ?? 0} pages · frontier ${ev.frontier ?? '—'} · rows appended to datasets/browser-ops.jsonl</div>`);
      clearInterval(timer);
      document.getElementById('go').disabled = false; document.getElementById('stopb').disabled = true;
    }
  }
}
function esc(s){ const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/api/explore")
def api_explore(body: dict):
    with _lock:
        if _state["running"]:
            return {"ok": False, "error": "a walk is already running"}
        seed = (body.get("url") or "").strip()
        if not seed.startswith("http"):
            return {"ok": False, "error": "seed must be an http(s) URL"}
        pages = max(1, min(60, int(body.get("pages") or 15)))
        if body.get("laya_url"):
            os.environ["LAYA_URL"] = body["laya_url"]
        else:
            os.environ.pop("LAYA_URL", None)
        _state.update(running=True, stop=False, seed=seed, started=time.time())
    try:
        _load_key()
    except Exception:
        pass

    def worker():
        def on_event(ev):
            with _lock:
                _feed.append(ev)

        def should_stop():
            return _state["stop"]

        try:
            explorer.explore(seed, pages, on_event=on_event, should_stop=should_stop)
        except Exception as e:
            with _lock:
                _feed.append({"type": "done", "error": str(e)})
        finally:
            with _lock:
                _state["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
def api_stop():
    _state["stop"] = True
    with _lock:
        if _state["running"]:
            _feed.append(
                {"type": "stopped", "n": sum(1 for e in _feed if e["type"] == "page")}
            )
    return {"ok": True}


@app.get("/api/feed")
def api_feed(since: int = 0):
    with _lock:
        events = [{**e, "i": i} for i, e in enumerate(_feed) if i >= since]
        return {"events": events, "running": _state["running"]}


def _load_key():
    env = Path.home() / ".secrets/typesafe.env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    _load_key()
    uvicorn.run(app, host="127.0.0.1", port=ap.parse_args().port, log_level="warning")
