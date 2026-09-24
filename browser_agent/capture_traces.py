"""capture_traces.py — Phase A: run jev-ultrafast on real tasks with the cloud
teacher (Jev API). Records (state, decision) pairs as training data for the
browser-ops fine-tune. Each completed task contributes verified pairs.

Usage (from browser_agent/):
  PYTHONPATH=<jev-ultrafast repo> TYPESAFE_API_KEY=... ../.browser-venv/bin/python capture_traces.py

Requires: headless chromium with CDP already running:
  chromium --headless=new --remote-debugging-port=9222 \\
           --user-data-dir=$HOME/.config/chromium --no-first-run about:blank
"""

import json, os, sys, time
from pathlib import Path

# run the stock jev-ultrafast agent with recording enabled
# every (page_state, chosen_action) pair gets logged
# only completed tasks contribute training pairs (the task check verifies success)
TASKS_FILE = os.path.join(os.path.dirname(__file__), "tasks.jsonl")


def run_task(task, trace_dir: Path) -> dict:
    from jev_ultrafast.agent import Agent

    trace_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    agent = Agent(task["url"], task["goal"], record_dir=str(trace_dir))
    last = None
    try:
        for snap in agent.run():  # streams one snapshot per cycle
            last = snap
            payload = {k: v for k, v in snap.items() if k != "screenshot"}
            (trace_dir / "state.json").write_text(
                json.dumps(payload, indent=1, default=str)
            )
        # final verification text (what smoke.py does) — proves the goal state visibly
        try:
            last["verification_text"] = agent.browser.evaluate(
                "document.body.innerText"
            )[:4000]
        except Exception:
            pass
        (trace_dir / "state.json").write_text(
            json.dumps(
                {k: v for k, v in (last or {}).items() if k != "screenshot"},
                indent=1,
                default=str,
            )
        )
        status = (last or {}).get("status")
        verified = status == "done"
        (trace_dir / "summary.json").write_text(
            json.dumps(
                {
                    "ms": round((time.time() - t0) * 1000),
                    "verified": verified,
                    "status": status,
                    "decisions": len((last or {}).get("decisions", [])),
                    "actions": len((last or {}).get("history", [])),
                },
                indent=1,
            )
        )
        return {
            "status": status,
            "verified": verified,
            "decisions": len((last or {}).get("decisions", [])),
            "actions": len((last or {}).get("history", [])),
        }
    finally:
        try:
            agent.close()
        except Exception:
            pass


def main():
    tasks_file = Path(TASKS_FILE)
    if not tasks_file.exists():
        print(f"Create {TASKS_FILE} with one task per line:")
        print(
            json.dumps(
                {
                    "url": "https://example.com",
                    "goal": "Search for X and click the first result",
                },
                indent=1,
            )
        )
        return 1
    only = sys.argv[1:]  # optional indices to (re)run
    tasks = [json.loads(l) for l in open(tasks_file) if l.strip()]
    outroot = Path(__file__).resolve().parent.parent / "traces"
    outroot.mkdir(exist_ok=True)
    results = []
    for i, task in enumerate(tasks):
        if only and str(i) not in only:
            continue
        trace_dir = outroot / f"task_{i:04d}"
        print(f"task {i + 1}/{len(tasks)}: {task['goal'][:60]}…", flush=True)
        try:
            r = run_task(task, trace_dir)
            results.append((i, r))
            print(
                f"  {r['status']} verified={r['verified']} decisions={r['decisions']} "
                f"actions={r['actions']} → {trace_dir}",
                flush=True,
            )
        except Exception as e:
            results.append((i, {"status": "error", "error": str(e)}))
            print(f"  FAILED: {e}", flush=True)
    ok = sum(1 for _, r in results if r.get("verified"))
    print(f"\n{ok}/{len(results)} tasks verified-DONE; traces in {outroot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
