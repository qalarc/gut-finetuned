#!/usr/bin/env python
"""kaggle_watch.py — poll qalarc/laya-domains until COMPLETE, then pull the outputs
(checkpoints + gate tables) into checkpoints/kaggle/ and print the verdicts.

Run: setsid nohup laya-venv/bin/python scripts/kaggle_watch.py > scripts/kaggle_watch.log 2>&1 &
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "qalarc/chanalyse-v2"
OUT = ROOT / "checkpoints" / "kaggle"
POLL = 300  # 5 min


def status():
    r = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL], capture_output=True, text=True
    )
    return (r.stdout + r.stderr).strip()


def main():
    print(f"watching {KERNEL} (poll {POLL}s)", flush=True)
    while True:
        s = status()
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
        if "COMPLETE" in s:
            break
        if "ERROR" in s or "CANCEL" in s:
            print(
                "kernel FAILED — pull logs:",
                subprocess.run(
                    ["kaggle", "kernels", "output", KERNEL, "-p", "/tmp/kaggle_fail"],
                    capture_output=True,
                    text=True,
                ).stdout,
                flush=True,
            )
            sys.exit(1)
        time.sleep(POLL)

    OUT.mkdir(parents=True, exist_ok=True)
    print("COMPLETE — downloading output…", flush=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(OUT)], check=True)
    print("\n================ GATE VERDICTS ================", flush=True)
    for ej in sorted(OUT.glob("*-ckpt/eval.json")):
        try:
            res = json.loads(ej.read_text())
            print(
                f"{ej.parent.name}: agreement={res['teacher_argmax_agreement']} "
                f"ece={res['ece_post_temperature']} gate={res['gate']}",
                flush=True,
            )
            if res["gate"] == "PASS":
                print(
                    f"  → PASS: eligible for finetune_register "
                    f"(copy to checkpoints/{ej.parent.name} + registry)",
                    flush=True,
                )
        except Exception as e:
            print(f"{ej}: unreadable ({e})", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
