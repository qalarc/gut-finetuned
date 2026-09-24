#!/usr/bin/env bash
# train_geo_after_chain.sh — waits for the in-flight 4-epoch training chain
# (launched 2026-09-22 11:44) to finish, then trains the two NEW GEO domains
# sequentially on CPU. Launched by the 2026-09-22 afternoon session.
#
# The morning chain PIDs: 3321814 (wrapper bash) + 3321815 (monitor train.py).
# It then runs gmux/osint/rfai sequentially inside that same wrapper — so we
# wait on the WRAPPER pid, then confirm no train.py remains, then start.
set -u
cd "$(dirname "$0")/.."
LV="$HOME/projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python"
LOG=scripts/train_geo_chain.log
echo "[$(date '+%F %T')] waiting for morning training chain (pid 3321814)…" >> "$LOG"
while kill -0 3321814 2>/dev/null; do sleep 120; done
# extra guard: wait until NO train.py is running
while pgrep -f 'mcp_finetune/train.py' >/dev/null 2>&1; do sleep 120; done
echo "[$(date '+%F %T')] chain finished — starting GEO fine-tunes" >> "$LOG"
for dom in geo-attribution geo-audit-triage; do
  echo "[$(date '+%F %T')] === $dom (1 epoch, CPU) ===" >> "$LOG"
  "$LV" mcp_finetune/train.py --dataset "datasets/$dom.jsonl" --epochs 1 \
    --domain "$dom" --device cpu --out "checkpoints/$dom-r1" >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] $dom exit=$?" >> "$LOG"
done
echo "[$(date '+%F %T')] GEO chain done" >> "$LOG"
