# TRAINING AREAS — handoff to Tulpa Studio

**Date:** 2026-09-22 · **Owner of training from here:** Tulpa Studio
**Repo:** `~/projects/GLM_projects/Laya_integrations` (github.com/qalarc/laya-integrations)
**Scope split:** this repo owns datasets + protocol + serving. Tulpa Studio owns WHEN and
WHAT gets trained. Everything Tulpa needs to run a fine-tune is in this file.

---

## 0. The 60-second context

Jev (TypeSafe cloud) decides zero-shot; every Jev verdict can be logged as a training row
(soft probability distributions). Laya (open 421M local model) is fine-tuned per domain on
those rows and serves the SAME wire protocol locally (`POST :8798/v1/systemone`). Training
was wired this session (`mcp_finetune/train.py`): GRPO-style RLCD with proper-scoring-rule
rewards + soft cross-entropy, per-(qtype, option-bucket) temperature calibration on a
held-out split, and a hard eval gate. **Tulpa already ran it** (geo-attribution-r1,
geo-audit-triage-r1) — the runner works headless.

## 1. Entry points (pick either)

```bash
# A) MCP tools (registered as `laya-finetune` in qalcode2 + Claude Code):
#    finetune_start(domain, epochs=4, base="english")  → background run + progress
#    finetune_status() / finetune_eval(domain) / finetune_register(domain, path)

# B) CLI (what the MCP wraps):
LV=~/projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python
$LV mcp_finetune/train.py --dataset datasets/<domain>.jsonl --epochs 4 \
    --device cpu --out checkpoints/<domain>-<tag>          # trains + auto-gates
# eval an existing ckpt:  $LV train.py --eval --dataset datasets/<d>.jsonl --ckpt <dir>

# register ONLY on gate PASS (writes checkpoints/registry.json; serve.py hot-reloads it):
#   finetune_register("<domain>", "checkpoints/<dir>")
```

## 2. Hardware reality (measured 2026-09-22) — READ BEFORE LAUNCHING

- **CPU only.** venv torch is `2.14.0+cu130` — `torch.version.hip` is None, so the 8060S
  iGPU + ROCm stack are unusable to torch. GPU training needs a ROCm wheel swap first.
- One run ≈ **10 GB RSS** and eats all cores (torch uses ~14 threads).
  micro-batch 16 ≈ 6.5 s/step; monitor-triage (2.5k items × 4 epochs) ≈ 66 min.
- **The box is in a RAM crunch** (99/124 G used, 67 G swap, load ~29 while training ran).
  Rules of thumb: **one training at a time**; `--micro-batch 8` halves peak activation
  memory; consider `OMP_NUM_THREADS=8` to leave cores for the rest of the stack.
- tmux-pane processes run at `oom_score_adj=200` — under memory pressure YOUR run dies
  first, by design. If a kill happens: run `oom-report` and read the plain-English block;
  it names what died and which unit tipped the machine over. Rolling checkpoint
  (`checkpoint_latest/`) survives per epoch, so a kill loses ≤1 epoch.
- Best window: train when the fleet is quiet; prefer many small runs over one long one.

## 3. THE GATE (non-negotiable — it's the whole product)

Every run ends with a gate table on the held-out split (`eval.json` inside the ckpt dir):
- argmax agreement with the Jev teacher **≥ 0.85**
- ECE post-temperature **≤ 0.10**
- no label receiving <5% probability on >20% of examples (crash buckets)

**PASS → `finetune_register`. FAIL → more data, never register.** An uncalibrated
confident local model is worse than a cloud call — this rule has no exceptions.

**Current honest state (all 7 runs so far): agreement is already there, calibration is not.**

| domain | rows | agreement | ECE | crash buckets | gate |
|---|---|---|---|---|---|
| monitor-triage | 1,215 | **0.9323** ✓ | 0.3971 ✗ | 188/189 ✗ | FAIL |
| gmux-routing r1 | 606 (broken criteria) | **0.9083** ✓ | 0.3682 ✗ | 109/120 ✗ | FAIL |
| **gmux-routing r2 (clean data)** | 606 canonical | **0.9458** ✓ | 0.5040 ✗ | 117/120 ✗ | FAIL |
| osint-grading | 604 | **0.9167** ✓ | 0.6549 ✗ | 29/60 ✗ | FAIL |
| rfai-feed | 402 | **0.9625** ✓ | 0.4592 ✗ | 40/40 ✗ | FAIL |
| geo-attribution | 300 | **0.9167** ✓ | 0.5885 ✗ | 30/30 ✗ | FAIL |
| geo-audit-triage r2 | 309 | **0.9333** ✓ | 0.5934 ✗ | — ✗ | FAIL |

**gmux r2 lesson (2026-09-23):** fixing the 3-vs-4-option criteria mismatch (the dataset
had trained a different question than the router serves — see `scripts/normalize_gmux.py`)
lifted agreement (+0.04) but NOT calibration. At ~600 rows / ~60 held-out questions the
probability heads are still mushy and the ECE estimate itself is noisy. **Volume is the
binding constraint**: get each domain to 1.5–3k canonical rows (flywheels + bulk generator,
which now emits the exact serving shapes) and expect ECE readings to stabilize as held-out
grows past ~150 questions.

## 4. The training areas (all of them, lined up)

Format per area: purpose · questions · state · collection · now → next. Dataset contract
for every row (`notes/FINETUNING_PROTOCOL.md` §0):
`{"state": str, "questions": {name: {type, instructions, criteria?}}, "answers": {name:
<str | float | {"probabilities": {...}}}, "source": "...", "ts": "..."}` — **soft
`probabilities` labels are preferred** (that's what Jev returns; they train best).
`dataset_add` MCP appends one row; bulk generators live in `scripts/jev_shadow_label.py`.

### 4.1 monitor-triage — 1,215 rows · flywheels LIVE · closest to gate-ready
- **Decides:** alert severity (choice low/medium/high/critical), wake-the-owner-3am (noul),
  likely cause (choice origin_down/network/tls/upstream); plus contact-form intent.
- **Collection:** `scripts/monitor_shadow.py` labels every new concierge monitor event via
  Jev (reads hub.db READ-ONLY; `--once`/`--loop 300`/`--backfill N`). Systemd units in
  `scripts/systemd/` (not enabled — enable manually). Bulk: `--domain monitor-triage`.
- **Now → next:** at 1.2k rows agreement is 0.93. Push to ~3k (enable the timer + one bulk
  batch of 1k), re-run. First registration candidate — it's also the highest-value swap
  (alerts flow 24/7, and local = no alert text leaves home).

### 4.2 osint-grading — 604 rows · needs the post-scan hook wired
- **Decides:** evidence tier per found hit (choice verified/likely/weak/junk) — honest
  hedging matters here; Jev's own soft labels are the point.
- **Collection:** after every OSINT-hub scan, grade each hit with `jev_decide` →
  `dataset_add("osint-grading", ..., source="jev-live")`. Bulk: `--domain osint-grading`
  (synthetic sweeps). Later: user keep/discard = outcome labels (protocol §1.2).
- **Now → next:** hub is in lite mode (cloudcheck only) — wire the hook in the hub's scan
  backend so every real scan feeds rows. Target 2k.

### 4.3 gmux-routing — 606 rows · flywheel LIVE · Phase-B flip pending gate
- **Decides:** task complexity (choice) + risk (choice readonly/local-writes/destructive/
  external-side-effects) + needs_reasoning (noul) + context_size (score) → dispatch tier.
- **Collection:** `crates/gmux-router` (gmux_v7) logs every RoutingDecision JSONL →
  `scripts/gmux_collect.py` ingests into the dataset. Currently Phase A (Jev cloud);
  **when this domain PASSes the gate and registers, gmux flips to local with one URL**
  (`GMUX_SOURCE=laya` / `HttpSource::laya_local("gmux-routing")`).
- **Now → next:** 600 → 2k rows (the router log grows free with usage), re-run.

### 4.4 rfai-feed — 402 rows · schema ready for the real feed
- **Decides:** RF transmission class (choice sensor/keyfob/voice/noise) + operator-interest
  (noul).
- **Collection:** bulk generator now; extend with the real feature-vector fields the RFAI
  feed produces (freq/band/pattern/dBm/interval are already in the state text).
- **Now → next:** 400 → 1.5k. (Sister domain `rfai-intent` below is further along.)

### 4.5 rfai-intent — 477 rows · tulpa-created
- **Decides:** transmission intent (choice, e.g. navigation) + urgent (noul) for
  aviation-style radio calls ("Darwin Approach… request ILS runway 12").
- **Next:** fill missing `source` fields (rows exist with source=None — set
  `jev-shadow`/`human` provenance), grow to ~1.5k, train.

### 4.6 geo-attribution — 300 rows · tulpa-created · r1 trained
- **Decides:** AI-engine citation attribution for qalarc-brand answers (choice
  correct/wrong-person/no-citation/wrong-facts) + escalate (noul). Feeds the GEO
  monitoring loop ("what does ChatGPT Search say about our brands, and is it correct?").
- **Next:** grow to ~1.5k (one Jev-shadow batch over the monitored query set), re-run.

### 4.7 geo-audit-triage — 309 rows · tulpa-created · r1 TRAINING NOW (as of 14:25)
- **Decides:** GEO audit finding priority (choice p0/p1/p2/skip) + fixnow (noul).
- **Next:** same recipe — batch to ~1.5k, re-run. Pairs with 4.8 as the search-visibility
  stack.

### 4.8 seo-grading — NEW, not seeded yet · spec'd here so Tulpa can start it
The JUDGING layer of SEO (crawling/parsing stays with the scanner; this is the verdict
tier — see docs/EVALUATION.md's SEO row). Cousins with geo-audit-triage (4.7): same stack,
different object (human-search vs AI-search).
- **Questions (atomic, combinable):**
  - `page_quality` — score 0–4: thin/stubbed → comprehensive, well-structured, fresh
  - `title_meta` — score 0–3: title + meta description quality vs the target query
  - `intent_match` — choice: matches / partial / mismatched (page vs target query intent)
  - `cannibalization` — choice none/possible/likely (vs the named sibling page)
  - `change_significant` — noul (monitor diff: meaningful content change vs
    analytics-noise/template churn) — reuse monitor-triage plumbing for diffs
- **State:** compact page snapshot: url · title · meta · h1 · word count · headings ·
  last-modified · current position for target query · sibling url+summary (for
  cannibalization). PII-free by construction.
- **Collection:** Jev-shadow bulk first (extend `jev_shadow_label.py` with a
  `seo-grading` generator over the qalarc site portfolio — real pages, real queries), then
  scanner-produced audits + monitor diffs as live flywheels.
- **Target:** 1.5–2.5k rows → first fine-tune. High volume potential (every page × every
  monitored query).

### 4.8b compliance-marks — 810 rows · NEW (2026-09-23 evening)
- **Decides:** library compliance for Doof.ing catalog entries: keep (choice keep/review/
  remove) + canonical (noul) + on-theme (choice 3). Source: real catalog
  (`scripts/export_library_marks.py`, includes community votes in-state as future outcome
  labels) + Jev-shadow variations to gate-worthy volume.
- **Next:** user's browser localStorage dump (`libraryVotes`) becomes outcome labels
  (export command in the script docstring); retrain at r1 to set the baseline.

### 4.10 browser-ops — live flywheel (explorer + task runs + Playwright instrumentation)
- Runs jev-ultrafast (browser agent) stock on real tasks; traces convert losslessly:
  `scripts/traces_to_dataset.py <artifact-dirs>` (keeps verified-DONE runs only).
- **Volume need:** 5–10k verified cycles before training makes sense — schedule LAST,
  after the cheap domains pass. GPU (ROCm) practically required for the 33ms-class serving
  this domain exists for.

## 5. Priority order (recommendation)

1. monitor-triage → first registration (data flywheel already running, highest value)
2. **seo-grading (1,200 rows, 2026-09-23)** — first r1 training candidate among the new areas
3. **compliance-marks (810 rows)** + localStorage votes as outcome labels → r1
4. geo-audit-triage + geo-attribution (finish batches to ~1.5k)
5. gmux-routing (clean data r2 done — agree 0.946; grows free; Phase B on PASS)
6. browser-ops (explorer + task runs collecting now; train r1 at ~300+ rows)
7. rfai-intent / rfai-feed / osint-grading (hook wiring needed)

## 6. Rules that bind whoever trains

- **The gate is law.** No `finetune_register` without PASS. No fake numbers.
- Datasets + checkpoints are qalarc IP; never commit `checkpoints/` bulk or traces;
  datasets stay in `datasets/`.
- Privacy domains (phone/message/RF states, monitor text): local serving only; Jev may
  TEACH on them but states must stay PII-stripped where feasible.
- Attribution stays accurate in anything public (`ATTRIBUTION.md`).
- Every training run leaves: `train_meta.json`, `eval.json` (the gate table), rolling
  `checkpoint_latest/`. Keep them — they're the audit trail.

## 7. Quick-start for Tulpa

```bash
cd ~/projects/GLM_projects/Laya_integrations
LV=~/projects/GLM_projects/investigation/typesafe_jev/laya-venv/bin/python
# grow a domain (example: monitor-triage +500 more rows, ~2 min, pennies):
$LV scripts/jev_shadow_label.py --domain monitor-triage --n 500 --workers 4
# train it (one at a time; box is RAM-tight — see §2):
$LV mcp_finetune/train.py --dataset datasets/monitor-triage.jsonl --epochs 4 \
    --micro-batch 8 --device cpu --out checkpoints/monitor-triage-r2
# read the gate table:
cat checkpoints/monitor-triage-r2/eval.json
# PASS? register (then serve.py + laya_predict pick it up via X-Laya-Domain/domain):
#   MCP: finetune_register("monitor-triage", "checkpoints/monitor-triage-r2")
```
