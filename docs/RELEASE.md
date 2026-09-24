# RELEASE — GUT (Guided Unconscious Thinking): a fine-tuned decision engine trained on qalarc's own operational data

**Status: program release (v0.9).** The pipeline, protocol, router, and datasets are
complete and published; per-domain checkpoints ship as they pass the hard eval gate
(zero passes at release time — see Gate status). Nothing is registered that hasn't passed.

---

## What we're releasing

A complete, reproducible pipeline that takes qalarc's live operational decisions — the
verdicts our agents, monitors, crawlers, and routers make every day — and distills them
into a 421M-parameter local model (Laya, Apache-2.0) that serves the same wire protocol
as our cloud decision engine (TypeSafe Jev). Result: the same verdicts, on-device,
~33 ms, $0 marginal, with the state text never leaving home.

```
qalarc operations ──► Jev (cloud teacher, soft probability labels)
        │                        │
        │                        ▼
        │              datasets/*.jsonl  (10 domains, ~5,000 rows, all soft-labeled)
        │                        │
        │                        ▼
        │              RLCD fine-tune (GRPO-style, proper scoring rule)
        │                        │
        │                        ▼
        │              per-(question-type, option-bucket) temperature calibration
        │                        │
        │                        ▼
        │              HARD GATE on held-out: teacher-agreement ≥ 0.85,
        │              calibration error (ECE) ≤ 0.10, no crash buckets
        │                        │  (no pass → no register. no exceptions.)
        ▼                        ▼
local endpoint :8798  ◄──  checkpoints/registry.json (hot-reloaded)
same wire protocol — gmux-router / browser agents switch cloud→local with ONE URL
```

## Methodology — how it was trained on qalarc data

Every qalarc system that makes decisions now teaches while it works ("Jev-shadow"):
when Jev grades a hit, triages an alert, routes an agent task, or steers the browser, the
full typed decision (state, question set, **soft probability distribution**) is logged as
a training row. Soft labels are kept deliberately — training against the teacher's full
distribution preserves its hedging, which a hard argmax would destroy.

**Provenance — where the training data comes from (all qalarc systems):**

| Source system | What it contributed |
|---|---|
| **qal-monitor** (concierge) | live alert transitions: severity, wake-the-owner, likely-cause verdicts (`monitor_shadow.py` reads the event feed read-only) |
| **OSINT hub** (cloudcheck/maigret scans) | evidence-tier grading of found accounts: verified / likely / weak / junk |
| **gmux-router** (agent-fleet dispatch) | task complexity, risk, reasoning-need, context-size → tier routing decisions (Phase-A cloud decisions logged fleet-wide) |
| **chanalyse / GEO pipeline** (AI-search monitoring) | AI-engine citation attribution + audit-finding triage for qalarc brands |
| **RFAI feed** (RF monitoring) | transmission classification (sensor/keyfob/voice/noise) + operator-interest |
| **jev-ultrafast browser agent** | real browser sessions: operation + target decisions per cycle (`traces_to_dataset.py`, verified-DONE runs only) |
| **autonomous explorer** (`explore.py`) | frontier crawls where the decider picks each next link — exploration is data collection |
| **Doof.ing library** | compliance marks: catalog entries + community votes, graded for keep/canonical/on-theme (`export_library_marks.py`) |
| **SEO audits** (scanner + monitor diffs) | page quality, title/meta, intent-match, cannibalization, change-significance |
| **Jev-shadow generators** (`jev_shadow_label.py`) | synthetic-but-realistic states per domain, soft-labeled by Jev — used to reach gate-worthy volume; always labeled with the same teacher as production |

No customer PII is used as training state. Sensitive domains (alerts, RF, library) serve
locally only.

## The training loop (what "RLCD" means here)

1. **Per-question isolation** — one training item per (state, question); choice/score/noul
   each become a small classification head over rendered options (Laya's typed decisions).
2. **GRPO-style sampling** — G=4 noisy logit perturbations per item (σ annealed 0.4→0.1,
   zero-mean projected over options), scored by a **proper scoring rule**
   (log + spherical + ranked probability score), group-normalized to advantages.
3. **Policy gradient + soft cross-entropy guidance** — the RL signal moves probability
   mass toward teacher-preferred options while a full soft-CE term anchors the heads to
   the teacher's distributions.
4. **Calibration is part of training, not an afterthought** — per-(question-type,
   option-bucket) LBFGS temperature fitting on the held-out split, clamped to honest ranges.
5. **The gate is law** — agreement with the Jev teacher ≥ 0.85, ECE ≤ 0.10, no collapsed
   labels, evaluated on a row-level held-out split (no state leakage). Checkpoints that
   fail are kept as audit trail and never registered. An uncalibrated confident local
   model is worse than a cloud call.

## THE TRAINING POINTS — every decision the model learns

| # | Domain | Decision heads (questions) | Serves |
|---|---|---|---|
| 1 | **monitor-triage** | severity (choice 4) · wake-the-owner-3am (truthy) · likely-cause (choice 4) · contact-intent (choice 3) | alert pipeline, contact forms |
| 2 | **osint-grading** | evidence tier (choice 4: verified/likely/weak/junk) | OSINT hub hit triage |
| 3 | **gmux-routing** | complexity (choice 4) · risk (choice 4) · needs-reasoning (truthy) · context-size (score 3) | agent-fleet dispatch tier (local-ornith / glm-flash / glm-53 / jev-only / human) |
| 4 | **rfai-feed** | transmission class (choice 4) · operator-interest (truthy) | RF monitoring feed |
| 5 | **rfai-intent** | intent (choice) · urgent (truthy) | aviation-style radio call triage |
| 6 | **geo-attribution** | citation correctness (choice 4) · escalate (truthy) | AI-search brand monitoring |
| 7 | **geo-audit-triage** | finding priority (choice 4: p0/p1/p2/skip) · fix-now (truthy) | GEO audit backlog |
| 8 | **seo-grading** | page-quality (score 5) · title/meta (score 4) · intent-match (choice 3) · cannibalization (choice 3) · change-significant (truthy) | SEO audits + monitor diffs |
| 9 | **compliance-marks** | keep (choice 3) · canonical (truthy) · on-theme (choice 3) | Doof.ing library moderation |
| 10 | **browser-ops** | operation (choice 5: CLICK/TYPE_TEXT/SELECT/NAVIGATE/WAIT) + per-operation target head over the element/link table | browser agent System-1: action selection, workflow routing, error recovery, compliance gate, completion detection |

## Capabilities this unlocks

- **Local decision serving** — `POST :8798/v1/systemone`, Jev wire-compatible; anything
  built for the cloud API flips with one URL (`X-Laya-Domain` header picks the checkpoint).
- **One-URL fleet switch** — gmux-router (Rust, 12/12 tests) runs Jev cloud in Phase A
  (logging decisions), flips to local in Phase B, and falls back to a deterministic
  heuristic when offline (`GMUX_OFFLINE=1`) — never cloud-dependent.
- **Autonomous browser exploration** — `explore.py` maps sites end-to-end with a decider
  steering each hop; the same loop is the browser-ops flywheel today and the fine-tuned
  model's runtime tomorrow (swap = `LAYA_URL` env or `laya_source.py` in jev-ultrafast).
- **The 421M System-1 / 30B+ System-2 split** — fast pattern-matched verdicts on-device;
  novel situations escalate to bigger models or humans.
- **Self-improving loop** — every Phase-A cloud decision becomes Phase-B training data;
  the fleet teaches its own replacement and keeps the teacher for the edges.
- **Kaggle 2×T4 path** — `kaggle/kernel_v2` regenerates a push-ready kernel from the local
  trainer, so fine-tunes scale beyond the workstation.

## Gate status (honest, at release)

Seven checkpoint runs across six domains: **teacher-agreement 0.91–0.96 (all above the
0.85 bar), calibration 0.37–0.65 ECE (bar: 0.10) — all FAIL, none registered.** The r2
experiment (training-serving consistency fix) lifted agreement but confirmed calibration
is data-volume-bound at ~600 rows/domain. Flywheels keep feeding; domains retrain as they
reach 1.5–3k canonical rows. **We publish the gate because it's the product**: numbers
only ship when they pass.

## Attribution

Built on **Laya** (Convai Innovations, Apache-2.0) and the **Jev/System One** API
(TypeSafe AI); browser loop via **browser-use/jev-ultrafast** (MIT); automation via
**Playwright/Patchright**. Fine-tuning protocol, router, explorer, and integrations by
**qalarc**. Full credits: `ATTRIBUTION.md`.
