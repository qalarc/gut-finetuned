# ASIC company/business-name flow — Jev vs our fine-tuned Laya: gap investigation

**Date:** 2026-09-23 · **Question:** can Jev navigate ASIC to set up a company name, and
can our fine-tuned Laya do the same? What's the gap?

## What was run

1. **jev-ultrafast agent (Jev cloud brain, browser-harness CDP)** on
   `asicconnect.asic.gov.au` — the actual registration portal.
2. **Autonomous explorer** (`explore.py`: playwright + Jev typed decisions, no
   screenshot dependency) on `www.asic.gov.au` — 18-page walk.

## Findings

### 1. The registration portal breaks the current agent runtime — even with the cloud brain
`asicconnect.asic.gov.au` is a legacy Java webapp that redirects mid-navigation. The agent
crashed twice: a dead CDP websocket ("no close frame") and a mid-load evaluate
("Document is navigating"). **This is a runtime gap, not a policy gap** — the world's best
decision model can't act through a harness that dies on the page. Required hardening
(before policy quality even matters): navigation guards (wait-for-quiescence +
retry-on-navigating), per-page timeout budgets, and a screenshot-free observation mode
(element tables only — which is what our heads consume anyway).

### 2. The main site navigates beautifully — and the explorer found the whole workflow
The playwright-based explorer walked 18 ASIC pages with Jev picking every hop, landing
exactly on the workflow spine:
`Register a business name` → `How to register a business name with ASIC` →
`Register a company` → business-structure choice (sole trader / partnership / company /
trust) → ASIC online-services portals (**the login boundary**). 18 decision rows captured
into `browser-ops.jsonl` (source `playwright-explore`).

### 3. The flow has a hard human boundary by design
Business-name registration requires **myGovID login + payment** — no autonomous agent
(or human without credentials) completes it end-to-end. The realistic autonomy envelope
is: navigate, choose the right workflow, pick business structure guidance, fill the
availability check, and **stop at the credential wall**. That's also the compliance-safe
product boundary (see tradez.au note below).

### 4. The Laya gap, quantified
- **Gap 0 — no trained checkpoint exists yet.** browser-ops has ~70 rows; the model needs
  ~300+ before an r1 and 1.5–3k for gate-worthy calibration. **This is the dominant gap.**
- **Gap 1 — head coverage.** ASIC-style flows need the exact heads browser-ops trains:
  operation (CLICK/TYPE_TEXT/SELECT/NAVIGATE/WAIT) + target-over-link-table — already the
  dataset shape ✓ — plus workflow routing ("this is a business-name task") which is a
  NEW head to add when we capture flow labels.
- **Gap 2 — runtime hardening** (finding 1): navigation guards + robust observation,
  independent of the model.
- **Gap 3 — measurement:** `browser_agent/replay_gap.py` replays every captured Jev
  decision against the local endpoint and reports per-head argmax agreement + probability
  mass on the teacher's choice. Run it after the browser-ops r1 to get the number.

## Product direction (tradez.au)

The same capture pipeline powers a "start your business" helper on tradez.au: guide the
user through business-structure choice → name availability → ASIC Connect hand-off at the
credential wall. The Laya model makes the guidance instant and local; the agent does the
walking; the human does the signing. The API-key angle (ABR/ASIC web-services keys) fits
the same pattern: agent walks the key-application flow, stops at the user's own
credentials, and the interface manages keys afterwards.

## Next actions

1. Re-run the asicconnect capture on an idle box (screenshot budgets need CPU) —
   `browser_agent/browser_up.sh && capture_traces.py` (task list in `tasks.jsonl`).
2. Grow browser-ops to 300+ rows (explorer runs are ~18 rows/5 min) → r1 → `replay_gap.py`.
3. Add a `workflow` head (flow routing) to browser-ops during the next capture batch.
4. Prototype the tradez.au business-formation guide using the captured workflow spine.
