# Jev (TypeSafe AI) — Evaluation, Project Fit & Rating Framework

**2026-09-22 · key: `~/.secrets/typesafe.env` (600) · console: console.typesafe.ai · docs: docs.typesafe.ai**
**Already wired: `jev` MCP server registered in opencode/qalcode + Claude Code** (tools: `jev_ask`, `jev_decide`, `jev_status`).

## What Jev is (30 seconds)

Not a chat LLM. A **decision engine**: you POST a `state` + typed `questions`
(**choice** = pick from labeled options · **score** = rubric level · **noul** = truthy 0–1),
and get **probabilities + calibrated confidence** your code branches on. Measured on our key:
**0.7–0.9s round-trips**, ~400–500 input tokens/call, $42/B input tokens
(→ roughly **$0.00002 per decision-question**). Anti-LLM design: no prose, no parsing, no
context-rot (questions evaluated in parallel + isolation).

## Verified live (our key, real qalarc-shaped tests)

| Test | Result | Read |
|---|---|---|
| Concierge incident (503 ×4, 12 min) | severity=**high** (86%), page-owner-3am=**0.34 no**, cause=**origin_down** (90%) | Correctly severe but NOT wake-worthy — the calibration is the product |
| OSINT scan hit (GitHub, sherlock) | real-account=0.72, tier=**verified** 75% / likely 24% | Honest hedging — exactly what evidence grading needs |
| tradez lead ("3 trucks by Friday") | quality=**hot**, confidence=1.0 | Lead triage out of the box |

## The rating rubric (how to score ANY future tool like this)

Rate 1–5 on four axes; use when ≥16/20 and the first two axes are ≥4:
1. **Decision-shaped?** (verdict/classification/routing — not generation, not multi-hop reasoning)
2. **Volume × latency?** (many calls/day, sub-second matters, or in loops)
3. **Cost negligible?** (at our volumes: yes — pennies/day)
4. **Confidence actionable?** (does a threshold change behavior: auto-act vs escalate?)

Jev scores 5/5/5/5 for triage-shaped work; 1/2 on prose/reasoning (that stays with GLM 5.3).

## Project matrix (rated, highest first)

| Project | Use | Rating |
|---|---|---|
| **Concierge/monitor stack** | Incident triage (severity + wake-the-owner + likely-cause) on every alert transition; dedupe/escalate decisions; email classification (contact-form intent) | ★★★★★ — the perfect first integration; replaces ad-hoc threshold code |
| **OSINT hub** | Found-hit evidence grading (verified/likely/weak/junk per result), "is this person the same person" Noul checks across exhibits, report confidence labels | ★★★★★ — directly upgrades report quality; cheap enough per-hit |
| **qalarc_hub / phone AI** | Inbound message triage (for-whom routing, urgency, spam Noul) before the expensive Claude/GLM brain engages — Jev as the cheap pre-filter | ★★★★★ — two-stage brain: Jev gate (0.1s, pennies) → GLM/Claude only when warranted |
| **tulpa RAISE ▚ GATES** | Semantic eval probes with calibrated confidence (byte-gates stay; add Jev gates: "response stays in character", score-rubric voice/style) | ★★★★ — natural extension of the eval harness |
| **chanalyse / news / trading** | Headline classification (relevance/sentiment/asset-impact) at news volume; OpenAlice signal triage before execution consideration | ★★★★ — volume + decision-shaped |
| **SEO checking (your idea)** | Honest split: crawling/parsing isn't Jev — but the JUDGING layer is: per-page score rubrics (title/meta quality, content decay, cannibalization risk vs sibling pages), change-significance triage on monitor diffs ("meaningful content change vs analytics-noise") | ★★★★ as the judgment half of an SEO scanner |
| **gmux / agent fleet** | Output-verification gates (did this edit satisfy spec — Noul), permission-risk triage, agent result scoring | ★★★★ via the now-registered MCP |
| **ai_audio / exercise_cv / RFAI** | Occasional triage only | ★★ — not decision-shaped at core |

## Usage doctrine

- **Two-stage brain pattern** (the big one): Jev gates/routs at ~$0.00002 → GLM 5.3 reasons only on
  what passes. Flip it into concierge, phone hub, and monitor as each gets touched.
- **Atomic questions, combine in code**: not "rate this startup" but separate market/feasibility/team
  scores with your own coefficients (docs' own doctrine — fits how we already build).
- **Confidence = policy**: e.g. OSINT report auto-labels at conf ≥0.8, "needs human eye" at 0.5–0.8,
  discard <0.5. Wake-owner only at noul ≥0.7 AND severity ≥ high.
- **Complement, never replacement**: reasoning, synthesis, prose, tool-use → GLM. Verdicts at
  volume with honest uncertainty → Jev.
- **Security**: key only in `~/.secrets/typesafe.env`; never in repos; rotate from console if leaked.

## Cost reality check

A busy day across all ★★★★★ uses (~5k decision-questions × ~500 tok): ≈ **$0.10/day**.
Effectively free vs GLM for the same volume — that's the whole point of System One.

---

## UPDATE 2026-09-22 — Laya (open System One) + jev-ultrafast (browser agent)

### Laya — NandhaKishorM/laya (10.2k★, Apache-2.0, open weights on HF) — MEASURED LOCALLY

Same typed-decision primitive as Jev (choice/score/noul), 3 checkpoints + Router (100+ langs),
pip-installable, runs LOCAL (421M ModernBERT / 322M mmBERT). Cloned: `public_repos/decision_engines/laya/`.
Venv: `typesafe_jev/laya-venv/`.

**Head-to-head on OUR questions (zero-shot, CPU):**

| Question | Jev cloud | Laya local zero-shot |
|---|---|---|
| Incident severity (503×4) | **high** @ conf 0.81 (86%) | medium @ conf **0.16** |
| Wake owner 3am | no (0.34) | no (0.24) ✓ |
| OSINT hit tier | **verified** 75% conf 0.66 | verified @ conf 0.14, probs mushy (0.53/0.22/0.13/0.13) |
| Latency | 0.7–0.9s | 0.25–0.47s CPU (33ms class on GPU) after 67s load |

Confirms the repo's own honest warning: **base checkpoints are not zero-shot decision engines**
(near-chance on general typed-decisions; value comes from fine-tuning — their fine-tuned
checkpoint beats Jev 0.766 vs 0.727 ON ITS OWN benchmark). Ships over-confident; temperature
clamping warning on load.

**Verdict — NOT a better fit as a Jev replacement; it's the complementary LOCAL tier:**
- Jev cloud = general/zero-shot, confident, correct → stays the default
- Laya local = fine-tuned hot paths: monitor-triage at volume, OSINT grading, gmux routing —
  our domain data via the RLCD fine-tune notebook (2×T4 4–5h ≈ our 8060S iGPU) → $0 marginal,
  private (no state leaves home — matters for phone-hub message triage!), same interface so
  AB-swap is trivial. Jev keeps the edges: >20-option choices, soft distributions, zero-shot.

### jev-ultrafast — browser-use/jev-ultrafast (15.7k★, MIT)

A BROWSER AGENT: indexed element table → ONE Jev request per cycle (operation + target choice
heads) → small LLM writes text only for TYPE_TEXT. No screenshots, 7.1s Google-Flights demo,
100 protocol calls vs 1,092 before. Cloned: `public_repos/decision_engines/jev-ultrafast/`.

- **Can it be local? Partially today, fully with Laya.** Browser ✓ local (Browser Harness +
  your Chrome), text model ✓ (GLM explicitly supported via OpenAI-compat), but the decision
  core calls the Jev cloud API.
- **Link with Laya: YES — architecturally clean.** Its decision interface IS typed choice
  questions; Laya implements the identical primitive locally. Path: run with Jev cloud →
  capture traces (repo saves them) → fine-tune Laya on operation/target choices → implement a
  Laya-backed source in `model.py` → fully-local ultrafast browser agent. Needs GPU (ROCm)
  for the 33ms class; CPU cycles (~0.3-0.5s) would make it sluggish.
- **qalarc relevance:** SEO scanning's interactive tier, OSINT web tasks, gmux agent browser
  tool, tradez lead capture — the fastest/cheapest web-agent pattern available.

### The doctrine now (three tiers)

1. **Jev cloud** — zero-shot verdicts everywhere (registered MCP, keep as default)
2. **Laya local** — fine-tune per domain once data accumulates (monitor/OSINT/phone = privacy
   + volume winners); same primitives, swap behind one interface
3. **jev-ultrafast** — the ACTING tier: Jev/Laya decisions driving a real browser at speed
