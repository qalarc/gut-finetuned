# Browser-Agent Bridge — Laya-swapped jev-ultrafast

> Goal: a **fully local** ultrafast browser agent — jev-ultrafast's loop, Laya's decisions.
> jev-ultrafast clone: `investigation/public_repos/decision_engines/jev-ultrafast/` (MIT).

## Why it works architecturally

jev-ultrafast already speaks typed choice questions: every cycle sends ONE request with
operation + (speculative) target heads against the observed element table; only `TYPE_TEXT`
uses a small LLM. Laya implements the identical primitive locally. The swap surface is small:

| File (their repo) | Bridge change |
|---|---|
| `jev_ultrafast/model.py` | Implement `LayaSource` with the same interface as the TypeSafe client: build the operation/target choice questions from `questions.py` unchanged, call `laya.predict`, return the same answer shape (choice + probabilities). Fan-out already returns per-head probabilities — map them straight. |
| `.env` | `TYPESAFE_API_KEY` unused; `LAYA_CKPT=checkpoints/browser-ops` |
| text helper | already OpenAI-compat → point at local GLM (llama-server/Ollama `:11434/v1`) — GLM explicitly supported upstream |

## The data flywheel

1. **Phase A — capture (cloud teacher):** run jev-ultrafast stock (Jev API) on real tasks;
   it saves traces (repo feature). Convert traces → `traces/*.jsonl` → dataset rows:
   state = element-table snapshot (+goal), questions = operation/target heads,
   answers = the executed choices (verified outcomes only — DONE states pass the task check).
2. **Phase B — fine-tune:** `datasets/browser-ops.jsonl` → `finetune_start("browser-ops")`
   (protocol as usual; high-cardinality targets need `head_max_len` raised — see laya
   README honest-limits: 512/1024 for >20 element options, or `predict_shortlist`).
3. **Phase C — swap + shadow:** run with `LayaSource`; shadow-log both engines for N runs;
   register when agreement ≥ gate.
4. **Phase D — fully local:** browser (own Chrome) + Laya (iGPU, 33ms class) + GLM local
   for typing. Zero cloud deps.

## Qalarc uses once local

- SEO scanner interactive tier (log in, click through, verify rankings/fixes)
- OSINT hub web tasks behind MCP (platforms needing real browsers)
- gmux agent browser tool (fast, cheap, auditable — no screenshots by design)
- tradez lead capture / form automation

## Honest risks

- Fine-tune quality for operation/target heads is unproven until Phase B data exists
  (need ~5-10k verified cycles; generate with cloud teacher)
- Target heads can exceed 20 options on busy pages → head_max_len tuning required
- Their MVP limits: shadow DOM, iframes, canvas, uploads not handled — same for us
