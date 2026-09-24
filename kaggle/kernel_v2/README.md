# kernel_v2/ — coordination note (2026-09-23, qalcode session)

Two kernels live in this directory. **The tulpa agent owns `laya-domains`** (it was
pushed and ran 19:24 today, and the agent was still iterating — `chanalyse-v2` ran
22:44). Do not push from here without checking with the agent / owner first.

| File | What | Status |
|---|---|---|
| `laya_domains.py` + `kernel-metadata.json` | **Tulpa agent's multi-domain trainer** — one RLCD checkpoint per attached JSONL (excludes combined-*), T4, dataset `qalarc/laya-domain-data`. This is the reviewed fix, evolved past single-combined-all. | LIVE (ran 19:24), owned by tulpa agent |
| `laya_rlcd_v2.py` | Generator output of `scripts/make_kaggle_kernel.py` — single combined-all run **with an HF-publish tail** (uploads `/kaggle/working/ckpt` → `Qalarc/laya-combined-all`, private, gate-labeled, graceful skip if no `HF_TOKEN` secret). | Available, NOT pushed |

## The missing piece the agent's kernel doesn't have yet: train → HF

`laya_domains.py` saves checkpoints to `/kaggle/working/<domain>-ckpt/` but nothing
uploads them to the hub — outputs vanish with the kernel eventually. The tail to adopt
(same code as in `laya_rlcd_v2.py` / `make_kaggle_kernel.py` OUTRO): per domain, after
the gate table, upload to `Qalarc/laya-<name>` **private** via `HfApi.upload_folder`,
gated on the `HF_TOKEN` Kaggle Secret (Add-ons → Secrets → attach — one manual step per
kernel; skips gracefully when absent; eval.json rides along for auditability).

Rules for that upload (from `~/projects/GLM_projects/huggingface/AGENTS.md`):
- private always; the owner approves repo creation policy once per new repo name
- registration into `serve.py` stays a LOCAL `finetune_register` step — never automatic
- keep `rl_agent_config.json` (fine on private repos)

## Related

- `~/projects/GLM_projects/huggingface/` — the HF control point (RULE ZERO: permission
  before upload; gated publish scripts).
- `KAGGLE_RUN_REVIEW_20260923.md` — the review that killed the original T5 kernel.
- Datasets mirrored private on the hub: `Qalarc/laya-decisions-*` (5 domains).
