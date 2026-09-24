# License notes — qalarc/laya-integrations

**Short version:** free for anything noncommercial. Commercial use needs a separate
license from qalarc (contact via qalarc.com). Split by artifact:

| Artifact | License | File |
|---|---|---|
| Code (scripts, trainer, console, explorer, router, MCP server) | **PolyForm Noncommercial 1.0.0** | `LICENSE_POLYFORM.md` |
| Datasets (`datasets/*.jsonl`) | **CC BY-NC 4.0** | `LICENSE-DATA` (summary; full text at creativecommons.org/licenses/by-nc/4.0/legalcode) |
| Fine-tuned checkpoints (weights) | **CC BY-NC 4.0** | as distributed on HuggingFace (`license: cc-by-nc-4.0`) |

## Why this split

- The base model (Convai Laya) is Apache-2.0, which permits derivative fine-tunes under
  different terms — our NC terms are valid for the fine-tuned weights, with Apache
  attribution preserved in `ATTRIBUTION.md`.
- Everything qalarc-generated (code + collected/derived datasets + checkpoints) carries
  the noncommercial restriction per the owner's licensing decision.
- **Commercial licenses are available** — anyone wanting to commercialize should contact
  qalarc (qalarc.com). This is the standard PolyForm "dual licensing" pattern.

## Third-party obligations preserved

- Laya (Convai Innovations) — Apache-2.0: attribution + license notice retained in
  ATTRIBUTION.md and every model card.
- TypeSafe Jev — proprietary API: labels used per its terms; flag to legal before any
  public distribution of checkpoints trained on Jev outputs (owner review).
- browser-use/jev-ultrafast (MIT) — copyright notice preserved in ATTRIBUTION.md.
- Playwright/Patchright (Apache-2.0/MIT) — used as dependencies only.

## HF model-card mapping

- `license: cc-by-nc-4.0` on checkpoint repos (weights).
- Dataset cards: `license: cc-by-nc-4.0`.
