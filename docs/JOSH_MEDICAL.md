# JOSH × GUT — medical implementation write-up

**Date:** 2026-09-27/28 · **Status:** research demonstration, working end-to-end
**External project:** [SampleBias/Jev_Onco_Statistical_Hierarchy](https://github.com/SampleBias/Jev_Onco_Statistical_Hierarchy) ("JOSH", MIT)
**Our side:** GUT (Guided Unconscious Thinking) — fine-tuned Laya behind `gut.qalarc.com`
**Companion live page:** [gut.qalarc.com/cup/](https://gut.qalarc.com/cup/) (tunnel-hosted local inference)

---

## 1. What JOSH is, and why it matters to us

JOSH is an independent, MIT-licensed Rust workbench (TUI + CLI, six crates) for
**cancer-of-unknown-primary (CUP) research**. It loads molecular, expression and clinical
evidence from validated TSVs, asks the cloud model Jev **three independent questions in
one call** — a 24-way origin Choice (22 cancer classes + `insufficient_evidence` +
`other_origin`) plus two Noul checks (`evidence_sufficient`, `conflicting_evidence`) —
then validates, gates, ranks, charts, and explains the answer entirely in Rust.

It is the first *external* consumer of the System One wire protocol we've found building
serious domain tooling on it — and its engineering doctrine is nearly identical to ours:
independent bounded questions, soft probability preservation, validation that never
renormalizes failures, abstention as a first-class outcome, versioned prompts, bounded
live-call budgets, and "keep numerical work in Rust". JOSH is Jev-as-teacher with a
clinical discipline we respect.

## 2. What we built (working today)

| Piece | Status |
|---|---|
| JOSH compiled; both prepare examples run (originals + our fixtures) | ✅ |
| `prepare_gut_fixture` example: 250 synthetic workups → System One requests, 0 import failures (validated through the real `josh_ingest` importer) | ✅ |
| `datasets/cup-origin.jsonl`: 250 Jev-labeled rows (soft distributions, `molecular-origin-v5`) | ✅ |
| `qalarc/cup-origin` dataset + **`cup-origin-r1` trained on Kaggle** | ✅ COMPLETE |
| r1 verdict: agreement **0.80** on a 24-way problem at 250 rows (chance ≈ 4%) — best first-run agreement in the program; ECE 0.537 FAIL (expected at this volume); gates correctly force REVIEW | ✅ honest |
| **Web-hosted live inference**: `gut.qalarc.com/cup/` → Cloudflare tunnel → home-hosted GUT endpoint (`X-Laya-Domain: cup-origin`) | ✅ verified end-to-end |
| JOSH-style gates implemented client-side (top ≥ 0.75, margin ≥ 0.15, sufficiency ≥ 0.8, conflict ≤ 0.2 → else REVIEW REQUIRED) | ✅ |
| gut-trainer MCP + trainer console (:8797) carry the full playbook incl. the JOSH wiring | ✅ |

Demo transcript (case 001, pulmonary-pattern workup, no localized primary — through the
public tunnel to home hardware): COADREAD 0.46 · GIST 0.17 · WDTC 0.10 ·
insufficient_evidence 0.10 · sufficiency 0.52 · conflict 0.55 → **REVIEW REQUIRED** in
~4.4 s round trip. The r1 checkpoint is uncalibrated (research), and the gate did its job:
a confident-looking distribution was held for review because conflict fired.

## 3. The privacy architecture (why this matters)

Medical evidence is the canonical cannot-leave-the-building dataset. JOSH itself notes
the model sees no patient IDs, source paths, provenance hashes or evaluation partitions —
but the *evidence* still goes to the cloud API today. GUT closes that loop:

```
workup (stays local) → Rust validation (JOSH) → structured evidence
        → GUT local endpoint (home hardware, ~20 ms GPU / 2.5 s CPU)
        → origin distribution + uncertainty + review gates
        → (training-time only) Jev cloud teacher on sanitized evidence
```

At inference nothing leaves home. At training, sanitized synthetic-style evidence goes to
the teacher — the same trade every hospital ML project makes, except the *serving* model
is 421M parameters on a workstation instead of a cloud endpoint.

## 4. Skin cancer / dermatology extension — what the data requires

The natural clinical sibling (and the user's question: face tracking + camera systems +
skin cancer training data). What it takes, concretely:

**Public datasets (real, credentialed ground truth — better than synthetic for this domain):**
- **ISIC Archive** (International Skin Imaging Collaboration): 70,000+ dermoscopic
  images with public diagnosis labels; the SIIM-ISIC 2020 Melanoma Classification
  Kaggle challenge alone is 33,126 images with biopsy-confirmed labels + metadata
  (age, sex, anatomical site). This is the training substrate.
- **ISIC 2024 (SLICE-3D)**: 400k+ cropped lesion slices for malignancy prediction.
- **PAD-UFES-20 / HAM10000**: smaller but metadata-rich (local accounts for ~40% of
  HAM10000's labels via follow-up — the "operational outcome" pattern we already use).
- **TCGA / SEER**: for the CUP side, outcome and site ground truth (restricted access,
  dbGaP-style applications — a real compliance step, not a blocker).

**The pipeline for skin-cancer referral triage (research framing):**
1. **Capture** — camera/dermoscope image stays on-device (phone or their camera-biomarkers
   web demo: MediaPipe face/region landmarks on-device, browser-based, no upload).
2. **Vision feature extraction** — an open dermoscopy classifier (or fine-tuned ViT)
   converts the image to *structured findings*: lesion ABCDE descriptors, site, diameter,
   patient age/sex. Images are never sent anywhere; findings are text.
3. **GUT decision head** — a `derm-triage` domain: malignancy-risk distribution
   (Choice over risk bands) + biopsy-urgency (Noul) + adequate-image (Noul — the
   quality-abstention pattern from JOSH's evidence_sufficient).
4. **Gate + review** — JOSH thresholds → low-risk reassure, mid-risk monitor,
   high-risk refer. Every output is review-required by design; nothing is a diagnosis.
5. **The privacy loop** — training-time only: structured findings (no images) go to the
   teacher for soft labels; serving is fully local.

**Why their systems fit:** the face-tracking stack (MediaPipe landmarks, on-device) and
the volkus-scan on-device demographic demo prove the capture layer already runs
browser-side with zero upload. Camera-biomarkers is literally a camera-health-monitor
pattern — extending it from landmarks to lesion regions is a UI change plus a vision
model swap, and its output feeds GUT exactly like JOSH's molecular evidence does.

## 5. Honest constraints and requirements

- **Volume**: cup-origin at 250 rows hit 0.80 agreement but failed calibration; the
  pattern across all domains says 1.5–3k rows per domain before ECE passes. Dermatology
  gets there faster because ISIC labels are public and biopsy-confirmed.
- **Labels vs ground truth**: Jev-shadow rows inherit the teacher's errors; ISIC rows
  carry biopsy ground truth. Mixing both (teacher soft labels + outcome corrections)
  is the protocol §1.2 pattern.
- **No clinical claims**: everything is research framing; gates output
  review_required, never diagnoses. Any real deployment needs clinical validation,
  ethics review, and medical-grade UI — out of scope for this repo by design.
- **Compute**: training on Kaggle T4s (free); serving on a workstation; the HF Space
  path needs PRO for hosted inference (the space files are ready at
  `Qalarc/gut-cup-origin-space` design — blocked only on subscription, see below).

## 6. Roadmap

| When | Item |
|---|---|
| Now | Grow cup-origin toward 1.5k (extend the generator's archetype table with JOSH's feedback); rerun r2; adopt prompt-versioned dataset rows everywhere |
| Now | serve.py tunnel bootstrap script (`cup-boot.sh`: tunnel up → tunnel.json → deploy) so the live page survives reboots hands-off |
| Next | cup-origin r2 on Kaggle with the normalized single-shape data; run JOSH's `molecular compare-prompts` methodology on our domain_benchmark for A/B gate reporting |
| Next | **Fork PR upstream**: a `--decider-url` flag for josh-jev pointing at a local System One endpoint (the one-URL flip, upstreamable to SampleBias) |
| Next | **derm-triage prototype**: HAM10000/ISIC metadata + vision findings → GUT head; camera-biomarkers capture UI extension |
| Later | **GUT Studio Tauri app**: desktop shell (local_media_studio pattern) wrapping capture → train → gate → serve for non-technical users; JOSH TUI stays for researchers |
| Later | Shapley-style question attribution (josh-explain pattern) applied to browser-ops and derm decisions |
| Blocked on owner | HF PRO subscription → hosted inference Space (design ready, `Qalarc/gut-cup-origin-space`); transfer fork to qalarc org; SampleBias contact/collab |

## 7. Credits

JOSH — study design, fixtures, gates doctrine, and the external validation that our
wire protocol is a standard: **SampleBias** (MIT). OncoNPC motivation:
Moon et al., *Nature Medicine* 29:2057 (2023). GUT, the RLCD pipeline, the local
endpoint, and this integration: qalarc.
