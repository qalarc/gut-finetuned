#!/usr/bin/env python
"""train.py — Laya fine-tuning runner: RLCD loop wired from the laya repo's notebook.

Source of truth (adapted, DDP removed → single device):
  investigation/public_repos/decision_engines/laya/notebooks/
    laya_finetune_typed_decisions_2xT4_kaggle.ipynb  (cells 6 + 8 + 12/14)

Pipeline: protocol-format dataset (datasets/<domain>.jsonl) → row-level 90/10 split →
per-question training items (laya.common.build_sequence) → GRPO-style RLCD loop with
proper-scoring-rule rewards (G group samples, σ 0.4→0.1, policy gradient + soft-CE
guidance, AdamW encoder/head LRs, cosine schedule) → per-(qtype, option-bucket)
temperature fitting on the HELD-OUT split → gate eval (teacher-agreement ≥0.85,
ECE ≤0.10 post-temperature, no crash buckets) → checkpoint saved in laya.load() format.

Usage:
  train.py --dataset datasets/monitor-triage.jsonl [--base english] [--epochs 4] \
           [--device auto] [--out checkpoints/monitor-triage-...]
  train.py --eval --dataset ... --ckpt checkpoints/...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from laya.common import (
    QTYPES,
    QTYPE_NAMES,
    build_model,
    build_sequence,
    ece_score,
    proper_reward,
    render_options,
    temp_bucket,
)

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = Path(os.environ.get("LAYA_PROGRESS_FILE")
                  or Path(__file__).resolve().parent / "train_progress.json")

# ------------------------------------------------- hyperparameters (notebook cell 8)
EPOCHS = 4
MICRO_BATCH = 8
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4
SIGMA_END = 0.1
WEIGHT_DECAY = 0.01


def log(msg: str):
    print(msg, flush=True)


def write_progress(d: dict):
    try:
        PROGRESS.write_text(json.dumps({**d, "ts": time.time()}))
    except OSError:
        pass


# ===================================================================== dataset
def load_rows(path: Path):
    rows, bad = [], 0
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            assert "state" in r and "questions" in r and "answers" in r
            rows.append(r)
        except Exception:
            bad += 1
    return rows, bad


def validate(rows):
    problems = []
    for i, r in enumerate(rows):
        for name, q in r["questions"].items():
            t = q.get("type")
            if t not in ("choice", "score", "noul"):
                problems.append(f"row {i}: question {name!r} bad type {t!r}")
                continue
            if t in ("choice", "score") and not q.get("criteria"):
                problems.append(f"row {i}: question {name!r} {t} without criteria")
        if not r.get("answers"):
            problems.append(f"row {i}: no answers")
    return problems


def target_from_answer(qtype: str, qdef: dict, answer, crit) -> list[float] | None:
    """Build the target distribution from a protocol-format answer cell.

    Supports: plain str (one-hot), float (noul), {"choice": str} / {"noul": f},
    {"probabilities": {...}} soft labels (teacher distributions — train best).
    Option order follows render_options (choice: criteria key order; noul: [false, true];
    score: level 0..n-1).
    """
    keys = (
        list(crit.keys())
        if qtype == "choice"
        else (
            [str(i) for i in range(len(crit))]
            if qtype == "score"
            else ["false", "true"]
        )
    )
    k = len(keys)
    probs: dict[str, float] | None = None
    if isinstance(answer, str):
        probs = {answer: 1.0}
    elif isinstance(answer, bool):
        probs = {"true" if answer else "false": 1.0}
    elif isinstance(answer, (int, float)):
        probs = {"true": float(answer), "false": 1.0 - float(answer)}
    elif isinstance(answer, dict):
        if "probabilities" in answer and isinstance(answer["probabilities"], dict):
            probs = {str(a): float(b) for a, b in answer["probabilities"].items()}
        elif answer.get("choice") is not None:
            probs = {str(answer["choice"]): 1.0}
        elif answer.get("noul") is not None:
            v = float(answer["noul"])
            probs = {"true": v, "false": 1.0 - v}
    if probs is None:
        return None
    target = [float(probs.get(key, 0.0)) for key in keys]
    s = sum(target)
    if s <= 0:
        return None  # teacher gave mass only to labels outside this question's criteria
    return [v / s for v in target]


def build_items(tok, rows, cfg):
    """Notebook cell 6: one item per (row, question) — per-question isolation."""
    items, skipped = [], 0
    for r in rows:
        state = r["state"]
        for qid, qdef in r["questions"].items():
            ans = (r.get("answers") or {}).get(qid)
            if ans is None:
                continue
            t = qdef["type"]
            crit = qdef.get("criteria")
            if t == "choice" and isinstance(crit, list):
                crit = {c: None for c in crit}
                qdef = {**qdef, "criteria": crit}
            target = target_from_answer(t, qdef, ans, crit)
            if target is None:
                skipped += 1
                continue
            q = {"t": t, "ins": str(qdef.get("instructions", "")), "crit": crit}
            try:
                seq, markers = build_sequence(
                    tok, state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 192)
                )
            except Exception:
                skipped += 1
                continue
            k = len(render_options(q))
            if len(markers) != k or k != len(target):
                skipped += 1
                continue
            items.append(
                {
                    "ids": seq,
                    "markers": markers,
                    "qtype": QTYPES[t],
                    "target": target,
                    "label": int(np.argmax(target)),
                }
            )
    return items, skipped


# ===================================================================== training
def collate(items, pad_id):
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def save_checkpoint(
    model, tok, cfg, out: Path, temperatures: dict, temps_per_type: list
):
    """Save in the exact laya.load() format (notebook cell 8, rank-0 branch)."""
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, out / "model.safetensors")
    model.encoder.config.save_pretrained(out / "encoder")
    tok.save_pretrained(out / "tokenizer")
    save_cfg = dict(cfg)
    save_cfg["fine_tuned"] = True
    save_cfg["temperature"] = [round(t, 4) for t in temps_per_type]
    save_cfg["temperature_by_options"] = temperatures
    with open(out / "rl_agent_config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)


def fit_one_temp(sel):
    """Notebook cell 8: LBFGS temperature fit on NLL, clamped to sane range."""
    if len(sel) < 10:
        return None
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        return None
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def collect_logits(model, items, pad_id, device, amp_dtype, use_amp, chunk=32):
    """Forward the items once → [(qtype, logits_k, target_k)] for calibration/eval."""
    model.eval()
    out = []
    for i in range(0, len(items), chunk):
        c = items[i : i + chunk]
        b = collate(c, pad_id)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["marker_pos"].to(device),
                b["marker_mask"].to(device),
                b["qtype"].to(device),
            )
        lg = logits.float().cpu().numpy()
        for r, it in enumerate(c):
            k = len(it["markers"])
            out.append(
                (it["qtype"], lg[r, :k], np.asarray(it["target"], dtype=np.float64))
            )
    return out


def fit_temperatures(calib_preds):
    """Protocol §3: per-(qtype, option-bucket) temperatures; per-qtype fallback.

    Returns (temperature_by_options, per_qtype_list). Clamp: package runtime refuses to
    apply temps outside [0.5, 5.0] (they'd sharpen confidence dishonestly), so fit within.
    """
    from laya.common import clamp_temperature

    by_bucket: dict[str, list] = {}
    for qt, z, t in calib_preds:
        by_bucket.setdefault(temp_bucket(qt, len(z)), []).append((z, t))
    temperatures, per_type = {}, []
    for qt_name in ("choice", "score", "noul"):
        qt = QTYPES[qt_name]
        sel = [(z, t) for q, z, t in calib_preds if q == qt]
        t_qt = fit_one_temp(sel)
        per_type.append(round(clamp_temperature(t_qt if t_qt is not None else 1.0), 4))
    for bucket, sel in by_bucket.items():
        t = fit_one_temp(sel)
        if t is None:
            qt = (
                int(bucket.split(":")[0])
                if bucket[0].isdigit()
                else QTYPES[bucket.split(":")[0]]
            )
            t = per_type[qt] if qt < 3 else 1.0
        temperatures[bucket] = round(clamp_temperature(t), 4)
    return temperatures, per_type


# ===================================================================== evaluation
@torch.no_grad()
def evaluate_gated(
    model, tok, cfg, held_rows, device, amp_dtype, use_amp, temperatures, per_type
):
    """Gate eval on held-out rows (protocol §4). Runs through the model+temperatures the
    same way the Agent runtime applies them (temp_bucket lookup)."""
    items, skipped = build_items(tok, held_rows, cfg)
    preds = collect_logits(model, items, tok.pad_token_id, device, amp_dtype, use_amp)
    agree = correct = n = 0
    confs, corr = [], []
    crash = {}  # qtype → [n, n_crash]
    probs_for_ece = []
    for (qt, logits_k, target), it in zip(preds, items):
        k = len(it["markers"])
        bucket = temp_bucket(qt, k)
        t_scale = temperatures.get(bucket, per_type[qt] if qt < 3 else 1.0)
        z = logits_k / max(t_scale, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()
        gold = int(np.argmax(target))
        pred = int(np.argmax(p))
        n += 1
        agree += int(pred == gold)
        # confidence = normalized entropy (same as runtime), correctness for ECE
        from laya.common import confidence_from_probs

        conf = confidence_from_probs(p, k)
        confs.append(conf)
        c = int(pred == gold)
        corr.append(c)
        correct += c
        probs_for_ece.append((p, target))
        c_i = crash.setdefault(qt, [0, 0])
        c_i[0] += 1
        if p.min() < 0.05:
            c_i[1] += 1
    ece = ece_score(np.array(confs), np.array(corr))
    agreement = agree / max(1, n)
    crash_bad = {
        QTYPE_NAMES[qt]: f"{bad}/{tot}"
        for qt, (tot, bad) in crash.items()
        if tot and bad / tot > 0.20
    }
    passed = agreement >= 0.85 and ece <= 0.10 and not crash_bad
    return {
        "n_questions": n,
        "skipped": skipped,
        "teacher_argmax_agreement": round(agreement, 4),
        "ece_post_temperature": round(float(ece), 4),
        "accuracy": round(correct / max(1, n), 4),
        "crash_buckets": crash_bad or "none",
        "temperatures_by_options": temperatures,
        "temperature_per_qtype": per_type,
        "gate": "PASS" if passed else "FAIL",
        "gate_rule": "agreement>=0.85 AND ece<=0.10 AND no crash buckets",
    }


# ===================================================================== main
def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name in ("auto", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.version, "hip", None):
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def base_model_dir(base: str) -> Path | str:
    """Resolve a base checkpoint to a local snapshot dir (or HF id for laya to fetch)."""
    if base != "english":
        return f"convaiinnovations/laya:{base}"  # handled below via subfolder download
    return "convaiinnovations/laya"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--base",
        default="english",
        choices=["english", "multilingual", "typed-decisions"],
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--micro-batch", type=int, default=MICRO_BATCH)
    ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    ap.add_argument(
        "--eval", action="store_true", help="evaluate --ckpt on held-out split only"
    )
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--domain", default=None, help="domain tag for progress reporting")
    a = ap.parse_args()

    device = resolve_device(a.device)
    use_amp = device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    domain = a.domain or Path(a.dataset).stem

    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    # ----- base checkpoint (local snapshot preferred; HF download otherwise)
    if a.ckpt and a.eval:
        model_dir = a.ckpt
    else:
        sub = None if a.base == "english" else a.base
        _names = (
            "rl_agent_config.json",
            "model.safetensors",
            "tokenizer/*",
            "encoder/*",
        )
        if sub:
            _patterns = [f"{sub}/{n}" for n in _names]
        else:
            _patterns = list(_names)
        try:
            model_dir = snapshot_download(
                "convaiinnovations/laya", allow_patterns=_patterns, token=None
            )
            if sub:
                model_dir = str(Path(model_dir) / sub)
        except Exception as e:  # offline fallback: find the cached snapshot
            log(f"snapshot_download failed ({e}); trying cache")
            cache = (
                Path.home()
                / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots"
            )
            snaps = sorted(cache.glob("*")) if cache.is_dir() else []
            if not snaps:
                sys.exit("no base checkpoint available offline")
            model_dir = str(snaps[-1])
    _fix_tokenizer_config(str(model_dir))

    with open(Path(model_dir) / "rl_agent_config.json") as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(str(Path(model_dir) / "tokenizer"))

    write_progress({"domain": domain, "stage": "loading-data", "device": str(device)})

    # ----- data: split by ROW first (no state leaks into held-out), then build items
    rows, bad = load_rows(Path(a.dataset))
    problems = validate(rows)
    if problems:
        log(f"VALIDATION ISSUES ({len(problems)}):")
        for p in problems[:8]:
            log(f"  {p}")
        sys.exit(2)
    rng = random.Random(13)
    rng.shuffle(rows)
    n_hold = max(1, int(len(rows) * 0.1))
    held_rows, train_rows = rows[:n_hold], rows[n_hold:]
    train_items, skipped = build_items(tok, train_rows, cfg)
    held_items, held_skipped = build_items(tok, held_rows, cfg)
    log(
        f"dataset {Path(a.dataset).name}: rows={len(rows)} (bad={bad}) "
        f"train={len(train_rows)} held={len(held_rows)} | items train={len(train_items)} "
        f"held={len(held_items)} skipped={skipped + held_skipped}"
    )
    if len(train_items) < 50:
        sys.exit("too few training items — collect more data first")

    qtype_counts = np.bincount([it["qtype"] for it in train_items], minlength=3)
    log(
        f"items by qtype: choice={qtype_counts[0]} score={qtype_counts[1]} noul={qtype_counts[2]}"
    )

    if a.eval:
        if not a.ckpt:
            sys.exit("--eval needs --ckpt")
        model = build_model(cfg, encoder_dir=str(Path(a.ckpt) / "encoder"))
        from safetensors.torch import load_file

        model.load_state_dict(
            load_file(str(Path(a.ckpt) / "model.safetensors")), strict=True
        )
        model.to(device).eval()
        temps = cfg.get("temperature_by_options", {})
        per_type = cfg.get("temperature", [1.0, 1.0, 1.0])
        res = evaluate_gated(
            model, tok, cfg, held_rows, device, amp_dtype, use_amp, temps, per_type
        )
        log("\n=== GATE TABLE ===")
        for k, v in res.items():
            log(f"  {k}: {v}")
        (Path(a.ckpt) / "eval.json").write_text(json.dumps(res, indent=2))
        sys.exit(0 if res["gate"] == "PASS" else 1)

    # ----- model
    model = build_model(cfg, encoder_dir=str(Path(model_dir) / "encoder"))
    from safetensors.torch import load_file

    weights_path = Path(model_dir) / "model.safetensors"
    model.load_state_dict(load_file(str(weights_path)), strict=True)
    model.to(device)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True
    model.train()

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": LR_ENCODER},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    micro = a.micro_batch
    accum = a.grad_accum
    total_updates = max(1, (len(train_items) // (micro * accum)) * a.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    out = Path(
        a.out or ROOT / "checkpoints" / f"{domain}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    log(f"device={device} amp={use_amp} | out={out}")
    log(
        f"RLCD: epochs={a.epochs} micro={micro} accum={accum} group={GROUP_SIZE} "
        f"σ {SIGMA_START}→{SIGMA_END} lr(enc/head) {LR_ENCODER}/{LR_HEAD}"
    )

    write_progress(
        {
            "domain": domain,
            "stage": "training",
            "epochs": a.epochs,
            "items": len(train_items),
            "out": str(out),
            "status": "running",
        }
    )

    # ----- the RLCD loop (notebook cell 8, single-device)
    t0 = time.time()
    for epoch in range(a.epochs):
        random.seed(42 + epoch)
        random.shuffle(train_items)
        epoch_loss = n_batches = 0
        accum_step = 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, a.epochs - 1))

        for b_idx in range(0, len(train_items), micro):
            chunk = train_items[b_idx : b_idx + micro]
            if not chunk:
                continue
            batch = collate(chunk, tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, act = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device),
                    batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # 1. Sample G noisy logit distributions, zero-mean projected over options
            eps = (
                torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            )
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. Proper-scoring-rule reward → group-relative advantages
            with torch.no_grad():
                r = proper_reward(
                    q,
                    target.unsqueeze(0),
                    batch["qtype"].to(device),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. Policy gradient + full soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = (
                -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1))
                .sum(-1)
                .mean()
            )
            loss = (loss_rl + 1.0 * loss_ce) / accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % accum == 0 or (b_idx + micro) >= len(train_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accum
            n_batches += 1
            if n_batches % 25 == 0:
                log(
                    f"  epoch {epoch + 1}/{a.epochs} step {n_batches} "
                    f"loss {loss.item() * accum:.4f} reward {r.mean().item():.3f} "
                    f"({(time.time() - t0):.0f}s)"
                )
                write_progress(
                    {
                        "domain": domain,
                        "stage": "training",
                        "status": "running",
                        "epoch": epoch + 1,
                        "epochs": a.epochs,
                        "step": n_batches,
                        "loss": round(loss.item() * accum, 4),
                        "reward": round(float(r.mean()), 4),
                        "out": str(out),
                    }
                )

        log(
            f"=== epoch {epoch + 1}/{a.epochs} done in {time.time() - t0:.0f}s "
            f"avg_loss {epoch_loss / max(1, n_batches):.4f} ==="
        )

        # rolling checkpoint so a crash never loses the epoch
        save_checkpoint(model, tok, cfg, out / "checkpoint_latest", {}, [1.0, 1.0, 1.0])
        (out / "checkpoint_latest" / "checkpoint_meta.json").write_text(
            json.dumps(
                {"epoch": epoch + 1, "avg_loss": epoch_loss / max(1, n_batches)},
                indent=2,
            )
        )

    # ----- calibration on the HELD-OUT split (protocol §3 — stricter than the notebook,
    # which fits on a train subsample; the gate consumes the same split)
    log("fitting per-(qtype, bucket) temperatures on held-out…")
    del optimizer, scaler, scheduler
    model.eval()
    calib_preds = collect_logits(
        model, held_items, tok.pad_token_id, device, amp_dtype, use_amp
    )
    temperatures, per_type = fit_temperatures(calib_preds)
    log(f"temperatures: by_bucket={temperatures} per_qtype={per_type}")

    save_checkpoint(model, tok, cfg, out, temperatures, per_type)
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "dataset": str(a.dataset),
                "rows": len(rows),
                "train_rows": len(train_rows),
                "held_rows": len(held_rows),
                "items": len(train_items),
                "epochs": a.epochs,
                "device": str(device),
                "seconds": round(time.time() - t0, 1),
                "temperatures_by_options": temperatures,
                "temperature_per_qtype": per_type,
            },
            indent=2,
        )
    )
    log(f"checkpoint saved: {out}")

    # ----- gate eval immediately (protocol §4)
    res = evaluate_gated(
        model, tok, cfg, held_rows, device, amp_dtype, use_amp, temperatures, per_type
    )
    log("\n=== GATE TABLE ===")
    for k2, v in res.items():
        log(f"  {k2}: {v}")
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    write_progress(
        {
            "domain": domain,
            "stage": "done",
            "status": "finished",
            "out": str(out),
            "gate": res["gate"],
            "agreement": res["teacher_argmax_agreement"],
            "ece": res["ece_post_temperature"],
        }
    )
    log(f"gate: {res['gate']} — register only if PASS (finetune_register)")
    sys.exit(0 if res["gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
