#!/usr/bin/env python
"""GPU VRAM probe for EfficientNet-B0 on RTX 3060 Laptop.

Tests escalating batch sizes for Stage1, Stage2/3 balanced (multiples of 7),
and pseudo (eval-only).  Each candidate runs one real forward+loss+backward+step.

Usage:
    python src/training/probe_efficientnet_b0_batch_3060.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW

from model_mbv3 import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 7
USE_AMP = True

STAGE1_CANDIDATES = [32, 48, 64, 96, 128, 160, 192]
BALANCED_CANDIDATES = [63, 84, 105, 126, 147]
PSEUDO_CANDIDATES = [64, 128, 192, 256]


def vram_info():
    a = int(torch.cuda.memory_allocated() / 1024**2)
    r = int(torch.cuda.memory_reserved() / 1024**2)
    p = int(torch.cuda.max_memory_allocated() / 1024**2)
    return a, r, p


def probe_one(batch, training=True):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = get_model(
        variant="efficientnet_b0",
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=DEVICE,
        verbose=False,
    )
    if training:
        model.train()
        opt = AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    else:
        model.eval()

    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (batch,), device=DEVICE)

    result = {"batch": batch, "ok": False, "oom": False, "alloc_mb": 0, "reserv_mb": 0, "peak_mb": 0}

    try:
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = model(x)
            if training:
                loss = nn.functional.cross_entropy(out, y)

        if training:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        a, r, p = vram_info()
        result.update({"ok": True, "alloc_mb": a, "reserv_mb": r, "peak_mb": p})
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["oom"] = True
            torch.cuda.empty_cache()
        else:
            raise

    del model, x, y
    if training:
        del opt, scaler
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    total_mb = int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"total VRAM: {total_mb} MB ({total_mb/1024:.1f} GB)\n")

    # ── Stage1 ──────────────────────────────────────────────────
    print("=== Stage1 batch probe ===")
    s1_results = []
    for bs in STAGE1_CANDIDATES:
        r = probe_one(bs, training=True)
        s1_results.append(r)
        s = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {s:5s}  alloc={r['alloc_mb']:5d}  reserv={r['reserv_mb']:5d}  peak={r['peak_mb']:5d} MB")

    max_ok = max((r["batch"] for r in s1_results if r["ok"]), default=0)
    # Pick one level below max for safety
    ok_list = sorted([r["batch"] for r in s1_results if r["ok"]])
    rec_stage1 = ok_list[-2] if len(ok_list) >= 2 else ok_list[-1] if ok_list else 32

    # ── Stage2/3 balanced ──────────────────────────────────────
    print("\n=== Stage2/3 balanced batch probe ===")
    bal_results = []
    for bs in BALANCED_CANDIDATES:
        r = probe_one(bs, training=True)
        bal_results.append(r)
        s = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {s:5s}  alloc={r['alloc_mb']:5d}  reserv={r['reserv_mb']:5d}  peak={r['peak_mb']:5d} MB")

    ok_bal = sorted([r["batch"] for r in bal_results if r["ok"]])
    rec_bal = ok_bal[-2] if len(ok_bal) >= 2 else ok_bal[-1] if ok_bal else 63

    # ── Pseudo (eval only) ────────────────────────────────────
    print("\n=== Pseudo (eval) batch probe ===")
    pse_results = []
    for bs in PSEUDO_CANDIDATES:
        r = probe_one(bs, training=False)
        pse_results.append(r)
        s = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {s:5s}  alloc={r['alloc_mb']:5d}  reserv={r['reserv_mb']:5d}  peak={r['peak_mb']:5d} MB")

    ok_pse = sorted([r["batch"] for r in pse_results if r["ok"]])
    rec_pse = ok_pse[-2] if len(ok_pse) >= 2 else ok_pse[-1] if ok_pse else 64

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG")
    print("=" * 60)
    print(f"  recommended_stage1_batch       = {rec_stage1}")
    print(f"  recommended_stage2_3_batch     = {rec_bal}  (balanced_per_class = {rec_bal // 7})")
    print(f"  recommended_pseudo_batch       = {rec_pse}")
    print(f"  use_amp                        = True")
    print("=" * 60)


if __name__ == "__main__":
    main()
