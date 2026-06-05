from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import torch
import torch.nn as nn
from torch.optim import AdamW

from model_repvggplus import create_RepVGGplus_L2pse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
NUM_CLASSES = 7
USE_AMP = True
USE_CHECKPOINT = True

STAGE1_CANDIDATES = [2, 4, 6, 8, 10, 12, 16]
BALANCED_CANDIDATES = [7, 14, 21, 28]


def vram_info() -> tuple[int, int, int]:
    """Return (allocated_MB, reserved_MB, max_allocated_MB)."""
    return (
        int(torch.cuda.memory_allocated() / 1024**2),
        int(torch.cuda.memory_reserved() / 1024**2),
        int(torch.cuda.max_memory_allocated() / 1024**2),
    )


def probe_one(batch: int, num_classes: int, device: str) -> dict:
    """Try one forward+backward+step with the given batch size."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = create_RepVGGplus_L2pse(
        num_classes=num_classes, deploy=False, use_checkpoint=USE_CHECKPOINT, use_aux=True,
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    x = torch.randn(batch, 3, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, num_classes, (batch,), device=device)

    result: dict = {"batch": batch, "ok": False, "oom": False, "allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}

    try:
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            output = model(x)
            if isinstance(output, dict):
                logits = output["main"]
            else:
                logits = output
            loss = nn.functional.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        alloc, reserv, peak = vram_info()
        result.update({"ok": True, "allocated_mb": alloc, "reserved_mb": reserv, "max_allocated_mb": peak})
    except torch.cuda.OutOfMemoryError:
        result["oom"] = True
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            result["oom"] = True
            torch.cuda.empty_cache()
        else:
            raise

    del model, optimizer, scaler, x, y, loss
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available — cannot probe VRAM.")
        return

    total_mb = int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {total_mb} MB ({total_mb/1024:.1f} GB)")
    print()

    # ── Stage1 ──────────────────────────────────────────────────
    print("=== Stage1 batch probe ===")
    stage1_results = []
    for bs in STAGE1_CANDIDATES:
        r = probe_one(bs, NUM_CLASSES, DEVICE)
        stage1_results.append(r)
        status = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {status:5s}  alloc={r['allocated_mb']:5d} MB  reserved={r['reserved_mb']:5d} MB  peak={r['max_allocated_mb']:5d} MB")

    # Find max stable batch
    max_ok_bs = max((r["batch"] for r in stage1_results if r["ok"]), default=0)
    safe_bs = max(2, int(max_ok_bs * 0.75))  # 75% of max for safety
    if safe_bs < 2:
        safe_bs = 2

    # ── Stage2/3 balanced ──────────────────────────────────────
    print("\n=== Stage2/3 balanced batch probe ===")
    balanced_results = []
    for bs in BALANCED_CANDIDATES:
        r = probe_one(bs, NUM_CLASSES, DEVICE)
        balanced_results.append(r)
        status = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {status:5s}  alloc={r['allocated_mb']:5d} MB  reserved={r['reserved_mb']:5d} MB  peak={r['max_allocated_mb']:5d} MB")

    max_ok_balanced = max((r["batch"] for r in balanced_results if r["ok"]), default=7)
    safe_balanced = max(7, max_ok_balanced - 7)  # drop one level from max

    # ── Pseudo batch ───────────────────────────────────────────
    print("\n=== Pseudo (eval) batch probe ===")
    pseudo_candidates = [4, 8, 16, 32]
    pseudo_results = []
    for bs in pseudo_candidates:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        model = create_RepVGGplus_L2pse(
            num_classes=NUM_CLASSES, deploy=False, use_checkpoint=False, use_aux=False,
        ).to(DEVICE)
        model.eval()

        x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        r = {"batch": bs, "ok": False, "oom": False, "allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}
        try:
            with torch.no_grad():
                _ = model(x)
            alloc, reserv, peak = vram_info()
            r.update({"ok": True, "allocated_mb": alloc, "reserved_mb": reserv, "max_allocated_mb": peak})
        except torch.cuda.OutOfMemoryError:
            r["oom"] = True
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                r["oom"] = True
                torch.cuda.empty_cache()

        pseudo_results.append(r)
        status = "OK" if r["ok"] else ("OOM" if r["oom"] else "FAIL")
        print(f"  batch={bs:3d}  {status:5s}  alloc={r['allocated_mb']:5d} MB  reserved={r['reserved_mb']:5d} MB  peak={r['max_allocated_mb']:5d} MB")

        del model, x
        torch.cuda.empty_cache()
        gc.collect()

    max_ok_pseudo = max((r["batch"] for r in pseudo_results if r["ok"]), default=4)
    safe_pseudo = min(max_ok_pseudo, 16)  # cap at 16 for eval mode

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG")
    print("=" * 60)
    print(f"  Stage1 batch          : {safe_bs}")
    print(f"  Stage1 grad_accum     : {32 // safe_bs if safe_bs > 0 else 8}  (effective batch ~32)")
    print(f"  Stage2/3 batch        : {safe_balanced}  (must be multiple of 7)")
    acc2 = 28 // safe_balanced if safe_balanced > 0 else 4
    print(f"  Stage2/3 grad_accum   : {acc2}  (effective batch ~28)")
    print(f"  Pseudo batch (eval)   : {safe_pseudo}")
    print(f"  use_amp               : True")
    print(f"  use_checkpoint        : True  (training only)")
    print("=" * 60)


if __name__ == "__main__":
    main()
