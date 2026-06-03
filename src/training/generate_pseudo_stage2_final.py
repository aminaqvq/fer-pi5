from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from pseudo_core import run_pseudo_rebalance_generation, print_manifest_summary
except Exception as exc:
    print("\n[generate_pseudo_stage2_final] Import failed.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {THIS_DIR}")
    print("Expected     : pseudo_core.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"))


def _p(*parts: str) -> Path:
    return PROJECT_ROOT / Path(*parts)


def choose_existing(candidates: Iterable[Path], *, name: str) -> Path:
    checked = []
    for path in candidates:
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No valid {name} found. Checked:\n" + "\n".join(checked)
    )


# ---------------------------------------------------------------------------
# Teacher checkpoint
# ---------------------------------------------------------------------------
# Priority:
# 1. Locked historical best if you created it.
# 2. Existing Stage2 balanced clean alias.
# 3. Original run directory best checkpoint.
TEACHER_CKPT = choose_existing(
    [
        _p("checkpoints", "best_model_stage2_balanced_clean_0684451_LOCKED.pth"),
        _p("checkpoints", "best_model_stage2_balanced_clean.pth"),
        _p("runs", "training", "stage2_20260602_110224_seed42", "checkpoints", "best_model.pth"),
    ],
    name="Stage2 historical-best teacher checkpoint",
)


# ---------------------------------------------------------------------------
# Final Stage2 pseudo-label generation config
# ---------------------------------------------------------------------------
# This is deliberately conservative:
# - high confidence thresholds
# - margin filtering
# - per-class caps
# - output uses unique final name
#
# Do not use weighted_v2 / clean_v2 checkpoints here.
CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "stage_name": "stage2_final",

    "unlabeled_csv": str(_p("data", "csv", "unlabeled.csv")),
    "img_root": None,

    "teacher_ckpt": str(TEACHER_CKPT),
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
    "strict_checkpoint": True,
    "compile_model": False,

    "device": "cuda",
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "img_size": 224,
    "strict_pixels": True,
    "seed": 42,

    # Keep TTA on. It is cheap and makes pseudo-labels slightly safer.
    "tta_hflip": True,

    # Conservative class-adaptive thresholds.
    "default_min_conf": 0.92,
    "class_min_conf": {
        "anger": 0.92,
        "disgust": 0.88,
        "fear": 0.88,
        "happy": 0.96,
        "sad": 0.91,
        "surprise": 0.92,
        "neutral": 0.95,
    },

    # Modest caps: enough pseudo labels to complete Stage3,
    # but not enough to overwhelm labeled data.
    "default_max_per_class": None,
    "class_max_per_class": {
        "anger": 3000,
        "disgust": 2500,
        "fear": 2500,
        "happy": 4500,
        "sad": 3000,
        "surprise": 3000,
        "neutral": 4500,
    },

    # Extra safety. Previous pseudo generation used margin=0.
    # Here we require a small margin to reduce ambiguous pseudo labels.
    "min_margin": 0.08,
    "max_entropy": None,

    "run_base_dir": str(_p("runs", "pseudo_labels")),
    "output_dir": str(_p("data", "csv")),
    "output_csv_name": "pseudo_labeled_stage2_final.csv",

    # Compatibility alias so train_stage3_final.py can use either direct final CSV
    # or conventional stage3 pseudo name.
    "compatibility_alias": "pseudo_labeled_stage2.csv",

    "write_all_candidates": True,
    "include_probs_in_audit": True,
    "include_pixels_in_audit": False,
}


def preflight() -> None:
    print("\n[generate_pseudo_stage2_final] Preflight")
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"TEACHER_CKPT : {TEACHER_CKPT}")

    required = [
        Path(CONFIG["unlabeled_csv"]),
        Path(CONFIG["teacher_ckpt"]),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    if "weighted" in str(TEACHER_CKPT).lower() or "clean_v2" in str(TEACHER_CKPT).lower():
        raise ValueError(
            "Do not use weighted_v2 / clean_v2 failed checkpoints as Stage3 teacher. "
            "Use the historical best Stage2 balanced clean checkpoint."
        )


def main() -> None:
    preflight()
    print("\n=== Generate final Stage2 pseudo labels for Stage3 ===")
    print(f"Teacher: {CONFIG['teacher_ckpt']}")
    print(f"Output : {Path(CONFIG['output_dir']) / CONFIG['output_csv_name']}")
    manifest = run_pseudo_rebalance_generation(CONFIG)
    print_manifest_summary(manifest)

    selected_total = int(manifest.get("counts", {}).get("selected_total", 0))
    print(f"\n[selected_total] {selected_total}")
    if selected_total < 2000:
        print(
            "[warning] selected pseudo labels are fewer than 2000. "
            "Stage3 can still run, but impact may be limited."
        )


if __name__ == "__main__":
    main()