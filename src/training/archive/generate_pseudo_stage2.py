from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"F:\fer-pi5")
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pseudo_core import run_pseudo_rebalance_generation, print_manifest_summary


CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "stage_name": "stage2",

    "unlabeled_csv": str(PROJECT_ROOT / "data" / "csv" / "unlabeled.csv"),
    "img_root": None,

    "teacher_ckpt": str(PROJECT_ROOT / "checkpoints" / "best_model_stage2_refactored.pth"),
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
    "tta_hflip": True,

    # Stage2 teacher should be stronger; thresholds are higher while still keeping minority classes less strict.
    "default_min_conf": 0.90,
    "class_min_conf": {
        "anger": 0.90,
        "disgust": 0.86,
        "fear": 0.86,
        "happy": 0.95,
        "sad": 0.89,
        "surprise": 0.90,
        "neutral": 0.94,
    },

    "default_max_per_class": None,
    "class_max_per_class": {
        "anger": 6000,
        "disgust": 5000,
        "fear": 5000,
        "happy": 10000,
        "sad": 6000,
        "surprise": 6000,
        "neutral": 10000,
    },

    "min_margin": 0.00,
    "max_entropy": None,

    "run_base_dir": str(PROJECT_ROOT / "runs" / "pseudo_labels"),
    "output_dir": str(PROJECT_ROOT / "data" / "csv"),
    "output_csv_name": "pseudo_labeled_stage2_rebalanced.csv",
    "compatibility_alias": "pseudo_labeled_stage2.csv",

    "write_all_candidates": True,
    "include_probs_in_audit": True,
    "include_pixels_in_audit": False,
}


def main() -> None:
    print("=== PyCharm lazy-run mode: Stage2 rebalanced pseudo labels ===")
    print("Teacher:", CONFIG["teacher_ckpt"])
    print("Unlabeled:", CONFIG["unlabeled_csv"])
    manifest = run_pseudo_rebalance_generation(CONFIG)
    print_manifest_summary(manifest)


if __name__ == "__main__":
    main()
