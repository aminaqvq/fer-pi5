from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(r"D:\fer-pi5")
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pseudo_core import run_pseudo_rebalance_generation, print_manifest_summary


CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "stage_name": "stage1_clean",

    "unlabeled_csv": str(PROJECT_ROOT / "data" / "csv" / "unlabeled.csv"),
    "img_root": None,

    "teacher_ckpt": str(PROJECT_ROOT / "checkpoints" / "best_model_stage1_efficientnet_b0.pth"),
    "model_variant": "efficientnet_b0",
    "num_classes": 7,
    "pretrained": False,
    "strict_checkpoint": True,
    "compile_model": False,

    "device": "cuda",
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "img_size": 224,
    "strict_pixels": True,
    "seed": 42,

    # Keep horizontal-flip TTA during teacher scoring.
    # Teacher pseudo labels should be generated as accurately and stably as possible.
    "tta_hflip": True,

    # Class-adaptive confidence thresholds.
    # Keep weak classes slightly lower, but apply margin and entropy gates below.
    "default_min_conf": 0.85,
    "class_min_conf": {
        "anger": 0.88,
        "disgust": 0.82,
        "fear": 0.82,
        "happy": 0.93,
        "sad": 0.87,
        "surprise": 0.87,
        "neutral": 0.92,
    },

    # Per-class caps prevent happy/neutral from dominating the pseudo set.
    "default_max_per_class": None,
    "class_max_per_class": {
        "anger": 4000,
        "disgust": 3000,
        "fear": 3000,
        "happy": 8000,
        "sad": 4000,
        "surprise": 4000,
        "neutral": 8000,
    },

    # New quality gate 1:
    # Require the winning class to beat the second-best class by at least 0.05.
    # This rejects ambiguous samples even if top-1 confidence is high.
    "min_margin": 0.05,

    # New quality gate 2:
    # Entropy is computed from the full 7-class probability distribution.
    # For 7 classes, maximum entropy is ln(7) ~= 1.946.
    # 0.80 is conservative but still compatible with the 0.82 weak-class
    # confidence threshold when remaining probability is spread across classes.
    "max_entropy": 0.80,

    "run_base_dir": str(PROJECT_ROOT / "runs" / "pseudo_labels"),
    "output_dir": str(PROJECT_ROOT / "data" / "csv"),
    "output_csv_name": "pseudo_labeled_stage1_clean.csv",

    # Avoid silently overwriting the old compatibility alias used by older scripts.
    # train_stage2_clean.py points to pseudo_labeled_stage1_clean.csv explicitly.
    "compatibility_alias": "",

    # Audit artifacts.
    "write_all_candidates": True,
    "include_probs_in_audit": True,
    "include_pixels_in_audit": False,
}


def main() -> None:
    print("=== PyCharm lazy-run mode: Stage1 CLEAN pseudo labels ===")
    print("Teacher:", CONFIG["teacher_ckpt"])
    print("Unlabeled:", CONFIG["unlabeled_csv"])
    print("Output :", str(PROJECT_ROOT / "data" / "csv" / CONFIG["output_csv_name"]))
    print("Gates  : min_margin =", CONFIG["min_margin"], "| max_entropy =", CONFIG["max_entropy"])

    manifest = run_pseudo_rebalance_generation(CONFIG)
    print_manifest_summary(manifest)


if __name__ == "__main__":
    main()
