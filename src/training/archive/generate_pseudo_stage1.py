from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(r"F:\fer-pi5")
TRAINING_DIR = PROJECT_ROOT / "src" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pseudo_core import run_pseudo_rebalance_generation, print_manifest_summary


# =========================
# PyCharm lazy configuration
# =========================
CONFIG = {
    "project_root": str(PROJECT_ROOT),
    "stage_name": "stage1",

    "unlabeled_csv": str(PROJECT_ROOT / "data" / "csv" / "unlabeled.csv"),
    "img_root": None,

    "teacher_ckpt": str(PROJECT_ROOT / "checkpoints" / "best_model_stage1_refactored.pth"),
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
    "strict_checkpoint": True,
    "compile_model": False,

    "device": "cuda",  # automatically falls back inside PyTorch only if CUDA is unavailable? Set to "cpu" manually if needed.
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "img_size": 224,
    "strict_pixels": True,
    "seed": 42,
    "tta_hflip": True,

    # Stage1 teacher is weaker on fear/disgust and stronger on happy/neutral.
    # These thresholds intentionally tighten majority/easy classes and loosen hard minority classes.
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

    # Caps prevent happy/neutral from dominating Stage2.
    # Weak classes may still have fewer samples; the script will report that honestly.
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

    # Extra confidence quality gates. Keep min_margin=0.00 first; raise to 0.05 only if false positives are obvious.
    "min_margin": 0.00,
    "max_entropy": None,

    "run_base_dir": str(PROJECT_ROOT / "runs" / "pseudo_labels"),
    "output_dir": str(PROJECT_ROOT / "data" / "csv"),
    "output_csv_name": "pseudo_labeled_stage1_rebalanced.csv",
    "compatibility_alias": "pseudo_labeled.csv",

    # Full audit can be large, but unlabeled is about 44k in your current split, so this is acceptable.
    "write_all_candidates": True,
    "include_probs_in_audit": True,
    "include_pixels_in_audit": False,
}


def main() -> None:
    print("=== PyCharm lazy-run mode: Stage1 rebalanced pseudo labels ===")
    print("Teacher:", CONFIG["teacher_ckpt"])
    print("Unlabeled:", CONFIG["unlabeled_csv"])
    manifest = run_pseudo_rebalance_generation(CONFIG)
    print_manifest_summary(manifest)


if __name__ == "__main__":
    main()
