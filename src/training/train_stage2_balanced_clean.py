from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import train_core
    from balanced_sampler_patch import install_balanced_batch_sampler
except Exception as exc:
    print("\n[Stage2 BALANCED CLEAN] Import failed.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {_THIS_DIR}")
    print("Expected     : train_core.py and balanced_sampler_patch.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# Inject balanced-batch support into the existing refactored train_core.py.
install_balanced_batch_sampler(train_core)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# Override without editing this script:
#   set FER_PROJECT_ROOT=F:\fer-pi5
PROJECT_ROOT = Path(os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"))


def _p(*parts: str) -> str:
    """Return a project-relative path string."""
    return str(Path(*parts))


# ---------------------------------------------------------------------------
# Stage2 Balanced Clean
# ---------------------------------------------------------------------------
# Goal:
#   After Balanced Control, test whether clean pseudo labels still add value.
#
# This run:
#   - starts from best_model_stage1_refactored.pth;
#   - uses strict 7-class balanced batches;
#   - disables class weights to avoid double correction;
#   - uses clean pseudo labels with conservative pseudo_loss_scale=0.25.
#
# Run this only after train_stage2_balanced_control.py.
CONFIG: Dict[str, Any] = {
    # Identity
    "project_root": str(PROJECT_ROOT),
    "stage": "stage2",

    # Data
    "train_csv": _p("data", "csv", "train.csv"),
    "val_csv": _p("data", "csv", "val.csv"),
    "test_csv": _p("data", "csv", "test.csv"),
    "img_base": None,

    # Clean pseudo labels from the Stage1 teacher pipeline.
    "pseudo_csv": _p("data", "csv", "pseudo_labeled_stage1_clean.csv"),
    "require_pseudo_conf": True,

    # Start from the verified Stage1 teacher.
    "init_ckpt": _p("checkpoints", "best_model_stage1_efficientnet_b0.pth"),

    # Keep outputs isolated from normal Stage2 aliases.
    "best_alias_name": "best_model_stage2_efficientnet_b0_balanced_clean.pth",
    "log_alias_name": "train_stage2_efficientnet_b0_balanced_clean_log.csv",
    "write_checkpoint_alias": True,
    "alias_overwrite": True,

    # Output layout
    "runs_dir": _p("runs", "training"),
    "checkpoint_alias_dir": _p("checkpoints"),

    # Model
    "model_variant": "efficientnet_b0",
    "num_classes": 7,
    "pretrained": False,
    "aux_loss_weight": 0.0,
    "use_checkpoint": False,
    "compile_model": False,
    "strict_checkpoint_load": True,

    # Training
    "device": "cuda",
    "epochs": 200,
    "batch_size": 126,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "drop_last_train": True,

    # Optimizer
    "lr": 1e-5,
    "lr_floor": 1e-6,
    "warmup_epochs": 3,
    "weight_decay": 1e-4,
    "grad_accum_steps": 1,

    # Loss
    "label_smoothing": 0.05,
    "use_class_weights": False,
    "class_balance_beta": 0.995,
    "class_weights_from": "labeled_train",

    # Balanced sampler
    "sampling_strategy": "balanced_batch",
    "balanced_per_class": "auto_min",
    "balanced_per_class_source": "labeled_train",
    "balanced_samples_per_class_per_batch": 18,
    "balanced_strict_batch_size": True,
    "balanced_replacement": False,

    # Pseudo-label controls
    "pseudo_conf_min": 0.82,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 0.20,
    "pseudo_rampup_epochs": 10,

    # Validation / checkpointing
    "val_interval": 1,
    "early_stop_patience": 20,
    "best_metric": "global_macro_f1",
    "evaluate_test_at_end": True,
    "save_last_every_epoch": True,

    # Reproducibility
    "seed": 42,
    "deterministic_algorithms": False,
    "cudnn_benchmark": False,

    # Stability
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    "notes": "Stage2 balanced clean: clean pseudo labels plus strict 7-class 1:1 batches.",
}


if __name__ == "__main__":
    train_core.main("stage2", CONFIG)
