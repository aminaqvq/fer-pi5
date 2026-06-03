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
    print("\n[Stage2 BALANCED CONTROL] Import failed.")
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
# Stage2 Balanced Control
# ---------------------------------------------------------------------------
# Goal:
#   Test whether strict 7-class balanced training alone improves Stage2.
#
# This run:
#   - uses NO pseudo labels;
#   - starts from best_model_stage1_refactored.pth;
#   - disables class weights because the sampler already enforces 1:1 class balance;
#   - uses batch_size=126 = 7 classes * 18 samples per class.
#
# If the smallest labeled class has 5486 samples:
#   floor(5486 / 18) * 18 = 5472 samples per class per epoch
#   5472 / 18 = 304 batches per epoch
#   304 * 126 = 38304 samples per epoch
CONFIG: Dict[str, Any] = {
    # Identity
    "project_root": str(PROJECT_ROOT),
    "stage": "stage2",

    # Data
    "train_csv": _p("data", "csv", "train.csv"),
    "val_csv": _p("data", "csv", "val.csv"),
    "test_csv": _p("data", "csv", "test.csv"),
    "img_base": None,

    # Control arm: no pseudo labels.
    "pseudo_csv": None,
    "require_pseudo_conf": False,

    # Start from the verified Stage1 teacher.
    "init_ckpt": _p("checkpoints", "best_model_stage1_refactored.pth"),

    # Keep outputs isolated from normal Stage2 aliases.
    "best_alias_name": "best_model_stage2_balanced_control.pth",
    "log_alias_name": "train_stage2_balanced_control_log.csv",
    "write_checkpoint_alias": True,
    "alias_overwrite": True,

    # Output layout
    "runs_dir": _p("runs", "training"),
    "checkpoint_alias_dir": _p("checkpoints"),

    # Model
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
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
    "lr": 3e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,

    # Loss
    "label_smoothing": 0.04,
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

    # Pseudo fields are intentionally inert in the control arm.
    "pseudo_conf_min": 0.82,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 0.0,
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

    "notes": "Stage2 balanced control: no pseudo labels, strict 7-class 1:1 batches.",
}


if __name__ == "__main__":
    train_core.main("stage2", CONFIG)
