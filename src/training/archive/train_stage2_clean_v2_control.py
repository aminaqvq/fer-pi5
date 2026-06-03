from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
# This launcher is designed to live beside:
#   train_core.py
#   balanced_sampler_patch.py
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    import train_core
    from balanced_sampler_patch import install_balanced_batch_sampler
except Exception as exc:
    print("\n[Stage2 CLEAN_V2 CONTROL] Import failed.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {_THIS_DIR}")
    print("Expected     : train_core.py and balanced_sampler_patch.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# ---------------------------------------------------------------------------
# Install balanced-batch support
# ---------------------------------------------------------------------------
# train_core.py currently uses a standard DataLoader.
# balanced_sampler_patch.py monkey-patches train_core.make_loader so that
# CONFIG["sampling_strategy"] == "balanced_batch" enables strict 7-class
# balanced mini-batches.
install_balanced_batch_sampler(train_core)


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# Strong recommendation:
#   set FER_PROJECT_ROOT=F:\fer-pi5
#
# This avoids D:/F: ambiguity when running on different machines.
PROJECT_ROOT = Path(os.environ.get("FER_PROJECT_ROOT", r"/"))


def _p(*parts: str) -> str:
    """Return a project-relative path string."""
    return str(Path(*parts))


def _resolve_project_path(value: str | None) -> Path | None:
    """Resolve a config path relative to PROJECT_ROOT, preserving None."""
    if value in (None, "", "None", "none", "null"):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Stage2 Clean-V2 Control
# ---------------------------------------------------------------------------
# Purpose:
#   Verify whether manually reviewed clean_v2 data improves validation/test
#   performance before doing any more pseudo-labeling, teacher distillation,
#   or Stage3 work.
#
# This run intentionally does NOT use pseudo labels.
#
# Data:
#   train_csv = data/csv/clean_v2/train_v2_clean.csv
#   val_csv   = data/csv/val.csv
#   test_csv  = data/csv/test.csv
#
# Training:
#   init_ckpt = best_model_stage1_refactored.pth
#   sampler   = strict 7-class balanced batch
#   batch     = 126 = 7 classes * 18 samples per class
#
# Success condition:
#   Beat the current historical best:
#       val  Macro-F1 ~= 0.68657
#       test Macro-F1 ~= 0.68445
CONFIG: Dict[str, Any] = {
    # Identity
    "project_root": str(PROJECT_ROOT),
    "stage": "stage2",

    # Data
    "train_csv": _p("data", "csv", "clean_v2", "train_v2_clean.csv"),
    "val_csv": _p("data", "csv", "val.csv"),
    "test_csv": _p("data", "csv", "test.csv"),
    "img_base": None,

    # No pseudo labels in this diagnostic control.
    "pseudo_csv": None,
    "require_pseudo_conf": False,

    # Start from the verified supervised Stage1 checkpoint.
    "init_ckpt": _p("checkpoints", "best_model_stage1_refactored.pth"),

    # Output aliases.
    # These names are intentionally unique so they do not overwrite older
    # Stage2 balanced-clean aliases.
    "best_alias_name": "best_model_stage2_clean_v2_control.pth",
    "log_alias_name": "train_stage2_clean_v2_control_log.csv",
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
    # Important:
    #   use_class_weights=False because strict balanced sampling already
    #   corrects class frequency inside each mini-batch. Using both can
    #   over-correct minority classes.
    "label_smoothing": 0.04,
    "use_class_weights": False,
    "class_balance_beta": 0.995,
    "class_weights_from": "labeled_train",

    # Balanced sampler
    # batch_size=126 and balanced_samples_per_class_per_batch=18 means:
    #   18 anger
    #   18 disgust
    #   18 fear
    #   18 happy
    #   18 sad
    #   18 surprise
    #   18 neutral
    # per mini-batch.
    "sampling_strategy": "balanced_batch",
    "balanced_per_class": "auto_min",
    "balanced_per_class_source": "labeled_train",
    "balanced_samples_per_class_per_batch": 18,
    "balanced_strict_batch_size": True,
    "balanced_replacement": False,

    # Pseudo fields are intentionally inert here.
    # They are kept explicit so the manifest is easy to audit.
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

    # Optional explicit run_name. Leave None for timestamped run directory.
    "run_name": None,

    "notes": (
        "Stage2 Clean-V2 Control: train on manually reviewed clean_v2 train CSV, "
        "no pseudo labels, initialized from Stage1, strict 7-class balanced batch. "
        "This experiment tests whether the first 500 manually reviewed OOF issues "
        "improve val/test before continuing data cleaning or teacher distillation."
    ),
}


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
REQUIRED_RELATIVE_FILES: Tuple[str, ...] = (
    _p("data", "csv", "clean_v2", "train_v2_clean.csv"),
    _p("data", "csv", "val.csv"),
    _p("data", "csv", "test.csv"),
    _p("checkpoints", "best_model_stage1_refactored.pth"),
)


EXPECTED_CONTROL_FIELDS: Tuple[str, ...] = (
    "train_csv",
    "val_csv",
    "test_csv",
    "pseudo_csv",
    "init_ckpt",
    "epochs",
    "batch_size",
    "lr",
    "lr_floor",
    "warmup_epochs",
    "weight_decay",
    "label_smoothing",
    "use_class_weights",
    "sampling_strategy",
    "balanced_per_class",
    "balanced_per_class_source",
    "balanced_samples_per_class_per_batch",
    "balanced_strict_batch_size",
    "balanced_replacement",
    "val_interval",
    "early_stop_patience",
    "best_metric",
    "evaluate_test_at_end",
    "save_last_every_epoch",
    "seed",
    "pretrained",
    "use_amp",
    "grad_clip",
    "max_norm",
    "best_alias_name",
    "log_alias_name",
)


def _print_config_audit(keys: Iterable[str]) -> None:
    print("\n[Stage2 CLEAN_V2 CONTROL] Config audit:")
    for key in keys:
        print(f"  {key}: {CONFIG.get(key)!r}")

    print("\n[Stage2 CLEAN_V2 CONTROL] Intent:")
    print("  This is a no-pseudo diagnostic control.")
    print("  It tests clean_v2 data quality only.")
    print("  It should not overwrite historical best balanced-clean aliases.")


def _preflight() -> None:
    print("\n[Stage2 CLEAN_V2 CONTROL] Preflight")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")

    missing = []
    for rel in REQUIRED_RELATIVE_FILES:
        full = _resolve_project_path(rel)
        if full is None or not full.exists():
            missing.append(str(full))

    if missing:
        print("\n[Stage2 CLEAN_V2 CONTROL] Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Required files are missing. Check FER_PROJECT_ROOT and make sure "
            "apply_manual_review.py has generated data/csv/clean_v2/train_v2_clean.csv."
        )

    if CONFIG["pseudo_csv"] is not None:
        raise ValueError("Clean-V2 control must have pseudo_csv=None.")

    if CONFIG["require_pseudo_conf"] is not False:
        raise ValueError("Clean-V2 control must have require_pseudo_conf=False.")

    if CONFIG["use_class_weights"] is not False:
        raise ValueError("Clean-V2 balanced control must have use_class_weights=False.")

    if CONFIG["batch_size"] != 7 * CONFIG["balanced_samples_per_class_per_batch"]:
        raise ValueError(
            "batch_size must equal 7 * balanced_samples_per_class_per_batch. "
            f"Got batch_size={CONFIG['batch_size']} and "
            f"balanced_samples_per_class_per_batch={CONFIG['balanced_samples_per_class_per_batch']}."
        )

    if CONFIG["best_alias_name"] in {
        "best_model_stage2_balanced_clean.pth",
        "best_model_stage2_refactored.pth",
        "best_model_stage1_refactored.pth",
    }:
        raise ValueError(
            "Alias name is unsafe and may overwrite important historical checkpoints."
        )

    _print_config_audit(EXPECTED_CONTROL_FIELDS)


def run() -> None:
    _preflight()
    print("\n[Stage2 CLEAN_V2 CONTROL] Starting training through train_core.main(...)\n")
    train_core.main("stage2", CONFIG)


if __name__ == "__main__":
    run()