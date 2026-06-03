from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
# This launcher is designed to live beside train_core.py.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from train_core import main
except Exception as exc:  # pragma: no cover - diagnostic path
    print("\n[Stage2 MATCHED CONTROL] Failed to import train_core.main.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {_THIS_DIR}")
    print("Expected     : train_core.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# You may override this without editing the script:
#   set FER_PROJECT_ROOT=F:\fer-pi5
PROJECT_ROOT = Path(os.environ.get("FER_PROJECT_ROOT", r"/"))


def _p(*parts: str) -> str:
    """Return a Windows-friendly project-relative path string."""
    return str(Path(*parts))


# ---------------------------------------------------------------------------
# Exact matched-control configuration
# ---------------------------------------------------------------------------
# This intentionally mirrors the latest Stage2 CLEAN resolved config:
#
#   clean run_id: stage2_20260601_165014_seed42
#   lr: 3e-4
#   warmup_epochs: 2
#   batch_size: 128
#   class_weights_from: labeled_train
#   pseudo_loss_scale: 0.35
#   pseudo_rampup_epochs: 10
#
# The only causal intervention is:
#
#   pseudo_csv = None
#
CONFIG: Dict[str, Any] = {
    # Identity
    "project_root": str(PROJECT_ROOT),
    "stage": "stage2",

    # Data
    "train_csv": _p("data", "csv", "train.csv"),
    "val_csv": _p("data", "csv", "val.csv"),
    "test_csv": _p("data", "csv", "test.csv"),
    "img_base": None,

    # No pseudo labels. This is the treatment removal.
    "pseudo_csv": None,
    "require_pseudo_conf": False,

    # Start from the verified Stage1 teacher, exactly like CLEAN.
    "init_ckpt": _p("checkpoints", "best_model_stage1_refactored.pth"),

    # Keep this run auditable and separate from normal Stage2 / Clean aliases.
    "best_alias_name": "best_model_stage2_refactored.pth",
    "log_alias_name": "train_stage2_refactored_log.csv",
    "write_checkpoint_alias": True,
    "alias_overwrite": True,

    # Output layout. Keep consistent with refactored train_core.
    "runs_dir": _p("runs", "training"),
    "checkpoint_alias_dir": _p("checkpoints"),

    # Model
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
    "compile_model": False,
    "strict_checkpoint_load": True,

    # Training — matched to Stage2 CLEAN.
    "device": "cuda",
    "epochs": 80,
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "drop_last_train": True,

    # Optimizer — matched to Stage2 CLEAN.
    "lr": 3e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,

    # Loss / imbalance handling — matched to Stage2 CLEAN.
    "label_smoothing": 0.04,
    "class_balance_beta": 0.995,
    "use_class_weights": True,
    "class_weights_from": "labeled_train",

    # These pseudo fields are intentionally kept identical to CLEAN where safe.
    # With pseudo_csv=None they should have no learning effect, but keeping them
    # matched makes the resolved config easier to compare.
    "pseudo_conf_min": 0.82,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 0.35,
    "pseudo_rampup_epochs": 10,

    # Validation / checkpointing — matched to Stage2 CLEAN.
    "val_interval": 1,
    "early_stop_patience": 20,
    "best_metric": "global_macro_f1",
    "evaluate_test_at_end": True,
    "save_last_every_epoch": True,

    # Reproducibility — matched to Stage2 CLEAN.
    "seed": 42,
    "deterministic_algorithms": False,
    "cudnn_benchmark": False,

    # Stability — matched to Stage2 CLEAN.
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    "run_name": None,
    "notes": (
        "Stage2 MATCHED CONTROL: supervised-only fine-tuning from Stage1 checkpoint. "
        "This run matches Stage2 CLEAN hyperparameters exactly and disables only "
        "pseudo_csv, so it can isolate whether CLEAN's gain comes from pseudo labels "
        "or from normal continued fine-tuning / the lr-warmup schedule."
    ),
}


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------
REQUIRED_RELATIVE_FILES: Tuple[str, ...] = (
    _p("data", "csv", "train.csv"),
    _p("data", "csv", "val.csv"),
    _p("data", "csv", "test.csv"),
    _p("checkpoints", "best_model_stage1_refactored.pth"),
)

MATCHED_CLEAN_FIELDS: Tuple[str, ...] = (
    "epochs",
    "batch_size",
    "lr",
    "lr_floor",
    "warmup_epochs",
    "weight_decay",
    "label_smoothing",
    "class_balance_beta",
    "use_class_weights",
    "class_weights_from",
    "pseudo_conf_min",
    "pseudo_conf_power",
    "pseudo_loss_scale",
    "pseudo_rampup_epochs",
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
)


def _resolve_project_path(value: str | None) -> Path | None:
    """Resolve a config path relative to PROJECT_ROOT, preserving None."""
    if value in (None, "", "None"):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _print_config_audit(keys: Iterable[str]) -> None:
    print("\n[Stage2 MATCHED CONTROL] Matched CLEAN fields:")
    for key in keys:
        print(f"  {key}: {CONFIG.get(key)!r}")

    print("\n[Stage2 MATCHED CONTROL] Causal intervention:")
    print("  pseudo_csv: None")
    print("  require_pseudo_conf: False")
    print("  aliases:")
    print(f"    best_alias_name: {CONFIG['best_alias_name']}")
    print(f"    log_alias_name : {CONFIG['log_alias_name']}")


def _preflight() -> None:
    print("\n[Stage2 MATCHED CONTROL] Preflight")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")

    missing = []
    for rel in REQUIRED_RELATIVE_FILES:
        full = _resolve_project_path(rel)
        if full is None or not full.exists():
            missing.append(str(full))

    if missing:
        print("\n[Stage2 MATCHED CONTROL] Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Required project files are missing. "
            "Check FER_PROJECT_ROOT or put this script under F:\\fer-pi5\\src\\training."
        )

    if CONFIG["pseudo_csv"] is not None:
        raise ValueError("Matched control must have pseudo_csv=None.")

    if CONFIG["require_pseudo_conf"] is not False:
        raise ValueError("Matched control must have require_pseudo_conf=False.")

    _print_config_audit(MATCHED_CLEAN_FIELDS)


def run() -> None:
    _preflight()
    print("\n[Stage2 MATCHED CONTROL] Starting training through train_core.main(...)\n")
    main("stage2", CONFIG)


if __name__ == "__main__":
    run()
