"""PyCharm lazy-run launcher for refactored Stage 3 semi-supervised training."""
from __future__ import annotations

from train_core import main

CONFIG = {
    "project_root": r"F:\fer-pi5",
    "stage": "stage3",
    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "pseudo_csv": r"data\csv\pseudo_labeled_stage2.csv",
    "init_ckpt": r"checkpoints\best_model_stage2_refactored.pth",
    "img_base": None,
    "epochs": 200,
    "batch_size": 128,
    "num_workers": 4,
    "lr": 5e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,
    "label_smoothing": 0.04,
    "class_balance_beta": 0.995,
    "early_stop_patience": 20,
    "seed": 42,
    "pretrained": False,
    "use_amp": True,
    "pseudo_conf_min": 0.0,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 1.0,
    "pseudo_rampup_epochs": 5,
    "require_pseudo_conf": True,
    "notes": "Refactored Stage 3: strict pseudo confidence weighting with global Macro-F1 checkpointing.",
}

if __name__ == "__main__":
    main("stage3", CONFIG)
