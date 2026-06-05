from __future__ import annotations

from train_core import main

# Edit only this block when needed. Running this file directly starts training.
CONFIG = {
    "project_root": r"D:\fer-pi5",
    "stage": "stage1",
    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "img_base": None,
    "epochs": 200,
    "batch_size": 98,
    "num_workers": 4,
    "lr": 3e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 5,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "class_balance_beta": 0.995,
    "early_stop_patience": 20,
    "seed": 42,
    "pretrained": True,
    "aux_loss_weight": 0.0,
    "model_variant": "efficientnet_b0",
    "use_amp": True,
    "use_checkpoint": False,
    "compile_model": False,
    "grad_accum_steps": 1,
    "per_class_limit": 0,
    "best_alias_name": "best_model_stage1_efficientnet_b0.pth",
    "log_alias_name": "train_stage1_efficientnet_b0_log.csv",
    "write_checkpoint_alias": True,
    "alias_overwrite": True,
    "notes": "EfficientNet-B0 Stage 1 baseline with ImageNet pretrained weights.",
}

if __name__ == "__main__":
    main("stage1", CONFIG)
