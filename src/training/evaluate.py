from __future__ import annotations

import argparse
import csv
import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import FER2013Hybrid, IMG_SIZE
    from .metrics import LABELS, NUM_CLASSES, evaluate_model, save_confusion_png, save_metrics_json
    from .model_mbv3 import get_model, load_checkpoint_into_model
except ImportError:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES, evaluate_model, save_confusion_png, save_metrics_json
    from model_mbv3 import get_model, load_checkpoint_into_model

# =========================
# PyCharm lazy-run config
# =========================
DEFAULT_CONFIG: Dict[str, Any] = {
    "project_root": r"F:\fer-pi5",
    "csv_base": r"F:\fer-pi5\data\csv",
    "img_base": None,
    "save_dir": r"F:\fer-pi5\checkpoints",
    "checkpoint": r"F:\fer-pi5\checkpoints\best_model_stage3.pth",
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
    "batch_size": 128,
    "num_workers": 0,  # safest for PyCharm/Windows evaluation
    "pin_memory": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "splits": "both",  # val, test, both
    "tta_horizontal_flip": True,
    "strict_checkpoint": True,
    "write_confusion_png": True,
}


def resolve_path(project_root: str | Path, value: Optional[Any]) -> Optional[str]:
    if value in (None, "", "None"):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str(Path(project_root) / path)


def make_eval_loader(
    csv_path: str,
    split: str,
    cfg: Dict[str, Any],
) -> DataLoader:
    dataset = FER2013Hybrid(
        csv_path=csv_path,
        img_root=cfg.get("img_base"),
        split=split,
        img_size=int(cfg.get("img_size", IMG_SIZE)),
        two_views=False,
        include_label=True,
        strict=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )


def run_evaluation(
    cfg: Dict[str, Any],
    eval_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(cfg or {})
    if eval_overrides:
        merged.update(eval_overrides)

    project_root = Path(str(merged.get("project_root", r"F:\fer-pi5")))
    csv_base = Path(resolve_path(project_root, merged["csv_base"]) or "")
    save_dir = Path(resolve_path(project_root, merged["save_dir"]) or "")
    checkpoint = Path(resolve_path(project_root, merged["checkpoint"]) or "")
    save_dir.mkdir(parents=True, exist_ok=True)

    split_mode = str(merged.get("splits", "both")).lower()
    requested = ["val", "test"] if split_mode == "both" else [split_mode]
    for split in requested:
        if split not in {"val", "test"}:
            raise ValueError("splits must be one of: val, test, both")

    device = torch.device(str(merged.get("device", "cpu")))
    print(f"[evaluate] device={device}")
    print(f"[evaluate] checkpoint={checkpoint}")
    print(f"[evaluate] csv_base={csv_base}")

    model = get_model(
        variant=str(merged.get("model_variant", "large")),
        num_classes=int(merged.get("num_classes", NUM_CLASSES)),
        pretrained=bool(merged.get("pretrained", False)),
        device=device,
        verbose=True,
        compile_model=False,
    )
    load_checkpoint_into_model(
        model,
        checkpoint,
        device=device,
        strict=bool(merged.get("strict_checkpoint", True)),
    )
    model.to(device).eval()

    results: Dict[str, Any] = {
        "timestamp": int(time.time()),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "label_order": list(LABELS),
        "metric_contract": "global split-level confusion matrix; no averaged batch macro-F1",
        "tta_horizontal_flip": bool(merged.get("tta_horizontal_flip", True)),
        "splits": {},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in merged.items()},
    }

    for split in requested:
        csv_path = csv_base / f"{split}.csv"
        loader = make_eval_loader(str(csv_path), split, merged)
        metric = evaluate_model(
            model,
            loader,
            device=device,
            num_classes=int(merged.get("num_classes", NUM_CLASSES)),
            labels=LABELS,
            tta_horizontal_flip=bool(merged.get("tta_horizontal_flip", True)),
        )
        payload = metric.to_dict()
        results["splits"][split] = payload
        print(
            f"[evaluate:{split}] "
            f"loss={payload['loss']:.6f} "
            f"acc={payload['accuracy']:.6f} "
            f"global_macro_f1={payload['global_macro_f1']:.6f} "
            f"n={payload['total']}",
            flush=True,
        )
        if bool(merged.get("write_confusion_png", True)):
            save_confusion_png(
                payload["confusion"],
                save_dir / f"{split}_confusion_refactored.png",
                title=f"{split.capitalize()} Confusion Matrix",
                labels=LABELS,
            )

    summary_path = save_dir / "metrics_summary_refactored.json"
    save_metrics_json(results, summary_path)
    log_path = save_dir / "analysis_log_refactored.csv"
    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time", "checkpoint", "split", "loss", "accuracy", "global_macro_f1", "total", "tta_horizontal_flip"])
        for split, payload in results["splits"].items():
            writer.writerow([
                results["timestamp"],
                results["checkpoint_name"],
                split,
                payload["loss"],
                payload["accuracy"],
                payload["global_macro_f1"],
                payload["total"],
                int(results["tta_horizontal_flip"]),
            ])
    print(f"[evaluate] wrote {summary_path}")
    print(f"[evaluate] appended {log_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FER Pi5 checkpoint with global Macro-F1")
    parser.add_argument("--project-root", default=DEFAULT_CONFIG["project_root"])
    parser.add_argument("--csv-base", default=DEFAULT_CONFIG["csv_base"])
    parser.add_argument("--img-base", default=None)
    parser.add_argument("--save-dir", default=DEFAULT_CONFIG["save_dir"])
    parser.add_argument("--checkpoint", default=DEFAULT_CONFIG["checkpoint"])
    parser.add_argument("--model-variant", default=DEFAULT_CONFIG["model_variant"], choices=["small", "large"])
    parser.add_argument("--splits", default=DEFAULT_CONFIG["splits"], choices=["val", "test", "both"])
    parser.add_argument("--batch-size", type=int, default=int(DEFAULT_CONFIG["batch_size"]))
    parser.add_argument("--num-workers", type=int, default=int(DEFAULT_CONFIG["num_workers"]))
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--allow-nonstrict-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Lazy-run when launched directly from PyCharm with no arguments.
    if len(sys.argv) == 1:
        print("=== PyCharm lazy-run mode enabled ===")
        run_evaluation(DEFAULT_CONFIG)
        return
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(vars(args))
    cfg["tta_horizontal_flip"] = not bool(args.no_tta)
    cfg["strict_checkpoint"] = not bool(args.allow_nonstrict_checkpoint)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
