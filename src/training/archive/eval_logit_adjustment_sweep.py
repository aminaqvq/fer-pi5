from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES, MetricAccumulator, save_confusion_png, save_metrics_json
    from model_mbv3 import get_model, load_checkpoint_into_model
except Exception:
    print("[import-error] Put this script in src/training beside dataset.py, metrics.py, model_mbv3.py")
    raise


CONFIG: Dict[str, Any] = {
    # Override without editing:
    #   set FER_PROJECT_ROOT=F:\fer-pi5
    "project_root": os.environ.get("FER_PROJECT_ROOT", r"F:\fer-pi5"),

    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "img_base": None,

    "model_variant": "large",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,

    # The script skips missing checkpoints automatically.
    "checkpoints": [
        ["stage1", r"checkpoints\best_model_stage1_refactored.pth"],
        ["stage2_balanced_clean", r"checkpoints\best_model_stage2_balanced_clean.pth"],
        ["stage2_balanced_control", r"checkpoints\best_model_stage2_balanced_control.pth"],
        ["stage2_crt_head", r"checkpoints\best_model_stage2_crt_head.pth"],
    ],

    # tau=0 means original logits.
    # Positive tau subtracts tau * log(train_prior), usually helping rare classes.
    "taus": [
        -1.00, -0.75, -0.50, -0.25, 0.00, 0.10, 0.20, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10,
        1.20, 1.40, 1.60, 1.80, 2.00, 2.25, 2.50,
    ],

    "runs_dir": r"runs\evaluation",
    "alias_dir": r"checkpoints",
    "alias_name": "logit_adjustment_sweep_latest.json",
}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def resolve_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    root = Path(str(out["project_root"])).expanduser().resolve()
    out["project_root"] = str(root)
    for key in ("train_csv", "val_csv", "test_csv", "img_base", "runs_dir", "alias_dir"):
        out[key] = None if is_none_like(out.get(key)) else str(resolve_path(root, out[key]))

    ckpts = []
    for name, path in out["checkpoints"]:
        p = resolve_path(root, path)
        ckpts.append([str(name), str(p) if p is not None else None])
    out["checkpoints"] = ckpts
    return out


def class_counts_from_dataset(ds: Dataset, num_classes: int = NUM_CLASSES) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    if hasattr(ds, "samples"):
        for sample in getattr(ds, "samples"):
            y = int(sample.get("label", -1))
            if 0 <= y < num_classes:
                counts[y] += 1
        return counts

    for idx in range(len(ds)):
        item = ds[idx]
        y = int(item[1])
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts


def make_loaders(cfg: Mapping[str, Any]):
    train_ds = FER2013Hybrid(str(cfg["train_csv"]), cfg.get("img_base"), "train", img_size=IMG_SIZE, include_label=True, strict=True)
    val_ds = FER2013Hybrid(str(cfg["val_csv"]), cfg.get("img_base"), "val", img_size=IMG_SIZE, include_label=True, strict=True)
    test_ds = FER2013Hybrid(str(cfg["test_csv"]), cfg.get("img_base"), "test", img_size=IMG_SIZE, include_label=True, strict=True)

    num_workers = int(cfg.get("num_workers", 4))
    common = {
        "batch_size": int(cfg.get("batch_size", 256)),
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        common["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        common["persistent_workers"] = bool(cfg.get("persistent_workers", True))

    return train_ds, val_ds, test_ds, DataLoader(val_ds, **common), DataLoader(test_ds, **common)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    log_prior: torch.Tensor,
    tau: float,
) -> Dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    criterion = nn.CrossEntropyLoss()
    log_prior = log_prior.to(device)

    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if abs(float(tau)) > 1e-12:
            logits = logits - float(tau) * log_prior.view(1, -1)
        loss = criterion(logits, yb)
        acc.update(logits, yb, loss=loss)

    return acc.compute().to_dict()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def main() -> Dict[str, Any]:
    cfg = resolve_config(CONFIG)
    device = torch.device(str(cfg["device"]))

    run_id = f"logit_adjustment_sweep_{now_stamp()}"
    run_dir = Path(str(cfg["runs_dir"])) / run_id
    eval_dir = run_dir / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=== FER Logit Adjustment Sweep ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"device : {device}", flush=True)

    train_ds, val_ds, test_ds, val_loader, test_loader = make_loaders(cfg)
    train_counts = class_counts_from_dataset(train_ds)
    counts = torch.tensor([max(1, int(x)) for x in train_counts], dtype=torch.float32)
    priors = counts / counts.sum()
    log_prior = torch.log(priors)

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []

    summary_csv = run_dir / "logit_adjustment_sweep_summary.csv"
    csv_fields = [
        "checkpoint_name", "checkpoint_path", "tau",
        "val_global_macro_f1", "val_acc",
        "test_global_macro_f1", "test_acc",
    ]

    for ckpt_name, ckpt_path_raw in cfg["checkpoints"]:
        if ckpt_path_raw is None:
            continue
        ckpt_path = Path(ckpt_path_raw)
        if not ckpt_path.exists():
            skipped.append({"name": ckpt_name, "path": str(ckpt_path), "reason": "missing"})
            print(f"[skip] {ckpt_name}: missing {ckpt_path}", flush=True)
            continue

        print(f"[eval] {ckpt_name}: {ckpt_path}", flush=True)
        model = get_model(
            variant=str(cfg["model_variant"]),
            num_classes=NUM_CLASSES,
            pretrained=False,
            device=device,
            verbose=False,
            compile_model=False,
        )
        load_checkpoint_into_model(model, ckpt_path, device=device, strict=True)

        rows: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        for tau in [float(x) for x in cfg["taus"]]:
            val = evaluate(model, val_loader, device, log_prior=log_prior, tau=tau)
            test = evaluate(model, test_loader, device, log_prior=log_prior, tau=tau)
            row = {
                "checkpoint_name": ckpt_name,
                "checkpoint_path": str(ckpt_path),
                "tau": tau,
                "val_global_macro_f1": float(val["global_macro_f1"]),
                "val_acc": float(val["accuracy"]),
                "test_global_macro_f1": float(test["global_macro_f1"]),
                "test_acc": float(test["accuracy"]),
                "val": val,
                "test": test,
            }
            rows.append(row)
            write_csv_row(summary_csv, {k: row[k] for k in csv_fields}, csv_fields)
            if best is None or row["val_global_macro_f1"] > best["val_global_macro_f1"]:
                best = row

        assert best is not None
        ckpt_payload = {
            "checkpoint_name": ckpt_name,
            "checkpoint_path": str(ckpt_path),
            "rows": rows,
            "best_by_val_macro_f1": best,
        }
        write_json(eval_dir / f"{ckpt_name}_logit_sweep.json", ckpt_payload)
        save_metrics_json(best["val"], eval_dir / f"{ckpt_name}_best_tau_val_metrics.json")
        save_metrics_json(best["test"], eval_dir / f"{ckpt_name}_best_tau_test_metrics.json")
        save_confusion_png(best["val"]["confusion"], eval_dir / f"{ckpt_name}_best_tau_val_confusion.png", title=f"{ckpt_name} Validation Confusion")
        save_confusion_png(best["test"]["confusion"], eval_dir / f"{ckpt_name}_best_tau_test_confusion.png", title=f"{ckpt_name} Test Confusion")

        print(
            f"[best] {ckpt_name}: tau={best['tau']} "
            f"val_f1={best['val_global_macro_f1']:.6f} "
            f"test_f1={best['test_global_macro_f1']:.6f}",
            flush=True,
        )
        results.append(ckpt_payload)

    final = {
        "method": "post_hoc_logit_adjustment",
        "formula": "adjusted_logits = logits - tau * log(train_class_prior)",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "labels": list(LABELS),
        "train_class_counts": {LABELS[i]: int(train_counts[i]) for i in range(NUM_CLASSES)},
        "train_class_priors": {LABELS[i]: float(priors[i].item()) for i in range(NUM_CLASSES)},
        "taus": [float(x) for x in cfg["taus"]],
        "results": results,
        "skipped": skipped,
        "summary_csv": str(summary_csv),
    }

    if results:
        final["global_best_checkpoint_by_val_macro_f1"] = max(
            (r["best_by_val_macro_f1"] for r in results),
            key=lambda x: x["val_global_macro_f1"],
        )

    write_json(run_dir / "logit_adjustment_sweep_all.json", final)

    alias_path = Path(str(cfg["alias_dir"])) / str(cfg["alias_name"])
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(alias_path, final)

    print("=== Sweep complete ===", flush=True)
    if "global_best_checkpoint_by_val_macro_f1" in final:
        print(json.dumps(final["global_best_checkpoint_by_val_macro_f1"], indent=2, ensure_ascii=False), flush=True)
    print(f"summary_csv: {summary_csv}", flush=True)
    return final


if __name__ == "__main__":
    main()