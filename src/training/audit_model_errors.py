from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES, MetricAccumulator, save_confusion_png, save_metrics_json
    from model_mbv3 import get_model, load_checkpoint_into_model
except Exception as exc:
    print("[import-error] Put this script in src/training beside dataset.py, metrics.py, model_mbv3.py")
    print(f"[import-error] current_dir={THIS_DIR}")
    print(f"[import-error] original={type(exc).__name__}: {exc}")
    raise


DEFAULT_PROJECT_ROOT = os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "NULL", "null"}


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to read json: {path} ({type(exc).__name__}: {exc})", flush=True)
    return {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def label_name(label_id: int) -> str:
    if 0 <= int(label_id) < len(LABELS):
        return str(LABELS[int(label_id)])
    return f"unknown_{label_id}"


def class_counts_from_dataset(ds: Dataset, num_classes: int = NUM_CLASSES) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    samples = getattr(ds, "samples", None)

    if samples is not None:
        for sample in samples:
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


def sample_meta(ds: Dataset, idx: int) -> Dict[str, Any]:
    samples = getattr(ds, "samples", None)
    if samples is None:
        return {}
    if 0 <= int(idx) < len(samples):
        raw = samples[int(idx)]
        if isinstance(raw, Mapping):
            return dict(raw)
    return {}


def make_deterministic_eval_dataset(csv_path: Path, img_base: Optional[Path], split_name: str) -> FER2013Hybrid:
    ds = FER2013Hybrid(
        str(csv_path),
        None if img_base is None else str(img_base),
        split_name,
        img_size=IMG_SIZE,
        include_label=True,
        strict=True,
    )

    # FER2013Hybrid applies train augmentation whenever ds.split == "train".
    # For audit we need deterministic eval transforms. Samples are already loaded,
    # so changing split after init is safe.
    ds.split = "audit"
    return ds


def make_loader(ds: Dataset, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def find_checkpoint_from_registry(project_root: Path, registry_csv: Path, selector: str) -> Optional[Path]:
    if not registry_csv.exists():
        return None

    try:
        with registry_csv.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    if not rows:
        return None

    selector_norm = str(selector).strip().lower()

    def to_float(row: Mapping[str, Any], key: str) -> float:
        try:
            return float(row.get(key, ""))
        except Exception:
            return -999.0

    if selector_norm in {"best-val", "best_val", "val"}:
        row = max(rows, key=lambda r: to_float(r, "val_macro_f1"))
    elif selector_norm in {"best-test", "best_test", "test"}:
        row = max(rows, key=lambda r: to_float(r, "test_macro_f1"))
    else:
        row = next((r for r in rows if str(r.get("run_id", "")) == selector), None)
        if row is None:
            return None

    path_text = str(row.get("checkpoint_path", "") or row.get("checkpoint_path_rel", ""))
    if not path_text:
        return None

    p = Path(path_text)
    return p if p.is_absolute() else project_root / p


def make_log_prior(
    train_csv: Path,
    img_base: Optional[Path],
    tau: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if abs(float(tau)) <= 1e-12:
        return None

    train_ds = make_deterministic_eval_dataset(train_csv, img_base, "train")
    counts = torch.tensor([max(1, x) for x in class_counts_from_dataset(train_ds)], dtype=torch.float32)
    prior = counts / counts.sum()
    return torch.log(prior).to(device)


@torch.no_grad()
def audit_split(
    model: torch.nn.Module,
    ds: Dataset,
    loader: DataLoader,
    device: torch.device,
    *,
    split: str,
    tau: float,
    log_prior: Optional[torch.Tensor],
    tta_horizontal_flip: bool,
    topk: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    accumulator = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)

    rows: List[Dict[str, Any]] = []
    global_offset = 0

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Audit loader must yield at least (images, labels)")

        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        if tta_horizontal_flip:
            logits = 0.5 * (logits + model(torch.flip(xb, dims=[-1])))

        if log_prior is not None and abs(float(tau)) > 1e-12:
            logits = logits - float(tau) * log_prior.view(1, -1)

        losses = criterion(logits, yb)
        probs = F.softmax(logits, dim=1)
        top_values, top_indices = torch.topk(probs, k=min(int(topk), NUM_CLASSES), dim=1)

        pred = top_indices[:, 0]
        conf = top_values[:, 0]
        second = top_values[:, 1] if top_values.shape[1] > 1 else torch.zeros_like(conf)
        margin = conf - second

        accumulator.update(logits, yb, loss=losses.mean())

        batch_size = int(yb.shape[0])
        for i in range(batch_size):
            dataset_index = global_offset + i
            meta = sample_meta(ds, dataset_index)

            true_id = int(yb[i].detach().cpu().item())
            pred_id = int(pred[i].detach().cpu().item())
            loss_value = float(losses[i].detach().cpu().item())
            conf_value = float(conf[i].detach().cpu().item())
            margin_value = float(margin[i].detach().cpu().item())
            correct = int(true_id == pred_id)

            row: Dict[str, Any] = {
                "split": split,
                "dataset_index": dataset_index,
                "row_index": meta.get("row_index", ""),
                "source_csv": meta.get("source_csv", getattr(ds, "csv_path", "")),
                "path": meta.get("path", ""),
                "has_pixels": bool(str(meta.get("pixels", "") or "")),
                "true_label_id": true_id,
                "true_label": label_name(true_id),
                "pred_label_id": pred_id,
                "pred_label": label_name(pred_id),
                "correct": correct,
                "confidence": conf_value,
                "margin": margin_value,
                "loss": loss_value,
                "audit_priority": loss_value * (1.0 + conf_value) if not correct else loss_value,
                "is_error": int(not correct),
                "is_high_conf_error": int((not correct) and conf_value >= 0.80),
                "is_low_margin": int(margin_value <= 0.10),
            }

            for rank in range(top_values.shape[1]):
                cls_id = int(top_indices[i, rank].detach().cpu().item())
                prob_value = float(top_values[i, rank].detach().cpu().item())
                row[f"top{rank + 1}_label_id"] = cls_id
                row[f"top{rank + 1}_label"] = label_name(cls_id)
                row[f"top{rank + 1}_prob"] = prob_value

            rows.append(row)

        global_offset += batch_size

    metrics = accumulator.compute().to_dict()
    return rows, metrics


def confusion_pair_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in rows:
        if int(row.get("correct", 0)) == 1:
            continue

        key = (str(row.get("true_label")), str(row.get("pred_label")))
        item = pairs.setdefault(
            key,
            {
                "true_label": key[0],
                "pred_label": key[1],
                "count": 0,
                "mean_confidence": 0.0,
                "mean_margin": 0.0,
                "mean_loss": 0.0,
                "high_conf_error_count": 0,
            },
        )

        item["count"] += 1
        item["mean_confidence"] += float(row.get("confidence", 0.0))
        item["mean_margin"] += float(row.get("margin", 0.0))
        item["mean_loss"] += float(row.get("loss", 0.0))
        item["high_conf_error_count"] += int(row.get("is_high_conf_error", 0))

    out: List[Dict[str, Any]] = []
    for item in pairs.values():
        n = max(1, int(item["count"]))
        item["mean_confidence"] = item["mean_confidence"] / n
        item["mean_margin"] = item["mean_margin"] / n
        item["mean_loss"] = item["mean_loss"] / n
        out.append(item)

    out.sort(key=lambda r: (int(r["count"]), float(r["mean_loss"])), reverse=True)
    return out


def per_class_error_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_class: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        label = str(row.get("true_label"))
        item = by_class.setdefault(
            label,
            {
                "true_label": label,
                "support": 0,
                "correct": 0,
                "errors": 0,
                "high_conf_errors": 0,
                "mean_confidence": 0.0,
                "mean_loss": 0.0,
            },
        )

        item["support"] += 1
        item["correct"] += int(row.get("correct", 0))
        item["errors"] += int(row.get("is_error", 0))
        item["high_conf_errors"] += int(row.get("is_high_conf_error", 0))
        item["mean_confidence"] += float(row.get("confidence", 0.0))
        item["mean_loss"] += float(row.get("loss", 0.0))

    out: List[Dict[str, Any]] = []
    for item in by_class.values():
        n = max(1, int(item["support"]))
        item["accuracy"] = float(item["correct"]) / n
        item["error_rate"] = float(item["errors"]) / n
        item["mean_confidence"] = item["mean_confidence"] / n
        item["mean_loss"] = item["mean_loss"] / n
        out.append(item)

    out.sort(key=lambda r: (float(r["error_rate"]), int(r["support"])), reverse=True)
    return out


AUDIT_FIELDNAMES = [
    "split",
    "dataset_index",
    "row_index",
    "source_csv",
    "path",
    "has_pixels",
    "true_label_id",
    "true_label",
    "pred_label_id",
    "pred_label",
    "correct",
    "confidence",
    "margin",
    "loss",
    "audit_priority",
    "is_error",
    "is_high_conf_error",
    "is_low_margin",
    "top1_label_id",
    "top1_label",
    "top1_prob",
    "top2_label_id",
    "top2_label",
    "top2_prob",
    "top3_label_id",
    "top3_label",
    "top3_prob",
    "top4_label_id",
    "top4_label",
    "top4_prob",
    "top5_label_id",
    "top5_label",
    "top5_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-sample prediction/error audit CSVs for FER checkpoints."
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT, help="Project root, e.g. F:\\fer-pi5")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Overrides --checkpoint-selector.")
    parser.add_argument(
        "--checkpoint-selector",
        default="best-val",
        help="Use checkpoint_registry.csv selector: best-val, best-test, or a run_id.",
    )
    parser.add_argument(
        "--registry-csv",
        default=None,
        help="Registry CSV path. Default: <project-root>/checkpoints/checkpoint_registry.csv",
    )
    parser.add_argument("--train-csv", default=r"data\csv\train.csv")
    parser.add_argument("--val-csv", default=r"data\csv\val.csv")
    parser.add_argument("--test-csv", default=r"data\csv\test.csv")
    parser.add_argument("--pseudo-csv", default=None, help="Optional pseudo CSV to audit as split=pseudo.")
    parser.add_argument("--img-base", default=None)
    parser.add_argument("--splits", default="val,test", help="Comma-separated: train,val,test,pseudo")
    parser.add_argument("--output-dir", default=None, help="Default: <project-root>/runs/audit/<run_id>")
    parser.add_argument("--model-variant", default="large", choices=["small", "large"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tta-horizontal-flip", action="store_true", help="Average original and horizontal-flip logits.")
    parser.add_argument("--tau", type=float, default=0.0, help="Logit adjustment tau. Positive subtracts tau*log(train_prior).")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--top-errors", type=int, default=500, help="Rows to save in top error shortlist per split.")
    parser.add_argument("--high-conf-threshold", type=float, default=0.80, help="Only used in filename/manifest; row flag uses 0.80.")
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    set_seed(int(args.seed))

    project_root = Path(args.project_root).expanduser().resolve()
    registry_csv = Path(args.registry_csv) if args.registry_csv else project_root / "checkpoints" / "checkpoint_registry.csv"

    checkpoint = resolve_path(project_root, args.checkpoint)
    if checkpoint is None:
        checkpoint = find_checkpoint_from_registry(project_root, registry_csv, str(args.checkpoint_selector))

    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Provide --checkpoint explicitly or run build_checkpoint_registry.py first.\n"
            f"checkpoint={checkpoint}\nregistry_csv={registry_csv}"
        )

    run_id = f"audit_{checkpoint.stem}_{now_stamp()}_seed{args.seed}"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "runs" / "audit" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    train_csv = resolve_path(project_root, args.train_csv)
    val_csv = resolve_path(project_root, args.val_csv)
    test_csv = resolve_path(project_root, args.test_csv)
    pseudo_csv = resolve_path(project_root, args.pseudo_csv)
    img_base = resolve_path(project_root, args.img_base)

    if train_csv is None or val_csv is None or test_csv is None:
        raise ValueError("train_csv, val_csv and test_csv are required")

    device = torch.device(str(args.device))
    log_prior = make_log_prior(train_csv, img_base, float(args.tau), device)

    print("=== FER model error audit ===", flush=True)
    print(f"project_root: {project_root}", flush=True)
    print(f"checkpoint  : {checkpoint}", flush=True)
    print(f"output_dir  : {output_dir}", flush=True)
    print(f"device      : {device}", flush=True)
    print(f"tau         : {args.tau}", flush=True)
    print(f"tta_flip    : {args.tta_horizontal_flip}", flush=True)

    model = get_model(
        variant=str(args.model_variant),
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
        verbose=True,
        compile_model=False,
    )
    load_checkpoint_into_model(model, checkpoint, device=device, strict=bool(args.strict_load))

    split_to_csv: Dict[str, Optional[Path]] = {
        "train": train_csv,
        "val": val_csv,
        "test": test_csv,
        "pseudo": pseudo_csv,
    }

    requested = [s.strip().lower() for s in str(args.splits).split(",") if s.strip()]
    all_rows: List[Dict[str, Any]] = []
    split_summaries: Dict[str, Any] = {}

    for split in requested:
        csv_path = split_to_csv.get(split)
        if csv_path is None:
            print(f"[skip] split={split}: no csv configured", flush=True)
            continue
        if not csv_path.exists():
            print(f"[skip] split={split}: csv missing: {csv_path}", flush=True)
            continue

        # For pseudo CSVs, prefer split="pseudo" because rows may have Usage=pseudo.
        # If the CSV has no Usage column, FER2013Hybrid accepts all rows for any split.
        dataset_split = "pseudo" if split == "pseudo" else split
        ds = make_deterministic_eval_dataset(csv_path, img_base, dataset_split)
        loader = make_loader(ds, int(args.batch_size), int(args.num_workers), pin_memory=True)

        print(f"[audit] split={split} count={len(ds)} csv={csv_path}", flush=True)

        rows, metrics = audit_split(
            model,
            ds,
            loader,
            device,
            split=split,
            tau=float(args.tau),
            log_prior=log_prior,
            tta_horizontal_flip=bool(args.tta_horizontal_flip),
            topk=int(args.topk),
        )

        split_csv = output_dir / f"audit_{split}.csv"
        write_csv(split_csv, rows, AUDIT_FIELDNAMES)

        errors = [r for r in rows if int(r.get("is_error", 0)) == 1]
        errors.sort(
            key=lambda r: (int(r.get("is_high_conf_error", 0)), float(r.get("audit_priority", 0.0))),
            reverse=True,
        )

        top_errors_csv = output_dir / f"audit_{split}_top_errors.csv"
        write_csv(top_errors_csv, errors[: int(args.top_errors)], AUDIT_FIELDNAMES)

        high_conf_errors = [
            r for r in rows
            if int(r.get("is_error", 0)) == 1 and float(r.get("confidence", 0.0)) >= float(args.high_conf_threshold)
        ]

        high_conf_csv = output_dir / f"audit_{split}_high_conf_errors.csv"
        write_csv(high_conf_csv, high_conf_errors, AUDIT_FIELDNAMES)

        pair_summary = confusion_pair_summary(rows)
        pair_csv = output_dir / f"audit_{split}_confusion_pairs.csv"
        write_csv(
            pair_csv,
            pair_summary,
            ["true_label", "pred_label", "count", "mean_confidence", "mean_margin", "mean_loss", "high_conf_error_count"],
        )

        class_summary = per_class_error_summary(rows)
        class_csv = output_dir / f"audit_{split}_per_class_summary.csv"
        write_csv(
            class_csv,
            class_summary,
            ["true_label", "support", "correct", "errors", "high_conf_errors", "accuracy", "error_rate", "mean_confidence", "mean_loss"],
        )

        metrics_path = output_dir / f"audit_{split}_metrics.json"
        save_metrics_json(metrics, metrics_path)
        save_confusion_png(metrics["confusion"], output_dir / f"audit_{split}_confusion.png", title=f"{split} Confusion Matrix")

        split_summaries[split] = {
            "csv": str(csv_path),
            "count": len(ds),
            "metrics": metrics,
            "audit_csv": str(split_csv),
            "top_errors_csv": str(top_errors_csv),
            "high_conf_errors_csv": str(high_conf_csv),
            "confusion_pairs_csv": str(pair_csv),
            "per_class_summary_csv": str(class_csv),
            "error_count": len(errors),
            "high_conf_error_count": len(high_conf_errors),
        }

        all_rows.extend(rows)

        print(
            f"[done] split={split} macro_f1={metrics['global_macro_f1']:.6f} "
            f"acc={metrics['accuracy']:.6f} errors={len(errors)} high_conf={len(high_conf_errors)}",
            flush=True,
        )

    if all_rows:
        combined_csv = output_dir / "audit_all_requested_splits.csv"
        write_csv(combined_csv, all_rows, AUDIT_FIELDNAMES)

        combined_errors = [r for r in all_rows if int(r.get("is_error", 0)) == 1]
        combined_errors.sort(
            key=lambda r: (int(r.get("is_high_conf_error", 0)), float(r.get("audit_priority", 0.0))),
            reverse=True,
        )
        write_csv(output_dir / "audit_all_top_errors.csv", combined_errors[: int(args.top_errors)], AUDIT_FIELDNAMES)
    else:
        combined_csv = output_dir / "audit_all_requested_splits.csv"

    manifest = {
        "status": "finished",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "checkpoint": str(checkpoint),
        "registry_csv": str(registry_csv),
        "output_dir": str(output_dir),
        "model_variant": str(args.model_variant),
        "device": str(device),
        "splits": requested,
        "tau": float(args.tau),
        "tta_horizontal_flip": bool(args.tta_horizontal_flip),
        "topk": int(args.topk),
        "high_conf_threshold": float(args.high_conf_threshold),
        "combined_csv": str(combined_csv),
        "split_summaries": split_summaries,
    }

    write_json(output_dir / "audit_manifest.json", manifest)

    print("=== audit complete ===", flush=True)
    print(f"output_dir : {output_dir}", flush=True)
    print(f"manifest   : {output_dir / 'audit_manifest.json'}", flush=True)

    for split, item in split_summaries.items():
        m = item["metrics"]
        print(
            f"{split:>6}: macro_f1={m['global_macro_f1']:.6f} "
            f"acc={m['accuracy']:.6f} high_conf_errors={item['high_conf_error_count']}",
            flush=True,
        )

    return manifest


if __name__ == "__main__":
    main()