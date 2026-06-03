from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Override without editing:
    #   set FER_PROJECT_ROOT=F:\fer-pi5
    #   python out_of_fold_train_audit.py --project-root F:\fer-pi5
    "project_root": os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"),
    "train_csv": r"data\csv\train.csv",
    "img_base": None,

    # Output
    "runs_dir": r"runs\audit",
    "run_name": None,

    # OOF protocol
    "num_folds": 5,
    "inner_val_fraction": 0.10,
    "seed": 42,
    "resume_existing": True,

    # Model
    "model_variant": "large",
    "pretrained": True,
    "compile_model": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Optional warm start. IMPORTANT: For true OOF label audit, do NOT initialize
    # from a checkpoint trained on the full train.csv. Leave None unless the
    # checkpoint is external or otherwise guaranteed not to have seen train.csv.
    "init_ckpt": None,
    "strict_checkpoint_load": True,

    # Training each fold model
    "epochs": 100,
    "early_stop_patience": 8,
    "batch_size": 126,
    "samples_per_class_per_batch": 18,
    "lr": 5e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,
    "label_smoothing": 0.04,
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    # DataLoader
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,

    # Label issue ranking thresholds
    "high_conf_threshold": 0.80,
    "high_margin_threshold": 0.50,
    "low_margin_threshold": 0.15,
    "low_self_confidence_threshold": 0.20,

    # Manual review packet
    "top_issue_rows": 1000,
    "manual_review_rows": 500,
    "contact_sheet_top_n": 500,
    "contact_sheet_cols": 5,
    "contact_sheet_thumb": 128,
    "contact_sheet_per_pair_top_n": 60,
    "contact_sheet_top_pairs": 8,

    # File names
    "oof_predictions_name": "oof_train_predictions.csv",
    "label_issues_name": "oof_train_label_issues.csv",
    "confusion_pairs_name": "oof_train_confusion_pairs.csv",
    "manual_template_name": "manual_label_review_template.csv",
}


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
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
    for key in ("train_csv", "img_base", "runs_dir", "init_ckpt"):
        out[key] = None if is_none_like(out.get(key)) else str(resolve_path(root, out[key]))

    out["num_folds"] = int(out["num_folds"])
    out["inner_val_fraction"] = float(out["inner_val_fraction"])
    out["seed"] = int(out["seed"])
    out["batch_size"] = int(out["batch_size"])
    out["samples_per_class_per_batch"] = int(out["samples_per_class_per_batch"])

    if out["num_folds"] < 2:
        raise ValueError("num_folds must be >= 2")
    if not (0.0 < out["inner_val_fraction"] < 0.5):
        raise ValueError("inner_val_fraction should be in (0, 0.5)")
    if out["batch_size"] != NUM_CLASSES * out["samples_per_class_per_batch"]:
        raise ValueError(
            "For strict class-balanced training, batch_size must equal "
            "NUM_CLASSES * samples_per_class_per_batch. "
            f"Got {out['batch_size']} vs {NUM_CLASSES}*{out['samples_per_class_per_batch']}."
        )
    return out


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def append_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def get_sample_label(sample: Mapping[str, Any]) -> int:
    return int(sample.get("label", -1))


def dataset_labels(ds: Dataset) -> List[int]:
    if hasattr(ds, "samples"):
        return [get_sample_label(x) for x in getattr(ds, "samples")]
    labels: List[int] = []
    for i in range(len(ds)):
        item = ds[i]
        labels.append(int(item[1]))
    return labels


def class_counts(labels: Sequence[int], indices: Optional[Sequence[int]] = None) -> List[int]:
    counts = [0 for _ in range(NUM_CLASSES)]
    iterable = range(len(labels)) if indices is None else indices
    for idx in iterable:
        y = int(labels[int(idx)])
        if 0 <= y < NUM_CLASSES:
            counts[y] += 1
    return counts


class EvalViewDataset(Dataset):
    """Evaluation-transform view of a FER2013Hybrid dataset.

    FER2013Hybrid uses train augmentation when split == "train". For OOF
    predictions/contact-sheet ranking, we need deterministic eval transforms
    while preserving exactly the same sample order as train.csv.
    """

    def __init__(self, base: FER2013Hybrid) -> None:
        self.base = base
        self.samples = base.samples

    def __len__(self) -> int:
        return len(self.base.samples)

    def __getitem__(self, idx: int):
        item = self.base.samples[int(idx)]
        img = self.base._load_image(item)
        label = int(item.get("label", -1))
        return self.base.t_eval(img), torch.tensor(label, dtype=torch.long)

    def raw_image(self, idx: int) -> Image.Image:
        item = self.base.samples[int(idx)]
        return self.base._load_image(item).convert("RGB")


# ---------------------------------------------------------------------------
# Fold splitting
# ---------------------------------------------------------------------------
def make_stratified_folds(labels: Sequence[int], n_splits: int, seed: int) -> List[List[int]]:
    rng = random.Random(int(seed))
    buckets: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for idx, y in enumerate(labels):
        y = int(y)
        if 0 <= y < NUM_CLASSES:
            buckets[y].append(int(idx))
        else:
            raise ValueError(f"Invalid label at dataset index {idx}: {y}")

    for c, bucket in buckets.items():
        if len(bucket) < n_splits:
            raise ValueError(f"Class {LABELS[c]} has {len(bucket)} samples, fewer than n_splits={n_splits}")
        rng.shuffle(bucket)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for c in range(NUM_CLASSES):
        for pos, idx in enumerate(buckets[c]):
            folds[pos % n_splits].append(idx)

    for fold in folds:
        fold.sort()
    return folds


def stratified_inner_split(
    train_indices: Sequence[int],
    labels: Sequence[int],
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(int(seed))
    by_class: Dict[int, List[int]] = {c: [] for c in range(NUM_CLASSES)}
    for idx in train_indices:
        by_class[int(labels[int(idx)])].append(int(idx))

    fit: List[int] = []
    inner_val: List[int] = []
    for c in range(NUM_CLASSES):
        bucket = list(by_class[c])
        rng.shuffle(bucket)
        n_val = max(1, int(round(len(bucket) * float(val_fraction))))
        n_val = min(n_val, max(1, len(bucket) - 1))
        inner_val.extend(bucket[:n_val])
        fit.extend(bucket[n_val:])

    fit.sort()
    inner_val.sort()
    return fit, inner_val


# ---------------------------------------------------------------------------
# Balanced class sampler
# ---------------------------------------------------------------------------
class BalancedClassBatchSampler(Sampler[List[int]]):
    """Strict class-balanced batch sampler over a Subset-like index space.

    The labels passed here must be aligned to the dataset that DataLoader sees.
    If DataLoader receives Subset(train_dataset, fit_indices), labels must have
    length len(fit_indices), not len(train_dataset).
    """

    def __init__(
        self,
        labels: Sequence[int],
        *,
        batch_size: int,
        num_classes: int,
        samples_per_class_per_batch: int,
        seed: int,
        replacement: bool = False,
    ) -> None:
        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.samples_per_class_per_batch = int(samples_per_class_per_batch)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = -1

        expected = self.num_classes * self.samples_per_class_per_batch
        if self.batch_size != expected:
            raise ValueError(f"batch_size={self.batch_size}, expected {expected}")

        self.buckets: Dict[int, List[int]] = {c: [] for c in range(self.num_classes)}
        for local_idx, y in enumerate(self.labels):
            if 0 <= y < self.num_classes:
                self.buckets[y].append(int(local_idx))
            else:
                raise ValueError(f"Invalid label in sampler: {y}")

        self.class_counts = [len(self.buckets[c]) for c in range(self.num_classes)]
        if any(x <= 0 for x in self.class_counts):
            missing = [LABELS[c] for c, x in enumerate(self.class_counts) if x <= 0]
            raise ValueError(f"Missing classes in balanced sampler: {missing}")

        min_count = min(self.class_counts)
        self.per_class_per_epoch = min_count // self.samples_per_class_per_batch * self.samples_per_class_per_batch
        if self.per_class_per_epoch <= 0:
            raise ValueError("per_class_per_epoch resolved to zero; reduce samples_per_class_per_batch")
        self.batches_per_epoch = self.per_class_per_epoch // self.samples_per_class_per_batch

    def __len__(self) -> int:
        return int(self.batches_per_epoch)

    def summary(self) -> Dict[str, Any]:
        return {
            "sampler": self.__class__.__name__,
            "batch_size": self.batch_size,
            "samples_per_class_per_batch": self.samples_per_class_per_batch,
            "class_counts": {LABELS[i]: int(self.class_counts[i]) for i in range(self.num_classes)},
            "per_class_per_epoch": int(self.per_class_per_epoch),
            "batches_per_epoch": int(self.batches_per_epoch),
            "samples_per_epoch": int(self.batches_per_epoch * self.batch_size),
            "replacement": self.replacement,
        }

    def __iter__(self):
        self.epoch += 1
        rng = random.Random(self.seed + self.epoch * 100003)
        chosen_by_class: Dict[int, List[int]] = {}

        for c in range(self.num_classes):
            bucket = list(self.buckets[c])
            rng.shuffle(bucket)
            if len(bucket) >= self.per_class_per_epoch:
                chosen = bucket[: self.per_class_per_epoch]
            elif self.replacement:
                chosen = [rng.choice(bucket) for _ in range(self.per_class_per_epoch)]
            else:
                raise RuntimeError(f"Class {c} has too few samples")
            chosen_by_class[c] = chosen

        k = self.samples_per_class_per_batch
        batches: List[List[int]] = []
        for b in range(self.batches_per_epoch):
            start = b * k
            end = start + k
            batch: List[int] = []
            for c in range(self.num_classes):
                batch.extend(chosen_by_class[c][start:end])
            rng.shuffle(batch)
            batches.append(batch)

        rng.shuffle(batches)
        for batch in batches:
            yield batch


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def make_loader_common_kwargs(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    num_workers = int(cfg.get("num_workers", 4))
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", True)),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
    return kwargs


def make_fold_loaders(
    train_ds: FER2013Hybrid,
    eval_ds: EvalViewDataset,
    labels: Sequence[int],
    *,
    fit_indices: Sequence[int],
    inner_val_indices: Sequence[int],
    audit_indices: Sequence[int],
    cfg: Mapping[str, Any],
    fold_id: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    common = make_loader_common_kwargs(cfg)

    fit_subset = Subset(train_ds, list(fit_indices))
    inner_val_subset = Subset(eval_ds, list(inner_val_indices))
    audit_subset = Subset(eval_ds, list(audit_indices))

    fit_labels = [int(labels[int(i)]) for i in fit_indices]
    sampler = BalancedClassBatchSampler(
        fit_labels,
        batch_size=int(cfg["batch_size"]),
        num_classes=NUM_CLASSES,
        samples_per_class_per_batch=int(cfg["samples_per_class_per_batch"]),
        seed=int(cfg["seed"]) + 1000 * int(fold_id),
        replacement=False,
    )

    train_loader = DataLoader(fit_subset, batch_sampler=sampler, **common)
    inner_val_loader = DataLoader(
        inner_val_subset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=make_generator(int(cfg["seed"]) + 2000 + int(fold_id)),
        **common,
    )
    audit_loader = DataLoader(
        audit_subset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=make_generator(int(cfg["seed"]) + 3000 + int(fold_id)),
        **common,
    )

    meta = {
        "fit_count": len(fit_indices),
        "inner_val_count": len(inner_val_indices),
        "audit_count": len(audit_indices),
        "fit_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts(labels, fit_indices))},
        "inner_val_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts(labels, inner_val_indices))},
        "audit_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts(labels, audit_indices))},
        "sampler": sampler.summary(),
    }
    return train_loader, inner_val_loader, audit_loader, meta


# ---------------------------------------------------------------------------
# Model / optimization
# ---------------------------------------------------------------------------
def build_fold_model(cfg: Mapping[str, Any], device: torch.device) -> nn.Module:
    model = get_model(
        variant=str(cfg["model_variant"]),
        num_classes=NUM_CLASSES,
        pretrained=bool(cfg.get("pretrained", True)),
        device=device,
        verbose=True,
        compile_model=bool(cfg.get("compile_model", False)),
    )
    init_ckpt = cfg.get("init_ckpt")
    if not is_none_like(init_ckpt):
        print("[warning] init_ckpt is set. Ensure this checkpoint was NOT trained on full train.csv.", flush=True)
        load_checkpoint_into_model(
            model,
            str(init_ckpt),
            device=device,
            strict=bool(cfg.get("strict_checkpoint_load", True)),
        )
    return model


def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch_index0: int) -> float:
    if epoch_index0 < warmup_epochs:
        return base_lr * float(epoch_index0 + 1) / max(1, int(warmup_epochs))
    progress = (epoch_index0 - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scaler(use_amp: bool):
    if not use_amp:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except Exception:
        return torch.cuda.amp.GradScaler()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class AmpAutocast:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.ctx = None

    def __enter__(self):
        if self.enabled:
            try:
                self.ctx = torch.amp.autocast("cuda")
            except Exception:
                self.ctx = torch.cuda.amp.autocast()
        else:
            self.ctx = _NullContext()
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.ctx is not None
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    *,
    epoch: int,
    metric: float,
    fold_id: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "metric": float(metric),
            "fold_id": int(fold_id),
            "config": dict(cfg),
            "labels": list(LABELS),
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    use_amp: bool,
    scaler: Any,
    label_smoothing: float,
    grad_clip: bool,
    max_norm: float,
) -> Dict[str, Any]:
    model.train()
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    start = time.time()

    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with AmpAutocast(use_amp):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            optimizer.step()

        acc.update(logits.detach(), yb.detach(), loss=loss.detach())

    out = acc.compute().to_dict()
    out["duration_sec"] = float(time.time() - start)
    return out


@torch.no_grad()
def evaluate_model_dict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        acc.update(logits, yb, loss=loss)
    return acc.compute().to_dict()


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_parts: List[np.ndarray] = []
    labels_parts: List[np.ndarray] = []
    losses_parts: List[np.ndarray] = []

    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        loss = F.cross_entropy(logits, yb, reduction="none")
        probs_parts.append(probs.detach().cpu().numpy().astype(np.float32))
        labels_parts.append(yb.detach().cpu().numpy().astype(np.int64))
        losses_parts.append(loss.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(probs_parts, axis=0), np.concatenate(labels_parts, axis=0), np.concatenate(losses_parts, axis=0)


def train_fold(
    fold_id: int,
    cfg: Mapping[str, Any],
    run_dir: Path,
    train_loader: DataLoader,
    inner_val_loader: DataLoader,
    device: torch.device,
) -> Tuple[Path, Dict[str, Any]]:
    fold_dir = run_dir / "folds" / f"fold_{fold_id}"
    ckpt_dir = fold_dir / "checkpoints"
    best_path = ckpt_dir / "best_model.pth"
    last_path = ckpt_dir / "last_model.pth"
    metrics_csv = fold_dir / "metrics_epoch.csv"
    metrics_jsonl = fold_dir / "metrics_epoch.jsonl"

    if bool(cfg.get("resume_existing", True)) and best_path.exists():
        print(f"[fold {fold_id}] reuse existing checkpoint: {best_path}", flush=True)
        return best_path, {"status": "reused", "best_checkpoint": str(best_path)}

    fold_dir.mkdir(parents=True, exist_ok=True)
    model = build_fold_model(cfg, device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    use_amp = bool(cfg.get("use_amp", True) and device.type == "cuda")
    scaler = make_scaler(use_amp)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0

    csv_fields = [
        "fold", "epoch", "lr",
        "train_loss", "train_acc", "train_global_macro_f1",
        "inner_val_loss", "inner_val_acc", "inner_val_global_macro_f1",
        "best_inner_val_global_macro_f1", "best_epoch", "bad_epochs", "duration_sec",
    ]

    for epoch0 in range(int(cfg["epochs"])):
        epoch = epoch0 + 1
        lr = cosine_warmup_lr(
            float(cfg["lr"]),
            float(cfg["lr_floor"]),
            int(cfg["warmup_epochs"]),
            int(cfg["epochs"]),
            epoch0,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_m = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_amp=use_amp,
            scaler=scaler,
            label_smoothing=float(cfg["label_smoothing"]),
            grad_clip=bool(cfg["grad_clip"]),
            max_norm=float(cfg["max_norm"]),
        )
        val_m = evaluate_model_dict(model, inner_val_loader, device)
        metric = float(val_m["global_macro_f1"])

        improved = metric > best_metric + 1e-12
        if improved:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, cfg, epoch=epoch, metric=best_metric, fold_id=fold_id)
        else:
            bad_epochs += 1

        save_checkpoint(last_path, model, optimizer, cfg, epoch=epoch, metric=best_metric, fold_id=fold_id)

        row = {
            "fold": fold_id,
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_m["loss"],
            "train_acc": train_m["accuracy"],
            "train_global_macro_f1": train_m["global_macro_f1"],
            "inner_val_loss": val_m["loss"],
            "inner_val_acc": val_m["accuracy"],
            "inner_val_global_macro_f1": val_m["global_macro_f1"],
            "best_inner_val_global_macro_f1": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "duration_sec": train_m["duration_sec"],
        }
        append_csv_row(metrics_csv, row, csv_fields)
        append_jsonl(
            metrics_jsonl,
            {
                "fold": fold_id,
                "epoch": epoch,
                "train": train_m,
                "inner_val": val_m,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
            },
        )

        print(
            f"[fold {fold_id} epoch {epoch:03d}] lr={lr:.3e} "
            f"train_f1={train_m['global_macro_f1']:.6f} "
            f"inner_val_f1={val_m['global_macro_f1']:.6f} "
            f"best={best_metric:.6f}@{best_epoch} bad={bad_epochs}",
            flush=True,
        )

        if bad_epochs >= int(cfg["early_stop_patience"]):
            print(f"[fold {fold_id}] early stop after {bad_epochs} bad epochs", flush=True)
            break

    summary = {
        "status": "trained",
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_inner_val_global_macro_f1": best_metric,
        "best_checkpoint": str(best_path),
    }
    write_json(fold_dir / "fold_summary.json", summary)
    return best_path, summary


# ---------------------------------------------------------------------------
# OOF post-processing
# ---------------------------------------------------------------------------
def topk_info(probs: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    order = np.argsort(-probs)[:k]
    return [(int(i), float(probs[int(i)])) for i in order]


def compute_issue_type(is_error: bool, confidence: float, margin: float, p_true: float, cfg: Mapping[str, Any]) -> str:
    if is_error and confidence >= float(cfg["high_conf_threshold"]) and margin >= float(cfg["high_margin_threshold"]):
        return "wrong_high_confidence"
    if is_error:
        return "wrong_prediction"
    if p_true <= float(cfg["low_self_confidence_threshold"]):
        return "correct_but_low_self_confidence"
    if margin <= float(cfg["low_margin_threshold"]):
        return "ambiguous_low_margin"
    return "probably_ok"


def zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mu) / sigma


def build_oof_rows(
    eval_ds: EvalViewDataset,
    labels: Sequence[int],
    folds_for_index: Sequence[int],
    probs_all: np.ndarray,
    losses_all: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    losses_z = zscore(losses_all)
    confusion_counter: Counter[Tuple[int, int]] = Counter()

    for idx in range(len(eval_ds)):
        sample = eval_ds.samples[idx]
        probs = probs_all[idx]
        true_id = int(labels[idx])
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        p_true = float(probs[true_id])
        sorted_probs = np.sort(probs)[::-1]
        top2_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = float(confidence - top2_prob)
        loss = float(losses_all[idx])
        loss_z = float(losses_z[idx])
        is_error = bool(pred_id != true_id)
        issue_type = compute_issue_type(is_error, confidence, margin, p_true, cfg)

        issue_score = float(
            (1.0 - p_true)
            + (0.75 if is_error else 0.0)
            + (0.50 * confidence if is_error else 0.0)
            + (0.25 * margin if is_error else 0.0)
            + (0.15 * max(0.0, min(5.0, loss_z)))
        )

        if is_error:
            confusion_counter[(true_id, pred_id)] += 1

        top = topk_info(probs, k=min(5, NUM_CLASSES))
        row: Dict[str, Any] = {
            "dataset_index": int(idx),
            "row_index": int(sample.get("row_index", -1)),
            "source_csv": str(sample.get("source_csv", "")),
            "path": str(sample.get("path", "")),
            "has_pixels": bool(str(sample.get("pixels", "")).strip()),
            "fold": int(folds_for_index[idx]),
            "true_label_id": true_id,
            "true_label": LABELS[true_id],
            "pred_label_id": pred_id,
            "pred_label": LABELS[pred_id],
            "correct": int(not is_error),
            "is_error": int(is_error),
            "confidence": confidence,
            "p_true": p_true,
            "margin": margin,
            "loss": loss,
            "loss_z": loss_z,
            "issue_score": issue_score,
            "issue_type": issue_type,
            "is_high_conf_error": int(issue_type == "wrong_high_confidence"),
            "is_low_margin": int(margin <= float(cfg["low_margin_threshold"])),
        }
        for rank, (label_id, prob) in enumerate(top, start=1):
            row[f"top{rank}_label_id"] = label_id
            row[f"top{rank}_label"] = LABELS[label_id]
            row[f"top{rank}_prob"] = prob
        for label_id, label_name in enumerate(LABELS):
            row[f"prob_{label_name}"] = float(probs[label_id])
        rows.append(row)

    issue_rows = sorted(rows, key=lambda r: float(r["issue_score"]), reverse=True)
    for rank, row in enumerate(issue_rows, start=1):
        row["issue_rank"] = rank

    pair_rows: List[Dict[str, Any]] = []
    for (true_id, pred_id), count in confusion_counter.most_common():
        support = sum(1 for y in labels if int(y) == int(true_id))
        pair_rows.append({
            "true_label_id": int(true_id),
            "true_label": LABELS[int(true_id)],
            "pred_label_id": int(pred_id),
            "pred_label": LABELS[int(pred_id)],
            "count": int(count),
            "true_support": int(support),
            "rate_within_true_label": float(count / max(1, support)),
        })

    high_conf = [r for r in rows if int(r["is_high_conf_error"]) == 1]
    summary = {
        "total": len(rows),
        "errors": int(sum(int(r["is_error"]) for r in rows)),
        "high_conf_errors": int(len(high_conf)),
        "probably_ok": int(sum(1 for r in rows if r["issue_type"] == "probably_ok")),
        "issue_type_counts": dict(Counter(str(r["issue_type"]) for r in rows)),
        "top_confusion_pairs": pair_rows[:20],
    }
    return rows, issue_rows, pair_rows, summary


def oof_metrics_from_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    preds = torch.tensor([int(r["pred_label_id"]) for r in rows], dtype=torch.long)
    y = torch.tensor([int(r["true_label_id"]) for r in rows], dtype=torch.long)
    logits = torch.full((len(rows), NUM_CLASSES), -10.0)
    for i, p in enumerate(preds.tolist()):
        logits[i, int(p)] = 10.0
    losses = torch.tensor([float(r["loss"]) for r in rows], dtype=torch.float32)
    acc.update(logits, y, loss=float(torch.mean(losses).item()))
    return acc.compute().to_dict()


# ---------------------------------------------------------------------------
# Contact sheets and manual review templates
# ---------------------------------------------------------------------------
def load_default_font(size: int = 14):
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            if Path(c).exists():
                return ImageFont.truetype(c, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    max_chars: int,
) -> int:
    x, y = xy
    lines: List[str] = []
    for part in str(text).split("\n"):
        while len(part) > max_chars:
            lines.append(part[:max_chars])
            part = part[max_chars:]
        lines.append(part)
    line_h = max(12, int(getattr(font, "size", 14) * 1.2))
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y


def make_contact_sheet(
    eval_ds: EvalViewDataset,
    rows: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    title: str,
    cols: int,
    thumb: int,
) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = max(1, int(cols))
    thumb = max(64, int(thumb))
    label_h = 92
    pad = 10
    rows_n = int(math.ceil(len(rows) / cols))
    cell_w = thumb + pad * 2
    cell_h = thumb + label_h + pad * 2
    title_h = 36

    canvas = Image.new("RGB", (cols * cell_w, title_h + rows_n * cell_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = load_default_font(13)
    small = load_default_font(11)
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)

    for pos, row in enumerate(rows):
        r = pos // cols
        c = pos % cols
        x0 = c * cell_w + pad
        y0 = title_h + r * cell_h + pad
        idx = int(row["dataset_index"])
        try:
            img = eval_ds.raw_image(idx).resize((thumb, thumb), Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (thumb, thumb), color=(180, 180, 180))
        canvas.paste(img, (x0, y0))

        text = (
            f"#{row.get('issue_rank','?')} idx={idx} fold={row['fold']}\n"
            f"true={row['true_label']} pred={row['pred_label']}\n"
            f"conf={float(row['confidence']):.3f} ptrue={float(row['p_true']):.3f}\n"
            f"margin={float(row['margin']):.3f} loss={float(row['loss']):.3f}\n"
            f"{row['issue_type']}"
        )
        draw_wrapped_text(draw, (x0, y0 + thumb + 4), text, small, (0, 0, 0), max_chars=26)

    canvas.save(out_path)


def write_contact_sheets(
    eval_ds: EvalViewDataset,
    issue_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    cfg: Mapping[str, Any],
) -> None:
    sheet_dir = run_dir / "contact_sheets"
    top_n = min(int(cfg["contact_sheet_top_n"]), len(issue_rows))
    page_size = int(cfg["contact_sheet_cols"]) ** 2
    page_size = max(1, page_size)

    top_rows = list(issue_rows[:top_n])
    for start in range(0, len(top_rows), page_size):
        page = start // page_size + 1
        rows = top_rows[start:start + page_size]
        make_contact_sheet(
            eval_ds,
            rows,
            sheet_dir / f"top_issues_page_{page:03d}.png",
            title=f"OOF top label issues {start + 1}-{start + len(rows)}",
            cols=int(cfg["contact_sheet_cols"]),
            thumb=int(cfg["contact_sheet_thumb"]),
        )

    top_pairs = pair_rows[: int(cfg["contact_sheet_top_pairs"])]
    per_pair_n = int(cfg["contact_sheet_per_pair_top_n"])
    for pair in top_pairs:
        true_label = str(pair["true_label"])
        pred_label = str(pair["pred_label"])
        rows = [
            r for r in issue_rows
            if int(r["true_label_id"]) == int(pair["true_label_id"])
            and int(r["pred_label_id"]) == int(pair["pred_label_id"])
        ][:per_pair_n]
        safe = f"{true_label}_to_{pred_label}".replace("/", "_").replace("\\", "_")
        make_contact_sheet(
            eval_ds,
            rows,
            sheet_dir / "by_confusion_pair" / f"{safe}.png",
            title=f"OOF confusion pair: {true_label} -> {pred_label}",
            cols=int(cfg["contact_sheet_cols"]),
            thumb=int(cfg["contact_sheet_thumb"]),
        )


def build_manual_review_template(
    issue_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    n = min(int(cfg["manual_review_rows"]), len(issue_rows))
    rows: List[Dict[str, Any]] = []
    for row in issue_rows[:n]:
        out = dict(row)
        out.update({
            "review_action": "",  # keep / relabel / ignore / soft
            "new_label": "",
            "soft_label_json": "",
            "review_reason": "",
            "reviewer": "",
        })
        rows.append(out)

    preferred = [
        "issue_rank", "dataset_index", "row_index", "fold", "true_label", "pred_label",
        "confidence", "p_true", "margin", "loss", "issue_score", "issue_type",
        "path", "has_pixels", "source_csv",
        "top1_label", "top1_prob", "top2_label", "top2_prob", "top3_label", "top3_prob",
        "review_action", "new_label", "soft_label_json", "review_reason", "reviewer",
    ]
    remaining = [k for k in rows[0].keys()] if rows else []
    fieldnames = preferred + [k for k in remaining if k not in preferred]
    return rows, fieldnames


# ---------------------------------------------------------------------------
# Fold prediction persistence
# ---------------------------------------------------------------------------
def write_fold_oof_predictions(
    path: Path,
    audit_indices: Sequence[int],
    probs: np.ndarray,
    losses: np.ndarray,
    labels: Sequence[int],
) -> None:
    rows: List[Dict[str, Any]] = []
    fields = ["dataset_index", "true_label_id", "loss"] + [f"prob_{label}" for label in LABELS]
    for pos, idx in enumerate(audit_indices):
        row: Dict[str, Any] = {
            "dataset_index": int(idx),
            "true_label_id": int(labels[pos]),
            "loss": float(losses[pos]),
        }
        for c, label in enumerate(LABELS):
            row[f"prob_{label}"] = float(probs[pos, c])
        rows.append(row)
    write_csv_rows(path, rows, fields)


def read_fold_oof_predictions(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        idxs: List[int] = []
        losses: List[float] = []
        probs: List[List[float]] = []
        for row in reader:
            idxs.append(int(row["dataset_index"]))
            losses.append(float(row["loss"]))
            probs.append([float(row[f"prob_{label}"]) for label in LABELS])
    return np.array(idxs, dtype=np.int64), np.array(probs, dtype=np.float32), np.array(losses, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------
def run_oof_audit(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = resolve_config(cfg)
    set_global_seed(int(cfg["seed"]))

    train_csv = Path(str(cfg["train_csv"]))
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")

    run_name = cfg.get("run_name") or f"oof_train_audit_{now_stamp()}_seed{cfg['seed']}"
    run_dir = Path(str(cfg["runs_dir"])) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(str(cfg["device"]))

    print("=== FER out-of-fold train audit ===", flush=True)
    print(f"run_dir  : {run_dir}", flush=True)
    print(f"train_csv: {train_csv}", flush=True)
    print(f"device   : {device}", flush=True)
    print(f"folds    : {cfg['num_folds']}", flush=True)

    train_ds = FER2013Hybrid(str(train_csv), cfg.get("img_base"), "train", img_size=IMG_SIZE, include_label=True, strict=True)
    eval_ds = EvalViewDataset(train_ds)
    labels = dataset_labels(train_ds)
    n = len(labels)
    folds = make_stratified_folds(labels, int(cfg["num_folds"]), int(cfg["seed"]))

    manifest = {
        "status": "started",
        "created_at": now_stamp(),
        "method": "out_of_fold_train_label_audit",
        "why": "Each train sample is predicted by a fold model that did not train on that sample.",
        "config": dict(cfg),
        "labels": list(LABELS),
        "dataset": {
            "train_csv": str(train_csv),
            "total": n,
            "class_counts": {LABELS[i]: c for i, c in enumerate(class_counts(labels))},
        },
        "folds": [],
    }
    write_json(run_dir / "manifest.json", manifest)

    probs_all = np.full((n, NUM_CLASSES), np.nan, dtype=np.float32)
    losses_all = np.full((n,), np.nan, dtype=np.float32)
    fold_for_index = np.full((n,), -1, dtype=np.int64)
    fold_summaries: List[Dict[str, Any]] = []

    all_indices = set(range(n))
    for fold_id, audit_indices in enumerate(folds):
        audit_set = set(audit_indices)
        train_indices = sorted(all_indices - audit_set)
        fit_indices, inner_val_indices = stratified_inner_split(
            train_indices,
            labels,
            float(cfg["inner_val_fraction"]),
            int(cfg["seed"]) + fold_id * 13,
        )

        fold_dir = run_dir / "folds" / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_loader, inner_val_loader, audit_loader, fold_meta = make_fold_loaders(
            train_ds,
            eval_ds,
            labels,
            fit_indices=fit_indices,
            inner_val_indices=inner_val_indices,
            audit_indices=audit_indices,
            cfg=cfg,
            fold_id=fold_id,
        )
        fold_meta.update({
            "fold": fold_id,
            "audit_indices_min": int(min(audit_indices)),
            "audit_indices_max": int(max(audit_indices)),
        })
        write_json(fold_dir / "fold_manifest.json", fold_meta)
        print(f"[fold {fold_id}] {json.dumps(fold_meta, ensure_ascii=False)}", flush=True)

        pred_path = fold_dir / "fold_oof_predictions.csv"
        if bool(cfg.get("resume_existing", True)) and pred_path.exists():
            print(f"[fold {fold_id}] reuse existing predictions: {pred_path}", flush=True)
            idxs, fold_probs, fold_losses = read_fold_oof_predictions(pred_path)
            if set(idxs.tolist()) != set(map(int, audit_indices)):
                raise RuntimeError(f"Existing predictions do not match audit indices for fold {fold_id}")
        else:
            best_path, train_summary = train_fold(fold_id, cfg, run_dir, train_loader, inner_val_loader, device)
            model = build_fold_model(cfg, device)
            load_checkpoint_into_model(model, best_path, device=device, strict=True)
            fold_probs, fold_labels, fold_losses = predict_probs(model, audit_loader, device)
            expected_labels = np.array([labels[int(i)] for i in audit_indices], dtype=np.int64)
            if not np.array_equal(fold_labels, expected_labels):
                raise RuntimeError(f"Fold {fold_id} label order mismatch during audit prediction")
            idxs = np.array(audit_indices, dtype=np.int64)
            write_fold_oof_predictions(pred_path, audit_indices, fold_probs, fold_losses, fold_labels)
            fold_meta["training"] = train_summary

        for local_pos, global_idx in enumerate(idxs.tolist()):
            probs_all[int(global_idx)] = fold_probs[local_pos]
            losses_all[int(global_idx)] = fold_losses[local_pos]
            fold_for_index[int(global_idx)] = int(fold_id)

        fold_summaries.append(fold_meta)
        manifest["folds"] = fold_summaries
        write_json(run_dir / "manifest.json", manifest)

    if np.isnan(probs_all).any() or np.isnan(losses_all).any() or (fold_for_index < 0).any():
        missing = np.where(np.isnan(losses_all) | (fold_for_index < 0))[0].tolist()[:20]
        raise RuntimeError(f"OOF predictions incomplete. Missing examples: {missing}")

    rows, issue_rows, pair_rows, issue_summary = build_oof_rows(
        eval_ds,
        labels,
        fold_for_index.tolist(),
        probs_all,
        losses_all,
        cfg,
    )

    metrics = oof_metrics_from_rows(rows)
    issue_summary["oof_metrics"] = metrics

    all_fields = list(rows[0].keys()) if rows else []
    issue_fields = ["issue_rank"] + [k for k in all_fields if k != "issue_rank"]
    pair_fields = ["true_label_id", "true_label", "pred_label_id", "pred_label", "count", "true_support", "rate_within_true_label"]

    write_csv_rows(run_dir / str(cfg["oof_predictions_name"]), rows, all_fields)
    write_csv_rows(run_dir / str(cfg["label_issues_name"]), issue_rows[: int(cfg["top_issue_rows"])], issue_fields)
    write_csv_rows(run_dir / str(cfg["confusion_pairs_name"]), pair_rows, pair_fields)
    write_json(run_dir / "oof_train_audit_summary.json", issue_summary)
    save_metrics_json(metrics, run_dir / "oof_train_metrics.json")
    save_confusion_png(metrics["confusion"], run_dir / "oof_train_confusion.png", title="OOF Train Confusion Matrix")

    manual_rows, manual_fields = build_manual_review_template(issue_rows, cfg)
    write_csv_rows(run_dir / str(cfg["manual_template_name"]), manual_rows, manual_fields)
    write_contact_sheets(eval_ds, issue_rows, pair_rows, run_dir, cfg)

    manifest.update({
        "status": "finished",
        "finished_at": now_stamp(),
        "outputs": {
            "oof_predictions": str(run_dir / str(cfg["oof_predictions_name"])),
            "label_issues": str(run_dir / str(cfg["label_issues_name"])),
            "confusion_pairs": str(run_dir / str(cfg["confusion_pairs_name"])),
            "manual_review_template": str(run_dir / str(cfg["manual_template_name"])),
            "contact_sheets": str(run_dir / "contact_sheets"),
            "summary": str(run_dir / "oof_train_audit_summary.json"),
        },
        "summary": issue_summary,
    })
    write_json(run_dir / "manifest.json", manifest)

    print("=== OOF audit complete ===", flush=True)
    print(json.dumps({
        "run_dir": str(run_dir),
        "oof_macro_f1": metrics["global_macro_f1"],
        "errors": issue_summary["errors"],
        "high_conf_errors": issue_summary["high_conf_errors"],
        "label_issues": str(run_dir / str(cfg["label_issues_name"])),
        "manual_review_template": str(run_dir / str(cfg["manual_template_name"])),
        "contact_sheets": str(run_dir / "contact_sheets"),
    }, indent=2, ensure_ascii=False), flush=True)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FER out-of-fold train label audit")
    parser.add_argument("--project-root", type=str, default=None, help=r"Project root, e.g. F:\fer-pi5")
    parser.add_argument("--train-csv", type=str, default=None, help="Train CSV path, relative to project root or absolute")
    parser.add_argument("--img-base", type=str, default=None, help="Image base path, if CSV uses image paths")
    parser.add_argument("--runs-dir", type=str, default=None, help="Output runs directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional fixed run directory name")
    parser.add_argument("--folds", type=int, default=None, help="Number of OOF folds")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs per fold")
    parser.add_argument("--patience", type=int, default=None, help="Early stop patience per fold")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size, default 126")
    parser.add_argument("--per-class-batch", type=int, default=None, help="Samples per class per batch, default 18")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--init-ckpt", type=str, default=None, help="Optional external warm-start checkpoint; avoid full-train FER checkpoints")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained MobileNetV3")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing fold checkpoints/predictions")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    mapping = {
        "project_root": "project_root",
        "train_csv": "train_csv",
        "img_base": "img_base",
        "runs_dir": "runs_dir",
        "run_name": "run_name",
        "folds": "num_folds",
        "epochs": "epochs",
        "patience": "early_stop_patience",
        "batch_size": "batch_size",
        "per_class_batch": "samples_per_class_per_batch",
        "lr": "lr",
        "seed": "seed",
        "device": "device",
        "num_workers": "num_workers",
        "init_ckpt": "init_ckpt",
    }
    for arg_name, cfg_name in mapping.items():
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.no_amp:
        cfg["use_amp"] = False
    if args.no_resume:
        cfg["resume_existing"] = False
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    cfg = config_from_args(args)
    return run_oof_audit(cfg)


if __name__ == "__main__":
    main()