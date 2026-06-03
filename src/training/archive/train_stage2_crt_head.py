from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES, MetricAccumulator, evaluate_model, save_confusion_png, save_metrics_json
    from model_mbv3 import get_model, load_checkpoint_into_model
except Exception as exc:
    print("[import-error] Put this script in src/training beside dataset.py, metrics.py, model_mbv3.py")
    print(f"[import-error] current_dir={THIS_DIR}")
    raise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    # Override without editing:
    #   set FER_PROJECT_ROOT=F:\fer-pi5
    "project_root": os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"),

    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "img_base": None,

    # Prefer the current best representation. If missing, fallback to Stage1.
    "init_ckpt_candidates": [
        r"checkpoints\best_model_stage2_balanced_clean.pth",
        r"checkpoints\best_model_stage1_refactored.pth",
    ],

    "runs_dir": r"runs\training",
    "checkpoint_alias_dir": r"checkpoints",
    "best_alias_name": "best_model_stage2_crt_head.pth",
    "log_alias_name": "train_stage2_crt_head_log.csv",
    "sweep_alias_name": "stage2_crt_head_logit_sweep.json",

    "model_variant": "large",
    "num_classes": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    # cRT / head-only retraining
    "reset_classifier": True,
    "train_classifier_only": True,

    # Strict 7-class balanced mini-batches.
    "epochs": 80,
    "batch_size": 126,
    "samples_per_class_per_batch": 18,
    "early_stop_patience": 15,

    # Head-only learning can use a larger LR than full-model fine-tuning.
    "lr": 1e-3,
    "lr_floor": 1e-5,
    "warmup_epochs": 3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.02,

    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,

    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    # Post-training prior calibration sweep. tau=0 means no adjustment.
    # Positive tau subtracts tau * log(train_prior), helping minority classes.
    "logit_adjustment_taus": [
        -0.50, -0.25, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.40,
        1.60, 1.80, 2.00,
    ],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def resolve_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    root = Path(str(out["project_root"])).expanduser().resolve()
    out["project_root"] = str(root)

    for key in ("train_csv", "val_csv", "test_csv", "img_base", "runs_dir", "checkpoint_alias_dir"):
        out[key] = None if is_none_like(out.get(key)) else str(resolve_path(root, out[key]))

    candidates = []
    for item in out.get("init_ckpt_candidates", []):
        p = resolve_path(root, item)
        if p is not None:
            candidates.append(str(p))
    out["init_ckpt_candidates"] = candidates

    out["num_classes"] = int(out.get("num_classes", NUM_CLASSES))
    if out["num_classes"] != NUM_CLASSES:
        raise ValueError(f"Project expects {NUM_CLASSES} classes, got {out['num_classes']}")

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


def choose_checkpoint(candidates: Sequence[str]) -> Path:
    checked: List[str] = []
    for item in candidates:
        p = Path(item)
        checked.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError("No init checkpoint found. Checked:\n" + "\n".join(checked))


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


def labels_from_dataset(ds: Dataset) -> List[int]:
    if hasattr(ds, "samples"):
        return [int(sample.get("label", -1)) for sample in getattr(ds, "samples")]
    out: List[int] = []
    for idx in range(len(ds)):
        out.append(int(ds[idx][1]))
    return out


def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch_index0: int) -> float:
    if epoch_index0 < warmup_epochs:
        return base_lr * float(epoch_index0 + 1) / max(1, int(warmup_epochs))
    progress = (epoch_index0 - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv_row(path: Path, row: Mapping[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def copy_alias(src: Path, dst: Path, *, overwrite: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Alias exists: {dst}")
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

class BalancedClassBatchSampler(Sampler[List[int]]):
    """Strict class-balanced batch sampler.

    For FER-7, batch_size=126 and samples_per_class_per_batch=18 means:
        each batch = 18 samples from each of the 7 classes.
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

        expected = self.samples_per_class_per_batch * self.num_classes
        if expected != self.batch_size:
            raise ValueError(
                f"batch_size must equal num_classes * samples_per_class_per_batch. "
                f"Got batch_size={self.batch_size}, expected={expected}."
            )

        self.buckets: Dict[int, List[int]] = {i: [] for i in range(self.num_classes)}
        for idx, label in enumerate(self.labels):
            if 0 <= label < self.num_classes:
                self.buckets[label].append(idx)

        self.class_counts = [len(self.buckets[i]) for i in range(self.num_classes)]
        if any(c <= 0 for c in self.class_counts):
            missing = [LABELS[i] for i, c in enumerate(self.class_counts) if c <= 0]
            raise ValueError(f"Balanced sampler requires all classes. Missing: {missing}")

        min_count = min(self.class_counts)
        self.per_class_per_epoch = min_count // self.samples_per_class_per_batch * self.samples_per_class_per_batch
        if self.per_class_per_epoch <= 0:
            raise ValueError("per_class_per_epoch resolved to zero")

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

        selected_by_class: Dict[int, List[int]] = {}
        for y in range(self.num_classes):
            bucket = list(self.buckets[y])
            rng.shuffle(bucket)
            if len(bucket) >= self.per_class_per_epoch:
                selected = bucket[: self.per_class_per_epoch]
            elif self.replacement:
                selected = [rng.choice(bucket) for _ in range(self.per_class_per_epoch)]
            else:
                raise RuntimeError(f"class {y} has too few samples for balanced sampling")
            selected_by_class[y] = selected

        all_batches: List[List[int]] = []
        k = self.samples_per_class_per_batch
        for b in range(self.batches_per_epoch):
            start = b * k
            end = start + k
            batch: List[int] = []
            for y in range(self.num_classes):
                batch.extend(selected_by_class[y][start:end])
            rng.shuffle(batch)
            all_batches.append(batch)

        rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch


# ---------------------------------------------------------------------------
# Model control
# ---------------------------------------------------------------------------

def reset_classifier_linears(model: nn.Module) -> None:
    if not hasattr(model, "classifier"):
        raise TypeError("Expected MobileNetV3 model with .classifier")
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def freeze_all_but_classifier(model: nn.Module) -> List[nn.Parameter]:
    if not hasattr(model, "classifier"):
        raise TypeError("Expected MobileNetV3 model with .classifier")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters after freezing")
    return trainable


def count_trainable_params(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": int(total), "trainable_params": int(trainable)}


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def make_loaders(cfg: Mapping[str, Any]) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    train_ds = FER2013Hybrid(str(cfg["train_csv"]), cfg.get("img_base"), "train", img_size=IMG_SIZE, include_label=True, strict=True)
    val_ds = FER2013Hybrid(str(cfg["val_csv"]), cfg.get("img_base"), "val", img_size=IMG_SIZE, include_label=True, strict=True)
    test_ds = FER2013Hybrid(str(cfg["test_csv"]), cfg.get("img_base"), "test", img_size=IMG_SIZE, include_label=True, strict=True)

    sampler = BalancedClassBatchSampler(
        labels_from_dataset(train_ds),
        batch_size=int(cfg["batch_size"]),
        num_classes=NUM_CLASSES,
        samples_per_class_per_batch=int(cfg["samples_per_class_per_batch"]),
        seed=int(cfg["seed"]),
        replacement=False,
    )

    num_workers = int(cfg.get("num_workers", 4))
    common = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", True)),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        common["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        common["persistent_workers"] = bool(cfg.get("persistent_workers", True))

    train_loader = DataLoader(train_ds, batch_sampler=sampler, **common)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=make_generator(int(cfg["seed"]) + 10),
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        generator=make_generator(int(cfg["seed"]) + 20),
        **common,
    )

    meta = {
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
        "train_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts_from_dataset(train_ds))},
        "val_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts_from_dataset(val_ds))},
        "test_class_counts": {LABELS[i]: c for i, c in enumerate(class_counts_from_dataset(test_ds))},
        "sampler": sampler.summary(),
    }
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meta


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    label_smoothing: float,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
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

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=float(label_smoothing))
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_norm))
            optimizer.step()

        acc.update(logits.detach(), yb.detach(), loss=loss.detach())

    out = acc.compute().to_dict()
    out["duration_sec"] = float(time.time() - start)
    return out


@torch.no_grad()
def evaluate_with_logit_adjustment(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    log_prior: Optional[torch.Tensor] = None,
    tau: float = 0.0,
) -> Dict[str, Any]:
    model.eval()
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    criterion = nn.CrossEntropyLoss()

    if log_prior is not None:
        log_prior = log_prior.to(device)

    for batch in loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        if log_prior is not None and abs(float(tau)) > 1e-12:
            logits = logits - float(tau) * log_prior.view(1, -1)

        loss = criterion(logits, yb)
        acc.update(logits, yb, loss=loss)

    return acc.compute().to_dict()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any], *, epoch: int, best_metric: float, val_metrics: Mapping[str, Any], run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "val_metrics": dict(val_metrics),
            "config": dict(cfg),
            "run_id": run_id,
            "labels": list(LABELS),
        },
        path,
    )


def sweep_logit_adjustment(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    train_class_counts: Sequence[int],
    device: torch.device,
    taus: Sequence[float],
) -> Dict[str, Any]:
    counts = torch.tensor([max(1, int(x)) for x in train_class_counts], dtype=torch.float32)
    priors = counts / counts.sum()
    log_prior = torch.log(priors)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for tau in taus:
        val = evaluate_with_logit_adjustment(model, val_loader, device, log_prior=log_prior, tau=float(tau))
        test = evaluate_with_logit_adjustment(model, test_loader, device, log_prior=log_prior, tau=float(tau))
        row = {
            "tau": float(tau),
            "val_global_macro_f1": float(val["global_macro_f1"]),
            "val_acc": float(val["accuracy"]),
            "test_global_macro_f1": float(test["global_macro_f1"]),
            "test_acc": float(test["accuracy"]),
            "val": val,
            "test": test,
        }
        rows.append(row)
        if best is None or row["val_global_macro_f1"] > best["val_global_macro_f1"]:
            best = row

    assert best is not None
    return {
        "method": "post_hoc_logit_adjustment",
        "formula": "adjusted_logits = logits - tau * log(train_class_prior)",
        "train_class_counts": {LABELS[i]: int(train_class_counts[i]) for i in range(NUM_CLASSES)},
        "train_class_priors": {LABELS[i]: float(priors[i].item()) for i in range(NUM_CLASSES)},
        "rows": rows,
        "best_by_val_macro_f1": best,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict[str, Any]:
    cfg = resolve_config(CONFIG)
    set_global_seed(int(cfg["seed"]))

    run_id = f"stage2_crt_head_{now_stamp()}_seed{cfg['seed']}"
    run_dir = Path(str(cfg["runs_dir"])) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    metrics_csv = run_dir / "metrics_epoch.csv"
    metrics_jsonl = run_dir / "metrics_epoch.jsonl"
    run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(str(cfg["device"]))
    init_ckpt = choose_checkpoint(cfg["init_ckpt_candidates"])

    print("=== FER Stage2 cRT Head-Only Retraining ===", flush=True)
    print(f"run_id   : {run_id}", flush=True)
    print(f"run_dir  : {run_dir}", flush=True)
    print(f"device   : {device}", flush=True)
    print(f"init_ckpt: {init_ckpt}", flush=True)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, data_meta = make_loaders(cfg)
    write_json(run_dir / "resolved_config.json", cfg)
    write_json(run_dir / "sampler_audit.json", data_meta["sampler"])

    model = get_model(
        variant=str(cfg["model_variant"]),
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
        verbose=True,
        compile_model=False,
    )
    load_checkpoint_into_model(model, init_ckpt, device=device, strict=True)

    if bool(cfg["reset_classifier"]):
        print("[model] reset MobileNetV3 classifier Linear layers", flush=True)
        reset_classifier_linears(model)

    if bool(cfg["train_classifier_only"]):
        trainable_params = freeze_all_but_classifier(model)
        print("[model] froze backbone; training classifier only", flush=True)
    else:
        trainable_params = list(model.parameters())
        print("[model] training full model", flush=True)

    param_meta = count_trainable_params(model)
    print("[model]", param_meta, flush=True)

    optimizer = AdamW(trainable_params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    use_amp = bool(cfg.get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    manifest = {
        "status": "started",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "init_ckpt": str(init_ckpt),
        "labels": list(LABELS),
        "config": cfg,
        "data": data_meta,
        "params": param_meta,
    }
    write_json(run_dir / "manifest.json", manifest)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"

    csv_fields = [
        "epoch", "lr", "train_loss", "train_acc", "train_global_macro_f1",
        "val_loss", "val_acc", "val_global_macro_f1",
        "best_val_global_macro_f1", "best_epoch", "bad_epochs", "duration_sec",
    ]

    for epoch_index0 in range(int(cfg["epochs"])):
        epoch = epoch_index0 + 1
        lr = cosine_warmup_lr(
            float(cfg["lr"]), float(cfg["lr_floor"]), int(cfg["warmup_epochs"]), int(cfg["epochs"]), epoch_index0
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            label_smoothing=float(cfg["label_smoothing"]),
            use_amp=use_amp,
            scaler=scaler,
            grad_clip=bool(cfg["grad_clip"]),
            max_norm=float(cfg["max_norm"]),
        )

        val_metrics = evaluate_with_logit_adjustment(model, val_loader, device, tau=0.0)
        metric_value = float(val_metrics["global_macro_f1"])

        improved = metric_value > best_metric + 1e-12
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                cfg,
                epoch=epoch,
                best_metric=best_metric,
                val_metrics=val_metrics,
                run_id=run_id,
            )
            save_metrics_json(val_metrics, eval_dir / "val_best_metrics.json")
            save_confusion_png(val_metrics["confusion"], eval_dir / "val_best_confusion.png", title="Validation Confusion Matrix")
        else:
            bad_epochs += 1

        save_checkpoint(
            last_path,
            model,
            optimizer,
            cfg,
            epoch=epoch,
            best_metric=best_metric,
            val_metrics=val_metrics,
            run_id=run_id,
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_global_macro_f1": train_metrics["global_macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_global_macro_f1": val_metrics["global_macro_f1"],
            "best_val_global_macro_f1": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "duration_sec": train_metrics["duration_sec"],
        }
        write_csv_row(metrics_csv, row, csv_fields)
        append_jsonl(metrics_jsonl, {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics, "best_epoch": best_epoch, "best_metric": best_metric})

        print(
            f"[epoch {epoch:03d}] lr={lr:.3e} "
            f"train_f1={row['train_global_macro_f1']:.6f} "
            f"val_f1={row['val_global_macro_f1']:.6f} "
            f"best={best_metric:.6f}@{best_epoch} bad={bad_epochs}",
            flush=True,
        )

        if bad_epochs >= int(cfg["early_stop_patience"]):
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")

    # Reload best for final evaluation and logit-adjustment sweep.
    eval_model = get_model(
        variant=str(cfg["model_variant"]),
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
        verbose=False,
        compile_model=False,
    )
    load_checkpoint_into_model(eval_model, best_path, device=device, strict=True)

    val_best = evaluate_with_logit_adjustment(eval_model, val_loader, device, tau=0.0)
    test_best = evaluate_with_logit_adjustment(eval_model, test_loader, device, tau=0.0)
    save_metrics_json(val_best, eval_dir / "val_best_reloaded_metrics.json")
    save_metrics_json(test_best, eval_dir / "test_best_reloaded_metrics.json")
    save_confusion_png(val_best["confusion"], eval_dir / "val_best_reloaded_confusion.png", title="Validation Confusion Matrix")
    save_confusion_png(test_best["confusion"], eval_dir / "test_best_reloaded_confusion.png", title="Test Confusion Matrix")

    train_counts = class_counts_from_dataset(train_ds)
    sweep = sweep_logit_adjustment(
        eval_model,
        val_loader,
        test_loader,
        train_counts,
        device,
        taus=[float(x) for x in cfg["logit_adjustment_taus"]],
    )
    write_json(eval_dir / "logit_adjustment_sweep.json", sweep)

    # Save sweep CSV for quick inspection.
    sweep_csv = eval_dir / "logit_adjustment_sweep.csv"
    sweep_fields = ["tau", "val_global_macro_f1", "val_acc", "test_global_macro_f1", "test_acc"]
    for row in sweep["rows"]:
        write_csv_row(sweep_csv, {k: row[k] for k in sweep_fields}, sweep_fields)

    alias_dir = Path(str(cfg["checkpoint_alias_dir"]))
    best_alias = alias_dir / str(cfg["best_alias_name"])
    log_alias = alias_dir / str(cfg["log_alias_name"])
    sweep_alias = alias_dir / str(cfg["sweep_alias_name"])
    copy_alias(best_path, best_alias, overwrite=True)
    copy_alias(metrics_csv, log_alias, overwrite=True)
    copy_alias(eval_dir / "logit_adjustment_sweep.json", sweep_alias, overwrite=True)

    final = {
        "status": "finished",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_global_macro_f1": float(val_best["global_macro_f1"]),
        "best_test_global_macro_f1": float(test_best["global_macro_f1"]),
        "best_checkpoint": str(best_path),
        "best_alias": str(best_alias),
        "log_alias": str(log_alias),
        "logit_adjustment_sweep": str(eval_dir / "logit_adjustment_sweep.json"),
        "logit_adjustment_sweep_alias": str(sweep_alias),
        "logit_adjustment_best_by_val_macro_f1": sweep["best_by_val_macro_f1"],
        "final_val": val_best,
        "final_test": test_best,
    }
    manifest.update(final)
    manifest["status"] = "finished"
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "final_summary.json", final)

    print("=== cRT training complete ===", flush=True)
    print(json.dumps({
        "best_epoch": best_epoch,
        "val_f1_no_adjust": final["best_val_global_macro_f1"],
        "test_f1_no_adjust": final["best_test_global_macro_f1"],
        "best_logit_tau": sweep["best_by_val_macro_f1"]["tau"],
        "val_f1_adjusted": sweep["best_by_val_macro_f1"]["val_global_macro_f1"],
        "test_f1_adjusted": sweep["best_by_val_macro_f1"]["test_global_macro_f1"],
        "run_dir": str(run_dir),
    }, indent=2, ensure_ascii=False), flush=True)

    return final


if __name__ == "__main__":
    main()