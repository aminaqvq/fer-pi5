from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


def unpack_model_output(output: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Unpack model forward output into (main_logits, aux_logits_dict).

    Works with:
      - plain ``torch.Tensor`` (e.g. MobileNetV3) → (tensor, {})
      - ``dict`` with key ``"main"`` (e.g. RepVGGplus training) →
        (output["main"], {k: v for k, v in output.items() if k != "main"})
    """
    if isinstance(output, Mapping):
        if "main" not in output:
            raise ValueError("Model output dict must contain key 'main'")
        main = output["main"]
        aux = {k: v for k, v in output.items() if k != "main" and torch.is_tensor(v)}
        return main, aux
    if torch.is_tensor(output):
        return output, {}
    raise TypeError(f"Unsupported model output type: {type(output)}")


try:
    from .dataset import FER2013Hybrid, IMG_SIZE
    from .metrics import LABELS, NUM_CLASSES, MetricAccumulator, evaluate_model, save_confusion_png, save_metrics_json
    from .model_mbv3 import get_model, load_checkpoint_into_model
except ImportError:  # allows direct PyCharm execution from src/training
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES, MetricAccumulator, evaluate_model, save_confusion_png, save_metrics_json
    from model_mbv3 import get_model, load_checkpoint_into_model


DEFAULT_COMMON_CONFIG: Dict[str, Any] = {
    # Paths
    "project_root": r"F:\fer-pi5",
    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "img_base": None,
    "runs_dir": r"runs\training",
    "checkpoint_alias_dir": r"checkpoints",
    # Model
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": True,
    "compile_model": False,
    "strict_checkpoint_load": True,
    # Training
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "drop_last_train": True,
    "lr": 5e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,
    "label_smoothing": 0.04,
    "class_balance_beta": 0.995,
    "use_class_weights": True,
    "class_weights_from": "labeled_train",  # labeled_train only; do not include pseudo by default
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,
    "dynamic_sampling": False,
    "per_class_limit": 0,
    # Semi-supervised pseudo-label weighting
    "pseudo_csv": None,
    "pseudo_conf_min": 0.0,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 1.0,
    "pseudo_rampup_epochs": 5,
    "require_pseudo_conf": True,
    # Evaluation and checkpointing
    "val_interval": 1,
    "early_stop_patience": 20,
    "best_metric": "global_macro_f1",
    "evaluate_test_at_end": True,
    "save_last_every_epoch": True,
    "write_checkpoint_alias": True,
    "alias_overwrite": True,
    # Reproducibility
    "seed": 42,
    "deterministic_algorithms": False,
    "cudnn_benchmark": False,
    "run_name": None,
    "notes": "",
}


STAGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "stage1": {
        "stage": "stage1",
        "pretrained": True,
        "init_ckpt": None,
        "pseudo_csv": None,
        "best_alias_name": "best_model_stage1_refactored.pth",
        "log_alias_name": "train_stage1_refactored_log.csv",
    },
    "stage2": {
        "stage": "stage2",
        "pretrained": False,
        "init_ckpt": r"checkpoints\best_model_stage1_refactored.pth",
        "pseudo_csv": r"data\csv\pseudo_labeled.csv",
        "best_alias_name": "best_model_stage2_refactored.pth",
        "log_alias_name": "train_stage2_refactored_log.csv",
    },
    "stage3": {
        "stage": "stage3",
        "pretrained": False,
        "init_ckpt": r"checkpoints\best_model_stage2_refactored.pth",
        "pseudo_csv": r"data\csv\pseudo_labeled_stage2.csv",
        "best_alias_name": "best_model_stage3_refactored.pth",
        "log_alias_name": "train_stage3_refactored_log.csv",
    },
}


def merge_config(stage: str, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    stage_key = str(stage).lower().strip()
    if stage_key not in STAGE_PRESETS:
        raise ValueError(f"Unsupported stage: {stage!r}. Expected one of {tuple(STAGE_PRESETS)}")
    cfg = dict(DEFAULT_COMMON_CONFIG)
    cfg.update(STAGE_PRESETS[stage_key])
    if overrides:
        cfg.update(dict(overrides))
    return resolve_config(cfg)


def _is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if _is_none_like(value):
        return None
    p = Path(str(value))
    if p.is_absolute():
        return p
    return project_root / p


def resolve_config(cfg: MutableMapping[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    root = Path(str(out["project_root"])).expanduser().resolve()
    out["project_root"] = str(root)
    for key in (
        "train_csv", "val_csv", "test_csv", "img_base", "runs_dir", "checkpoint_alias_dir",
        "pseudo_csv", "init_ckpt",
    ):
        out[key] = None if _is_none_like(out.get(key)) else str(resolve_path(root, out.get(key)))
    out["num_classes"] = int(out.get("num_classes", NUM_CLASSES))
    if out["num_classes"] != NUM_CLASSES:
        raise ValueError(f"This project expects {NUM_CLASSES} classes, got num_classes={out['num_classes']}")
    return out


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_id(cfg: Mapping[str, Any]) -> str:
    custom = cfg.get("run_name")
    if not _is_none_like(custom):
        return str(custom).strip()
    return f"{cfg['stage']}_{now_stamp()}_seed{cfg['seed']}"


def set_global_seed(seed: int, *, deterministic_algorithms: bool = False, cudnn_benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    # Recommended PyTorch pattern: derive NumPy/Python seed from torch.initial_seed().
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch_index0: int) -> float:
    if epoch_index0 < warmup_epochs:
        return base_lr * float(epoch_index0 + 1) / max(1, int(warmup_epochs))
    progress = (epoch_index0 - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def rampup_weight(epoch_index0: int, rampup_epochs: int) -> float:
    if rampup_epochs <= 0:
        return 1.0
    # Smooth ramp, reaches 1 at rampup_epochs.
    t = max(0.0, min(1.0, float(epoch_index0 + 1) / float(rampup_epochs)))
    return float(math.exp(-5.0 * (1.0 - t) ** 2))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def source_file_hashes(paths: Sequence[str | Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        path = Path(p)
        if path.exists() and path.is_file():
            out[str(path)] = sha256_file(path)
    return out


def count_labels_from_dataset(ds: Dataset, num_classes: int = NUM_CLASSES) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    # Unwrap Subset if needed.
    if isinstance(ds, Subset):
        base = ds.dataset
        if hasattr(base, "samples"):
            for idx in ds.indices:
                y = int(base.samples[int(idx)].get("label", -1))
                if 0 <= y < num_classes:
                    counts[y] += 1
            return counts
    if hasattr(ds, "samples"):
        for s in getattr(ds, "samples"):
            y = int(s.get("label", -1))
            if 0 <= y < num_classes:
                counts[y] += 1
        return counts
    for i in range(len(ds)):
        item = ds[i]
        y = int(item[1]) if isinstance(item, (tuple, list)) and len(item) >= 2 else -1
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts


def effective_number_weights(counts: Sequence[int], beta: float, device: torch.device) -> torch.Tensor:
    counts_arr = np.asarray(list(counts), dtype=np.float64)
    if np.any(counts_arr <= 0):
        missing = [LABELS[i] for i, c in enumerate(counts_arr.tolist()) if c <= 0]
        raise ValueError(f"Every class must be present in training data. Missing: {missing}")
    beta = float(beta)
    if not (0.0 <= beta < 1.0):
        raise ValueError("class_balance_beta must be in [0,1)")
    if beta == 0.0:
        weights = np.ones_like(counts_arr)
    else:
        effective_num = 1.0 - np.power(beta, counts_arr)
        weights = (1.0 - beta) / effective_num
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def apply_per_class_limit(ds: FER2013Hybrid, limit: int, seed: int) -> Dataset:
    if int(limit) <= 0:
        return ds
    buckets: Dict[int, List[int]] = {i: [] for i in range(NUM_CLASSES)}
    for idx, sample in enumerate(ds.samples):
        y = int(sample.get("label", -1))
        if 0 <= y < NUM_CLASSES:
            buckets[y].append(idx)
    rng = random.Random(int(seed))
    selected: List[int] = []
    for y in range(NUM_CLASSES):
        idxs = buckets[y]
        rng.shuffle(idxs)
        selected.extend(idxs[: int(limit)])
    selected.sort()
    print(f"[data] per_class_limit={limit}; selected={len(selected)} from {len(ds)}", flush=True)
    return Subset(ds, selected)


class WeightedDataset(Dataset):
    """Wrap a dataset and return sample weights and pseudo flags."""

    def __init__(self, base: Dataset, weights: Sequence[float], is_pseudo: bool) -> None:
        if len(base) != len(weights):
            raise ValueError(f"weights length {len(weights)} != dataset length {len(base)}")
        self.base = base
        self.weights = [float(w) for w in weights]
        self.is_pseudo = bool(is_pseudo)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        return img, label, torch.tensor(self.weights[idx], dtype=torch.float32), torch.tensor(int(self.is_pseudo), dtype=torch.long)


def _row_get(row: Mapping[str, Any], *names: str) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        key = name.lower()
        if key in lower:
            return lower[key]
    return None


def read_pseudo_conf_by_row(csv_path: str | Path, *, require_conf: bool = True, conf_min: float = 0.0) -> Dict[int, float]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Pseudo CSV not found: {path}")
    conf_by_row: Dict[int, float] = {}
    errors: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Pseudo CSV has no header: {path}")
        has_conf = any(str(x).lower() == "conf" for x in reader.fieldnames)
        if require_conf and not has_conf:
            raise ValueError(f"Pseudo CSV must contain a conf column: {path}")
        for line_no, row in enumerate(reader, start=2):
            usage = str(_row_get(row, "Usage") or "").strip().lower()
            if usage and usage not in {"pseudo", "unlabeled", "u"}:
                continue
            value = _row_get(row, "conf")
            if value is None or str(value).strip() == "":
                if require_conf:
                    errors.append(f"line {line_no}: missing conf")
                    continue
                c = 1.0
            else:
                try:
                    c = float(value)
                except Exception:
                    errors.append(f"line {line_no}: invalid conf={value!r}")
                    continue
            if not (0.0 <= c <= 1.0):
                errors.append(f"line {line_no}: conf out of range [0,1]: {c}")
                continue
            if c >= float(conf_min):
                conf_by_row[line_no] = c
    if errors:
        raise ValueError("Invalid pseudo confidence values: " + " | ".join(errors[:20]))
    if not conf_by_row:
        raise RuntimeError(f"No pseudo rows remained after conf_min={conf_min}: {path}")
    return conf_by_row


def build_datasets(cfg: Mapping[str, Any]) -> Tuple[Dataset, FER2013Hybrid, FER2013Hybrid, Dict[str, Any]]:
    train_csv = str(cfg["train_csv"])
    val_csv = str(cfg["val_csv"])
    test_csv = str(cfg["test_csv"])
    img_base = cfg.get("img_base")
    img_size = int(cfg.get("img_size", IMG_SIZE))

    for name, p in (("train_csv", train_csv), ("val_csv", val_csv), ("test_csv", test_csv)):
        if not Path(p).exists():
            raise FileNotFoundError(f"{name} does not exist: {p}")

    train_base = FER2013Hybrid(train_csv, img_base, "train", img_size=img_size, include_label=True, strict=True)
    per_class_limit_val = int(cfg.get("per_class_limit") or 0)
    train_limited: Dataset = apply_per_class_limit(train_base, per_class_limit_val, int(cfg["seed"])) \
        if per_class_limit_val > 0 else train_base
    train_for_counts: Dataset = train_limited

    val_ds = FER2013Hybrid(val_csv, img_base, "val", img_size=img_size, include_label=True, strict=True)
    test_ds = FER2013Hybrid(test_csv, img_base, "test", img_size=img_size, include_label=True, strict=True)

    labeled_weights = [1.0 for _ in range(len(train_limited))]
    train_wrapped = WeightedDataset(train_limited, labeled_weights, is_pseudo=False)

    meta: Dict[str, Any] = {
        "train_labeled_count": len(train_limited),
        "train_full_labeled_count": len(train_base),
        "val_count": len(val_ds),
        "test_count": len(test_ds),
        "pseudo_count": 0,
        "label_counts": count_labels_from_dataset(train_for_counts),
    }

    pseudo_csv = cfg.get("pseudo_csv")
    if not _is_none_like(pseudo_csv):
        pseudo_path = Path(str(pseudo_csv))
        if not pseudo_path.exists():
            raise FileNotFoundError(f"This stage requires pseudo_csv but it does not exist: {pseudo_path}")
        pseudo_ds = FER2013Hybrid(str(pseudo_path), img_base, "pseudo", img_size=img_size, include_label=True, strict=True)
        conf_by_row = read_pseudo_conf_by_row(
            pseudo_path,
            require_conf=bool(cfg.get("require_pseudo_conf", True)),
            conf_min=float(cfg.get("pseudo_conf_min", 0.0)),
        )
        weights: List[float] = []
        valid_indices: List[int] = []
        for idx, sample in enumerate(pseudo_ds.samples):
            row_index = int(sample.get("row_index", -1))
            conf = conf_by_row.get(row_index)
            if conf is None:
                # Most often filtered by pseudo_conf_min. Invalid/missing conf values already raise in read_pseudo_conf_by_row.
                continue
            valid_indices.append(idx)
            weights.append((conf ** float(cfg.get("pseudo_conf_power", 2.0))) * float(cfg.get("pseudo_loss_scale", 1.0)))
        if not valid_indices:
            raise RuntimeError(f"No pseudo samples remained after filtering: {pseudo_path}")
        pseudo_filtered: Dataset = Subset(pseudo_ds, valid_indices)
        pseudo_wrapped = WeightedDataset(pseudo_filtered, weights, is_pseudo=True)
        train_ds: Dataset = ConcatDataset([train_wrapped, pseudo_wrapped])
        meta.update({
            "pseudo_count_total_csv": len(pseudo_ds),
            "pseudo_count": len(pseudo_filtered),
            "pseudo_filtered_out": int(len(pseudo_ds) - len(pseudo_filtered)),
            "pseudo_weight_min": float(min(weights)) if weights else 0.0,
            "pseudo_weight_mean": float(sum(weights) / max(1, len(weights))),
            "pseudo_weight_max": float(max(weights)) if weights else 0.0,
        })
    else:
        train_ds = train_wrapped

    return train_ds, val_ds, test_ds, meta


def make_loader(ds: Dataset, cfg: Mapping[str, Any], *, shuffle: bool, drop_last: bool, seed_offset: int = 0) -> DataLoader:
    num_workers = int(cfg.get("num_workers", 4))
    generator = make_generator(int(cfg["seed"]) + int(seed_offset))
    kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.get("batch_size", 128)),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(cfg.get("pin_memory", True)),
        "drop_last": bool(drop_last),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
    return DataLoader(ds, **kwargs)


def get_amp_objects(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    # Prefer the modern torch.amp API; fall back for older PyTorch.
    autocast_ctx = torch.amp.autocast
    scaler = None
    if enabled:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except Exception:
            from torch.cuda.amp import GradScaler  # type: ignore
            scaler = GradScaler()
    return enabled, autocast_ctx, scaler


def unpack_train_batch(batch: Sequence[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Training loader must yield at least (images, labels)")
    xb = batch[0].to(device, non_blocking=True)
    yb = batch[1].to(device, non_blocking=True)
    if len(batch) >= 4:
        sample_w = batch[2].to(device, non_blocking=True, dtype=torch.float32).view(-1)
        is_pseudo = batch[3].to(device, non_blocking=True, dtype=torch.long).view(-1)
    else:
        sample_w = torch.ones_like(yb, dtype=torch.float32, device=device)
        is_pseudo = torch.zeros_like(yb, dtype=torch.long, device=device)
    return xb, yb, sample_w, is_pseudo


def weighted_ce_loss(
    logits: torch.Tensor,
    yb: torch.Tensor,
    sample_w: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    per_sample = F.cross_entropy(
        logits,
        yb,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    weights = sample_w.to(per_sample.dtype)
    denom = torch.clamp(weights.sum(), min=1.0)
    return (per_sample * weights).sum() / denom


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Mapping[str, Any],
    epoch_index0: int,
    class_weights: Optional[torch.Tensor],
    scaler: Any,
    autocast_ctx: Any,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.train()
    acc = MetricAccumulator(num_classes=NUM_CLASSES, labels=LABELS)
    loss_sum = 0.0
    weight_sum = 0.0
    pseudo_seen = 0
    labeled_seen = 0
    pseudo_weight_sum = 0.0
    ramp = rampup_weight(epoch_index0, int(cfg.get("pseudo_rampup_epochs", 0)))
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    t0 = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        xb, yb, sample_w, is_pseudo = unpack_train_batch(batch, device)
        if is_pseudo.numel() == sample_w.numel():
            pseudo_mask = is_pseudo.bool()
            sample_w = sample_w.clone()
            sample_w[pseudo_mask] *= float(ramp)
            pseudo_seen += int(pseudo_mask.sum().item())
            labeled_seen += int((~pseudo_mask).sum().item())
            pseudo_weight_sum += float(sample_w[pseudo_mask].sum().item()) if pseudo_mask.any() else 0.0

        with autocast_ctx(device_type=device.type, enabled=bool(amp_enabled)):
            output = model(xb)
            logits, aux_logits = unpack_model_output(output)
            main_loss = weighted_ce_loss(
                logits,
                yb,
                sample_w,
                class_weights=class_weights,
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
            )
            loss = main_loss

            aux_loss_weight = float(cfg.get("aux_loss_weight", 0.0))
            if aux_loss_weight > 0.0 and aux_logits:
                aux_loss = torch.tensor(0.0, device=logits.device)
                for _aux_name, aux_out in aux_logits.items():
                    aux_loss = aux_loss + weighted_ce_loss(
                        aux_out,
                        yb,
                        sample_w,
                        class_weights=class_weights,
                        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                    )
                aux_loss = aux_loss / max(1, len(aux_logits))
                loss = main_loss + aux_loss_weight * aux_loss

        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps

        if scaler is not None and amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        n = int(yb.numel())
        loss_sum += float(loss.item()) * float(n)
        weight_sum += float(n)
        acc.update(logits.detach(), yb.detach(), loss=None)

        # Optimizer step only after accumulation
        is_accum_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(loader))
        if is_accum_step:
            if scaler is not None and amp_enabled:
                if bool(cfg.get("grad_clip", True)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("max_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                if bool(cfg.get("grad_clip", True)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("max_norm", 1.0)))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    metrics = acc.compute().to_dict()
    metrics.update({
        "loss": float(loss_sum / max(1.0, weight_sum)),
        "duration_sec": float(time.perf_counter() - t0),
        "pseudo_rampup_weight": float(ramp),
        "pseudo_seen": int(pseudo_seen),
        "labeled_seen": int(labeled_seen),
        "pseudo_effective_weight_sum": float(pseudo_weight_sum),
    })
    return metrics


def write_csv_row(path: Path, row: Mapping[str, Any], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


def get_model_state_dict(model: nn.Module) -> Mapping[str, torch.Tensor]:
    # torch.compile wraps the original module as _orig_mod.
    base = getattr(model, "_orig_mod", model)
    return base.state_dict()


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    *,
    epoch: int,
    best_metric: float,
    val_metrics: Mapping[str, Any],
    run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "run_id": run_id,
        "stage": cfg.get("stage"),
        "epoch": int(epoch),
        "best_metric_name": cfg.get("best_metric", "global_macro_f1"),
        "best_metric": float(best_metric),
        "model_state_dict": get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(cfg),
        "val_metrics": dict(val_metrics),
        "label_order": list(LABELS),
        "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def copy_alias(src: Path, dst: Path, *, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        print(f"[alias] exists, skip: {dst}", flush=True)
        return
    shutil.copy2(src, dst)
    print(f"[alias] {src} -> {dst}", flush=True)


def write_manifest(path: Path, cfg: Mapping[str, Any], run_id: str, dataset_meta: Mapping[str, Any], status: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    files_to_hash = [
        cfg.get("train_csv"), cfg.get("val_csv"), cfg.get("test_csv"), cfg.get("pseudo_csv"), cfg.get("init_ckpt"),
        Path(__file__), Path(__file__).with_name("dataset.py"), Path(__file__).with_name("metrics.py"), Path(__file__).with_name("model_mbv3.py"),
    ]
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "created_or_updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "labels": list(LABELS),
        "config": dict(cfg),
        "dataset_meta": dict(dataset_meta),
        "file_sha256": source_file_hashes([p for p in files_to_hash if not _is_none_like(p)]),
    }
    if extra:
        manifest.update(dict(extra))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_training(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = resolve_config(dict(cfg))
    run_id = make_run_id(cfg)
    run_dir = Path(str(cfg["runs_dir"])) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "evaluation"
    log_csv = run_dir / "metrics_epoch.csv"
    metrics_jsonl = run_dir / "metrics_epoch.jsonl"
    config_json = run_dir / "resolved_config.json"
    manifest_json = run_dir / "manifest.json"
    config_json.write_text(json.dumps(dict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== FER Pi5 refactored training ===", flush=True)
    print(f"stage    : {cfg['stage']}", flush=True)
    print(f"run_id   : {run_id}", flush=True)
    print(f"run_dir  : {run_dir}", flush=True)
    print(f"device   : {cfg['device']}", flush=True)
    print("label order:", ", ".join(f"{i}:{name}" for i, name in enumerate(LABELS)), flush=True)

    set_global_seed(
        int(cfg["seed"]),
        deterministic_algorithms=bool(cfg.get("deterministic_algorithms", False)),
        cudnn_benchmark=bool(cfg.get("cudnn_benchmark", False)),
    )
    device = torch.device(str(cfg["device"]))

    train_ds, val_ds, test_ds, dataset_meta = build_datasets(cfg)
    write_manifest(manifest_json, cfg, run_id, dataset_meta, status="started")

    train_loader = make_loader(train_ds, cfg, shuffle=True, drop_last=bool(cfg.get("drop_last_train", True)), seed_offset=0)
    val_loader = make_loader(val_ds, cfg, shuffle=False, drop_last=False, seed_offset=10)
    test_loader = make_loader(test_ds, cfg, shuffle=False, drop_last=False, seed_offset=20)

    model = get_model(
        variant=str(cfg.get("model_variant", "large")),
        num_classes=NUM_CLASSES,
        pretrained=bool(cfg.get("pretrained", True)),
        device=device,
        verbose=True,
        compile_model=bool(cfg.get("compile_model", False)),
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
    )
    init_ckpt = cfg.get("init_ckpt")
    if not _is_none_like(init_ckpt):
        print(f"[checkpoint] loading init checkpoint: {init_ckpt}", flush=True)
        load_checkpoint_into_model(model, str(init_ckpt), device=device, strict=bool(cfg.get("strict_checkpoint_load", True)))

    class_weights = None
    if bool(cfg.get("use_class_weights", True)):
        class_counts = dataset_meta["label_counts"]
        class_weights = effective_number_weights(class_counts, float(cfg.get("class_balance_beta", 0.995)), device=device)
        print("[loss] class counts:", dict(zip(LABELS, class_counts)), flush=True)
        print("[loss] class weights:", [round(float(x), 4) for x in class_weights.detach().cpu().tolist()], flush=True)

    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    amp_enabled, autocast_ctx, scaler = get_amp_objects(device, bool(cfg.get("use_amp", True)))
    print(f"[amp] enabled={amp_enabled}", flush=True)

    best_metric = -float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = checkpoint_dir / "best_model.pth"
    last_path = checkpoint_dir / "last_model.pth"
    csv_fields = [
        "epoch", "lr", "train_loss", "train_acc", "train_global_macro_f1",
        "val_loss", "val_acc", "val_global_macro_f1", "best_val_global_macro_f1",
        "best_epoch", "bad_epochs", "pseudo_rampup_weight", "pseudo_seen", "duration_sec",
    ]

    for epoch_index0 in range(int(cfg["epochs"])):
        epoch = epoch_index0 + 1
        lr = cosine_warmup_lr(
            float(cfg["lr"]), float(cfg["lr_floor"]), int(cfg["warmup_epochs"]), int(cfg["epochs"]), epoch_index0,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, cfg, epoch_index0, class_weights, scaler, autocast_ctx, amp_enabled
        )
        should_validate = epoch % int(cfg.get("val_interval", 1)) == 0
        if should_validate:
            criterion_eval = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
            val_result = evaluate_model(model, val_loader, device, criterion=criterion_eval, num_classes=NUM_CLASSES, labels=LABELS)
            val_metrics = val_result.to_dict()
            metric_value = float(val_metrics["global_macro_f1"])
        else:
            val_metrics = {"loss": None, "accuracy": None, "global_macro_f1": None}
            metric_value = -float("inf")

        improved = should_validate and metric_value > best_metric + 1e-12
        if improved:
            best_metric = metric_value
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, cfg, epoch=epoch, best_metric=best_metric, val_metrics=val_metrics, run_id=run_id)
            save_metrics_json(val_metrics, eval_dir / "val_best_metrics.json")
            save_confusion_png(val_metrics["confusion"], eval_dir / "val_best_confusion.png", title="Validation Confusion Matrix")
        elif should_validate:
            bad_epochs += 1

        if bool(cfg.get("save_last_every_epoch", True)):
            save_checkpoint(last_path, model, optimizer, cfg, epoch=epoch, best_metric=best_metric, val_metrics=val_metrics, run_id=run_id)

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_global_macro_f1": train_metrics["global_macro_f1"],
            "val_loss": val_metrics.get("loss"),
            "val_acc": val_metrics.get("accuracy"),
            "val_global_macro_f1": val_metrics.get("global_macro_f1"),
            "best_val_global_macro_f1": best_metric,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "pseudo_rampup_weight": train_metrics.get("pseudo_rampup_weight"),
            "pseudo_seen": train_metrics.get("pseudo_seen"),
            "duration_sec": train_metrics.get("duration_sec"),
        }
        write_csv_row(log_csv, row, fieldnames=csv_fields)
        append_jsonl(metrics_jsonl, {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics, "best_epoch": best_epoch, "best_metric": best_metric})
        print(
            f"[epoch {epoch:03d}] lr={lr:.3e} train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_global_macro_f1']:.4f} val_f1={row['val_global_macro_f1']:.4f} "
            f"best={best_metric:.4f}@{best_epoch} bad={bad_epochs}",
            flush=True,
        )
        if should_validate and bad_epochs >= int(cfg.get("early_stop_patience", 20)):
            print(f"[early-stop] no improvement for {bad_epochs} validation rounds", flush=True)
            break

    if not best_path.exists():
        raise RuntimeError("Training finished without saving a best checkpoint. Check val_interval and validation data.")

    final_extra: Dict[str, Any] = {"best_epoch": best_epoch, "best_metric": best_metric, "best_checkpoint": str(best_path)}
    if bool(cfg.get("evaluate_test_at_end", True)):
        print("[final] loading best checkpoint for final validation/test evaluation", flush=True)
        # Build a fresh uncompiled model to avoid wrapper state names.
        eval_model = get_model(
            variant=str(cfg.get("model_variant", "large")),
            num_classes=NUM_CLASSES,
            pretrained=False,
            device=device,
            verbose=False,
            compile_model=False,
            use_checkpoint=False,
        )
        load_checkpoint_into_model(
            eval_model,
            str(best_path),
            device=device,
            strict=bool(cfg.get("strict_checkpoint_load", True)),
        )
        eval_model.eval()
        criterion_eval = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.0)
        val_best = evaluate_model(eval_model, val_loader, device, criterion=criterion_eval, num_classes=NUM_CLASSES, labels=LABELS)
        test_best = evaluate_model(eval_model, test_loader, device, criterion=criterion_eval, num_classes=NUM_CLASSES, labels=LABELS)
        val_payload = val_best.to_dict()
        test_payload = test_best.to_dict()
        save_metrics_json(val_payload, eval_dir / "val_best_reloaded_metrics.json")
        save_metrics_json(test_payload, eval_dir / "test_best_reloaded_metrics.json")
        save_confusion_png(val_payload["confusion"], eval_dir / "val_best_reloaded_confusion.png", title="Validation Confusion Matrix")
        save_confusion_png(test_payload["confusion"], eval_dir / "test_best_reloaded_confusion.png", title="Test Confusion Matrix")
        final_extra.update({"final_val": val_payload, "final_test": test_payload})
        print(f"[final] val_f1={val_payload['global_macro_f1']:.6f} test_f1={test_payload['global_macro_f1']:.6f}", flush=True)

    if bool(cfg.get("write_checkpoint_alias", True)):
        alias_dir = Path(str(cfg["checkpoint_alias_dir"]))
        best_alias = alias_dir / str(cfg.get("best_alias_name", f"best_model_{cfg['stage']}_refactored.pth"))
        copy_alias(best_path, best_alias, overwrite=bool(cfg.get("alias_overwrite", True)))
        log_alias = alias_dir / str(cfg.get("log_alias_name", f"train_{cfg['stage']}_refactored_log.csv"))
        copy_alias(log_csv, log_alias, overwrite=bool(cfg.get("alias_overwrite", True)))
        final_extra.update({"best_alias": str(best_alias), "log_alias": str(log_alias)})

    write_manifest(manifest_json, cfg, run_id, dataset_meta, status="finished", extra=final_extra)
    print("=== Training complete ===", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"best  : {best_path}", flush=True)
    return {"run_id": run_id, "run_dir": str(run_dir), "best_checkpoint": str(best_path), **final_extra}


def main(stage: str, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg = merge_config(stage, overrides)
    return run_training(cfg)
