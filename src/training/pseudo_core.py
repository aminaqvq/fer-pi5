from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import inspect
import json
import logging
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "torchvision is required for pseudo label generation. "
        "Install the same torchvision version used by your training environment."
    ) from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

# Import project modules after sys.path has been prepared by the caller script.
try:
    from model_mbv3 import get_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cannot import get_model from model_mbv3.py. "
        "Run this script from src/training or make sure F:\\fer-pi5\\src\\training is on sys.path."
    ) from exc

try:
    from dataset import IMG_SIZE as PROJECT_IMG_SIZE
    from dataset import IMAGENET_MEAN, IMAGENET_STD
except Exception:
    PROJECT_IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


LABEL_NAMES: Tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

LABEL_ALIASES: Dict[str, int] = {
    "0": 0, "anger": 0, "angry": 0,
    "1": 1, "disgust": 1, "disgusted": 1,
    "2": 2, "fear": 2, "fearful": 2,
    "3": 3, "happy": 3, "happiness": 3,
    "4": 4, "sad": 4, "sadness": 4,
    "5": 5, "surprise": 5, "surprised": 5,
    "6": 6, "neutral": 6,
}


@dataclass(frozen=True)
class Candidate:
    source_index: int
    sample_id: str
    pred_label: int
    pred_name: str
    confidence: float
    margin: float
    entropy: float
    threshold_used: float
    selected: bool
    reject_reason: str
    raw_row: Dict[str, str]
    probs: Tuple[float, ...]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    # PyTorch recommended pattern: each worker gets a unique, bounded seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pseudo_rebalance_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "pseudo_rebalance_generation.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def normalize_thresholds(class_min_conf: Mapping[Any, Any], default: float) -> Dict[int, float]:
    out: Dict[int, float] = {i: float(default) for i in range(len(LABEL_NAMES))}
    for k, v in (class_min_conf or {}).items():
        if isinstance(k, int):
            label = k
        else:
            key = str(k).strip().lower()
            if key not in LABEL_ALIASES:
                raise ValueError(f"Unknown class key in class_min_conf: {k!r}")
            label = LABEL_ALIASES[key]
        value = float(v)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Threshold for class {label} must be in [0,1], got {value}")
        out[label] = value
    return out


def normalize_caps(class_max_per_class: Mapping[Any, Any], default: Optional[int]) -> Dict[int, Optional[int]]:
    out: Dict[int, Optional[int]] = {i: (None if default in (None, 0, "0", "None", "") else int(default)) for i in range(len(LABEL_NAMES))}
    for k, v in (class_max_per_class or {}).items():
        if isinstance(k, int):
            label = k
        else:
            key = str(k).strip().lower()
            if key not in LABEL_ALIASES:
                raise ValueError(f"Unknown class key in class_max_per_class: {k!r}")
            label = LABEL_ALIASES[key]
        if v in (None, 0, "0", "None", ""):
            out[label] = None
        else:
            iv = int(v)
            if iv <= 0:
                raise ValueError(f"Cap for class {label} must be positive or None, got {iv}")
            out[label] = iv
    return out


def build_eval_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_fer_pixels_strict(pixels: str) -> np.ndarray:
    if pixels is None or str(pixels).strip() == "":
        raise ValueError("empty pixels field")
    arr = np.fromstring(str(pixels), dtype=np.uint8, sep=" ")
    if arr.size != 48 * 48:
        raise ValueError(f"expected 2304 pixel values, got {arr.size}")
    return arr.reshape(48, 48)


def to_rgb_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


class PseudoUnlabeledCsvDataset(Dataset):
    """
    Strict unlabeled CSV reader.

    Accepted input columns:
    - pixels
    - optional path/filepath/image
    - optional sample_id
    - optional emotion/label is preserved for audit only and never used for selection

    The output order is exactly the CSV row order after optional Usage filtering.
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_root: Optional[str | Path],
        img_size: int,
        require_pixels_or_path: bool = True,
        strict_pixels: bool = True,
        usage_filter: Optional[Sequence[str]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.img_root = None if img_root in (None, "", "None") else Path(str(img_root))
        self.transform = build_eval_transform(int(img_size))
        self.strict_pixels = bool(strict_pixels)
        self.rows: List[Dict[str, str]] = []

        if not self.csv_path.exists():
            raise FileNotFoundError(f"unlabeled CSV not found: {self.csv_path}")

        usage_allow = None
        if usage_filter:
            usage_allow = {str(x).strip().lower() for x in usage_filter}

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {self.csv_path}")

            lowered = {name.lower(): name for name in reader.fieldnames}
            has_pixels = "pixels" in lowered
            has_path = any(k in lowered for k in ("path", "filepath", "image"))
            if require_pixels_or_path and not (has_pixels or has_path):
                raise ValueError("unlabeled CSV must contain either pixels or path/filepath/image")

            for row_idx, row in enumerate(reader):
                usage = str(row.get("Usage") or row.get("usage") or "").strip().lower()
                if usage_allow is not None and usage and usage not in usage_allow:
                    continue

                normalized = {str(k): ("" if v is None else str(v)) for k, v in row.items()}
                if "sample_id" not in normalized or not normalized.get("sample_id"):
                    raw_for_id = normalized.get("pixels") or normalized.get("path") or normalized.get("filepath") or normalized.get("image") or str(row_idx)
                    normalized["sample_id"] = sha256_text(raw_for_id)[:16]
                normalized["_source_index"] = str(len(self.rows))
                self.rows.append(normalized)

        if not self.rows:
            raise RuntimeError(f"No usable rows from unlabeled CSV: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, row: Mapping[str, str]) -> Optional[Path]:
        raw = row.get("path") or row.get("filepath") or row.get("image") or ""
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute() and self.img_root is not None:
            p = self.img_root / p
        return p

    def _load_image(self, row: Mapping[str, str]) -> Image.Image:
        p = self._resolve_path(row)
        if p is not None:
            if not p.exists():
                raise FileNotFoundError(f"image path not found: {p}")
            return Image.open(p).convert("RGB")

        pixels = row.get("pixels", "")
        if self.strict_pixels:
            return to_rgb_pil(load_fer_pixels_strict(pixels))

        arr = np.fromstring(str(pixels), dtype=np.uint8, sep=" ")
        if arr.size != 48 * 48:
            side = int(math.sqrt(max(arr.size, 1)))
            side = max(8, side)
            padded = np.zeros(side * side, dtype=np.uint8)
            padded[: min(arr.size, side * side)] = arr[: min(arr.size, side * side)]
            arr = padded.reshape(side, side)
        else:
            arr = arr.reshape(48, 48)
        return to_rgb_pil(arr)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = self._load_image(row)
        return self.transform(img), idx


def safe_torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    kwargs = {"map_location": map_location}
    try:
        sig = inspect.signature(torch.load)
        if "weights_only" in sig.parameters:
            kwargs["weights_only"] = True
    except Exception:
        pass
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def extract_state_dict(ckpt: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(ckpt, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = ckpt.get(key)
            if isinstance(value, Mapping):
                ckpt = value
                break

    if not isinstance(ckpt, Mapping):
        raise TypeError(f"checkpoint does not contain a state dict; got {type(ckpt)!r}")

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if not torch.is_tensor(v):
            continue
        name = str(k)
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        cleaned[name] = v

    if not cleaned:
        raise ValueError("checkpoint state dict contains no tensor weights")
    return cleaned


def load_teacher_model(cfg: Mapping[str, Any], logger: logging.Logger):
    requested_device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    variant = str(cfg.get("model_variant", "large"))
    num_classes = int(cfg.get("num_classes", len(LABEL_NAMES)))
    pretrained = bool(cfg.get("pretrained", False))
    compile_model = bool(cfg.get("compile_model", False))

    model = get_model(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        device=str(device),
        verbose=True,
        compile_model=compile_model,
    )

    ckpt_path = Path(str(cfg["teacher_ckpt"]))
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")

    logger.info("Loading teacher checkpoint: %s", ckpt_path)
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    state = extract_state_dict(ckpt)
    strict = bool(cfg.get("strict_checkpoint", True))
    result = model.load_state_dict(state, strict=strict)

    missing = list(getattr(result, "missing_keys", []) or [])
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if missing or unexpected:
        logger.warning("Checkpoint load result: missing=%s unexpected=%s", missing, unexpected)
        if strict:
            raise RuntimeError(f"strict checkpoint load failed: missing={missing}, unexpected={unexpected}")

    model.to(device).eval()
    return model, device, {"missing_keys": missing, "unexpected_keys": unexpected}


@torch.inference_mode()
def score_unlabeled_pool(cfg: Mapping[str, Any], run_dir: Path, logger: logging.Logger) -> Tuple[List[Dict[str, Any]], PseudoUnlabeledCsvDataset]:
    seed_everything(int(cfg.get("seed", 42)))

    dataset = PseudoUnlabeledCsvDataset(
        csv_path=cfg["unlabeled_csv"],
        img_root=cfg.get("img_root"),
        img_size=int(cfg.get("img_size", PROJECT_IMG_SIZE)),
        strict_pixels=bool(cfg.get("strict_pixels", True)),
        usage_filter=cfg.get("usage_filter"),
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)))

    num_workers = int(cfg.get("num_workers", 0))
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=False,
        generator=generator,
    )
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = worker_init_fn
        loader_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        loader_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", False))

    loader = DataLoader(dataset, **loader_kwargs)

    model, device, load_info = load_teacher_model(cfg, logger)
    run_dir.joinpath("checkpoint_load_info.json").write_text(json.dumps(load_info, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Unlabeled samples: %s", len(dataset))
    logger.info("Scoring full unlabeled pool before selection. TTA hflip=%s", bool(cfg.get("tta_hflip", True)))

    candidates: List[Dict[str, Any]] = []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Scoring unlabeled", ncols=100)

    for xb, indices in iterator:
        xb = xb.to(device, non_blocking=True)

        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits.get("main", next(iter(logits.values())))
        if bool(cfg.get("tta_hflip", True)):
            logits_flip = model(torch.flip(xb, dims=[-1]))
            if isinstance(logits_flip, dict):
                logits_flip = logits_flip.get("main", next(iter(logits_flip.values())))
            logits = 0.5 * (logits + logits_flip)

        probs = F.softmax(logits, dim=1)
        if probs.ndim != 2 or probs.shape[1] != len(LABEL_NAMES):
            raise RuntimeError(f"model output must be [batch, 7], got {tuple(probs.shape)}")

        top2 = torch.topk(probs, k=2, dim=1)
        conf = top2.values[:, 0]
        second = top2.values[:, 1]
        pred = top2.indices[:, 0]
        margin = conf - second
        entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=1)

        for row_idx, p, c, m, e in zip(indices.tolist(), pred.tolist(), conf.tolist(), margin.tolist(), entropy.tolist()):
            row = dataset.rows[int(row_idx)]
            prob_tuple = tuple(float(x) for x in probs[indices.tolist().index(row_idx)].detach().cpu().tolist()) if False else ()
            # Avoid indexing complexity above; fill below from per-sample tensor.
            candidates.append({
                "source_index": int(row_idx),
                "sample_id": row.get("sample_id", ""),
                "pred_label": int(p),
                "pred_name": LABEL_NAMES[int(p)],
                "confidence": float(c),
                "margin": float(m),
                "entropy": float(e),
                "raw_row": row,
                "probs": None,
            })

        # Fill probabilities correctly for this batch.
        batch_probs = probs.detach().cpu().tolist()
        batch_indices = indices.tolist()
        for local_idx, src_idx in enumerate(batch_indices):
            candidates[-len(batch_indices) + local_idx]["probs"] = tuple(float(x) for x in batch_probs[local_idx])

    return candidates, dataset


def select_pseudo_labels(
    candidates: Sequence[Mapping[str, Any]],
    class_min_conf: Mapping[int, float],
    class_max_per_class: Mapping[int, Optional[int]],
    min_margin: float = 0.0,
    max_entropy: Optional[float] = None,
) -> Tuple[List[Candidate], List[Candidate]]:
    """Select by complete-pool class buckets, confidence desc, margin desc, entropy asc."""
    buckets: Dict[int, List[Mapping[str, Any]]] = {i: [] for i in range(len(LABEL_NAMES))}
    rejected_initial: List[Candidate] = []

    for cand in candidates:
        y = int(cand["pred_label"])
        thr = float(class_min_conf[y])
        confidence = float(cand["confidence"])
        margin = float(cand["margin"])
        entropy = float(cand["entropy"])

        reason = ""
        if confidence < thr:
            reason = "below_class_threshold"
        elif margin < float(min_margin):
            reason = "below_min_margin"
        elif max_entropy is not None and entropy > float(max_entropy):
            reason = "above_max_entropy"

        if reason:
            rejected_initial.append(Candidate(
                source_index=int(cand["source_index"]),
                sample_id=str(cand.get("sample_id", "")),
                pred_label=y,
                pred_name=LABEL_NAMES[y],
                confidence=confidence,
                margin=margin,
                entropy=entropy,
                threshold_used=thr,
                selected=False,
                reject_reason=reason,
                raw_row=dict(cand["raw_row"]),
                probs=tuple(float(x) for x in cand["probs"]),
            ))
        else:
            buckets[y].append(cand)

    selected: List[Candidate] = []
    rejected_cap: List[Candidate] = []

    for y, bucket in buckets.items():
        bucket_sorted = sorted(
            bucket,
            key=lambda x: (
                -float(x["confidence"]),
                -float(x["margin"]),
                float(x["entropy"]),
                int(x["source_index"]),
            ),
        )
        cap = class_max_per_class.get(y)
        if cap is None:
            keep = len(bucket_sorted)
        else:
            keep = min(int(cap), len(bucket_sorted))

        for pos, cand in enumerate(bucket_sorted):
            is_selected = pos < keep
            selected_flag = is_selected
            reject_reason = "" if is_selected else "over_class_cap"
            record = Candidate(
                source_index=int(cand["source_index"]),
                sample_id=str(cand.get("sample_id", "")),
                pred_label=y,
                pred_name=LABEL_NAMES[y],
                confidence=float(cand["confidence"]),
                margin=float(cand["margin"]),
                entropy=float(cand["entropy"]),
                threshold_used=float(class_min_conf[y]),
                selected=selected_flag,
                reject_reason=reject_reason,
                raw_row=dict(cand["raw_row"]),
                probs=tuple(float(x) for x in cand["probs"]),
            )
            if is_selected:
                selected.append(record)
            else:
                rejected_cap.append(record)

    selected = sorted(selected, key=lambda x: (x.pred_label, -x.confidence, -x.margin, x.entropy, x.source_index))
    rejected = rejected_initial + rejected_cap
    rejected = sorted(rejected, key=lambda x: (x.pred_label, x.reject_reason, -x.confidence, x.source_index))
    return selected, rejected


def summarize_records(records: Sequence[Candidate]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total": len(records),
        "per_class": {},
        "confidence": {},
        "margin": {},
        "entropy": {},
    }
    if not records:
        return summary

    confs = np.array([r.confidence for r in records], dtype=np.float64)
    margins = np.array([r.margin for r in records], dtype=np.float64)
    ents = np.array([r.entropy for r in records], dtype=np.float64)

    def stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "min": float(np.min(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p25": float(np.quantile(arr, 0.25)),
            "median": float(np.quantile(arr, 0.50)),
            "mean": float(np.mean(arr)),
            "p75": float(np.quantile(arr, 0.75)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }

    summary["confidence"] = stats(confs)
    summary["margin"] = stats(margins)
    summary["entropy"] = stats(ents)

    for y, name in enumerate(LABEL_NAMES):
        cls = [r for r in records if r.pred_label == y]
        arr = np.array([r.confidence for r in cls], dtype=np.float64) if cls else np.array([], dtype=np.float64)
        summary["per_class"][name] = {
            "label": y,
            "count": len(cls),
            "confidence": stats(arr) if len(arr) else {},
        }
    return summary


def write_candidates_csv(path: Path, records: Sequence[Candidate], include_pixels: bool, include_probs: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selected",
        "reject_reason",
        "source_index",
        "sample_id",
        "label",
        "emotion",
        "pred_name",
        "conf",
        "margin",
        "entropy",
        "threshold_used",
        "Usage",
        "path",
    ]
    if include_pixels:
        fields.append("pixels")
    if include_probs:
        fields.extend([f"prob_{name}" for name in LABEL_NAMES])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            raw = rec.raw_row
            row = {
                "selected": int(rec.selected),
                "reject_reason": rec.reject_reason,
                "source_index": rec.source_index,
                "sample_id": rec.sample_id,
                "label": rec.pred_label,
                "emotion": rec.pred_label,
                "pred_name": rec.pred_name,
                "conf": f"{rec.confidence:.8f}",
                "margin": f"{rec.margin:.8f}",
                "entropy": f"{rec.entropy:.8f}",
                "threshold_used": f"{rec.threshold_used:.8f}",
                "Usage": "pseudo",
                "path": raw.get("path") or raw.get("filepath") or raw.get("image") or "",
            }
            if include_pixels:
                row["pixels"] = raw.get("pixels", "")
            if include_probs:
                for name, value in zip(LABEL_NAMES, rec.probs):
                    row[f"prob_{name}"] = f"{float(value):.8f}"
            writer.writerow(row)


def write_training_pseudo_csv(path: Path, selected: Sequence[Candidate]) -> None:
    """CSV consumed by Stage2/Stage3 training scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "emotion",
        "pixels",
        "path",
        "Usage",
        "conf",
        "sample_id",
        "source_index",
        "teacher_pred_name",
        "margin",
        "entropy",
        "threshold_used",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in selected:
            raw = rec.raw_row
            writer.writerow({
                "label": rec.pred_label,
                "emotion": rec.pred_label,
                "pixels": raw.get("pixels", ""),
                "path": raw.get("path") or raw.get("filepath") or raw.get("image") or "",
                "Usage": "pseudo",
                "conf": f"{rec.confidence:.8f}",
                "sample_id": rec.sample_id,
                "source_index": rec.source_index,
                "teacher_pred_name": rec.pred_name,
                "margin": f"{rec.margin:.8f}",
                "entropy": f"{rec.entropy:.8f}",
                "threshold_used": f"{rec.threshold_used:.8f}",
            })


def write_class_counts(path: Path, selected: Sequence[Candidate], rejected: Sequence[Candidate], thresholds: Mapping[int, float], caps: Mapping[int, Optional[int]]) -> None:
    fields = [
        "label",
        "class_name",
        "selected",
        "rejected_below_threshold",
        "rejected_below_margin",
        "rejected_above_entropy",
        "rejected_over_cap",
        "threshold",
        "cap",
        "selected_conf_min",
        "selected_conf_median",
        "selected_conf_mean",
        "selected_conf_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for y, name in enumerate(LABEL_NAMES):
            sel = [r for r in selected if r.pred_label == y]
            rej = [r for r in rejected if r.pred_label == y]
            confs = np.array([r.confidence for r in sel], dtype=np.float64)
            row = {
                "label": y,
                "class_name": name,
                "selected": len(sel),
                "rejected_below_threshold": sum(1 for r in rej if r.reject_reason == "below_class_threshold"),
                "rejected_below_margin": sum(1 for r in rej if r.reject_reason == "below_min_margin"),
                "rejected_above_entropy": sum(1 for r in rej if r.reject_reason == "above_max_entropy"),
                "rejected_over_cap": sum(1 for r in rej if r.reject_reason == "over_class_cap"),
                "threshold": thresholds[y],
                "cap": "" if caps.get(y) is None else caps[y],
                "selected_conf_min": "" if len(confs) == 0 else f"{np.min(confs):.8f}",
                "selected_conf_median": "" if len(confs) == 0 else f"{np.quantile(confs, 0.50):.8f}",
                "selected_conf_mean": "" if len(confs) == 0 else f"{np.mean(confs):.8f}",
                "selected_conf_max": "" if len(confs) == 0 else f"{np.max(confs):.8f}",
            }
            writer.writerow(row)


def run_pseudo_rebalance_generation(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    project_root = Path(str(cfg.get("project_root", r"F:\fer-pi5")))
    run_base = Path(str(cfg.get("run_base_dir", project_root / "runs" / "pseudo_labels")))
    stage_name = str(cfg.get("stage_name", "stage1"))
    run_id = str(cfg.get("run_id") or f"{stage_name}_rebalanced_{now_stamp()}")

    run_dir = run_base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)

    logger.info("=== Pseudo-label rebalanced generation started ===")
    logger.info("Run ID: %s", run_id)

    thresholds = normalize_thresholds(
        cfg.get("class_min_conf", {}),
        float(cfg.get("default_min_conf", 0.85)),
    )
    caps = normalize_caps(
        cfg.get("class_max_per_class", {}),
        cfg.get("default_max_per_class", None),
    )

    logger.info("Class thresholds: %s", {LABEL_NAMES[k]: v for k, v in thresholds.items()})
    logger.info("Class caps: %s", {LABEL_NAMES[k]: v for k, v in caps.items()})

    candidates_raw, dataset = score_unlabeled_pool(cfg, run_dir, logger)
    logger.info("Candidate pool scored: %s", len(candidates_raw))

    selected, rejected = select_pseudo_labels(
        candidates_raw,
        class_min_conf=thresholds,
        class_max_per_class=caps,
        min_margin=float(cfg.get("min_margin", 0.0)),
        max_entropy=(None if cfg.get("max_entropy") in (None, "", "None") else float(cfg.get("max_entropy"))),
    )

    logger.info("Selected pseudo labels: %s", len(selected))
    logger.info("Rejected pseudo labels: %s", len(rejected))

    include_pixels_in_audit = bool(cfg.get("include_pixels_in_audit", False))
    include_probs_in_audit = bool(cfg.get("include_probs_in_audit", True))

    output_dir = Path(str(cfg.get("output_dir", project_root / "data" / "csv")))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = str(cfg.get("output_csv_name", f"pseudo_labeled_{stage_name}_rebalanced.csv"))
    out_csv = output_dir / out_name

    run_pseudo_csv = run_dir / out_name
    write_training_pseudo_csv(run_pseudo_csv, selected)
    shutil.copy2(run_pseudo_csv, out_csv)

    write_candidates_csv(run_dir / "selected_pseudo_labels_audit.csv", selected, include_pixels=include_pixels_in_audit, include_probs=include_probs_in_audit)
    write_candidates_csv(run_dir / "rejected_pseudo_labels_audit.csv", rejected, include_pixels=False, include_probs=include_probs_in_audit)
    write_class_counts(run_dir / "class_counts.csv", selected, rejected, thresholds, caps)

    if bool(cfg.get("write_all_candidates", True)):
        # Full audit with every candidate can be useful but large. Pixels are intentionally omitted.
        all_records: List[Candidate] = []
        selected_keys = {(r.source_index, r.pred_label) for r in selected}
        rejected_by_index = {(r.source_index, r.pred_label): r for r in rejected}
        for cand in candidates_raw:
            y = int(cand["pred_label"])
            key = (int(cand["source_index"]), y)
            if key in selected_keys:
                continue
            if key in rejected_by_index:
                continue
            # Should not happen, but keep an explicit record if selection logic changes later.
            rejected.append(Candidate(
                source_index=int(cand["source_index"]),
                sample_id=str(cand.get("sample_id", "")),
                pred_label=y,
                pred_name=LABEL_NAMES[y],
                confidence=float(cand["confidence"]),
                margin=float(cand["margin"]),
                entropy=float(cand["entropy"]),
                threshold_used=float(thresholds[y]),
                selected=False,
                reject_reason="unclassified_reject",
                raw_row=dict(cand["raw_row"]),
                probs=tuple(float(x) for x in cand["probs"]),
            ))
        all_records = list(selected) + list(rejected)
        write_candidates_csv(run_dir / "all_candidates_audit.csv", all_records, include_pixels=False, include_probs=include_probs_in_audit)

    compatibility_alias = cfg.get("compatibility_alias")
    alias_path = None
    if compatibility_alias:
        alias_path = output_dir / str(compatibility_alias)
        shutil.copy2(out_csv, alias_path)
        logger.info("Compatibility alias written: %s", alias_path)

    selected_summary = summarize_records(selected)
    rejected_summary = summarize_records(rejected)
    manifest = {
        "run_id": run_id,
        "stage_name": stage_name,
        "created_at_utc": iso_now(),
        "project_root": str(project_root),
        "unlabeled_csv": str(Path(str(cfg["unlabeled_csv"]))),
        "unlabeled_csv_sha256": sha256_file(Path(str(cfg["unlabeled_csv"]))),
        "teacher_ckpt": str(Path(str(cfg["teacher_ckpt"]))),
        "teacher_ckpt_sha256": sha256_file(Path(str(cfg["teacher_ckpt"]))),
        "output_csv": str(out_csv),
        "output_csv_sha256": sha256_file(out_csv),
        "compatibility_alias": str(alias_path) if alias_path else "",
        "label_order": list(LABEL_NAMES),
        "selection_policy": {
            "name": "complete_pool_class_adaptive_threshold_topk",
            "class_min_conf": {LABEL_NAMES[k]: thresholds[k] for k in range(len(LABEL_NAMES))},
            "class_max_per_class": {LABEL_NAMES[k]: caps[k] for k in range(len(LABEL_NAMES))},
            "min_margin": float(cfg.get("min_margin", 0.0)),
            "max_entropy": cfg.get("max_entropy"),
            "ranking": ["confidence desc", "margin desc", "entropy asc", "source_index asc"],
        },
        "counts": {
            "unlabeled_total": len(dataset),
            "candidate_total": len(candidates_raw),
            "selected_total": len(selected),
            "rejected_total": len(rejected),
        },
        "selected_summary": selected_summary,
        "rejected_summary": rejected_summary,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(cfg).items()},
        "audit_files": {
            "run_pseudo_csv": str(run_pseudo_csv),
            "selected_audit": str(run_dir / "selected_pseudo_labels_audit.csv"),
            "rejected_audit": str(run_dir / "rejected_pseudo_labels_audit.csv"),
            "all_candidates_audit": str(run_dir / "all_candidates_audit.csv"),
            "class_counts": str(run_dir / "class_counts.csv"),
            "log": str(run_dir / "pseudo_rebalance_generation.log"),
        },
    }
    manifest_path = run_dir / "pseudo_rebalance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Manifest written: %s", manifest_path)
    logger.info("Output pseudo CSV: %s", out_csv)
    if alias_path:
        logger.info("Compatibility output: %s", alias_path)
    logger.info("Per-class selected counts:")
    for name, stats in selected_summary.get("per_class", {}).items():
        logger.info("  %-8s selected=%s", name, stats.get("count", 0))
    logger.info("=== Pseudo-label rebalanced generation finished ===")
    return manifest


def print_manifest_summary(manifest: Mapping[str, Any]) -> None:
    print("\n=== Pseudo-label Rebalance Summary ===")
    print(f"run_id          : {manifest.get('run_id')}")
    print(f"selected_total  : {manifest.get('counts', {}).get('selected_total')}")
    print(f"output_csv      : {manifest.get('output_csv')}")
    print(f"manifest        : {manifest.get('audit_files', {}).get('class_counts')}")
    print("\nPer-class selected:")
    for name, stats in manifest.get("selected_summary", {}).get("per_class", {}).items():
        print(f"  {name:<8} {stats.get('count', 0):>8}")
