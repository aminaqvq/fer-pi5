from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import RandAugment

try:
    from .metrics import LABELS, NUM_CLASSES
except ImportError:  # direct script imports from train_stage*.py
    from metrics import LABELS, NUM_CLASSES

IMG_SIZE = 224
FER_PIXEL_SIDE = 48
FER_PIXEL_COUNT = FER_PIXEL_SIDE * FER_PIXEL_SIDE
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABELS)}
LABEL_ALIASES: Dict[str, str] = {
    "angry": "anger",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "neutral": "neutral",
}


def get_labeled_transforms(split: str = "train", img_size: int = IMG_SIZE) -> transforms.Compose:
    split = str(split).lower()
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_weak_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_strong_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.80, 1.00)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        RandAugment(num_ops=2, magnitude=7),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])


def _normalize_label(value: Any, *, allow_missing: bool) -> int:
    if value is None or str(value).strip() == "":
        if allow_missing:
            return -1
        raise ValueError("missing label")
    raw = str(value).strip()
    try:
        label_id = int(raw)
    except ValueError:
        key = raw.lower().replace(" ", "_")
        canonical = LABEL_ALIASES.get(key)
        if canonical is None:
            raise ValueError(f"unknown label: {value!r}")
        return LABEL_TO_ID[canonical]
    if 0 <= label_id < NUM_CLASSES:
        return label_id
    raise ValueError(f"label id out of range: {label_id}")


def _row_get(row: Mapping[str, Any], *names: str) -> Any:
    lower = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _split_matches(row_usage: str, split: str) -> bool:
    usage = row_usage.strip().lower()
    split = split.strip().lower()
    if not usage:
        return True
    groups = {
        "train": {"train", "training"},
        "val": {"val", "validation", "private", "privatetest"},
        "validation": {"val", "validation", "private", "privatetest"},
        "test": {"test", "public", "publictest"},
        "publictest": {"test", "public", "publictest"},
        "unlabeled": {"unlabeled", "pseudo"},
        "pseudo": {"unlabeled", "pseudo"},
    }
    return usage in groups.get(split, {split})


def _load_fer_pixels_strict(pixels: str) -> np.ndarray:
    text = str(pixels or "").strip()
    if not text:
        raise ValueError("empty FER pixels field")
    arr = np.fromstring(text, dtype=np.int16, sep=" ")
    if arr.size != FER_PIXEL_COUNT:
        raise ValueError(f"expected {FER_PIXEL_COUNT} pixel values, got {arr.size}")
    if arr.size and (arr.min() < 0 or arr.max() > 255):
        raise ValueError("pixel values must be within [0,255]")
    return arr.astype(np.uint8).reshape(FER_PIXEL_SIDE, FER_PIXEL_SIDE)


def _to_pil_rgb(img_arr: np.ndarray) -> Image.Image:
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    return Image.fromarray(img_arr.astype(np.uint8), mode="RGB")


def _resolve_csvs(csv_base: str | os.PathLike[str]) -> Tuple[str, str, str, Optional[str]]:
    base = Path(csv_base)
    if not base.is_dir():
        raise FileNotFoundError(f"csv_base not found: {base}")
    train_csv = base / "train.csv"
    val_csv = base / "val.csv"
    test_csv = base / "test.csv"
    unlabeled_csv = base / "unlabeled.csv"
    missing = [str(p) for p in (train_csv, val_csv, test_csv) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required CSV files: " + ", ".join(missing))
    return str(train_csv), str(val_csv), str(test_csv), str(unlabeled_csv) if unlabeled_csv.exists() else None


@dataclass(frozen=True)
class SampleRecord:
    label: int
    pixels: str
    path: str
    row_index: int
    source_csv: str

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "pixels": self.pixels,
            "path": self.path,
            "row_index": self.row_index,
            "source_csv": self.source_csv,
        }


class FER2013Hybrid(Dataset):
    """CSV-backed FER dataset compatible with existing train scripts.

    Supported row formats:
    - ``emotion,pixels`` for the current processed CSV pipeline.
    - ``emotion,path`` or ``label,path`` for future image-path training.
    - optional ``Usage`` column; if absent, the whole CSV is treated as the
      requested split.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: Optional[str],
        split: str,
        img_size: int = IMG_SIZE,
        two_views: bool = False,
        include_label: bool = True,
        *,
        strict: bool = True,
        validate_at_init: bool = False,
    ) -> None:
        self.csv_path = str(csv_path)
        self.img_root = None if img_root in (None, "", "None") else str(img_root)
        self.split = str(split).lower()
        self.img_size = int(img_size)
        self.two_views = bool(two_views)
        self.include_label = bool(include_label)
        self.strict = bool(strict)
        self.validate_at_init = bool(validate_at_init)

        path = Path(self.csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        self.samples: List[Dict[str, Any]] = []
        errors: List[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            for row_index, row in enumerate(reader, start=2):
                usage = str(_row_get(row, "Usage") or "")
                if not _split_matches(usage, self.split):
                    continue
                try:
                    label = _normalize_label(
                        _row_get(row, "label", "emotion", "mapped_emotion"),
                        allow_missing=not self.include_label,
                    )
                    pixels = str(_row_get(row, "pixels") or "").strip()
                    rel_path = str(_row_get(row, "path", "filepath", "image", "file") or "").strip()
                    if not pixels and not rel_path:
                        raise ValueError("row has neither pixels nor path")
                    if self.include_label and label < 0:
                        raise ValueError("labeled split row has no label")
                    record = SampleRecord(label, pixels, rel_path, row_index, str(path)).to_legacy_dict()
                    self.samples.append(record)
                except Exception as exc:
                    errors.append(f"row {row_index}: {exc}")
                    if len(errors) >= 10 and self.strict:
                        break

        if errors and self.strict:
            raise ValueError(f"Invalid rows in {path}: " + " | ".join(errors[:10]))
        if not self.samples:
            raise RuntimeError(f"Empty dataset after split filtering: csv={path}, split={self.split}")

        if self.validate_at_init:
            for idx in range(len(self.samples)):
                self._load_image(self.samples[idx])

        self.t_train = get_labeled_transforms("train", self.img_size)
        self.t_eval = get_labeled_transforms("eval", self.img_size)
        self.t_weak = get_weak_transforms(self.img_size)
        self.t_strong = get_strong_transforms(self.img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def label_counts(self) -> Dict[int, int]:
        return dict(Counter(int(s.get("label", -1)) for s in self.samples if int(s.get("label", -1)) >= 0))

    def _resolve_image_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if self.img_root and not path.is_absolute():
            path = Path(self.img_root) / path
        return path

    def _load_image(self, item: Mapping[str, Any]) -> Image.Image:
        rel_path = str(item.get("path") or "")
        if rel_path:
            path = self._resolve_image_path(rel_path)
            if not path.exists():
                raise FileNotFoundError(f"image path not found: {path}")
            with Image.open(path) as img:
                return img.convert("RGB")
        pixels = str(item.get("pixels") or "")
        return _to_pil_rgb(_load_fer_pixels_strict(pixels))

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = self._load_image(item)
        label = int(item.get("label", -1))

        if self.two_views:
            y = label if (self.include_label and label >= 0) else -1
            return self.t_weak(img), self.t_strong(img), torch.tensor(y, dtype=torch.long)
        if self.split == "train":
            return self.t_train(img), torch.tensor(label, dtype=torch.long)
        return self.t_eval(img), torch.tensor(label, dtype=torch.long)


def verify_paths(ds: FER2013Hybrid) -> List[str]:
    missing: List[str] = []
    for s in ds.samples:
        path = str(s.get("path") or "")
        if path and not ds._resolve_image_path(path).exists():
            missing.append(path)
    if missing:
        print(f"Missing {len(missing)} image files")
    return missing


def _apply_per_class_limit(ds: FER2013Hybrid, per_class: Optional[int], seed: int = 42) -> Subset:
    if per_class is None or int(per_class) <= 0:
        return Subset(ds, list(range(len(ds))))
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, item in enumerate(ds.samples):
        lab = int(item.get("label", -1))
        if lab >= 0:
            buckets[lab].append(idx)
    rng = random.Random(int(seed))
    picked: List[int] = []
    for lab in sorted(buckets):
        idxs = list(buckets[lab])
        rng.shuffle(idxs)
        picked.extend(idxs[: int(per_class)])
    return Subset(ds, sorted(picked))


def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    if int(num_workers) <= 0:
        persistent_workers = False
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if persistent_workers:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def get_dataloaders_hybrid(
    csv_base: Optional[str] = None,
    img_base: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    dynamic_sampling: bool = False,
    per_class: Optional[int] = None,
    include_unlabeled: bool = False,
    unlabeled_two_views: bool = True,
    prefetch_factor: int = 2,
    img_size: int = IMG_SIZE,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    unlabeled_csv: Optional[str] = None,
    seed: int = 42,
    strict: bool = True,
):
    """Return train, val, test and optional unlabeled loaders."""
    if csv_base:
        t_csv, v_csv, te_csv, u_csv = _resolve_csvs(csv_base)
        train_csv = train_csv or t_csv
        val_csv = val_csv or v_csv
        test_csv = test_csv or te_csv
        unlabeled_csv = unlabeled_csv or u_csv
    if not (train_csv and val_csv and test_csv):
        raise ValueError("Provide csv_base or explicit train_csv/val_csv/test_csv paths")

    train_set: Dataset = FER2013Hybrid(train_csv, img_base, "train", img_size=img_size, include_label=True, strict=strict)
    val_set = FER2013Hybrid(val_csv, img_base, "val", img_size=img_size, include_label=True, strict=strict)
    test_set = FER2013Hybrid(test_csv, img_base, "test", img_size=img_size, include_label=True, strict=strict)

    if dynamic_sampling and per_class is not None and int(per_class) > 0:
        train_set = _apply_per_class_limit(train_set, int(per_class), seed=seed)  # type: ignore[arg-type]

    train_loader = _make_loader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
    )
    val_loader = _make_loader(
        val_set,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = _make_loader(
        test_set,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    unlabeled_loader = None
    if include_unlabeled and unlabeled_csv and Path(unlabeled_csv).exists():
        u_set = FER2013Hybrid(
            unlabeled_csv,
            img_base,
            "unlabeled",
            img_size=img_size,
            two_views=unlabeled_two_views,
            include_label=False,
            strict=strict,
        )
        unlabeled_loader = _make_loader(
            u_set,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )
    return train_loader, val_loader, test_loader, unlabeled_loader


__all__ = [
    "IMG_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LABELS",
    "NUM_CLASSES",
    "FER2013Hybrid",
    "get_labeled_transforms",
    "get_weak_transforms",
    "get_strong_transforms",
    "get_dataloaders_hybrid",
    "verify_paths",
]
