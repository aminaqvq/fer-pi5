import os
import csv
from typing import Optional, List, Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import RandAugment

# =========================
# Global
# =========================
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================
# Transforms
# =========================
def get_labeled_transforms(split: str = "train", img_size: int = IMG_SIZE) -> transforms.Compose:
    """给有标签样本的增强：训练适度几何变化，评估仅做归一化"""
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
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_weak_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """无标签 weak：尽量稳定语义，用于产生伪标签"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_strong_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """无标签 strong：更强的几何/局部扰动以正则化"""
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


# =========================
# Dataset helpers
# =========================
def _load_fer_pixels(pixels: str) -> np.ndarray:
    """从 FER2013 CSV 的像素串还原 48x48 灰度图"""
    arr = np.fromstring(pixels, dtype=np.uint8, sep=' ')
    if arr.size != 48 * 48:
        # 容错：若像素数异常，尽量 reshape 成接近方阵
        side = int(np.sqrt(max(arr.size, 1)))
        side = max(8, side)
        padded = np.zeros(side * side, dtype=np.uint8)
        n = min(arr.size, side * side)
        padded[:n] = arr[:n]
        arr = padded.reshape(side, side)
    else:
        arr = arr.reshape(48, 48)
    return arr


def _to_pil(img_arr: np.ndarray) -> Image.Image:
    """灰度转 3 通道 PIL（与 ImageNet 预训练保持一致的接口）"""
    if img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    return Image.fromarray(img_arr)


def _discover_csvs(csv_base: str) -> Tuple[str, str, str, Optional[str]]:
    """在 csv_base 目录下自动探测 train/val/test/unlabeled 文件"""
    assert os.path.isdir(csv_base), f"csv_base not found: {csv_base}"
    files = [f for f in os.listdir(csv_base) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files in {csv_base}")

    def pick(preds):
        cand = [f for f in files if any(p in f.lower() for p in preds)]
        if not cand:
            return None
        # 优先精确名字
        for name in ("train.csv", "val.csv", "validation.csv", "test.csv", "publictest.csv", "privatetest.csv", "unlabeled.csv"):
            for f in cand:
                if f.lower() == name:
                    return os.path.join(csv_base, f)
        # 否则返回第一个匹配
        return os.path.join(csv_base, sorted(cand)[0])

    train_csv = pick(["train"])
    val_csv = pick(["val", "validation", "private"])
    test_csv = pick(["test", "public"])
    unlabeled_csv = pick(["unlabeled", "pseudo", "u"])

    if train_csv is None:
        raise FileNotFoundError("Cannot locate train CSV under csv_base.")
    if val_csv is None:
        raise FileNotFoundError("Cannot locate val/validation CSV under csv_base.")
    if test_csv is None:
        raise FileNotFoundError("Cannot locate test/public CSV under csv_base.")
    return train_csv, val_csv, test_csv, unlabeled_csv


def _apply_per_class_limit(ds: "FER2013Hybrid", per_class: int) -> Subset:
    """将数据集裁成“每类最多 per_class 个样本”，用于 dynamic_sampling"""
    if per_class is None or per_class <= 0:
        return Subset(ds, list(range(len(ds))))
    buckets: Dict[int, List[int]] = {}
    for idx, item in enumerate(ds.samples):
        lab = int(item.get("label", -1))
        if lab < 0:
            continue
        if lab not in buckets:
            buckets[lab] = []
        if len(buckets[lab]) < per_class:
            buckets[lab].append(idx)
    picked = []
    for lab, idxs in buckets.items():
        picked.extend(idxs)
    # 若存在无标签项，保持这些样本（仅 train split 会用到）
    if any(int(s.get("label", -1)) < 0 for s in ds.samples):
        picked.extend([i for i, s in enumerate(ds.samples) if int(s.get("label", -1)) < 0])
    return Subset(ds, picked)


# =========================
# Dataset
# =========================
class FER2013Hybrid(Dataset):
    """
    兼容两种 CSV 格式：
      A) 经典 FER2013：列含 "emotion", "pixels", "Usage"
      B) 扩展格式：列含 "label/emotion" 与 "path/filepath/image"（外部图片）
    参数：
      - split: 'train'|'val'|'publictest'|'test'|'unlabeled'
      - two_views: True 时返回 (weak, strong, label) 以用于 FixMatch
      - include_label: 无标签集设 False，则 label 永远为 -1
    """
    def __init__(
        self,
        csv_path: str,
        img_root: Optional[str],
        split: str,
        img_size: int = IMG_SIZE,
        two_views: bool = False,
        include_label: bool = True,
    ) -> None:
        self.csv_path = csv_path
        self.img_root = img_root
        self.split = split.lower()
        self.img_size = img_size
        self.two_views = two_views
        self.include_label = include_label

        # ---- 替换 FER2013Hybrid.__init__ 里“读取 CSV”这段 ----
        self.samples: List[Dict[str, Any]] = []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 如果没有 Usage 列，则整份 CSV 都视为当前 split
            fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]
            has_usage = ("usage" in fieldnames)

            for row in reader:
                usage = (row.get("Usage") or row.get("usage") or "").lower()

                take = True
                if has_usage:
                    is_train = usage in ("train", "training")
                    is_val = usage in ("val", "validation", "privatetest")
                    is_test = usage in ("publictest", "test")
                    is_u = usage in ("unlabeled", "pseudo", "u")

                    if self.split == "train" and not is_train: take = False
                    if self.split in ("val", "validation") and not is_val: take = False
                    if self.split in ("test", "publictest") and not is_test: take = False
                    if self.split == "unlabeled" and not is_u: take = False

                if not take:
                    continue

                item: Dict[str, Any] = {}
                label_str = row.get("label", row.get("emotion", None))
                if label_str is None or label_str == "":
                    item["label"] = -1
                else:
                    try:
                        item["label"] = int(label_str)
                    except Exception:
                        item["label"] = -1

                item["pixels"] = row.get("pixels", "")
                item["path"] = row.get("path") or row.get("filepath") or row.get("image") or ""
                self.samples.append(item)

        if len(self.samples) == 0:
            # 兜底：如果还是空，直接不按 usage 过滤再读一遍（防止特殊命名）
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    item: Dict[str, Any] = {}
                    label_str = row.get("label", row.get("emotion", None))
                    if label_str is None or label_str == "":
                        item["label"] = -1
                    else:
                        try:
                            item["label"] = int(label_str)
                        except Exception:
                            item["label"] = -1
                    item["pixels"] = row.get("pixels", "")
                    item["path"] = row.get("path") or row.get("filepath") or row.get("image") or ""
                    self.samples.append(item)

        if len(self.samples) == 0:
            raise RuntimeError(f"Empty dataset from: {csv_path}")

        # transforms
        self.t_train = get_labeled_transforms("train", img_size)
        self.t_eval = get_labeled_transforms("eval", img_size)
        self.t_weak = get_weak_transforms(img_size)
        self.t_strong = get_strong_transforms(img_size)

    def __len__(self) -> int:
        return len(self.samples)

    def verify_paths(ds):
        missing = [s["path"] for s in ds.samples if s["path"] and not os.path.exists(s["path"])]
        print(f"⚠️ Missing {len(missing)} files")

    def _load_image(self, item: Dict[str, Any]) -> Image.Image:
        # 优先外部路径，其次像素串
        if item["path"]:
            path = item["path"]
            if self.img_root and not os.path.isabs(path):
                path = os.path.join(self.img_root, path)
            img = Image.open(path).convert("RGB")
            return img
        if item["pixels"]:
            arr = _load_fer_pixels(item["pixels"])
            return _to_pil(arr)
        raise RuntimeError("Neither `path` nor `pixels` available in CSV row.")

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = self._load_image(item)
        label = int(item.get("label", -1))

        # 监督训练/验证/测试
        if self.split == "train" and not self.two_views:
            return self.t_train(img), torch.tensor(label, dtype=torch.long)

        if self.split in ("val", "validation", "test", "publictest") and not self.two_views:
            return self.t_eval(img), torch.tensor(label, dtype=torch.long)

        # 半监督两视图
        if self.two_views:
            v1 = self.t_weak(img)    # weak
            v2 = self.t_strong(img)  # strong
            y = torch.tensor(label if (self.include_label and label >= 0) else -1, dtype=torch.long)
            return v1, v2, y

        # fallback（理论上不会走到）
        return self.t_eval(img), torch.tensor(label, dtype=torch.long)


# =========================
# Dataloaders
# =========================
def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    drop_last: bool = False,
):
    # Windows 下 persistent_workers 只有在 num_workers>0 时才允许
    if num_workers <= 0:
        persistent_workers = False
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    # prefetch_factor 只有在多进程时有效
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if persistent_workers:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def get_dataloaders_hybrid(
    # --- 新用法：与 train.py / evaluate.py 一致 ---
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
    # --- 旧用法：直接给路径也支持（保持兼容） ---
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    unlabeled_csv: Optional[str] = None,
):
    """
    返回：(train_loader, val_loader, test_loader, unlabeled_loader or None)

    兼容两种入口：
    1) 新：get_dataloaders_hybrid(csv_base=..., dynamic_sampling=..., per_class=..., ...)
    2) 旧：get_dataloaders_hybrid(train_csv=..., val_csv=..., test_csv=..., unlabeled_csv=..., ...)
    """
    # --- 解析 CSV 路径 ---
    if csv_base:
        t_csv, v_csv, te_csv, u_csv = _discover_csvs(csv_base)
        train_csv = train_csv or t_csv
        val_csv = val_csv or v_csv
        test_csv = test_csv or te_csv
        unlabeled_csv = unlabeled_csv or u_csv
    else:
        if not (train_csv and val_csv and test_csv):
            raise ValueError("Either provide csv_base OR explicit train_csv/val_csv/test_csv paths.")

    # --- 构造 Dataset ---
    train_set = FER2013Hybrid(train_csv, img_base, "train", img_size=img_size, two_views=False, include_label=True)
    val_set = FER2013Hybrid(val_csv, img_base, "val", img_size=img_size, two_views=False, include_label=True)
    test_set = FER2013Hybrid(test_csv, img_base, "test", img_size=img_size, two_views=False, include_label=True)

    # --- 可选：按类上限抽样（与 dynamic_sampling 配合）---
    if dynamic_sampling and (per_class is not None) and per_class > 0:
        train_set = _apply_per_class_limit(train_set, per_class)

    # --- Dataloader ---
    train_loader = _make_loader(
        train_set, batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, drop_last=True
    )
    val_loader = _make_loader(
        val_set, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )
    test_loader = _make_loader(
        test_set, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor
    )

    unlabeled_loader = None
    if include_unlabeled and unlabeled_csv and os.path.exists(unlabeled_csv):
        u_set = FER2013Hybrid(
            unlabeled_csv, img_base, "unlabeled",
            img_size=img_size, two_views=unlabeled_two_views, include_label=False
        )
        unlabeled_loader = _make_loader(
            u_set, batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor, drop_last=True
        )

    return train_loader, val_loader, test_loader, unlabeled_loader