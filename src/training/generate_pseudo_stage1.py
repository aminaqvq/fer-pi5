import os
import math
import json
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import FER2013Hybrid, IMG_SIZE
from model_mbv3 import get_model


# ============================================================
# 配置
# ============================================================
CFG: Dict[str, object] = dict(
    csv_base=r"D:\fer-pi5\data\csv",
    unlabeled_csv=r"D:\fer-pi5\data\csv\unlabeled.csv",
    img_base=None,
    teacher_ckpt=r"D:\fer-pi5\checkpoints\best_model_stage1.pth",

    save_dir=r"D:\fer-pi5\data\csv",
    out_csv_name="pseudo_labeled.csv",
    out_stats_name="pseudo_stats.json",

    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    img_size=IMG_SIZE,
    tta=True,
    num_classes=7,

    min_conf=0.85,
    max_per_class=20000,
    seed=42,
)


# ============================================================
# 工具函数
# ============================================================
def seed_all(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    """包装一个 Dataset，使 __getitem__ 同时返回 (data, idx)"""
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        return x, idx


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool,
                 num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# ============================================================
# 第一阶段：伪标签生成（带 tqdm）
# ============================================================
@torch.no_grad()
def generate_pseudo_first_stage(cfg: Dict[str, object]):
    """
    第一阶段伪标签生成：使用教师模型推理 unlabeled 数据并生成 pseudo_labeled.csv
    """
    device = str(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 42)))

    os.makedirs(str(cfg["save_dir"]), exist_ok=True)

    out_csv_path = os.path.join(cfg["save_dir"], cfg["out_csv_name"])
    out_stats_path = os.path.join(cfg["save_dir"], cfg["out_stats_name"])

    # ----------------------------
    # 找 unlabeled.csv
    # ----------------------------
    unlabeled_csv = cfg.get("unlabeled_csv")
    if unlabeled_csv in (None, "", "None"):
        base = str(cfg["csv_base"])
        cand = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
        u_list = [f for f in cand if "unlabeled" in f.lower()]
        if not u_list:
            raise FileNotFoundError("❌ 未找到 unlabeled CSV！")
        unlabeled_csv = os.path.join(base, sorted(u_list)[0])

    unlabeled_csv = os.path.abspath(unlabeled_csv)
    print(f"[Stage1] Using unlabeled CSV: {unlabeled_csv}")

    # ----------------------------
    # DataLoader
    # ----------------------------
    u_set = FER2013Hybrid(
        csv_path=unlabeled_csv,
        img_root=(None if cfg["img_base"] in (None, "", "None") else cfg["img_base"]),
        split="unlabeled",
        img_size=cfg["img_size"],
        two_views=False,
        include_label=False,
    )

    loader = _make_loader(
        IndexedDataset(u_set),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )

    print(f"[Stage1] Unlabeled samples: {len(u_set)}")

    # ----------------------------
    # 加载 Teacher 模型
    # ----------------------------
    model = get_model(
        variant="large",
        num_classes=cfg["num_classes"],
        pretrained=False,
        device=device,
        verbose=True,
        compile_model=False,
    )

    ckpt = cfg["teacher_ckpt"]
    print(f"[Stage1] Loading teacher checkpoint: {ckpt}")

    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device).eval()

    # ----------------------------
    # 伪标签生成
    # ----------------------------
    min_conf = cfg["min_conf"]
    max_per_class = cfg["max_per_class"]
    num_classes = cfg["num_classes"]

    per_class_counts = [0] * num_classes
    selected_indices, selected_labels, selected_confs = [], [], []

    tta = cfg["tta"]

    print("[Stage1] === Generating pseudo labels with tqdm ===")

    for xb, idx in tqdm(loader, desc="Stage1 Generating Pseudo", ncols=100):
        xb = xb.to(device, non_blocking=True)

        # TTA 推理
        if tta:
            logits = model(xb)
            logits_flip = model(torch.flip(xb, dims=[-1]))
            logits = 0.5 * (logits + logits_flip)
        else:
            logits = model(xb)

        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

        for c, y, i in zip(conf.tolist(), pred.tolist(), idx.tolist()):
            if c < min_conf:
                continue
            if max_per_class and per_class_counts[y] >= max_per_class:
                continue

            selected_indices.append(i)
            selected_labels.append(y)
            selected_confs.append(float(c))
            per_class_counts[y] += 1

    print(f"[Stage1] Selected {len(selected_indices)} samples (conf >= {min_conf})")
    print("[Stage1] Per-class counts:", per_class_counts)

    # ----------------------------
    # 写入 CSV
    # ----------------------------
    import csv
    rows = []
    for i, y, c in zip(selected_indices, selected_labels, selected_confs):
        s = u_set.samples[i]
        rows.append({
            "label": y,
            "pixels": s.get("pixels", ""),
            "path": s.get("path", ""),
            "Usage": "pseudo",
            "conf": f"{c:.6f}",
        })

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "pixels", "path", "Usage", "conf"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Stage1] Saved pseudo_labeled.csv → {out_csv_path}")

    # ----------------------------
    # 保存统计信息
    # ----------------------------
    stats = dict(
        total_unlabeled=len(u_set),
        selected=len(selected_indices),
        per_class_counts=per_class_counts,
        min_conf=min_conf,
        max_per_class=max_per_class,
        unlabeled_csv=unlabeled_csv,
        out_csv=out_csv_path,
    )

    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[Stage1] Stats saved → {out_stats_path}")
    return stats


# ============================================================
# main
# ============================================================
def main():
    generate_pseudo_first_stage(CFG)


if __name__ == "__main__":
    main()