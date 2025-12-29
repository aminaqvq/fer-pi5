import os
import json
from typing import Dict, List

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
    # 基础路径（按需要修改）
    csv_base=r"D:\fer-pi5\data\csv",
    unlabeled_csv=r"D:\fer-pi5\data\csv\unlabeled.csv",
    img_base=None,

    # 教师模型 ckpt（初始为 stage1 的模型）
    teacher_ckpt=r"D:\fer-pi5\checkpoints\best_model_stage1.pth",

    # 伪标签输出（本脚本生成的二阶段伪标签）
    out_csv_name="pseudo_labeled_stage2.csv",
    out_stats_name="pseudo_stats_stage2.json",

    # 训练相关
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    img_size=IMG_SIZE,
    tta=True,
    num_classes=7,

    # 伪标签筛选规则
    min_conf=0.85,
    max_per_class=20000,

    # 其他
    seed=42,
    save_dir=r"D:\fer-pi5\data\csv",
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
    """
    包装一个 Dataset，使 __getitem__ 返回 (image, index)
    方便根据 index 回到 samples 列表取 pixels/path 等原始信息
    """
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        return x, idx


def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
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
# Stage1：使用指定 teacher ckpt 对 unlabeled 生成伪标签
# ============================================================
@torch.no_grad()
def generate_pseudo_stage1(cfg: Dict[str, object]) -> Dict[str, object]:
    device = str(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 42)))

    os.makedirs(str(cfg["save_dir"]), exist_ok=True)
    out_csv_path = os.path.join(str(cfg["save_dir"]), str(cfg["out_csv_name"]))
    out_stats_path = os.path.join(str(cfg["save_dir"]), str(cfg["out_stats_name"]))

    # --- 确定 unlabeled CSV ---
    unlabeled_csv = cfg.get("unlabeled_csv")
    if unlabeled_csv in (None, "", "None"):
        base = str(cfg["csv_base"])
        cand = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
        u_list = [f for f in cand if "unlabeled" in f.lower()]
        if not u_list:
            raise FileNotFoundError("未找到 unlabeled CSV，请在 CFG['unlabeled_csv'] 显式指定路径")
        unlabeled_csv = os.path.join(base, sorted(u_list)[0])
    unlabeled_csv = os.path.abspath(unlabeled_csv)

    print(f"[Stage1] Using unlabeled CSV: {unlabeled_csv}")
    # --- 构造 Dataset & DataLoader ---
    u_set = FER2013Hybrid(
        csv_path=unlabeled_csv,
        img_root=(None if cfg.get("img_base") in (None, "", "None") else str(cfg["img_base"])),
        split="unlabeled",
        img_size=int(cfg.get("img_size", IMG_SIZE)),
        two_views=False,
        include_label=False,
    )
    idx_ds = IndexedDataset(u_set)
    loader = _make_loader(
        idx_ds,
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )
    print(f"[Stage1] Unlabeled samples: {len(u_set)}")

    # --- 加载 Teacher 模型 ---
    num_classes = int(cfg.get("num_classes", 7))
    model = get_model(
        variant="large",
        num_classes=num_classes,
        pretrained=False,
        device=device,
        verbose=False,
        compile_model=False,
    )
    ckpt_path = str(cfg["teacher_ckpt"])
    print(f"[Stage1] Loading teacher checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    min_conf = float(cfg.get("min_conf", 0.85))
    max_per_class_cfg = cfg.get("max_per_class", None)
    if max_per_class_cfg is None or (isinstance(max_per_class_cfg, (int, float)) and max_per_class_cfg <= 0):
        max_per_class = None
    else:
        max_per_class = int(max_per_class_cfg)

    tta = bool(cfg.get("tta", True))
    num_classes = int(cfg.get("num_classes", 7))

    selected_indices: List[int] = []
    selected_labels: List[int] = []
    selected_confs: List[float] = []
    per_class_counts = [0 for _ in range(num_classes)]

    print("[Stage1] === Generating pseudo labels with tqdm ===")
    pbar = tqdm(loader, desc="Stage1 Generating Pseudo", ncols=80)

    for xb, idx in pbar:
        xb = xb.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

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
            if max_per_class is not None and per_class_counts[y] >= max_per_class:
                continue
            selected_indices.append(i)
            selected_labels.append(y)
            selected_confs.append(float(c))
            per_class_counts[y] += 1

    print(f"[Stage1] Selected {len(selected_indices)} samples (conf >= {min_conf})")
    print(f"[Stage1] Per-class counts: {per_class_counts}")

    # --- 导出 CSV ---
    out_rows: List[Dict[str, str]] = []
    for i, y, c in zip(selected_indices, selected_labels, selected_confs):
        s = u_set.samples[i]
        row = {
            "label": str(int(y)),
            "pixels": s.get("pixels", ""),
            "path": s.get("path", ""),
            "Usage": "pseudo",
            "conf": f"{c:.6f}",
        }
        out_rows.append(row)

    if out_rows:
        fieldnames = ["label", "pixels", "path", "Usage", "conf"]
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"[Stage1] Saved pseudo_labeled.csv → {out_csv_path}")
    else:
        print("[Stage1] No samples selected, CSV not written.")

    stats = dict(
        total_unlabeled=len(u_set),
        selected=len(selected_indices),
        per_class_counts=per_class_counts,
        min_conf=min_conf,
        max_per_class=max_per_class,
        teacher_ckpt=os.path.abspath(ckpt_path),
        unlabeled_csv=unlabeled_csv,
        out_csv=os.path.abspath(out_csv_path),
    )
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[Stage1] Stats saved → {out_stats_path}")

    return stats


# ============================================================
# Stage2：用 pseudo_labeled.csv 微调 teacher，然后用新模型再生成伪标签
# ============================================================
def fine_tune_teacher_and_regenerate(cfg: Dict[str, object]):
    device = str(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 42)))

    # -------- 加载第一阶段伪标签数据（原始 pseudo_labeled.csv）--------
    pseudo_csv = os.path.join(str(cfg["save_dir"]), "pseudo_labeled.csv")
    print(f"\n[Stage2] Loading pseudo-labeled data from {pseudo_csv}")
    if not os.path.exists(pseudo_csv):
        raise FileNotFoundError(f"找不到 {pseudo_csv}，请先运行原来的 generate_pseudo.py 生成 pseudo_labeled.csv")

    pseudo_set = FER2013Hybrid(
        csv_path=pseudo_csv,
        img_root=(None if cfg.get("img_base") in (None, "", "None") else str(cfg["img_base"])),
        split="train",
        img_size=int(cfg.get("img_size", IMG_SIZE)),
        two_views=False,
        include_label=True,
    )
    loader = _make_loader(
        pseudo_set,
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
    )
    print(f"[Stage2] Loaded {len(pseudo_set)} pseudo-labeled samples")

    # -------- 加载教师模型 --------
    print("\n[Stage2] Loading teacher model...")
    num_classes = int(cfg.get("num_classes", 7))
    model = get_model(
        variant="large",
        num_classes=num_classes,
        pretrained=False,
        device=device,
        verbose=False,
        compile_model=False,
    )
    ckpt_path = str(cfg["teacher_ckpt"])
    print(f"[Stage2] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)

    # 启用梯度、设为 train 模式
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # -------- 微调 --------
    num_epochs = 5
    print("\n[Stage2] ===== Fine-tuning Teacher Model =====")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_n = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=80)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val * xb.size(0)
            total_n += xb.size(0)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = total_loss / max(1, total_n)
        print(f"[Stage2] Epoch {epoch+1} Finished | Avg Loss = {avg_loss:.5f}")

    # -------- 保存新教师模型 --------
    old_ckpt = str(cfg["teacher_ckpt"])
    if "stage1" in old_ckpt:
        new_ckpt = old_ckpt.replace("stage1", "stage2")
    else:
        base, ext = os.path.splitext(old_ckpt)
        new_ckpt = base + "_stage2" + ext
    torch.save(model.state_dict(), new_ckpt)
    print(f"\n[Stage2] Saved new teacher model → {new_ckpt}")

    # -------- 用新教师模型重新生成伪标签（写入 pseudo_labeled_stage2.csv）--------
    print("\n[Stage2] Regenerating pseudo labels using fine-tuned teacher ...\n")
    cfg2 = dict(cfg)  # 复制一份 cfg 避免污染
    cfg2["teacher_ckpt"] = new_ckpt
    # out_csv_name/out_stats_name 已经在 CFG 里设置为 stage2 专用文件名
    generate_pseudo_stage1(cfg2)


# ============================================================
# main
# ============================================================
def main():
    # 第一步：用当前 teacher_ckpt（stage1 模型）在 unlabeled 上生成一份二阶段伪标签
    #         （你已经跑通过这一部分，保留也没问题，可以视作对照组）
    generate_pseudo_stage1(CFG)

    # 第二步：用首轮伪标签微调教师模型，然后用新模型重新生成伪标签到 pseudo_labeled_stage2.csv
    fine_tune_teacher_and_regenerate(CFG)


if __name__ == "__main__":
    main()