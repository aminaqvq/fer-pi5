import os
import csv
import math
import time
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.utils import resample

from dataset import get_dataloaders_hybrid  # 从 dataset.py 导入数据加载函数
from model_mbv3 import get_model  # 从 model_mbv3.py 导入模型

# -------------------------
# 配置
# -------------------------
CFG: Dict[str, object] = dict(
    # 路径（按需修改）
    csv_base=r"D:\fer-pi5\data\csv",
    img_base=None,
    save_dir=r"D:\fer-pi5\checkpoints",
    best_ckpt=r"D:\fer-pi5\checkpoints\best_model_stage1.pth",
    log_csv=r"D:\fer-pi5\checkpoints\train_stage1_log.csv",

    # 设备 & 训练
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=200,          # 上限，配合 early stop，一般不会跑满
    batch_size=128,
    num_workers=4,
    lr=5e-4,
    lr_floor=1e-6,
    warmup_epochs=2,

    # 数据加载
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,

    # 动态采样：控制每类最多多少有标样本参与训练
    dynamic_sampling=True,
    per_class_limit=5000,

    # 类别不平衡处理
    beta=0.995,            # effective number
    label_smoothing=0.04,  # 类别平滑

    # 是否启用 mixup（可选，默认关闭）
    use_mixup=False,
    mixup_alpha=0.2,

    # 稳定性
    use_amp=True,          # 显存和算子支持的话建议开
    grad_clip=True,
    max_norm=1.0,

    # 验证与早停
    val_interval=1,
    early_stop_patience=20,

    # 随机种子
    seed=42,
)

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 工具函数
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch: int) -> float:
    """Cosine annealing + warmup"""
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))

def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(1) == target).float().mean().item())

def macro_f1(logits: torch.Tensor, target: torch.Tensor, C: int = 7) -> float:
    pred = logits.argmax(1)
    f1s = []
    for c in range(C):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s))

def _iter_labels_from_dataset(ds):
    """
    与你原 train.py 中逻辑一致：稳健获取所有标签，用于统计类别数量
    """
    from torch.utils.data import Subset

    if hasattr(ds, "samples"):
        for s in ds.samples:
            y = int(s.get("label", -1))
            if y >= 0:
                yield y
        return

    if isinstance(ds, Subset):
        base, idxs = ds.dataset, list(ds.indices)
        if hasattr(base, "samples"):
            for i in idxs:
                y = int(base.samples[i].get("label", -1))
                if y >= 0:
                    yield y
            return
        for i in idxs:
            _, y = base[i]
            y = int(y)
            if y >= 0:
                yield y
        return

    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, tuple) and len(item) >= 2:
            y = int(item[1])
            if y >= 0:
                yield y

# -------------------------
# 焦点损失（Focal Loss）
# -------------------------
def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """
    Focal loss formula: FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
    """
    p_t = F.softmax(logits, dim=1)
    p_t = p_t.gather(1, target.view(-1, 1))  # 获取正确类别的概率
    loss = -alpha * (1 - p_t)**gamma * torch.log(p_t)
    return loss.mean()

# -------------------------
# 一个 epoch 的训练
# -------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, CFG, criterion_sup, scaler=None):
    model.train()
    total_loss = total_acc = total_f1 = 0.0
    total_n = 0

    use_mixup = bool(CFG.get("use_mixup", False))
    alpha = float(CFG.get("mixup_alpha", 0.2))

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            lam = np.random.beta(alpha, alpha)
            idx = torch.randperm(xb.size(0), device=device)
            mixed = lam * xb + (1 - lam) * xb[idx]

            with autocast(device_type=AMP_DEVICE, enabled=bool(CFG.get("use_amp", False))):
                logits = model(mixed)
                loss = lam * criterion_sup(logits, yb) + (1 - lam) * criterion_sup(logits, yb[idx])
        else:
            with autocast(device_type=AMP_DEVICE, enabled=bool(CFG.get("use_amp", False))):
                logits = model(xb)
                loss = criterion_sup(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            if bool(CFG.get("grad_clip", False)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.get("max_norm", 1.0)))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if bool(CFG.get("grad_clip", False)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.get("max_norm", 1.0)))
            optimizer.step()

        with torch.no_grad():
            bs = yb.size(0)
            total_loss += float(loss.item()) * bs
            total_acc += accuracy(logits, yb) * bs
            total_f1 += macro_f1(logits, yb) * bs
            total_n += bs

    return total_loss / max(1, total_n), total_acc / max(1, total_n), total_f1 / max(1, total_n)

# -------------------------
# 验证
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, CFG):
    model.eval()
    total_loss = total_acc = total_f1 = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        bs = yb.size(0)
        total_loss += float(loss.item()) * bs
        total_acc += accuracy(logits, yb) * bs
        total_f1 += macro_f1(logits, yb) * bs
        total_n += bs

    return total_loss / max(1, total_n), total_acc / max(1, total_n), total_f1 / max(1, total_n)

# -------------------------
# 主流程
# -------------------------
def main():
    os.makedirs(str(CFG["save_dir"]), exist_ok=True)
    for k in ("log_csv", "best_ckpt"):
        d = os.path.dirname(str(CFG[k]))
        if d:
            os.makedirs(d, exist_ok=True)

    seed_all(int(CFG["seed"]))
    device = str(CFG["device"])
    print(f"Device: {device}", flush=True)

    # Data：这里只用有标签的 train/val/test，不加载 unlabeled
    train_loader, val_loader, test_loader, _ = get_dataloaders_hybrid(
        csv_base=str(CFG["csv_base"]),
        img_base=(None if CFG["img_base"] in (None, "None", "") else str(CFG["img_base"])),
        batch_size=int(CFG["batch_size"]),
        num_workers=int(CFG["num_workers"]),
        pin_memory=bool(CFG.get("pin_memory", True)),
        persistent_workers=bool(CFG.get("persistent_workers", True)),
        prefetch_factor=int(CFG.get("prefetch_factor", 2)),
        dynamic_sampling=bool(CFG["dynamic_sampling"]),
        per_class=int(CFG["per_class_limit"]),
        include_unlabeled=False,
        unlabeled_two_views=False,
    )
    print("Dataloaders ready: Train=%d, Val=%d, Test=%d" %
          (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)),
          flush=True)

    # Model
    model = get_model(
        variant="large",
        num_classes=7,
        pretrained=True,   # 使用 ImageNet 预训练
        device=device,
        verbose=True,
        compile_model=False,
    )

    # 类别权重计算（有效类别数）
    labels = list(_iter_labels_from_dataset(train_loader.dataset))
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=7).astype(np.float32)
    print("Class counts:", counts.tolist(), flush=True)

    beta = float(CFG["beta"])
    eff_num = (1.0 - beta) / (1.0 - np.power(beta, counts + 1e-8))
    cb = eff_num / eff_num.mean()  # 类别平衡权重
    class_w = torch.tensor(cb, dtype=torch.float32, device=device)
    print("Class weights (CB):", cb.tolist(), flush=True)

    def criterion_sup(logits, target):
        return focal_loss(logits, target)  # 使用焦点损失

    # 优化器
    optimizer = AdamW(model.parameters(), lr=float(CFG["lr"]), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=bool(CFG.get("use_amp", False)))

    # 日志表头
    if not os.path.exists(str(CFG["log_csv"])):
        with open(str(CFG["log_csv"]), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "lr",
                "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1"
            ])

    best_f1 = -1.0
    no_improve = 0
    total_epochs = int(CFG["epochs"])

    for epoch in range(total_epochs):
        lr = cosine_warmup_lr(
            float(CFG["lr"]), float(CFG["lr_floor"]),
            int(CFG["warmup_epochs"]), total_epochs, epoch
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, optimizer, train_loader, device, epoch, CFG, criterion_sup, scaler
        )
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device, CFG)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1}/{total_epochs} | lr {lr:.6f} | "
            f"Train L/A/F1 {tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
            f"Val L/A/F1 {va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f} | "
            f"{elapsed:.1f}s",
            flush=True
        )

        # 写日志
        with open(str(CFG["log_csv"]), "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1, lr,
                tr_loss, tr_acc, tr_f1,
                va_loss, va_acc, va_f1,
            ])

        # 选择 best（按 Val F1）
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), str(CFG["best_ckpt"]))
            print(f"  -> Saved best to {CFG['best_ckpt']} (Val F1={best_f1:.4f})", flush=True)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(CFG["early_stop_patience"]):
                print("Early stop: no improvement.", flush=True)
                break

    print("\nTraining finished.", flush=True)

if __name__ == "__main__":
    main()