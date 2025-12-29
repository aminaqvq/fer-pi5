"""
Stage 1：FER 监督训练（用于产出一个“老师模型”）
- 只使用 train.csv（有标签部分），不使用 unlabeled.csv
- 利用：
    * 动态采样（per_class_limit）
    * effective number 类别权重
    * 轻量的数据增强（沿用 dataset.py）
    * Cosine + warmup 学习率
    * 可选 AMP 和梯度裁剪
- 训练完成后，会保存 best_model_stage1.pth
"""

import os
import csv
import math
import time
import random
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Subset

from dataset import get_dataloaders_hybrid  # 直接复用你现有的 dataloader 逻辑
from model_mbv3 import get_model            # 复用 MobileNetV3 定义

# -------------------------
# 配置
# -------------------------
CFG: Dict[str, object] = dict(
    # 路径（按需改成你自己的）
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
    label_smoothing=0.04,  # 比原来 0.08 略小一点

    # 是否启用 mixup（表情任务先关掉）
    use_mixup=False,
    mixup_alpha=0.2,

    # 稳定性
    use_amp=True,          # 显存和算子支持的话建议开
    grad_clip=True,
    max_norm=1.0,

    # 验证与早停
    val_interval=1,
    early_stop_patience=20,

    seed=42,
)

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 小工具
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(base_lr: float, floor: float,
                     warmup_epochs: int, total_epochs: int, epoch: int) -> float:
    """线性 warmup + cosine 衰减"""
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


def _iter_labels_from_dataset(ds) -> Iterable[int]:
    """
    与你原 train.py 中逻辑一致：稳健获取所有标签，用于统计类别数量
    """
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

            with torch.amp.autocast(AMP_DEVICE, enabled=bool(CFG.get("use_amp", False))):
                logits = model(mixed)
                loss = lam * criterion_sup(logits, yb) + (1 - lam) * criterion_sup(logits, yb[idx])
        else:
            with torch.amp.autocast(AMP_DEVICE, enabled=bool(CFG.get("use_amp", False))):
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

        # 这里不再使用 autocast，直接前向即可
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
        pretrained=True,
        device=device,
        verbose=True,
        compile_model=False,
    )

    # 统计类别数量，计算 effective number 权重
    labels = list(_iter_labels_from_dataset(train_loader.dataset))
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=7).astype(np.float32)
    print("Class counts:", counts.tolist(), flush=True)

    beta = float(CFG["beta"])
    eff_num = (1.0 - beta) / (1.0 - np.power(beta, counts + 1e-8))
    cb = eff_num / eff_num.mean()
    class_w = torch.tensor(cb, dtype=torch.float32, device=device)
    print("Class weights (CB):", cb.tolist(), flush=True)

    def criterion_sup(logits, target):
        return F.cross_entropy(
            logits, target,
            weight=class_w,
            label_smoothing=float(CFG["label_smoothing"])
        )

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

    # 结束时可以用 evaluate.py 再跑一次详细评估（生成混淆矩阵 PNG / JSON）
    try:
        from evaluate import run_evaluation
        _ = run_evaluation(
            dict(
                device=device,
                csv_base=str(CFG["csv_base"]),
                img_base=CFG["img_base"],
                save_dir=str(CFG["save_dir"]),
                best_ckpt=str(CFG["best_ckpt"]),
                batch_size=int(CFG["batch_size"]),
                num_workers=4,
                pin_memory=True,
                persistent_workers=False,
                per_class_limit=5000,
                model_variant="large",
            ),
            dict(split="both", tta=True, ckpt=str(CFG["best_ckpt"]))
        )
        print("[final-eval] confusion matrix & per-class metrics written.", flush=True)
    except Exception as e:
        print(f"[final-eval] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()