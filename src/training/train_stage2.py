import os
import csv
import math
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from torch.amp import autocast
from torch.cuda.amp import GradScaler

from dataset import FER2013Hybrid, IMG_SIZE
from model_mbv3 import get_model


# -------------------------
# 配置（按需修改）
# -------------------------
CFG: Dict[str, object] = dict(
    # 数据路径
    train_csv=r"F:\fer-pi5\data\csv\train.csv",
    val_csv=r"F:\fer-pi5\data\csv\val.csv",
    test_csv=r"F:\fer-pi5\data\csv\test.csv",
    pseudo_csv=r"F:\fer-pi5\data\csv\pseudo_labeled.csv",  # 需要包含列：label,pixels,path,Usage,conf
    img_base=None,  # 若 path 是相对路径，这里填图片根目录

    # 初始化（强烈建议：加载 stage1 best）
    init_ckpt=r"F:\fer-pi5\checkpoints\best_model_stage1.pth",

    # 输出
    save_dir=r"F:\fer-pi5\checkpoints",
    best_ckpt=r"F:\fer-pi5\checkpoints\best_model_stage2.pth",
    log_csv=r"F:\fer-pi5\checkpoints\train_stage2_log.csv",

    # 训练
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=200,
    batch_size=128,
    num_workers=4,
    lr=5e-4,
    lr_floor=1e-6,
    warmup_epochs=2,
    weight_decay=1e-4,

    # 类别不平衡（CB loss）
    beta=0.995,
    label_smoothing=0.04,
    # 计算 class weight 时是否把 pseudo 也算进去（更推荐 False：只用真实数据统计更稳）
    cb_include_pseudo=False,

    # pseudo(conf) 策略
    pseudo_conf_min=0.0,         # 伪标签额外过滤阈值（pseudo 生成阶段已经过滤过的话可设 0）
    pseudo_conf_power=2.0,       # conf 幂次：>1 会“压低低置信度样本”的影响
    pseudo_loss_scale=1.0,       # 伪标签整体权重（常见 0.5~1.0）
    pseudo_rampup_epochs=5,      # 前 N 个 epoch 逐步引入 pseudo（更稳）

    # 稳定性
    use_amp=True,
    grad_clip=True,
    max_norm=1.0,

    # 早停
    val_interval=1,
    early_stop_patience=20,

    # 其它
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


def _make_loader(ds, batch_size, shuffle, cfg) -> DataLoader:
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=bool(shuffle),
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# -------------------------
# 读取 pseudo conf（按 Usage 过滤）
# -------------------------
def read_pseudo_confs(csv_path: str, usage_keep=("pseudo", "unlabeled", "u"), conf_min: float = 0.0) -> List[float]:
    confs: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]
        has_usage = ("usage" in fieldnames)
        for row in reader:
            usage = (row.get("Usage") or row.get("usage") or "").lower()
            if has_usage and (usage not in usage_keep):
                continue
            c = row.get("conf", row.get("Conf", None))
            try:
                c = float(c) if c is not None else 1.0
            except Exception:
                c = 1.0
            if c < float(conf_min):
                continue
            confs.append(float(c))
    return confs


# -------------------------
# Dataset wrapper：给每个样本附带 weight
# -------------------------
class WeightedDataset(Dataset):
    def __init__(self, base: Dataset, weights: Optional[List[float]] = None, default_w: float = 1.0):
        self.base = base
        self.default_w = float(default_w)
        if weights is None:
            self.weights = None
        else:
            assert len(weights) == len(base), f"weights({len(weights)}) != dataset({len(base)})"
            self.weights = [float(w) for w in weights]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.default_w if (self.weights is None) else self.weights[idx]
        return x, y, torch.tensor(w, dtype=torch.float32)


# -------------------------
# 训练 / 验证
# -------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, cfg, class_w: Optional[torch.Tensor], scaler: GradScaler):
    model.train()
    total_loss = total_acc = total_f1 = 0.0
    total_wsum = 0.0

    use_amp = bool(cfg.get("use_amp", False)) and str(device).startswith("cuda")

    # pseudo rampup
    ramp_epochs = int(cfg.get("pseudo_rampup_epochs", 0))
    ramp = 1.0 if ramp_epochs <= 0 else min(1.0, float(epoch + 1) / float(ramp_epochs))

    for xb, yb, wb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb = wb.to(device, non_blocking=True)

        # wb 里已经包含 pseudo_loss_scale/conf_power（在构造 pseudo weights 时做），这里只做 rampup
        wb = wb * ramp

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)

            # per-sample CE
            loss_vec = F.cross_entropy(
                logits, yb,
                weight=class_w,
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                reduction="none",
            )
            # 加权归一化（避免 batch 里权重变化导致 loss 尺度漂移）
            wsum = wb.sum().clamp_min(1e-6)
            loss = (loss_vec * wb).sum() / wsum

        scaler.scale(loss).backward()
        if bool(cfg.get("grad_clip", False)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("max_norm", 1.0)))
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            bs_w = float(wsum.item())
            total_loss += float(loss.item()) * bs_w
            total_acc += accuracy(logits, yb) * bs_w
            total_f1 += macro_f1(logits, yb) * bs_w
            total_wsum += bs_w

    denom = max(1e-6, total_wsum)
    return total_loss / denom, total_acc / denom, total_f1 / denom


@torch.no_grad()
def evaluate_simple(model, loader, device):
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


def compute_cb_weights(labels: List[int], num_classes: int, beta: float, device: str) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    eff = 1.0 - np.power(beta, counts)
    cb = (1.0 - beta) / np.maximum(eff, 1e-8)
    cb = cb / cb.mean()
    return torch.tensor(cb, dtype=torch.float32, device=device)


def extract_labels(ds: Dataset) -> List[int]:
    labs: List[int] = []
    for i in range(len(ds)):
        _, y = ds[i]
        y = int(y)
        if y >= 0:
            labs.append(y)
    return labs


def load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: str):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        print(f"[stage2] init_ckpt not found: {ckpt_path} (skip)")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[stage2] Loaded init_ckpt: {ckpt_path}")
    if missing:
        print(f"[stage2]  missing keys: {len(missing)}")
    if unexpected:
        print(f"[stage2]  unexpected keys: {len(unexpected)}")


def main():
    os.makedirs(str(CFG["save_dir"]), exist_ok=True)
    for k in ("log_csv", "best_ckpt"):
        d = os.path.dirname(str(CFG[k]))
        if d:
            os.makedirs(d, exist_ok=True)

    seed_all(int(CFG["seed"]))
    device = str(CFG["device"])
    print(f"Device: {device}", flush=True)

    # ----- datasets -----
    img_root = None if CFG["img_base"] in (None, "", "None") else str(CFG["img_base"])

    train_csv = os.path.abspath(str(CFG["train_csv"]))
    pseudo_csv = os.path.abspath(str(CFG["pseudo_csv"]))
    val_csv = os.path.abspath(str(CFG["val_csv"]))
    test_csv = os.path.abspath(str(CFG["test_csv"]))

    ds_real = FER2013Hybrid(train_csv, img_root, "train", img_size=int(IMG_SIZE), two_views=False, include_label=True)

    # pseudo：用 split="unlabeled" 过滤 Usage=pseudo，然后把 split 改成 train 用训练增强（你原脚本也这么做）
    ds_pseudo = FER2013Hybrid(pseudo_csv, img_root, "unlabeled", img_size=int(IMG_SIZE), two_views=False, include_label=True)
    ds_pseudo.split = "train"

    # 读 conf 并构造权重（conf -> conf^power -> *scale）
    conf_min = float(CFG.get("pseudo_conf_min", 0.0))
    conf_power = float(CFG.get("pseudo_conf_power", 1.0))
    pseudo_scale = float(CFG.get("pseudo_loss_scale", 1.0))

    pseudo_confs = read_pseudo_confs(pseudo_csv, conf_min=conf_min)
    if len(pseudo_confs) != len(ds_pseudo):
        raise RuntimeError(f"Pseudo conf length mismatch: confs={len(pseudo_confs)} ds_pseudo={len(ds_pseudo)}. "
                           f"检查 pseudo_csv 的 Usage 过滤/是否有空行。")

    pseudo_weights = [(max(0.0, min(1.0, c)) ** conf_power) * pseudo_scale for c in pseudo_confs]

    ds_real_w = WeightedDataset(ds_real, weights=None, default_w=1.0)
    ds_pseudo_w = WeightedDataset(ds_pseudo, weights=pseudo_weights, default_w=1.0)

    ds_train = ConcatDataset([ds_real_w, ds_pseudo_w])
    train_loader = _make_loader(ds_train, CFG["batch_size"], True, CFG)

    ds_val = FER2013Hybrid(val_csv, img_root, "val", img_size=int(IMG_SIZE), two_views=False, include_label=True)
    ds_test = FER2013Hybrid(test_csv, img_root, "test", img_size=int(IMG_SIZE), two_views=False, include_label=True)
    val_loader = _make_loader(ds_val, CFG["batch_size"], False, CFG)
    test_loader = _make_loader(ds_test, CFG["batch_size"], False, CFG)

    print(f"[stage2] Real train samples:   {len(ds_real)}")
    print(f"[stage2] Pseudo train samples: {len(ds_pseudo)}")
    print(f"[stage2] Val samples:          {len(ds_val)}")
    print(f"[stage2] Test samples:         {len(ds_test)}", flush=True)

    # ----- model -----
    model = get_model("large", num_classes=7, pretrained=True, device=device, verbose=True, compile_model=False)

    # init from stage1 best
    load_ckpt_into_model(model, str(CFG.get("init_ckpt", "")), device)

    # ----- class-balanced weights -----
    beta = float(CFG.get("beta", 0.999))
    include_pseudo = bool(CFG.get("cb_include_pseudo", False))
    if include_pseudo:
        # 用真实+伪标签统计（不一定更好，默认 False）
        real_labels = extract_labels(ds_real)
        pseudo_labels = extract_labels(ds_pseudo)
        labels = real_labels + pseudo_labels
        print(f"[stage2] CB labels: real({len(real_labels)}) + pseudo({len(pseudo_labels)})")
    else:
        labels = extract_labels(ds_real)
        print(f"[stage2] CB labels: real only ({len(labels)})")

    class_w = compute_cb_weights(labels, num_classes=7, beta=beta, device=device)
    print("[stage2] Class weights (CB):", class_w.detach().cpu().numpy().tolist(), flush=True)

    # ----- optim / amp -----
    optimizer = AdamW(model.parameters(), lr=float(CFG["lr"]), weight_decay=float(CFG["weight_decay"]))
    use_amp = bool(CFG.get("use_amp", False)) and str(device).startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    # ----- log header -----
    if not os.path.exists(str(CFG["log_csv"])):
        with open(str(CFG["log_csv"]), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "time", "epoch", "lr",
                "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1",
                "test_loss", "test_acc", "test_f1",
            ])

    best_f1 = -1.0
    no_improve = 0
    total_epochs = int(CFG["epochs"])

    for epoch in range(total_epochs):
        lr = cosine_warmup_lr(float(CFG["lr"]), float(CFG["lr_floor"]), int(CFG["warmup_epochs"]), total_epochs, epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, optimizer, train_loader, device, epoch, CFG, class_w, scaler)
        va_loss, va_acc, va_f1 = evaluate_simple(model, val_loader, device)
        te_loss, te_acc, te_f1 = evaluate_simple(model, test_loader, device)
        elapsed = time.time() - t0

        print(
            f"[stage2] Epoch {epoch+1}/{total_epochs} | lr {lr:.6f} | "
            f"Train L/A/F1 {tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
            f"Val   L/A/F1 {va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f} | "
            f"Test  L/A/F1 {te_loss:.4f}/{te_acc:.4f}/{te_f1:.4f} | "
            f"{elapsed:.1f}s",
            flush=True
        )

        with open(str(CFG["log_csv"]), "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                int(time.time()), epoch + 1, lr,
                tr_loss, tr_acc, tr_f1,
                va_loss, va_acc, va_f1,
                te_loss, te_acc, te_f1,
            ])

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), str(CFG["best_ckpt"]))
            print(f"[stage2]  -> Saved best to {CFG['best_ckpt']} (Val F1={best_f1:.4f})", flush=True)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(CFG["early_stop_patience"]):
                print("[stage2] Early stop: no improvement.", flush=True)
                break

    print("\nTraining finished.", flush=True)


if __name__ == "__main__":
    main()