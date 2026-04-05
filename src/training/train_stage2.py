#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage2传统.py

第二阶段训练：真实标注数据 + 伪标签数据联合训练。
采用置信度加权策略，支持渐进式引入伪标签（ramp-up）。
与train_stage1.py保持完全统一的代码风格和配置结构。

核心创新：
    1. 真实样本与伪标签样本的差异化权重（真实=1.0，伪标签=conf^power）
    2. Ramp-up策略：前N个epoch逐步增加伪标签贡献，稳定训练初期
    3. 类别均衡：沿用CB loss，仅基于真实样本统计（更稳定）
"""

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

from dataset import FER2013Hybrid, IMG_SIZE, get_labeled_transforms
from model_mbv3 import get_model


# ============================================================
# 配置（与Stage1完全统一结构）
# ============================================================
CFG: Dict[str, object] = dict(
    # 数据路径
    train_csv=r"F:\fer-pi5\data\csv\train.csv",
    val_csv=r"F:\fer-pi5\data\csv\val.csv",
    test_csv=r"F:\fer-pi5\data\csv\test.csv",
    pseudo_csv=r"F:\fer-pi5\data\csv\pseudo_labeled.csv",  # Stage1生成的伪标签
    
    # 图像根目录（若path为相对路径）
    img_base=None,
    
    # 模型初始化（强烈建议加载Stage1最优模型）
    init_ckpt=r"F:\fer-pi5\checkpoints\best_model_stage1.pth",
    
    # 输出路径
    save_dir=r"F:\fer-pi5\checkpoints",
    best_ckpt=r"F:\fer-pi5\checkpoints\best_model_stage2.pth",
    log_csv=r"F:\fer-pi5\checkpoints\train_stage2_log.csv",
    
    # 设备与训练
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=200,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    
    # 优化器
    lr=5e-4,
    lr_floor=1e-6,
    warmup_epochs=2,
    weight_decay=1e-4,
    
    # 类别均衡（CB loss）
    beta=0.995,                # effective number衰减系数
    label_smoothing=0.04,      # 标签平滑
    # 计算class weight时是否包含伪标签（推荐False，更稳定）
    cb_include_pseudo=False,
    
    # 伪标签加权策略（Stage2核心创新）
    pseudo_conf_min=0.0,       # 伪标签额外过滤阈值（生成时已过滤则设0）
    pseudo_conf_power=2.0,       # 置信度幂次：>1强烈压低低置信度样本
    pseudo_loss_scale=1.0,       # 伪标签整体权重系数
    pseudo_rampup_epochs=5,      # ramp-up周期：前N个epoch渐进引入伪标签
    
    # 稳定性
    use_amp=True,              # 自动混合精度
    grad_clip=True,            # 梯度裁剪
    max_norm=1.0,              # 梯度裁剪阈值
    
    # 早停
    val_interval=1,            # 每N个epoch验证一次
    early_stop_patience=20,    # 验证F1不提升的容忍轮数
    
    # 可复现性
    seed=42,
)

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 工具函数（与Stage1完全统一）
# ============================================================
def seed_all(seed: int = 42):
    """设置全局随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(
    base_lr: float, 
    floor: float, 
    warmup_epochs: int, 
    total_epochs: int, 
    epoch: int
) -> float:
    """
    线性warmup + cosine衰减学习率调度
    """
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """计算批次准确率"""
    return float((logits.argmax(1) == target).float().mean().item())


def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> float:
    """计算宏平均F1（类别不均衡时更可靠）"""
    pred = logits.argmax(1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s))


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, cfg: Dict) -> DataLoader:
    """
    构造DataLoader，参数与Stage1完全统一
    """
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=bool(shuffle),    # 训练时丢弃不完整批次（BN稳定）
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
    return DataLoader(ds, **kwargs)


# ============================================================
# 伪标签置信度读取
# ============================================================
def read_pseudo_confs(
    csv_path: str, 
    usage_keep: Tuple[str, ...] = ("pseudo", "unlabeled", "u"),
    conf_min: float = 0.0
) -> List[float]:
    """
    从伪标签CSV读取置信度列表，按Usage过滤
    
    返回:
        confs: 与CSV行数一一对应的置信度列表（非目标Usage为0，会被过滤）
    """
    confs: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [fn.lower() for fn in (reader.fieldnames or [])]
        has_usage = "usage" in fieldnames
        
        for row in reader:
            usage = (row.get("Usage") or row.get("usage") or "").lower()
            if has_usage and (usage not in usage_keep):
                # 非目标Usage，标记为-1（后续过滤）
                confs.append(-1.0)
                continue
            
            c = row.get("conf", row.get("Conf", None))
            try:
                c = float(c) if c is not None else 1.0
            except Exception:
                c = 1.0
            
            if c < float(conf_min):
                confs.append(-1.0)  # 低于阈值，标记为无效
            else:
                confs.append(float(c))
    
    return confs


# ============================================================
# 带权重的数据集包装（Stage2核心：区分真实/伪标签权重）
# ============================================================
class WeightedDataset(Dataset):
    """
    为每个样本附加权重，支持真实样本与伪标签样本的差异化处理
    
    权重设计:
        - 真实样本: weight = 1.0（基准）
        - 伪标签样本: weight = (conf^power) * scale * ramp(t)
    """
    def __init__(
        self, 
        base: Dataset, 
        weights: Optional[List[float]] = None,
        is_pseudo_flags: Optional[List[bool]] = None,
        default_w: float = 1.0
    ):
        self.base = base
        self.default_w = float(default_w)
        self.weights = weights
        self.is_pseudo_flags = is_pseudo_flags  # 标记哪些是伪标签样本
        
        # 验证长度一致性
        if self.weights is not None:
            assert len(self.weights) == len(base), \
                f"weights长度({len(self.weights)})与数据集({len(base)})不匹配"
        if self.is_pseudo_flags is not None:
            assert len(self.is_pseudo_flags) == len(base), \
                f"is_pseudo_flags长度({len(self.is_pseudo_flags)})与数据集不匹配"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.default_w if (self.weights is None) else self.weights[idx]
        is_pseudo = False if (self.is_pseudo_flags is None) else self.is_pseudo_flags[idx]
        # 返回5元组：(图像, 标签, 权重, 是否伪标签, 索引)
        return x, y, torch.tensor(w, dtype=torch.float32), is_pseudo, idx


# ============================================================
# 类别均衡权重计算（CB loss）
# ============================================================
def compute_cb_weights(
    labels: List[int], 
    num_classes: int, 
    beta: float, 
    device: str
) -> torch.Tensor:
    """
    计算Class-Balanced权重: w_c ∝ (1-β)/(1-β^n_c)
    """
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    # 防止除零
    counts = np.maximum(counts, 1.0)
    
    # effective number
    eff_num = 1.0 - np.power(beta, counts)
    cb = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    # 归一化
    cb = cb / cb.mean()
    
    return torch.tensor(cb, dtype=torch.float32, device=device)


def extract_labels(ds: Dataset) -> List[int]:
    """从数据集提取所有标签（用于CB权重统计）"""
    labs: List[int] = []
    for i in range(len(ds)):
        _, y = ds[i]
        y = int(y)
        if y >= 0:
            labs.append(y)
    return labs


# ============================================================
# 模型加载
# ============================================================
def load_ckpt_into_model(model: torch.nn.Module, ckpt_path: str, device: str):
    """
    安全加载检查点，兼容多种格式（state_dict包装、多卡module.前缀）
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Stage2] 警告: 检查点不存在或路径为空: {ckpt_path}")
        return
    
    state = torch.load(ckpt_path, map_location="cpu")
    
    # 解包state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    # 移除module.前缀（多卡训练遗留）
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            new_state[nk] = v
        state = new_state
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Stage2] 加载检查点: {ckpt_path}")
    if missing:
        print(f"[Stage2]   缺失键: {len(missing)}个")
    if unexpected:
        print(f"[Stage2]   意外键: {len(unexpected)}个")


# ============================================================
# 训练与验证（核心：修复后的ramp-up逻辑）
# ============================================================
def train_one_epoch(
    model: torch.nn.Module,
    optimizer: AdamW,
    loader: DataLoader,
    device: str,
    epoch: int,
    cfg: Dict,
    class_w: Optional[torch.Tensor],
    scaler: GradScaler
) -> Tuple[float, float, float]:
    """
    训练一个epoch，支持真实/伪标签差异化权重和ramp-up
    
    修复后的ramp-up逻辑:
        - ramp只作用于伪标签样本的权重
        - 真实样本权重保持1.0不变
        - 这样ramp-up才能真正控制伪标签的渐进引入
    """
    model.train()
    total_loss = total_acc = total_f1 = 0.0
    total_wsum = 0.0  # 加权样本数（用于归一化）
    
    use_amp = bool(cfg.get("use_amp", False)) and str(device).startswith("cuda")
    
    # ramp-up系数：前pseudo_rampup_epochs个epoch从0线性增长到1
    ramp_epochs = int(cfg.get("pseudo_rampup_epochs", 0))
    if ramp_epochs <= 0:
        ramp = 1.0
    else:
        ramp = min(1.0, (epoch + 1) / ramp_epochs)
    
    for batch in loader:
        # 解包5元组: (图像, 标签, 权重, 是否伪标签, 索引)
        xb, yb, wb_base, is_pseudo_flags, _ = batch
        
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        wb_base = wb_base.to(device, non_blocking=True)
        
        # ==================== 修复后的ramp-up逻辑 ====================
        # 创建布尔掩码：区分真实样本和伪标签样本
        is_pseudo_mask = torch.tensor(is_pseudo_flags, dtype=torch.bool, device=device)
        
        # 真实样本：权重保持wb_base（即1.0）
        # 伪标签样本：权重乘以ramp系数
        wb = wb_base.clone()
        wb[is_pseudo_mask] = wb_base[is_pseudo_mask] * ramp
        
        # 打印调试信息（仅第一个batch的前3个epoch）
        if epoch < 3 and total_wsum == 0:
            real_count = (~is_pseudo_mask).sum().item()
            pseudo_count = is_pseudo_mask.sum().item()
            print(f"  [Debug Epoch {epoch+1}] real={real_count}, pseudo={pseudo_count}, "
                  f"ramp={ramp:.3f}, pseudo_weight_mean={wb[is_pseudo_mask].mean().item():.3f}")
        # ============================================================
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)
            
            # 逐样本计算CE（reduction="none"以便加权）
            loss_vec = F.cross_entropy(
                logits, yb,
                weight=class_w,
                label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                reduction="none",
            )
            
            # 加权归一化：避免batch内权重分布变化导致loss尺度漂移
            wsum = wb.sum().clamp_min(1e-6)
            loss = (loss_vec * wb).sum() / wsum
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if bool(cfg.get("grad_clip", False)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=float(cfg.get("max_norm", 1.0))
            )
        
        scaler.step(optimizer)
        scaler.update()
        
        # 统计（使用实际权重）
        with torch.no_grad():
            bs_w = float(wsum.item())
            total_loss += float(loss.item()) * bs_w
            total_acc += accuracy(logits, yb) * bs_w
            total_f1 += macro_f1(logits, yb) * bs_w
            total_wsum += bs_w
    
    # 归一化返回
    denom = max(1e-6, total_wsum)
    return total_loss / denom, total_acc / denom, total_f1 / denom


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str
) -> Tuple[float, float, float]:
    """
    验证/测试评估（不使用加权，标准评估）
    """
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


# ============================================================
# 主流程
# ============================================================
def main():
    # ------------------- 初始化 -------------------
    os.makedirs(str(CFG["save_dir"]), exist_ok=True)
    for k in ("log_csv", "best_ckpt"):
        d = os.path.dirname(str(CFG[k]))
        if d:
            os.makedirs(d, exist_ok=True)
    
    seed_all(int(CFG["seed"]))
    device = str(CFG["device"])
    print(f"[Stage2] 设备: {device}")
    
    # ------------------- 数据集构造 -------------------
    img_root = None if CFG["img_base"] in (None, "", "None") else str(CFG["img_base"])
    
    train_csv = os.path.abspath(str(CFG["train_csv"]))
    pseudo_csv = os.path.abspath(str(CFG["pseudo_csv"]))
    val_csv = os.path.abspath(str(CFG["val_csv"]))
    test_csv = os.path.abspath(str(CFG["test_csv"]))
    
    # 真实标注数据
    ds_real = FER2013Hybrid(
        train_csv, img_root, "train",
        img_size=int(IMG_SIZE),
        two_views=False,
        include_label=True
    )
    
    # 伪标签数据：先用"unlabeled"过滤Usage=pseudo，再改为train用训练增强
    ds_pseudo = FER2013Hybrid(
        pseudo_csv, img_root, "unlabeled",
        img_size=int(IMG_SIZE),
        two_views=False,
        include_label=True
    )
    ds_pseudo.split = "train"  # 关键：切换为train模式以应用训练增强
    
    # ------------------- 伪标签权重计算 -------------------
    conf_min = float(CFG.get("pseudo_conf_min", 0.0))
    conf_power = float(CFG.get("pseudo_conf_power", 1.0))
    pseudo_scale = float(CFG.get("pseudo_loss_scale", 1.0))
    
    print(f"[Stage2] 读取伪标签置信度: {pseudo_csv}")
    pseudo_confs = read_pseudo_confs(pseudo_csv, conf_min=conf_min)
    
    # 过滤无效项（conf=-1表示被过滤掉）
    valid_indices = [i for i, c in enumerate(pseudo_confs) if c >= 0]
    valid_confs = [pseudo_confs[i] for i in valid_indices]
    
    print(f"[Stage2] 伪标签总数: {len(pseudo_confs)}, 有效: {len(valid_confs)} "
          f"(过滤率: {100*(1-len(valid_confs)/max(len(pseudo_confs),1)):.1f}%)")
    
    # 构建权重列表和伪标签标记
    # 真实样本：weight=1.0, is_pseudo=False
    # 伪标签样本：weight=(conf^power)*scale, is_pseudo=True
    real_weights = [1.0] * len(ds_real)
    real_is_pseudo = [False] * len(ds_real)
    
    pseudo_weights = [
        (max(0.0, min(1.0, c)) ** conf_power) * pseudo_scale 
        for c in valid_confs
    ]
    pseudo_is_pseudo = [True] * len(valid_confs)
    
    # 创建加权数据集（使用Subset只保留有效的伪标签）
    from torch.utils.data import Subset
    ds_real_w = WeightedDataset(
        ds_real, 
        weights=real_weights, 
        is_pseudo_flags=real_is_pseudo
    )
    ds_pseudo_w = WeightedDataset(
        Subset(ds_pseudo, valid_indices),
        weights=pseudo_weights,
        is_pseudo_flags=pseudo_is_pseudo
    )
    
    # 合并训练集
    ds_train = ConcatDataset([ds_real_w, ds_pseudo_w])
    train_loader = _make_loader(ds_train, CFG["batch_size"], True, CFG)
    
    # 验证/测试集（标准Dataset，无加权）
    ds_val = FER2013Hybrid(val_csv, img_root, "val", img_size=int(IMG_SIZE))
    ds_test = FER2013Hybrid(test_csv, img_root, "test", img_size=int(IMG_SIZE))
    val_loader = _make_loader(ds_val, CFG["batch_size"], False, CFG)
    test_loader = _make_loader(ds_test, CFG["batch_size"], False, CFG)
    
    print(f"[Stage2] 训练样本: 真实={len(ds_real)}, 伪标签={len(valid_confs)}, 总计={len(ds_train)}")
    print(f"[Stage2] 验证样本: {len(ds_val)}, 测试样本: {len(ds_test)}")
    
    # ------------------- 模型 -------------------
    model = get_model(
        "large",
        num_classes=7,
        pretrained=True,  # 先加载ImageNet，再覆盖自己的ckpt
        device=device,
        verbose=True,
        compile_model=False
    )
    
    # 加载Stage1最优模型初始化
    load_ckpt_into_model(model, str(CFG.get("init_ckpt", "")), device)
    
    # ------------------- CB loss权重 -------------------
    beta = float(CFG.get("beta", 0.999))
    include_pseudo = bool(CFG.get("cb_include_pseudo", False))
    
    if include_pseudo:
        # 不推荐：伪标签分布可能有偏
        real_labels = extract_labels(ds_real)
        pseudo_labels = [ds_pseudo.samples[i]["label"] for i in valid_indices]
        labels = real_labels + pseudo_labels
        print(f"[Stage2] CB统计: 真实({len(real_labels)}) + 伪标签({len(pseudo_labels)})")
    else:
        # 推荐：仅用真实样本统计，更稳定
        labels = extract_labels(ds_real)
        print(f"[Stage2] CB统计: 仅用真实样本 ({len(labels)}个)")
    
    class_w = compute_cb_weights(labels, num_classes=7, beta=beta, device=device)
    print(f"[Stage2] CB权重: {class_w.detach().cpu().numpy().round(4).tolist()}")
    
    # ------------------- 优化器与AMP -------------------
    optimizer = AdamW(
        model.parameters(),
        lr=float(CFG["lr"]),
        weight_decay=float(CFG["weight_decay"])
    )
    
    use_amp = bool(CFG.get("use_amp", False)) and str(device).startswith("cuda")
    scaler = GradScaler(enabled=use_amp)
    
    # ------------------- 日志初始化 -------------------
    if not os.path.exists(str(CFG["log_csv"])):
        with open(str(CFG["log_csv"]), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "time", "epoch", "lr",
                "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1",
                "test_loss", "test_acc", "test_f1",
                "ramp_coeff",  # 新增：记录ramp-up系数
            ])
    
    # ------------------- 训练循环 -------------------
    best_f1 = -1.0
    no_improve = 0
    total_epochs = int(CFG["epochs"])
    
    print(f"\n[Stage2] ===== 开始训练 (共{total_epochs}轮, ramp-up={CFG['pseudo_rampup_epochs']}轮) =====\n")
    
    for epoch in range(total_epochs):
        # 学习率调度
        lr = cosine_warmup_lr(
            float(CFG["lr"]),
            float(CFG["lr_floor"]),
            int(CFG["warmup_epochs"]),
            total_epochs,
            epoch
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # ramp-up系数（用于日志）
        ramp_epochs = int(CFG.get("pseudo_rampup_epochs", 0))
        ramp = 1.0 if ramp_epochs <= 0 else min(1.0, (epoch + 1) / ramp_epochs)
        
        # 训练
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, optimizer, train_loader, device, epoch, CFG, class_w, scaler
        )
        
        # 验证
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device)
        te_loss, te_acc, te_f1 = evaluate(model, test_loader, device)
        
        elapsed = time.time() - t0
        
        # 打印进度
        print(
            f"[Stage2] Epoch {epoch+1:3d}/{total_epochs} | lr={lr:.6f} | ramp={ramp:.3f} | "
            f"Train={tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
            f"Val={va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f} | "
            f"Test={te_loss:.4f}/{te_acc:.4f}/{te_f1:.4f} | "
            f"{elapsed:.1f}s"
        )
        
        # 写日志
        with open(str(CFG["log_csv"]), "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                int(time.time()), epoch + 1, lr,
                tr_loss, tr_acc, tr_f1,
                va_loss, va_acc, va_f1,
                te_loss, te_acc, te_f1,
                f"{ramp:.4f}",
            ])
        
        # 早停判断（以val_f1为准）
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), str(CFG["best_ckpt"]))
            print(f"  -> 保存最优模型 (Val F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(CFG["early_stop_patience"]):
                print(f"[Stage2] 早停: {epoch+1}轮无提升")
                break
    
    print(f"\n[Stage2] ===== 训练完成 =====")
    print(f"最优验证F1: {best_f1:.4f}")
    print(f"最优模型: {CFG['best_ckpt']}")
    
    # ------------------- 最终评估 -------------------
    try:
        from evaluate import run_evaluation
        print("\n[Stage2] 运行最终评估...")
        _ = run_evaluation(
            dict(
                device=device,
                csv_base=os.path.dirname(str(CFG["train_csv"])),
                img_base=CFG["img_base"],
                save_dir=str(CFG["save_dir"]),
                best_ckpt=str(CFG["best_ckpt"]),
                batch_size=int(CFG["batch_size"]),
                num_workers=0,  # 评估用单进程
                pin_memory=True,
                persistent_workers=False,
                per_class_limit=5000,
                model_variant="large",
            ),
            dict(split="both", tta=True, ckpt=str(CFG["best_ckpt"]))
        )
        print("[Stage2] 评估完成，混淆矩阵已生成")
    except Exception as e:
        print(f"[Stage2] 最终评估跳过: {e}")


if __name__ == "__main__":
    main()