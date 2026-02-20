import os
import sys
import csv
import math
import json
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torch.amp import autocast
from torch.cuda.amp import GradScaler

# 项目内导入
from dataset import FER2013Hybrid, IMG_SIZE
from model_mbv3 import get_model

# ============================================================
# 全局默认配置（Stage1基准）
# ============================================================
DEFAULT_CFG: Dict[str, object] = {
    # === 路径根目录（按你的项目修改）===
    "project_root": r"F:\fer-pi5",

    # === 数据路径（相对project_root或绝对路径）===
    "train_csv": r"data\csv\train.csv",
    "val_csv": r"data\csv\val.csv",
    "test_csv": r"data\csv\test.csv",
    "unlabeled_csv": r"data\csv\unlabeled.csv",
    "img_base": None,

    # === 训练参数 ===
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 200,
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,

    # === 优化器 ===
    "lr": 5e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,

    # === 类别均衡（CB loss）===
    "beta": 0.995,
    "label_smoothing": 0.04,
    "cb_include_pseudo": False,  # Stage2+时是否用伪标签统计CB权重

    # === 半监督特有（Stage2+生效）===
    "pseudo_conf_min": 0.0,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 1.0,
    "pseudo_rampup_epochs": 5,

    # === 稳定性 ===
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    # === 早停 ===
    "early_stop_patience": 20,
    "val_interval": 1,

    # === 可复现性 ===
    "seed": 42,

    # === 模型 ===
    "model_variant": "large",
    "pretrained": True,
}

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 工具函数
# ============================================================
def seed_all(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch: int) -> float:
    """线性warmup + cosine衰减"""
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """计算批次准确率"""
    return float((logits.argmax(1) == target).float().mean().item())


def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> float:
    """计算宏平均F1"""
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
    """构造DataLoader"""
    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        drop_last=bool(shuffle),
    )
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
    return DataLoader(ds, **kwargs)


# ============================================================
# 伪标签置信度读取（Stage2+需要）
# ============================================================
def read_pseudo_confs(csv_path: str, conf_min: float = 0.0) -> Tuple[List[float], List[int]]:
    """
    读取伪标签CSV的置信度，返回(conf列表, 有效索引列表)
    无效项（conf < conf_min）标记为-1并过滤
    """
    import csv as csv_mod

    confs: List[float] = []
    valid_indices: List[int] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for idx, row in enumerate(reader):
            c = row.get("conf", row.get("Conf", None))
            try:
                c = float(c) if c is not None else 1.0
            except Exception:
                c = 1.0

            if c < conf_min:
                confs.append(-1.0)  # 标记无效
            else:
                confs.append(float(c))
                valid_indices.append(idx)

    return confs, valid_indices


# ============================================================
# 带权重的数据集包装（Stage2+核心）
# ============================================================
class WeightedDataset(Dataset):
    """
    为样本附加权重，区分真实/伪标签
    返回5元组: (image, label, weight, is_pseudo, idx)
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
        self.is_pseudo_flags = is_pseudo_flags

        if self.weights is not None:
            assert len(self.weights) == len(base)
        if self.is_pseudo_flags is not None:
            assert len(self.is_pseudo_flags) == len(base)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        w = self.default_w if (self.weights is None) else self.weights[idx]
        is_pseudo = False if (self.is_pseudo_flags is None) else self.is_pseudo_flags[idx]
        return x, y, torch.tensor(w, dtype=torch.float32), is_pseudo, idx


# ============================================================
# CB权重计算
# ============================================================
def compute_cb_weights(labels: List[int], num_classes: int, beta: float, device: str) -> torch.Tensor:
    """计算Class-Balanced权重"""
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    eff = 1.0 - np.power(beta, counts)
    cb = (1.0 - beta) / np.maximum(eff, 1e-8)
    cb = cb / cb.mean()
    return torch.tensor(cb, dtype=torch.float32, device=device)


def extract_labels(ds: Dataset) -> List[int]:
    """从数据集提取所有标签"""
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
    """安全加载检查点"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}, training from scratch")
        return False

    state = torch.load(ckpt_path, map_location="cpu")

    # 解包state_dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 移除module.前缀
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    return True


# ============================================================
# 训练与验证
# ============================================================
def train_one_epoch(
        model: torch.nn.Module,
        optimizer: AdamW,
        loader: DataLoader,
        device: str,
        epoch: int,
        cfg: Dict,
        class_w: Optional[torch.Tensor],
        scaler: GradScaler,
        is_ssl: bool = False
) -> Tuple[float, float, float]:
    """
    训练一个epoch

    Args:
        is_ssl: 是否为半监督模式（Stage2+）
    """
    model.train()
    total_loss = total_acc = total_f1 = 0.0
    total_wsum = 0.0

    use_amp = bool(cfg.get("use_amp", False)) and str(device).startswith("cuda")

    # Ramp-up系数（仅SSL模式）
    ramp = 1.0
    if is_ssl:
        ramp_epochs = int(cfg.get("pseudo_rampup_epochs", 0))
        if ramp_epochs > 0:
            ramp = min(1.0, (epoch + 1) / ramp_epochs)

    for batch in loader:
        # 解包
        if is_ssl:
            # 5元组: (img, label, weight, is_pseudo, idx)
            xb, yb, wb_base, is_pseudo_flags, _ = batch

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb_base = wb_base.to(device, non_blocking=True)

            # Ramp-up: 仅伪标签样本受影响
            is_pseudo_mask = torch.tensor(is_pseudo_flags, dtype=torch.bool, device=device)
            wb = wb_base.clone()
            wb[is_pseudo_mask] = wb_base[is_pseudo_mask] * ramp

            # 调试信息（前3epoch首batch）
            if epoch < 3 and total_wsum == 0:
                real_cnt = (~is_pseudo_mask).sum().item()
                pseudo_cnt = is_pseudo_mask.sum().item()
                print(f"  [Debug Epoch {epoch + 1}] real={real_cnt}, pseudo={pseudo_cnt}, "
                      f"ramp={ramp:.3f}, pseudo_w_mean={wb[is_pseudo_mask].mean().item():.3f}")
        else:
            # 3元组: (img, label) - Stage1纯监督
            xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = torch.ones(xb.size(0), device=device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(xb)

            # 计算损失
            if class_w is not None:
                loss_vec = F.cross_entropy(
                    logits, yb,
                    weight=class_w,
                    label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                    reduction="none"
                )
            else:
                loss_vec = F.cross_entropy(
                    logits, yb,
                    label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                    reduction="none"
                )

            # 加权归一化
            wsum = wb.sum().clamp_min(1e-6)
            loss = (loss_vec * wb).sum() / wsum

        # 反向传播
        scaler.scale(loss).backward()

        if bool(cfg.get("grad_clip", False)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(cfg.get("max_norm", 1.0))
            )

        scaler.step(optimizer)
        scaler.update()

        # 统计
        with torch.no_grad():
            bs_w = float(wsum.item())
            total_loss += float(loss.item()) * bs_w
            total_acc += accuracy(logits, yb) * bs_w
            total_f1 += macro_f1(logits, yb) * bs_w
            total_wsum += bs_w

    denom = max(1e-6, total_wsum)
    return total_loss / denom, total_acc / denom, total_f1 / denom


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: str
) -> Tuple[float, float, float]:
    """验证/测试评估"""
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
# 混淆矩阵与详细评估
# ============================================================
@torch.no_grad()
def confusion_and_per_class(
        model: torch.nn.Module,
        loader: DataLoader,
        num_classes: int = 7,
        device: str = "cuda"
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    """计算混淆矩阵和每类指标"""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1)

        for t, p in zip(yb.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    cm = cm.cpu().numpy()
    prec, rec, f1 = [], [], []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        prec.append(float(p))
        rec.append(float(r))
        f1.append(float(2 * p * r / (p + r + 1e-8)))

    return cm, prec, rec, f1


def save_confusion_png(cm: np.ndarray, path: str, title: str = "Confusion Matrix"):
    """保存混淆矩阵图为PNG"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6), dpi=150)
        plt.imshow(cm, interpolation="nearest", cmap="viridis")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()

        # 添加数值标注
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=8)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Saved confusion matrix: {path}")
    except Exception as e:
        print(f"[WARN] Failed to save confusion matrix: {e}")


# ============================================================
# 主流程
# ============================================================
def get_stage_config(stage: int, cfg: Dict) -> Dict:
    """
    根据阶段生成完整配置
    自动推断路径：stage N 加载 stage N-1 的模型，使用 stage N-1 的伪标签
    """
    stage_cfg = dict(cfg)
    root = cfg["project_root"]

    # 确保是绝对路径
    def make_path(p):
        if os.path.isabs(p):
            return p
        return os.path.join(root, p)

    # 数据路径
    stage_cfg["train_csv"] = make_path(cfg["train_csv"])
    stage_cfg["val_csv"] = make_path(cfg["val_csv"])
    stage_cfg["test_csv"] = make_path(cfg["test_csv"])

    # 阶段特定配置
    if stage == 1:
        # Stage1: 纯监督，无伪标签
        stage_cfg["stage_name"] = "Stage1"
        stage_cfg["is_ssl"] = False
        stage_cfg["init_ckpt"] = None  # 从预训练开始
        stage_cfg["pseudo_csv"] = None
        stage_cfg["best_ckpt"] = make_path(rf"checkpoints\best_model_stage1.pth")
        stage_cfg["log_csv"] = make_path(rf"checkpoints\train_stage1_log.csv")

    else:
        # Stage2+: 半监督
        stage_cfg["stage_name"] = f"Stage{stage}"
        stage_cfg["is_ssl"] = True

        # 自动推断输入
        prev_stage = stage - 1
        stage_cfg["init_ckpt"] = make_path(rf"checkpoints\best_model_stage{prev_stage}.pth")
        stage_cfg["pseudo_csv"] = make_path(rf"data\csv\pseudo_labeled_stage{prev_stage}.csv")

        # 输出路径
        stage_cfg["best_ckpt"] = make_path(rf"checkpoints\best_model_stage{stage}.pth")
        stage_cfg["log_csv"] = make_path(rf"checkpoints\train_stage{stage}_log.csv")

        # 检查文件存在性
        if not os.path.exists(stage_cfg["init_ckpt"]):
            raise FileNotFoundError(
                f"Stage {stage} requires Stage {prev_stage} checkpoint: {stage_cfg['init_ckpt']}\n"
                f"请先完成 Stage {prev_stage} 训练"
            )
        if not os.path.exists(stage_cfg["pseudo_csv"]):
            raise FileNotFoundError(
                f"Stage {stage} requires pseudo labels from Stage {prev_stage}: {stage_cfg['pseudo_csv']}\n"
                f"请先运行 generate_pseudo_stage{prev_stage}.py 生成伪标签"
            )

    # 创建目录
    os.makedirs(os.path.dirname(stage_cfg["best_ckpt"]), exist_ok=True)

    return stage_cfg


def main():
    parser = argparse.ArgumentParser(description="通用多阶段训练脚本")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="训练阶段: 1=监督, 2+=半监督")
    parser.add_argument("--project-root", type=str, default=None,
                        help="项目根目录，覆盖默认配置")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖训练轮数")
    parser.add_argument("--lr", type=float, default=None,
                        help="覆盖学习率")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖批次大小")

    args = parser.parse_args()

    # 构建配置
    cfg = dict(DEFAULT_CFG)
    if args.project_root:
        cfg["project_root"] = args.project_root
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr:
        cfg["lr"] = args.lr
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    # 获取阶段配置
    stage_cfg = get_stage_config(args.stage, cfg)
    is_ssl = stage_cfg["is_ssl"]

    print(f"\n{'=' * 60}")
    print(f"开始 {stage_cfg['stage_name']} 训练")
    print(f"{'=' * 60}")
    print(f"模式: {'半监督 (SSL)' if is_ssl else '纯监督 (Supervised)'}")
    print(f"初始化: {stage_cfg['init_ckpt'] or 'ImageNet预训练'}")
    if is_ssl:
        print(f"伪标签: {stage_cfg['pseudo_csv']}")
    print(f"输出模型: {stage_cfg['best_ckpt']}")
    print(f"{'=' * 60}\n")

    # 设置随机种子
    seed_all(int(stage_cfg["seed"]))
    device = str(stage_cfg["device"])

    # ========================================================
    # 数据集构造
    # ========================================================
    img_root = None if stage_cfg["img_base"] in (None, "", "None") else str(stage_cfg["img_base"])

    # 真实标注数据（始终使用）
    ds_real = FER2013Hybrid(
        stage_cfg["train_csv"], img_root, "train",
        img_size=int(IMG_SIZE),
        two_views=False,
        include_label=True
    )

    if is_ssl:
        # 半监督模式：加载伪标签
        print(f"[INFO] 加载伪标签数据...")
        conf_min = float(stage_cfg.get("pseudo_conf_min", 0.0))
        conf_power = float(stage_cfg.get("pseudo_conf_power", 1.0))
        pseudo_scale = float(stage_cfg.get("pseudo_loss_scale", 1.0))

        # 读取置信度
        pseudo_confs, valid_indices = read_pseudo_confs(stage_cfg["pseudo_csv"], conf_min)
        print(f"  伪标签总数: {len(pseudo_confs)}, 有效: {len(valid_indices)} "
              f"(过滤率: {100 * (1 - len(valid_indices) / max(len(pseudo_confs), 1)):.1f}%)")

        # 加载伪标签数据集
        ds_pseudo = FER2013Hybrid(
            stage_cfg["pseudo_csv"], img_root, "unlabeled",
            img_size=int(IMG_SIZE),
            two_views=False,
            include_label=True
        )
        ds_pseudo.split = "train"  # 使用训练增强

        # 构建权重
        real_weights = [1.0] * len(ds_real)
        real_is_pseudo = [False] * len(ds_real)

        valid_confs = [pseudo_confs[i] for i in valid_indices]
        pseudo_weights = [
            (max(0.0, min(1.0, c)) ** conf_power) * pseudo_scale
            for c in valid_confs
        ]
        pseudo_is_pseudo = [True] * len(valid_indices)

        # 包装加权数据集
        ds_real_w = WeightedDataset(ds_real, real_weights, real_is_pseudo)
        ds_pseudo_w = WeightedDataset(Subset(ds_pseudo, valid_indices), pseudo_weights, pseudo_is_pseudo)

        # 合并
        ds_train = ConcatDataset([ds_real_w, ds_pseudo_w])
        train_loader = _make_loader(ds_train, stage_cfg["batch_size"], True, stage_cfg)

        print(f"[INFO] 训练集: 真实={len(ds_real)}, 伪标签={len(valid_indices)}, 总计={len(ds_train)}")
    else:
        # 纯监督模式
        train_loader = _make_loader(ds_real, stage_cfg["batch_size"], True, stage_cfg)
        print(f"[INFO] 训练集: 真实={len(ds_real)} (纯监督)")

    # 验证/测试集
    ds_val = FER2013Hybrid(stage_cfg["val_csv"], img_root, "val", img_size=int(IMG_SIZE))
    ds_test = FER2013Hybrid(stage_cfg["test_csv"], img_root, "test", img_size=int(IMG_SIZE))
    val_loader = _make_loader(ds_val, stage_cfg["batch_size"], False, stage_cfg)
    test_loader = _make_loader(ds_test, stage_cfg["batch_size"], False, stage_cfg)

    print(f"[INFO] 验证集: {len(ds_val)}, 测试集: {len(ds_test)}")

    # ========================================================
    # 模型
    # ========================================================
    model = get_model(
        str(stage_cfg.get("model_variant", "large")),
        num_classes=7,
        pretrained=bool(stage_cfg.get("pretrained", True)) and args.stage == 1,
        device=device,
        verbose=True,
        compile_model=False
    )

    # 加载前一阶段权重（Stage2+）
    if stage_cfg["init_ckpt"]:
        load_ckpt_into_model(model, stage_cfg["init_ckpt"], device)

    # ========================================================
    # CB Loss权重
    # ========================================================
    beta = float(stage_cfg.get("beta", 0.999))
    include_pseudo = is_ssl and bool(stage_cfg.get("cb_include_pseudo", False))

    if include_pseudo:
        # 不推荐：包含伪标签统计
        real_labels = extract_labels(ds_real)
        pseudo_labels = [ds_pseudo.samples[i]["label"] for i in valid_indices]
        labels = real_labels + pseudo_labels
        print(f"[INFO] CB统计: 真实({len(real_labels)}) + 伪标签({len(pseudo_labels)})")
    else:
        # 推荐：仅用真实样本
        labels = extract_labels(ds_real)
        print(f"[INFO] CB统计: 仅用真实样本 ({len(labels)}个)")

    class_w = compute_cb_weights(labels, num_classes=7, beta=beta, device=device)
    print(f"[INFO] CB权重: {class_w.detach().cpu().numpy().round(4).tolist()}")

    # ========================================================
    # 优化器与AMP
    # ========================================================
    optimizer = AdamW(
        model.parameters(),
        lr=float(stage_cfg["lr"]),
        weight_decay=float(stage_cfg["weight_decay"])
    )

    use_amp = bool(stage_cfg.get("use_amp", False)) and str(device).startswith("cuda")
    scaler = GradScaler(enabled=use_amp)

    # ========================================================
    # 日志初始化
    # ========================================================
    log_csv = stage_cfg["log_csv"]
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time", "epoch", "lr", "train_loss", "train_acc", "train_f1",
                      "val_loss", "val_acc", "val_f1", "test_loss", "test_acc", "test_f1"]
            if is_ssl:
                header.append("ramp_coeff")
            writer.writerow(header)

    # ========================================================
    # 训练循环
    # ========================================================
    best_f1 = -1.0
    no_improve = 0
    total_epochs = int(stage_cfg["epochs"])

    print(f"\n{'=' * 60}")
    print(f"开始训练 (共{total_epochs}轮)")
    if is_ssl:
        print(f"Ramp-up: 前{stage_cfg['pseudo_rampup_epochs']}轮渐进引入伪标签")
    print(f"{'=' * 60}\n")

    for epoch in range(total_epochs):
        # 学习率调度
        lr = cosine_warmup_lr(
            float(stage_cfg["lr"]),
            float(stage_cfg["lr_floor"]),
            int(stage_cfg["warmup_epochs"]),
            total_epochs,
            epoch
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Ramp-up系数
        ramp = 1.0
        if is_ssl:
            ramp_epochs = int(stage_cfg.get("pseudo_rampup_epochs", 0))
            if ramp_epochs > 0:
                ramp = min(1.0, (epoch + 1) / ramp_epochs)

        # 训练
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, optimizer, train_loader, device, epoch, stage_cfg, class_w, scaler, is_ssl
        )

        # 验证
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, device)
        te_loss, te_acc, te_f1 = evaluate(model, test_loader, device)

        elapsed = time.time() - t0

        # 打印进度
        msg = (f"[{stage_cfg['stage_name']}] Epoch {epoch + 1:3d}/{total_epochs} | "
               f"lr={lr:.6f} | "
               f"Train={tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
               f"Val={va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f} | "
               f"Test={te_loss:.4f}/{te_acc:.4f}/{te_f1:.4f} | "
               f"{elapsed:.1f}s")
        if is_ssl:
            msg += f" | ramp={ramp:.3f}"
        print(msg)

        # 写日志
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [int(time.time()), epoch + 1, lr,
                   tr_loss, tr_acc, tr_f1,
                   va_loss, va_acc, va_f1,
                   te_loss, te_acc, te_f1]
            if is_ssl:
                row.append(f"{ramp:.4f}")
            writer.writerow(row)

        # 早停判断（以Val F1为准）
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), stage_cfg["best_ckpt"])
            print(f"  -> 保存最优模型 (Val F1={best_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(stage_cfg["early_stop_patience"]):
                print(f"[INFO] 早停: {epoch + 1}轮无提升")
                break

    # ========================================================
    # 训练结束：最终评估与混淆矩阵
    # ========================================================
    print(f"\n{'=' * 60}")
    print(f"{stage_cfg['stage_name']} 训练完成")
    print(f"最优验证F1: {best_f1:.4f}")
    print(f"最优模型: {stage_cfg['best_ckpt']}")
    print(f"{'=' * 60}")

    # 加载最优模型生成详细评估
    print("\n[INFO] 生成详细评估报告...")
    load_ckpt_into_model(model, stage_cfg["best_ckpt"], device)

    # 验证集混淆矩阵
    cm_val, prec_val, rec_val, f1_val = confusion_and_per_class(model, val_loader, device=device)
    save_confusion_png(cm_val,
                       os.path.join(os.path.dirname(stage_cfg["best_ckpt"]), f"val_confusion_stage{args.stage}.png"),
                       f"Stage{args.stage} Val Confusion")

    # 测试集混淆矩阵
    cm_test, prec_test, rec_test, f1_test = confusion_and_per_class(model, test_loader, device=device)
    save_confusion_png(cm_test,
                       os.path.join(os.path.dirname(stage_cfg["best_ckpt"]), f"test_confusion_stage{args.stage}.png"),
                       f"Stage{args.stage} Test Confusion")

    # 保存详细指标
    metrics = {
        "stage": args.stage,
        "stage_name": stage_cfg["stage_name"],
        "best_val_f1": float(best_f1),
        "final_val_f1": float(va_f1),
        "final_test_f1": float(te_f1),
        "val_per_class": {"precision": prec_val, "recall": rec_val, "f1": f1_val},
        "test_per_class": {"precision": prec_test, "recall": rec_test, "f1": f1_test},
        "config": {k: str(v) if isinstance(v, os.PathLike) else v
                   for k, v in stage_cfg.items() if k != "device"},
        "timestamp": int(time.time())
    }

    metrics_path = os.path.join(os.path.dirname(stage_cfg["best_ckpt"]),
                                f"metrics_stage{args.stage}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 详细指标已保存: {metrics_path}")
    print(f"\n[SUMMARY] {stage_cfg['stage_name']} 最终测试F1: {te_f1:.4f}")

    return metrics


if __name__ == "__main__":
    main()