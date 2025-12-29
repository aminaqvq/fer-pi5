import os as _os
_os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import os
import math
import csv
import json
import time
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import get_dataloaders_hybrid
from model_mbv3 import get_model

# ================= Default CFG（改这里→直接运行） =================
CFG: Dict[str, object] = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",

    # 路径：按你的项目结构修改
    csv_base=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv")),
    img_base=None,  # 若 CSV 里路径需要前缀根目录，填字符串；否则保持 None
    save_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")),
    log_csv=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "train_log.csv")),
    best_ckpt=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_model_ssl.pth")),

    # data & loader
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    dynamic_sampling=True,
    per_class_limit=5000,
    prefetch_factor=2,

    # train
    epochs=100,
    lr=5e-4,
    lr_floor=1e-6,
    weight_decay=1e-4,
    warmup_epochs=2,
    min_train_epochs=5,

    # 有标签分支增强
    use_mixup=True,
    mixup_alpha=0.2,     # 0 关闭；建议 0.2~0.4

    # ssl（半监督）
    use_unlabeled=True,
    unlabeled_two_views=True,  # 需要 dataset 返回 (weak, strong, meta) 两视图
    weak_strong=True,          # True: 弱增强生成伪标签，强增强训练
    lambda_u=1.0,              # 无标签损失系数上限
    ramp_epochs=10,            # 前若干 epoch 线性 ramp 到 lambda_u
    u_cap=0.03,                # 无标签 per-sample 损失裁顶（稳训练）
    sharpen_T=0.5,             # 温度锐化 T<1 更果断；=1 等于不锐化
    use_da=True,               # Distribution Alignment：校正伪标签类别分布
    da_momentum=0.999,         # 模型分布 EMA 动量
    da_eps=1e-6,               # 避免除零

    # 动态按类阈值（来自 v2；本版加入 per-class 动量）
    per_class_init_thr=[0.85, 0.92, 0.85, 0.90, 0.82, 0.85, 0.88],
    thr_momentum=0.90,         # 默认 EMA 动量
    weak_classes=[1, 4],       # 弱类列表（按你数据调整）
    thr_momentum_weak=0.80,    # 弱类更快更新（数值更小→更新更快）
    thr_floor=0.70,
    thr_ceil=0.95,
    thr_scale=0.98,
    weak_bias=-0.03,           # 弱类额外降低阈值

    # Focal Loss（仅用于无标签伪监督）
    focal_gamma=2.0,

    # stability
    use_amp=True,
    grad_clip=True,
    max_norm=1.0,
    ema=True,
    ema_decay=0.999,

    # eval toggles in training
    tta=True,  # 训练期评估是否做 TTA

    # early stop
    early_stop_patience=10,
    early_stop_min_delta=1e-4,

    # reproducibility
    seed=42,

    # model
    model_variant="large",
    pretrained=True,
    compile_model=False,

    # === 自动评估（训练结束后） ===
    auto_eval=True,      # 训练结束后自动评估
    eval_split="both",   # "val" / "test" / "both"
    eval_tta=True,       # 评估是否做 TTA
)

AMP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= Utils =================
def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cosine_warmup_lr(base_lr: float, floor: float, warmup_epochs: int, total_epochs: int, epoch: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return floor + (base_lr - floor) * 0.5 * (1 + math.cos(math.pi * progress))

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}; self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[name] = p.data.clone()
            p.data = self.shadow[name].clone()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data = self.backup[name].clone()
        self.backup = {}

# ================= Metrics =================
def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(1) == target).float().mean().item())

def macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> float:
    pred = logits.argmax(1)
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (target == c)).sum().item()
        fp = ((pred == c) & (target != c)).sum().item()
        fn = ((pred != c) & (target == c)).sum().item()
        p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return float(np.mean(f1s))

@torch.no_grad()
def confusion_and_per_class(model, loader, num_classes=7, tta=True, device="cuda"):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logits = (model(xb) + model(torch.flip(xb, dims=[-1]))) * 0.5 if tta else model(xb)
        pred = logits.argmax(1)
        for t, p in zip(yb.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    cm = cm.cpu().numpy()
    prec, rec, f1 = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        prec.append(float(p)); rec.append(float(r)); f1.append(float(2*p*r/(p+r+1e-8)))
    return cm, prec, rec, f1

# ===== MixUp（有标签分支） =====
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        lam = 1.0
        return x, y, y, lam
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_ce(logits: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

# ============== Focal utils（无标签分支的伪监督） ==============
def focal_factor(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    p = torch.softmax(logits, dim=1)
    pt = p.gather(1, target.view(-1, 1)).clamp_(1e-8, 1.0)
    return (1.0 - pt).pow(gamma).view(-1)

# ============== 简单的概率锐化与分布对齐 ==============
def sharpen(p: torch.Tensor, T: float) -> torch.Tensor:
    if T == 1.0: return p
    ps = p.pow(1.0 / T)
    return ps / ps.sum(dim=1, keepdim=True).clamp_min(1e-8)

class DistAlign:
    """ 轻量 DA：维护模型预测分布的 EMA，并向有标签先验对齐。 """
    def __init__(self, num_classes: int, momentum: float = 0.999, eps: float = 1e-6, device="cuda"):
        self.m = momentum
        self.eps = eps
        self.q_model = torch.ones(num_classes, device=device) / num_classes  # 模型分布 EMA
        self.q_target = torch.ones(num_classes, device=device) / num_classes # 目标分布（由有标签先验设定）
    @torch.no_grad()
    def set_target_from_counts(self, counts: np.ndarray):
        q = torch.tensor(counts / np.maximum(counts.sum(), 1.0), dtype=torch.float32, device=self.q_model.device)
        self.q_target = q.clamp_min(self.eps)
    @torch.no_grad()
    def update_and_align(self, p: torch.Tensor) -> torch.Tensor:
        # p: (B, C) 未对齐分布
        batch_mean = p.mean(dim=0)
        self.q_model = self.m * self.q_model + (1 - self.m) * batch_mean
        # 校正：p' ∝ p * (q_target / q_model)
        ratio = (self.q_target / self.q_model).clamp_min(self.eps)
        p_adj = p * ratio[None, :]
        return p_adj / p_adj.sum(dim=1, keepdim=True).clamp_min(self.eps)

# ================= Train/Eval loops =================
def train_one_epoch(model, ema_helper, loader_l, loader_u, optimizer, criterion, scaler, epoch, cfg,
                    thr_state: Dict[str, torch.Tensor], da: Optional[DistAlign]):
    model.train()
    device = cfg["device"]; use_amp = bool(cfg["use_amp"])
    total_loss = total_acc = total_f1 = 0.0; total_n = 0

    # SSL 诊断
    ssl_masked = ssl_total = 0
    ssl_conf_sum = ssl_loss_wsum = 0.0

    unlabeled_iter = iter(loader_u) if (loader_u is not None and cfg["use_unlabeled"]) else None
    start_t = time.time()

    # SSL 温度/强度调度
    lam_u = float(cfg["lambda_u"]) * min((epoch + 1) / max(1, int(cfg.get("ramp_epochs", 10))), 1.0)
    T = float(cfg.get("sharpen_T", 1.0))
    gamma = float(cfg.get("focal_gamma", 2.0))

    for xb, yb in loader_l:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)

        # ---- 有标签分支：可选 MixUp ----
        if bool(cfg.get("use_mixup", False)):
            xb, y_a, y_b, lam = mixup(xb, yb, float(cfg.get("mixup_alpha", 0.2)))
            with autocast(device_type=AMP_DEVICE, enabled=use_amp):
                logits = model(xb)
                loss_s = mixup_ce(logits, y_a, y_b, lam)
        else:
            with autocast(device_type=AMP_DEVICE, enabled=use_amp):
                logits = model(xb)
                loss_s = criterion(logits, yb)

        loss = loss_s

        # ---- 无标签分支：weak->strong + sharpen + DA + 动态阈值 + Focal ----
        if unlabeled_iter is not None and lam_u > 0:
            try:
                if cfg["unlabeled_two_views"]:
                    u_weak, u_strong, _ = next(unlabeled_iter)
                else:
                    u_weak, _, _ = next(unlabeled_iter); u_strong = u_weak
            except StopIteration:
                unlabeled_iter = iter(loader_u)
                if cfg["unlabeled_two_views"]:
                    u_weak, u_strong, _ = next(unlabeled_iter)
                else:
                    u_weak, _, _ = next(unlabeled_iter); u_strong = u_weak

            u_weak = u_weak.to(device, non_blocking=True)
            u_strong = u_strong.to(device, non_blocking=True)

            with torch.no_grad():
                # teacher on weak
                logits_w = model(u_weak)
                p = torch.softmax(logits_w, dim=1)

                # 分布对齐（可选）
                if da is not None and bool(cfg.get("use_da", False)):
                    p = da.update_and_align(p)

                # 温度锐化
                p = sharpen(p, T)
                conf, pseudo = torch.max(p, dim=1)

                # ---- 动态阈值（每类 EMA；弱类更快动量 + 偏置）----
                ema_conf = thr_state["ema_conf"]  # shape [C]
                C = ema_conf.numel()
                base_m = float(cfg["thr_momentum"])
                weak_m = float(cfg.get("thr_momentum_weak", base_m))
                weak_set = set(int(c) for c in cfg.get("weak_classes", []))

                for c in range(C):
                    mc = (pseudo == c)
                    if mc.any():
                        batch_mean = conf[mc].mean()
                        m = weak_m if c in weak_set else base_m
                        ema_conf[c] = m * ema_conf[c] + (1 - m) * batch_mean

                thr = (ema_conf * float(cfg["thr_scale"])).clone()
                for c in weak_set: thr[c] = thr[c] + float(cfg.get("weak_bias", -0.03))
                thr = thr.clamp(float(cfg["thr_floor"]), float(cfg["thr_ceil"]))
                thr_each = thr[pseudo]
                mask = (conf >= thr_each).float()

            with autocast(device_type=AMP_DEVICE, enabled=use_amp):
                # student on strong
                logits_s = model(u_strong)
                mod = focal_factor(logits_s, pseudo, gamma=gamma).detach()
                per_sample = F.cross_entropy(logits_s, pseudo, reduction="none") * mask * mod
                cap = float(cfg.get("u_cap", 0.03))
                if cap > 0: per_sample = torch.clamp(per_sample, max=cap)
                loss_u = per_sample.sum() / (mask.sum() + 1e-8) if mask.sum() > 0 else torch.tensor(0.0, device=device)

            loss = loss + lam_u * loss_u

            # 诊断
            with torch.no_grad():
                msel = mask.bool()
                ssl_masked += int(msel.sum().item())
                ssl_total  += int(mask.numel())
                if msel.any():
                    ssl_conf_sum += float(conf[msel].mean().item()) * int(msel.sum().item())
                    ssl_loss_wsum += float(loss_u.detach().item() * msel.sum().item())

        # ---- 反向传播 ----
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["max_norm"]))
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["max_norm"]))
            optimizer.step()

        if ema_helper is not None: ema_helper.update(model)

        with torch.no_grad():
            total_loss += loss_s.detach().item() * yb.size(0)
            total_acc  += accuracy(logits.detach(), yb) * yb.size(0)
            total_f1   += macro_f1(logits.detach(), yb) * yb.size(0)
            total_n    += yb.size(0)

    elapsed = time.time() - start_t
    if ssl_total > 0:
        ssl_mask_rate = ssl_masked / ssl_total
        ssl_avg_conf  = ssl_conf_sum / max(ssl_masked, 1)
        ssl_loss_avg  = ssl_loss_wsum / max(ssl_masked, 1)
    else:
        ssl_mask_rate = 0.0; ssl_avg_conf = 0.0; ssl_loss_avg = 0.0

    diag = dict(
        epoch_time=elapsed,
        throughput=total_n / max(elapsed, 1e-6),
        ssl_mask_rate=ssl_mask_rate,
        ssl_avg_conf=ssl_avg_conf,
        ssl_loss=ssl_loss_avg,
        thr_snapshot=thr_state["ema_conf"].detach().cpu().tolist(),
    )
    return total_loss / max(1, total_n), total_acc / max(1, total_n), total_f1 / max(1, total_n), diag

@torch.no_grad()
def evaluate(model, loader: DataLoader, criterion, cfg, tta=True):
    model.eval(); device = cfg["device"]
    total_loss = total_acc = total_f1 = 0.0; total_n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        logits = (model(xb) + model(torch.flip(xb, dims=[-1]))) * 0.5 if tta else model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        total_acc  += accuracy(logits, yb) * yb.size(0)
        total_f1   += macro_f1(logits, yb) * yb.size(0)
        total_n    += yb.size(0)
    return total_loss / max(1, total_n), total_acc / max(1, total_n), total_f1 / max(1, total_n)

# ================= Main =================
def main():
    # 目录与随机种子
    os.makedirs(CFG["save_dir"], exist_ok=True)
    for k in ("log_csv", "best_ckpt"):
        base = os.path.dirname(CFG[k]); os.makedirs(base, exist_ok=True)
    seed_all(int(CFG["seed"]))
    device = str(CFG["device"]); print(f"Device: {device}", flush=True)

    # Data
    train_loader, val_loader, test_loader, unlabeled_loader = get_dataloaders_hybrid(
        csv_base=str(CFG["csv_base"]),
        img_base=(None if CFG["img_base"] in (None, "None", "") else str(CFG["img_base"])),
        batch_size=int(CFG["batch_size"]),
        num_workers=int(CFG["num_workers"]),
        pin_memory=bool(CFG["pin_memory"]),
        persistent_workers=bool(CFG["persistent_workers"]),
        dynamic_sampling=bool(CFG["dynamic_sampling"]),
        per_class=int(CFG["per_class_limit"]),
        include_unlabeled=bool(CFG["use_unlabeled"]),
        unlabeled_two_views=bool(CFG["unlabeled_two_views"]),
        prefetch_factor=int(CFG.get("prefetch_factor", 2)),
    )
    print("Dataloaders ready: Train=%d, Val=%d, Test=%d, Unlabeled=%s" %
          (len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset),
           "yes" if unlabeled_loader is not None else "no"), flush=True)

    # Model / Optimizer / Loss
    model = get_model(
        str(CFG.get("model_variant", "large")), num_classes=7,
        pretrained=bool(CFG.get("pretrained", True)), device=device,
        verbose=True, compile_model=bool(CFG.get("compile_model", False)),
    )

    # Class-Balanced 权重（β=0.999）
    subset = train_loader.dataset
    full_dataset = subset.dataset
    subset_indices = subset.indices
    labels = [full_dataset[i][1] for i in subset_indices if full_dataset[i][1] >= 0]
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=7).astype(np.float32)
    beta = 0.999; eff = 1.0 - np.power(beta, counts)
    cb = (1.0 - beta) / np.maximum(eff, 1e-8); cb = cb / cb.mean()
    class_w = torch.tensor(cb, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=float(CFG["lr"]), weight_decay=float(CFG["weight_decay"]))
    scaler = GradScaler(device=AMP_DEVICE, enabled=bool(CFG["use_amp"]))
    ema_helper = EMA(model, decay=float(CFG["ema_decay"])) if CFG["ema"] else None

    # ---- 动态阈值状态（每类一条 EMA 置信度）----
    C = 7
    init_thr = torch.tensor(list(CFG["per_class_init_thr"]), device=device, dtype=torch.float32)
    thr_state = {
        # 用“初始阈值 / thr_scale”的倒推值作为 EMA 初始值（保证第一轮就有合理阈值）
        "ema_conf": (init_thr / float(CFG["thr_scale"])).clamp(float(CFG["thr_floor"]), float(CFG["thr_ceil"])).clone(),
    }

    # ---- 分布对齐（DA）对象 ----
    da = None
    if bool(CFG.get("use_da", False)):
        da = DistAlign(num_classes=C, momentum=float(CFG.get("da_momentum", 0.999)),
                       eps=float(CFG.get("da_eps", 1e-6)), device=device)
        da.set_target_from_counts(counts)  # 以有标签先验作为目标分布

    # 主训练日志（追加 SSL 指标与阈值快照）
    if not os.path.exists(CFG["log_csv"]):
        with open(CFG["log_csv"], "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "epoch", "lr",
                "train_loss", "train_acc", "train_f1",
                "val_loss_eval", "val_acc_eval", "val_f1_eval",
                "val_f1_cm",
                "ssl_mask_rate", "ssl_avg_conf", "ssl_loss",
                "thr_snapshot"  # JSON 串
            ])

    best_f1 = -1.0; no_improve = 0
    total_epochs = int(CFG["epochs"])

    for epoch in range(total_epochs):
        lr = cosine_warmup_lr(float(CFG["lr"]), float(CFG["lr_floor"]),
                              int(CFG["warmup_epochs"]), total_epochs, epoch)
        for pg in optimizer.param_groups: pg["lr"] = lr

        tr_loss, tr_acc, tr_f1, diag = train_one_epoch(
            model, ema_helper, train_loader, unlabeled_loader, optimizer, criterion, scaler, epoch, CFG, thr_state, da
        )

        # 评估（标准）
        va_loss, va_acc, va_f1 = evaluate(model, val_loader, criterion, CFG, tta=bool(CFG["tta"]))

        # 评估（CM-F1 用于挑 best；EMA shadow）
        eval_model = model
        if ema_helper is not None: ema_helper.apply_shadow(eval_model)
        cm, pc, rc, fc = confusion_and_per_class(eval_model, val_loader, num_classes=C,
                                                 tta=bool(CFG["tta"]), device=device)
        if ema_helper is not None: ema_helper.restore(eval_model)
        va_f1_cm = float(np.mean(fc))

        print(
            f"Epoch {epoch+1}/{total_epochs} | lr {lr:.6f} | "
            f"Train L/A/F1 {tr_loss:.4f}/{tr_acc:.4f}/{tr_f1:.4f} | "
            f"Val(eval) L/A/F1 {va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f} | "
            f"Val(CM) F1 {va_f1_cm:.4f} | "
            f"SSL mask {diag['ssl_mask_rate']:.2%}, conf {diag['ssl_avg_conf']:.3f}, loss_u {diag['ssl_loss']:.4f} | "
            f"{diag['throughput']:.1f} img/s, {diag['epoch_time']:.1f}s",
            flush=True
        )

        # 写日志（含阈值快照）
        with open(CFG["log_csv"], "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch+1, lr, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1, va_f1_cm,
                diag["ssl_mask_rate"], diag["ssl_avg_conf"], diag["ssl_loss"],
                json.dumps(diag["thr_snapshot"])
            ])

        # 以 CM-F1 挑最佳
        improved = va_f1_cm > (best_f1 + float(CFG["early_stop_min_delta"]))
        if improved or best_f1 < 0:
            best_f1 = va_f1_cm
            torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in model.state_dict().items()}, str(CFG["best_ckpt"]))
            print(f"  -> Saved best to {CFG['best_ckpt']} (CM-F1={best_f1:.4f})", flush=True)
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) >= int(CFG["min_train_epochs"]) and no_improve >= int(CFG["early_stop_patience"]):
            print("Early stop: no CM-F1 improvement.", flush=True)
            break

    # 训练结束：summary
    summary = {
        "best_f1_cm_val": float(best_f1),
        "epochs_run": int(epoch + 1),
        "cfg": {k: (str(v) if isinstance(v, os.PathLike) else v) for k, v in CFG.items()},
        "timestamp": int(time.time()),
    }
    with open(os.path.join(str(CFG["save_dir"]), "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nTraining finished.", flush=True)

    # === 自动评估 ===
    if bool(CFG.get("auto_eval", False)):
        try:
            from evaluate import run_evaluation
            print("\n[auto-eval] Start ...", flush=True)
            _ = run_evaluation(CFG, dict(
                split=str(CFG.get("eval_split", "both")),
                tta=bool(CFG.get("eval_tta", True)),
                ckpt=str(CFG.get("best_ckpt")),
            ))
            print("[auto-eval] Done.", flush=True)
        except Exception as e:
            print(f"[auto-eval] Failed: {e}", flush=True)

if __name__ == "__main__":
    main()