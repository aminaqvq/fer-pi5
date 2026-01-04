import os
import json
import csv
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders_hybrid
from model_mbv3 import get_model

# 与 train.py 保持一致的最小 CFG（可按需改路径）
CFG: Dict[str, object] = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",
    csv_base=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "csv")),
    img_base=None,
    save_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")),
    best_ckpt=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "best_model_stage2.pth")),
    batch_size=64,
    num_workers=4,            # 仅占位；run_evaluation 内会强制改为 0
    pin_memory=True,
    persistent_workers=True,  # 仅占位；run_evaluation 内会强制改为 False
    per_class_limit=5000,
    model_variant="large",
)

# 评估默认选项（可被 run_evaluation 的 eval_overrides 覆盖）
EVAL_DEFAULT: Dict[str, object] = dict(
    split="both",  # "val" / "test" / "both"
    tta=True,
    ckpt=None,     # None → 使用 CFG["best_ckpt"]
)


# ---------- 轻量评估工具（避免循环依赖） ----------
def _accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(1)
    return float((pred == target).float().mean().item())


def _macro_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> float:
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


@torch.no_grad()
def _evaluate_simple(model, loader, criterion, cfg, tta=True):
    model.eval()
    device = cfg["device"]
    total_loss = total_acc = total_f1 = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = (model(xb) + model(torch.flip(xb, dims=[-1]))) * 0.5 if tta else model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * yb.size(0)
        total_acc  += _accuracy(logits, yb) * yb.size(0)
        total_f1   += _macro_f1(logits, yb) * yb.size(0)
        total_n    += yb.size(0)
    return total_loss / max(1, total_n), total_acc / max(1, total_n), total_f1 / max(1, total_n)


@torch.no_grad()
def _confusion_and_per_class(model, loader, num_classes=7, tta=True, device="cuda"):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = (model(xb) + model(torch.flip(xb, dims=[-1]))) * 0.5 if tta else model(xb)
        pred = logits.argmax(1)
        for t, p in zip(yb.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    cm = cm.cpu().numpy()
    prec, rec, f1 = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
        prec.append(float(p)); rec.append(float(r)); f1.append(float(2 * p * r / (p + r + 1e-8)))
    return cm, prec, rec, f1


def _save_confusion_png(cm, path_png: str, title: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5), dpi=150)
        plt.imshow(cm, interpolation="nearest")
        plt.title(title); plt.xlabel("Pred"); plt.ylabel("True")
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        plt.tight_layout(); plt.savefig(path_png); plt.close()
    except Exception as e:
        print(f"[warn] Failed to save {path_png}: {e}", flush=True)


# ---------- 公开入口：供 train.py 调用，也可直接运行 ----------
def run_evaluation(cfg: Dict[str, object], eval_overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """
    离线评估入口。返回 summary 字典（含 val/test 指标）。
    **Windows 修复**：评估阶段强制单进程 DataLoader（num_workers=0, persistent_workers=False）。
    """
    # 合并配置
    E = dict(EVAL_DEFAULT)
    if eval_overrides:
        E.update(eval_overrides)

    os.makedirs(str(cfg["save_dir"]), exist_ok=True)
    log_csv_path = os.path.join(str(cfg["save_dir"]), "analysis_log.csv")

    # --- 评估 DataLoader：强制单进程，避免 WinError 1114 / shm.dll 冲突 ---
    train_loader, val_loader, test_loader, _ = get_dataloaders_hybrid(
        csv_base=str(cfg["csv_base"]),
        img_base=(None if cfg.get("img_base") in (None, "None", "") else str(cfg["img_base"])),
        batch_size=int(cfg.get("batch_size", 64)),
        num_workers=0,                 # <<<<<< 关键：评估用单进程
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=False,      # <<<<<< 关键：与 num_workers=0 搭配
        dynamic_sampling=False,
        per_class=int(cfg.get("per_class_limit", 5000)),
        include_unlabeled=False,
        unlabeled_two_views=False,
        prefetch_factor=2,
    )

    device = str(cfg.get("device", "cpu"))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    ckpt_path = str(E["ckpt"] or cfg["best_ckpt"])
    do_tta = bool(E["tta"])
    which = str(E["split"]).lower()

    # 模型
    print(f"[evaluate] Loading checkpoint: {ckpt_path}", flush=True)
    model = get_model(str(cfg.get("model_variant", "large")), num_classes=7,
                      pretrained=False, device=device, verbose=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state); model.to(device).eval()

    def run_one(name: str, loader):
        loss, acc, f1_eval = _evaluate_simple(model, loader, criterion, cfg, tta=do_tta)
        cm, pc, rc, fc = _confusion_and_per_class(model, loader, num_classes=7, tta=do_tta, device=device)
        f1_cm = float(np.mean(fc))
        print(f"[evaluate:{name}] Loss={loss:.4f}, Acc={acc:.4f}, F1(eval)={f1_eval:.4f}, F1(CM)={f1_cm:.4f}", flush=True)
        # 产物
        _save_confusion_png(cm, os.path.join(str(cfg["save_dir"]), f"{name}_confusion.png"),
                            f"{name.capitalize()} Confusion")
        with open(os.path.join(str(cfg["save_dir"]), f"{name}_per_class.json"), "w", encoding="utf-8") as jf:
            json.dump({"precision": pc, "recall": rc, "f1": fc, "confusion": cm.tolist()}, jf, indent=2, ensure_ascii=False)
        return dict(loss=float(loss), acc=float(acc), f1_eval=float(f1_eval), f1_cm=f1_cm)

    summary = {"timestamp": int(time.time()),
               "cfg": {k: (str(v) if isinstance(v, os.PathLike) else v) for k, v in cfg.items()},
               "ckpt": os.path.basename(ckpt_path),
               "tta": int(do_tta)}

    if which in ("val", "both"):
        summary["val"] = run_one("val", val_loader)
    if which in ("test", "both"):
        summary["test"] = run_one("test", test_loader)

    # 评估记录（便于多 ckpt 对比）
    if not os.path.exists(log_csv_path):
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["time", "ckpt", "split", "loss", "acc", "f1_eval", "f1_cm", "tta"])


    for sp in ("val", "test"):
        if sp in summary:
            with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([summary["timestamp"], summary["ckpt"], sp,
                                        summary[sp]["loss"], summary[sp]["acc"], summary[sp]["f1_eval"],
                                        summary[sp]["f1_cm"], summary["tta"]])

    # metrics_summary.json（覆盖/补充）
    with open(os.path.join(str(cfg["save_dir"]), "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[evaluate] Finished. Artifacts ->", str(cfg["save_dir"]), flush=True)
    return summary


def main():
    # 允许直接运行 evaluate.py（用本地 CFG/EVAL_DEFAULT）
    run_evaluation(CFG, None)


if __name__ == "__main__":
    main()