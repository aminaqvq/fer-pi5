#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pseudo_stage2传统.py

第二阶段伪标签生成：使用Stage2训练后的模型，在unlabeled数据上生成迭代优化的伪标签。
与generate_pseudo_stage1.py保持完全统一的代码风格、配置结构和日志格式。
"""

import os
import json
import csv
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import FER2013Hybrid, IMG_SIZE
from model_mbv3 import get_model


# ============================================================
# 配置（与Stage1完全统一结构）
# ============================================================
CFG: Dict[str, object] = dict(
    # 基础路径
    csv_base=r"F:\fer-pi5\data\csv",
    unlabeled_csv=r"F:\fer-pi5\data\csv\unlabeled.csv",
    img_base=None,
    
    # 教师模型：Stage2训练后的最优模型
    teacher_ckpt=r"F:\fer-pi5\checkpoints\best_model_stage2.pth",
    
    # 输出路径
    save_dir=r"F:\fer-pi5\data\csv",
    out_csv_name="pseudo_labeled_stage2.csv",      # 明确标识为Stage2生成
    out_stats_name="pseudo_stats_stage2.json",
    
    # 设备与推理
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    img_size=IMG_SIZE,
    tta=True,                    # 测试时增强，提升伪标签稳定性
    num_classes=7,
    
    # 伪标签筛选策略（与Stage1一致，可独立调整）
    min_conf=0.90,               # Stage2模型更可靠，可提高阈值
    max_per_class=25000,         # Stage2可适当增加上限
    
    # 可复现性
    seed=42,
)


# ============================================================
# 工具函数（与Stage1完全统一）
# ============================================================
def seed_all(seed: int = 42):
    """设置全局随机种子，保证结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    """
    包装Dataset，使__getitem__返回(image, index)
    用于根据index回溯原始样本信息（pixels/path）
    """
    def __init__(self, base: Dataset):
        self.base = base
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx: int):
        x, _ = self.base[idx]  # 忽略原始label（unlabeled为-1）
        return x, idx


def _make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    构造DataLoader，参数与Stage1完全统一
    """
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,        # 伪标签生成不丢弃样本
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# ============================================================
# 核心：第二阶段伪标签生成
# ============================================================
@torch.no_grad()
def generate_pseudo_stage2(cfg: Dict[str, object]) -> Dict[str, object]:
    """
    使用Stage2模型在unlabeled数据上生成伪标签。
    
    流程：
        1. 加载Stage2教师模型（best_model_stage2.pth）
        2. 对unlabeled数据进行TTA推理
        3. 按置信度阈值和每类上限筛选高质量伪标签
        4. 输出pseudo_labeled_stage2.csv和统计信息
    
    返回:
        stats: 包含生成数量、每类分布、筛选参数等信息的字典
    """
    device = str(cfg.get("device", "cpu"))
    seed_all(int(cfg.get("seed", 42)))
    
    # 确保输出目录存在
    os.makedirs(str(cfg["save_dir"]), exist_ok=True)
    out_csv_path = os.path.join(cfg["save_dir"], cfg["out_csv_name"])
    out_stats_path = os.path.join(cfg["save_dir"], cfg["out_stats_name"])
    
    # ------------------------------
    # 定位unlabeled.csv（与Stage1统一逻辑）
    # ------------------------------
    unlabeled_csv = cfg.get("unlabeled_csv")
    if unlabeled_csv in (None, "", "None"):
        base = str(cfg["csv_base"])
        cand = [f for f in os.listdir(base) if f.lower().endswith(".csv")]
        u_list = [f for f in cand if "unlabeled" in f.lower()]
        if not u_list:
            raise FileNotFoundError("❌ 未找到unlabeled CSV，请在CFG['unlabeled_csv']显式指定路径")
        unlabeled_csv = os.path.join(base, sorted(u_list)[0])
    
    unlabeled_csv = os.path.abspath(unlabeled_csv)
    print(f"[Stage2] 使用unlabeled数据: {unlabeled_csv}")
    print(f"[Stage2] 输出伪标签: {out_csv_path}")
    
    # ------------------------------
    # 构造Dataset和DataLoader
    # ------------------------------
    u_set = FER2013Hybrid(
        csv_path=unlabeled_csv,
        img_root=(None if cfg["img_base"] in (None, "", "None") else cfg["img_base"]),
        split="unlabeled",
        img_size=cfg["img_size"],
        two_views=False,           # 伪标签生成不需要双视图
        include_label=False,       # unlabeled无真实标签
    )
    
    loader = _make_loader(
        IndexedDataset(u_set),
        batch_size=cfg["batch_size"],
        shuffle=False,             # 保持顺序，便于调试
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
    )
    
    print(f"[Stage2] 无标签样本总数: {len(u_set)}")
    
    # ------------------------------
    # 加载Stage2教师模型
    # ------------------------------
    model = get_model(
        variant="large",
        num_classes=cfg["num_classes"],
        pretrained=False,          # 加载自己的ckpt，不需要ImageNet预训练
        device=device,
        verbose=True,
        compile_model=False,
    )
    
    ckpt = cfg["teacher_ckpt"]
    print(f"[Stage2] 加载Stage2教师模型: {ckpt}")
    
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    model.load_state_dict(state)
    model.to(device).eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Stage2] 模型参数量: {total_params/1e6:.2f}M")
    
    # ------------------------------
    # 伪标签生成（带进度条）
    # ------------------------------
    min_conf = cfg["min_conf"]
    max_per_class = cfg["max_per_class"]
    num_classes = cfg["num_classes"]
    tta = cfg["tta"]
    
    # 统计容器
    per_class_counts = [0] * num_classes
    selected_indices: List[int] = []      # 选中的样本在u_set中的索引
    selected_labels: List[int] = []       # 伪标签
    selected_confs: List[float] = []      # 置信度
    
    print(f"[Stage2] 开始生成伪标签 (min_conf={min_conf}, max_per_class={max_per_class}, TTA={tta})")
    
    for xb, idx in tqdm(loader, desc="Stage2 Generating", ncols=100):
        xb = xb.to(device, non_blocking=True)
        
        # TTA推理：原始 + 水平翻转，取平均
        if tta:
            logits = model(xb)
            logits_flip = model(torch.flip(xb, dims=[-1]))
            logits = 0.5 * (logits + logits_flip)
        else:
            logits = model(xb)
        
        # Softmax概率和预测
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        
        # 逐样本筛选
        for c, y, i in zip(conf.tolist(), pred.tolist(), idx.tolist()):
            # 条件1：置信度达标
            if c < min_conf:
                continue
            # 条件2：该类未达到上限
            if max_per_class and per_class_counts[y] >= max_per_class:
                continue
            
            selected_indices.append(i)
            selected_labels.append(y)
            selected_confs.append(float(c))
            per_class_counts[y] += 1
    
    total_selected = len(selected_indices)
    print(f"\n[Stage2] 伪标签生成完成")
    print(f"  - 选中样本数: {total_selected} / {len(u_set)} ({100*total_selected/len(u_set):.1f}%)")
    print(f"  - 每类分布: {per_class_counts}")
    print(f"  - 平均置信度: {np.mean(selected_confs):.4f}" if selected_confs else "  - 无选中样本")
    
    # ------------------------------
    # 写入CSV（与Stage1完全统一格式）
    # ------------------------------
    if total_selected == 0:
        print("⚠️ 警告: 未选中任何伪标签，请检查阈值设置")
        # 仍写入空文件，避免下游脚本报错
        rows = []
    else:
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
    
    print(f"[Stage2] 伪标签CSV已保存: {out_csv_path}")
    
    # ------------------------------
    # 保存统计信息（JSON，便于后续分析）
    # ------------------------------
    stats = {
        "stage": "stage2",
        "total_unlabeled": len(u_set),
        "selected": total_selected,
        "selection_rate": total_selected / len(u_set) if len(u_set) > 0 else 0,
        "per_class_counts": per_class_counts,
        "per_class_rates": [c/len(u_set) for c in per_class_counts],
        "avg_confidence": float(np.mean(selected_confs)) if selected_confs else 0.0,
        "min_conf": min_conf,
        "max_per_class": max_per_class,
        "tta": tta,
        "teacher_ckpt": os.path.abspath(ckpt),
        "unlabeled_csv": unlabeled_csv,
        "out_csv": os.path.abspath(out_csv_path),
        "timestamp": int(os.path.getmtime(out_csv_path)) if os.path.exists(out_csv_path) else None,
    }
    
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"[Stage2] 统计信息已保存: {out_stats_path}")
    
    return stats


# ============================================================
# 主入口
# ============================================================
def main():
    """命令行入口，支持直接运行"""
    # 可选：支持命令行覆盖配置
    import argparse
    parser = argparse.ArgumentParser(description="Stage2伪标签生成")
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="覆盖教师模型路径")
    parser.add_argument("--min-conf", type=float, default=None, help="覆盖置信度阈值")
    parser.add_argument("--max-per-class", type=int, default=None, help="覆盖每类上限")
    args = parser.parse_args()
    
    # 应用命令行覆盖
    cfg = dict(CFG)
    if args.teacher_ckpt:
        cfg["teacher_ckpt"] = args.teacher_ckpt
    if args.min_conf is not None:
        cfg["min_conf"] = args.min_conf
    if args.max_per_class is not None:
        cfg["max_per_class"] = args.max_per_class
    
    # 执行生成
    stats = generate_pseudo_stage2(cfg)
    
    # 简要报告
    print("\n" + "="*50)
    print("Stage2伪标签生成完成")
    print(f"选中率: {stats['selection_rate']*100:.1f}%")
    print(f"输出文件: {stats['out_csv']}")
    print("="*50)


if __name__ == "__main__":
    main()