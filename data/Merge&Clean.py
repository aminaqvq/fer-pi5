import os
import math
import pandas as pd
from collections import defaultdict

# ============== 配置区（按需修改） ==============
BASE_DIR = r"F:\fer-pi5\data\csv"  # 你的 csv 目录
MERGED_PATH = os.path.join(BASE_DIR, "merged.csv")

TRAIN_OUT      = os.path.join(BASE_DIR, "train.csv")
VAL_OUT        = os.path.join(BASE_DIR, "val.csv")
TEST_OUT       = os.path.join(BASE_DIR, "test.csv")
UNLABELED_OUT  = os.path.join(BASE_DIR, "unlabeled.csv")

LABEL_COL = "emotion"
RNG_SEED  = 42

# 选择 unlabeled 生成模式： "ratio" 或 "target"
MODE = "ratio"        # "ratio" | "target"

# MODE="ratio" 时生效：每个类别预留比例
UNLABELED_RATIO = 0.20   # 建议 0.1 ~ 0.4；例如 0.20 约留 20%

# MODE="target" 时生效：全局目标无标签总数
UNLABELED_TARGET = 50000

# 验证/测试集的每类最小样本数保护（labeled 部分）
MIN_VAL_PER_CLASS  = 1    # 若不需要可设 0
MIN_TEST_PER_CLASS = 1    # 若不需要可设 0
# ==============================================


def _print_header(msg: str):
    print("\n" + msg)
    print("-" * len(msg))


def _stats_block(name: str, df: pd.DataFrame, label_col: str):
    total = len(df)
    if total == 0:
        print(f"{name:<10}: {total:>6} 样本 | (空)")
        return
    counts = df[label_col].value_counts().sort_index().to_dict()
    print(f"{name:<10}: {total:>6} 样本 | 类别分布: {counts}")


def _load_or_merge(base_dir: str, merged_path: str, label_col: str) -> pd.DataFrame:
    train_csv = os.path.join(base_dir, "train_old.csv")
    val_csv   = os.path.join(base_dir, "val_old.csv")
    test_csv  = os.path.join(base_dir, "test_old.csv")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    assert set(train_df.columns) == set(val_df.columns) == set(test_df.columns), "CSV 列名不一致！"

    merged_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"✅ 合并完成，共 {len(merged_df)} 条样本。")
    merged_df.to_csv(merged_path, index=False)
    print(f"💾 已保存合并文件: {merged_path}")

    # 统一列名小写
    merged_df.columns = [c.lower().strip() for c in merged_df.columns]
    if label_col not in merged_df.columns:
        raise ValueError(f"合并后的 CSV 缺少列: {label_col}")
    return merged_df


def _strict_8_1_1_from_labeled_slice(g: pd.DataFrame, t: int):
    """
    从同一类别 g（已随机打乱）中，按整份配额 t 产出：
      - Train: 8t
      - Val:   1t
      - Test:  1t
    返回 (train_df, val_df, test_df, leftover_df)
    leftover_df 是 labeled 剩余的碎片（不足10的样本），需并入 unlabeled。
    """
    n_train, n_val, n_test = 8 * t, t, t
    take = n_train + n_val + n_test
    train = g.iloc[:n_train]
    val   = g.iloc[n_train:n_train + n_val]
    test  = g.iloc[n_train + n_val:take]
    leftover = g.iloc[take:]  # labeled 剩余碎片
    return train, val, test, leftover


def main():
    merged_df = _load_or_merge(BASE_DIR, MERGED_PATH, LABEL_COL)

    # 全局打乱，类内会再次打乱
    merged_df = merged_df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

    labels = sorted(merged_df[LABEL_COL].unique())
    per_class_counts = merged_df[LABEL_COL].value_counts().reindex(labels).fillna(0).astype(int)
    print("🧮 每类样本数:", per_class_counts.to_dict())

    train_parts, val_parts, test_parts, unlab_parts = [], [], [], []

    if MODE not in {"ratio", "target"}:
        raise ValueError("MODE 必须是 'ratio' 或 'target'。")

    if MODE == "target":
        total = len(merged_df)
        target = min(UNLABELED_TARGET, total - 30)  # 留一些给 labeled
        # 按类别占比分配 unlabeled 配额（整数四舍五入），最后再用一个“差额修正”确保总数准确
        raw_alloc = {lab: int(round(target * (per_class_counts[lab] / total))) for lab in labels}
        diff = target - sum(raw_alloc.values())
        # 对差额进行修正：按各类样本数从大到小逐个 +1/-1
        if diff != 0:
            order = sorted(labels, key=lambda l: per_class_counts[l], reverse=(diff > 0))
            i = 0
            while diff != 0 and i < len(order):
                lab = order[i % len(order)]
                raw_alloc[lab] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
                i += 1

    # ======== 按类别执行：先确定 unlabeled，再对剩余做严格 8:1:1 ========
    for lab, g_all in merged_df.groupby(LABEL_COL, sort=True):
        # 类内打乱
        g_all = g_all.sample(frac=1.0, random_state=RNG_SEED)

        n_total = len(g_all)

        # 1) 计算 unlabeled 预留
        if MODE == "ratio":
            n_unlab = int(n_total * UNLABELED_RATIO)
        else:  # MODE == "target"
            n_unlab = raw_alloc.get(lab, 0)

        # 限制不要抽空（至少给 labeled 留出最小需要）
        min_needed = max(10, MIN_VAL_PER_CLASS + MIN_TEST_PER_CLASS)  # 至少能凑出 1 份 8:1:1，或满足最小 val/test
        if n_total - n_unlab < min_needed:
            n_unlab = max(0, n_total - min_needed)

        # 2) 切片得到 unlabeled 预留 & labeled pool
        #   这里先把“预留 unlabeled”放到末尾，便于后续 labeled 配额切片
        g_labeled  = g_all.iloc[:n_total - n_unlab]
        g_unlab_pre= g_all.iloc[n_total - n_unlab:]

        # 3) 在 labeled pool 内做严格 8:1:1 —— 需要整份配额 t
        n_lab = len(g_labeled)
        t = n_lab // 10  # 整份配额数
        # 若开启最小 val/test 保护，确保 t 至少能满足各自最小需求：
        t_min_by_val  = math.ceil(MIN_VAL_PER_CLASS) if MIN_VAL_PER_CLASS > 0 else 0
        t_min_by_test = math.ceil(MIN_TEST_PER_CLASS) if MIN_TEST_PER_CLASS > 0 else 0
        t_required = max(t_min_by_val, t_min_by_test)
        t = min(t, t) if t_required == 0 else max(0, min(t, t))  # 为可读，保持 t 不变
        if t < t_required:
            # labeled pool 不足以满足最小 val/test，则退而求其次：把全部进 unlabeled，避免破坏 8:1:1
            unlab_parts.append(g_all)
            continue

        # 4) 分配 8:1:1，并把 labeled leftover 也放进 unlabeled
        tr, va, te, leftover_lab = _strict_8_1_1_from_labeled_slice(g_labeled, t)

        # 5) 组装各部分
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)

        # unlabeled = 预留 + labeled 的 leftover 碎片
        if len(leftover_lab) > 0:
            unlab_parts.append(leftover_lab)
        if len(g_unlab_pre) > 0:
            unlab_parts.append(g_unlab_pre)

    # ======== 拼接并导出 ========
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=merged_df.columns)
    val_df   = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame(columns=merged_df.columns)
    test_df  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame(columns=merged_df.columns)
    unlab_df = pd.concat(unlab_parts, ignore_index=True) if unlab_parts else pd.DataFrame(columns=merged_df.columns)

    # 守恒校验
    assert len(train_df) + len(val_df) + len(test_df) + len(unlab_df) == len(merged_df), "样本数不守恒！"

    _print_header("📊 划分结果")
    _stats_block("Train",     train_df, LABEL_COL)
    _stats_block("Val",       val_df,   LABEL_COL)
    _stats_block("Test",      test_df,  LABEL_COL)
    _stats_block("Unlabeled", unlab_df, LABEL_COL)

    # 导出
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    unlab_df.to_csv(UNLABELED_OUT, index=False)

    _print_header("✅ 导出完成")
    print(f"Train     -> {TRAIN_OUT}")
    print(f"Val       -> {VAL_OUT}")
    print(f"Test      -> {TEST_OUT}")
    print(f"Unlabeled -> {UNLABELED_OUT}")


if __name__ == "__main__":
    main()