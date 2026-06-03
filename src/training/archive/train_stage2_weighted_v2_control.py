from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import train_core
    from balanced_sampler_patch import install_balanced_batch_sampler
except Exception as exc:
    print("\n[Stage2 WEIGHTED_V2 CONTROL] Import failed.")
    print(f"Current file : {__file__}")
    print(f"Current dir  : {THIS_DIR}")
    print("Expected     : train_core.py and balanced_sampler_patch.py in the same directory.")
    print(f"Original err : {type(exc).__name__}: {exc}")
    raise


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
# Recommended:
#   set FER_PROJECT_ROOT=F:\fer-pi5
PROJECT_ROOT = Path(os.environ.get("FER_PROJECT_ROOT", r"/"))


def _p(*parts: str) -> str:
    """Return a project-relative path string."""
    return str(Path(*parts))


def _is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def _resolve_project_path(value: str | None) -> Optional[Path]:
    if _is_none_like(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Sample-weight support patch
# ---------------------------------------------------------------------------
def _read_weight_by_csv_line(csv_path: Path, weight_col: str) -> Dict[int, float]:
    """Return {csv_line_number_starting_at_2: sample_weight}.

    FER2013Hybrid stores each sample's original CSV line number in sample["row_index"].
    This lets us align sample weights even when the dataset filters by Usage.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Weighted train CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if weight_col not in fields:
            raise ValueError(
                f"Weighted train CSV must contain column {weight_col!r}. "
                f"Found fields: {fields}"
            )

        out: Dict[int, float] = {}
        errors: List[str] = []
        for line_no, row in enumerate(reader, start=2):
            raw = str(row.get(weight_col, "")).strip()
            if raw == "":
                weight = 1.0
            else:
                try:
                    weight = float(raw)
                except Exception:
                    errors.append(f"line {line_no}: invalid {weight_col}={raw!r}")
                    continue
            if not (0.0 <= weight <= 10.0):
                errors.append(f"line {line_no}: {weight_col} out of allowed range [0,10]: {weight}")
                continue
            out[line_no] = float(weight)

    if errors:
        raise ValueError("Invalid sample weights: " + " | ".join(errors[:30]))
    if not out:
        raise RuntimeError(f"No sample weights were read from {csv_path}")
    return out


def _sample_row_indices(ds: Any) -> List[int]:
    """Return CSV row_index values aligned to ds indices.

    Supports:
      FER2013Hybrid
      Subset(FER2013Hybrid, indices)
    """
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        base = ds.dataset
        indices = [int(i) for i in ds.indices]
    else:
        base = ds
        indices = list(range(len(ds)))

    if not hasattr(base, "samples"):
        raise TypeError("Expected FER2013Hybrid-like dataset with .samples for sample-weight alignment.")

    row_indices: List[int] = []
    for base_idx in indices:
        sample = base.samples[int(base_idx)]
        row_indices.append(int(sample.get("row_index", -1)))
    return row_indices


def _labels_from_dataset_like(ds: Any) -> List[int]:
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        base = ds.dataset
        return [int(base.samples[int(i)].get("label", -1)) for i in ds.indices]
    if hasattr(ds, "samples"):
        return [int(s.get("label", -1)) for s in ds.samples]
    labels: List[int] = []
    for i in range(len(ds)):
        item = ds[i]
        labels.append(int(item[1]))
    return labels


def _class_weight_sums(labels: Sequence[int], weights: Sequence[float]) -> Dict[str, float]:
    names = list(getattr(train_core, "LABELS"))
    sums = {name: 0.0 for name in names}
    for y, w in zip(labels, weights):
        if 0 <= int(y) < len(names):
            sums[names[int(y)]] += float(w)
    return sums


def install_labeled_sample_weight_patch(core: Any, *, weight_col: str = "sample_weight") -> None:
    """Patch train_core.build_datasets so labeled train rows can carry sample_weight.

    Existing train_core already supports WeightedDataset and weighted_ce_loss.
    The missing piece is only reading the labeled sample_weight column from CSV.
    """
    original_build_datasets = core.build_datasets

    def build_datasets_with_labeled_weights(cfg: Mapping[str, Any]):
        train_ds, val_ds, test_ds, meta = original_build_datasets(cfg)

        pseudo_csv = cfg.get("pseudo_csv")
        if not _is_none_like(pseudo_csv):
            raise ValueError(
                "weighted_v2_control must run without pseudo labels. "
                "Set pseudo_csv=None for this diagnostic experiment."
            )

        train_csv = Path(str(cfg["train_csv"]))
        weight_by_line = _read_weight_by_csv_line(train_csv, str(cfg.get("sample_weight_col", weight_col)))

        if not hasattr(train_ds, "base") or not hasattr(train_ds, "weights"):
            raise TypeError(
                "Expected train_core.build_datasets to return a WeightedDataset for labeled training data. "
                f"Got {type(train_ds)}"
            )

        base_ds = train_ds.base
        row_indices = _sample_row_indices(base_ds)
        weights: List[float] = []
        missing_rows: List[int] = []
        for row_index in row_indices:
            w = weight_by_line.get(int(row_index))
            if w is None:
                missing_rows.append(int(row_index))
                w = 1.0
            weights.append(float(w))

        train_ds.weights = weights

        labels = _labels_from_dataset_like(base_ds)
        meta.update({
            "sample_weight_enabled": True,
            "sample_weight_col": str(cfg.get("sample_weight_col", weight_col)),
            "sample_weight_source_csv": str(train_csv),
            "sample_weight_count": len(weights),
            "sample_weight_missing_rows": len(missing_rows),
            "sample_weight_missing_rows_preview": missing_rows[:20],
            "sample_weight_min": float(min(weights)) if weights else 0.0,
            "sample_weight_mean": float(sum(weights) / max(1, len(weights))),
            "sample_weight_max": float(max(weights)) if weights else 0.0,
            "sample_weight_downweighted_count": int(sum(1 for w in weights if float(w) < 0.999)),
            "sample_weight_zero_count": int(sum(1 for w in weights if float(w) <= 0.0)),
            "sample_weight_class_sums": _class_weight_sums(labels, weights),
        })

        print("[sample_weight_patch]", json.dumps({
            "enabled": True,
            "count": meta["sample_weight_count"],
            "min": meta["sample_weight_min"],
            "mean": meta["sample_weight_mean"],
            "max": meta["sample_weight_max"],
            "downweighted_count": meta["sample_weight_downweighted_count"],
            "missing_rows": meta["sample_weight_missing_rows"],
            "class_sums": meta["sample_weight_class_sums"],
        }, ensure_ascii=False), flush=True)

        return train_ds, val_ds, test_ds, meta

    core.build_datasets = build_datasets_with_labeled_weights


install_labeled_sample_weight_patch(train_core, weight_col="sample_weight")
install_balanced_batch_sampler(train_core)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # Identity
    "project_root": str(PROJECT_ROOT),
    "stage": "stage2",

    # Data
    "train_csv": _p("data", "csv", "clean_v2", "train_v2_weighted.csv"),
    "val_csv": _p("data", "csv", "val.csv"),
    "test_csv": _p("data", "csv", "test.csv"),
    "img_base": None,

    # Sample weight column generated by build_weighted_train_from_review.py
    "sample_weight_col": "sample_weight",

    # No pseudo labels in this diagnostic control.
    "pseudo_csv": None,
    "require_pseudo_conf": False,

    # Start from Stage1, same as previous clean_v2 control.
    "init_ckpt": _p("checkpoints", "best_model_stage1_refactored.pth"),

    # Unique aliases. Do not overwrite the historical best balanced-clean alias.
    "best_alias_name": "best_model_stage2_weighted_v2_control.pth",
    "log_alias_name": "train_stage2_weighted_v2_control_log.csv",
    "write_checkpoint_alias": True,
    "alias_overwrite": True,

    # Output layout
    "runs_dir": _p("runs", "training"),
    "checkpoint_alias_dir": _p("checkpoints"),

    # Model
    "model_variant": "large",
    "num_classes": 7,
    "pretrained": False,
    "compile_model": False,
    "strict_checkpoint_load": True,

    # Training
    "device": "cuda",
    "epochs": 200,
    "batch_size": 126,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
    "drop_last_train": True,

    # Optimizer
    "lr": 3e-4,
    "lr_floor": 1e-6,
    "warmup_epochs": 2,
    "weight_decay": 1e-4,

    # Loss
    # Do not use class weights together with balanced batches and sample weights.
    "label_smoothing": 0.04,
    "use_class_weights": False,
    "class_balance_beta": 0.995,
    "class_weights_from": "labeled_train",

    # Balanced sampler
    "sampling_strategy": "balanced_batch",
    "balanced_per_class": "auto_min",
    "balanced_per_class_source": "labeled_train",
    "balanced_samples_per_class_per_batch": 18,
    "balanced_strict_batch_size": True,
    "balanced_replacement": False,

    # Pseudo fields are intentionally inert here.
    "pseudo_conf_min": 0.82,
    "pseudo_conf_power": 2.0,
    "pseudo_loss_scale": 0.0,
    "pseudo_rampup_epochs": 10,

    # Validation / checkpointing
    "val_interval": 1,
    "early_stop_patience": 20,
    "best_metric": "global_macro_f1",
    "evaluate_test_at_end": True,
    "save_last_every_epoch": True,

    # Reproducibility
    "seed": 42,
    "deterministic_algorithms": False,
    "cudnn_benchmark": False,

    # Stability
    "use_amp": True,
    "grad_clip": True,
    "max_norm": 1.0,

    "run_name": None,

    "notes": (
        "Stage2 Weighted-V2 Control: train_v2_weighted.csv, no pseudo labels, "
        "strict balanced batches, labeled sample_weight enabled. This is a final "
        "one-shot check before moving on from Stage2 cleanup experiments."
    ),
}


REQUIRED_RELATIVE_FILES: Tuple[str, ...] = (
    _p("data", "csv", "clean_v2", "train_v2_weighted.csv"),
    _p("data", "csv", "val.csv"),
    _p("data", "csv", "test.csv"),
    _p("checkpoints", "best_model_stage1_refactored.pth"),
)


def _print_config_audit(keys: Iterable[str]) -> None:
    print("\n[Stage2 WEIGHTED_V2 CONTROL] Config audit:")
    for key in keys:
        print(f"  {key}: {CONFIG.get(key)!r}")


def _preflight() -> None:
    print("\n[Stage2 WEIGHTED_V2 CONTROL] Preflight")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")

    missing: List[str] = []
    for rel in REQUIRED_RELATIVE_FILES:
        full = _resolve_project_path(rel)
        if full is None or not full.exists():
            missing.append(str(full))

    if missing:
        print("\n[Stage2 WEIGHTED_V2 CONTROL] Missing required files:")
        for item in missing:
            print(f"  - {item}")
        raise FileNotFoundError(
            "Required files are missing. First run build_weighted_train_from_review.py, "
            "and check FER_PROJECT_ROOT."
        )

    train_csv = _resolve_project_path(CONFIG["train_csv"])
    assert train_csv is not None
    with train_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
    if CONFIG["sample_weight_col"] not in fields:
        raise ValueError(
            f"{train_csv} does not contain sample_weight column {CONFIG['sample_weight_col']!r}."
        )

    if CONFIG["pseudo_csv"] is not None:
        raise ValueError("weighted_v2_control must have pseudo_csv=None.")

    if CONFIG["use_class_weights"] is not False:
        raise ValueError("weighted_v2_control must have use_class_weights=False.")

    if CONFIG["batch_size"] != 7 * CONFIG["balanced_samples_per_class_per_batch"]:
        raise ValueError(
            "batch_size must equal 7 * balanced_samples_per_class_per_batch. "
            f"Got batch_size={CONFIG['batch_size']} and "
            f"balanced_samples_per_class_per_batch={CONFIG['balanced_samples_per_class_per_batch']}."
        )

    unsafe_aliases = {
        "best_model_stage2_balanced_clean.pth",
        "best_model_stage2_refactored.pth",
        "best_model_stage1_refactored.pth",
    }
    if CONFIG["best_alias_name"] in unsafe_aliases:
        raise ValueError(f"Unsafe alias name: {CONFIG['best_alias_name']}")

    _print_config_audit((
        "train_csv",
        "val_csv",
        "test_csv",
        "sample_weight_col",
        "pseudo_csv",
        "init_ckpt",
        "epochs",
        "batch_size",
        "lr",
        "label_smoothing",
        "use_class_weights",
        "sampling_strategy",
        "balanced_samples_per_class_per_batch",
        "early_stop_patience",
        "best_alias_name",
        "log_alias_name",
    ))


def run() -> None:
    _preflight()
    print("\n[Stage2 WEIGHTED_V2 CONTROL] Starting train_core.main(...)\n")
    train_core.main("stage2", CONFIG)


if __name__ == "__main__":
    run()