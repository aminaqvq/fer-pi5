from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_PROJECT_ROOT = os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "NULL", "null"}


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] failed to read json: {path} ({type(exc).__name__}: {exc})", flush=True)
    return {}


def read_jsonl_last(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    last = text
        return json.loads(last) if last else {}
    except Exception as exc:
        print(f"[warn] failed to read jsonl last row: {path} ({type(exc).__name__}: {exc})", flush=True)
        return {}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        print(f"[warn] failed to read csv: {path} ({type(exc).__name__}: {exc})", flush=True)
        return []


def try_float(value: Any) -> Optional[float]:
    if is_none_like(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def try_int(value: Any) -> Optional[int]:
    if is_none_like(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def get_nested(obj: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and not is_none_like(value):
            return value
    return None


def safe_rel(path: Optional[Path], root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def sha256_file(path: Path, max_mb: Optional[int] = None) -> str:
    if not path.exists() or not path.is_file():
        return ""
    limit_bytes = None if max_mb is None or max_mb <= 0 else int(max_mb) * 1024 * 1024
    h = hashlib.sha256()
    read_total = 0
    with path.open("rb") as f:
        while True:
            if limit_bytes is not None and read_total >= limit_bytes:
                break
            chunk_size = 1024 * 1024
            if limit_bytes is not None:
                chunk_size = min(chunk_size, limit_bytes - read_total)
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            read_total += len(chunk)
    suffix = "" if limit_bytes is None else f"_first_{max_mb}MB"
    return h.hexdigest() + suffix


def looks_like_run_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = [
        "manifest.json",
        "final_summary.json",
        "resolved_config.json",
        "metrics_epoch.csv",
        "metrics_epoch.jsonl",
        "sampler_audit.json",
    ]
    return any((path / name).exists() for name in markers)


def discover_run_dirs(runs_roots: Sequence[Path], max_depth: int = 4) -> List[Path]:
    found: List[Path] = []
    seen = set()

    for root in runs_roots:
        if not root.exists():
            print(f"[warn] runs root not found: {root}", flush=True)
            continue

        if looks_like_run_dir(root):
            key = str(root.resolve())
            if key not in seen:
                found.append(root)
                seen.add(key)

        stack: List[Tuple[Path, int]] = [(root, 0)]
        while stack:
            cur, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                children = list(cur.iterdir())
            except Exception:
                continue

            for child in children:
                if not child.is_dir():
                    continue
                if looks_like_run_dir(child):
                    key = str(child.resolve())
                    if key not in seen:
                        found.append(child)
                        seen.add(key)
                else:
                    stack.append((child, depth + 1))

    return sorted(found, key=lambda p: str(p).lower())


def metric_from_payload(payload: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    metric_obj = first_present(
        payload.get(f"final_{prefix}"),
        payload.get(f"{prefix}_best"),
        payload.get(prefix),
        payload.get(f"{prefix}_metrics"),
    )

    if not isinstance(metric_obj, Mapping):
        metric_obj = {}

    macro = first_present(
        payload.get(f"best_{prefix}_global_macro_f1"),
        payload.get(f"best_{prefix}_macro_f1"),
        payload.get(f"{prefix}_global_macro_f1"),
        payload.get(f"{prefix}_macro_f1"),
        metric_obj.get("global_macro_f1"),
        metric_obj.get("macro_f1"),
    )

    acc = first_present(
        payload.get(f"{prefix}_acc"),
        payload.get(f"{prefix}_accuracy"),
        metric_obj.get("accuracy"),
    )

    loss = first_present(
        payload.get(f"{prefix}_loss"),
        metric_obj.get("loss"),
    )

    return {
        f"{prefix}_macro_f1": try_float(macro),
        f"{prefix}_accuracy": try_float(acc),
        f"{prefix}_loss": try_float(loss),
    }


def metrics_from_csv(metrics_csv: Path) -> Dict[str, Any]:
    rows = read_csv_rows(metrics_csv)
    if not rows:
        return {}

    def row_score(row: Mapping[str, Any]) -> float:
        for key in (
            "val_global_macro_f1",
            "val_macro_f1",
            "val_f1",
            "best_val_global_macro_f1",
            "best_val_macro_f1",
        ):
            v = try_float(row.get(key))
            if v is not None:
                return v
        return -1.0

    best_row = max(rows, key=row_score)
    last_row = rows[-1]

    out: Dict[str, Any] = {
        "csv_epoch_count": len(rows),
        "csv_best_epoch": try_int(first_present(best_row.get("epoch"), best_row.get("Epoch"))),
        "csv_last_epoch": try_int(first_present(last_row.get("epoch"), last_row.get("Epoch"))),
    }

    for prefix in ("train", "val", "test"):
        macro = first_present(
            best_row.get(f"{prefix}_global_macro_f1"),
            best_row.get(f"{prefix}_macro_f1"),
            best_row.get(f"{prefix}_f1"),
        )
        acc = first_present(best_row.get(f"{prefix}_acc"), best_row.get(f"{prefix}_accuracy"))
        loss = first_present(best_row.get(f"{prefix}_loss"), best_row.get(f"{prefix}_Loss"))
        out[f"csv_best_{prefix}_macro_f1"] = try_float(macro)
        out[f"csv_best_{prefix}_accuracy"] = try_float(acc)
        out[f"csv_best_{prefix}_loss"] = try_float(loss)

    return out


def read_eval_metric(run_dir: Path, name: str) -> Dict[str, Any]:
    candidates = [
        run_dir / "evaluation" / f"{name}_best_reloaded_metrics.json",
        run_dir / "evaluation" / f"{name}_best_metrics.json",
        run_dir / "evaluation" / f"{name}_metrics.json",
    ]
    for path in candidates:
        payload = read_json(path)
        if payload:
            return payload
    return {}


def find_checkpoint_candidates(run_dir: Path, manifest: Mapping[str, Any], project_root: Path) -> List[Path]:
    candidates: List[Path] = []

    for key in ("best_checkpoint", "checkpoint", "best_model", "best_path"):
        value = manifest.get(key)
        p = resolve_path(project_root, value)
        if p is not None:
            candidates.append(p)

    local_candidates = [
        run_dir / "checkpoints" / "best_model.pth",
        run_dir / "best_model.pth",
        run_dir / "checkpoint_best.pth",
        run_dir / "model_best.pth",
    ]
    candidates.extend(local_candidates)

    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def parse_run(run_dir: Path, project_root: Path, *, hash_mode: str = "none") -> Dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    final_summary = read_json(run_dir / "final_summary.json")
    resolved_config = read_json(run_dir / "resolved_config.json")
    sampler_audit = read_json(run_dir / "sampler_audit.json")
    jsonl_last = read_jsonl_last(run_dir / "metrics_epoch.jsonl")
    csv_metrics = metrics_from_csv(run_dir / "metrics_epoch.csv")
    val_eval = read_eval_metric(run_dir, "val")
    test_eval = read_eval_metric(run_dir, "test")

    payload: Dict[str, Any] = {}
    for src in (resolved_config, manifest.get("config", {}) if isinstance(manifest.get("config"), Mapping) else {}, final_summary, manifest):
        if isinstance(src, Mapping):
            payload.update(src)

    cfg = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else resolved_config
    if not isinstance(cfg, Mapping):
        cfg = {}

    run_id = first_present(manifest.get("run_id"), final_summary.get("run_id"), run_dir.name)
    status = first_present(manifest.get("status"), final_summary.get("status"), "unknown")

    val_metrics = metric_from_payload(
        {**manifest, **final_summary, "final_val": first_present(final_summary.get("final_val"), manifest.get("final_val"), val_eval)},
        "val",
    )
    test_metrics = metric_from_payload(
        {**manifest, **final_summary, "final_test": first_present(final_summary.get("final_test"), manifest.get("final_test"), test_eval)},
        "test",
    )

    if val_metrics["val_macro_f1"] is None:
        val_metrics["val_macro_f1"] = try_float(first_present(val_eval.get("global_macro_f1"), csv_metrics.get("csv_best_val_macro_f1")))
    if test_metrics["test_macro_f1"] is None:
        test_metrics["test_macro_f1"] = try_float(first_present(test_eval.get("global_macro_f1"), csv_metrics.get("csv_best_test_macro_f1")))

    if val_metrics["val_accuracy"] is None:
        val_metrics["val_accuracy"] = try_float(first_present(val_eval.get("accuracy"), csv_metrics.get("csv_best_val_accuracy")))
    if test_metrics["test_accuracy"] is None:
        test_metrics["test_accuracy"] = try_float(first_present(test_eval.get("accuracy"), csv_metrics.get("csv_best_test_accuracy")))

    best_epoch = try_int(first_present(
        final_summary.get("best_epoch"),
        manifest.get("best_epoch"),
        csv_metrics.get("csv_best_epoch"),
        get_nested(jsonl_last, ["best_epoch"]),
        get_nested(jsonl_last, ["epoch"]),
    ))

    best_metric = try_float(first_present(
        final_summary.get("best_metric"),
        manifest.get("best_metric"),
        final_summary.get("best_val_global_macro_f1"),
        manifest.get("best_val_global_macro_f1"),
        val_metrics.get("val_macro_f1"),
    ))

    ckpt_candidates = find_checkpoint_candidates(run_dir, manifest, project_root)
    existing_ckpts = [p for p in ckpt_candidates if p.exists()]
    best_ckpt = existing_ckpts[0] if existing_ckpts else (ckpt_candidates[0] if ckpt_candidates else None)

    ckpt_size = ""
    ckpt_mtime = ""
    ckpt_sha = ""
    if best_ckpt is not None and best_ckpt.exists():
        stat = best_ckpt.stat()
        ckpt_size = str(stat.st_size)
        ckpt_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        if hash_mode == "full":
            ckpt_sha = sha256_file(best_ckpt, max_mb=None)
        elif hash_mode.startswith("first"):
            m = re.match(r"first(\d+)", hash_mode)
            max_mb = int(m.group(1)) if m else 64
            ckpt_sha = sha256_file(best_ckpt, max_mb=max_mb)

    dataset_meta = manifest.get("dataset_meta") if isinstance(manifest.get("dataset_meta"), Mapping) else {}
    data_meta = manifest.get("data") if isinstance(manifest.get("data"), Mapping) else {}
    train_counts = first_present(
        dataset_meta.get("train_class_counts") if isinstance(dataset_meta, Mapping) else None,
        data_meta.get("train_class_counts") if isinstance(data_meta, Mapping) else None,
        sampler_audit.get("class_counts") if isinstance(sampler_audit, Mapping) else None,
        sampler_audit.get("source_class_counts") if isinstance(sampler_audit, Mapping) else None,
    )

    logit_best = final_summary.get("logit_adjustment_best_by_val_macro_f1")
    if not isinstance(logit_best, Mapping):
        sweep = read_json(run_dir / "evaluation" / "logit_adjustment_sweep.json")
        logit_best = sweep.get("best_by_val_macro_f1") if isinstance(sweep.get("best_by_val_macro_f1"), Mapping) else {}

    row = {
        "run_id": str(run_id),
        "status": str(status),
        "run_dir": str(run_dir),
        "run_dir_rel": safe_rel(run_dir, project_root),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "val_macro_f1": val_metrics["val_macro_f1"],
        "test_macro_f1": test_metrics["test_macro_f1"],
        "val_accuracy": val_metrics["val_accuracy"],
        "test_accuracy": test_metrics["test_accuracy"],
        "val_loss": val_metrics["val_loss"],
        "test_loss": test_metrics["test_loss"],
        "csv_epoch_count": csv_metrics.get("csv_epoch_count"),
        "csv_last_epoch": csv_metrics.get("csv_last_epoch"),
        "checkpoint_path": str(best_ckpt) if best_ckpt is not None else "",
        "checkpoint_path_rel": safe_rel(best_ckpt, project_root) if best_ckpt is not None else "",
        "checkpoint_exists": bool(best_ckpt is not None and best_ckpt.exists()),
        "checkpoint_size_bytes": ckpt_size,
        "checkpoint_mtime": ckpt_mtime,
        "checkpoint_sha256": ckpt_sha,
        "best_alias": str(first_present(manifest.get("best_alias"), final_summary.get("best_alias"), "")),
        "log_alias": str(first_present(manifest.get("log_alias"), final_summary.get("log_alias"), "")),
        "project_root_in_config": str(cfg.get("project_root", "")),
        "train_csv": str(cfg.get("train_csv", "")),
        "val_csv": str(cfg.get("val_csv", "")),
        "test_csv": str(cfg.get("test_csv", "")),
        "pseudo_csv": str(cfg.get("pseudo_csv", "")),
        "init_ckpt": str(first_present(cfg.get("init_ckpt"), cfg.get("init_checkpoint"), cfg.get("init_ckpt_candidates"), "")),
        "model_variant": str(cfg.get("model_variant", "")),
        "batch_size": cfg.get("batch_size", ""),
        "epochs": cfg.get("epochs", ""),
        "lr": cfg.get("lr", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "seed": cfg.get("seed", ""),
        "sampling_strategy": str(cfg.get("sampling_strategy", "")),
        "balanced_per_class": str(cfg.get("balanced_per_class", "")),
        "balanced_samples_per_class_per_batch": str(cfg.get("balanced_samples_per_class_per_batch", "")),
        "balanced_sampler_samples_per_epoch": sampler_audit.get("samples_per_epoch", "") if isinstance(sampler_audit, Mapping) else "",
        "balanced_sampler_per_class_per_epoch": sampler_audit.get("per_class_per_epoch", "") if isinstance(sampler_audit, Mapping) else "",
        "use_class_weights": str(cfg.get("use_class_weights", "")),
        "pseudo_loss_scale": str(cfg.get("pseudo_loss_scale", "")),
        "pseudo_conf_min": str(cfg.get("pseudo_conf_min", "")),
        "logit_best_tau": logit_best.get("tau", "") if isinstance(logit_best, Mapping) else "",
        "logit_best_val_macro_f1": logit_best.get("val_global_macro_f1", "") if isinstance(logit_best, Mapping) else "",
        "logit_best_test_macro_f1": logit_best.get("test_global_macro_f1", "") if isinstance(logit_best, Mapping) else "",
        "train_class_counts_json": json.dumps(train_counts, ensure_ascii=False) if train_counts else "",
    }
    return row


FIELDNAMES = [
    "rank_by_val",
    "rank_by_test",
    "run_id",
    "status",
    "run_dir",
    "run_dir_rel",
    "best_epoch",
    "best_metric",
    "val_macro_f1",
    "test_macro_f1",
    "val_accuracy",
    "test_accuracy",
    "val_loss",
    "test_loss",
    "csv_epoch_count",
    "csv_last_epoch",
    "checkpoint_path",
    "checkpoint_path_rel",
    "checkpoint_exists",
    "checkpoint_size_bytes",
    "checkpoint_mtime",
    "checkpoint_sha256",
    "best_alias",
    "log_alias",
    "project_root_in_config",
    "train_csv",
    "val_csv",
    "test_csv",
    "pseudo_csv",
    "init_ckpt",
    "model_variant",
    "batch_size",
    "epochs",
    "lr",
    "weight_decay",
    "seed",
    "sampling_strategy",
    "balanced_per_class",
    "balanced_samples_per_class_per_batch",
    "balanced_sampler_samples_per_epoch",
    "balanced_sampler_per_class_per_epoch",
    "use_class_weights",
    "pseudo_loss_scale",
    "pseudo_conf_min",
    "logit_best_tau",
    "logit_best_val_macro_f1",
    "logit_best_test_macro_f1",
    "train_class_counts_json",
]


def sort_and_rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def val_key(row: Mapping[str, Any]) -> float:
        v = try_float(row.get("val_macro_f1"))
        return -999.0 if v is None else v

    def test_key(row: Mapping[str, Any]) -> float:
        v = try_float(row.get("test_macro_f1"))
        return -999.0 if v is None else v

    rows_sorted = sorted(rows, key=lambda r: (val_key(r), test_key(r)), reverse=True)

    test_sorted_ids = {
        id(row): rank + 1
        for rank, row in enumerate(sorted(rows, key=lambda r: (test_key(r), val_key(r)), reverse=True))
    }

    for i, row in enumerate(rows_sorted, start=1):
        row["rank_by_val"] = i
        row["rank_by_test"] = test_sorted_ids[id(row)]

    return rows_sorted


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a registry CSV/JSON for FER training checkpoints and runs."
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT, help="Project root, e.g. F:\\fer-pi5")
    parser.add_argument(
        "--runs-dir",
        action="append",
        default=None,
        help="Run directory to scan. Can be passed multiple times. Default: <root>/runs/training",
    )
    parser.add_argument("--output-csv", default=None, help="Output registry CSV path.")
    parser.add_argument("--output-json", default=None, help="Output registry JSON path.")
    parser.add_argument("--max-depth", type=int, default=4, help="Max recursion depth under each runs dir.")
    parser.add_argument(
        "--hash",
        choices=["none", "first16", "first64", "full"],
        default="none",
        help="Optional checkpoint hashing. full can be slow for large .pth files.",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()

    if args.runs_dir:
        runs_roots = [resolve_path(project_root, item) for item in args.runs_dir]
        runs_roots = [p for p in runs_roots if p is not None]
    else:
        runs_roots = [project_root / "runs" / "training"]

    output_csv = Path(args.output_csv) if args.output_csv else project_root / "checkpoints" / "checkpoint_registry.csv"
    output_json = Path(args.output_json) if args.output_json else project_root / "checkpoints" / "checkpoint_registry.json"

    run_dirs = discover_run_dirs(runs_roots, max_depth=int(args.max_depth))
    rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        try:
            rows.append(parse_run(run_dir, project_root, hash_mode=str(args.hash)))
        except Exception as exc:
            print(f"[warn] failed to parse run: {run_dir} ({type(exc).__name__}: {exc})", flush=True)

    rows = sort_and_rank(rows)
    write_csv(output_csv, rows)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version,
        "project_root": str(project_root),
        "runs_roots": [str(p) for p in runs_roots],
        "run_count": len(rows),
        "output_csv": str(output_csv),
        "rows": rows,
    }
    write_json(output_json, payload)

    print("=== checkpoint registry complete ===", flush=True)
    print(f"project_root: {project_root}", flush=True)
    print(f"run_count   : {len(rows)}", flush=True)
    print(f"csv         : {output_csv}", flush=True)
    print(f"json        : {output_json}", flush=True)

    if rows:
        best_val = rows[0]
        best_test = sorted(rows, key=lambda r: try_float(r.get("test_macro_f1")) or -999.0, reverse=True)[0]
        print(
            f"best by val : {best_val['run_id']} val={best_val.get('val_macro_f1')} test={best_val.get('test_macro_f1')}",
            flush=True,
        )
        print(
            f"best by test: {best_test['run_id']} val={best_test.get('val_macro_f1')} test={best_test.get('test_macro_f1')}",
            flush=True,
        )

    return payload


if __name__ == "__main__":
    main()