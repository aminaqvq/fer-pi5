from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

ACTION_ALIASES = {
    "keep": "keep",
    "保留": "keep",
    "relabel": "relabel",
    "change": "relabel",
    "更换": "relabel",
    "修改": "relabel",
    "改": "relabel",
    "ignore": "ignore",
    "delete": "ignore",
    "remove": "ignore",
    "drop": "ignore",
    "删除": "ignore",
    "丢弃": "ignore",
    "soft": "soft",
    "软标签": "soft",
}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields:
        raise ValueError(f"CSV has no header: {path}")
    return rows, fields


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def find_latest_manual_review(project_root: Path) -> Path:
    audit_root = project_root / "runs" / "audit"
    candidates: List[Path] = []
    if audit_root.exists():
        candidates.extend(audit_root.glob("oof_train_audit_*/manual_label_review.csv"))
        candidates.extend(audit_root.glob("*/manual_label_review.csv"))
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find manual_label_review.csv under runs/audit. "
            "Please pass --review-csv explicitly."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_label_column(fields: Sequence[str]) -> str:
    for name in ("label", "emotion", "target", "class", "class_id", "emotion_id", "mapped_emotion"):
        if name in fields:
            return name
    raise ValueError(f"Could not detect label column from fields: {fields}")


def label_style(rows: Sequence[Mapping[str, Any]], label_col: str) -> str:
    for row in rows:
        value = str(row.get(label_col, "")).strip()
        if not value:
            continue
        if value.lower() in LABEL_TO_ID:
            return "name"
        try:
            int(float(value))
            return "id"
        except Exception:
            return "raw"
    return "id"


def parse_label_id(value: Any) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in LABEL_TO_ID:
        return LABEL_TO_ID[lower]
    try:
        idx = int(float(text))
        if 0 <= idx < len(LABELS):
            return idx
    except Exception:
        pass
    return None


def format_label(label_id: int, style: str) -> str:
    return LABELS[int(label_id)] if style == "name" else str(int(label_id))


def normalize_action(value: Any) -> str:
    text = str(value).strip().lower()
    return ACTION_ALIASES.get(text, "")


def resolve_review_index(row: Mapping[str, Any], train_len: int) -> Optional[int]:
    text = str(row.get("dataset_index", "")).strip()
    if text:
        try:
            idx = int(float(text))
            if 0 <= idx < train_len:
                return idx
        except Exception:
            pass

    text = str(row.get("row_index", "")).strip()
    if text:
        try:
            raw = int(float(text))
            if 0 <= raw < train_len:
                return raw
            line_as_idx = raw - 2
            if 0 <= line_as_idx < train_len:
                return line_as_idx
        except Exception:
            pass
    return None


def ensure_fields(base_fields: Sequence[str], extra_fields: Sequence[str]) -> List[str]:
    out = list(base_fields)
    for field in extra_fields:
        if field not in out:
            out.append(field)
    return out


def class_counts(rows: Sequence[Mapping[str, Any]], label_col: str) -> Dict[str, int]:
    counts = {name: 0 for name in LABELS}
    unknown = 0
    for row in rows:
        idx = parse_label_id(row.get(label_col, ""))
        if idx is None:
            unknown += 1
        else:
            counts[LABELS[idx]] += 1
    if unknown:
        counts["__unknown__"] = unknown
    return counts


def class_weight_sums(rows: Sequence[Mapping[str, Any]], label_col: str, weight_col: str) -> Dict[str, float]:
    sums = {name: 0.0 for name in LABELS}
    unknown = 0.0
    for row in rows:
        try:
            w = float(row.get(weight_col, 1.0))
        except Exception:
            w = 1.0
        idx = parse_label_id(row.get(label_col, ""))
        if idx is None:
            unknown += w
        else:
            sums[LABELS[idx]] += w
    if unknown:
        sums["__unknown__"] = unknown
    return sums


def build_weighted_train(
    *,
    train_csv: Path,
    review_csv: Path,
    output_dir: Path,
    output_name: str,
    report_name: str,
    sample_weight_col: str,
    keep_weight: float,
    relabel_weight: float,
    relabel_same_weight: float,
    ignore_weight: float,
    soft_weight: float,
    strict: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    train_rows, train_fields = read_csv_dicts(train_csv)
    review_rows, _review_fields = read_csv_dicts(review_csv)
    label_col = detect_label_column(train_fields)
    style = label_style(train_rows, label_col)

    output_fields = ensure_fields(train_fields, [
        sample_weight_col,
        "review_action",
        "review_new_label",
        "review_old_label",
        "review_pred_label",
        "review_issue_type",
        "review_issue_rank",
        "review_confidence",
        "review_p_true",
        "review_margin",
        "review_reason",
        "review_source",
        "soft_label_json",
    ])

    reviews_by_index: Dict[int, Dict[str, str]] = {}
    duplicate_indices: Counter[int] = Counter()
    invalid_reviews: List[Dict[str, Any]] = []

    for review_pos, review in enumerate(review_rows):
        action = normalize_action(review.get("review_action", ""))
        if not action:
            continue
        idx = resolve_review_index(review, len(train_rows))
        if idx is None:
            invalid_reviews.append({
                "review_csv_row": review_pos + 2,
                "reason": "cannot_resolve_dataset_index",
                "review": review,
            })
            continue
        duplicate_indices[idx] += 1
        review = dict(review)
        review["__normalized_action"] = action
        review["__review_csv_row"] = str(review_pos + 2)
        reviews_by_index[idx] = review

    weighted_rows: List[Dict[str, Any]] = []
    relabeled_rows: List[Dict[str, Any]] = []
    downweighted_rows: List[Dict[str, Any]] = []
    soft_rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    action_counts: Counter[str] = Counter()
    effective_action_counts: Counter[str] = Counter()
    label_changes: Counter[str] = Counter()

    for idx, original in enumerate(train_rows):
        row: Dict[str, Any] = dict(original)
        old_label_id = parse_label_id(row.get(label_col, ""))
        old_label_name = LABELS[old_label_id] if old_label_id is not None else str(row.get(label_col, ""))

        row[sample_weight_col] = f"{1.0:.6f}"
        row["review_action"] = ""
        row["review_new_label"] = ""
        row["review_old_label"] = old_label_name
        row["review_pred_label"] = ""
        row["review_issue_type"] = ""
        row["review_issue_rank"] = ""
        row["review_confidence"] = ""
        row["review_p_true"] = ""
        row["review_margin"] = ""
        row["review_reason"] = ""
        row["review_source"] = ""
        row["soft_label_json"] = ""

        review = reviews_by_index.get(idx)
        if review is None:
            weighted_rows.append(row)
            continue

        action = str(review["__normalized_action"])
        action_counts[action] += 1

        row["review_action"] = action
        row["review_pred_label"] = str(review.get("pred_label", ""))
        row["review_issue_type"] = str(review.get("issue_type", ""))
        row["review_issue_rank"] = str(review.get("issue_rank", ""))
        row["review_confidence"] = str(review.get("confidence", ""))
        row["review_p_true"] = str(review.get("p_true", ""))
        row["review_margin"] = str(review.get("margin", ""))
        row["review_reason"] = str(review.get("review_reason", ""))
        row["review_source"] = str(review_csv)
        row["soft_label_json"] = str(review.get("soft_label_json", ""))

        if action == "keep":
            row[sample_weight_col] = f"{float(keep_weight):.6f}"
            effective_action_counts["keep"] += 1
            weighted_rows.append(row)
            continue

        if action == "ignore":
            row[sample_weight_col] = f"{float(ignore_weight):.6f}"
            effective_action_counts["downweight_ignore"] += 1
            downweighted_rows.append(dict(row))
            weighted_rows.append(row)
            continue

        if action == "soft":
            row[sample_weight_col] = f"{float(soft_weight):.6f}"
            effective_action_counts["downweight_soft"] += 1
            soft_rows.append(dict(row))
            weighted_rows.append(row)
            continue

        if action == "relabel":
            new_label_id = parse_label_id(review.get("new_label_id", ""))
            if new_label_id is None:
                new_label_id = parse_label_id(review.get("new_label", ""))

            if new_label_id is None:
                warning = {
                    "dataset_index": idx,
                    "reason": "relabel_missing_or_invalid_new_label",
                    "review_csv_row": review.get("__review_csv_row", ""),
                    "new_label": review.get("new_label", ""),
                    "new_label_id": review.get("new_label_id", ""),
                }
                warnings.append(warning)
                if strict:
                    raise ValueError(json.dumps(warning, ensure_ascii=False))
                row[sample_weight_col] = f"{float(keep_weight):.6f}"
                effective_action_counts["invalid_relabel_kept"] += 1
                weighted_rows.append(row)
                continue

            new_label_name = LABELS[new_label_id]
            row["review_new_label"] = new_label_name

            if old_label_id == new_label_id:
                row[sample_weight_col] = f"{float(relabel_same_weight):.6f}"
                effective_action_counts["relabel_same_as_keep"] += 1
                weighted_rows.append(row)
                continue

            row[label_col] = format_label(new_label_id, style)
            row[sample_weight_col] = f"{float(relabel_weight):.6f}"
            label_changes[f"{old_label_name}->{new_label_name}"] += 1
            effective_action_counts["relabel_changed"] += 1
            relabeled_rows.append(dict(row))
            weighted_rows.append(row)
            continue

        warning = {
            "dataset_index": idx,
            "reason": f"unsupported_action={action}",
            "review_csv_row": review.get("__review_csv_row", ""),
        }
        warnings.append(warning)
        if strict:
            raise ValueError(json.dumps(warning, ensure_ascii=False))
        weighted_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    weighted_csv = output_dir / output_name
    relabeled_csv = output_dir / "train_v2_weighted_relabeled.csv"
    downweighted_csv = output_dir / "train_v2_weighted_downweighted.csv"
    soft_csv = output_dir / "train_v2_weighted_soft.csv"
    report_json = output_dir / report_name

    report = {
        "status": "dry_run" if dry_run else "finished",
        "created_at": now_stamp(),
        "method": "manual_review_to_sample_weighted_train",
        "train_csv": str(train_csv),
        "review_csv": str(review_csv),
        "weighted_csv": str(weighted_csv),
        "label_column": label_col,
        "label_style": style,
        "sample_weight_col": sample_weight_col,
        "weights": {
            "keep_weight": float(keep_weight),
            "relabel_weight": float(relabel_weight),
            "relabel_same_weight": float(relabel_same_weight),
            "ignore_weight": float(ignore_weight),
            "soft_weight": float(soft_weight),
        },
        "input_train_rows": len(train_rows),
        "output_weighted_rows": len(weighted_rows),
        "review_rows_total": len(review_rows),
        "review_rows_valid_with_action": len(reviews_by_index),
        "invalid_review_count": len(invalid_reviews),
        "warning_count": len(warnings),
        "duplicate_review_indices": {str(k): int(v) for k, v in duplicate_indices.items() if v > 1},
        "action_counts_raw": dict(action_counts),
        "effective_action_counts": dict(effective_action_counts),
        "label_changes": dict(label_changes),
        "class_counts_before": class_counts(train_rows, label_col),
        "class_counts_after_hard_labels": class_counts(weighted_rows, label_col),
        "class_weight_sums_after": class_weight_sums(weighted_rows, label_col, sample_weight_col),
        "relabeled_rows": len(relabeled_rows),
        "downweighted_rows": len(downweighted_rows),
        "soft_rows": len(soft_rows),
        "invalid_reviews_preview": invalid_reviews[:20],
        "warnings_preview": warnings[:20],
        "outputs": {
            "weighted_csv": str(weighted_csv),
            "relabeled_csv": str(relabeled_csv),
            "downweighted_csv": str(downweighted_csv),
            "soft_csv": str(soft_csv),
            "report_json": str(report_json),
        },
        "next_step": (
            "Run train_stage2_weighted_v2_control.py once. "
            "If it does not beat historical best, stop minor data-cleaning experiments and proceed."
        ),
    }

    if not dry_run:
        write_csv_dicts(weighted_csv, weighted_rows, output_fields)
        write_csv_dicts(relabeled_csv, relabeled_rows, output_fields)
        write_csv_dicts(downweighted_csv, downweighted_rows, output_fields)
        write_csv_dicts(soft_csv, soft_rows, output_fields)
        write_json(report_json, report)

    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample-weighted FER train CSV from manual label review.")
    parser.add_argument("--project-root", type=str, default=os.environ.get("FER_PROJECT_ROOT", r"/"))
    parser.add_argument("--train-csv", type=str, default=r"data\csv\train.csv")
    parser.add_argument("--review-csv", type=str, default=None, help="manual_label_review.csv. Default: latest under runs/audit.")
    parser.add_argument("--output-dir", type=str, default=r"data\csv\clean_v2")
    parser.add_argument("--output-name", type=str, default="train_v2_weighted.csv")
    parser.add_argument("--report-name", type=str, default="train_v2_weighted_report.json")
    parser.add_argument("--sample-weight-col", type=str, default="sample_weight")
    parser.add_argument("--keep-weight", type=float, default=1.0)
    parser.add_argument("--relabel-weight", type=float, default=0.8)
    parser.add_argument("--relabel-same-weight", type=float, default=1.0)
    parser.add_argument("--ignore-weight", type=float, default=0.25)
    parser.add_argument("--soft-weight", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    train_csv = resolve_path(project_root, args.train_csv)
    review_csv = resolve_path(project_root, args.review_csv) if args.review_csv else find_latest_manual_review(project_root)
    output_dir = resolve_path(project_root, args.output_dir)
    assert train_csv is not None and review_csv is not None and output_dir is not None

    report = build_weighted_train(
        train_csv=train_csv,
        review_csv=review_csv,
        output_dir=output_dir,
        output_name=str(args.output_name),
        report_name=str(args.report_name),
        sample_weight_col=str(args.sample_weight_col),
        keep_weight=float(args.keep_weight),
        relabel_weight=float(args.relabel_weight),
        relabel_same_weight=float(args.relabel_same_weight),
        ignore_weight=float(args.ignore_weight),
        soft_weight=float(args.soft_weight),
        strict=bool(args.strict),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()