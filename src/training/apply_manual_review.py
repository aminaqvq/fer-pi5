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

REVIEW_ACTION_ALIASES = {
    "keep": "keep",
    "保留": "keep",
    "relabel": "relabel",
    "change": "relabel",
    "更换": "relabel",
    "改": "relabel",
    "ignore": "ignore",
    "delete": "ignore",
    "remove": "ignore",
    "drop": "ignore",
    "删除": "ignore",
    "soft": "soft",
}

REVIEW_META_COLUMNS = [
    "review_action", "new_label", "new_label_id", "soft_label_json",
    "review_reason", "reviewer", "reviewed_at",
]


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return rows, fields


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    tmp.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def find_latest_manual_review(project_root: Path) -> Path:
    audit_root = project_root / "runs" / "audit"
    candidates = list(audit_root.glob("oof_train_audit_*/manual_label_review.csv")) if audit_root.exists() else []
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        raise FileNotFoundError("Could not auto-find runs/audit/oof_train_audit_*/manual_label_review.csv. Use --review-csv.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def detect_label_column(fields: Sequence[str]) -> str:
    candidates = ["label", "emotion", "target", "class", "class_id", "emotion_id"]
    for c in candidates:
        if c in fields:
            return c
    raise ValueError(f"Could not detect label column. CSV fields={fields}")


def label_value_style(train_rows: Sequence[Mapping[str, str]], label_col: str) -> str:
    for row in train_rows:
        value = str(row.get(label_col, "")).strip()
        if value == "":
            continue
        low = value.lower()
        if low in LABEL_TO_ID:
            return "name"
        try:
            int(float(value))
            return "id"
        except Exception:
            return "raw"
    return "id"


def parse_label_id(value: Any) -> Optional[int]:
    text = str(value).strip()
    if text == "":
        return None
    low = text.lower()
    if low in LABEL_TO_ID:
        return LABEL_TO_ID[low]
    try:
        i = int(float(text))
        if 0 <= i < len(LABELS):
            return i
    except Exception:
        pass
    return None


def format_label(label_id: int, style: str) -> str:
    if style == "name":
        return LABELS[int(label_id)]
    return str(int(label_id))


def normalize_action(value: Any) -> str:
    text = str(value).strip().lower()
    return REVIEW_ACTION_ALIASES.get(text, "")


def resolve_review_index(row: Mapping[str, Any], train_len: int) -> Optional[int]:
    for col in ("dataset_index", "row_index"):
        text = str(row.get(col, "")).strip()
        if text == "":
            continue
        try:
            idx = int(float(text))
        except Exception:
            continue
        if 0 <= idx < train_len:
            return idx
    return None


def merged_removed_row(train_row: Mapping[str, Any], review_row: Mapping[str, Any], label_col: str) -> Dict[str, Any]:
    out = dict(train_row)
    out["__old_label"] = train_row.get(label_col, "")
    for col in REVIEW_META_COLUMNS:
        out[f"__{col}"] = review_row.get(col, "")
    for col in ["dataset_index", "row_index", "true_label", "pred_label", "confidence", "p_true", "margin", "loss", "issue_score", "issue_rank"]:
        if col in review_row:
            out[f"__{col}"] = review_row.get(col, "")
    return out


def build_output_fields(train_fields: Sequence[str], extra_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    out = list(train_fields)
    for row in extra_rows:
        for key in row.keys():
            if key not in out:
                out.append(key)
    return out


def apply_manual_review(
    *,
    train_csv: Path,
    review_csv: Path,
    output_dir: Path,
    output_prefix: str,
    soft_policy: str,
    strict: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not review_csv.exists():
        raise FileNotFoundError(f"review_csv not found: {review_csv}")

    train_rows, train_fields = read_csv_dicts(train_csv)
    review_rows, review_fields = read_csv_dicts(review_csv)
    label_col = detect_label_column(train_fields)
    style = label_value_style(train_rows, label_col)

    reviews_by_idx: Dict[int, Dict[str, str]] = {}
    invalid_reviews: List[Dict[str, Any]] = []
    duplicate_counter: Counter[int] = Counter()

    for rpos, review in enumerate(review_rows):
        action = normalize_action(review.get("review_action", ""))
        if not action:
            continue
        idx = resolve_review_index(review, len(train_rows))
        if idx is None:
            invalid_reviews.append({"review_row": rpos, "reason": "cannot_resolve_dataset_index", "row": review})
            continue
        duplicate_counter[idx] += 1
        review = dict(review)
        review["__normalized_action"] = action
        review["__review_csv_row"] = str(rpos)
        reviews_by_idx[idx] = review

    clean_rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    soft_rows: List[Dict[str, Any]] = []
    relabeled_rows: List[Dict[str, Any]] = []

    action_counts: Counter[str] = Counter()
    label_changes: Counter[str] = Counter()
    warnings: List[Dict[str, Any]] = []

    for idx, original in enumerate(train_rows):
        row = dict(original)
        review = reviews_by_idx.get(idx)
        if review is None:
            clean_rows.append(row)
            continue

        action = str(review["__normalized_action"])
        action_counts[action] += 1

        old_label_id = parse_label_id(row.get(label_col, ""))
        old_label_name = LABELS[old_label_id] if old_label_id is not None else str(row.get(label_col, ""))

        if action == "keep":
            clean_rows.append(row)
            continue

        if action == "ignore":
            removed_rows.append(merged_removed_row(row, review, label_col))
            continue

        if action == "relabel":
            new_id = parse_label_id(review.get("new_label_id", ""))
            if new_id is None:
                new_id = parse_label_id(review.get("new_label", ""))
            if new_id is None:
                warning = {"dataset_index": idx, "reason": "relabel_missing_or_invalid_new_label", "review": review}
                warnings.append(warning)
                if strict:
                    raise ValueError(json.dumps(warning, ensure_ascii=False))
                clean_rows.append(row)
                continue

            row[label_col] = format_label(new_id, style)
            new_label_name = LABELS[new_id]
            label_changes[f"{old_label_name}->{new_label_name}"] += 1

            relabeled_meta = merged_removed_row(original, review, label_col)
            relabeled_meta["__new_label"] = new_label_name
            relabeled_meta["__new_label_id"] = int(new_id)
            relabeled_rows.append(relabeled_meta)
            clean_rows.append(row)
            continue

        if action == "soft":
            soft_row = merged_removed_row(row, review, label_col)
            soft_rows.append(soft_row)
            if soft_policy == "remove":
                removed_rows.append(soft_row)
            elif soft_policy == "keep":
                clean_rows.append(row)
            else:
                warning = {"dataset_index": idx, "reason": f"unknown_soft_policy={soft_policy}"}
                warnings.append(warning)
                if strict:
                    raise ValueError(json.dumps(warning, ensure_ascii=False))
                clean_rows.append(row)
            continue

        warning = {"dataset_index": idx, "reason": f"unsupported_action={action}", "review": review}
        warnings.append(warning)
        if strict:
            raise ValueError(json.dumps(warning, ensure_ascii=False))
        clean_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = output_dir / f"{output_prefix}_clean.csv"
    removed_csv = output_dir / f"{output_prefix}_removed.csv"
    relabeled_csv = output_dir / f"{output_prefix}_relabeled.csv"
    soft_csv = output_dir / f"{output_prefix}_soft.csv"
    report_json = output_dir / f"{output_prefix}_apply_report.json"

    removed_fields = build_output_fields(train_fields, removed_rows + relabeled_rows + soft_rows)
    relabeled_fields = build_output_fields(train_fields, relabeled_rows)
    soft_fields = build_output_fields(train_fields, soft_rows)

    report = {
        "status": "dry_run" if dry_run else "finished",
        "created_at": now_stamp(),
        "train_csv": str(train_csv),
        "review_csv": str(review_csv),
        "output_dir": str(output_dir),
        "label_column": label_col,
        "label_style": style,
        "soft_policy": soft_policy,
        "total_train_rows": len(train_rows),
        "review_rows_total": len(review_rows),
        "review_rows_nonempty_valid": len(reviews_by_idx),
        "duplicate_review_indices": {str(k): v for k, v in duplicate_counter.items() if v > 1},
        "action_counts": dict(action_counts),
        "label_changes": dict(label_changes),
        "clean_rows": len(clean_rows),
        "removed_rows": len(removed_rows),
        "relabeled_rows": len(relabeled_rows),
        "soft_rows": len(soft_rows),
        "invalid_review_count": len(invalid_reviews),
        "warning_count": len(warnings),
        "invalid_reviews_preview": invalid_reviews[:20],
        "warnings_preview": warnings[:20],
        "outputs": {
            "clean_csv": str(clean_csv),
            "removed_csv": str(removed_csv),
            "relabeled_csv": str(relabeled_csv),
            "soft_csv": str(soft_csv),
            "report_json": str(report_json),
        },
    }

    if not dry_run:
        write_csv_dicts(clean_csv, clean_rows, train_fields)
        write_csv_dicts(removed_csv, removed_rows, removed_fields)
        write_csv_dicts(relabeled_csv, relabeled_rows, relabeled_fields)
        write_csv_dicts(soft_csv, soft_rows, soft_fields)
        write_json(report_json, report)

    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply manual FER label review to train.csv")
    parser.add_argument("--project-root", type=str, default=os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"))
    parser.add_argument("--train-csv", type=str, default=r"data\csv\train.csv")
    parser.add_argument("--review-csv", type=str, default=None, help="manual_label_review.csv. Default: latest under runs/audit")
    parser.add_argument("--output-dir", type=str, default=r"data\csv\clean_v2")
    parser.add_argument("--output-prefix", type=str, default="train_v2")
    parser.add_argument("--soft-policy", type=str, choices=["keep", "remove"], default="keep")
    parser.add_argument("--strict", action="store_true", help="Fail on invalid relabel/review rows")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    train_csv = resolve_path(project_root, args.train_csv)
    if args.review_csv:
        review_csv = resolve_path(project_root, args.review_csv)
    else:
        review_csv = find_latest_manual_review(project_root)
    output_dir = resolve_path(project_root, args.output_dir)
    assert train_csv is not None and review_csv is not None and output_dir is not None

    report = apply_manual_review(
        train_csv=train_csv,
        review_csv=review_csv,
        output_dir=output_dir,
        output_prefix=str(args.output_prefix),
        soft_policy=str(args.soft_policy),
        strict=bool(args.strict),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()