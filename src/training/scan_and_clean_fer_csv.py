from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

FER_PIXEL_COUNT = 48 * 48


def row_get(row: Mapping[str, Any], *names: str) -> str:
    lower = {str(k).lower(): "" if v is None else str(v) for k, v in row.items()}
    for name in names:
        key = name.lower()
        if key in lower:
            return lower[key]
    return ""


def check_row(row: Mapping[str, Any], *, img_root: Optional[Path]) -> tuple[bool, str, int]:
    raw_path = row_get(row, "path", "filepath", "image", "file").strip()
    pixels = row_get(row, "pixels").strip()

    if raw_path:
        path = Path(raw_path)
        if img_root is not None and not path.is_absolute():
            path = img_root / path
        if not path.exists():
            return False, f"image path not found: {path}", -1
        return True, "ok_path", -1

    if not pixels:
        return False, "row has neither path nor pixels", 0

    arr = np.fromstring(pixels, dtype=np.int16, sep=" ")
    if int(arr.size) != FER_PIXEL_COUNT:
        return False, f"expected {FER_PIXEL_COUNT} pixel values, got {int(arr.size)}", int(arr.size)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > 255):
        return False, "pixel values outside [0,255]", int(arr.size)
    return True, "ok_pixels", int(arr.size)


def scan_one(csv_path: Path, *, img_root: Optional[Path], write_clean: bool, suffix: str) -> Dict[str, Any]:
    valid_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    invalid_report: List[Dict[str, Any]] = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        for line_no, row in enumerate(reader, start=2):
            ok, reason, pixel_count = check_row(row, img_root=img_root)
            row_dict = {str(k): ("" if v is None else str(v)) for k, v in row.items()}
            if ok:
                valid_rows.append(row_dict)
            else:
                invalid_rows.append(row_dict)
                invalid_report.append({
                    "line": line_no,
                    "reason": reason,
                    "pixel_count": pixel_count,
                    "label": row_get(row, "label", "emotion", "mapped_emotion"),
                    "usage": row_get(row, "Usage", "usage"),
                    "sample_id": row_get(row, "sample_id"),
                    "source_index": row_get(row, "source_index"),
                })

    result = {
        "csv": str(csv_path),
        "total": len(valid_rows) + len(invalid_rows),
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "first_invalid": invalid_report[:20],
    }

    if write_clean:
        clean_path = csv_path.with_name(csv_path.stem + suffix + csv_path.suffix)
        bad_path = csv_path.with_name(csv_path.stem + ".invalid_rows" + csv_path.suffix)
        report_path = csv_path.with_name(csv_path.stem + ".invalid_rows.json")

        with clean_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid_rows)

        with bad_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(invalid_rows)

        report_path.write_text(json.dumps(invalid_report, indent=2, ensure_ascii=False), encoding="utf-8")
        result.update({
            "clean_csv": str(clean_path),
            "invalid_csv": str(bad_path),
            "invalid_report": str(report_path),
        })

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan FER CSV files for corrupt 48x48 pixel rows and optionally write cleaned copies.")
    parser.add_argument("--csv", nargs="+", required=True, help="CSV files to scan.")
    parser.add_argument("--img-root", default="", help="Optional image root for relative path rows.")
    parser.add_argument("--write-clean", action="store_true", help="Write *.valid.csv and *.invalid_rows.csv files.")
    parser.add_argument("--suffix", default=".valid", help="Suffix for cleaned CSV copies. Default: .valid")
    args = parser.parse_args()

    img_root = Path(args.img_root) if args.img_root else None
    results = []
    for item in args.csv:
        results.append(scan_one(Path(item), img_root=img_root, write_clean=bool(args.write_clean), suffix=str(args.suffix)))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
