from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Raise CSV field limit for large FER pixel-string columns.
try:
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
except OverflowError:
    csv.field_size_limit(2 ** 31 - 1)

LABEL_ORDER: List[str] = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABEL_ORDER)}
LABEL_ALIASES: Dict[str, str] = {
    # canonical names
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
    # common directory / dataset variants
    "angry": "anger",
    "happiness": "happy",
    "sadness": "sad",
    "surprised": "surprise",
    "surprise_": "surprise",
    "neutrality": "neutral",
    # numeric FER order used by this project
    "0": "anger",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}
KNOWN_PROCESSED_CSVS: List[Tuple[str, str]] = [
    ("emotion-domestic", "emotion-domestic/emotion-domestic_processed.csv"),
    ("fer2013plus", "fer2013plus/fer2013plus_processed.csv"),
    ("MMAFEDB", "MMAFEDB/MMAFEDB_processed.csv"),
    ("Rafdb", "Rafdb/Rafdb_processed.csv"),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PIXEL_COUNT = 48 * 48


PYCHARM_LAZY_RUN = True

# 项目数据根目录。Windows 路径推荐使用 r"..." 原始字符串。
DEFAULT_DATA_ROOT = r"D:\fer-pi5\data"
DEFAULT_OUTPUT_DIR = r"D:\fer-pi5\data\csv"

# 当前推荐模式：直接使用四个已经处理好的 emotion,pixels CSV。
DEFAULT_SOURCE_MODE = "processed-csv"

# processed CSV 输入源。格式是：数据集名称=相对 data-root 的路径
DEFAULT_SOURCE_CSVS = [
    r"emotion-domestic=emotion-domestic\emotion-domestic_processed.csv",
    r"fer2013plus=fer2013plus\fer2013plus_processed.csv",
    r"MMAFEDB=MMAFEDB\MMAFEDB_processed.csv",
    r"Rafdb=Rafdb\Rafdb_processed.csv",
]

# 划分策略：
# 每个类别先预留 20% 做 unlabeled；
# 剩余 80% labeled pool 再按 8:1:1 切成 train/val/test。
DEFAULT_UNLABELED_RATIO = "0.20"
DEFAULT_TRAIN_RATIO = "0.8"
DEFAULT_VAL_RATIO = "0.1"
DEFAULT_TEST_RATIO = "0.1"

# 固定随机种子，保证同一份输入 CSV 生成同一套划分。
DEFAULT_SEED = "42"

# exact：按样本内容 SHA-256 精确去重。
# none：完全不去重，不推荐。
DEFAULT_DEDUPE = "exact"

# count：只检查像素数量是否为 48*48
# basic：检查数量，并抽样检查像素是否为 0-255 整数，推荐
# full：逐像素完整检查，最严格但更慢
DEFAULT_PIXEL_CHECK = "full"

# True：直接覆盖旧的 train.csv / val.csv / test.csv / unlabeled.csv
# False：如果输出已存在则直接报错，适合保守模式
DEFAULT_OVERWRITE = True

# True：只审计和打印统计，不写最终 split CSV
# False：正式写出结果
DEFAULT_DRY_RUN = False


def build_pycharm_default_argv() -> List[str]:
    """Build default CLI arguments for no-argument PyCharm runs."""
    argv: List[str] = [
        "--data-root", DEFAULT_DATA_ROOT,
        "--source-mode", DEFAULT_SOURCE_MODE,
        "--output-dir", DEFAULT_OUTPUT_DIR,
        "--unlabeled-ratio", DEFAULT_UNLABELED_RATIO,
        "--train-ratio", DEFAULT_TRAIN_RATIO,
        "--val-ratio", DEFAULT_VAL_RATIO,
        "--test-ratio", DEFAULT_TEST_RATIO,
        "--seed", DEFAULT_SEED,
        "--dedupe", DEFAULT_DEDUPE,
        "--pixel-check", DEFAULT_PIXEL_CHECK,
    ]
    for source_csv in DEFAULT_SOURCE_CSVS:
        argv.extend(["--source-csv", source_csv])
    if DEFAULT_OVERWRITE:
        argv.append("--overwrite")
    if DEFAULT_DRY_RUN:
        argv.append("--dry-run")
    return argv



@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: str
    kind: str
    sha256: Optional[str] = None


@dataclass
class Counters:
    seen: int = 0
    accepted: int = 0
    dropped_duplicate_same_label: int = 0
    dropped_duplicate_conflicting_label: int = 0
    quarantined: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes_iter(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    def chunks() -> Iterator[bytes]:
        with path.open("rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                yield b
    return sha256_bytes_iter(chunks())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def normalize_label(value: object) -> Tuple[int, str]:
    raw = "" if value is None else str(value).strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_")
    canonical = LABEL_ALIASES.get(raw)
    if canonical is None:
        raise ValueError(f"unknown label: {value!r}")
    return LABEL_TO_ID[canonical], canonical


def normalize_pixels(value: object, pixel_check: str) -> str:
    if value is None:
        raise ValueError("missing pixels")
    pixels = " ".join(str(value).strip().split())
    if not pixels:
        raise ValueError("empty pixels")
    parts = pixels.split(" ")
    if len(parts) != PIXEL_COUNT:
        raise ValueError(f"pixel count {len(parts)} != {PIXEL_COUNT}")
    if pixel_check == "full":
        for p in parts:
            try:
                v = int(p)
            except Exception as exc:
                raise ValueError(f"non-integer pixel value: {p!r}") from exc
            if v < 0 or v > 255:
                raise ValueError(f"pixel value out of range [0,255]: {v}")
    elif pixel_check == "basic":
        # Cheap validation: check a few positions without parsing all 2304 values.
        probe = parts[:8] + parts[-8:]
        for p in probe:
            if not p.isdigit():
                raise ValueError(f"non-integer pixel value in probe: {p!r}")
    elif pixel_check == "count":
        pass
    else:
        raise ValueError(f"unsupported pixel_check: {pixel_check}")
    return pixels


def lower_key_row(row: Dict[str, str]) -> Dict[str, str]:
    return {str(k).strip().lower(): v for k, v in row.items() if k is not None}


def parse_source_csv_args(values: Optional[Sequence[str]], data_root: Path) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    if values:
        for value in values:
            if "=" in value:
                name, path_text = value.split("=", 1)
                name = name.strip()
                path = Path(path_text.strip())
            else:
                path = Path(value.strip())
                name = path.parent.name or path.stem.replace("_processed", "")
            if not path.is_absolute():
                path = data_root / path
            specs.append(SourceSpec(name=name, path=str(path), kind="processed_csv"))
        return specs

    for name, rel in KNOWN_PROCESSED_CSVS:
        path = data_root / rel
        if path.exists():
            specs.append(SourceSpec(name=name, path=str(path), kind="processed_csv"))

    if specs:
        return specs

    # Fallback: discover *_processed.csv one level below data_root.
    for path in sorted(data_root.glob("*/*_processed.csv")):
        specs.append(SourceSpec(name=path.parent.name, path=str(path), kind="processed_csv"))
    return specs


def discover_image_sources(data_root: Path, source_dirs: Optional[Sequence[str]]) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    candidates: List[Path]
    if source_dirs:
        candidates = []
        for value in source_dirs:
            p = Path(value)
            if not p.is_absolute():
                p = data_root / p
            candidates.append(p)
    else:
        candidates = [data_root / name for name in ["emotion-domestic", "fer2013plus", "MMAFEDB", "Rafdb"]]
        candidates = [p for p in candidates if p.exists()]
    for p in candidates:
        if not p.is_dir():
            raise FileNotFoundError(f"source image directory not found: {p}")
        specs.append(SourceSpec(name=p.name, path=str(p), kind="image_dir"))
    return specs



def relative_path_or_absolute(path: Path, root: Path) -> str:
    """Return path relative to root when possible; otherwise return the absolute path."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)

def init_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT UNIQUE NOT NULL,
            source_dataset TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            label INTEGER NOT NULL,
            label_name TEXT NOT NULL,
            pixels TEXT NOT NULL,
            path TEXT NOT NULL,
            sample_sha256 TEXT NOT NULL,
            split TEXT,
            write_order INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_split ON samples(split)")
    conn.commit()
    return conn


def open_quarantine(path: Path) -> Tuple[csv.DictWriter, object]:
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "reason",
            "source_dataset",
            "source_file",
            "source_row",
            "label_raw",
            "sample_sha256",
            "first_sample_id",
            "first_label",
            "preview",
        ],
    )
    writer.writeheader()
    return writer, f


def write_quarantine(
    writer: csv.DictWriter,
    reason: str,
    source_dataset: str,
    source_file: str,
    source_row: int,
    label_raw: object = "",
    sample_sha256: str = "",
    first_sample_id: str = "",
    first_label: object = "",
    preview: str = "",
) -> None:
    writer.writerow(
        {
            "reason": reason,
            "source_dataset": source_dataset,
            "source_file": source_file,
            "source_row": source_row,
            "label_raw": label_raw,
            "sample_sha256": sample_sha256,
            "first_sample_id": first_sample_id,
            "first_label": first_label,
            "preview": preview[:200],
        }
    )


def insert_sample(
    conn: sqlite3.Connection,
    *,
    sample_id: str,
    source_dataset: str,
    source_file: str,
    source_row: int,
    label: int,
    label_name: str,
    pixels: str,
    path: str,
    sample_sha256: str,
) -> None:
    conn.execute(
        """
        INSERT INTO samples (
            sample_id, source_dataset, source_file, source_row,
            label, label_name, pixels, path, sample_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sample_id, source_dataset, source_file, int(source_row), int(label), label_name, pixels, path, sample_sha256),
    )


def load_processed_csvs(
    conn: sqlite3.Connection,
    sources: Sequence[SourceSpec],
    args: argparse.Namespace,
    quarantine_writer: csv.DictWriter,
) -> Tuple[Counters, Dict[str, Tuple[int, str]]]:
    counters = Counters()
    seen_hashes: Dict[str, Tuple[int, str]] = {}
    commit_every = max(1, int(args.commit_every))

    for spec in sources:
        path = Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f"processed CSV not found: {path}")
        print(f"[load] {spec.name}: {path}")
        with path.open("r", encoding=args.encoding, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"empty CSV or missing header: {path}")
            headers = [h.strip().lower() for h in reader.fieldnames]
            if "emotion" not in headers and "label" not in headers:
                raise ValueError(f"{path} must contain 'emotion' or 'label' column")
            if "pixels" not in headers and "path" not in headers and "filepath" not in headers and "image" not in headers:
                raise ValueError(f"{path} must contain 'pixels' or image path column")

            for row_num, raw_row in enumerate(reader, start=2):
                counters.seen += 1
                row = lower_key_row(raw_row)
                label_raw = row.get("emotion", row.get("label", ""))
                try:
                    label, label_name = normalize_label(label_raw)
                    pixels = ""
                    rel_path = ""
                    if row.get("pixels", ""):
                        pixels = normalize_pixels(row.get("pixels"), args.pixel_check)
                        sample_hash = sha256_text(pixels)
                    else:
                        rel_path = str(row.get("path") or row.get("filepath") or row.get("image") or "").strip()
                        if not rel_path:
                            raise ValueError("missing both pixels and path")
                        p = Path(rel_path)
                        if not p.is_absolute():
                            # If a processed CSV uses paths relative to its own folder, keep a stable relative path.
                            candidate = path.parent / p
                            if candidate.exists():
                                rel_path = relative_path_or_absolute(candidate, args.data_root)
                        sample_hash = sha256_text(rel_path) if args.path_hash_mode == "path" else sha256_file(Path(rel_path) if Path(rel_path).is_absolute() else args.data_root / rel_path)
                except Exception as exc:
                    counters.quarantined += 1
                    write_quarantine(
                        quarantine_writer,
                        reason=str(exc),
                        source_dataset=spec.name,
                        source_file=str(path),
                        source_row=row_num,
                        label_raw=label_raw,
                        preview=str(raw_row),
                    )
                    continue

                if args.dedupe != "none":
                    first = seen_hashes.get(sample_hash)
                    if first is not None:
                        first_label, first_sample_id = first
                        if first_label != label:
                            counters.dropped_duplicate_conflicting_label += 1
                            write_quarantine(
                                quarantine_writer,
                                reason="duplicate_conflicting_label",
                                source_dataset=spec.name,
                                source_file=str(path),
                                source_row=row_num,
                                label_raw=label_raw,
                                sample_sha256=sample_hash,
                                first_sample_id=first_sample_id,
                                first_label=first_label,
                                preview=str(raw_row),
                            )
                            continue
                        counters.dropped_duplicate_same_label += 1
                        if args.log_same_label_duplicates:
                            write_quarantine(
                                quarantine_writer,
                                reason="duplicate_same_label_dropped",
                                source_dataset=spec.name,
                                source_file=str(path),
                                source_row=row_num,
                                label_raw=label_raw,
                                sample_sha256=sample_hash,
                                first_sample_id=first_sample_id,
                                first_label=first_label,
                                preview=str(raw_row),
                            )
                        continue

                sample_id = f"{spec.name}:{path.name}:{row_num}:{sample_hash[:16]}"
                try:
                    insert_sample(
                        conn,
                        sample_id=sample_id,
                        source_dataset=spec.name,
                        source_file=str(path),
                        source_row=row_num,
                        label=label,
                        label_name=label_name,
                        pixels=pixels,
                        path=rel_path,
                        sample_sha256=sample_hash,
                    )
                except sqlite3.IntegrityError as exc:
                    counters.quarantined += 1
                    write_quarantine(
                        quarantine_writer,
                        reason=f"sqlite_integrity_error: {exc}",
                        source_dataset=spec.name,
                        source_file=str(path),
                        source_row=row_num,
                        label_raw=label_raw,
                        sample_sha256=sample_hash,
                        preview=str(raw_row),
                    )
                    continue
                seen_hashes[sample_hash] = (label, sample_id)
                counters.accepted += 1
                if counters.accepted % commit_every == 0:
                    conn.commit()
                    print(f"  accepted={counters.accepted:,} quarantined={counters.quarantined:,}")
    conn.commit()
    return counters, seen_hashes


def iter_images_for_source(source: SourceSpec, data_root: Path) -> Iterator[Tuple[int, Path, int, str]]:
    root = Path(source.path)
    row_num = 0
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            label, label_name = normalize_label(class_dir.name)
        except Exception:
            continue
        files: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(class_dir.rglob(f"*{ext}"))
            files.extend(class_dir.rglob(f"*{ext.upper()}"))
        for image_path in sorted(set(files)):
            row_num += 1
            yield row_num, image_path, label, label_name


def load_image_dirs(
    conn: sqlite3.Connection,
    sources: Sequence[SourceSpec],
    args: argparse.Namespace,
    quarantine_writer: csv.DictWriter,
) -> Counters:
    counters = Counters()
    seen_hashes: Dict[str, Tuple[int, str]] = {}
    commit_every = max(1, int(args.commit_every))

    for spec in sources:
        print(f"[scan] {spec.name}: {spec.path}")
        for row_num, image_path, label, label_name in iter_images_for_source(spec, args.data_root):
            counters.seen += 1
            try:
                if args.image_hash_mode == "bytes":
                    sample_hash = sha256_file(image_path)
                elif args.image_hash_mode == "path":
                    sample_hash = sha256_text(str(image_path.resolve()))
                else:
                    sample_hash = f"nohash:{spec.name}:{row_num}:{image_path.name}"
                rel_path = relative_path_or_absolute(image_path, args.data_root)
            except Exception as exc:
                counters.quarantined += 1
                write_quarantine(
                    quarantine_writer,
                    reason=str(exc),
                    source_dataset=spec.name,
                    source_file=str(image_path),
                    source_row=row_num,
                    label_raw=label_name,
                    preview=str(image_path),
                )
                continue

            if args.dedupe != "none":
                first = seen_hashes.get(sample_hash)
                if first is not None:
                    first_label, first_sample_id = first
                    if first_label != label:
                        counters.dropped_duplicate_conflicting_label += 1
                        write_quarantine(
                            quarantine_writer,
                            reason="duplicate_conflicting_label",
                            source_dataset=spec.name,
                            source_file=str(image_path),
                            source_row=row_num,
                            label_raw=label_name,
                            sample_sha256=sample_hash,
                            first_sample_id=first_sample_id,
                            first_label=first_label,
                            preview=str(image_path),
                        )
                        continue
                    counters.dropped_duplicate_same_label += 1
                    continue

            sample_id = f"{spec.name}:{row_num}:{sample_hash[:16]}"
            insert_sample(
                conn,
                sample_id=sample_id,
                source_dataset=spec.name,
                source_file=str(image_path),
                source_row=row_num,
                label=label,
                label_name=label_name,
                pixels="",
                path=rel_path,
                sample_sha256=sample_hash,
            )
            seen_hashes[sample_hash] = (label, sample_id)
            counters.accepted += 1
            if counters.accepted % commit_every == 0:
                conn.commit()
                print(f"  accepted={counters.accepted:,} quarantined={counters.quarantined:,}")
    conn.commit()
    return counters


def allocate_counts(total: int, ratios: Sequence[float], minimums: Sequence[int]) -> List[int]:
    if len(ratios) != len(minimums):
        raise ValueError("ratios and minimums length mismatch")
    if total < sum(minimums):
        raise ValueError(f"total {total} < required minimum {sum(minimums)}")
    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0:
        raise ValueError("ratio sum must be positive")
    raw = [total * (r / ratio_sum) for r in ratios]
    counts = [int(x) for x in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1

    # Enforce minimums by borrowing from the largest buckets above their minimum.
    changed = True
    while changed:
        changed = False
        for i, minimum in enumerate(minimums):
            if counts[i] < minimum:
                deficit = minimum - counts[i]
                donors = sorted(
                    [j for j in range(len(counts)) if counts[j] > minimums[j]],
                    key=lambda j: counts[j] - minimums[j],
                    reverse=True,
                )
                if not donors:
                    raise ValueError("cannot satisfy minimum split counts")
                for j in donors:
                    take = min(deficit, counts[j] - minimums[j])
                    counts[j] -= take
                    counts[i] += take
                    deficit -= take
                    changed = True
                    if deficit == 0:
                        break
    assert sum(counts) == total
    return counts


def build_splits(conn: sqlite3.Connection, args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    rng = random.Random(args.seed)
    split_summary: Dict[str, Dict[str, int]] = defaultdict(dict)
    minimums = [args.min_train_per_class, args.min_val_per_class, args.min_test_per_class]
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]

    for label_id, label_name in enumerate(LABEL_ORDER):
        ids = [row[0] for row in conn.execute("SELECT id FROM samples WHERE label=?", (label_id,)).fetchall()]
        n = len(ids)
        if n == 0:
            raise RuntimeError(f"no accepted samples for label {label_id} ({label_name})")
        rng.shuffle(ids)

        n_unlabeled = int(round(n * args.unlabeled_ratio)) if args.unlabeled_ratio > 0 else 0
        min_labeled_needed = sum(minimums)
        if n - n_unlabeled < min_labeled_needed:
            n_unlabeled = max(0, n - min_labeled_needed)
        n_labeled = n - n_unlabeled
        train_count, val_count, test_count = allocate_counts(n_labeled, ratios, minimums)

        train_ids = ids[:train_count]
        val_ids = ids[train_count:train_count + val_count]
        test_ids = ids[train_count + val_count:train_count + val_count + test_count]
        unlabeled_ids = ids[train_count + val_count + test_count:]

        for split_name, split_ids in [
            ("train", train_ids),
            ("val", val_ids),
            ("test", test_ids),
            ("unlabeled", unlabeled_ids),
        ]:
            conn.executemany("UPDATE samples SET split=? WHERE id=?", [(split_name, i) for i in split_ids])
            split_summary[split_name][label_name] = len(split_ids)

    conn.commit()

    # Assign deterministic write orders per split so output CSVs are not grouped by class.
    for split_name in ["train", "val", "test", "unlabeled"]:
        split_ids = [row[0] for row in conn.execute("SELECT id FROM samples WHERE split=?", (split_name,)).fetchall()]
        rng_split = random.Random(f"{args.seed}:{split_name}")
        if args.shuffle_output:
            rng_split.shuffle(split_ids)
        conn.executemany("UPDATE samples SET write_order=? WHERE id=?", [(idx, sample_id) for idx, sample_id in enumerate(split_ids)])
    conn.commit()
    return {split: dict(counts) for split, counts in split_summary.items()}


def ensure_outputs_can_be_written(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["train.csv", "val.csv", "test.csv", "unlabeled.csv"]
    existing = [output_dir / n for n in names if (output_dir / n).exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(p) for p in existing)
        raise FileExistsError(
            "output files already exist. Pass --overwrite to replace them:\n  " + joined
        )


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(final_path)


def write_split_csvs(conn: sqlite3.Connection, output_dir: Path, args: argparse.Namespace) -> Dict[str, str]:
    output_paths: Dict[str, str] = {}
    fieldnames_labeled = [
        "emotion",
        "pixels",
        "path",
        "sample_id",
        "source_dataset",
        "source_row",
        "sample_sha256",
        "label_name",
    ]
    fieldnames_unlabeled = [
        "pixels",
        "path",
        "sample_id",
        "source_dataset",
        "source_row",
        "sample_sha256",
    ]
    if not args.hide_unlabeled_labels:
        fieldnames_unlabeled = ["emotion"] + fieldnames_unlabeled + ["label_name"]

    for split_name in ["train", "val", "test", "unlabeled"]:
        final_path = output_dir / f"{split_name}.csv"
        tmp_path = output_dir / f".{split_name}.csv.tmp.{os.getpid()}"
        fieldnames = fieldnames_unlabeled if split_name == "unlabeled" else fieldnames_labeled
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            query = (
                "SELECT label, pixels, path, sample_id, source_dataset, source_row, sample_sha256, label_name "
                "FROM samples WHERE split=? ORDER BY write_order"
            )
            for label, pixels, path, sample_id, source_dataset, source_row, sample_sha256, label_name in conn.execute(query, (split_name,)):
                if split_name == "unlabeled" and args.hide_unlabeled_labels:
                    row = {
                        "pixels": pixels,
                        "path": path,
                        "sample_id": sample_id,
                        "source_dataset": source_dataset,
                        "source_row": source_row,
                        "sample_sha256": sample_sha256,
                    }
                else:
                    row = {
                        "emotion": label,
                        "pixels": pixels,
                        "path": path,
                        "sample_id": sample_id,
                        "source_dataset": source_dataset,
                        "source_row": source_row,
                        "sample_sha256": sample_sha256,
                        "label_name": label_name,
                    }
                writer.writerow(row)
        atomic_replace(tmp_path, final_path)
        output_paths[split_name] = str(final_path)
    return output_paths


def write_unlabeled_ground_truth(conn: sqlite3.Connection, audit_dir: Path) -> str:
    path = audit_dir / "unlabeled_ground_truth_audit.csv"
    tmp_path = audit_dir / f".unlabeled_ground_truth_audit.csv.tmp.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["emotion", "label_name", "sample_id", "source_dataset", "source_row", "sample_sha256"],
        )
        writer.writeheader()
        for row in conn.execute(
            "SELECT label, label_name, sample_id, source_dataset, source_row, sample_sha256 "
            "FROM samples WHERE split='unlabeled' ORDER BY write_order"
        ):
            writer.writerow(
                {
                    "emotion": row[0],
                    "label_name": row[1],
                    "sample_id": row[2],
                    "source_dataset": row[3],
                    "source_row": row[4],
                    "sample_sha256": row[5],
                }
            )
    atomic_replace(tmp_path, path)
    return str(path)


def write_class_counts(split_summary: Dict[str, Dict[str, int]], audit_dir: Path) -> str:
    path = audit_dir / "class_counts.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label_id", "label_name", "count"])
        writer.writeheader()
        for split_name in ["train", "val", "test", "unlabeled"]:
            for label_id, label_name in enumerate(LABEL_ORDER):
                writer.writerow(
                    {
                        "split": split_name,
                        "label_id": label_id,
                        "label_name": label_name,
                        "count": int(split_summary.get(split_name, {}).get(label_name, 0)),
                    }
                )
    return str(path)


def get_source_counts(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source_dataset, split, count in conn.execute(
        "SELECT source_dataset, split, COUNT(*) FROM samples GROUP BY source_dataset, split"
    ):
        result[source_dataset][split] = int(count)
    return {source: dict(counts) for source, counts in result.items()}


def get_total_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    return {split: int(count) for split, count in conn.execute("SELECT split, COUNT(*) FROM samples GROUP BY split")}


def write_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    sources: Sequence[SourceSpec],
    counters: Counters,
    split_summary: Dict[str, Dict[str, int]],
    output_paths: Dict[str, str],
    audit_paths: Dict[str, str],
    conn: sqlite3.Connection,
) -> None:
    enriched_sources = []
    for spec in sources:
        source_path = Path(spec.path)
        sha = None
        if source_path.is_file():
            sha = sha256_file(source_path)
        enriched_sources.append({**asdict(spec), "sha256": sha})

    manifest = {
        "created_at_utc": utc_now_iso(),
        "tool": "MergeClean_refactored.py",
        "source_mode": args.source_mode,
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "audit_dir": str(args.audit_dir),
        "seed": args.seed,
        "label_order": LABEL_ORDER,
        "label_to_id": LABEL_TO_ID,
        "ratios": {
            "unlabeled_ratio_total_per_class": args.unlabeled_ratio,
            "labeled_train_ratio": args.train_ratio,
            "labeled_val_ratio": args.val_ratio,
            "labeled_test_ratio": args.test_ratio,
        },
        "policies": {
            "dedupe": args.dedupe,
            "hide_unlabeled_labels": args.hide_unlabeled_labels,
            "pixel_check": args.pixel_check,
            "shuffle_output": args.shuffle_output,
        },
        "source_files": enriched_sources,
        "counters": asdict(counters),
        "split_class_counts": split_summary,
        "split_total_counts": get_total_counts(conn),
        "source_split_counts": get_source_counts(conn),
        "outputs": output_paths,
        "audit_outputs": audit_paths,
        "notes": [
            "No train_old.csv, val_old.csv, test_old.csv, or merged.csv is created.",
            "Unlabeled labels are hidden in unlabeled.csv by default and stored only in audit/unlabeled_ground_truth_audit.csv.",
            "Exact duplicate removal is based on normalized pixel SHA-256 for processed CSV mode and image bytes SHA-256 for image mode unless configured otherwise.",
        ],
    }
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    atomic_replace(tmp_path, path)


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fer-pi5 train/val/test/unlabeled CSVs from processed CSVs or image folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT), help="Root data directory, e.g. F:/fer-pi5/data")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR), help="Output CSV directory. Defaults to <data-root>/csv")
    parser.add_argument("--audit-dir", type=Path, default=None, help="Audit output directory. Defaults to <output-dir>/audit")
    parser.add_argument("--source-mode", choices=["processed-csv", "images"], default="processed-csv")
    parser.add_argument(
        "--source-csv",
        action="append",
        default=None,
        help="Processed CSV source. Repeatable. Use either path or name=path. Relative paths are resolved under data-root.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Image source directory. Repeatable. Relative paths are resolved under data-root.",
    )
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding for processed inputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlabeled-ratio", type=float, default=0.20, help="Per-class fraction withheld as unlabeled")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio within the labeled pool")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val ratio within the labeled pool")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio within the labeled pool")
    parser.add_argument("--min-train-per-class", type=int, default=1)
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument("--min-test-per-class", type=int, default=1)
    parser.add_argument("--dedupe", choices=["exact", "none"], default="exact")
    parser.add_argument("--pixel-check", choices=["count", "basic", "full"], default="basic")
    parser.add_argument("--path-hash-mode", choices=["bytes", "path"], default="bytes", help="Hash mode when a processed CSV row contains a path instead of pixels")
    parser.add_argument("--image-hash-mode", choices=["bytes", "path", "none"], default="bytes")
    parser.add_argument("--hide-unlabeled-labels", type=parse_bool, default=True)
    parser.add_argument("--shuffle-output", type=parse_bool, default=True)
    parser.add_argument("--log-same-label-duplicates", action="store_true", help="Also write same-label duplicates to quarantine.csv")
    parser.add_argument("--commit-every", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing train/val/test/unlabeled outputs")
    parser.add_argument("--keep-sqlite", action="store_true", help="Keep the temporary SQLite database under audit-dir")
    parser.add_argument("--dry-run", action="store_true", help="Build and audit, but do not write final split CSVs")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    args.data_root = args.data_root.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.data_root / "csv"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    if args.audit_dir is None:
        args.audit_dir = args.output_dir / "audit"
    else:
        args.audit_dir = args.audit_dir.expanduser().resolve()

    if not args.data_root.exists():
        raise FileNotFoundError(f"data-root does not exist: {args.data_root}")
    if args.unlabeled_ratio < 0 or args.unlabeled_ratio >= 1:
        raise ValueError("--unlabeled-ratio must be in [0, 1)")
    for name in ["train_ratio", "val_ratio", "test_ratio"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.train_ratio + args.val_ratio + args.test_ratio <= 0:
        raise ValueError("train/val/test ratios must sum to a positive number")


def print_summary(counters: Counters, split_summary: Dict[str, Dict[str, int]]) -> None:
    print("\n=== Build summary ===")
    for key, value in asdict(counters).items():
        print(f"{key:38s}: {value:,}")
    print("\n=== Split class counts ===")
    header = ["split"] + LABEL_ORDER + ["total"]
    print("\t".join(header))
    for split_name in ["train", "val", "test", "unlabeled"]:
        counts = [int(split_summary.get(split_name, {}).get(label, 0)) for label in LABEL_ORDER]
        print("\t".join([split_name] + [str(c) for c in counts] + [str(sum(counts))]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    # PyCharm lazy-run mode:
    # When this file is launched with no command-line arguments, use the defaults
    # configured at the top of the script. If any CLI argument is provided, keep
    # argparse behavior unchanged so advanced runs still work.
    if argv is None and PYCHARM_LAZY_RUN and len(sys.argv) == 1:
        argv = build_pycharm_default_argv()
        print("Using default arguments from the configuration block at the top of this script.")
        print("Equivalent argv:")
        print("  " + " ".join(argv))
        print()

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(args)

    ensure_outputs_can_be_written(args.output_dir, args.overwrite or args.dry_run)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    if args.source_mode == "processed-csv":
        sources = parse_source_csv_args(args.source_csv, args.data_root)
    else:
        sources = discover_image_sources(args.data_root, args.source_dir)
    if not sources:
        raise RuntimeError(f"no sources found for mode={args.source_mode!r}")

    print("=== Sources ===")
    for spec in sources:
        print(f"- {spec.name}: {spec.path}")

    db_path = args.audit_dir / "build_splits.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = init_database(db_path)

    quarantine_path = args.audit_dir / "quarantine.csv"
    quarantine_writer, quarantine_file = open_quarantine(quarantine_path)
    try:
        if args.source_mode == "processed-csv":
            counters, _seen = load_processed_csvs(conn, sources, args, quarantine_writer)
        else:
            counters = load_image_dirs(conn, sources, args, quarantine_writer)
        quarantine_file.flush()
    finally:
        quarantine_file.close()

    if counters.accepted == 0:
        raise RuntimeError("no valid samples accepted; check quarantine.csv")

    split_summary = build_splits(conn, args)
    print_summary(counters, split_summary)

    audit_paths: Dict[str, str] = {
        "quarantine_csv": str(quarantine_path),
        "class_counts_csv": write_class_counts(split_summary, args.audit_dir),
        "unlabeled_ground_truth_audit_csv": write_unlabeled_ground_truth(conn, args.audit_dir),
    }

    output_paths: Dict[str, str] = {}
    if args.dry_run:
        print("\n[dry-run] Final train/val/test/unlabeled CSVs were not written.")
    else:
        output_paths = write_split_csvs(conn, args.output_dir, args)
        print("\n=== Outputs ===")
        for split_name, path in output_paths.items():
            print(f"{split_name:10s}: {path}")

    manifest_path = args.audit_dir / "split_manifest.json"
    write_manifest(
        path=manifest_path,
        args=args,
        sources=sources,
        counters=counters,
        split_summary=split_summary,
        output_paths=output_paths,
        audit_paths=audit_paths,
        conn=conn,
    )
    print(f"manifest  : {manifest_path}")
    print(f"audit dir : {args.audit_dir}")

    conn.close()
    if not args.keep_sqlite and db_path.exists():
        db_path.unlink()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
