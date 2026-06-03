from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk


# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from dataset import FER2013Hybrid, IMG_SIZE
    from metrics import LABELS, NUM_CLASSES
except Exception as exc:
    print("[import-error] Put this script in src/training beside dataset.py and metrics.py")
    print(f"[import-error] current_dir={THIS_DIR}")
    print(f"[import-error] original={type(exc).__name__}: {exc}")
    raise


REVIEW_COLUMNS = [
    "review_action",       # keep / relabel / ignore / soft
    "new_label",           # label name when review_action == relabel
    "new_label_id",        # numeric label id when review_action == relabel
    "soft_label_json",     # optional, not required for the fast workflow
    "review_reason",
    "reviewer",
    "reviewed_at",
]

ACTION_KEEP = "keep"
ACTION_RELABEL = "relabel"
ACTION_IGNORE = "ignore"
ACTION_SOFT = "soft"


# ---------------------------------------------------------------------------
# Path / CSV helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def is_none_like(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null"}


def resolve_path(project_root: Path, value: Any) -> Optional[Path]:
    if is_none_like(value):
        return None
    p = Path(str(value))
    return p if p.is_absolute() else project_root / p


def read_csv_dicts(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(path)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def find_latest_issue_csv(project_root: Path) -> Path:
    candidates: List[Path] = []
    audit_root = project_root / "runs" / "audit"
    if audit_root.exists():
        patterns = [
            "oof_train_audit_*/*manual_label_review_template.csv",
            "oof_train_audit_*/*oof_train_label_issues.csv",
            "*manual_label_review_template.csv",
            "*oof_train_label_issues.csv",
        ]
        for pattern in patterns:
            candidates.extend(audit_root.glob(pattern))
    candidates = [p for p in candidates if p.exists() and p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-find manual_label_review_template.csv or oof_train_label_issues.csv. "
            "Use --issues-csv explicitly."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def row_key(row: Mapping[str, Any]) -> str:
    # Prefer dataset_index because it is aligned with FER2013Hybrid(train.csv) sample order.
    for col in ("dataset_index", "row_index"):
        value = row.get(col, "")
        if str(value).strip() != "":
            return str(int(float(str(value))))
    raise ValueError("Issue row has neither dataset_index nor row_index")


def parse_index(row: Mapping[str, Any], dataset_len: int) -> int:
    for col in ("dataset_index", "row_index"):
        value = row.get(col, "")
        if str(value).strip() == "":
            continue
        idx = int(float(str(value)))
        if 0 <= idx < dataset_len:
            return idx
    raise IndexError(f"Cannot resolve dataset index from row: {row}")


def merge_fieldnames(base: Sequence[str]) -> List[str]:
    out = list(base)
    for col in REVIEW_COLUMNS:
        if col not in out:
            out.append(col)
    return out


def action_is_done(row: Mapping[str, Any]) -> bool:
    return str(row.get("review_action", "")).strip().lower() in {
        ACTION_KEEP,
        ACTION_RELABEL,
        ACTION_IGNORE,
        ACTION_SOFT,
        "delete",
        "remove",
        "drop",
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_default_font(size: int = 16):
    for candidate in [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def error_image(text: str, size: Tuple[int, int] = (420, 420)) -> Image.Image:
    img = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(img)
    font = load_default_font(16)
    draw.text((16, 16), text, fill=(150, 0, 0), font=font)
    return img


def resize_for_display(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(float(max_side) / max(w, h), 1.0 if max(w, h) > max_side else float(max_side) / max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((new_w, new_h), resampling)


class RawImageProvider:
    def __init__(self, train_csv: Path, img_base: Optional[Path]) -> None:
        self.train_csv = train_csv
        self.img_base = img_base
        self.dataset = FER2013Hybrid(
            str(train_csv),
            None if img_base is None else str(img_base),
            "train",
            img_size=IMG_SIZE,
            include_label=True,
            strict=True,
        )
        self.samples = self.dataset.samples

    def __len__(self) -> int:
        return len(self.samples)

    def image_for_issue(self, issue_row: Mapping[str, Any]) -> Image.Image:
        idx = parse_index(issue_row, len(self.samples))
        sample = self.samples[idx]
        try:
            return self.dataset._load_image(sample).convert("RGB")
        except Exception as exc:
            return error_image(f"Image load failed\nidx={idx}\n{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Review store
# ---------------------------------------------------------------------------
class ReviewStore:
    def __init__(self, issue_rows: List[Dict[str, str]], issue_fields: List[str], output_csv: Path, reviewer: str) -> None:
        self.issue_rows = issue_rows
        self.issue_fields = issue_fields
        self.output_csv = output_csv
        self.reviewer = reviewer
        self.fieldnames = merge_fieldnames(issue_fields)
        self.reviews: Dict[str, Dict[str, str]] = {}
        self.history: List[Tuple[str, Dict[str, str]]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.output_csv.exists():
            return
        rows, _fields = read_csv_dicts(self.output_csv)
        for row in rows:
            try:
                key = row_key(row)
            except Exception:
                continue
            if action_is_done(row):
                self.reviews[key] = {k: str(v) for k, v in row.items()}

    def reviewed_count(self) -> int:
        return len(self.reviews)

    def is_reviewed(self, issue_row: Mapping[str, Any]) -> bool:
        return row_key(issue_row) in self.reviews

    def review_for(self, issue_row: Mapping[str, Any]) -> Dict[str, str]:
        return self.reviews.get(row_key(issue_row), {})

    def set_review(
        self,
        issue_row: Mapping[str, Any],
        *,
        action: str,
        new_label: str = "",
        new_label_id: str = "",
        soft_label_json: str = "",
        reason: str = "",
    ) -> None:
        key = row_key(issue_row)
        previous = dict(self.reviews.get(key, {}))
        self.history.append((key, previous))
        merged = {k: str(issue_row.get(k, "")) for k in self.fieldnames}
        merged.update({k: str(v) for k, v in issue_row.items()})
        merged.update({
            "review_action": action,
            "new_label": new_label,
            "new_label_id": str(new_label_id),
            "soft_label_json": soft_label_json,
            "review_reason": reason,
            "reviewer": self.reviewer,
            "reviewed_at": now_iso(),
        })
        self.reviews[key] = merged
        self.save()

    def clear_review(self, issue_row: Mapping[str, Any]) -> None:
        key = row_key(issue_row)
        previous = dict(self.reviews.get(key, {}))
        self.history.append((key, previous))
        self.reviews.pop(key, None)
        self.save()

    def undo(self) -> None:
        if not self.history:
            return
        key, previous = self.history.pop()
        if previous:
            self.reviews[key] = previous
        else:
            self.reviews.pop(key, None)
        self.save()

    def save(self) -> None:
        rows_out: List[Dict[str, Any]] = []
        for issue_row in self.issue_rows:
            base = {k: issue_row.get(k, "") for k in self.fieldnames}
            key = row_key(issue_row)
            if key in self.reviews:
                base.update(self.reviews[key])
            rows_out.append(base)
        write_csv_dicts(self.output_csv, rows_out, self.fieldnames)
        write_json(self.output_csv.with_suffix(".progress.json"), {
            "output_csv": str(self.output_csv),
            "reviewed_count": self.reviewed_count(),
            "total_issue_rows": len(self.issue_rows),
            "updated_at": now_iso(),
        })


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ReviewApp(tk.Tk):
    def __init__(
        self,
        *,
        image_provider: RawImageProvider,
        store: ReviewStore,
        issue_rows: List[Dict[str, str]],
        start_index: int,
        max_display_side: int,
        auto_next: bool,
        skip_reviewed: bool,
    ) -> None:
        super().__init__()
        self.title("FER Label Issue Reviewer")
        self.geometry("1180x760")
        self.minsize(1000, 680)

        self.image_provider = image_provider
        self.store = store
        self.issue_rows = issue_rows
        self.idx = max(0, min(int(start_index), max(0, len(issue_rows) - 1)))
        self.max_display_side = int(max_display_side)
        self.auto_next = bool(auto_next)
        self.skip_reviewed = bool(skip_reviewed)
        self.photo: Optional[ImageTk.PhotoImage] = None

        self._build_widgets()
        self._bind_keys()
        if self.skip_reviewed:
            self._go_next_unreviewed(start_at_current=True)
        self.show_current()

    def _build_widgets(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(root, width=420)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))

        self.image_label = ttk.Label(left, anchor=tk.CENTER)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, pady=(8, 0))

        self.info_text = tk.Text(right, width=52, height=22, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X)
        self.info_text.configure(state=tk.DISABLED)

        ttk.Label(right, text="审核原因/备注，可空：").pack(anchor=tk.W, pady=(10, 0))
        self.reason_var = tk.StringVar(value="")
        ttk.Entry(right, textvariable=self.reason_var).pack(fill=tk.X)

        action_frame = ttk.LabelFrame(right, text="操作")
        action_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(action_frame, text="保留 Keep  [K]", command=self.action_keep).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(action_frame, text="删除/忽略 Ignore  [D]", command=self.action_ignore).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(action_frame, text="撤销当前审核 Clear", command=self.action_clear_current).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(action_frame, text="撤销上一步 Undo  [U]", command=self.action_undo).pack(fill=tk.X, padx=6, pady=4)

        relabel_frame = ttk.LabelFrame(right, text="改为标签 Relabel：快捷键 1-7")
        relabel_frame.pack(fill=tk.X, pady=(12, 0))
        for i, label in enumerate(LABELS):
            ttk.Button(
                relabel_frame,
                text=f"{i + 1}. {label}",
                command=lambda label=label, i=i: self.action_relabel(label, i),
            ).pack(fill=tk.X, padx=6, pady=2)

        nav_frame = ttk.LabelFrame(right, text="导航")
        nav_frame.pack(fill=tk.X, pady=(12, 0))
        row1 = ttk.Frame(nav_frame)
        row1.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(row1, text="上一张 [←]", command=self.prev_item).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row1, text="下一张 [→/Space]", command=self.next_item).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(nav_frame, text="跳到下一个未审核", command=lambda: self._go_next_unreviewed(start_at_current=False)).pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(nav_frame, text="保存并退出 [Q]", command=self.save_and_quit).pack(fill=tk.X, padx=6, pady=4)

    def _bind_keys(self) -> None:
        self.bind("k", lambda _e: self.action_keep())
        self.bind("K", lambda _e: self.action_keep())
        self.bind("d", lambda _e: self.action_ignore())
        self.bind("D", lambda _e: self.action_ignore())
        self.bind("u", lambda _e: self.action_undo())
        self.bind("U", lambda _e: self.action_undo())
        self.bind("q", lambda _e: self.save_and_quit())
        self.bind("Q", lambda _e: self.save_and_quit())
        self.bind("<Right>", lambda _e: self.next_item())
        self.bind("<space>", lambda _e: self.next_item())
        self.bind("<Left>", lambda _e: self.prev_item())
        for i in range(min(9, len(LABELS))):
            self.bind(str(i + 1), lambda _e, i=i: self.action_relabel(LABELS[i], i))

    def current_row(self) -> Dict[str, str]:
        return self.issue_rows[self.idx]

    def _review_reason(self) -> str:
        return self.reason_var.get().strip()

    def _after_action(self) -> None:
        if self.auto_next:
            self._go_next_unreviewed(start_at_current=False)
        self.show_current()

    def action_keep(self) -> None:
        self.store.set_review(self.current_row(), action=ACTION_KEEP, reason=self._review_reason())
        self._after_action()

    def action_ignore(self) -> None:
        self.store.set_review(self.current_row(), action=ACTION_IGNORE, reason=self._review_reason())
        self._after_action()

    def action_relabel(self, label: str, label_id: int) -> None:
        self.store.set_review(
            self.current_row(),
            action=ACTION_RELABEL,
            new_label=label,
            new_label_id=str(label_id),
            reason=self._review_reason(),
        )
        self._after_action()

    def action_clear_current(self) -> None:
        self.store.clear_review(self.current_row())
        self.show_current()

    def action_undo(self) -> None:
        self.store.undo()
        self.show_current()

    def next_item(self) -> None:
        if self.idx < len(self.issue_rows) - 1:
            self.idx += 1
        self.show_current()

    def prev_item(self) -> None:
        if self.idx > 0:
            self.idx -= 1
        self.show_current()

    def _go_next_unreviewed(self, *, start_at_current: bool) -> None:
        start = self.idx if start_at_current else self.idx + 1
        for j in range(start, len(self.issue_rows)):
            if not self.store.is_reviewed(self.issue_rows[j]):
                self.idx = j
                return
        messagebox.showinfo("完成", "后面没有未审核样本了。")

    def save_and_quit(self) -> None:
        self.store.save()
        self.destroy()

    def show_current(self) -> None:
        if not self.issue_rows:
            messagebox.showerror("错误", "没有可审核的样本。")
            self.destroy()
            return

        row = self.current_row()
        review = self.store.review_for(row)
        self.reason_var.set(str(review.get("review_reason", "")))

        img = self.image_provider.image_for_issue(row)
        img_disp = resize_for_display(img, self.max_display_side)
        self.photo = ImageTk.PhotoImage(img_disp)
        self.image_label.configure(image=self.photo)

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, self._format_info(row, review))
        self.info_text.configure(state=tk.DISABLED)

        self.status_var.set(
            f"{self.idx + 1}/{len(self.issue_rows)} | reviewed={self.store.reviewed_count()} | "
            f"output={self.store.output_csv}"
        )

    def _format_info(self, row: Mapping[str, Any], review: Mapping[str, Any]) -> str:
        lines: List[str] = []
        lines.append("当前样本 / Current sample")
        lines.append("-" * 46)
        for key in [
            "issue_rank", "dataset_index", "row_index", "fold", "issue_type",
            "true_label", "pred_label", "confidence", "p_true", "margin", "loss", "issue_score",
        ]:
            if str(row.get(key, "")).strip() != "":
                lines.append(f"{key}: {row.get(key)}")
        lines.append("")
        lines.append("Top probabilities:")
        for rank in range(1, 6):
            l = row.get(f"top{rank}_label", "")
            p = row.get(f"top{rank}_prob", "")
            if str(l).strip():
                lines.append(f"  {rank}. {l}: {p}")
        lines.append("")
        lines.append("Path / source:")
        for key in ["path", "has_pixels", "source_csv"]:
            if str(row.get(key, "")).strip() != "":
                lines.append(f"{key}: {row.get(key)}")
        lines.append("")
        lines.append("当前审核 / Current review")
        lines.append("-" * 46)
        if review:
            for key in REVIEW_COLUMNS:
                if str(review.get(key, "")).strip() != "":
                    lines.append(f"{key}: {review.get(key)}")
        else:
            lines.append("未审核")
        lines.append("")
        lines.append("快捷键: K=保留, D=删除/忽略, 1-7=改标签, ←/→=前后, U=撤销, Q=退出")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI reviewer for FER OOF label issues")
    parser.add_argument("--project-root", type=str, default=os.environ.get("FER_PROJECT_ROOT", r"D:\fer-pi5"))
    parser.add_argument("--issues-csv", type=str, default=None, help="manual_label_review_template.csv or oof_train_label_issues.csv")
    parser.add_argument("--train-csv", type=str, default=r"data\csv\train.csv")
    parser.add_argument("--img-base", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None, help="Default: same folder as issues CSV / manual_label_review.csv")
    parser.add_argument("--reviewer", type=str, default=os.environ.get("USERNAME", "reviewer"))
    parser.add_argument("--limit", type=int, default=500, help="Review only top N rows from issues CSV; use 0 for all")
    parser.add_argument("--start", type=int, default=0, help="0-based starting row inside loaded issue rows")
    parser.add_argument("--max-side", type=int, default=520, help="Max displayed image side")
    parser.add_argument("--no-auto-next", action="store_true", help="Do not automatically advance after action")
    parser.add_argument("--show-reviewed", action="store_true", help="Do not skip already reviewed rows on startup/next-unreviewed")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    issues_csv = resolve_path(project_root, args.issues_csv) if args.issues_csv else find_latest_issue_csv(project_root)
    assert issues_csv is not None
    train_csv = resolve_path(project_root, args.train_csv)
    img_base = resolve_path(project_root, args.img_base)
    if train_csv is None or not train_csv.exists():
        raise FileNotFoundError(f"train_csv not found: {train_csv}")
    if not issues_csv.exists():
        raise FileNotFoundError(f"issues_csv not found: {issues_csv}")

    if args.output_csv:
        output_csv = resolve_path(project_root, args.output_csv)
        assert output_csv is not None
    else:
        output_csv = issues_csv.parent / "manual_label_review.csv"

    rows, fields = read_csv_dicts(issues_csv)
    if args.limit and int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError(f"No rows loaded from {issues_csv}")

    provider = RawImageProvider(train_csv, img_base)
    store = ReviewStore(rows, fields, output_csv, reviewer=str(args.reviewer))

    print("=== FER Label Issue Reviewer ===")
    print(f"project_root: {project_root}")
    print(f"issues_csv  : {issues_csv}")
    print(f"train_csv   : {train_csv}")
    print(f"output_csv  : {output_csv}")
    print(f"rows_loaded : {len(rows)}")
    print(f"reviewed    : {store.reviewed_count()}")

    app = ReviewApp(
        image_provider=provider,
        store=store,
        issue_rows=rows,
        start_index=int(args.start),
        max_display_side=int(args.max_side),
        auto_next=not bool(args.no_auto_next),
        skip_reviewed=not bool(args.show_reviewed),
    )
    app.mainloop()
    store.save()
    print(f"[done] saved: {output_csv}")


if __name__ == "__main__":
    main()