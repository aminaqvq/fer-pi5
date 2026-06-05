from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json

import torch
import torch.nn as nn

LABELS: Tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)
NUM_CLASSES: int = len(LABELS)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


@dataclass
class MetricResult:
    loss: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total: int
    confusion: List[List[int]]
    per_class: Dict[str, Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "global_macro_precision": self.macro_precision,
            "global_macro_recall": self.macro_recall,
            "global_macro_f1": self.macro_f1,
            "total": self.total,
            "confusion": self.confusion,
            "per_class": self.per_class,
        }


class MetricAccumulator:
    """Accumulate split-level metrics from logits and labels."""

    def __init__(self, num_classes: int = NUM_CLASSES, labels: Sequence[str] = LABELS) -> None:
        self.num_classes = int(num_classes)
        self.labels = tuple(labels)
        if len(self.labels) != self.num_classes:
            raise ValueError("labels length must equal num_classes")
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
        self.loss_sum = 0.0
        self.loss_count = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss: Optional[torch.Tensor | float] = None,
    ) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected logits with shape [N,{self.num_classes}], got {tuple(logits.shape)}"
            )
        target_cpu = target.detach().to("cpu", dtype=torch.long).view(-1)
        pred_cpu = logits.detach().to("cpu").argmax(dim=1).view(-1)
        if target_cpu.numel() != pred_cpu.numel():
            raise ValueError("target and prediction sizes differ")

        valid = (target_cpu >= 0) & (target_cpu < self.num_classes)
        target_cpu = target_cpu[valid]
        pred_cpu = pred_cpu[valid]
        for t, p in zip(target_cpu.tolist(), pred_cpu.tolist()):
            self.confusion[int(t), int(p)] += 1

        if loss is not None and int(valid.sum().item()) > 0:
            loss_value = float(loss.item() if hasattr(loss, "item") else loss)
            n = int(valid.sum().item())
            self.loss_sum += loss_value * n
            self.loss_count += n

    def compute(self) -> MetricResult:
        cm = self.confusion.clone().to(torch.long)
        total = int(cm.sum().item())
        correct = int(torch.diag(cm).sum().item())
        accuracy = _safe_div(correct, total)

        per_class: Dict[str, Dict[str, float]] = {}
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        for idx, label in enumerate(self.labels):
            tp = int(cm[idx, idx].item())
            fp = int(cm[:, idx].sum().item()) - tp
            fn = int(cm[idx, :].sum().item()) - tp
            support = int(cm[idx, :].sum().item())
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            f1 = _safe_div(2.0 * precision * recall, precision + recall)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            per_class[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(support),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
            }

        return MetricResult(
            loss=_safe_div(self.loss_sum, self.loss_count),
            accuracy=accuracy,
            macro_precision=float(sum(precisions) / max(1, len(precisions))),
            macro_recall=float(sum(recalls) / max(1, len(recalls))),
            macro_f1=float(sum(f1s) / max(1, len(f1s))),
            total=total,
            confusion=cm.tolist(),
            per_class=per_class,
        )


def logits_from_predictions(pred: Sequence[int], num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Create deterministic logits for metric tests from class predictions."""
    out = torch.full((len(pred), num_classes), -10.0)
    for i, p in enumerate(pred):
        out[i, int(p)] = 10.0
    return out


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable,
    device: str | torch.device,
    criterion: Optional[nn.Module] = None,
    *,
    num_classes: int = NUM_CLASSES,
    labels: Sequence[str] = LABELS,
    tta_horizontal_flip: bool = False,
) -> MetricResult:
    """Evaluate a model using split-level global metrics."""
    model.eval()
    accumulator = MetricAccumulator(num_classes=num_classes, labels=labels)
    device = torch.device(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise ValueError("Evaluation loader must yield at least (images, labels)")
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        # RepVGGplus (and similar models) may return a dict in training mode.
        # eval() mode should return a plain tensor, but be defensive.
        if isinstance(logits, dict):
            logits = logits.get("main", next(iter(logits.values())))
        if tta_horizontal_flip:
            logits_flip = model(torch.flip(xb, dims=[-1]))
            if isinstance(logits_flip, dict):
                logits_flip = logits_flip.get("main", next(iter(logits_flip.values())))
            logits = 0.5 * (logits + logits_flip)
        loss = criterion(logits, yb)
        accumulator.update(logits, yb, loss=loss)
    return accumulator.compute()


def save_metrics_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion_png(
    confusion: Sequence[Sequence[int]],
    path: str | Path,
    *,
    title: str,
    labels: Sequence[str] = LABELS,
) -> None:
    """Save a confusion matrix plot. Uses English labels for artifact portability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        cm = torch.tensor(confusion, dtype=torch.long).numpy()
        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        im = ax.imshow(cm, interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=list(labels),
            yticklabels=list(labels),
            ylabel="True label",
            xlabel="Predicted label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(int(cm[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=7,
                )
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    except Exception as exc:  # plotting should not invalidate evaluation
        print(f"[warn] failed to save confusion plot {path}: {exc}", flush=True)
