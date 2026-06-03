from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset


def _label_name(core: Any, idx: int) -> str:
    labels = getattr(core, "LABELS", None)
    if labels is None:
        return str(idx)
    try:
        return str(labels[idx])
    except Exception:
        return str(idx)


def _counts_to_dict(core: Any, counts: Sequence[int]) -> Dict[str, int]:
    return {_label_name(core, i): int(v) for i, v in enumerate(counts)}


def dataset_labels(ds: Dataset) -> List[int]:
    """Return labels aligned with dataset indices without loading images when possible."""
    if hasattr(ds, "base") and ds.__class__.__name__ == "WeightedDataset":
        return dataset_labels(ds.base)

    if isinstance(ds, Subset):
        base_labels = dataset_labels(ds.dataset)
        return [int(base_labels[int(i)]) for i in ds.indices]

    if isinstance(ds, ConcatDataset):
        labels: List[int] = []
        for child in ds.datasets:
            labels.extend(dataset_labels(child))
        return labels

    if hasattr(ds, "samples"):
        return [int(sample.get("label", -1)) for sample in getattr(ds, "samples")]

    labels: List[int] = []
    for idx in range(len(ds)):
        item = ds[idx]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Cannot infer labels from dataset item")
        labels.append(int(item[1]))
    return labels


def count_labels_from_any_dataset(ds: Dataset, num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for label in dataset_labels(ds):
        y = int(label)
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts


def resolve_balanced_per_class(value: Any, counts: Sequence[int], batch_class_quota: int) -> int:
    counts_int = [int(x) for x in counts]
    if any(x <= 0 for x in counts_int):
        missing = [i for i, x in enumerate(counts_int) if x <= 0]
        raise ValueError(f"Balanced sampling requires every class to be present. Missing class indices: {missing}")

    if value is None or str(value).strip() in {"", "0", "None", "none", "null"}:
        limit = 0
    elif isinstance(value, str) and value.strip().lower() in {"auto", "auto_min", "min", "minimum"}:
        limit = min(counts_int)
    else:
        limit = int(value)

    if limit <= 0:
        limit = min(counts_int)

    quota = max(1, int(batch_class_quota))
    return int(limit // quota * quota)


class BalancedClassBatchSampler(Sampler[List[int]]):
    """Strict N-class balanced mini-batch sampler.

    Example for 7-class FER:
        batch_size = 126
        samples_per_class_per_batch = 18

    Then every batch contains exactly 18 samples from each class.
    Majority classes rotate across epochs instead of being permanently discarded.
    """

    def __init__(
        self,
        labels: Sequence[int],
        *,
        batch_size: int,
        num_classes: int,
        samples_per_class_per_batch: int = 0,
        per_class: Any = "auto_min",
        seed: int = 42,
        strict_batch_size: bool = True,
        replacement: bool = False,
        reference_counts: Optional[Sequence[int]] = None,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.labels = [int(x) for x in labels]
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.strict_batch_size = bool(strict_batch_size)
        self.replacement = bool(replacement)
        self.label_names = [str(x) for x in label_names] if label_names is not None else [
            str(i) for i in range(self.num_classes)
        ]
        self.epoch_index = -1

        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if int(samples_per_class_per_batch) > 0:
            self.samples_per_class_per_batch = int(samples_per_class_per_batch)
            expected = self.samples_per_class_per_batch * self.num_classes
            if self.strict_batch_size and expected != self.batch_size:
                raise ValueError(
                    f"batch_size={self.batch_size}, but samples_per_class_per_batch="
                    f"{self.samples_per_class_per_batch} gives {expected}. "
                    "Use batch_size = num_classes * samples_per_class_per_batch."
                )
            self.effective_batch_size = expected
        else:
            if self.strict_batch_size and self.batch_size % self.num_classes != 0:
                raise ValueError(
                    f"batch_size={self.batch_size} is not divisible by num_classes={self.num_classes}. "
                    "For strict 7-class FER, use values like 126 = 7 * 18."
                )
            self.samples_per_class_per_batch = max(1, self.batch_size // self.num_classes)
            self.effective_batch_size = self.samples_per_class_per_batch * self.num_classes

        self.buckets: Dict[int, List[int]] = {i: [] for i in range(self.num_classes)}
        ignored = 0
        for idx, label in enumerate(self.labels):
            if 0 <= label < self.num_classes:
                self.buckets[label].append(int(idx))
            else:
                ignored += 1

        self.counts = [len(self.buckets[i]) for i in range(self.num_classes)]
        self.reference_counts = [int(x) for x in reference_counts] if reference_counts is not None else list(self.counts)

        if len(self.reference_counts) != self.num_classes:
            raise ValueError(f"reference_counts must contain {self.num_classes} values")

        if any(c <= 0 for c in self.counts):
            missing = [self.label_names[i] for i, c in enumerate(self.counts) if c <= 0]
            raise ValueError(f"Balanced sampler cannot run; missing classes: {missing}")

        resolved = resolve_balanced_per_class(
            per_class,
            self.reference_counts,
            batch_class_quota=self.samples_per_class_per_batch,
        )

        if resolved <= 0:
            raise ValueError("Resolved per-class epoch size is zero")

        if not self.replacement and any(c < resolved for c in self.counts):
            raise ValueError(
                f"per_class={resolved} exceeds at least one available class count without replacement: {self.counts}"
            )

        self.per_class = int(resolved)
        self.num_batches = self.per_class // self.samples_per_class_per_batch
        self.ignored = int(ignored)

        if self.num_batches <= 0:
            raise ValueError("Balanced sampler would produce zero batches")

    def __len__(self) -> int:
        return int(self.num_batches)

    def summary(self) -> Dict[str, Any]:
        source = {self.label_names[i]: int(self.counts[i]) for i in range(self.num_classes)}
        ref = {self.label_names[i]: int(self.reference_counts[i]) for i in range(self.num_classes)}
        return {
            "sampler": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_count": len(self.labels),
            "ignored_label_count": self.ignored,
            "source_class_counts": source,
            "reference_class_counts": ref,
            "batch_size_requested": self.batch_size,
            "batch_size_effective": self.effective_batch_size,
            "samples_per_class_per_batch": self.samples_per_class_per_batch,
            "per_class_per_epoch": self.per_class,
            "batches_per_epoch": self.num_batches,
            "samples_per_epoch": self.num_batches * self.effective_batch_size,
            "replacement": self.replacement,
        }

    def __iter__(self):
        self.epoch_index += 1
        rng = random.Random(self.seed + self.epoch_index)

        selected_by_class: Dict[int, List[int]] = {}

        for y in range(self.num_classes):
            bucket = list(self.buckets[y])
            rng.shuffle(bucket)

            if len(bucket) >= self.per_class:
                chosen = bucket[: self.per_class]
            elif self.replacement:
                chosen = [rng.choice(bucket) for _ in range(self.per_class)]
            else:
                raise RuntimeError("Balanced sampler internal error: class count below resolved per_class")

            selected_by_class[y] = chosen

        k = self.samples_per_class_per_batch
        batches: List[List[int]] = []

        for batch_idx in range(self.num_batches):
            start = batch_idx * k
            end = start + k
            batch: List[int] = []

            for y in range(self.num_classes):
                batch.extend(selected_by_class[y][start:end])

            rng.shuffle(batch)
            batches.append(batch)

        rng.shuffle(batches)

        for batch in batches:
            yield batch


def balanced_reference_counts(ds: Dataset, source: str, num_classes: int) -> Optional[List[int]]:
    normalized = str(source or "sampler").lower().strip()

    if normalized in {"", "sampler", "all", "all_samples"}:
        return None

    if normalized in {"labeled", "labeled_train", "labeled_only"}:
        if isinstance(ds, ConcatDataset) and len(ds.datasets) >= 1:
            return count_labels_from_any_dataset(ds.datasets[0], num_classes)
        return count_labels_from_any_dataset(ds, num_classes)

    raise ValueError(f"Unsupported balanced_per_class_source={source!r}; use 'sampler' or 'labeled_train'.")


def install_balanced_batch_sampler(core: Any) -> None:
    """Monkey-patch train_core.make_loader to add sampling_strategy='balanced_batch'.

    Keep the original train_core.py unchanged. Put this file beside train_core.py,
    import it in the launcher, then call install_balanced_batch_sampler(train_core).
    """
    original_make_loader = core.make_loader
    num_classes = int(getattr(core, "NUM_CLASSES"))
    label_names = [str(x) for x in getattr(core, "LABELS", [str(i) for i in range(num_classes)])]

    def make_loader_patched(
        ds: Dataset,
        cfg: Mapping[str, Any],
        *,
        shuffle: bool,
        drop_last: bool,
        seed_offset: int = 0,
    ) -> DataLoader:
        strategy = str(cfg.get("sampling_strategy", "standard") or "standard").lower().strip()

        # Only patch the training loader. Val/test remain exhaustive and sequential.
        if not (bool(shuffle) and strategy == "balanced_batch"):
            return original_make_loader(ds, cfg, shuffle=shuffle, drop_last=drop_last, seed_offset=seed_offset)

        num_workers = int(cfg.get("num_workers", 4))

        common_kwargs: Dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": bool(cfg.get("pin_memory", True)),
            "worker_init_fn": getattr(core, "seed_worker") if num_workers > 0 else None,
        }

        if num_workers > 0:
            common_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
            common_kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))

        labels = dataset_labels(ds)

        reference_counts = balanced_reference_counts(
            ds,
            str(cfg.get("balanced_per_class_source", "sampler")),
            num_classes,
        )

        sampler = BalancedClassBatchSampler(
            labels,
            batch_size=int(cfg.get("batch_size", 128)),
            num_classes=num_classes,
            samples_per_class_per_batch=int(cfg.get("balanced_samples_per_class_per_batch") or 0),
            per_class=cfg.get("balanced_per_class", "auto_min"),
            seed=int(cfg.get("seed", 42)) + int(seed_offset),
            strict_batch_size=bool(cfg.get("balanced_strict_batch_size", True)),
            replacement=bool(cfg.get("balanced_replacement", False)),
            reference_counts=reference_counts,
            label_names=label_names,
        )

        print("[balanced_sampler]", json.dumps(sampler.summary(), ensure_ascii=False), flush=True)

        # Important:
        # When batch_sampler is used, do NOT also pass batch_size, shuffle,
        # sampler, or drop_last. Those options are mutually exclusive in PyTorch.
        return DataLoader(ds, batch_sampler=sampler, **common_kwargs)

    core.BalancedClassBatchSampler = BalancedClassBatchSampler
    core.make_loader = make_loader_patched
