from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import inspect
import warnings

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights

SUPPORTED_VARIANTS: Tuple[str, ...] = ("small", "large")


def _classifier_in_features(model: nn.Module) -> int:
    if not hasattr(model, "classifier"):
        raise TypeError("Expected a torchvision MobileNetV3 model with a classifier attribute")
    classifier = getattr(model, "classifier")
    if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Linear):
        raise TypeError("Unexpected MobileNetV3 classifier layout")
    return int(classifier[-1].in_features)


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    in_features = _classifier_in_features(model)
    model.classifier[-1] = nn.Linear(in_features, int(num_classes))
    return model


def get_model(
    variant: str = "large",
    num_classes: int = 7,
    pretrained: bool = True,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    compile_model: bool = False,
    *,
    strict_compile: bool = False,
) -> nn.Module:
    """Build the canonical FER MobileNetV3 classifier.

    Unlike the old version, unknown variants fail fast instead of silently
    falling back to ``large``.
    """
    normalized = str(variant).lower().strip()
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported MobileNetV3 variant: {variant!r}. Expected one of {SUPPORTED_VARIANTS}.")

    if normalized == "small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

    replace_classifier(model, int(num_classes))
    model.to(torch.device(device))

    if compile_model:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)  # type: ignore[assignment]
            except Exception as exc:
                if strict_compile:
                    raise
                warnings.warn(f"torch.compile failed; continuing without compilation: {exc}")
        elif strict_compile:
            raise RuntimeError("compile_model=True but torch.compile is unavailable")

    if verbose:
        print(
            f"MobileNetV3-{normalized} initialized "
            f"(pretrained={pretrained}, num_classes={num_classes}, device={device})",
            flush=True,
        )
    return model


build_model = get_model


def _strip_known_prefixes(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    cleaned: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("module.", "model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    """Accept raw state_dicts and common checkpoint dictionaries."""
    if isinstance(obj, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = obj.get(key)  # type: ignore[arg-type]
            if isinstance(value, Mapping):
                return _strip_known_prefixes(value)  # type: ignore[arg-type]
        if obj and all(hasattr(v, "shape") for v in obj.values()):  # raw state dict
            return _strip_known_prefixes(obj)  # type: ignore[arg-type]
    raise TypeError("Checkpoint does not look like a PyTorch state_dict or known checkpoint dict")


def torch_load_safe(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a local trusted checkpoint using weights_only when supported."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    kwargs: Dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(str(path), **kwargs)


def load_checkpoint_into_model(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    checkpoint = torch_load_safe(ckpt_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        message = f"Checkpoint key mismatch. missing={missing}, unexpected={unexpected}"
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    return model
