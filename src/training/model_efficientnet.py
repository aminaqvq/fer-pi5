from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def create_efficientnet_b0(
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: Optional[float] = None,
) -> nn.Module:
    """Build an EfficientNet-B0 classifier for FER.

    Uses torchvision's official EfficientNet-B0 and only replaces the
    classifier head.  Returns a plain ``nn.Module`` whose forward produces
    a ``[B, num_classes]`` tensor.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if not hasattr(model, "classifier"):
        raise RuntimeError("Unexpected EfficientNet-B0 structure: missing classifier")

    classifier = model.classifier
    if isinstance(classifier, nn.Sequential):
        last = classifier[-1]
        if not isinstance(last, nn.Linear):
            raise RuntimeError(f"Unexpected EfficientNet-B0 classifier[-1]: {type(last)}")
        in_features = last.in_features

        if dropout is None:
            # Keep torchvision's original dropout layer, only replace the final Linear.
            modules = list(classifier.children())
            modules[-1] = nn.Linear(in_features, num_classes)
            model.classifier = nn.Sequential(*modules)
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=float(dropout), inplace=True),
                nn.Linear(in_features, num_classes),
            )
    else:
        raise RuntimeError(f"Unexpected EfficientNet-B0 classifier type: {type(classifier)}")

    return model


def get_efficientnet_info(model: nn.Module) -> dict:
    """Return architecture metadata for an EfficientNet model."""
    return {
        "arch": "efficientnet_b0",
        "classifier": repr(getattr(model, "classifier", None)),
        "num_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
