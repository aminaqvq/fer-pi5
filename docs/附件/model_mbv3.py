import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

def get_model(variant="large", num_classes=7, pretrained=True, device="cuda", verbose=True, compile_model=False):
    variant = variant.lower()
    if variant == "small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    else:
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

    if hasattr(model, "classifier"):
        last = model.classifier[-1]
        in_features = last.in_features if isinstance(last, nn.Linear) else 1280
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        model = nn.Sequential(model, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(1280, num_classes))

    model.to(device)
    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass
    if verbose:
        print(f"MobileNetV3-{variant} initialized (pretrained={pretrained}, num_classes={num_classes})")
    return model
