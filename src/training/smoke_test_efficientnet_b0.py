#!/usr/bin/env python
"""Smoke test for EfficientNet-B0 integration.

Usage:
    python src/training/smoke_test_efficientnet_b0.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "src" / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT))

import torch

from model_mbv3 import get_model, load_checkpoint_into_model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # ── 1. Create EfficientNet-B0 ──────────────────────────────────
    model = get_model(
        variant="efficientnet_b0",
        num_classes=7,
        pretrained=True,
        device=device,
        verbose=True,
        compile_model=False,
    )

    params = count_params(model)
    print(f"params={params:,}")
    print(f"classifier: {model.classifier}")

    x = torch.randn(2, 3, 224, 224, device=device)

    # ── 2. Train forward ──────────────────────────────────────────
    model.train()
    y_train = model(x)
    assert torch.is_tensor(y_train), f"Expected tensor, got {type(y_train)}"
    assert y_train.shape == (2, 7), f"Expected (2,7), got {y_train.shape}"
    print("train forward: tensor [2,7] OK")

    # ── 3. Eval forward ───────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    assert torch.is_tensor(y_eval), f"Expected tensor, got {type(y_eval)}"
    assert y_eval.shape == (2, 7)
    print("eval forward: tensor [2,7] OK")

    # ── 4. Checkpoint save/load ────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp = f.name
    torch.save({"model": model.state_dict()}, tmp)

    model2 = get_model(
        variant="efficientnet_b0",
        num_classes=7,
        pretrained=False,
        device=device,
        verbose=False,
        compile_model=False,
    )
    load_checkpoint_into_model(model2, tmp, device=torch.device(device), strict=True)

    model2.eval()
    with torch.no_grad():
        y2 = model2(x)
    max_diff = (y_eval - y2).abs().max().item()
    print(f"reload max_diff={max_diff:.3e}")
    assert max_diff < 1e-5, f"Reload diff too large: {max_diff}"

    # ── 5. MobileNetV3 backward compat ────────────────────────────
    _ = get_model(variant="small", num_classes=7, pretrained=False, device=device, verbose=False)
    _ = get_model(variant="large", num_classes=7, pretrained=False, device=device, verbose=False)
    print("MobileNetV3 small/large: OK")

    print("\nEfficientNet-B0 smoke test PASSED")


if __name__ == "__main__":
    main()
