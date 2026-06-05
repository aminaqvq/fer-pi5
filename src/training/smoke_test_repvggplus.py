from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch

from model_repvggplus import (
    RepVGGplus,
    RepVGGplusBlock,
    RepVGGplusStage,
    create_RepVGGplus_L2pse,
    repvgg_model_convert,
)

# ────────────────────────────────────────────────────────────────
# Configuration — tune these for your hardware
# ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2
IMG_SIZE = 224
NUM_CLASSES = 7

print(f"device={DEVICE}  batch={BATCH}  img_size={IMG_SIZE}  num_classes={NUM_CLASSES}")


# ────────────────────────────────────────────────────────────────
# 1.  Build L2pse model
# ────────────────────────────────────────────────────────────────
model = create_RepVGGplus_L2pse(num_classes=NUM_CLASSES, deploy=False, use_aux=True)
model.to(DEVICE)
model.train()
print(f"\n[1] Model built: {type(model).__name__}")

params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    Total params:   {params:,}")
print(f"    Trainable:      {trainable:,}")


# ────────────────────────────────────────────────────────────────
# 2.  Architecture checks
# ────────────────────────────────────────────────────────────────
print("\n[2] Architecture checks")

# Stage 0: 3 -> 64
s0 = model.stage0
assert isinstance(s0, RepVGGplusBlock), f"stage0 type: {type(s0)}"
assert s0.in_channels == 3, f"stage0.in_channels={s0.in_channels}"
assert s0.out_channels == 64, f"stage0.out_channels={s0.out_channels}"
print("    stage0: RepVGGplusBlock(3 -> 64)  [OK]")

# Stage block counts
stages = ["stage1", "stage2", "stage3_first", "stage3_second", "stage4"]
expected_blocks = [8, 14, 12, 12, 1]
for sname, expected in zip(stages, expected_blocks):
    stage = getattr(model, sname)
    assert isinstance(stage, RepVGGplusStage), f"{sname} type: {type(stage)}"
    n = len(list(stage.blocks.children()))
    assert n == expected, f"{sname}: expected {expected} blocks, got {n}"
    print(f"    {sname}: {n} blocks  [OK]")

# Classifier: 2560 -> 7
assert model.linear.in_features == 2560, f"linear.in_features={model.linear.in_features}"
assert model.linear.out_features == NUM_CLASSES, f"linear.out_features={model.linear.out_features}"
print(f"    linear: {model.linear.in_features} -> {model.linear.out_features}  [OK]")


# ────────────────────────────────────────────────────────────────
# 3.  Training forward (should return dict with aux logits)
# ────────────────────────────────────────────────────────────────
print("\n[3] Training forward")
x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
out = model(x)
assert isinstance(out, dict), f"Expected dict, got {type(out).__name__}"
assert "main" in out, f"Missing 'main' in keys: {list(out.keys())}"
assert out["main"].shape == (BATCH, NUM_CLASSES), f"main shape: {out['main'].shape}"
for aux_key in ("stage1_aux", "stage2_aux", "stage3_first_aux"):
    assert aux_key in out, f"Missing {aux_key}"
    assert out[aux_key].shape == (BATCH, NUM_CLASSES), f"{aux_key} shape: {out[aux_key].shape}"
print("    Training forward -> dict with main + 3 aux  [OK]")


# ────────────────────────────────────────────────────────────────
# 4.  Eval forward (should return plain tensor)
# ────────────────────────────────────────────────────────────────
print("\n[4] Eval forward")
model.eval()
with torch.no_grad():
    out_eval = model(x)
assert torch.is_tensor(out_eval), f"Expected tensor, got {type(out_eval).__name__}"
assert out_eval.shape == (BATCH, NUM_CLASSES), f"shape: {out_eval.shape}"
print(f"    Eval forward -> tensor shape={tuple(out_eval.shape)}  [OK]")


# ────────────────────────────────────────────────────────────────
# 5.  Block deploy fusion  (eval vs deploy — BN uses running stats)
# ────────────────────────────────────────────────────────────────
print("\n[5] Block deploy fusion")
model.eval()
test_block = list(model.stage1.blocks.children())[0]
x_block = torch.randn(BATCH, 64, IMG_SIZE // 4, IMG_SIZE // 4, device=DEVICE)

with torch.no_grad():
    out_eval_block = test_block(x_block).detach().clone()

test_block.switch_to_deploy()
assert test_block.deploy, "Block should be in deploy mode"
test_block.eval()
with torch.no_grad():
    out_deploy_block = test_block(x_block).detach()

max_diff = float((out_eval_block - out_deploy_block).abs().max().item())
print(f"    eval vs deploy max_diff = {max_diff:.2e}")
assert max_diff < 1e-3, f"Fusion diff too large: {max_diff}"
print("    Block deploy fusion passed  [OK]")


# ────────────────────────────────────────────────────────────────
# 6.  Full model deploy conversion  (eval vs deploy)
# ────────────────────────────────────────────────────────────────
print("\n[6] Full model deploy conversion")
model2 = create_RepVGGplus_L2pse(num_classes=NUM_CLASSES, deploy=False, use_aux=True)
model2.to(DEVICE)
model2.eval()
with torch.no_grad():
    out_before = model2(x).detach().clone()

model2.switch_repvggplus_to_deploy()
assert model2.deploy, "Model should be in deploy mode"
assert model2.use_aux is False, "use_aux should be False after deploy"
assert model2.stage1_aux is None, "stage1_aux should be None after deploy"
assert model2.stage2_aux is None
assert model2.stage3_first_aux is None

model2.eval()
with torch.no_grad():
    out_after = model2(x)

assert torch.is_tensor(out_after), f"Expected tensor after deploy, got {type(out_after)}"
print(f"    Deploy forward -> tensor shape={tuple(out_after.shape)}  [OK]")

full_max_diff = float((out_before - out_after).abs().max().item())
print(f"    eval vs deploy max_diff = {full_max_diff:.2e}")
assert full_max_diff < 5e-3, f"Full model fusion diff too large: {full_max_diff}"
print("    Full model deploy fusion passed  [OK]")
print("    Aux heads stripped  [OK]")


# ────────────────────────────────────────────────────────────────
# 7.  get_model integration
# ────────────────────────────────────────────────────────────────
print("\n[7] get_model integration")
from model_mbv3 import get_model

model3 = get_model("repvggplus-l2pse", num_classes=7, pretrained=False, device=DEVICE)
model3.train()
out3 = model3(x)
assert isinstance(out3, dict) and "main" in out3
model3.eval()
with torch.no_grad():
    out3e = model3(x)
assert torch.is_tensor(out3e)
print("    get_model('repvggplus-l2pse') -> train dict + eval tensor  [OK]")


print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)