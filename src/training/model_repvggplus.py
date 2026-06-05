from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

try:
    from .se_block import SEBlock
except ImportError:
    from se_block import SEBlock  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    """Conv2d + BatchNorm2d, no activation."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(out_channels))
    return result


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    """Conv2d + BatchNorm2d + ReLU."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(out_channels))
    result.add_module("relu", nn.ReLU(inplace=True))
    return result


# ---------------------------------------------------------------------------
# RepVGGplus Building Block
# ---------------------------------------------------------------------------
class RepVGGplusBlock(nn.Module):
    """Multi-branch training block that fuses to a single 3×3 conv at deploy.

    Training branches:
      - 3×3 conv + BN
      - 1×1 conv + BN
      - identity + BN (only when in==out and stride==1)

    Post-processing order:  branch-sum → ReLU → SE  (faithful to original).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deploy: bool = False,
        use_post_se: bool = False,
    ) -> None:
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_identity: Optional[nn.BatchNorm2d] = (
                nn.BatchNorm2d(in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels, out_channels, 1,
                stride=stride, padding=0, groups=groups,
            )

        self.nonlinearity = nn.ReLU(inplace=True)
        self.se: Optional[SEBlock] = (
            SEBlock(out_channels, out_channels // 4) if use_post_se else None
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            out = self.nonlinearity(self.rbr_reparam(x))
            if self.se is not None:
                out = self.se(out)
            return out

        id_out: Optional[torch.Tensor] = None
        if self.rbr_identity is not None:
            id_out = self.rbr_identity(x)

        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if id_out is not None:
            out = out + id_out

        out = self.nonlinearity(out)          # ReLU
        if self.se is not None:
            out = self.se(out)                # SE after ReLU
        return out

    # ------------------------------------------------------------------
    # Reparameterisation helpers
    # ------------------------------------------------------------------
    def _get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(
            self.rbr_identity
        ) if self.rbr_identity is not None else (
            torch.zeros(1, device=kernel3x3.device),
            torch.zeros(1, device=kernel3x3.device),
        )

        kernel1x1_padded = nn.functional.pad(kernel1x1, [1, 1, 1, 1])
        kernelid_padded: torch.Tensor
        if self.rbr_identity is not None:
            kernelid_padded = self._pad_identity_kernel(kernelid)
        else:
            kernelid_padded = torch.zeros_like(kernel3x3)

        return (
            kernel3x3 + kernel1x1_padded + kernelid_padded,
            bias3x3 + bias1x1 + biasid,
        )

    @staticmethod
    def _conv_from_branch(branch: nn.Sequential) -> nn.Conv2d:
        return branch.conv if hasattr(branch, "conv") else branch[0]  # type: ignore[return-value]

    @staticmethod
    def _bn_from_branch(branch: nn.Sequential) -> nn.BatchNorm2d:
        return branch.bn if hasattr(branch, "bn") else branch[1]  # type: ignore[return-value]

    @staticmethod
    def _fuse_bn_tensor(
        branch: nn.Sequential | nn.BatchNorm2d | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if branch is None:
            return torch.zeros(1), torch.zeros(1)

        if isinstance(branch, nn.BatchNorm2d):
            c = branch.num_features
            kernel = torch.zeros(
                c, c, 1, 1, dtype=branch.weight.dtype, device=branch.weight.device
            )
            for i in range(c):
                kernel[i, i, 0, 0] = 1.0
            bn_weight = branch.weight / torch.sqrt(branch.running_var + branch.eps)
            bn_bias = branch.bias - branch.running_mean * bn_weight
            return kernel * bn_weight.view(c, 1, 1, 1), bn_bias

        conv = RepVGGplusBlock._conv_from_branch(branch)
        bn = RepVGGplusBlock._bn_from_branch(branch)
        kernel = conv.weight.detach().clone()
        bn_weight = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_weight = bn_weight.view(-1, 1, 1, 1)
        bn_bias = bn.bias - bn.running_mean * bn_weight.flatten()
        return kernel * bn_weight, bn_bias

    @staticmethod
    def _pad_identity_kernel(kernel_id: torch.Tensor) -> torch.Tensor:
        c = kernel_id.shape[0]
        if kernel_id.ndim == 4 and kernel_id.shape[2] == 1:
            kernel_id = kernel_id.view(c, c, 1, 1)
        return nn.functional.pad(kernel_id, [1, 1, 1, 1])

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for attr in ("rbr_dense", "rbr_1x1", "rbr_identity"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True


# ---------------------------------------------------------------------------
# RepVGGplus Stage
# ---------------------------------------------------------------------------
class RepVGGplusStage(nn.Module):
    """A sequence of RepVGGplusBlocks, optionally with gradient checkpointing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_checkpoint: bool = False,
        use_post_se: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint

        blocks: List[nn.Module] = []
        # First block carries the stride.
        blocks.append(
            RepVGGplusBlock(
                in_channels,
                out_channels,
                stride=stride,
                deploy=deploy,
                use_post_se=use_post_se,
            )
        )
        for _ in range(1, num_blocks):
            blocks.append(
                RepVGGplusBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    deploy=deploy,
                    use_post_se=use_post_se,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                try:
                    x = checkpoint.checkpoint(block, x, use_reentrant=False)
                except Exception:
                    x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


# ---------------------------------------------------------------------------
# RepVGGplus Full Model
# ---------------------------------------------------------------------------
class RepVGGplus(nn.Module):
    """RepVGGplus adapted for FER (faithful to the original paper).

    Architecture (L2pse):
      stage0:       RepVGGplusBlock  (3 → 64,  stride=2)
      stage1:       8  RepVGGplus blocks  (64 → 160,  stride=2)
      stage2:       14 RepVGGplus blocks  (160 → 320, stride=2)
      stage3_first: 12 RepVGGplus blocks  (320 → 640, stride=2)
      stage3_second:12 RepVGGplus blocks  (640 → 640, stride=1)
      stage4:       1  RepVGGplus block   (640 → 2560,stride=2)
      head:         GAP → Flatten → Linear(num_classes)

    Aux classifiers after stage1 / stage2 / stage3_first when training.
    """

    def __init__(
        self,
        num_blocks: Sequence[int],
        num_classes: int,
        width_multiplier: Sequence[float],
        deploy: bool = False,
        use_post_se: bool = False,
        use_checkpoint: bool = False,
        use_aux: bool = True,
    ) -> None:
        super().__init__()
        if len(num_blocks) != 4:
            raise ValueError(
                f"num_blocks must have length 4, got {len(num_blocks)}"
            )
        if len(width_multiplier) != 4:
            raise ValueError(
                f"width_multiplier must have length 4, got {len(width_multiplier)}"
            )

        self.deploy = deploy
        self.use_aux = use_aux
        self.use_checkpoint = use_checkpoint
        self.num_classes = int(num_classes)

        base_channels = [64, 128, 256, 512]
        stage_channels = [
            int(base_channels[i] * width_multiplier[i]) for i in range(4)
        ]

        # ── Stage 0 ──────────────────────────────────────────────────
        in_channels = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGplusBlock(
            in_channels=3,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_post_se=use_post_se,
        )

        # ── Stage 1 ──────────────────────────────────────────────────
        self.stage1 = RepVGGplusStage(
            in_channels,
            stage_channels[0],
            num_blocks[0],
            stride=2,
            use_checkpoint=use_checkpoint,
            use_post_se=use_post_se,
            deploy=deploy,
        )

        # ── Stage 2 ──────────────────────────────────────────────────
        self.stage2 = RepVGGplusStage(
            stage_channels[0],
            stage_channels[1],
            num_blocks[1],
            stride=2,
            use_checkpoint=use_checkpoint,
            use_post_se=use_post_se,
            deploy=deploy,
        )

        # ── Stage 3 (split for aux head placement) ──────────────────
        half3 = num_blocks[2] // 2
        self.stage3_first = RepVGGplusStage(
            stage_channels[1],
            stage_channels[2],
            half3,
            stride=2,
            use_checkpoint=use_checkpoint,
            use_post_se=use_post_se,
            deploy=deploy,
        )
        self.stage3_second = RepVGGplusStage(
            stage_channels[2],
            stage_channels[2],
            num_blocks[2] - half3,
            stride=1,
            use_checkpoint=use_checkpoint,
            use_post_se=use_post_se,
            deploy=deploy,
        )

        # ── Stage 4 ──────────────────────────────────────────────────
        self.stage4 = RepVGGplusStage(
            stage_channels[2],
            stage_channels[3],
            num_blocks[3],
            stride=2,
            use_checkpoint=use_checkpoint,
            use_post_se=use_post_se,
            deploy=deploy,
        )

        # ── Classification head ──────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(stage_channels[3], num_classes)

        # ── Auxiliary classifiers ────────────────────────────────────
        self.stage1_aux: Optional[nn.Module] = None
        self.stage2_aux: Optional[nn.Module] = None
        self.stage3_first_aux: Optional[nn.Module] = None

        if not self.deploy and self.use_aux:
            self.stage1_aux = self._build_aux_for_stage(self.stage1)
            self.stage2_aux = self._build_aux_for_stage(self.stage2)
            self.stage3_first_aux = self._build_aux_for_stage(self.stage3_first)

    # ------------------------------------------------------------------
    # Auxiliary head builder (original paper structure)
    # ------------------------------------------------------------------
    def _build_aux_for_stage(self, stage: RepVGGplusStage) -> nn.Sequential:
        last_block = list(stage.blocks.children())[-1]
        # rbr_dense is nn.Sequential with named children: conv / bn.
        if hasattr(last_block.rbr_dense, "conv"):
            stage_out_channels = last_block.rbr_dense.conv.out_channels
        else:
            stage_out_channels = last_block.rbr_dense[0].out_channels

        downsample = conv_bn_relu(
            in_channels=stage_out_channels,
            out_channels=stage_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        fc = nn.Linear(stage_out_channels, self.num_classes, bias=True)
        return nn.Sequential(
            downsample,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            fc,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        out = self.stage0(x)

        out = self.stage1(out)
        stage1_aux: Optional[torch.Tensor] = None
        if self.training and not self.deploy and self.use_aux and self.stage1_aux is not None:
            stage1_aux = self.stage1_aux(out)

        out = self.stage2(out)
        stage2_aux: Optional[torch.Tensor] = None
        if self.training and not self.deploy and self.use_aux and self.stage2_aux is not None:
            stage2_aux = self.stage2_aux(out)

        out = self.stage3_first(out)
        stage3_first_aux: Optional[torch.Tensor] = None
        if self.training and not self.deploy and self.use_aux and self.stage3_first_aux is not None:
            stage3_first_aux = self.stage3_first_aux(out)

        out = self.stage3_second(out)
        out = self.stage4(out)

        y: torch.Tensor = self.gap(out)
        y = self.flatten(y)
        y = self.linear(y)

        if self.training and not self.deploy and self.use_aux:
            return {
                "main": y,
                "stage1_aux": stage1_aux,
                "stage2_aux": stage2_aux,
                "stage3_first_aux": stage3_first_aux,
            }

        return y

    # ------------------------------------------------------------------
    # Deploy conversion
    # ------------------------------------------------------------------
    def switch_repvggplus_to_deploy(self) -> None:
        """Convert every RepVGGplusBlock to deploy mode and strip aux heads."""
        if self.deploy:
            return

        for module in self.modules():
            if isinstance(module, RepVGGplusBlock) and not module.deploy:
                module.switch_to_deploy()

        for attr in ("stage1_aux", "stage2_aux", "stage3_first_aux"):
            if hasattr(self, attr):
                setattr(self, attr, None)

        self.deploy = True
        self.use_aux = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def repvgg_model_convert(
    model: nn.Module, save_path: Optional[str] = None
) -> nn.Module:
    """Convert a RepVGGplus model to deploy mode and optionally save it."""
    if hasattr(model, "switch_repvggplus_to_deploy"):
        model.switch_repvggplus_to_deploy()
    else:
        for module in model.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_RepVGGplus_L2pse(
    num_classes: int = 7,
    deploy: bool = False,
    use_checkpoint: bool = False,
    use_aux: bool = True,
) -> RepVGGplus:
    """Create a RepVGGplus-L2pse model for FER.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 7 for FER).
    deploy : bool
        Build the single-branch deploy graph directly.
    use_checkpoint : bool
        Enable gradient checkpointing in every stage (saves VRAM).
    use_aux : bool
        Attach auxiliary classifiers after stage1, stage2, stage3_first.
    """
    return RepVGGplus(
        num_blocks=[8, 14, 24, 1],
        num_classes=num_classes,
        width_multiplier=[2.5, 2.5, 2.5, 5.0],
        deploy=deploy,
        use_post_se=True,
        use_checkpoint=use_checkpoint,
        use_aux=use_aux,
    )
