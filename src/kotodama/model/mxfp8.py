"""MXFP8 training linears (7B throughput program, SPEC §C2).

torchao 0.17 removed the dense MXLinear module (only the MoE-training path
kept a swap API), but the differentiable building block survived:
`_to_mxfp8_then_scaled_mm` quantizes input/weight/grad to MXFP8
(block_size=32, e4m3, per-32-element block scales — the native B200 tensor
core format) around all three GEMMs of a linear, with torch.compile support
(`allow_in_graph` on the autograd.Function). This module is the thin
nn.Linear wrapper torchao no longer ships.

Weights stay high-precision nn.Parameter (dynamic quantization per matmul),
so Muon/DDP/checkpointing see the exact same state as bf16 or tensorwise
Float8Linear training.

Tier-C lever: any use is gated by the SPEC §5 validation protocol
(smoke logits → canary loss overlay → proxy soak) before production.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

try:
    from torchao.prototype.mx_formats.config import ScaleCalculationMode
    from torchao.prototype.mx_formats.mx_linear import _to_mxfp8_then_scaled_mm
    from torchao.quantization.quantize_.common.kernel_preference import (
        KernelPreference,
    )

    MXFP8_AVAILABLE = True
except ImportError:
    MXFP8_AVAILABLE = False

# MXFP8 block scales cover 32 contiguous elements along the contraction dim.
_MX_BLOCK_SIZE = 32


class _GradContiguous(torch.autograd.Function):
    """Identity forward; makes the incoming gradient contiguous.

    torchao's triton_to_mxfp8_dim0 asserts contiguity on grad_output, but
    autograd can deliver expanded/strided grads (e.g. the stride-0 ones
    tensor from a .sum() loss). .contiguous() is a no-op when the grad
    already is — zero cost on the hot path.
    """

    @staticmethod
    def forward(ctx, t: torch.Tensor) -> torch.Tensor:
        return t

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        return grad.contiguous()


class MXFP8Linear(nn.Linear):
    """nn.Linear whose matmuls (fwd, dgrad, wgrad) run in MXFP8.

    RCEIL scale mode = cuBLAS's 1D-block-quantization rounding (the
    hardware-native recipe); KernelPreference.AUTO picks the best available
    cast/GEMM kernels per op.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = _to_mxfp8_then_scaled_mm(
            input.contiguous(),
            self.weight,
            KernelPreference.AUTO,
            ScaleCalculationMode.RCEIL,
        )
        out = _GradContiguous.apply(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "MXFP8Linear":
        with torch.device("meta"):
            new = cls(
                mod.in_features, mod.out_features, bias=mod.bias is not None
            )
        new.weight = mod.weight
        new.bias = mod.bias
        return new


def convert_to_mxfp8_training(model: nn.Module) -> int:
    """Swap every eligible nn.Linear for MXFP8Linear, in place.

    Mirrors convert_to_float8_training's default scope (all nn.Linear);
    skips (and logs) any layer whose contraction dims aren't divisible by
    the MX block size — none exist in the 7B/3B shapes, but a silent
    fallback beats a launch-time kernel error if shapes ever change.
    Returns the number of swapped modules.
    """
    if not MXFP8_AVAILABLE:
        raise ImportError(
            "MXFP8 requires torchao with prototype.mx_formats "
            "(needs the cu128 cp312 wheel: torchao==0.17.0+cu128)"
        )
    swapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is not nn.Linear:
                continue
            if (
                child.in_features % _MX_BLOCK_SIZE
                or child.out_features % _MX_BLOCK_SIZE
            ):
                logger.warning(
                    "MXFP8: skipping %s (%dx%d not divisible by %d)",
                    name,
                    child.in_features,
                    child.out_features,
                    _MX_BLOCK_SIZE,
                )
                continue
            setattr(parent, name, MXFP8Linear.from_float(child))
            swapped += 1
    logger.info("MXFP8: converted %d nn.Linear modules", swapped)
    return swapped
