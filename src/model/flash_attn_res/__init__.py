"""Fused Triton kernels for Block Attention Residual routing.

Vendored from https://github.com/LuxiaSL/flash-attention-residuals (sublayer-routing branch).
Phase 1: inter-block attention (softmax over committed blocks).
Phase 2: online softmax merge with intra-block partial sum.
"""

from .ops.phase_1 import phase_1_batched_attention_triton_op
from .ops.phase_2 import phase_2_online_softmax_merge_triton_op
