"""LigerRMSNorm with launch settings tuned for B200 at the model's shapes: Liger's own kernels and math, other grids.

Liger launches its one-row kernels whenever the row is wider than 256 (forward: one row per program; backward: one
program per SM walking n_rows / n_SMs rows in sequence) and its block kernels (16 rows per tile, one program per SM)
below that. Measured cold on B200 (scripts/benchmark/rmsnorm_roofline.py, bf16, llama casting):
- hidden rows (32768 x 1536): block forward, 2 rows x 8 warps: 57 -> 46 us, Y bitwise equal; block backward,
  2 rows x 8 warps over 16 programs per SM: 232 -> 99 us, dX within 4 bf16 ulps on ~1e-6 of the elements.
- head rows (QK-norm, 393216 x 128): block backward, 16 rows x 2 warps over 16 programs per SM: 274 -> 90 us, dX bitwise
  equal. (Liger's forward there is already near bandwidth and is kept.)
dW sums its per-program partials over more programs (reduction grouping only). Other shapes / casting modes fall back
to Liger's launch.

``AVAILABLE`` is False when liger-kernel's internals (0.7.x layout) cannot be imported; callers then use LigerRMSNorm.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    from liger_kernel.ops.rms_norm import (
        LigerRMSNormFunction,
        _block_rms_norm_backward_kernel,
        _block_rms_norm_forward_kernel,
        rms_norm_backward,
        rms_norm_forward,
    )
    from liger_kernel.ops.utils import calculate_settings, ensure_contiguous, torch_to_triton_dtype
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    AVAILABLE = True
except Exception:  # liger missing, or its internals moved
    AVAILABLE = False

_LLAMA = 0  # liger _CASTING_MODE_LLAMA
_MIN_ROWS = 4096 * 8  # below this Liger's row kernels are the right launch (few rows)
# (block_row, num_warps, programs per SM; 0 = one program per row tile)
_FWD_WIDE = (2, 8, 0)
_BWD_WIDE = (2, 8, 16)
_BWD_NARROW = (16, 2, 16)


def _casting_int(casting_mode) -> Optional[int]:
    if isinstance(casting_mode, int):
        return casting_mode
    return {"llama": 0, "gemma": 1, "none": -1}.get(casting_mode)


def tuned_forward(X: torch.Tensor, W: Optional[torch.Tensor], eps: float, offset: float, casting_mode, row_mode):
    """rms_norm_forward with the tuned block launch for wide llama-cast rows (else Liger's own launch)."""
    shape = X.shape
    dim = shape[-1]
    X2 = X.view(-1, dim)
    n_rows, n_cols = X2.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    cm = _casting_int(casting_mode)
    if W is None or cm != _LLAMA or row_mode or BLOCK_SIZE <= 256 or n_rows < _MIN_ROWS:
        return rms_norm_forward(X, W, eps, offset, casting_mode, row_mode)
    block_row, warps, _ = _FWD_WIDE
    Y = torch.empty((n_rows, n_cols), dtype=X2.dtype, device=X2.device)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X2.device)
    _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, block_row),)](
        Y, Y.stride(0), X2, X2.stride(0), W, W.stride(0), RSTD, RSTD.stride(0), n_rows, n_cols, eps, offset, cm,
        elementwise_affine=True, BLOCK_SIZE=BLOCK_SIZE, num_warps=warps, BLOCK_ROW=block_row)
    return Y.view(*shape), X2, RSTD, BLOCK_SIZE, num_warps, cm


def tuned_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode):
    """rms_norm_backward with the tuned block launches (llama casting, affine, many rows; else Liger's own)."""
    shape = dY.shape
    dim = shape[-1]
    dY2 = dY.view(-1, dim)
    n_rows, n_cols = dY2.shape
    cm = _casting_int(casting_mode)
    if W is None or cm != _LLAMA or row_mode or n_rows < _MIN_ROWS:
        return rms_norm_backward(dY, X, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place, row_mode)
    block_row, warps, per_sm = _BWD_WIDE if BLOCK_SIZE > 256 else _BWD_NARROW
    n_tiles = triton.cdiv(n_rows, block_row)
    sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    grid = min(n_tiles, sms * per_sm)
    dX = dY2 if in_place else torch.empty_like(dY2)
    dW_part = torch.empty((grid, n_cols), dtype=torch.float32, device=W.device)
    _block_rms_norm_backward_kernel[(grid,)](
        dY2, dY2.stride(0), dX, dX.stride(0), X, X.stride(0), torch_to_triton_dtype[X.dtype], W, W.stride(0),
        RSTD, RSTD.stride(0), dW_part, dW_part.stride(0), n_rows, n_cols, offset, cm,
        elementwise_affine=True, BLOCK_SIZE=BLOCK_SIZE, num_warps=warps, BLOCK_ROW=block_row)
    return dX.view(*shape), dW_part.sum(dim=0).to(W.dtype)


if AVAILABLE:

    class LigerRMSNormTunedFunction(LigerRMSNormFunction):
        @staticmethod
        @ensure_contiguous
        def forward(ctx, X, W, eps, offset=0.0, casting_mode="llama", in_place=True, row_mode=None):
            Y, X, RSTD, BLOCK_SIZE, num_warps, casting_mode = tuned_forward(X, W, eps, offset, casting_mode, row_mode)
            ctx.offset = offset
            ctx.casting_mode = casting_mode
            ctx.in_place = in_place
            ctx.row_mode = row_mode
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.elementwise_affine = W is not None
            if W is not None:
                ctx.save_for_backward(X, W, RSTD)
            else:
                ctx.save_for_backward(X, RSTD)
            return Y

        @staticmethod
        @ensure_contiguous
        def backward(ctx, dY):
            if ctx.elementwise_affine:
                X, W, RSTD = ctx.saved_tensors
            else:
                X, RSTD = ctx.saved_tensors
                W = None
            dX, dW = tuned_backward(dY, X, W, RSTD, ctx.offset, ctx.casting_mode, ctx.BLOCK_SIZE, ctx.num_warps,
                                    ctx.in_place, ctx.row_mode)
            return dX, dW, None, None, None, None, None

    class LigerRMSNormTuned(LigerRMSNorm):
        """LigerRMSNorm (same parameters, kernels and math) with the tuned launches."""

        def forward(self, hidden_states):
            return LigerRMSNormTunedFunction.apply(hidden_states, self.weight, self.variance_epsilon, self.offset,
                                                   self.casting_mode, self.in_place, self.row_mode)
