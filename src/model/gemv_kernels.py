"""Custom Triton GEMV for single-stream decode (M=1).

Replaces inductor's prologue-fused reduction-mm at the decode shapes where
its template choice runs far below HBM bandwidth. Measured (round-2 baseline
profile, B200): qkv-proj fusion 20% BW, o-proj 8-33% BW — while the
gate_up/down/lm_head fusions hit 77-84% and STAY on the inductor path
(the custom kernel measured at-or-below them once un-fused epilogues are
priced in; see scripts/benchmark/gemv_microbench.py).

The op is wrapped in torch.library.custom_op (pure, no mutation) so
torch.compile schedules it as an opaque extern node — no functionalization
hazards, capture-safe under cudagraph trees. fp32 accumulation = same
rounding class as cuBLAS/inductor mm (sweep measured identical max-err vs
fp64 ground truth: 0.0039 on both).

Opt-in via KOTODAMA_CUSTOM_GEMV (read at import): '0'/unset = off (every
call site stays on F.linear / the inductor path), 'all' = every tuned
shape, or a comma list of site names. Default OFF — the deploy config
enables it explicitly once validated.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# name -> ((N, K), (BLOCK_N, BLOCK_K, num_warps, num_stages))
# Tuned on B200 via gemv_microbench.py full sweep (2026-06-11,
# results/profiles/gemv_sweep_gpu3.json — rankings flat across num_stages and
# stable across the top-5 per shape, so these are not noise winners).
_SHAPES: dict[str, tuple[tuple[int, int], tuple[int, int, int, int]]] = {
    "qkv":     ((5120, 3072),  (8, 1024, 4, 3)),   # 5.7us vs 19.5us inductor norm+mm
    "o":       ((3072, 3072),  (4, 1024, 4, 2)),   # 4.4us vs 7.1-31us inductor mm
    "gate_up": ((16384, 3072), (4, 1024, 4, 2)),   # 14.8us ~= 15.6us fused (wash alone)
    "down":    ((3072, 8192),  (4, 1024, 4, 4)),   # 8.8us ~= 8.2us fused (wash alone)
    "lm_head": ((49152, 3072), (4, 1024, 4, 3)),   # 45.2us ~= 47.7us fused (wash alone)
}


def _parse_enabled() -> frozenset[str]:
    """KOTODAMA_CUSTOM_GEMV: '0'/'' = off, '1'/'all' = every tuned shape,
    or a comma list of site names ('qkv,o').

    Round-2 finding that motivates the granularity: replacing a SUBSET of mms
    lets inductor re-fuse the displaced routing/norm prologues into whichever
    mm remains, recreating the same bad reduction-mm template elsewhere
    (measured: qkv+o-only moved the 19.5us pathology from qkv onto down_proj
    and regressed 410 -> 327 tok/s). 'all' leaves no mm to poison.
    """
    raw = os.environ.get("KOTODAMA_CUSTOM_GEMV", "0").strip().lower()
    if raw in ("", "0", "off", "false"):
        return frozenset()
    if raw in ("1", "all", "true"):
        return frozenset(_SHAPES)
    names = frozenset(p.strip() for p in raw.split(",") if p.strip())
    unknown = names - set(_SHAPES)
    if unknown:
        logger.warning("KOTODAMA_CUSTOM_GEMV: unknown site(s) %s ignored (have %s)",
                       sorted(unknown), sorted(_SHAPES))
    return names & set(_SHAPES)


_ENABLED_SITES = _parse_enabled()
_TUNED_CONFIGS: dict[tuple[int, int], tuple[int, int, int, int]] = {
    shape: cfg for name, (shape, cfg) in _SHAPES.items() if name in _ENABLED_SITES
}
_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    _AVAILABLE = True
except Exception as _e:  # pragma: no cover — CPU-only envs
    logger.warning("triton unavailable (%s) — custom decode GEMV disabled", _e)


if _AVAILABLE:

    @triton.jit
    def _gemv_kernel(
        w_ptr, x_ptr, y_ptr,
        K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """y[n] = sum_k W[n, k] * x[k]; fp32 accumulate, bf16 out.

        No bounds masks: N % BLOCK_N == 0 and K % BLOCK_K == 0 are guaranteed
        by the host-side config table (all decode shapes divide cleanly).
        """
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        w_base = w_ptr + rn[:, None] * K
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            xv = tl.load(x_ptr + k0 + rk).to(tl.float32)
            wv = tl.load(w_base + k0 + rk[None, :]).to(tl.float32)
            acc += tl.sum(wv * xv[None, :], axis=1)
        tl.store(y_ptr + rn, acc.to(tl.bfloat16))

    @torch.library.custom_op("kotodama::decode_gemv", mutates_args=())
    def _decode_gemv_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n, k = w.shape
        if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise TypeError(f"decode_gemv expects bf16, got x={x.dtype} w={w.dtype}")
        if w.stride(1) != 1 or w.stride(0) != k:
            raise ValueError("decode_gemv expects a contiguous row-major weight")
        cfg = _TUNED_CONFIGS.get((n, k))
        if cfg is None:
            raise KeyError(f"decode_gemv has no tuned config for shape ({n}, {k})")
        bn, bk, nw, ns = cfg
        x_flat = x.contiguous().view(k)
        y = torch.empty((n,), device=x.device, dtype=torch.bfloat16)
        _gemv_kernel[(n // bn,)](
            w, x_flat, y, K=k, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=ns
        )
        return y.view(*x.shape[:-1], n)

    @_decode_gemv_op.register_fake
    def _decode_gemv_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], w.shape[0]))


def use_custom_gemv(w: torch.Tensor) -> bool:
    """True if this weight's decode GEMV should use the custom kernel.

    Evaluated at trace time (shapes are static in the decode graph), so the
    branch costs nothing at replay.
    """
    if not _AVAILABLE:
        return False
    n, k = w.shape
    return (n, k) in _TUNED_CONFIGS and n * k < 2**31


def decode_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Custom GEMV: x (..., K) bf16 @ w (N, K) bf16 -> (..., N) bf16."""
    return _decode_gemv_op(x, w)
