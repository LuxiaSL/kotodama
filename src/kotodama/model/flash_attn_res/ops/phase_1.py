import os

import torch
import triton
from torch.library import triton_op, wrap_triton

from ..kernels import phase_1
from ..kernels.configs import PHASE1_BWD_MIN_BLOCK_BT
from ..kernels.reduce import reduce_grad_queries_kernel

EPS = torch.finfo(torch.float32).eps

# Chunked two-pass backward (register-spill fix, per-program query partials,
# no reduce kernel). Default OFF until its parity gate passes; enable with
# KOTODAMA_P1_BWD_V2=1. Same playbook as phase-2 v2 (2026-07-05).
_P1_BWD_V2 = os.environ.get("KOTODAMA_P1_BWD_V2", "0") == "1"

# Torch-native backward: the whole gradient factors into batched GEMMs
# (einsums) + tiny elementwise glue, so cuBLAS does the heavy lifting and
# Inductor fuses the rest into the compiled backward graph. Written per the
# Window-2 redesign brief (SPEC §"p1-bwd/fwd redesign notes") as the
# baseline-to-beat before any further Triton work. Takes precedence over
# _P1_BWD_V2 when both are set.
#
# Measured (2026-07-09, eager micro-bench): ~flat in S vs v1's superlinear
# spill blowup — 2.1x faster at S=7, 2.8x SLOWER at S=3. num_active is a
# trace-time int (compile specializes per call site), so dispatch is hybrid:
# torch for num_active >= MIN_S, v1 below. MIN_S=5 puts the crossover
# between the measured S=3 (v1 wins) and S=7 (torch wins) points.
_P1_BWD_TORCH = os.environ.get("KOTODAMA_P1_BWD_TORCH", "0") == "1"
_P1_BWD_TORCH_MIN_S = int(os.environ.get("KOTODAMA_P1_BWD_TORCH_MIN_S", "5"))


def _batched_attention_backward_torch(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    lses: torch.Tensor,
    inverse_rms_norms: torch.Tensor,
    attention_logits: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
    grad_lses: torch.Tensor,
    has_grad_lses: bool,
    eps: float,
    num_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM-decomposed backward. Same math as v1 up to fp reassociation.

    Forward being differentiated (per query n, source block s, token (b,t)):
        irms[b,t,s]   = rsqrt(mean_d(src^2) + eps)
        logits[n,b,t,s] = (src[s,b,t,:] . q[n,:]) * irms[b,t,s]
        lse[n,b,t]    = logsumexp_s(logits)
        out[n,b,t,:]  = sum_s softmax_s(logits) * src[s,b,t,:]

    Columns of the saved aux tensors at s >= num_active are uninitialized
    (torch.empty in the forward) — everything below slices to the active
    prefix first.
    """
    num_source_blocks = block_representations.shape[0]
    B, T, D = block_representations.shape[1:]

    src = block_representations[:num_active]  # (S,B,T,D) bf16
    logits = attention_logits[..., :num_active]  # (NQ,B,T,S) fp32
    irms = inverse_rms_norms[..., :num_active]  # (B,T,S) fp32
    probs = torch.exp(logits - lses.unsqueeze(-1))  # (NQ,B,T,S) fp32

    gm = grad_softmax_outputs.to(src.dtype)  # (NQ,B,T,D)

    # dot(grad_out, src) per (n,b,t,s): batched (NQ,D)@(D,S) GEMMs over (b,t),
    # bf16 inputs / fp32 accumulation in cuBLAS.
    g_probs = torch.einsum("nbtd,sbtd->nbts", gm, src).float()

    # Softmax backward + the lse gradient path (d lse/d logits = probs).
    grad_logits = probs * (
        g_probs - (probs * g_probs).sum(dim=-1, keepdim=True)
    )
    if has_grad_lses:
        grad_logits = grad_logits + probs * grad_lses.unsqueeze(-1)

    # grad_src, three terms:
    #   value path:   sum_n probs * grad_out
    #   query path:   irms * sum_n grad_logits * q
    #   irms path:    -(irms^2 / D) * (sum_n grad_logits * logits) * src
    # (dot = logits/irms, so dot * d irms/d src = -logits * irms^2 * src / D.)
    grad_src = torch.einsum("nbts,nbtd->sbtd", probs.to(src.dtype), gm)
    gl_irms = grad_logits * irms.unsqueeze(0)  # (NQ,B,T,S) fp32
    grad_src = grad_src + torch.einsum(
        "nbts,nd->sbtd", gl_irms.to(src.dtype), pseudo_queries.to(src.dtype)
    )
    irms_coef = (
        (irms * irms / D) * (grad_logits * logits).sum(dim=0)
    )  # (B,T,S) fp32
    grad_src = grad_src - (
        irms_coef.permute(2, 0, 1).unsqueeze(-1).to(src.dtype) * src
    )

    # grad_q[n,:] = sum_{b,t,s} grad_logits * irms * src — one big contraction.
    # fp32 inputs: bf16 rounding here costs ~3e-3 rel error on the query grads
    # (measured) because the sum runs over B*T*S elements; the output is tiny
    # (NQ, D) so the extra fp32 read bandwidth is negligible.
    grad_pseudo_queries = torch.einsum(
        "nbts,sbtd->nd", gl_irms, src.float()
    )

    grad_block_representations = grad_src.to(block_representations.dtype)
    if num_active < num_source_blocks:
        pad = torch.zeros(
            (num_source_blocks - num_active, B, T, D),
            device=grad_block_representations.device,
            dtype=grad_block_representations.dtype,
        )
        grad_block_representations = torch.cat(
            [grad_block_representations, pad], dim=0
        )
    return grad_block_representations, grad_pseudo_queries


@triton_op(
    "flash_attn_res::_phase_1_batched_attention_backward_v2",
    mutates_args={},
)
def _batched_attention_backward_v2_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    lses: torch.Tensor,
    inverse_rms_norms: torch.Tensor,
    attention_logits: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
    grad_lses: torch.Tensor,
    has_grad_lses: bool,
    eps: float,
    num_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]
    BT = B * T
    padded_src = triton.next_power_of_2(num_source_blocks)

    # Grads for the source blocks land directly in the input dtype (one bf16
    # round, identical to v1's fp32-store + .to() conversion).
    grad_block_representations = torch.empty(
        (num_source_blocks, B, T, D),
        device=block_representations.device,
        dtype=block_representations.dtype,
    )

    max_programs = triton.cdiv(BT, PHASE1_BWD_MIN_BLOCK_BT)
    grad_pseudo_queries_partials = torch.zeros(
        (num_queries, max_programs, D),
        device=pseudo_queries.device,
        dtype=torch.float32,
    )

    # Per-(query, token, src) dot sums; pass 1 fills it (first-chunk store,
    # so no zero-init needed), pass 2 consumes it.
    dot_scratch = torch.empty(
        (num_queries, BT, padded_src),
        device=block_representations.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_1.phase_1_batched_attention_backward_v2_kernel)[
        lambda META: (triton.cdiv(BT, META["BLOCK_BT"]),)
    ](
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries_partials,
        dot_scratch,
        eps,
        num_active,
        BT,
        max_programs,
        num_source_blocks,
        D,
        num_queries,
        padded_src,
        has_grad_lses,
    )

    grad_pseudo_queries = grad_pseudo_queries_partials.sum(dim=1)

    return grad_block_representations, grad_pseudo_queries


@triton_op(
    "flash_attn_res::_phase_1_batched_attention_forward_with_aux",
    mutates_args={},
)
def _phase_1_batched_attention_forward_with_aux_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    eps: float,
    num_active: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]
    BT = B * T

    softmax_outputs = torch.empty(
        (num_queries, B, T, D),
        device=block_representations.device,
        dtype=torch.bfloat16,
    )

    lses = torch.empty(
        (num_queries, B, T),
        device=block_representations.device,
        dtype=torch.float32,
    )

    inverse_rms_norms = torch.empty(
        (B, T, num_source_blocks),
        device=block_representations.device,
        dtype=torch.float32,
    )

    attention_logits = torch.empty(
        (num_queries, B, T, num_source_blocks),
        device=block_representations.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_1.phase_1_batched_attention_forward_kernel)[(BT,)](
        block_representations,
        pseudo_queries,
        softmax_outputs,
        lses,
        inverse_rms_norms,
        attention_logits,
        eps,
        num_active,
        num_source_blocks,
        BT,
        D,
        num_queries,
        triton.next_power_of_2(num_source_blocks),
        triton.next_power_of_2(D),
    )

    return softmax_outputs, lses, inverse_rms_norms, attention_logits


def phase_1_batched_attention_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    eps: float,
    num_active: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_active is None:
        num_active = block_representations.shape[0]
    softmax_outputs, lses, _inverse_rms_norms, _attention_logits = (
        _phase_1_batched_attention_forward_with_aux_triton_op(
            block_representations,
            pseudo_queries,
            eps,
            num_active,
        )
    )
    return softmax_outputs, lses


@triton_op(
    "flash_attn_res::_phase_1_batched_attention_backward",
    mutates_args={},
)
def _batched_attention_backward_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    lses: torch.Tensor,
    inverse_rms_norms: torch.Tensor,
    attention_logits: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
    grad_lses: torch.Tensor,
    has_grad_lses: bool,
    eps: float,
    num_active: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]

    grad_block_representations = torch.empty(
        (num_source_blocks, B, T, D),
        device=block_representations.device,
        dtype=torch.float32,
    )

    grad_pseudo_queries = torch.zeros(
        (num_queries, D),
        device=pseudo_queries.device,
        dtype=torch.float32,
    )

    grad_pseudo_queries_partial = torch.empty(
        (num_queries, B, T, D),
        device=pseudo_queries.device,
        dtype=torch.float32,
    )

    _batched_attention_backward_accumulate(
        block_representations,
        pseudo_queries,
        lses,
        grad_softmax_outputs,
        grad_lses if has_grad_lses else None,
        grad_block_representations,
        grad_pseudo_queries,
        grad_pseudo_queries_partial,
        eps,
        False,
        inverse_rms_norms,
        attention_logits,
        num_active,
    )

    return grad_block_representations, grad_pseudo_queries


def _batched_attention_backward_accumulate(
    block_representations,
    pseudo_queries,
    lses,
    grad_softmax_outputs,
    grad_lses,
    grad_block_representations,
    grad_pseudo_queries,
    grad_pseudo_queries_partial,
    eps,
    accumulate_grad_blocks,
    inverse_rms_norms,
    attention_logits,
    num_active=None,
) -> None:
    num_source_blocks = block_representations.shape[0]
    if num_active is None:
        num_active = num_source_blocks
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]
    BT = B * T

    has_grad_lses = grad_lses is not None

    if grad_lses is None:
        grad_lses = lses

    wrap_triton(phase_1.phase_1_batched_attention_backward_kernel)[(BT,)](
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries_partial,
        eps,
        num_active,
        num_source_blocks,
        BT,
        D,
        num_queries,
        triton.next_power_of_2(num_source_blocks),
        has_grad_lses,
        accumulate_grad_blocks,
        triton.next_power_of_2(D),
    )

    wrap_triton(reduce_grad_queries_kernel)[
        lambda META: (
            triton.cdiv(BT, META["BLOCK_BATCH_SEQ"]),
            num_queries,
            triton.cdiv(D, META["BLOCK_HIDDEN"]),
        )
    ](
        grad_pseudo_queries_partial,
        grad_pseudo_queries,
        BT,
        D,
    )


def setup_context(ctx, inputs, output):
    block_representations, pseudo_queries, eps, num_active = inputs
    _softmax_outputs, lses, inverse_rms_norms, attention_logits = output

    ctx.save_for_backward(
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
    )
    ctx.eps = eps
    ctx.num_active = num_active


def backward(
    ctx,
    grad_softmax_outputs,
    grad_lses,
    _grad_inverse_rms_norms,
    _grad_attention_logits,
):
    (
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
    ) = ctx.saved_tensors

    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]

    if grad_softmax_outputs is None:
        grad_softmax_outputs = torch.zeros(
            (num_queries, B, T, D),
            device=block_representations.device,
            dtype=torch.float32,
        )
    else:
        grad_softmax_outputs = grad_softmax_outputs.contiguous()

    has_grad_lses = grad_lses is not None

    if grad_lses is None:
        grad_lses = lses
    else:
        grad_lses = grad_lses.contiguous()

    if _P1_BWD_TORCH and ctx.num_active >= _P1_BWD_TORCH_MIN_S:
        grad_block_representations, grad_pseudo_queries = (
            _batched_attention_backward_torch(
                block_representations,
                pseudo_queries,
                lses,
                inverse_rms_norms,
                attention_logits,
                grad_softmax_outputs,
                grad_lses,
                has_grad_lses,
                ctx.eps,
                ctx.num_active,
            )
        )
        # Same output contract as v2: block grads already in input dtype.
        return (
            grad_block_representations if ctx.needs_input_grad[0] else None,
            (
                grad_pseudo_queries.to(pseudo_queries.dtype)
                if ctx.needs_input_grad[1]
                else None
            ),
            None,
            None,
        )

    if _P1_BWD_V2:
        grad_block_representations, grad_pseudo_queries = (
            _batched_attention_backward_v2_triton_op(
                block_representations,
                pseudo_queries,
                lses,
                inverse_rms_norms,
                attention_logits,
                grad_softmax_outputs,
                grad_lses,
                has_grad_lses,
                ctx.eps,
                ctx.num_active,
            )
        )
        # v2 emits block grads in the input dtype already; query grads fp32.
        return (
            grad_block_representations if ctx.needs_input_grad[0] else None,
            (
                grad_pseudo_queries.to(pseudo_queries.dtype)
                if ctx.needs_input_grad[1]
                else None
            ),
            None,
            None,
        )

    grad_block_representations, grad_pseudo_queries = (
        _batched_attention_backward_triton_op(
            block_representations,
            pseudo_queries,
            lses,
            inverse_rms_norms,
            attention_logits,
            grad_softmax_outputs,
            grad_lses,
            has_grad_lses,
            ctx.eps,
            ctx.num_active,
        )
    )

    return (
        (
            grad_block_representations.to(block_representations.dtype)
            if ctx.needs_input_grad[0]
            else None
        ),
        (
            grad_pseudo_queries.to(pseudo_queries.dtype)
            if ctx.needs_input_grad[1]
            else None
        ),
        None,
        None,
    )


_phase_1_batched_attention_forward_with_aux_triton_op.register_autograd(
    backward,
    setup_context=setup_context,
)
