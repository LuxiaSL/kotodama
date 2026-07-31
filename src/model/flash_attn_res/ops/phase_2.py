import os

import torch
import triton
from torch.library import triton_op, wrap_triton

from ..kernels import phase_2
from ..kernels.configs import PHASE2_BWD_MIN_BLOCK_BT

# v2 backward (bf16-native stores, deterministic pseudo-query reduce, no
# atomics, pipelined configs). Same math as v1 — grads that v1 stored fp32
# and then converted are rounded to the same bf16 values; grad_pseudo_query
# changes summation ORDER only (v1's atomic order was nondeterministic).
# Default ON since 2026-07-05: parity 31/31 x2, micro-bench 7.0-7.3x, canary
# +10.6% tok/s, loss overlay reconverged to +-0.001 vs same-seed baseline
# (SPEC Window-2 table). KOTODAMA_P2_BWD_V2=0 to fall back to v1.
_P2_BWD_V2 = os.environ.get("KOTODAMA_P2_BWD_V2", "1") == "1"


def phase_2_online_softmax_merge_triton_op(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    interblock_normalized_output: torch.Tensor,
    interblock_lse: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    merged_output, _phase2_intrablock_logit, _intrablock_inverse_rms_norm = (
        _phase_2_online_softmax_merge_forward_with_aux_triton_op(
            intrablock_partial_sum,
            pseudo_query,
            interblock_normalized_output,
            interblock_lse,
            eps,
        )
    )
    return merged_output


@triton_op(
    "flash_attn_res::_phase_2_online_softmax_merge_forward_with_aux",
    mutates_args={},
)
def _phase_2_online_softmax_merge_forward_with_aux_triton_op(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    interblock_normalized_output: torch.Tensor,
    interblock_lse: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = intrablock_partial_sum.shape
    BT = B * T

    merged_output = torch.empty(
        (B, T, D),
        device=interblock_normalized_output.device,
        dtype=interblock_normalized_output.dtype,
    )

    phase2_intrablock_logit = torch.empty(
        (B, T),
        device=intrablock_partial_sum.device,
        dtype=torch.float32,
    )

    intrablock_inverse_rms_norm = torch.empty(
        (B, T),
        device=intrablock_partial_sum.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_2.phase_2_online_softmax_merge_forward_kernel)[(BT,)](
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        merged_output,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
        eps,
        D,
        triton.next_power_of_2(D),
    )

    return merged_output, phase2_intrablock_logit, intrablock_inverse_rms_norm


@triton_op(
    "flash_attn_res::_phase_2_online_softmax_merge_backward",
    mutates_args={},
)
def _online_softmax_merge_backward_triton_op(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    phase1_interblock_normalized_output: torch.Tensor,
    phase1_interblock_logsumexp: torch.Tensor,
    phase2_intrablock_logit: torch.Tensor,
    intrablock_inverse_rms_norm: torch.Tensor,
    grad_merged_attention_output: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = intrablock_partial_sum.shape

    grad_intrablock_partial_sum = torch.empty(
        (B, T, D),
        device=intrablock_partial_sum.device,
        dtype=torch.float32,
    )

    grad_pseudo_query = torch.zeros(
        (D,),
        device=pseudo_query.device,
        dtype=torch.float32,
    )

    grad_phase1_interblock_normalized_output = torch.empty(
        (B, T, D),
        device=phase1_interblock_normalized_output.device,
        dtype=torch.float32,
    )

    grad_phase1_interblock_logsumexp = torch.empty(
        (B, T),
        device=phase1_interblock_logsumexp.device,
        dtype=torch.float32,
    )

    _online_softmax_merge_backward_accumulate(
        intrablock_partial_sum,
        pseudo_query,
        phase1_interblock_normalized_output,
        phase1_interblock_logsumexp,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
        grad_merged_attention_output,
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
        eps=eps,
        accumulate_grad_intrablock=False,
    )

    return (
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
    )


def _online_softmax_merge_backward_accumulate(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    phase1_interblock_normalized_output: torch.Tensor,
    phase1_interblock_logsumexp: torch.Tensor,
    phase2_intrablock_logit: torch.Tensor,
    intrablock_inverse_rms_norm: torch.Tensor,
    grad_merged_attention_output: torch.Tensor,
    grad_intrablock_partial_sum: torch.Tensor,
    grad_pseudo_query: torch.Tensor,
    grad_phase1_interblock_normalized_output: torch.Tensor,
    grad_phase1_interblock_logsumexp: torch.Tensor,
    eps: float,
    accumulate_grad_intrablock: bool,
) -> None:
    B, T, D = intrablock_partial_sum.shape
    BT = B * T

    wrap_triton(phase_2.phase_2_online_softmax_merge_backward_kernel)[
        lambda META: (triton.cdiv(BT, META["BLOCK_BT"]),)
    ](
        intrablock_partial_sum,
        pseudo_query,
        phase1_interblock_normalized_output,
        phase1_interblock_logsumexp,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
        grad_merged_attention_output,
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
        eps,
        BT,
        D,
        accumulate_grad_intrablock,
        triton.next_power_of_2(D),
    )


@triton_op(
    "flash_attn_res::_phase_2_online_softmax_merge_backward_v2",
    mutates_args={},
)
def _online_softmax_merge_backward_v2_triton_op(
    intrablock_partial_sum: torch.Tensor,
    pseudo_query: torch.Tensor,
    phase1_interblock_normalized_output: torch.Tensor,
    phase1_interblock_logsumexp: torch.Tensor,
    phase2_intrablock_logit: torch.Tensor,
    intrablock_inverse_rms_norm: torch.Tensor,
    grad_merged_attention_output: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, D = intrablock_partial_sum.shape
    BT = B * T

    # Grads for the two (B, T, D) inputs land directly in the input dtype
    # (v1 stored fp32 and the autograd wrapper converted afterwards — same
    # rounding, twice the store traffic plus two extra kernels).
    grad_intrablock_partial_sum = torch.empty(
        (B, T, D),
        device=intrablock_partial_sum.device,
        dtype=intrablock_partial_sum.dtype,
    )

    grad_phase1_interblock_normalized_output = torch.empty(
        (B, T, D),
        device=phase1_interblock_normalized_output.device,
        dtype=phase1_interblock_normalized_output.dtype,
    )

    grad_phase1_interblock_logsumexp = torch.empty(
        (B, T),
        device=phase1_interblock_logsumexp.device,
        dtype=torch.float32,
    )

    # Per-program partials, deterministically reduced below. Sized for the
    # smallest BLOCK_BT the autotuner may pick; zero-filled so rows beyond
    # the chosen grid contribute nothing.
    max_programs = triton.cdiv(BT, PHASE2_BWD_MIN_BLOCK_BT)
    grad_pseudo_query_partials = torch.zeros(
        (max_programs, D),
        device=pseudo_query.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_2.phase_2_online_softmax_merge_backward_v2_kernel)[
        lambda META: (triton.cdiv(BT, META["BLOCK_BT"]),)
    ](
        intrablock_partial_sum,
        pseudo_query,
        phase1_interblock_normalized_output,
        phase1_interblock_logsumexp,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
        grad_merged_attention_output,
        grad_intrablock_partial_sum,
        grad_pseudo_query_partials,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
        eps,
        BT,
        D,
    )

    grad_pseudo_query = grad_pseudo_query_partials.sum(dim=0)

    return (
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_phase1_interblock_normalized_output,
        grad_phase1_interblock_logsumexp,
    )


def setup_context(ctx, inputs, output):
    (
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        eps,
    ) = inputs

    (
        _merged_output,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
    ) = output

    ctx.save_for_backward(
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
    )
    ctx.eps = eps


def backward(
    ctx,
    grad_merged_output,
    _grad_phase2_intrablock_logit=None,
    _grad_intrablock_inverse_rms_norm=None,
):
    if grad_merged_output is None:
        return None, None, None, None, None

    (
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
    ) = ctx.saved_tensors

    if _P2_BWD_V2:
        (
            grad_intrablock_partial_sum,
            grad_pseudo_query,
            grad_interblock_normalized_output,
            grad_interblock_lse,
        ) = _online_softmax_merge_backward_v2_triton_op(
            intrablock_partial_sum,
            pseudo_query,
            interblock_normalized_output,
            interblock_lse,
            phase2_intrablock_logit,
            intrablock_inverse_rms_norm,
            grad_merged_output.contiguous(),
            ctx.eps,
        )
        # v2 emits grad_intrablock / grad_interblock_normalized in the input
        # dtype already; pseudo-query and lse grads are fp32.
        return (
            grad_intrablock_partial_sum if ctx.needs_input_grad[0] else None,
            (
                grad_pseudo_query.to(pseudo_query.dtype)
                if ctx.needs_input_grad[1]
                else None
            ),
            (
                grad_interblock_normalized_output
                if ctx.needs_input_grad[2]
                else None
            ),
            (
                grad_interblock_lse.to(interblock_lse.dtype)
                if ctx.needs_input_grad[3]
                else None
            ),
            None,
        )

    (
        grad_intrablock_partial_sum,
        grad_pseudo_query,
        grad_interblock_normalized_output,
        grad_interblock_lse,
    ) = _online_softmax_merge_backward_triton_op(
        intrablock_partial_sum,
        pseudo_query,
        interblock_normalized_output,
        interblock_lse,
        phase2_intrablock_logit,
        intrablock_inverse_rms_norm,
        grad_merged_output.contiguous(),
        ctx.eps,
    )

    return (
        (
            grad_intrablock_partial_sum.to(intrablock_partial_sum.dtype)
            if ctx.needs_input_grad[0]
            else None
        ),
        grad_pseudo_query.to(pseudo_query.dtype) if ctx.needs_input_grad[1] else None,
        (
            grad_interblock_normalized_output.to(interblock_normalized_output.dtype)
            if ctx.needs_input_grad[2]
            else None
        ),
        (
            grad_interblock_lse.to(interblock_lse.dtype)
            if ctx.needs_input_grad[3]
            else None
        ),
        None,
    )


_phase_2_online_softmax_merge_forward_with_aux_triton_op.register_autograd(
    backward,
    setup_context=setup_context,
)
