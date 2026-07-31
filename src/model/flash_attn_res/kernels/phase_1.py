import triton
import triton.language as tl
from .configs import (
    forward_configs,
    phase1_backward_configs,
    phase1_backward_v2_configs,
)


@triton.autotune(
    configs=forward_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
)
@triton.jit
def phase_1_batched_attention_forward_kernel(
    block_representations_ptr,
    pseudo_queries_ptr,
    softmax_normalized_output_ptr,
    lse_ptr,
    inverse_rms_norm_ptr,
    attention_logits_ptr,
    eps,
    num_active,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
    PADDED_HIDDEN: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)

    source_block_range = tl.arange(0, PADDED_SRC)[:, None]
    hidden_dim_range = tl.arange(0, PADDED_HIDDEN)[None, :]
    valid_block_mask_2d = source_block_range < num_active
    valid_hidden_mask_2d = hidden_dim_range < HIDDEN_DIM
    load_mask_2d = valid_block_mask_2d & valid_hidden_mask_2d

    valid_block_mask_1d = tl.arange(0, PADDED_SRC) < num_active
    valid_hidden_mask_1d = tl.arange(0, PADDED_HIDDEN) < HIDDEN_DIM

    source_block_values = tl.load(
        block_representations_ptr
        + source_block_range * (BT * HIDDEN_DIM)
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        mask=load_mask_2d,
        other=0.0,
    ).to(tl.float32)

    squared_norm_sum = tl.sum(source_block_values * source_block_values, axis=1)
    inverse_rms_norm = tl.rsqrt(squared_norm_sum / float(HIDDEN_DIM) + eps)

    source_block_range_1d = tl.arange(0, PADDED_SRC)
    source_block_range = source_block_range_1d[:, None]

    tl.store(
        inverse_rms_norm_ptr
        + batch_seq_idx * NUM_SOURCE_BLOCKS
        + source_block_range_1d,
        inverse_rms_norm,
        mask=valid_block_mask_1d,
    )

    hidden_dim_range_1d = tl.arange(0, PADDED_HIDDEN)

    for layer_offset in tl.static_range(NUM_QUERIES_PER_BLOCK):
        pseudo_query_vector = tl.load(
            pseudo_queries_ptr + layer_offset * HIDDEN_DIM + hidden_dim_range,
            mask=valid_hidden_mask_2d,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        raw_attention_logits = (
            tl.sum(source_block_values * pseudo_query_vector, axis=1) * inverse_rms_norm
        )

        tl.store(
            attention_logits_ptr
            + layer_offset * (BT * NUM_SOURCE_BLOCKS)
            + batch_seq_idx * NUM_SOURCE_BLOCKS
            + source_block_range_1d,
            raw_attention_logits,
            mask=valid_block_mask_1d,
        )

        attention_logits = tl.where(
            valid_block_mask_1d, raw_attention_logits, float("-inf")
        )

        max_attention_logit = tl.max(attention_logits)
        exp_attention_logits = tl.exp(attention_logits - max_attention_logit)
        exp_sum = tl.sum(exp_attention_logits)

        unnormalized_output = tl.sum(
            exp_attention_logits[:, None] * source_block_values, axis=0
        )
        normalized_output = (unnormalized_output / exp_sum).to(tl.bfloat16)

        tl.store(
            softmax_normalized_output_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
            normalized_output,
            mask=valid_hidden_mask_1d,
        )
        tl.store(
            lse_ptr + layer_offset * BT + batch_seq_idx,
            max_attention_logit + tl.log(exp_sum),
        )


@triton.autotune(
    configs=phase1_backward_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
    restore_value=[
        "grad_block_representations_accumulator_ptr",
        "grad_pseudo_queries_partial_ptr",
    ],
)
@triton.jit
def phase_1_batched_attention_backward_kernel(
    block_representations_ptr,
    pseudo_queries_ptr,
    lse_ptr,
    inverse_rms_norm_ptr,
    attention_logits_ptr,
    grad_softmax_normalized_output_ptr,
    grad_lse_ptr,
    grad_block_representations_accumulator_ptr,
    grad_pseudo_queries_partial_ptr,
    eps,
    num_active,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    BT: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
    HAS_GRAD_LSE: tl.constexpr,
    ACCUMULATE_GRAD_BLOCKS: tl.constexpr,
    PADDED_HIDDEN: tl.constexpr,
):
    batch_seq_idx = tl.program_id(0)

    source_block_range = tl.arange(0, PADDED_SRC)[:, None]
    source_block_range_1d = tl.arange(0, PADDED_SRC)

    hidden_dim_range = tl.arange(0, PADDED_HIDDEN)[None, :]
    hidden_dim_range_1d = tl.arange(0, PADDED_HIDDEN)

    valid_block_mask_2d = source_block_range < num_active
    valid_block_mask_1d = source_block_range_1d < num_active
    valid_hidden_mask_2d = hidden_dim_range < HIDDEN_DIM
    valid_hidden_mask_1d = hidden_dim_range_1d < HIDDEN_DIM
    load_mask_2d = valid_block_mask_2d & valid_hidden_mask_2d

    source_block_values = tl.load(
        block_representations_ptr
        + source_block_range * (BT * HIDDEN_DIM)
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range,
        mask=load_mask_2d,
        other=0.0,
    ).to(tl.float32)

    inverse_rms_norm = tl.load(
        inverse_rms_norm_ptr
        + batch_seq_idx * NUM_SOURCE_BLOCKS
        + source_block_range_1d,
        mask=valid_block_mask_1d,
        other=0.0,
    ).to(tl.float32)

    inverse_rms_norm_squared = inverse_rms_norm * inverse_rms_norm

    grad_source_accumulator = tl.zeros((PADDED_SRC, PADDED_HIDDEN), tl.float32)

    for layer_offset in tl.static_range(NUM_QUERIES_PER_BLOCK):
        pseudo_query_vector = tl.load(
            pseudo_queries_ptr + layer_offset * HIDDEN_DIM + hidden_dim_range,
            mask=valid_hidden_mask_2d,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        grad_attention_output = tl.load(
            grad_softmax_normalized_output_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
            mask=valid_hidden_mask_1d,
            other=0.0,
        ).to(tl.float32)

        if HAS_GRAD_LSE:
            grad_logsumexp = tl.load(
                grad_lse_ptr + layer_offset * BT + batch_seq_idx
            ).to(tl.float32)
        else:
            grad_logsumexp = 0.0

        forward_logsumexp = tl.load(lse_ptr + layer_offset * BT + batch_seq_idx).to(
            tl.float32
        )

        saved_attention_logits = tl.load(
            attention_logits_ptr
            + layer_offset * (BT * NUM_SOURCE_BLOCKS)
            + batch_seq_idx * NUM_SOURCE_BLOCKS
            + source_block_range_1d,
            mask=valid_block_mask_1d,
            other=0.0,
        ).to(tl.float32)

        attention_logits = tl.where(
            valid_block_mask_1d,
            saved_attention_logits,
            float("-inf"),
        )

        softmax_probabilities = tl.exp(attention_logits - forward_logsumexp)

        grad_output_dot_source_values = tl.sum(
            source_block_values * grad_attention_output[None, :],
            axis=1,
        )

        grad_output_dot_expected_value = tl.sum(
            softmax_probabilities * grad_output_dot_source_values,
            axis=0,
        )

        grad_attention_logits = softmax_probabilities * (
            grad_logsumexp
            + grad_output_dot_source_values
            - grad_output_dot_expected_value
        )

        grad_source_from_value_path = (
            softmax_probabilities[:, None] * grad_attention_output[None, :]
        )

        grad_source_from_logit_path = grad_attention_logits[:, None] * (
            inverse_rms_norm[:, None] * pseudo_query_vector
            - saved_attention_logits[:, None]
            * inverse_rms_norm_squared[:, None]
            * source_block_values
            / float(HIDDEN_DIM)
        )

        grad_source_block_values = (
            grad_source_from_value_path + grad_source_from_logit_path
        )

        grad_source_accumulator += tl.where(
            load_mask_2d,
            grad_source_block_values,
            0.0,
        )

        grad_pseudo_query = tl.sum(
            grad_attention_logits[:, None]
            * inverse_rms_norm[:, None]
            * source_block_values,
            axis=0,
        )

        tl.store(
            grad_pseudo_queries_partial_ptr
            + layer_offset * BT * HIDDEN_DIM
            + batch_seq_idx * HIDDEN_DIM
            + hidden_dim_range_1d,
            grad_pseudo_query,
            mask=valid_hidden_mask_1d,
        )

    grad_block_ptr = (
        grad_block_representations_accumulator_ptr
        + source_block_range * (BT * HIDDEN_DIM)
        + batch_seq_idx * HIDDEN_DIM
        + hidden_dim_range
    )

    if ACCUMULATE_GRAD_BLOCKS:
        prev_grad_block = tl.load(
            grad_block_ptr,
            mask=load_mask_2d,
            other=0.0,
        ).to(tl.float32)

        grad_source_accumulator += prev_grad_block

    tl.store(
        grad_block_ptr,
        grad_source_accumulator,
        mask=load_mask_2d,
    )


@triton.autotune(
    configs=phase1_backward_v2_configs,
    key=["NUM_SOURCE_BLOCKS", "HIDDEN_DIM", "NUM_QUERIES_PER_BLOCK", "PADDED_SRC"],
    # Different BLOCK_BT trials write different partials-row counts; restore
    # between autotune trials (same reasoning as phase-2 v2). The dot-scratch
    # buffer self-heals (first chunk STOREs, later chunks RMW), so it needs
    # no restore.
    restore_value=["grad_pseudo_queries_partials_ptr"],
)
@triton.jit
def phase_1_batched_attention_backward_v2_kernel(
    block_representations_ptr,
    pseudo_queries_ptr,
    lse_ptr,
    inverse_rms_norm_ptr,
    attention_logits_ptr,
    grad_softmax_normalized_output_ptr,
    grad_lse_ptr,
    grad_block_representations_ptr,
    grad_pseudo_queries_partials_ptr,
    dot_scratch_ptr,
    eps,
    num_active,
    BT,
    NUM_PROGRAMS,
    NUM_SOURCE_BLOCKS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES_PER_BLOCK: tl.constexpr,
    PADDED_SRC: tl.constexpr,
    HAS_GRAD_LSE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_BT: tl.constexpr,
):
    # v1 rewrite, same math, register-spill cure (phase-2 v2 lesson: v1 holds
    # a (PADDED_SRC, 4096) src tile PLUS a same-sized grad accumulator across
    # a NUM_QUERIES-long loop -> guaranteed spill).
    #
    #   Pass 1 (d-chunk outer, query inner): accumulate, into a small global
    #     scratch (NQ, BT, PADDED_SRC), the per-(query, token, source) dots
    #     grad_out . src_values. First chunk stores, later chunks
    #     read-modify-write (only this program touches its rows).
    #   Pass 2 (d-chunk outer, query inner): rebuild each query's scalar
    #     pipeline (probs, grad_logits) from the completed scratch — cheap
    #     (BLOCK_BT, PADDED_SRC) loads — and emit:
    #       grad_block_representations chunk (output dtype = bf16 in
    #         training; one bf16 round, identical to v1 fp32-store + .to()),
    #       per-PROGRAM pseudo-query grad partials, layout (NQ, NUM_PROGRAMS,
    #         D) (v1 wrote per-TOKEN (NQ, BT, D) fp32 + a reduce kernel —
    #         BLOCK_BT x the traffic).
    bt_block_idx = tl.program_id(0)

    bt_offsets = bt_block_idx * BLOCK_BT + tl.arange(0, BLOCK_BT)
    valid_bt = bt_offsets < BT

    src_range = tl.arange(0, PADDED_SRC)
    valid_src = src_range < num_active
    mask_bt_src = valid_bt[:, None] & valid_src[None, :]

    # ── Pass 1: dot accumulation into scratch ─────────────────────────
    for d_start in range(0, HIDDEN_DIM, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        valid_d = d_offsets < HIDDEN_DIM
        mask_src_3d = (
            valid_src[:, None, None]
            & valid_bt[None, :, None]
            & valid_d[None, None, :]
        )
        src_chunk = tl.load(
            block_representations_ptr
            + src_range[:, None, None] * (BT * HIDDEN_DIM)
            + bt_offsets[None, :, None] * HIDDEN_DIM
            + d_offsets[None, None, :],
            mask=mask_src_3d,
            other=0.0,
        ).to(tl.float32)

        for q in tl.static_range(NUM_QUERIES_PER_BLOCK):
            gm_chunk = tl.load(
                grad_softmax_normalized_output_ptr
                + q * (BT * HIDDEN_DIM)
                + bt_offsets[:, None] * HIDDEN_DIM
                + d_offsets[None, :],
                mask=valid_bt[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)

            # (PADDED_SRC, BLOCK_BT) dot contribution for this chunk
            dot_chunk = tl.sum(src_chunk * gm_chunk[None, :, :], axis=2)

            scratch_offs = (
                q * (BT * PADDED_SRC)
                + bt_offsets[None, :] * PADDED_SRC
                + src_range[:, None]
            )
            scratch_mask = valid_bt[None, :] & valid_src[:, None]
            if d_start == 0:
                tl.store(dot_scratch_ptr + scratch_offs, dot_chunk,
                         mask=scratch_mask)
            else:
                prev = tl.load(dot_scratch_ptr + scratch_offs,
                               mask=scratch_mask, other=0.0)
                tl.store(dot_scratch_ptr + scratch_offs, prev + dot_chunk,
                         mask=scratch_mask)

    # Row scalars shared by every query's pass-2 pipeline
    inverse_rms_norm = tl.load(
        inverse_rms_norm_ptr
        + bt_offsets[:, None] * NUM_SOURCE_BLOCKS
        + src_range[None, :],
        mask=mask_bt_src,
        other=0.0,
    ).to(tl.float32)
    inverse_rms_norm_squared = inverse_rms_norm * inverse_rms_norm

    # ── Pass 2: emit grads chunk-by-chunk ─────────────────────────────
    for d_start in range(0, HIDDEN_DIM, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        valid_d = d_offsets < HIDDEN_DIM
        mask_src_3d = (
            valid_src[:, None, None]
            & valid_bt[None, :, None]
            & valid_d[None, None, :]
        )
        src_offs_3d = (
            src_range[:, None, None] * (BT * HIDDEN_DIM)
            + bt_offsets[None, :, None] * HIDDEN_DIM
            + d_offsets[None, None, :]
        )
        src_chunk = tl.load(
            block_representations_ptr + src_offs_3d,
            mask=mask_src_3d,
            other=0.0,
        ).to(tl.float32)

        grad_src_chunk = tl.zeros(
            (PADDED_SRC, BLOCK_BT, BLOCK_D), dtype=tl.float32
        )

        for q in tl.static_range(NUM_QUERIES_PER_BLOCK):
            forward_logsumexp = tl.load(
                lse_ptr + q * BT + bt_offsets, mask=valid_bt, other=0.0,
            ).to(tl.float32)
            if HAS_GRAD_LSE:
                grad_logsumexp = tl.load(
                    grad_lse_ptr + q * BT + bt_offsets,
                    mask=valid_bt, other=0.0,
                ).to(tl.float32)
            else:
                grad_logsumexp = tl.zeros((BLOCK_BT,), dtype=tl.float32)

            saved_logits = tl.load(
                attention_logits_ptr
                + q * (BT * NUM_SOURCE_BLOCKS)
                + bt_offsets[:, None] * NUM_SOURCE_BLOCKS
                + src_range[None, :],
                mask=mask_bt_src, other=0.0,
            ).to(tl.float32)
            logits = tl.where(
                valid_src[None, :], saved_logits, float("-inf")
            )
            probs = tl.exp(logits - forward_logsumexp[:, None])

            dots = tl.load(
                dot_scratch_ptr
                + q * (BT * PADDED_SRC)
                + bt_offsets[:, None] * PADDED_SRC
                + src_range[None, :],
                mask=mask_bt_src, other=0.0,
            )
            dot_expected = tl.sum(probs * dots, axis=1)
            grad_logits = probs * (
                grad_logsumexp[:, None] + dots - dot_expected[:, None]
            )

            gm_chunk = tl.load(
                grad_softmax_normalized_output_ptr
                + q * (BT * HIDDEN_DIM)
                + bt_offsets[:, None] * HIDDEN_DIM
                + d_offsets[None, :],
                mask=valid_bt[:, None] & valid_d[None, :],
                other=0.0,
            ).to(tl.float32)
            query_chunk = tl.load(
                pseudo_queries_ptr + q * HIDDEN_DIM + d_offsets,
                mask=valid_d, other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)

            probs_t = tl.trans(probs)
            grad_logits_t = tl.trans(grad_logits)
            irms_t = tl.trans(inverse_rms_norm)
            irms2_t = tl.trans(inverse_rms_norm_squared)
            logits_t = tl.trans(saved_logits)

            grad_src_chunk += probs_t[:, :, None] * gm_chunk[None, :, :]
            grad_src_chunk += grad_logits_t[:, :, None] * (
                irms_t[:, :, None] * query_chunk[None, None, :]
                - logits_t[:, :, None]
                * irms2_t[:, :, None]
                * src_chunk
                / float(HIDDEN_DIM)
            )

            # pseudo-query partial: sum over (src, token) of
            # grad_logits * irms * src — one row per (q, program)
            coef = grad_logits_t * irms_t  # (PADDED_SRC, BLOCK_BT)
            gq_chunk = tl.sum(
                tl.sum(coef[:, :, None] * src_chunk, axis=0), axis=0
            )
            tl.store(
                grad_pseudo_queries_partials_ptr
                + q * (NUM_PROGRAMS * HIDDEN_DIM)
                + bt_block_idx * HIDDEN_DIM
                + d_offsets,
                gq_chunk,
                mask=valid_d,
            )

        tl.store(
            grad_block_representations_ptr + src_offs_3d,
            grad_src_chunk.to(
                grad_block_representations_ptr.dtype.element_ty
            ),
            mask=mask_src_3d,
        )
