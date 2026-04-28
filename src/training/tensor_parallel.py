"""
Tensor parallelism for kotodama pretraining.

Manual TP with conjugate autograd communication primitives:
- Column-parallel (Q/K/V, gate/up): identity forward, all-reduce backward
- Row-parallel (O, down): all-reduce forward, identity backward

TP linear layers remain plain nn.Linear with sharded weights, preserving
compatibility with FP8 (torchao), torch.compile, and Liger kernels.
Communication ops live in parent module forwards (GQAttention, SwiGLUFFN).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ── Autograd communication primitives ──────────────────────────────────────


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Identity forward, all-reduce backward.

    At attention/FFN input: backward sums partial gradients from
    sharded projections (Q/K/V, gate/up) across TP ranks.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce forward, identity backward.

    At o_proj/down_proj output: sums partial matmul results across
    TP ranks into Replicated hidden states.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=group)
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def copy_to_parallel_region(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Identity forward, all-reduce backward."""
    return _CopyToModelParallelRegion.apply(x, group)


def reduce_from_parallel_region(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """All-reduce forward, identity backward."""
    return _ReduceFromModelParallelRegion.apply(x, group)


# ── Process group creation ─────────────────────────────────────────────────


def create_process_groups(
    world_size: int,
    tp_size: int,
) -> tuple[dist.ProcessGroup, Optional[dist.ProcessGroup]]:
    """Create tensor-parallel and data-parallel process groups.

    TP groups use contiguous ranks: TP=4 on 8 GPUs → [0,1,2,3], [4,5,6,7].
    DP groups span same-position ranks across TP groups: [0,4], [1,5], ...

    Returns (tp_group, dp_group). dp_group is None when dp_size == 1.
    """
    if world_size % tp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
        )

    dp_size = world_size // tp_size
    rank = dist.get_rank()

    tp_group: Optional[dist.ProcessGroup] = None
    for start in range(0, world_size, tp_size):
        ranks = list(range(start, start + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    dp_group: Optional[dist.ProcessGroup] = None
    if dp_size > 1:
        for tp_pos in range(tp_size):
            ranks = list(range(tp_pos, world_size, tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                dp_group = group

    assert tp_group is not None
    return tp_group, dp_group


# ── Weight sharding helpers ────────────────────────────────────────────────


def _column_shard_linear(
    parent: torch.nn.Module,
    attr_name: str,
    tp_rank: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
) -> None:
    """Shard a Linear's weight along output dim (dim 0) in-place.

    Column-parallel: each rank holds out_features/tp_size rows.
    Used for Q, K, V, gate, up projections.
    """
    linear = getattr(parent, attr_name)
    if linear.out_features % tp_size != 0:
        raise ValueError(
            f"{attr_name}: out_features ({linear.out_features}) "
            f"not divisible by tp_size ({tp_size})"
        )

    shard_size = linear.out_features // tp_size
    start = tp_rank * shard_size

    sharded = linear.weight.data[start : start + shard_size, :].contiguous()
    linear.weight = torch.nn.Parameter(sharded)
    linear.out_features = shard_size

    linear.weight._tp_shard_dim = 0
    linear.weight._tp_group = tp_group
    linear.weight._tp_size = tp_size
    linear.weight._tp_rank = tp_rank


def _row_shard_linear(
    parent: torch.nn.Module,
    attr_name: str,
    tp_rank: int,
    tp_size: int,
    tp_group: dist.ProcessGroup,
) -> None:
    """Shard a Linear's weight along input dim (dim 1) in-place.

    Row-parallel: each rank holds in_features/tp_size columns.
    Used for O, down projections.
    """
    linear = getattr(parent, attr_name)
    if linear.in_features % tp_size != 0:
        raise ValueError(
            f"{attr_name}: in_features ({linear.in_features}) "
            f"not divisible by tp_size ({tp_size})"
        )

    shard_size = linear.in_features // tp_size
    start = tp_rank * shard_size

    sharded = linear.weight.data[:, start : start + shard_size].contiguous()
    linear.weight = torch.nn.Parameter(sharded)
    linear.in_features = shard_size

    linear.weight._tp_shard_dim = 1
    linear.weight._tp_group = tp_group
    linear.weight._tp_size = tp_size
    linear.weight._tp_rank = tp_rank


# ── Model-level TP application ─────────────────────────────────────────────


def apply_tensor_parallelism(
    model: torch.nn.Module,
    tp_group: dist.ProcessGroup,
) -> None:
    """Apply tensor parallelism to a LuxiaBaseModel.

    Shards projection weights in-place and sets TP metadata on modules.
    Linear layers remain nn.Linear — no custom subclasses — preserving
    compatibility with FP8, torch.compile, and Liger.

    Column-parallel (shard output dim): Q, K, V, gate, up
    Row-parallel (shard input dim): O, down
    Replicated (unchanged): embeddings, LM head, norms, AttnRes queries
    """
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)

    config = model.config
    if config.num_attention_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads ({config.num_attention_heads}) "
            f"must be divisible by tp_size ({tp_size})"
        )
    if config.num_kv_heads % tp_size != 0:
        raise ValueError(
            f"num_kv_heads ({config.num_kv_heads}) "
            f"must be divisible by tp_size ({tp_size})"
        )

    for layer in model.layers:
        attn = layer.attn
        ffn = layer.ffn

        # Attention: column-shard Q/K/V, row-shard O
        _column_shard_linear(attn, "q_proj", tp_rank, tp_size, tp_group)
        _column_shard_linear(attn, "k_proj", tp_rank, tp_size, tp_group)
        _column_shard_linear(attn, "v_proj", tp_rank, tp_size, tp_group)
        _row_shard_linear(attn, "o_proj", tp_rank, tp_size, tp_group)

        # MLP: column-shard gate/up, row-shard down
        _column_shard_linear(ffn, "gate_proj", tp_rank, tp_size, tp_group)
        _column_shard_linear(ffn, "up_proj", tp_rank, tp_size, tp_group)
        _row_shard_linear(ffn, "down_proj", tp_rank, tp_size, tp_group)

        # Adjust local head counts
        attn.num_heads = config.num_attention_heads // tp_size
        attn.num_kv_heads = config.num_kv_heads // tp_size
        attn.num_kv_groups = attn.num_heads // attn.num_kv_heads

        # Set TP group for communication in forward pass
        attn.tp_group = tp_group
        ffn.tp_group = tp_group

    if dist.get_rank() == 0:
        sharded = sum(
            p.numel() for p in model.parameters() if hasattr(p, "_tp_shard_dim")
        )
        replicated = sum(
            p.numel() for p in model.parameters() if not hasattr(p, "_tp_shard_dim")
        )
        logger.info(
            "TP=%d applied: %d sharded params (local), %d replicated params",
            tp_size,
            sharded,
            replicated,
        )


# ── Muon helper ────────────────────────────────────────────────────────────


def all_gather_along_dim(
    tensor: torch.Tensor,
    dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-gather tensor along a specific dimension across the TP group."""
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return tensor

    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)
