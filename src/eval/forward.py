"""Unified forward pass with state capture for evaluation and extraction.

Consolidates forward_with_states and attention weight computation from
extract_shapes.py, extract_shapes_v2.py, and src/monitoring/geometric.py.

Handles both standard and AttnRes forward paths transparently.
Convention: states[0] = embedding, states[i+1] = after layer i.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.model.llama import LuxiaBaseModel, apply_rope

logger = logging.getLogger(__name__)


@dataclass
class ForwardResult:
    """Result of a forward pass with state capture.

    Attributes:
        states: Hidden states at each layer. states[0] = embedding output,
            states[i+1] = output after layer i.
        attention_weights: Optional per-layer attention weight matrices.
            Keys are layer indices, values are (n_heads, seq, seq) arrays.
    """

    states: list[torch.Tensor]
    attention_weights: dict[int, np.ndarray] = field(default_factory=dict)


def compute_attention_weights(
    layer: torch.nn.Module,
    x_normed: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    seq_len: int,
    batch_index: int = 0,
) -> np.ndarray:
    """Compute raw attention weight matrix for a single layer.

    Manually computes Q @ K^T / sqrt(d) and applies causal softmax.
    This bypasses SDPA/Flash Attention which don't return weights.

    Handles QK-norm and GQA (key head expansion) automatically.

    Args:
        layer: TransformerBlock instance.
        x_normed: Pre-normed input — ``layer.attn_norm(h)`` for AttnRes,
            ``layer.attn_norm(x)`` for standard.
        rope_cos: RoPE cosine frequencies.
        rope_sin: RoPE sine frequencies.
        seq_len: Sequence length.
        batch_index: Which batch element to return weights for.

    Returns:
        Attention weights array of shape (n_heads, seq, seq).
    """
    bsz = x_normed.shape[0]
    attn = layer.attn

    q = attn.q_proj(x_normed).view(
        bsz, seq_len, attn.num_heads, attn.head_dim
    ).transpose(1, 2)
    k = attn.k_proj(x_normed).view(
        bsz, seq_len, attn.num_kv_heads, attn.head_dim
    ).transpose(1, 2)

    if attn.qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)

    q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
    k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

    # Expand KV heads for GQA
    if attn.num_kv_groups > 1:
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)

    scale = attn.head_dim**-0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    # Causal mask
    causal = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores.masked_fill_(causal, float("-inf"))
    weights = F.softmax(scores, dim=-1)

    return weights[batch_index].cpu().numpy()  # (n_heads, seq, seq)


@torch.no_grad()
def forward_with_states(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    capture_attention: bool = False,
    attention_layers: list[int] | None = None,
) -> ForwardResult:
    """Run forward pass and return hidden states at every layer.

    Handles both standard and AttnRes forward paths transparently.
    For AttnRes models, captures the AttnRes-aggregated state ``h`` at each
    layer, and overrides the last layer with the final aggregation output.

    When ``capture_attention=True``, also computes raw attention weight
    matrices at the specified layers (or all sampled layers if None).

    Args:
        model: The model in eval mode.
        input_ids: Token IDs, shape (batch, seq_len).
        capture_attention: Whether to compute attention weight matrices.
        attention_layers: Which layers to capture attention at. If None and
            capture_attention is True, captures at all layers (expensive).

    Returns:
        ForwardResult with states and optional attention weights.
    """
    states: list[torch.Tensor] = []
    attn_result: dict[int, np.ndarray] = {}

    attn_layers_set: set[int] | None = None
    if capture_attention:
        if attention_layers is not None:
            attn_layers_set = set(attention_layers)
        else:
            attn_layers_set = set(range(model.config.num_layers))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = model.embed_tokens(input_ids)
        states.append(x.float())

        rope_cos = model.rope_cos
        rope_sin = model.rope_sin
        seq_len = input_ids.shape[1]

        if model.config.attn_res:
            states, attn_result = _forward_attn_res(
                model, x, rope_cos, rope_sin, seq_len, attn_layers_set
            )
            # Prepend embedding (the AttnRes path builds its own states list)
            states = [x.float()] + states
        else:
            for i, layer in enumerate(model.layers):
                # Capture attention weights before running the full layer
                if attn_layers_set is not None and i in attn_layers_set:
                    try:
                        x_normed = layer.attn_norm(x)
                        attn_result[i] = compute_attention_weights(
                            layer, x_normed, rope_cos, rope_sin, seq_len
                        )
                    except Exception as e:
                        logger.debug("Attention capture failed at layer %d: %s", i, e)

                x = layer(x, rope_cos, rope_sin)
                states.append(x.float())

    return ForwardResult(states=states, attention_weights=attn_result)


def _forward_attn_res(
    model: LuxiaBaseModel,
    embed: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    seq_len: int,
    attn_layers_set: set[int] | None,
) -> tuple[list[torch.Tensor], dict[int, np.ndarray]]:
    """AttnRes forward path with state capture.

    Uses _block_attn_res_from_list (the compat shim) for clarity.

    Returns:
        (states, attention_weights) where states does NOT include the
        embedding (caller prepends it).
    """
    states: list[torch.Tensor] = []
    attn_result: dict[int, np.ndarray] = {}

    committed: list[torch.Tensor] = []
    partial = embed
    boundary_set = model._attn_res_boundary_set

    for i, layer in enumerate(model.layers):
        # Pre-attention routing
        sources = committed + [partial]
        h = model._block_attn_res_from_list(
            sources, layer.attn_res_query, layer.attn_res_norm
        )

        # Block boundary
        if i in boundary_set:
            committed.append(partial.clone())

        # Capture attention weights from the AttnRes-aggregated state
        if attn_layers_set is not None and i in attn_layers_set:
            try:
                x_normed = layer.attn_norm(h)
                attn_result[i] = compute_attention_weights(
                    layer, x_normed, rope_cos, rope_sin, seq_len
                )
            except Exception as e:
                logger.debug("Attention capture failed at layer %d: %s", i, e)

        # Attention sub-layer
        attn_out = layer.attn(layer.attn_norm(h), rope_cos, rope_sin)
        partial = partial + attn_out

        # Pre-MLP routing
        sources = committed + [partial]
        h = model._block_attn_res_from_list(
            sources, layer.mlp_res_query, layer.mlp_res_norm
        )

        # MLP sub-layer
        mlp_out = layer.ffn(layer.ffn_norm(h))
        partial = partial + mlp_out

        # Capture state after this layer
        states.append(h.float())

    # Final aggregation — override last layer with actual output
    sources = committed + [partial]
    final_h = model._block_attn_res_from_list(
        sources, model.final_res_query, model.final_res_norm
    )
    states[-1] = final_h.float()

    return states, attn_result
