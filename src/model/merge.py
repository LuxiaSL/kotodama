"""Weight-space merges for kotodama fine-tunes that preserve Block AttnRes.

All constructions here keep the residual stream at ``hidden_size`` and the layer
count fixed.  That is the load-bearing invariant: every Block Attention Residual
tensor (``attn_res_query`` / ``mlp_res_query`` / ``final_res_query`` and their
norms) is a ``hidden_size`` vector and the block boundaries are a function of
``num_layers`` only, so as long as neither changes the AttnRes routing — masks,
block ranges, Triton kernels — stays valid.  Depth/passthrough merges break this;
the merges below do not.

Two merges are provided:

* :func:`build_routing_transplant` — keep one model's body, swap in another
  model's AttnRes routing params.  Localises "what the routing carries".

* :func:`build_width_state_dict` — the "A+M" width merge.  Widen the attention
  head count and the MLP intermediate by ``k`` (= number of source models) by
  *folded-stacking* the k sublayers, recombining into the unchanged D-dim
  residual.  At initialisation this computes the exact per-layer output-mean
  ``(1/k) * sum_m sublayer_m(x)``.

Folding trick (makes the width merge exact): ``RMSNorm(x) = (x / rms(x)) * w`` and
``rms(x)`` is shared across branches, so each source's pre-norm weight is folded
into its projection columns (``W_m * w_m``) and the merged pre-norm weight is set
to ones.  Every branch then sees the identical normed input while applying its own
learned scale.  Verified numerically: FFN block matches to fp32 eps; attention to
~4e-5 (the only residual is qk-norm averaging, which cannot be folded because it
acts per-head after projection — the cross-model spread there is ~6e-3).
"""
from __future__ import annotations

from typing import Mapping

import torch

StateDict = dict[str, torch.Tensor]

# Block AttnRes routing parameters (per-layer suffixes + model-level names).
_ROUTING_LAYER_SUFFIXES: tuple[str, ...] = (
    "attn_res_query",
    "attn_res_norm.weight",
    "mlp_res_query",
    "mlp_res_norm.weight",
)
_ROUTING_MODEL_KEYS: tuple[str, ...] = (
    "final_res_query",
    "final_res_norm.weight",
)


def _unwrap(ckpt: object) -> StateDict:
    """Return the tensor state dict from a raw checkpoint object."""
    if isinstance(ckpt, Mapping):
        inner = ckpt.get("model", ckpt)
        if isinstance(inner, Mapping):
            return {k: v for k, v in inner.items() if torch.is_tensor(v)}
    raise TypeError(f"Unsupported checkpoint object: {type(ckpt)!r}")


def load_state_dict(path: str) -> StateDict:
    """Load a (possibly ``{'model': ...}``-wrapped) checkpoint to a flat dict."""
    return _unwrap(torch.load(path, map_location="cpu", weights_only=False))


def widen_config(base_config: dict, k: int) -> dict:
    """Return a config dict widened by ``k`` lanes (heads/kv/intermediate × k).

    ``hidden_size``, ``num_layers``, ``head_dim`` and the AttnRes settings are
    unchanged — only the per-layer sublayer widths grow.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    cfg = dict(base_config)
    cfg["num_attention_heads"] = base_config["num_attention_heads"] * k
    cfg["num_kv_heads"] = base_config["num_kv_heads"] * k
    cfg["intermediate_size"] = base_config["intermediate_size"] * k
    return cfg


def _check_aligned(sds: list[StateDict]) -> None:
    if len(sds) < 1:
        raise ValueError("need at least one source state dict")
    ref = set(sds[0])
    for i, sd in enumerate(sds[1:], 1):
        if set(sd) != ref:
            missing = ref ^ set(sd)
            raise ValueError(f"source {i} key mismatch (symmetric diff size {len(missing)}); "
                             f"examples: {sorted(missing)[:3]}")


def build_routing_transplant(body: StateDict, routing: StateDict) -> StateDict:
    """``body``'s weights with AttnRes routing params replaced by ``routing``'s.

    Same shapes/config as the sources (no widening).
    """
    _check_aligned([body, routing])
    out: StateDict = {k: v.clone() for k, v in body.items()}
    for key in out:
        if key in _ROUTING_MODEL_KEYS or key.endswith(_ROUTING_LAYER_SUFFIXES):
            out[key] = routing[key].clone()
    return out


def build_width_state_dict(
    sds: list[StateDict],
    *,
    num_layers: int,
    hidden_size: int,
    dtype: torch.dtype = torch.float32,
) -> StateDict:
    """Build the A+M width-merge state dict (config from :func:`widen_config`).

    Args:
        sds: ``k`` source state dicts (aligned keys, base config).
        num_layers / hidden_size: base architecture dims.
        dtype: output tensor dtype.

    Returns:
        State dict for a ``LuxiaBaseModel`` built with ``widen_config(base, k)``.
        At init it computes ``(1/k) * sum_m sublayer_m(x)`` per layer.
    """
    _check_aligned(sds)
    k = len(sds)

    def mean(key: str) -> torch.Tensor:
        return torch.stack([sd[key].float() for sd in sds]).mean(0).to(dtype)

    def fold_stack(key: str, norm_key: str) -> torch.Tensor:
        # Fold each source's pre-norm into its projection columns, stack on out-dim.
        return torch.cat(
            [(sd[key].float() * sd[norm_key].float()).to(dtype) for sd in sds], dim=0
        )

    def cat_out(key: str) -> torch.Tensor:
        # Concatenate output projections on the in-dim and average (1/k).
        return (torch.cat([sd[key].float() for sd in sds], dim=1) / k).to(dtype)

    out: StateDict = {
        "embed_tokens.weight": mean("embed_tokens.weight"),
        "norm.weight": mean("norm.weight"),
        "final_res_query": mean("final_res_query"),
        "final_res_norm.weight": mean("final_res_norm.weight"),
    }
    ones = torch.ones(hidden_size, dtype=dtype)
    for layer in range(num_layers):
        p = f"layers.{layer}."
        # Pre-norms folded into the projections -> identity scale on the merged norm.
        out[p + "attn_norm.weight"] = ones.clone()
        out[p + "ffn_norm.weight"] = ones.clone()
        # Attention: q/k/v fold attn_norm + stack on out-dim; o concatenated in-dim /k.
        out[p + "attn.q_proj.weight"] = fold_stack(p + "attn.q_proj.weight", p + "attn_norm.weight")
        out[p + "attn.k_proj.weight"] = fold_stack(p + "attn.k_proj.weight", p + "attn_norm.weight")
        out[p + "attn.v_proj.weight"] = fold_stack(p + "attn.v_proj.weight", p + "attn_norm.weight")
        out[p + "attn.o_proj.weight"] = cat_out(p + "attn.o_proj.weight")
        # qk-norm is per-head, post-projection -> cannot fold; average (sub-noise).
        out[p + "attn.q_norm.weight"] = mean(p + "attn.q_norm.weight")
        out[p + "attn.k_norm.weight"] = mean(p + "attn.k_norm.weight")
        # FFN: gate/up fold ffn_norm + stack on out-dim; down concatenated in-dim /k.
        out[p + "ffn.gate_proj.weight"] = fold_stack(p + "ffn.gate_proj.weight", p + "ffn_norm.weight")
        out[p + "ffn.up_proj.weight"] = fold_stack(p + "ffn.up_proj.weight", p + "ffn_norm.weight")
        out[p + "ffn.down_proj.weight"] = cat_out(p + "ffn.down_proj.weight")
        # AttnRes routing: residual unchanged -> averaged, structure untouched.
        for suf in _ROUTING_LAYER_SUFFIXES:
            out[p + suf] = mean(p + suf)
    return out
