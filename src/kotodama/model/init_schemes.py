"""Alternative weight initializations, applied after the model's standard init (`--init_scheme`).

``standard`` is the model's own ``_init_weights`` (N(0, 0.02); residual projections × 1/sqrt(2L)) and is a no-op
here, so every existing run is unchanged. The others re-initialize transformer-block matrices in place, drawing from
the global torch RNG (seeded by ``--seed`` before model construction), so a scheme is deterministic per seed.

  mimetic         per attention head, W_Q W_Kᵀ ≈ a·Z/√d + b·I and W_O W_V ≈ c·Z'/√d − e·I (Trockman & Kolter 2023,
                  "Mimetic Initialization of Self-Attention Layers"), built from the SVD of those d×d targets and
                  truncated to head_dim; GQA: the K/V factors are per KV head, each query head in a group gets its
                  own small perturbation of the shared factor (breaks symmetry). Rescaled to the standard init's
                  Frobenius norm per matrix (QK-norm makes the QK scale irrelevant; the structure is what counts).
  orthogonal      every block Linear = a random orthogonal matrix, rescaled to the standard init's Frobenius norm
                  (residual projections keep their 1/sqrt(2L) factor).
  depth_graded    standard weights; the QK-norm gains are raised so attention logits are DEPTH_PEAK× sharper in
                  the mid-stack (ramp up L4→L12, plateau L12–L17, ramp down L17→L25). Requires qk_norm (with QK-norm,
                  scaling W_Q/W_K would not change the logits at all).
  small_attn_out  attention o_proj × 0.1 (attention starts near-silent; which heads organize is left to the data).

Embeddings, the LM head, norms (except depth_graded's QK gains) and AttnRes routing parameters are untouched.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

SCHEMES: tuple[str, ...] = ("standard", "mimetic", "orthogonal", "depth_graded", "small_attn_out")
STD = 0.02
MIMETIC_QK = (0.7, 0.7)      # (a, b): W_Q W_Kᵀ ≈ a·Z/√d + b·I   (paper defaults)
MIMETIC_VO = (0.4, 0.4)      # (c, e): W_O W_V ≈ c·Z/√d − e·I
GQA_JITTER = 0.1             # relative perturbation giving each query head in a KV group its own W_Q
DEPTH_PEAK = 1.5             # logit sharpening at the mid-stack plateau
SMALL_O = 0.1


def _target_norm(w: torch.Tensor, scale: float = 1.0) -> float:
    """Frobenius norm the standard init has in expectation for this matrix."""
    return STD * scale * math.sqrt(w.numel())


def _rescale_(w: torch.Tensor, norm: float) -> None:
    w.mul_(norm / w.norm().clamp_min(1e-12))


def _factor(m: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-k factors (A, B) with A @ B ≈ m: A = U_k sqrt(S_k), B = sqrt(S_k) V_kᵀ."""
    u, s, vh = torch.linalg.svd(m.double(), full_matrices=False)
    r = s[:k].sqrt()
    return (u[:, :k] * r).float(), (r[:, None] * vh[:k]).float()


def _layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("model has no .layers")
    return list(layers)


@torch.no_grad()
def _mimetic(model: nn.Module) -> None:
    rs = 1.0 / math.sqrt(2 * len(_layers(model)))
    for layer in _layers(model):
        at = layer.attn
        d, hd, nh, nkv = at.q_proj.in_features, at.head_dim, at.num_heads, at.num_kv_heads
        if hd > d:
            raise ValueError(f"mimetic init needs head_dim <= hidden ({hd} > {d})")
        group = nh // nkv
        dev = at.q_proj.weight.device
        wq = torch.empty(nh * hd, d); wk = torch.empty(nkv * hd, d)
        wv = torch.empty(nkv * hd, d); wo = torch.empty(d, nh * hd)
        eye = torch.eye(d)
        for g in range(nkv):
            a, b = MIMETIC_QK
            m_qk = a * torch.randn(d, d) / math.sqrt(d) + b * eye           # W_Qᵀ W_K target (x·qk·x)
            uq, vk = _factor(m_qk, hd)                                      # m ≈ uq @ vk ; uq: d×hd, vk: hd×d
            wk[g * hd:(g + 1) * hd] = vk
            c, e = MIMETIC_VO
            m_vo = c * torch.randn(d, d) / math.sqrt(d) - e * eye           # W_O W_V target
            uo, vv = _factor(m_vo, hd)                                      # uo: d×hd (W_O block), vv: hd×d (W_V)
            wv[g * hd:(g + 1) * hd] = vv
            for j in range(group):
                h = g * group + j
                q = uq.T.clone()                                            # hd×d, so W_Qᵀ W_K = uq @ vk
                q += GQA_JITTER * q.norm() / math.sqrt(q.numel()) * torch.randn_like(q)
                wq[h * hd:(h + 1) * hd] = q
                wo[:, h * hd:(h + 1) * hd] = uo / group                     # summed over the group ≈ the target
        for lin, w, sc in ((at.q_proj, wq, 1.0), (at.k_proj, wk, 1.0), (at.v_proj, wv, 1.0), (at.o_proj, wo, rs)):
            _rescale_(w, _target_norm(lin.weight, sc))
            lin.weight.copy_(w.to(dev, lin.weight.dtype))


@torch.no_grad()
def _orthogonal(model: nn.Module) -> None:
    rs = 1.0 / math.sqrt(2 * len(_layers(model)))
    for layer in _layers(model):
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                residual = name.endswith("o_proj") or name.endswith("down_proj")
                w = torch.empty_like(mod.weight, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(w)
                _rescale_(w, _target_norm(mod.weight, rs if residual else 1.0))
                mod.weight.copy_(w.to(mod.weight.device, mod.weight.dtype))


def depth_gain(layer: int, n_layers: int) -> float:
    """Logit multiplier at ``layer`` (1.0 outside the ramp; DEPTH_PEAK on the plateau), defined on a 28-layer
    stack and mapped proportionally for other depths."""
    x = layer * 27.0 / max(n_layers - 1, 1)
    if x <= 4 or x >= 25:
        return 1.0
    if x < 12:
        t = (x - 4) / 8
    elif x <= 17:
        t = 1.0
    else:
        t = (25 - x) / 8
    return 1.0 + (DEPTH_PEAK - 1.0) * t


@torch.no_grad()
def _depth_graded(model: nn.Module) -> None:
    layers = _layers(model)
    for i, layer in enumerate(layers):
        at = layer.attn
        if not getattr(at, "qk_norm", False):
            raise ValueError("depth_graded needs qk_norm (logits are only temperature-controlled through the norm gains)")
        g = math.sqrt(depth_gain(i, len(layers)))     # q·k scales with g_q·g_k
        at.q_norm.weight.mul_(g)
        at.k_norm.weight.mul_(g)


@torch.no_grad()
def _small_attn_out(model: nn.Module) -> None:
    for layer in _layers(model):
        layer.attn.o_proj.weight.mul_(SMALL_O)


def apply_init_scheme(model: nn.Module, scheme: str) -> None:
    """Re-initialize ``model`` in place under ``scheme`` (``standard`` = no-op)."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown init scheme {scheme!r}; choose from {SCHEMES}")
    {"standard": lambda m: None, "mimetic": _mimetic, "orthogonal": _orthogonal,
     "depth_graded": _depth_graded, "small_attn_out": _small_attn_out}[scheme](model)
