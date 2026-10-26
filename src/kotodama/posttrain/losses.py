"""Preference losses on sequence log-probs (sums over completion tokens).

Pairwise:
  dpo   −log σ(β[(π_c − ref_c) − (π_r − ref_r)])
  dpop  DPO-Positive: inside the sigmoid, −λ·max(0, ref_c − π_c) — guards chosen-logprob collapse
  ipo   (h − 1/(2β))² — margins saturate instead of growing
The reported margin β·h is the vanilla-DPO currency for every variant (cross-run comparability).

Unpaired — KTO (Ethayarajh et al. 2024, "KTO: Model Alignment as Prospect Theoretic Optimization"):
  r   = π − ref                                   (implied reward / β)
  z0  = max(0, KL estimate), detached             (the reference point)
  desirable    λ_D · (1 − σ(β(r − z0)))
  undesirable  λ_U · (1 − σ(β(z0 − r)))
The paper's KL estimate is the batch mean of π − ref on MISMATCHED (prompt_i, completion_j) pairs; ``kto_loss``
takes it as an argument so the caller chooses the estimator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAIRWISE = ("dpo", "dpop", "ipo")


def pairwise_loss(lp_c: torch.Tensor, lp_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor,
                  beta: float, weight: torch.Tensor | None = None, loss_type: str = "dpo",
                  dpop_lambda: float = 50.0) -> tuple[torch.Tensor, torch.Tensor]:
    """(weighted mean loss, vanilla margin β·h per pair)."""
    if loss_type not in PAIRWISE:
        raise ValueError(f"loss_type {loss_type!r}; expected one of {PAIRWISE}")
    w = torch.ones_like(lp_c) if weight is None else weight
    h = (lp_c - ref_c) - (lp_r - ref_r)
    margin = beta * h
    if loss_type == "dpop":
        per = -F.logsigmoid(beta * (h - dpop_lambda * torch.clamp(ref_c - lp_c, min=0.0)))
    elif loss_type == "ipo":
        per = (h - 1.0 / (2.0 * beta)) ** 2
    else:
        per = -F.logsigmoid(margin)
    return (w * per).sum() / w.sum(), margin


def kto_loss(lp: torch.Tensor, ref: torch.Tensor, desirable: torch.Tensor, kl: torch.Tensor | float,
             beta: float, lambda_d: float = 1.0, lambda_u: float = 1.0,
             weight: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(weighted mean KTO loss, reward β·(π − ref) per item). ``desirable`` is a bool tensor; ``kl`` the KL
    estimate (clamped at 0 and detached here)."""
    w = torch.ones_like(lp) if weight is None else weight
    z0 = torch.as_tensor(kl, dtype=lp.dtype, device=lp.device).detach().clamp(min=0.0)
    r = lp - ref
    good = lambda_d * (1.0 - torch.sigmoid(beta * (r - z0)))
    bad = lambda_u * (1.0 - torch.sigmoid(beta * (z0 - r)))
    per = torch.where(desirable, good, bad)
    return (w * per).sum() / w.sum(), beta * r


def kto_kl_estimate(lp_mismatch: torch.Tensor, ref_mismatch: torch.Tensor) -> torch.Tensor:
    """The paper's reference point: mean(π − ref) over mismatched pairs, clamped at 0, detached."""
    return (lp_mismatch - ref_mismatch).mean().detach().clamp(min=0.0)


def kto_balance(n_desirable: int, n_undesirable: int, lambda_d: float, lambda_u: float) -> float:
    """λ_D·n_D / (λ_U·n_U); the paper recommends keeping it in [1, 4/3]."""
    return (lambda_d * n_desirable) / max(lambda_u * n_undesirable, 1e-9)
