"""
Muon optimizer for luxia-base.

Muon (Momentum Orthogonalized by Newton-Schulz) performs steepest descent
under the spectral norm by orthogonalizing the momentum matrix. This treats
all singular directions of the weight space proportionally, resisting the
low-rank collapse that Adam-family optimizers promote.

For the luxia-base project, this is a deliberate choice: we hypothesize that
Muon's balanced spectral updates produce richer representational geometry
(more diverse dynamical motifs, higher shattering dimensionality).

Reference: MoonshotAI/Muon (github.com/MoonshotAI/Muon)
           Moonlight paper: Muon is Scalable for LLM Training
           Gram-NS coefficients: Dao-AILab/Gram-Newton-Schulz

Usage:
    optimizer = build_hybrid_optimizer(model, muon_config, adamw_config)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.optim import AdamW, Optimizer

logger = logging.getLogger(__name__)


# ── NS coefficient presets ──────────────────────────────────────────────────
#
# Per-iteration (a, b, c) coefficients for the Newton-Schulz polar iteration:
#   X_{k+1} = (a_k I + b_k X_k X_k^T + c_k (X_k X_k^T)^2) X_k
#
# "original": Fixed coefficients from MoonshotAI/Muon, same polynomial every
#   iteration. Standard in all prior proxy ablations.
# "gram_ns": Per-iteration optimized from Gram-Newton-Schulz (Dao-AILab).
#   Earlier iterations more aggressive, later ones polish. Better convergence
#   in the same 5 iterations — free quality improvement.
# "polar_express": From the same repo, with safety factor adjustment (÷1.05
#   on a, ÷1.05^3 on b, ÷1.05^5 on c). More conservative than "gram_ns".

NS_COEFFICIENT_PRESETS: dict[str, list[tuple[float, float, float]]] = {
    "original": [
        (3.4445, -4.7750, 2.0315),
    ] * 5,
    "gram_ns": [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        (
            8.28721201814563 / 1.05,
            -23.595886519098837 / 1.05**3,
            17.300387312530933 / 1.05**5,
        ),
        (
            4.107059111542203 / 1.05,
            -2.9478499167379106 / 1.05**3,
            0.5448431082926601 / 1.05**5,
        ),
        (
            3.9486908534822946 / 1.05,
            -2.908902115962949 / 1.05**3,
            0.5518191394370137 / 1.05**5,
        ),
        (
            3.3184196573706015 / 1.05,
            -2.488488024314874 / 1.05**3,
            0.51004894012372 / 1.05**5,
        ),
        (
            2.300652019954817 / 1.05,
            -1.6689039845747493 / 1.05**3,
            0.4188073119525673 / 1.05**5,
        ),
    ],
}


def _resolve_ns_coefficients(
    preset: Optional[str],
    num_iterations: int,
) -> list[tuple[float, float, float]]:
    """Resolve NS coefficient preset name to per-iteration coefficient list."""
    if preset is None:
        preset = "original"
    if preset not in NS_COEFFICIENT_PRESETS:
        raise ValueError(
            f"Unknown NS coefficient preset '{preset}'. "
            f"Available: {list(NS_COEFFICIENT_PRESETS.keys())}"
        )
    coeffs = NS_COEFFICIENT_PRESETS[preset]
    if len(coeffs) < num_iterations:
        coeffs = coeffs + [coeffs[-1]] * (num_iterations - len(coeffs))
    return coeffs[:num_iterations]


def newton_schulz_orthogonalize(
    M: torch.Tensor,
    num_iterations: int = 5,
    ns_coefficients: Optional[str] = None,
) -> torch.Tensor:
    """
    Approximate orthogonalization via Newton-Schulz iteration.

    Computes the nearest orthogonal matrix to M (in Frobenius norm).
    Converges cubically — 5 iterations is typically sufficient.

    This is the core of Muon: by orthogonalizing the momentum,
    the update treats all singular directions equally.

    Args:
        M: 2D matrix to orthogonalize
        num_iterations: Number of NS iterations
        ns_coefficients: Coefficient preset name ("original", "gram_ns",
            "polar_express"). None defaults to "original".

    Optimizations from reference implementation (KellerJordan/Muon):
    - BF16 computation for ~2x speedup on modern GPUs
    - Tall-matrix transpose for faster convergence (NS converges
      faster when ncols >= nrows)
    """
    assert M.ndim == 2, f"Newton-Schulz requires 2D matrix, got {M.ndim}D"

    coeffs = _resolve_ns_coefficients(ns_coefficients, num_iterations)
    orig_dtype = M.dtype
    X = M.half()

    # Transpose tall matrices: NS converges faster when ncols >= nrows
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    # Normalize to unit Frobenius norm for numerical stability
    X = X / (X.norm() + 1e-7)

    for a, b, c in coeffs:
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(orig_dtype)


def _batched_newton_schulz(
    M: torch.Tensor,
    num_iterations: int = 5,
    ns_coefficients: Optional[str] = None,
) -> torch.Tensor:
    """
    Batched Newton-Schulz orthogonalization.

    M: (batch, rows, cols) — all matrices must have the same shape.
    Returns: (batch, rows, cols) orthogonalized matrices.
    """
    coeffs = _resolve_ns_coefficients(ns_coefficients, num_iterations)
    orig_dtype = M.dtype
    X = M.half()

    # Transpose if tall
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
        transposed = True

    # Normalize each matrix
    norms = X.flatten(start_dim=-2).norm(dim=-1, keepdim=True).unsqueeze(-1)
    X = X / (norms + 1e-7)

    for a, b, c in coeffs:
        A = X @ X.transpose(-2, -1)  # batched matmul
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.transpose(-2, -1)

    return X.to(orig_dtype)


class Muon(Optimizer):
    """
    Muon: Momentum Orthogonalized by Newton-Schulz.

    Applies to 2D weight matrices only. Uses Newton-Schulz iteration
    to orthogonalize the momentum matrix before stepping.

    Args:
        params: Parameters to optimize (must all be 2D)
        lr: Learning rate (typically 0.01-0.04 for LLM pretraining)
        momentum: Momentum coefficient (typically 0.95)
        nesterov: Whether to use Nesterov momentum
        weight_decay: Decoupled weight decay coefficient
        ns_iterations: Number of Newton-Schulz iterations (5 is standard)
        ns_coefficients: Coefficient preset for NS iteration. Options:
            "original" (MoonshotAI fixed), "gram_ns" (Dao-AILab per-iteration),
            "polar_express" (Dao-AILab conservative). None defaults to "original".
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.03,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        ns_iterations: int = 5,
        ns_coefficients: Optional[str] = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_iterations=ns_iterations,
        )
        super().__init__(params, defaults)
        self._ns_coefficients = ns_coefficients
        _resolve_ns_coefficients(ns_coefficients, ns_iterations)  # validate early

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["ns_coefficients"] = self._ns_coefficients
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        ns_coeff = state_dict.pop("ns_coefficients", None)
        super().load_state_dict(state_dict)
        if ns_coeff is not None:
            if self._ns_coefficients is not None and self._ns_coefficients != ns_coeff:
                logger.warning(
                    "Muon ns_coefficients mismatch: checkpoint has '%s', "
                    "CLI has '%s'. Using checkpoint value for consistency.",
                    ns_coeff, self._ns_coefficients,
                )
            self._ns_coefficients = ns_coeff

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            ns_iterations = group["ns_iterations"]

            # Collect params with gradients and update momentum buffers
            params_with_grads: list[torch.Tensor] = []
            updates_to_orthogonalize: list[torch.Tensor] = []
            scales: list[float] = []
            tp_infos: list[Optional[dict[str, Any]]] = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    update = momentum * buf + grad
                else:
                    update = buf.clone()

                # TP: all-gather to full matrix for NS, compute scale from full shape
                tp_info: Optional[dict[str, Any]] = None
                if hasattr(p, "_tp_shard_dim"):
                    from src.training.tensor_parallel import all_gather_along_dim
                    tp_info = {
                        "shard_dim": p._tp_shard_dim,
                        "group": p._tp_group,
                        "tp_rank": p._tp_rank,
                        "local_size": update.shape[p._tp_shard_dim],
                    }
                    update = all_gather_along_dim(update, p._tp_shard_dim, p._tp_group)

                updates_to_orthogonalize.append(update)
                params_with_grads.append(p)
                tp_infos.append(tp_info)
                scales.append(max(1.0, (update.shape[0] / update.shape[1]) ** 0.5))

            if not params_with_grads:
                continue

            # Batch NS by shape: group matrices with same shape
            # and orthogonalize them as a single batched operation.
            # Under TP, updates are already gathered to full shape,
            # so batching groups match the non-TP case.
            shape_groups: dict[tuple[int, int], list[int]] = {}
            for i, M in enumerate(updates_to_orthogonalize):
                shape = (M.shape[0], M.shape[1])
                if shape not in shape_groups:
                    shape_groups[shape] = []
                shape_groups[shape].append(i)

            orthogonalized: list[Optional[torch.Tensor]] = [None] * len(updates_to_orthogonalize)

            for shape, indices in shape_groups.items():
                if len(indices) == 1:
                    idx = indices[0]
                    orthogonalized[idx] = newton_schulz_orthogonalize(
                        updates_to_orthogonalize[idx],
                        num_iterations=ns_iterations,
                        ns_coefficients=self._ns_coefficients,
                    )
                else:
                    batch = torch.stack(
                        [updates_to_orthogonalize[i] for i in indices]
                    )
                    result = _batched_newton_schulz(
                        batch, ns_iterations,
                        ns_coefficients=self._ns_coefficients,
                    )
                    for j, idx in enumerate(indices):
                        orthogonalized[idx] = result[j]

            # Apply updates — slice TP params back to local shard
            for i, p in enumerate(params_with_grads):
                orth = orthogonalized[i]

                if tp_infos[i] is not None:
                    info = tp_infos[i]
                    start = info["tp_rank"] * info["local_size"]
                    orth = orth.narrow(
                        info["shard_dim"], start, info["local_size"]
                    ).contiguous()

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(orth, alpha=-lr * scales[i])

        return loss


def build_hybrid_optimizer(
    model: torch.nn.Module,
    muon_lr: float = 0.03,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    muon_ns_iterations: int = 5,
    muon_ns_coefficients: Optional[str] = None,
    adamw_lr: float = 6e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_weight_decay: float = 0.1,
) -> tuple[Muon, AdamW]:
    """
    Build hybrid Muon + AdamW optimizer for a transformer model.

    Muon handles all 2D weight matrices (Q/K/V/O projections, FFN weights).
    AdamW handles everything else (embeddings, norms, LM head if untied).

    Returns separate optimizers that should both be stepped each training step.
    """
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_count = sum(p.numel() for p in muon_params)
    adamw_count = sum(p.numel() for p in adamw_params)
    total = muon_count + adamw_count

    # Only print on rank 0 (or when not distributed)
    try:
        import torch.distributed as dist
        should_print = not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        should_print = True

    if should_print:
        print(f"Muon params:  {muon_count:>12,} ({muon_count/total*100:.1f}%)")
        print(f"AdamW params: {adamw_count:>12,} ({adamw_count/total*100:.1f}%)")

    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=True,
        weight_decay=muon_weight_decay,
        ns_iterations=muon_ns_iterations,
        ns_coefficients=muon_ns_coefficients,
    )

    adamw_opt = AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=adamw_betas,
        weight_decay=adamw_weight_decay,
    )

    return muon_opt, adamw_opt


class HybridScheduler:
    """
    WSD (Warmup-Stable-Decay) schedule applied to both Muon and AdamW.

    Both optimizers follow the same schedule shape but with different base LRs.
    """

    def __init__(
        self,
        muon_opt: Muon,
        adamw_opt: AdamW,
        warmup_steps: int = 2000,
        total_steps: int = 100000,
        decay_start_pct: float = 0.90,
        decay_type: str = "sqrt",
    ) -> None:
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_step = int(total_steps * decay_start_pct)
        self.decay_type = decay_type

        # Store base LRs
        self.muon_base_lr = muon_opt.defaults["lr"]
        self.adamw_base_lr = adamw_opt.defaults["lr"]

    def get_lr_multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        elif step < self.decay_start_step:
            # Stable phase
            return 1.0
        else:
            # Decay to zero
            decay_steps = self.total_steps - self.decay_start_step
            progress = (step - self.decay_start_step) / max(decay_steps, 1)

            if self.decay_type == "sqrt":
                return 1.0 - progress ** 0.5
            elif self.decay_type == "linear":
                return 1.0 - progress
            elif self.decay_type == "cosine":
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            else:
                raise ValueError(f"Unknown decay type: {self.decay_type}")

    def step(self, step: int) -> None:
        mult = self.get_lr_multiplier(step)
        for group in self.muon_opt.param_groups:
            group["lr"] = self.muon_base_lr * mult
        for group in self.adamw_opt.param_groups:
            group["lr"] = self.adamw_base_lr * mult

    def get_last_lr(self) -> dict[str, float]:
        muon_lr = (
            self.muon_opt.param_groups[0]["lr"]
            if self.muon_opt.param_groups
            else 0.0
        )
        adamw_lr = (
            self.adamw_opt.param_groups[0]["lr"]
            if self.adamw_opt.param_groups
            else 0.0
        )
        return {"muon_lr": muon_lr, "adamw_lr": adamw_lr}
