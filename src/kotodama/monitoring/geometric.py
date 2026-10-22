"""
Geometric health monitoring for kotodama pretraining.

Metrics computed per monitoring step:
  - RankMe (effective rank) on probe batch hidden states
  - Stable rank per layer (weight-space)
  - Anisotropy (average pairwise cosine similarity)
  - Dead unit fraction per layer
  - Attention entropy distribution
  - TwoNN intrinsic dimensionality at sampled layers
  - EoC Jacobian spectral radius per layer

EoS sharpness is computed separately (requires live gradients).

Usage in training loop::

    monitor = GeometricMonitor(model, config)
    # In training loop:
    if monitor.should_monitor(step):
        sharpness = monitor.compute_sharpness(step)  # before optimizer.step()
        ...
        metrics = monitor.compute_all(step)           # after optimizer.step()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MonitorSchedule:
    """Step-function schedule for decaying monitoring cadence.

    Parsed from "start:interval,..." e.g. "0:25,2000:50,10000:200,50000:500".
    At each step, the last phase whose start_step <= step determines the interval.
    """

    phases: list[tuple[int, int]] = field(default_factory=lambda: [(0, 500)])

    @classmethod
    def from_string(cls, s: str) -> MonitorSchedule:
        phases: list[tuple[int, int]] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            start_s, interval_s = part.split(":")
            phases.append((int(start_s), int(interval_s)))
        phases.sort(key=lambda x: x[0])
        if not phases:
            raise ValueError(f"Empty schedule: {s!r}")
        return cls(phases=phases)

    @classmethod
    def fixed(cls, every: int) -> MonitorSchedule:
        return cls(phases=[(0, every)])

    def should_fire(self, step: int) -> bool:
        interval = self.interval_at(step)
        return interval > 0 and step % interval == 0

    def interval_at(self, step: int) -> int:
        result = self.phases[0][1]
        for start, interval in self.phases:
            if step >= start:
                result = interval
            else:
                break
        return result


@dataclass
class MonitorConfig:
    """Configuration for geometric monitoring."""

    schedule: Optional[MonitorSchedule] = None

    # Probe / layer sampling
    tier1_probe_size: int = 1024  # number of samples in probe batch
    tier1_sample_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select

    tier2_twonn_samples: int = 3000  # samples for TwoNN ID estimation
    tier2_twonn_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select (5 evenly spaced)

    # Legacy cadence fields (used when schedule is None)
    tier1_every: int = 500
    tier2_every: int = 5000

    # General
    device: str = "cuda"


class GeometricMonitor:
    """
    Geometric health monitor for transformer pretraining.

    Computes metrics at configurable intervals during training.
    All metrics are returned as flat dictionaries suitable for
    wandb/logging.

    The monitor holds a fixed probe batch for longitudinal
    comparability — the same inputs are used at every measurement
    point so changes reflect model development, not data variance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MonitorConfig] = None,
    ) -> None:
        self.config = config or MonitorConfig()

        # Unwrap DDP and torch.compile wrappers
        m = model.module if hasattr(model, "module") else model
        # torch.compile wraps in OptimizedModule — unwrap to get the real model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        self.model = m

        # Determine layer sampling positions
        num_layers = len(self.model.layers)
        if not self.config.tier1_sample_layers:
            # Sample 5 evenly spaced layers + first and last
            step = max(1, num_layers // 4)
            self.config.tier1_sample_layers = sorted(
                set([0, step, 2 * step, 3 * step, num_layers - 1])
            )
        if not self.config.tier2_twonn_layers:
            # 5 evenly spaced positions (input, 25%, 50%, 75%, output)
            self.config.tier2_twonn_layers = [
                0,
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
                num_layers - 1,
            ]

        # Storage for probe batch (set via set_probe_batch)
        self._probe_batch: Optional[torch.Tensor] = None

        # Hook storage for capturing intermediate activations
        self._hooks: list[Any] = []
        self._captured_hidden: dict[int, torch.Tensor] = {}
        self._captured_attn: dict[int, torch.Tensor] = {}

        # AttnRes diagnostics (populated during _probe_forward_attn_res)
        self._attn_res_diagnostics: dict[str, float] = {}

        # Warm-start vectors for Jacobian spectral radius (EoC)
        self._warm_vectors: dict[str, torch.Tensor] = {}
        self._eoc_call_count: int = 0

        logger.info(
            "GeometricMonitor: %d layers, tier1 layers=%s, tier2 ID layers=%s",
            num_layers,
            self.config.tier1_sample_layers,
            self.config.tier2_twonn_layers,
        )

    def set_probe_batch(self, input_ids: torch.Tensor) -> None:
        """
        Set the fixed probe batch for longitudinal monitoring.

        This should be called once with a diverse, fixed set of token
        sequences that will be used for all forward-pass-based metrics.
        """
        self._probe_batch = input_ids.clone()
        logger.info("Probe batch set: shape %s", tuple(input_ids.shape))

    def should_monitor(self, step: int) -> bool:
        """Check if monitoring should fire at this step."""
        if self.config.schedule is not None:
            return self.config.schedule.should_fire(step)
        return self.config.tier1_every > 0 and step % self.config.tier1_every == 0

    # =========================================================================
    # Edge of Stability: finite-difference sharpness along gradient direction
    # =========================================================================

    def compute_sharpness(
        self, step: int, epsilon: float = 0.1, n_sequences: int = 8,
    ) -> dict[str, float]:
        """
        Compute directional sharpness along the current gradient direction.

        Must be called while gradients are still on model parameters (before
        optimizer.zero_grad). Uses finite differences on a probe batch subset:
            sharpness = (L(θ+εĝ) - 2L(θ) + L(θ-εĝ)) / ε²
        where ĝ = g/‖g‖ is the unit gradient direction.

        At edge of stability: sharpness ≈ 2/lr. Values exceeding this indicate
        the optimizer is actively reducing sharpness.

        Note: ε=0.1 (not 0.01) because at 3B+ scale, unit-normalized gradient
        components are O(1/√D) ≈ 6e-6, and smaller ε causes per-parameter
        perturbations to fall below bf16 precision.
        """
        if self._probe_batch is None:
            return {}

        params_with_grad = [
            p for p in self.model.parameters() if p.grad is not None
        ]
        if not params_with_grad:
            logger.warning("No gradients available for sharpness computation")
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        torch.cuda.empty_cache()
        batch = self._probe_batch[:n_sequences].to(device)

        grad_norm = torch.sqrt(
            sum(p.grad.float().pow(2).sum() for p in params_with_grad)
        )
        if grad_norm < 1e-10:
            return {}

        embed_params = {id(self.model.embed_tokens.weight)}
        embed_grad_sq = sum(
            p.grad.float().pow(2).sum()
            for p in params_with_grad if id(p) in embed_params
        )
        embed_grad_frac = (embed_grad_sq / grad_norm.pow(2)).item()

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}
        # Exact snapshot (CPU, so 7B rank 0 does not need 2x weight memory). In-place
        # +e / -2e / +e adds on bf16 weights do not restore exactly and would leave a
        # residual on rank 0 only -> DDP ranks silently diverge.
        snapshot = [p.detach().to("cpu", copy=True) for p in params_with_grad]
        inv_norm = 1.0 / grad_norm

        def _set_perturbed(sign: float) -> float:
            """Weights := original + sign*eps*g_hat (computed in fp32, one rounding).
            Returns the norm of the displacement that actually landed."""
            landed_sq = torch.zeros((), device=device, dtype=torch.float32)
            for p, orig in zip(params_with_grad, snapshot):
                o = orig.to(device, non_blocking=True).float()
                new = (o + p.grad.float() * (sign * epsilon * inv_norm)).to(p.dtype)
                landed_sq += (new.float() - o).pow(2).sum()
                p.data.copy_(new)
            return landed_sq.sqrt().item()

        def _restore() -> None:
            for p, orig in zip(params_with_grad, snapshot):
                p.data.copy_(orig.to(device, non_blocking=True))

        try:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L0 = self.model(batch, labels=batch)["loss"].float().item()
                landed_plus = _set_perturbed(+1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L_plus = self.model(batch, labels=batch)["loss"].float().item()
                _set_perturbed(-1.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    L_minus = self.model(batch, labels=batch)["loss"].float().item()

            sharpness = (L_plus - 2.0 * L0 + L_minus) / (epsilon ** 2)
            metrics["eos/sharpness"] = sharpness
            metrics["eos/L0_probe"] = L0
            metrics["eos/L_plus"] = L_plus
            metrics["eos/L_minus"] = L_minus
            metrics["eos/grad_norm"] = grad_norm.item()
            metrics["eos/embed_grad_frac"] = embed_grad_frac
            # < 1 means bf16 rounding swallowed part of the probe step: sharpness
            # is then measured along a truncated direction (read it with that in mind).
            metrics["eos/eps_landed_frac"] = landed_plus / epsilon
            metrics["eos/time_s"] = time.time() - t0

            logger.info(
                "EoS [step %d]: sharpness=%.4f, L0=%.4f, embed_grad=%.1f%%, landed=%.2f, time=%.2fs",
                step, sharpness, L0, embed_grad_frac * 100, landed_plus / epsilon,
                time.time() - t0,
            )

        except Exception as e:
            logger.warning("Sharpness computation failed: %s", e)
        finally:
            try:
                with torch.no_grad():
                    _restore()
            except Exception:
                logger.error("CRITICAL: failed to restore weights after sharpness probe")
                raise

        if was_training:
            self.model.train()

        return metrics

    # =========================================================================
    # Edge of Chaos: per-layer Jacobian spectral radius
    # =========================================================================

    def compute_jacobian_spectral(
        self, step: int, n_iters: int = 5, n_sequences: int = 4,
    ) -> dict[str, float]:
        """
        Compute per-layer Jacobian spectral norm via warm-started power iteration.

        For each sampled layer, estimates σ_max(J_ℓ) where J_ℓ = ∂output/∂input.
        σ_max > 1 → layer amplifies perturbations (chaotic regime).
        σ_max < 1 → layer damps perturbations (ordered regime).
        σ_max ≈ 1 → edge of chaos (critical regime).

        For AttnRes models, computes separate σ for attn and MLP sub-layers
        (since they receive independently routed inputs in the actual forward).

        Uses warm-started eigenvectors from previous call for faster convergence.
        """
        if self._probe_batch is None:
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        torch.cuda.empty_cache()
        batch = self._probe_batch[:n_sequences].to(device)

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}

        try:
            # Get hidden states at each sampled layer via probe forward
            with torch.no_grad():
                hidden_states, _ = self._probe_forward(batch)

            self._eoc_call_count += 1
            metrics.update(self._eoc_metrics(hidden_states, device, n_iters))
            metrics["eoc/time_s"] = time.time() - t0

            logger.info(
                "EoC [step %d]: σ_max=%.4f, σ_mean=%.4f, layer0=%.1f, time=%.2fs",
                step,
                metrics.get("eoc/sigma_max", 0),
                metrics.get("eoc/sigma_mean", 0),
                metrics.get("eoc/layer0_sigma", 0),
                time.time() - t0,
            )

        except Exception as e:
            logger.warning("Jacobian spectral computation failed: %s", e)

        if was_training:
            self.model.train()

        return metrics

    COLD_ITERS = 20      # iterations from a random vector (cold start AND re-randomize)
    EOC_REL_TOL = 1e-3   # stop once σ changes by less than this (relative)

    def _power_iterate(
        self,
        h_detached: torch.Tensor,
        layer_fn: Any,
        warm_key: str,
        device: torch.device,
        n_iters: int,
    ) -> tuple[float, float]:
        """Power iteration on the Jacobian J = ∂f/∂h of layer_fn at h_detached.

        Iterates vector-Jacobian products only (v ← Jᵀv), so σ is the dominant
        |eigenvalue| of J — the spectral RADIUS, the edge-of-chaos quantity — not
        the spectral norm (that needs Jᵀ J, i.e. forward-mode AD, which the FA4
        attention path does not support). Returns (σ, relative gain); the
        relative gain σ·‖h‖/‖f(h)‖ is scale-free (log-log amplification), so layers
        whose input sits at a very different scale (layer 0 reads the raw embedding)
        compare with the rest — raw σ there is ~‖W‖/rms(embedding), not chaos.

        Re-randomizes the warm vector every 10 calls to avoid drifting onto a
        subdominant direction. A random start (cold OR re-randomized) gets
        COLD_ITERS iterations: the warm budget (5) from scratch under-converges and
        prints a sawtooth into every σ series (a dip each 10th call, then a creep
        back up).
        Iteration stops early once σ is stable to EOC_REL_TOL.
        """
        cold_start = warm_key not in self._warm_vectors
        rerandomize = (self._eoc_call_count % 10 == 0) and not cold_start

        v = None if (cold_start or rerandomize) else self._warm_vectors[warm_key].to(device)
        if v is None or v.shape != h_detached.shape:
            v = torch.randn_like(h_detached)
            n_iters = max(n_iters, self.COLD_ITERS)
        v = v / v.norm()

        sigma, prev = 0.0, None
        for it in range(n_iters):
            h_input = h_detached.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                layer_out = layer_fn(h_input)
            Jt_v = torch.autograd.grad(
                (layer_out * v).sum(), h_input,
            )[0].float()
            sigma = Jt_v.norm().item()
            if sigma < 1e-10:
                break
            v = (Jt_v / sigma).detach()
            if prev is not None and it >= 2 and abs(sigma - prev) <= self.EOC_REL_TOL * sigma:
                break
            prev = sigma

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_norm = layer_fn(h_detached).float().norm().item()
        in_norm = h_detached.float().norm().item()
        rel = sigma * in_norm / out_norm if out_norm > 0 else float("nan")

        self._warm_vectors[warm_key] = v.cpu()
        return sigma, rel

    def _eoc_metrics(
        self,
        hidden_states: dict[int, torch.Tensor],
        device: torch.device,
        n_iters: int,
        n_seq: Optional[int] = None,
    ) -> dict[str, float]:
        """Per-layer Jacobian σ (raw) + relative gain, and interior aggregates.

        AttnRes: attn and MLP sub-layers are probed at their own routed inputs
        (pre-attn from attn_norm, pre-MLP from ffn_norm).
        """
        metrics: dict[str, float] = {}
        use_attn_res = getattr(self.model.config, "attn_res", False)
        rope_cos, rope_sin = self.model.rope_cos, self.model.rope_sin
        mlp_inputs = getattr(self, "_mlp_inputs", {})
        cut = (lambda t: t[:n_seq]) if n_seq else (lambda t: t)

        for layer_idx in self.config.tier1_sample_layers:
            if layer_idx not in hidden_states:
                continue
            layer = self.model.layers[layer_idx]
            h_attn = cut(hidden_states[layer_idx]).detach()
            base = f"layer_{layer_idx}"
            if use_attn_res:
                h_mlp = cut(mlp_inputs.get(layer_idx, hidden_states[layer_idx])).detach()
                for name, h, fn in [
                    ("attn", h_attn, lambda h, _l=layer: h + _l.attn(_l.attn_norm(h), rope_cos, rope_sin)),
                    ("mlp", h_mlp, lambda h, _l=layer: h + _l.ffn(_l.ffn_norm(h))),
                ]:
                    sigma, rel = self._power_iterate(h, fn, f"jacobian_v_{layer_idx}_{name}", device, n_iters)
                    metrics[f"eoc/jacobian_sigma/{base}/{name}"] = sigma
                    metrics[f"eoc/jacobian_gain/{base}/{name}"] = rel
                metrics[f"eoc/jacobian_sigma/{base}"] = max(
                    metrics[f"eoc/jacobian_sigma/{base}/attn"], metrics[f"eoc/jacobian_sigma/{base}/mlp"])
                metrics[f"eoc/jacobian_gain/{base}"] = max(
                    metrics[f"eoc/jacobian_gain/{base}/attn"], metrics[f"eoc/jacobian_gain/{base}/mlp"])
            else:
                sigma, rel = self._power_iterate(
                    h_attn, lambda h, _l=layer: _l(h, rope_cos, rope_sin),
                    f"jacobian_v_{layer_idx}", device, n_iters)
                metrics[f"eoc/jacobian_sigma/{base}"] = sigma
                metrics[f"eoc/jacobian_gain/{base}"] = rel

        # Layer 0 raw σ is an embedding-scale artefact ("Spike No More", Takase et al.
        # COLM 2025): report it apart and keep it out of the raw aggregates. The
        # scale-free gain aggregates include every layer.
        layer0_key = "eoc/jacobian_sigma/layer_0"
        if layer0_key in metrics:
            metrics["eoc/layer0_sigma"] = metrics[layer0_key]
        for fam, agg, skip in [("eoc/jacobian_sigma/layer_", "eoc/sigma", layer0_key),
                               ("eoc/jacobian_gain/layer_", "eoc/gain", None)]:
            vals = [v for k, v in metrics.items()
                    if k.startswith(fam) and k[-1].isdigit() and k != skip]
            if vals:
                metrics[f"{agg}_max"] = max(vals)
                metrics[f"{agg}_min"] = min(vals)
                metrics[f"{agg}_mean"] = sum(vals) / len(vals)
        return metrics

    # =========================================================================
    # Unified monitoring: shared probe forward, all metrics
    # =========================================================================

    def compute_all(self, step: int, n_eoc_iters: int = 5) -> dict[str, float]:
        """
        Unified geometric monitoring with a single shared probe forward.

        Computes tier1 metrics (RankMe, stable rank, anisotropy, dead units,
        attention entropy), TwoNN intrinsic dimensionality, and EoC Jacobian
        spectral radius from one forward pass.

        NOT decorated @torch.no_grad — EoC power iteration needs autograd.
        The probe forward and no-grad metrics use explicit context blocks.
        """
        if self._probe_batch is None:
            logger.warning("No probe batch set — skipping compute_all")
            return {}

        t0 = time.time()
        device = next(self.model.parameters()).device
        torch.cuda.empty_cache()
        batch = self._probe_batch.to(device)

        was_training = self.model.training
        self.model.eval()

        metrics: dict[str, float] = {}

        try:
            # == Phase 1: probe forward + activation-based metrics (no grad) ==
            with torch.no_grad():
                hidden_states, attn_weights = self._probe_forward(batch)

                # -- RankMe (effective rank) on last-layer hidden states --
                last_layer_idx = len(self.model.layers) - 1
                if last_layer_idx in hidden_states:
                    last_hidden = hidden_states[last_layer_idx]
                    H = last_hidden.reshape(-1, last_hidden.shape[-1]).float()
                    metrics["geo/rankme_last"] = _rankme(H)

                # -- Per-layer metrics --
                for layer_idx in self.config.tier1_sample_layers:
                    prefix = f"geo/layer_{layer_idx}"
                    layer = self.model.layers[layer_idx]

                    for name, param in [
                        ("q_proj", layer.attn.q_proj.weight),
                        ("k_proj", layer.attn.k_proj.weight),
                        ("o_proj", layer.attn.o_proj.weight),
                        ("gate_proj", layer.ffn.gate_proj.weight),
                        ("down_proj", layer.ffn.down_proj.weight),
                    ]:
                        metrics[f"{prefix}/stable_rank_{name}"] = _stable_rank(param)

                    metrics.update(self._mlp_unit_metrics(layer_idx, prefix))
                    if layer_idx in hidden_states:
                        h = hidden_states[layer_idx]
                        h_flat = h.reshape(-1, h.shape[-1])
                        metrics[f"{prefix}/anisotropy"] = _anisotropy(h_flat, max_samples=512)

                    if layer_idx in attn_weights:
                        ent_mean, ent_std = _attention_entropy_stats(attn_weights[layer_idx])
                        metrics[f"{prefix}/attn_entropy_mean"] = ent_mean
                        metrics[f"{prefix}/attn_entropy_std"] = ent_std

                # -- AttnRes diagnostics --
                if self._attn_res_diagnostics:
                    metrics.update(self._attn_res_diagnostics)

                # -- TwoNN intrinsic dimensionality --
                for layer_idx in self.config.tier2_twonn_layers:
                    if layer_idx in hidden_states:
                        h = hidden_states[layer_idx].reshape(
                            -1, hidden_states[layer_idx].shape[-1]
                        )
                        n = min(self.config.tier2_twonn_samples, h.shape[0])
                        idx = torch.randperm(h.shape[0])[:n]
                        h_sub = h[idx].float()
                        id_est = _twonn_id(h_sub)
                        if id_est is not None:
                            metrics[f"geo/twonn_id/layer_{layer_idx}"] = id_est

            # == Phase 2: EoC Jacobian spectral radius (needs autograd) ==
            self._eoc_call_count += 1
            metrics.update(self._eoc_metrics(hidden_states, device, n_eoc_iters, n_seq=4))

        except Exception as e:
            logger.warning("compute_all failed: %s", e, exc_info=True)

        if was_training:
            self.model.train()

        elapsed = time.time() - t0
        metrics["geo/compute_all_time_s"] = elapsed
        metrics["geo/step"] = float(step)

        logger.info(
            "Geo [step %d]: RankMe=%.1f, σ_max=%.4f, time=%.2fs",
            step,
            metrics.get("geo/rankme_last", 0),
            metrics.get("eoc/sigma_max", 0),
            elapsed,
        )

        return metrics

    # =========================================================================
    # Tier 1: Lightweight streaming metrics (< 1% overhead)
    # =========================================================================

    @torch.no_grad()
    def tier1(self, step: int, probe_batch: Optional[torch.Tensor] = None) -> dict[str, float]:
        """
        Compute Tier 1 geometric health metrics.

        Requires a forward pass on the probe batch. Returns a flat dict
        of metric name → value.
        """
        t0 = time.time()
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is None:
            logger.warning("No probe batch set — skipping Tier 1 metrics")
            return {}

        device = next(self.model.parameters()).device
        batch = batch.to(device)

        metrics: dict[str, float] = {}

        # Run forward pass with hooks to capture hidden states and attention
        hidden_states, attn_weights = self._probe_forward(batch)

        # -- RankMe (effective rank) on last-layer hidden states --
        last_hidden = hidden_states[len(self.model.layers) - 1]  # [batch, seq, hidden]
        # Flatten to [n_samples, hidden_dim]
        H = last_hidden.reshape(-1, last_hidden.shape[-1]).float()
        metrics["geo/rankme_last"] = _rankme(H)

        # -- Per-layer metrics --
        for layer_idx in self.config.tier1_sample_layers:
            prefix = f"geo/layer_{layer_idx}"

            # Stable rank of weight matrices
            layer = self.model.layers[layer_idx]
            for name, param in [
                ("q_proj", layer.attn.q_proj.weight),
                ("k_proj", layer.attn.k_proj.weight),
                ("o_proj", layer.attn.o_proj.weight),
                ("gate_proj", layer.ffn.gate_proj.weight),
                ("down_proj", layer.ffn.down_proj.weight),
            ]:
                sr = _stable_rank(param)
                metrics[f"{prefix}/stable_rank_{name}"] = sr

            # Dead unit fraction (fraction of neurons with near-zero activation)
            metrics.update(self._mlp_unit_metrics(layer_idx, prefix))

            # Anisotropy (average pairwise cosine similarity)
            if layer_idx in hidden_states:
                h = hidden_states[layer_idx].reshape(
                    -1, hidden_states[layer_idx].shape[-1]
                )
                aniso = _anisotropy(h, max_samples=512)
                metrics[f"{prefix}/anisotropy"] = aniso

            # Attention entropy
            if layer_idx in attn_weights:
                attn = attn_weights[layer_idx]  # [batch, n_heads, seq, seq]
                ent_mean, ent_std = _attention_entropy_stats(attn)
                metrics[f"{prefix}/attn_entropy_mean"] = ent_mean
                metrics[f"{prefix}/attn_entropy_std"] = ent_std

        # Merge AttnRes diagnostics if present
        if self._attn_res_diagnostics:
            metrics.update(self._attn_res_diagnostics)

        elapsed = time.time() - t0
        metrics["geo/tier1_time_s"] = elapsed
        metrics["geo/step"] = float(step)

        logger.info(
            "Tier 1 [step %d]: RankMe=%.1f, time=%.2fs",
            step,
            metrics.get("geo/rankme_last", 0),
            elapsed,
        )

        return metrics

    # =========================================================================
    # Tier 2: Checkpoint-level metrics (minutes)
    # =========================================================================

    @torch.no_grad()
    def tier2(self, step: int, probe_batch: Optional[torch.Tensor] = None) -> dict[str, float]:
        """
        Compute Tier 2 geometric health metrics.

        TwoNN intrinsic dimensionality at sampled layers.
        """
        t0 = time.time()
        metrics: dict[str, float] = {}

        # -- TwoNN intrinsic dimensionality at sampled layers --
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is not None:
            device = next(self.model.parameters()).device
            batch = batch.to(device)
            hidden_states, _ = self._probe_forward(batch)

            for layer_idx in self.config.tier2_twonn_layers:
                if layer_idx in hidden_states:
                    h = hidden_states[layer_idx].reshape(
                        -1, hidden_states[layer_idx].shape[-1]
                    )
                    # Subsample for speed
                    n = min(self.config.tier2_twonn_samples, h.shape[0])
                    idx = torch.randperm(h.shape[0])[:n]
                    h_sub = h[idx].float()
                    id_est = _twonn_id(h_sub)
                    if id_est is not None:
                        metrics[f"geo/twonn_id/layer_{layer_idx}"] = id_est

        elapsed = time.time() - t0
        metrics["geo/tier2_time_s"] = elapsed

        logger.info(
            "Tier 2 [step %d]: TwoNN layers=%d, time=%.1fs",
            step,
            sum(1 for k in metrics if k.startswith("geo/twonn_id")),
            elapsed,
        )

        return metrics

    # =========================================================================
    # Forward pass with hooks
    # =========================================================================

    def _probe_forward(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        Run a forward pass on the probe batch, capturing hidden states
        and attention weights at sampled layers.

        When AttnRes is enabled, uses the AttnRes forward path to capture
        per-layer hidden states that match what training actually computes.
        The "last layer" hidden state is the final AttnRes aggregation output.

        Returns (hidden_states, attn_weights) dicts keyed by layer index.
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}

        # Determine which layers we need activations from
        needed_layers = set(self.config.tier1_sample_layers) | set(
            self.config.tier2_twonn_layers
        )
        needed_layers.add(len(self.model.layers) - 1)  # always need last layer

        was_training = self.model.training
        self.model.eval()

        use_attn_res = getattr(self.model.config, "attn_res", False)

        # Cap probe batch: the uncompiled model forward under DDP crashes with
        # large batches (64 seq) but works fine with <=16.  16 × seq_len tokens
        # is more than enough for geometric statistics.
        if use_attn_res and input_ids.shape[0] > 16:
            input_ids = input_ids[:16]

        # MLP neuron activity: the SwiGLU output feeding down_proj, reduced per
        # neuron inside the hook (the residual stream has no "dead" dims to find).
        self._mlp_mean_abs = {}
        self._mlp_inputs: dict[int, torch.Tensor] = {}  # AttnRes pre-MLP routed inputs
        mlp_hooks: list[Any] = []

        def _make_mlp_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple) -> None:
                act = args[0].detach().float().abs()
                self._mlp_mean_abs[layer_idx] = act.reshape(-1, act.shape[-1]).mean(dim=0)
            return hook

        for i in self.config.tier1_sample_layers:
            ffn = getattr(self.model.layers[i], "ffn", None)
            if ffn is not None and hasattr(ffn, "down_proj"):
                mlp_hooks.append(ffn.down_proj.register_forward_pre_hook(_make_mlp_hook(i)))

        try:
            self._probe_forward_inner(input_ids, needed_layers, use_attn_res,
                                      hidden_states, attn_weights)
        finally:
            for h in mlp_hooks:
                h.remove()

        if was_training:
            self.model.train()

        return hidden_states, attn_weights

    def _probe_forward_inner(
        self,
        input_ids: torch.Tensor,
        needed_layers: set[int],
        use_attn_res: bool,
        hidden_states: dict[int, torch.Tensor],
        attn_weights: dict[int, torch.Tensor],
    ) -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if use_attn_res:
                hs, aw = self._probe_forward_attn_res(input_ids, needed_layers)
                hidden_states.update(hs)
                attn_weights.update(aw)
            else:
                x = self.model.embed_tokens(input_ids)
                rope_cos = self.model.rope_cos
                rope_sin = self.model.rope_sin

                for i, layer in enumerate(self.model.layers):
                    x = layer(x, rope_cos, rope_sin)
                    if i in needed_layers:
                        hidden_states[i] = x.detach()

                        if i in self.config.tier1_sample_layers:
                            attn_w = self._get_attention_weights(
                                layer, self.model.layers[i].attn_norm(x),
                                rope_cos, rope_sin, input_ids.shape[1]
                            )
                            if attn_w is not None:
                                attn_weights[i] = attn_w

    def _mlp_unit_metrics(self, layer_idx: int, prefix: str) -> dict[str, float]:
        """dead_mlp_units: neurons with mean |act| < 1e-6 over the probe;
        weak_mlp_units: below 1% of the layer's median neuron activity."""
        m = getattr(self, "_mlp_mean_abs", {}).get(layer_idx)
        if m is None:
            return {}
        med = m.median()
        return {
            f"{prefix}/dead_mlp_units": (m < 1e-6).float().mean().item(),
            f"{prefix}/weak_mlp_units": (m < 0.01 * med).float().mean().item(),
        }

    def _probe_forward_attn_res(
        self, input_ids: torch.Tensor, needed_layers: set[int]
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        AttnRes-aware probe forward pass using hooks on the model's own forward.

        Instead of manually re-implementing the AttnRes routing (which uses
        different Triton kernel call patterns than training and can crash under
        DDP), we hook into the model's own forward path — the same one that
        compute_sharpness uses successfully.

        Forward pre-hooks on attn_norm capture the routed hidden state h (input
        to normalization) at each layer. A pre-hook on the final norm captures
        the last-layer AttnRes aggregation output.
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}
        captured_normed: dict[int, torch.Tensor] = {}

        model = self.model
        last_layer_idx = len(model.layers) - 1
        hooks: list[Any] = []

        def _make_norm_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple, output: torch.Tensor) -> None:
                hidden_states[layer_idx] = args[0].detach()
                if layer_idx in self.config.tier1_sample_layers:
                    captured_normed[layer_idx] = output.detach()
            return hook

        def _final_norm_hook(_module: nn.Module, args: tuple) -> None:
            hidden_states[last_layer_idx] = args[0].detach()

        # The MLP sub-layer reads its OWN routed input (pre-MLP AttnRes), not the
        # pre-attn one; EoC must probe its Jacobian there (probing the MLP at the
        # pre-attn input puts layer 0's MLP on the raw embedding it never sees).
        def _make_ffn_norm_hook(layer_idx: int):
            def hook(_module: nn.Module, args: tuple) -> None:
                self._mlp_inputs[layer_idx] = args[0].detach()
            return hook

        for i in needed_layers:
            if i < len(model.layers):
                hooks.append(
                    model.layers[i].attn_norm.register_forward_hook(
                        _make_norm_hook(i)
                    )
                )
                if i in self.config.tier1_sample_layers:
                    hooks.append(
                        model.layers[i].ffn_norm.register_forward_pre_hook(
                            _make_ffn_norm_hook(i)
                        )
                    )
        hooks.append(model.norm.register_forward_pre_hook(_final_norm_hook))

        try:
            model(input_ids)
        finally:
            for h in hooks:
                h.remove()

        rope_cos = model.rope_cos
        rope_sin = model.rope_sin
        for layer_idx, normed_h in captured_normed.items():
            attn_w = self._get_attention_weights(
                model.layers[layer_idx], normed_h,
                rope_cos, rope_sin, input_ids.shape[1],
            )
            if attn_w is not None:
                attn_weights[layer_idx] = attn_w

        # --- AttnRes diagnostics (computed by _forward_attn_res during eval) ---
        self._attn_res_diagnostics.clear()
        if hasattr(model, "_last_attn_res_diagnostics"):
            self._attn_res_diagnostics.update(model._last_attn_res_diagnostics)

        return hidden_states, attn_weights

    def _get_attention_weights(
        self,
        layer: nn.Module,
        x_normed: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        seq_len: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute raw attention weights for a layer.

        We compute Q @ K^T / sqrt(d) and softmax manually to get the
        weights without relying on SDPA (which doesn't return them).
        Uses a subset of the batch to keep memory bounded.
        """
        try:
            attn = layer.attn
            # Limit to first few sequences to save memory
            x_sub = x_normed[:4]
            bsz = x_sub.shape[0]

            q = attn.q_proj(x_sub).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x_sub).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            if attn.qk_norm:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            from kotodama.model.llama import apply_rope

            q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
            k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

            # Expand KV for GQA
            if attn.num_kv_groups > 1:
                k = k.repeat_interleave(attn.num_kv_groups, dim=1)

            # Compute attention weights
            scale = attn.head_dim ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

            # Causal mask
            causal = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores.masked_fill_(causal, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            return weights.detach()
        except Exception as e:
            logger.debug("Failed to compute attention weights: %s", e)
            return None


# =============================================================================
# Metric computation functions (stateless, pure)
# =============================================================================


def _rankme(H: torch.Tensor, eps: float = 1e-7) -> float:
    """
    Compute RankMe (effective rank) of a representation matrix.

    RankMe = exp(entropy of normalized singular values)
    Higher = more diverse/spread representation.
    """
    # H: [n_samples, hidden_dim]
    # SVD of the (potentially large) matrix — use only singular values
    try:
        S = torch.linalg.svdvals(H)
        S = S / (S.sum() + eps)
        S = S[S > eps]
        entropy = -(S * torch.log(S)).sum()
        return torch.exp(entropy).item()
    except Exception:
        return 0.0


def _stable_rank(W: torch.Tensor) -> float:
    """
    Compute stable rank: ||W||_F^2 / ||W||_2^2.

    Measures effective dimensionality of a weight matrix.
    Stable rank 1 = rank-1 matrix. Higher = more distributed.
    """
    try:
        W_f = W.float()
        frob_sq = W_f.pow(2).sum()
        spectral_sq = torch.linalg.svdvals(W_f)[0].pow(2)
        return (frob_sq / (spectral_sq + 1e-10)).item()
    except Exception:
        return 0.0


def _anisotropy(H: torch.Tensor, max_samples: int = 512) -> float:
    """
    Average pairwise cosine similarity (anisotropy measure).

    High anisotropy (close to 1) = representations are clustered.
    Low anisotropy (close to 0) = representations are spread.
    """
    if H.shape[0] > max_samples:
        idx = torch.randperm(H.shape[0])[:max_samples]
        H = H[idx]

    H = H.float()
    # Normalize rows
    H_norm = F.normalize(H, dim=-1)
    # Compute mean cosine similarity (excluding self-similarity)
    sim = H_norm @ H_norm.T
    n = sim.shape[0]
    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    mean_sim = sim[mask].mean()
    return mean_sim.item()


def _attention_entropy_stats(
    attn: torch.Tensor, eps: float = 1e-10
) -> tuple[float, float]:
    """
    Compute mean and std of per-head attention entropy.

    attn: [batch, n_heads, seq, seq]
    Returns (mean_entropy, std_entropy) across heads.
    """
    # Entropy per head per position: -sum(p * log(p))
    attn_clamped = attn.float().clamp(min=eps)
    entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # [batch, heads, seq]
    # Average over batch and sequence positions
    per_head = entropy.mean(dim=(0, 2))  # [heads]
    return per_head.mean().item(), per_head.std().item()




def _twonn_id(X: torch.Tensor) -> Optional[float]:
    """
    Estimate intrinsic dimensionality using the TwoNN method.

    Facco et al. (2017): "Estimating the intrinsic dimension of
    datasets by a minimal neighborhood information."

    Uses the ratio of second-nearest to nearest neighbor distances.
    """
    try:
        n = X.shape[0]
        if n < 10:
            return None

        # Compute pairwise distances
        # For memory efficiency, compute in chunks if large
        if n > 5000:
            X = X[:5000]
            n = 5000

        dists = torch.cdist(X, X)
        # Set self-distance to infinity
        dists.fill_diagonal_(float("inf"))

        # Get two nearest neighbors
        topk = dists.topk(2, dim=1, largest=False)
        r1 = topk.values[:, 0]  # nearest neighbor distance
        r2 = topk.values[:, 1]  # second nearest

        # Filter out zero distances
        valid = r1 > 1e-10
        if valid.sum() < 10:
            return None

        mu = r2[valid] / r1[valid]

        # Sort mu values
        mu_sorted = mu.sort().values
        n_valid = len(mu_sorted)

        # Empirical CDF
        i = torch.arange(1, n_valid + 1, dtype=torch.float32, device=X.device)
        # log(1 - F(mu)) = -d * log(mu), where F(mu) = i/n
        # So d = -log(1 - i/n) / log(mu)
        # Use linear regression: log(1 - i/(n+1)) vs log(mu)
        log_survival = torch.log(1.0 - i / (n_valid + 1))
        log_mu = torch.log(mu_sorted)

        # Linear regression (slope = -d)
        x = log_mu
        y = log_survival
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).pow(2).sum() + 1e-10)

        d = -slope.item()

        # Sanity check
        if d < 0.5 or d > 10000:
            return None

        return d
    except Exception:
        return None


