"""
Geometric health monitoring for luxia-base pretraining.

Implements the three-tier monitoring framework from the deep research
synthesis (research/pretraining/deep-research-synthesis.md):

Tier 1 (every N steps, <1% overhead):
  - RankMe (effective rank) on probe batch hidden states
  - Stable rank per layer (weight-space)
  - Anisotropy (average pairwise cosine similarity)
  - Dead unit fraction per layer
  - Attention entropy distribution

Tier 2 (every checkpoint, minutes):
  - WeightWatcher alpha per layer (weight-only, no forward pass)
  - TwoNN intrinsic dimensionality at sampled layers
  - Eigenspectrum decay rate (alpha-ReQ proxy)

Tier 3 (5 key checkpoints, longer):
  - Full WeightWatcher analysis
  - Full Anamnesis extraction
  - (Deferred to Track D)

Usage in training loop::

    monitor = GeometricMonitor(model, config)
    # In training loop:
    if step % tier1_every == 0:
        metrics = monitor.tier1(probe_batch, step)
    if step % tier2_every == 0:
        metrics = monitor.tier2(step)
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
class MonitorConfig:
    """Configuration for geometric monitoring."""

    # Tier 1: lightweight streaming metrics
    tier1_every: int = 500  # steps between Tier 1 measurements
    tier1_probe_size: int = 1024  # number of samples in probe batch
    tier1_sample_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select

    # Tier 2: checkpoint-level metrics
    tier2_every: int = 5000  # steps between Tier 2 measurements
    tier2_twonn_samples: int = 3000  # samples for TwoNN ID estimation
    tier2_twonn_layers: list[int] = field(
        default_factory=lambda: []
    )  # empty = auto-select (5 evenly spaced)

    # Tier 3: deep analysis (at specific token counts)
    tier3_token_checkpoints: list[int] = field(
        default_factory=lambda: [
            0,
            8_000_000_000,
            24_000_000_000,
            48_000_000_000,
            80_000_000_000,
        ]
    )

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
            if layer_idx in hidden_states:
                h = hidden_states[layer_idx]
                dead_frac = _dead_unit_fraction(h)
                metrics[f"{prefix}/dead_units"] = dead_frac

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

        Weight-space metrics (WeightWatcher alpha) use the real WeightWatcher
        library for proper power-law fitting. TwoNN ID estimation requires
        a forward pass.
        """
        t0 = time.time()
        metrics: dict[str, float] = {}

        # -- WeightWatcher alpha (real library) --
        try:
            import weightwatcher as ww

            watcher = ww.WeightWatcher(model=self.model)
            details = watcher.analyze(min_evals=10, plot=False)

            all_alphas = details["alpha"].dropna().tolist()
            if all_alphas:
                metrics["geo/ww_alpha_mean"] = sum(all_alphas) / len(all_alphas)
                metrics["geo/ww_alpha_std"] = _std(all_alphas)
                metrics["geo/ww_alpha_min"] = min(all_alphas)
                metrics["geo/ww_alpha_max"] = max(all_alphas)
                healthy = sum(1 for a in all_alphas if 2.0 < a < 4.0)
                metrics["geo/ww_alpha_healthy_frac"] = healthy / len(all_alphas)

                # Per-weight-type breakdown (more useful than per-layer)
                for idx, row in details.iterrows():
                    name = str(row.get("name", ""))
                    alpha = row["alpha"]
                    if str(alpha) == "nan":
                        continue
                    # Classify by weight type
                    for wtype in ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]:
                        if wtype in name:
                            key = f"geo/ww_alpha_by_type/{wtype}"
                            if key not in metrics:
                                metrics[key] = []
                            metrics[key].append(alpha)

                # Average per weight type
                for key in list(metrics.keys()):
                    if key.startswith("geo/ww_alpha_by_type/") and isinstance(metrics[key], list):
                        vals = metrics[key]
                        wtype = key.split("/")[-1]
                        metrics[f"geo/ww_alpha_by_type/{wtype}"] = sum(vals) / len(vals)

        except ImportError:
            logger.warning("weightwatcher not installed — using proxy alpha")
            # Fallback to proxy computation for all layers
            for layer_idx in range(len(self.model.layers)):
                layer = self.model.layers[layer_idx]
                for name, param in [
                    ("q_proj", layer.attn.q_proj.weight),
                    ("o_proj", layer.attn.o_proj.weight),
                    ("gate_proj", layer.ffn.gate_proj.weight),
                    ("down_proj", layer.ffn.down_proj.weight),
                ]:
                    alpha = _weightwatcher_alpha(param)
                    if alpha is not None:
                        metrics[f"geo/ww_alpha/layer_{layer_idx}/{name}"] = alpha
            all_alphas = [v for k, v in metrics.items() if "ww_alpha" in k and isinstance(v, (int, float))]
            if all_alphas:
                metrics["geo/ww_alpha_mean"] = sum(all_alphas) / len(all_alphas)
                healthy = sum(1 for a in all_alphas if 2.0 < a < 4.0)
                metrics["geo/ww_alpha_healthy_frac"] = healthy / len(all_alphas)
        except Exception as e:
            logger.warning("WeightWatcher analysis failed (%s): %s — falling back to proxy alpha", type(e).__name__, e)
            # Fall back to proxy computation
            for layer_idx in range(len(self.model.layers)):
                layer = self.model.layers[layer_idx]
                for name, param in [
                    ("q_proj", layer.attn.q_proj.weight),
                    ("o_proj", layer.attn.o_proj.weight),
                    ("gate_proj", layer.ffn.gate_proj.weight),
                    ("down_proj", layer.ffn.down_proj.weight),
                ]:
                    alpha = _weightwatcher_alpha(param)
                    if alpha is not None:
                        metrics[f"geo/ww_alpha/layer_{layer_idx}/{name}"] = alpha
            all_alphas = [v for k, v in metrics.items() if "ww_alpha" in k and isinstance(v, (int, float))]
            if all_alphas:
                metrics["geo/ww_alpha_mean"] = sum(all_alphas) / len(all_alphas)
                healthy = sum(1 for a in all_alphas if 2.0 < a < 4.0)
                metrics["geo/ww_alpha_healthy_frac"] = healthy / len(all_alphas)

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
            "Tier 2 [step %d]: WW_alpha_mean=%.2f, healthy=%.0f%%, time=%.1fs",
            step,
            metrics.get("geo/ww_alpha_mean", 0),
            metrics.get("geo/ww_alpha_healthy_frac", 0) * 100,
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

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if use_attn_res:
                hidden_states, attn_weights = self._probe_forward_attn_res(
                    input_ids, needed_layers
                )
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

        if was_training:
            self.model.train()

        return hidden_states, attn_weights

    def _probe_forward_attn_res(
        self, input_ids: torch.Tensor, needed_layers: set[int]
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """
        AttnRes-aware probe forward pass.

        Runs the actual AttnRes routing (matching training) while capturing
        the pre-attention hidden state `h` at each layer. For the last layer,
        captures the final AttnRes aggregation (the actual model output).
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}

        model = self.model
        embed = model.embed_tokens(input_ids)
        committed: list[torch.Tensor] = []
        partial = embed
        boundary_set = model._attn_res_boundary_set
        rope_cos = model.rope_cos
        rope_sin = model.rope_sin
        last_layer_idx = len(model.layers) - 1

        max_s = model._attn_res_max_sources
        masks = model._attn_res_masks
        zero = torch.zeros_like(embed)

        def _pad_and_stack(committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
            sources = committed + [partial]
            while len(sources) < max_s:
                sources.append(zero)
            return torch.stack(sources, dim=0)

        for i, layer in enumerate(model.layers):
            # AttnRes routing (matches _forward_attn_res exactly)
            buf = _pad_and_stack(committed, partial)
            h = model._route_static(buf, layer.attn_res_query, layer.attn_res_norm, masks[2 * i])

            # Block boundary
            if i in boundary_set:
                committed.append(partial.clone())
                partial = torch.zeros_like(embed)

            # Capture h
            if i in needed_layers:
                hidden_states[i] = h.detach()

                if i in self.config.tier1_sample_layers:
                    attn_w = self._get_attention_weights(
                        layer, layer.attn_norm(h),
                        rope_cos, rope_sin, input_ids.shape[1]
                    )
                    if attn_w is not None:
                        attn_weights[i] = attn_w

            # Attention sub-layer
            attn_out = layer.attn(layer.attn_norm(h), rope_cos, rope_sin)
            partial = partial + attn_out

            # Pre-MLP routing
            buf = _pad_and_stack(committed, partial)
            h = model._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1])

            # MLP sub-layer
            mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        # Final aggregation
        buf = _pad_and_stack(committed, partial)
        final_h = model._route_static(buf, model.final_res_query, model.final_res_norm, masks[2 * len(model.layers)])
        hidden_states[last_layer_idx] = final_h.detach()

        # --- AttnRes diagnostics ---
        self._attn_res_diagnostics.clear()
        try:
            # Compute routing weights for the final aggregation
            qw = model.final_res_query * model.final_res_norm.weight
            eps = model.final_res_norm.eps
            rsqrt = torch.rsqrt(buf.pow(2).mean(-1) + eps)
            logits = (buf * qw).sum(-1) * rsqrt
            final_mask = masks[2 * len(model.layers)]
            logits = logits.masked_fill(~final_mask.view(-1, 1, 1), float("-inf"))
            alpha_weights = F.softmax(logits, dim=0)
            avg_alpha = alpha_weights.mean(dim=(1, 2)).detach()
            n_active = final_mask.sum().item()
            for block_idx in range(n_active):
                self._attn_res_diagnostics[f"attnres/final_alpha/block_{block_idx}"] = avg_alpha[block_idx].item()
                norm_val = buf[block_idx].detach().float().norm(dim=-1).mean().item()
                self._attn_res_diagnostics[f"attnres/block_norm/{block_idx}"] = norm_val
        except Exception as e:
            logger.debug("AttnRes diagnostics failed: %s", e)

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

            from src.model.llama import apply_rope

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


def _dead_unit_fraction(
    hidden: torch.Tensor, threshold: float = 1e-6
) -> float:
    """
    Fraction of neurons with near-zero mean absolute activation.

    hidden: [batch, seq_len, hidden_dim]
    """
    # Mean absolute activation per neuron across batch and sequence
    mean_abs = hidden.float().abs().mean(dim=(0, 1))  # [hidden_dim]
    dead = (mean_abs < threshold).float().mean()
    return dead.item()


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


def _weightwatcher_alpha(
    W: torch.Tensor, min_sv: int = 10
) -> Optional[float]:
    """
    Estimate WeightWatcher power-law alpha for a weight matrix.

    Fits a power law to the eigenspectrum of W^T W.
    Alpha in (2, 4) = well-trained. Alpha > 6 = undertrained.

    This is a simplified version — full WeightWatcher uses
    more sophisticated fitting (KS test, xmin estimation).
    We use a log-log linear regression on the sorted eigenvalues
    as a fast proxy.
    """
    try:
        W_f = W.float().detach()
        # Get eigenvalues of W^T W (= squared singular values)
        S = torch.linalg.svdvals(W_f)
        eigs = S.pow(2)

        if len(eigs) < min_sv:
            return None

        # Sort descending, take top portion (skip very small eigenvalues)
        eigs_sorted = eigs.sort(descending=True).values
        # Use all eigenvalues above a threshold
        threshold = eigs_sorted[0] * 1e-10
        eigs_valid = eigs_sorted[eigs_sorted > threshold]

        if len(eigs_valid) < min_sv:
            return None

        # Log-log linear regression: log(eig) ~ -alpha * log(rank)
        n = len(eigs_valid)
        log_rank = torch.log(torch.arange(1, n + 1, dtype=torch.float32, device=W.device))
        log_eig = torch.log(eigs_valid)

        # Simple linear regression
        x = log_rank
        y = log_eig
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean).pow(2).sum() + 1e-10)

        # Alpha is the negative slope (eigenvalues decay as rank^{-alpha})
        # But WeightWatcher alpha is defined differently — it's the tail exponent
        # of the empirical spectral density. For a Marchenko-Pastur + power-law
        # tail, alpha ~ 1 + 1/|slope|. We use the simpler |slope| as our proxy.
        alpha = -slope.item()

        # Sanity check
        if alpha < 0.1 or alpha > 20:
            return None

        return alpha
    except Exception:
        return None


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


def _std(values: list[float]) -> float:
    """Standard deviation of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5
