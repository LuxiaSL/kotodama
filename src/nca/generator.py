"""
NCA trajectory generator for luxia-base pre-pre-training.

Generates Neural Cellular Automata trajectories as tokenized sequences
for training attention circuits before language pretraining.

Adapted from Han et al. (arxiv.org/html/2603.10055v1) with extensions
per the luxia-base design brief (spec Section 5.1):

1. **Multi-channel parallel tracking** — n_groups channels evolve in
   parallel within each rule. Channels are serialized as sequential
   frames for tokenization (avoids vocab explosion while teaching the
   model to track multiple simultaneous information threads).

2. **Mixed complexity** — rules are sampled from a distribution of
   architectures: varying kernel sizes (3×3, 5×5), hidden depths
   (1-3 layers), identity biases (0.5-2.0), temperatures (0.3-1.0).
   The model sees diverse dynamics, not a single regime.

3. **Larger grids** — 32×32 to 64×64 (spec says 32-64), producing
   longer, richer spatial patterns than the reference's 12×12.

4. **Class IV filtering** — gzip complexity in 0.4-0.7 targets
   edge-of-chaos dynamics (neither fixed points nor pure noise).

5. **Context sensitivity** — deeper rule networks (2-3 hidden layers)
   + larger kernels mean transitions depend on broader context, not
   just immediate neighbors.

Usage::

    python -m src.nca.generator \\
        --output data/nca_trajectories.bin \\
        --tokens 300_000_000 \\
        --device cuda:0

Reference: github.com/danihyunlee/nca-pre-pretraining
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NCAConfig:
    """Configuration for NCA trajectory generation."""

    # Grid parameters
    grid_size: int = 32  # H × W grid (spec: 32-64)
    d_state: int = 10  # discrete cell states per channel
    n_groups: int = 4  # parallel channels (spec: 4-8)

    # Rule network (these are defaults — overridden by mixed sampling)
    kernel_size: int = 3  # neighborhood size (3 or 5)
    hidden_dim: int = 32  # hidden layer width
    num_hidden_layers: int = 2  # hidden layers in rule network

    # Dynamics control
    identity_bias: float = 1.0  # logit bonus for current state
    temperature: float = 0.5  # sampling temperature

    # Trajectory
    num_steps: int = 128  # steps per trajectory
    burn_in: int = 10  # initial steps to discard

    # Complexity filtering
    filter_enabled: bool = True
    gzip_lower: float = 0.4
    gzip_upper: float = 0.7
    filter_steps: int = 30  # steps used for filtering evaluation

    # Tokenization
    patch_size: int = 2  # 2×2 patches → single tokens

    # Mixed complexity — if True, sample rule architecture per rule
    mixed_complexity: bool = True

    @property
    def vocab_size(self) -> int:
        """NCA token vocabulary size (per single-channel frame)."""
        return self.d_state ** (self.patch_size**2) + 2  # +2 for START, END

    @property
    def patches_per_frame(self) -> int:
        """Number of patch tokens per single-channel grid frame."""
        return (self.grid_size // self.patch_size) ** 2

    @property
    def tokens_per_frame(self) -> int:
        """Tokens per single-channel frame (patches + START + END)."""
        return self.patches_per_frame + 2

    @property
    def tokens_per_timestep(self) -> int:
        """Tokens per full timestep (all channels serialized)."""
        return self.tokens_per_frame * self.n_groups

    @property
    def tokens_per_trajectory(self) -> int:
        """Total tokens per trajectory."""
        return self.tokens_per_timestep * self.num_steps


# =============================================================================
# Rule network
# =============================================================================


class NCARule(nn.Module):
    """
    Neural network defining an NCA transition rule.

    Maps (d_state × n_groups) one-hot neighborhood → next-state logits.
    Channels interact through the shared perception convolution, enabling
    cross-channel information flow.

    Each random initialization produces a different automaton.
    """

    def __init__(
        self,
        d_state: int,
        n_groups: int,
        kernel_size: int,
        hidden_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.n_groups = n_groups
        in_channels = d_state * n_groups
        out_channels = d_state * n_groups
        pad = kernel_size // 2

        # Perception: gathers neighborhood across ALL channels
        # This is where cross-channel interaction happens
        self.perception = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode="circular",
        )

        # Hidden layers
        layers: list[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)

        # Output logits per channel per state
        self.output = nn.Conv2d(hidden_dim, out_channels, 1)

        self._random_init()

    def _random_init(self) -> None:
        """Random init with moderate variance for diverse dynamics."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.5)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.1)

    def forward(self, state_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_onehot: (B, d_state*n_groups, H, W)
        Returns:
            logits: (B, d_state*n_groups, H, W)
        """
        x = self.perception(state_onehot)
        x = F.relu(x)
        x = self.hidden(x)
        return self.output(x)


def sample_rule_config(config: NCAConfig) -> dict:
    """
    Sample a random rule architecture from the design space.

    Returns kwargs for NCARule plus dynamics parameters.
    When mixed_complexity is False, returns fixed config values.
    """
    if not config.mixed_complexity:
        return {
            "kernel_size": config.kernel_size,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "identity_bias": config.identity_bias,
            "temperature": config.temperature,
        }

    return {
        "kernel_size": random.choice([3, 5]),
        "hidden_dim": random.choice([16, 32, 48]),
        "num_hidden_layers": random.choice([1, 2, 3]),
        "identity_bias": random.uniform(0.5, 2.0),
        "temperature": random.uniform(0.3, 1.0),
    }


# =============================================================================
# Simulation
# =============================================================================


@torch.no_grad()
def simulate_trajectory(
    rule: NCARule,
    grid_size: int,
    d_state: int,
    n_groups: int,
    num_steps: int,
    burn_in: int,
    identity_bias: float,
    temperature: float,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Simulate an NCA trajectory.

    Returns:
        states: (batch, num_steps, n_groups, H, W) integer cell states
    """
    H, W = grid_size, grid_size
    d = d_state
    G = n_groups

    # Random initial state per channel
    # shape: (B, G, H, W)
    state = torch.randint(0, d, (batch_size, G, H, W), device=device)

    def step(s: torch.Tensor) -> torch.Tensor:
        """One NCA step: state (B, G, H, W) → next state (B, G, H, W)."""
        # One-hot encode: (B, G, H, W) → (B, G*d, H, W)
        s_flat = s.reshape(batch_size * G, H, W)
        onehot = F.one_hot(s_flat.long(), d)  # (B*G, H, W, d)
        onehot = onehot.permute(0, 3, 1, 2).float()  # (B*G, d, H, W)
        onehot = onehot.reshape(batch_size, G * d, H, W)  # (B, G*d, H, W)

        logits = rule(onehot)  # (B, G*d, H, W)

        # Add identity bias
        if identity_bias > 0:
            logits = logits + identity_bias * onehot

        # Reshape to (B, G, d, H, W) then sample per channel
        logits = logits.reshape(batch_size, G, d, H, W)
        logits = logits.permute(0, 1, 3, 4, 2)  # (B, G, H, W, d)
        logits_flat = logits.reshape(-1, d)  # (B*G*H*W, d)

        probs = F.softmax(logits_flat / max(temperature, 1e-6), dim=-1)
        next_state = torch.multinomial(probs, 1)  # (B*G*H*W, 1)
        return next_state.reshape(batch_size, G, H, W)

    # Burn-in
    for _ in range(burn_in):
        state = step(state)

    # Record trajectory
    trajectory = torch.zeros(
        batch_size, num_steps, G, H, W, dtype=torch.long, device=device
    )
    for t in range(num_steps):
        trajectory[:, t] = state
        state = step(state)

    return trajectory


# =============================================================================
# Tokenization
# =============================================================================


def tokenize_trajectory(
    trajectory: torch.Tensor,
    d_state: int,
    patch_size: int,
) -> np.ndarray:
    """
    Convert multi-channel grid trajectories to token sequences.

    Multi-channel strategy: each channel is serialized as a separate
    frame within each timestep. For 4 channels, timestep t becomes:
    [START, ch0_patches..., END, START, ch1_patches..., END, ...]

    This teaches the transformer to track parallel information streams
    through sequential attention, matching how it processes language.

    Args:
        trajectory: (B, T, G, H, W) integer cell states

    Returns:
        tokens: flat uint16 array
    """
    B, T, G, H, W = trajectory.shape
    ps = patch_size
    d = d_state
    nph = H // ps
    npw = W // ps

    START_TOKEN = d ** (ps * ps)
    END_TOKEN = START_TOKEN + 1

    all_tokens: list[int] = []

    traj_np = trajectory.cpu().numpy()

    for b in range(B):
        for t in range(T):
            # Serialize each channel as a separate frame
            for g in range(G):
                all_tokens.append(START_TOKEN)

                grid = traj_np[b, t, g]
                for ph in range(nph):
                    for pw in range(npw):
                        patch = grid[
                            ph * ps : (ph + 1) * ps,
                            pw * ps : (pw + 1) * ps,
                        ].flatten()
                        token = 0
                        for i, val in enumerate(patch):
                            token += int(val) * (d**i)
                        all_tokens.append(token)

                all_tokens.append(END_TOKEN)

    return np.array(all_tokens, dtype=np.uint16)


def compute_gzip_complexity(tokens: np.ndarray) -> float:
    """
    Gzip compression ratio as complexity proxy.

    Low = simple/repetitive. High = random/noisy.
    Class IV edge-of-chaos targets 0.4-0.7.
    """
    raw = tokens.tobytes()
    compressed = gzip.compress(raw)
    return len(compressed) / max(len(raw), 1)


# =============================================================================
# Rule filtering
# =============================================================================


def evaluate_rule_complexity(
    rule: NCARule,
    config: NCAConfig,
    rule_params: dict,
    device: torch.device,
    eval_sims: int = 4,
) -> float:
    """Simulate a short trajectory and return gzip complexity ratio."""
    traj = simulate_trajectory(
        rule=rule,
        grid_size=config.grid_size,
        d_state=config.d_state,
        n_groups=config.n_groups,
        num_steps=config.filter_steps,
        burn_in=config.burn_in,
        identity_bias=rule_params["identity_bias"],
        temperature=rule_params["temperature"],
        batch_size=eval_sims,
        device=device,
    )
    tokens = tokenize_trajectory(traj, config.d_state, config.patch_size)
    return compute_gzip_complexity(tokens)


def generate_and_filter_rules(
    config: NCAConfig,
    num_rules: int,
    device: torch.device,
    eval_sims: int = 4,
) -> list[tuple[NCARule, dict]]:
    """
    Generate random NCA rules with mixed architectures, filter by complexity.

    Returns list of (rule, params_dict) tuples.
    """
    accepted: list[tuple[NCARule, dict]] = []
    candidates_tested = 0
    t0 = time.time()

    while len(accepted) < num_rules:
        # Sample rule architecture
        params = sample_rule_config(config)
        rule = NCARule(
            d_state=config.d_state,
            n_groups=config.n_groups,
            kernel_size=params["kernel_size"],
            hidden_dim=params["hidden_dim"],
            num_hidden_layers=params["num_hidden_layers"],
        ).to(device)
        candidates_tested += 1

        if not config.filter_enabled:
            accepted.append((rule, params))
            continue

        ratio = evaluate_rule_complexity(rule, config, params, device, eval_sims)

        if config.gzip_lower <= ratio <= config.gzip_upper:
            accepted.append((rule, params))
            if len(accepted) % 100 == 0:
                elapsed = time.time() - t0
                rate = len(accepted) / candidates_tested
                logger.info(
                    "  %d/%d rules accepted (%.0f%% rate, %.1fs, last gzip=%.3f)",
                    len(accepted), num_rules, rate * 100, elapsed, ratio,
                )

    elapsed = time.time() - t0
    rate = num_rules / max(candidates_tested, 1)
    logger.info(
        "Rule generation: %d accepted from %d candidates (%.0f%%, %.1fs)",
        num_rules, candidates_tested, rate * 100, elapsed,
    )

    # Log architecture distribution
    if config.mixed_complexity:
        kernels = [p["kernel_size"] for _, p in accepted]
        depths = [p["num_hidden_layers"] for _, p in accepted]
        biases = [p["identity_bias"] for _, p in accepted]
        temps = [p["temperature"] for _, p in accepted]
        logger.info(
            "  kernel: %d×3x3, %d×5x5 | depth: %s | "
            "bias: %.2f-%.2f | temp: %.2f-%.2f",
            kernels.count(3), kernels.count(5),
            {d: depths.count(d) for d in sorted(set(depths))},
            min(biases), max(biases), min(temps), max(temps),
        )

    return accepted


# =============================================================================
# Full dataset generation
# =============================================================================


def generate_nca_dataset(
    config: NCAConfig,
    output_path: str | Path,
    max_tokens: int = 300_000_000,
    num_rules: int = 5000,
    sims_per_rule: int = 8,
    device: torch.device = torch.device("cpu"),
) -> int:
    """
    Generate a complete NCA trajectory dataset.

    Pipeline:
    1. Generate and filter rules (mixed architectures)
    2. For each rule, simulate multiple trajectories
    3. Tokenize (multi-channel → serialized frames) and write binary

    The output format is identical to language data (flat uint16),
    so it works directly with TokenizedDataset.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("NCA Dataset Generation")
    logger.info("=" * 60)
    logger.info(
        "Grid: %d×%d, %d states, %d channels",
        config.grid_size, config.grid_size, config.d_state, config.n_groups,
    )
    logger.info(
        "Mixed complexity: %s, filter: [%.2f, %.2f]",
        config.mixed_complexity, config.gzip_lower, config.gzip_upper,
    )
    logger.info(
        "Tokens/timestep: %d (%d channels × %d tok/frame)",
        config.tokens_per_timestep, config.n_groups, config.tokens_per_frame,
    )
    logger.info(
        "Tokens/trajectory: %d (%d steps)",
        config.tokens_per_trajectory, config.num_steps,
    )
    logger.info(
        "Target: %d rules × %d sims → %.1fM tokens max",
        num_rules, sims_per_rule, max_tokens / 1e6,
    )

    # Step 1: Generate filtered rules
    rules = generate_and_filter_rules(config, num_rules, device)

    # Step 2: Simulate and tokenize
    total_tokens = 0
    rules_used = 0
    t0 = time.time()

    with open(output_path, "wb") as f:
        for rule, params in rules:
            traj = simulate_trajectory(
                rule=rule,
                grid_size=config.grid_size,
                d_state=config.d_state,
                n_groups=config.n_groups,
                num_steps=config.num_steps,
                burn_in=config.burn_in,
                identity_bias=params["identity_bias"],
                temperature=params["temperature"],
                batch_size=sims_per_rule,
                device=device,
            )
            tokens = tokenize_trajectory(traj, config.d_state, config.patch_size)

            f.write(tokens.tobytes())
            total_tokens += len(tokens)
            rules_used += 1

            if rules_used % 100 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  %d/%d rules, %.1fM tokens, %.0f tok/s",
                    rules_used, len(rules),
                    total_tokens / 1e6,
                    total_tokens / max(elapsed, 1e-6),
                )

            if total_tokens >= max_tokens:
                break

    # Trim to exact max_tokens
    if total_tokens > max_tokens:
        with open(output_path, "r+b") as f:
            f.truncate(max_tokens * 2)
        total_tokens = max_tokens

    elapsed = time.time() - t0
    file_size_mb = output_path.stat().st_size / 1e6
    logger.info("=" * 60)
    logger.info(
        "Done: %d rules, %.1fM tokens, %.1f MB, %.1fs (%.0f tok/s)",
        rules_used, total_tokens / 1e6, file_size_mb,
        elapsed, total_tokens / max(elapsed, 1e-6),
    )

    return total_tokens


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Generate NCA trajectory dataset")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--tokens", type=int, default=300_000_000)
    p.add_argument("--num_rules", type=int, default=5000)
    p.add_argument("--sims_per_rule", type=int, default=8)

    # Grid
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--d_state", type=int, default=10)
    p.add_argument("--n_groups", type=int, default=4)

    # Rule defaults (overridden when mixed_complexity=True)
    p.add_argument("--kernel_size", type=int, default=3, choices=[3, 5])
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--num_hidden_layers", type=int, default=2)
    p.add_argument("--identity_bias", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.5)

    # Filtering
    p.add_argument("--gzip_lower", type=float, default=0.4)
    p.add_argument("--gzip_upper", type=float, default=0.7)
    p.add_argument("--no_filter", action="store_true")
    p.add_argument("--no_mixed", action="store_true",
                    help="Disable mixed complexity (use fixed architecture)")

    # Trajectory
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--burn_in", type=int, default=10)

    # Runtime
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)

    # Verify mode
    p.add_argument("--verify", type=str, default=None)

    args = p.parse_args()

    if args.verify:
        data = np.memmap(args.verify, dtype=np.uint16, mode="r")
        print(f"File: {args.verify}")
        print(f"Tokens: {len(data):,} ({len(data)/1e6:.1f}M)")
        print(f"Size: {Path(args.verify).stat().st_size / 1e6:.1f} MB")
        print(f"Vocab range: [{data.min()}, {data.max()}]")
        print(f"Unique tokens: {len(np.unique(data))}")
        print(f"First 50: {data[:50].tolist()}")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = NCAConfig(
        grid_size=args.grid_size,
        d_state=args.d_state,
        n_groups=args.n_groups,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_steps=args.num_steps,
        burn_in=args.burn_in,
        identity_bias=args.identity_bias,
        temperature=args.temperature,
        filter_enabled=not args.no_filter,
        gzip_lower=args.gzip_lower,
        gzip_upper=args.gzip_upper,
        mixed_complexity=not args.no_mixed,
    )

    generate_nca_dataset(
        config=config,
        output_path=args.output,
        max_tokens=args.tokens,
        num_rules=args.num_rules,
        sims_per_rule=args.sims_per_rule,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
