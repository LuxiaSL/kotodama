#!/usr/bin/env python3
"""Unified activation geometry extraction (Track 5a).

Extracts geometric objects from model checkpoints for visualization and
analysis: point clouds, eigenspectra, trajectories, attention patterns,
head entropy, effective rank, and Procrustes alignment.

Replaces: extract_shapes.py, extract_shapes_v2.py, extract_shapes_lang_full.py.

Usage::

    # All components for all registry checkpoints
    python -m scripts.extract_activation_geometry --device cuda:0

    # Specific components
    python -m scripts.extract_activation_geometry --device cuda:0 \\
        --components point_clouds,eigenspectra,trajectories

    # CPU-only components (no GPU needed)
    python -m scripts.extract_activation_geometry --components eigenspectra,effective_rank,procrustes

    # Specific checkpoints
    python -m scripts.extract_activation_geometry --device cuda:0 \\
        --checkpoints P3-Muon-002,NCA-002

    # Single checkpoint by path
    python -m scripts.extract_activation_geometry --device cuda:0 \\
        --checkpoint path/to/ckpt.pt --name my-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.forward import ForwardResult, compute_attention_weights, forward_with_states
from src.eval.model_loader import load_checkpoint_registry, load_model

logger = logging.getLogger(__name__)

# Component classification
GPU_COMPONENTS = {
    "point_clouds", "topo_clouds", "trajectories",
    "attention_weights", "attention_outputs", "head_entropy",
}
CPU_COMPONENTS = {"eigenspectra", "effective_rank", "procrustes"}
ALL_COMPONENTS = GPU_COMPONENTS | CPU_COMPONENTS

# Default extraction parameters
DEFAULTS = {
    "n_seqs_clouds": 500,
    "n_seqs_topo": 50,
    "n_seqs_attn_outputs": 500,
    "n_seqs_head_entropy": 200,
    "seq_len": 512,
    "batch_size": 16,
    "topo_subsample": 2000,
    "topo_layers": [0, 14, 27],
    "attn_layers": [0, 7, 14, 27],
    "procrustes_components": 50,
}

WEIGHT_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# =============================================================================
# Extraction functions
# =============================================================================


@torch.no_grad()
def extract_point_clouds(
    model: Any, eval_tokens: np.ndarray,
    n_seqs: int, seq_len: int, batch_size: int, device: str,
) -> dict[int, np.ndarray]:
    """Mean-pooled hidden states per layer. Returns {layer_idx: (n_seqs, hidden)}."""
    n_layers = model.config.num_layers
    clouds: dict[int, list[np.ndarray]] = {i: [] for i in range(-1, n_layers)}

    for batch_start in range(0, n_seqs, batch_size):
        actual_bs = min(batch_size, n_seqs - batch_start)
        input_ids = _load_batch(eval_tokens, batch_start, actual_bs, seq_len, device)
        result = forward_with_states(model, input_ids)

        clouds[-1].append(result.states[0].mean(dim=1).cpu().numpy())
        for i in range(n_layers):
            clouds[i].append(result.states[i + 1].mean(dim=1).cpu().numpy())

        if (batch_start // batch_size) % 10 == 0:
            logger.info("  Point clouds: %d/%d", batch_start + actual_bs, n_seqs)

    return {k: np.concatenate(v, axis=0) for k, v in clouds.items()}


@torch.no_grad()
def extract_topo_clouds(
    model: Any, eval_tokens: np.ndarray,
    n_seqs: int, seq_len: int, batch_size: int,
    layers: list[int], subsample: int, device: str,
) -> dict[int, np.ndarray]:
    """Token-level hidden states for persistent homology. Returns {layer_idx: (subsample, hidden)}."""
    all_tokens: dict[int, list[np.ndarray]] = {i: [] for i in layers}

    for batch_start in range(0, n_seqs, batch_size):
        actual_bs = min(batch_size, n_seqs - batch_start)
        input_ids = _load_batch(eval_tokens, batch_start, actual_bs, seq_len, device)
        result = forward_with_states(model, input_ids)

        for i in layers:
            state = result.states[i + 1]
            flat = state.reshape(-1, state.shape[-1]).cpu().numpy()
            all_tokens[i].append(flat)

    rng = np.random.RandomState(42)
    out = {}
    for layer_idx, arrays in all_tokens.items():
        full = np.concatenate(arrays, axis=0)
        indices = rng.choice(len(full), size=min(subsample, len(full)), replace=False)
        out[layer_idx] = full[indices]
    return out


@torch.no_grad()
def extract_trajectories(
    model: Any, prompt_ids_list: list[torch.Tensor], device: str,
) -> dict[int, np.ndarray]:
    """Last-token hidden state through all layers. Returns {prompt_idx: (n_layers+1, hidden)}."""
    out = {}
    for j, prompt_ids in enumerate(prompt_ids_list):
        result = forward_with_states(model, prompt_ids.to(device))
        trajectory = [s[:, -1, :].cpu().numpy().squeeze(0) for s in result.states]
        out[j] = np.stack(trajectory, axis=0)
    return out


def extract_eigenspectra(model: Any) -> dict[str, np.ndarray]:
    """SVD of all weight matrices. CPU-only. Returns {f"{wtype}_layer_{i}": sv_array}."""
    out = {}
    for i, layer in enumerate(model.layers):
        weight_map = {
            "q_proj": layer.attn.q_proj.weight,
            "k_proj": layer.attn.k_proj.weight,
            "v_proj": layer.attn.v_proj.weight,
            "o_proj": layer.attn.o_proj.weight,
            "gate_proj": layer.ffn.gate_proj.weight,
            "up_proj": layer.ffn.up_proj.weight,
            "down_proj": layer.ffn.down_proj.weight,
        }
        for wtype, W in weight_map.items():
            sv = torch.linalg.svdvals(W.detach().float().cpu()).numpy()
            out[f"{wtype}_layer_{i}"] = sv
    return out


@torch.no_grad()
def extract_attention_weights_all(
    model: Any, prompt_ids_list: list[torch.Tensor],
    prompt_indices: list[int], layers: list[int], device: str,
) -> dict[str, np.ndarray]:
    """Per-head attention weight matrices. Returns {f"prompt_{idx}_layer_{i}": (n_heads, seq, seq)}."""
    out: dict[str, np.ndarray] = {}
    layers_set = set(layers)

    for prompt_ids, orig_idx in zip(prompt_ids_list, prompt_indices):
        result = forward_with_states(
            model, prompt_ids.to(device),
            capture_attention=True,
            attention_layers=list(layers_set),
        )
        for layer_idx, weights in result.attention_weights.items():
            out[f"prompt_{orig_idx}_layer_{layer_idx}"] = weights

    return out


@torch.no_grad()
def extract_attention_outputs(
    model: Any, eval_tokens: np.ndarray,
    n_seqs: int, seq_len: int, batch_size: int, device: str,
) -> dict[int, np.ndarray]:
    """Attention-only hidden states (before residual), mean-pooled. Returns {layer: (n_seqs, hidden)}."""
    from src.model.llama import LuxiaBaseModel
    n_layers = model.config.num_layers
    use_attn_res = model.config.attn_res
    attn_outs: dict[int, list[np.ndarray]] = {i: [] for i in range(n_layers)}

    for batch_start in range(0, n_seqs, batch_size):
        actual_bs = min(batch_size, n_seqs - batch_start)
        input_ids = _load_batch(eval_tokens, batch_start, actual_bs, seq_len, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.embed_tokens(input_ids)
            rope_cos = model.rope_cos
            rope_sin = model.rope_sin

            if use_attn_res:
                committed: list[torch.Tensor] = []
                partial = x
                boundary_set = model._attn_res_boundary_set

                for i, layer in enumerate(model.layers):
                    sources = committed + [partial]
                    h = model._block_attn_res_from_list(
                        sources, layer.attn_res_query, layer.attn_res_norm
                    )
                    if i in boundary_set:
                        committed.append(partial.clone())

                    attn_out = layer.attn(layer.attn_norm(h), rope_cos, rope_sin)
                    attn_outs[i].append(attn_out.float().mean(dim=1).cpu().numpy())
                    partial = partial + attn_out

                    sources = committed + [partial]
                    h = model._block_attn_res_from_list(
                        sources, layer.mlp_res_query, layer.mlp_res_norm
                    )
                    mlp_out = layer.ffn(layer.ffn_norm(h))
                    partial = partial + mlp_out
            else:
                for i, layer in enumerate(model.layers):
                    attn_out = layer.attn(layer.attn_norm(x), rope_cos, rope_sin)
                    attn_outs[i].append(attn_out.float().mean(dim=1).cpu().numpy())
                    x = x + attn_out
                    x = x + layer.ffn(layer.ffn_norm(x))

        if (batch_start // batch_size) % 10 == 0:
            logger.info("  Attention outputs: %d/%d", batch_start + actual_bs, n_seqs)

    return {k: np.concatenate(v, axis=0) for k, v in attn_outs.items()}


@torch.no_grad()
def extract_head_entropy(
    model: Any, eval_tokens: np.ndarray,
    n_seqs: int, seq_len: int, batch_size: int, device: str,
) -> dict[int, np.ndarray]:
    """Per-head attention entropy averaged over sequences. Returns {layer: (n_heads,)}."""
    from src.model.llama import apply_rope
    import torch.nn.functional as F

    n_layers = model.config.num_layers
    n_heads = model.config.num_attention_heads
    use_attn_res = model.config.attn_res
    eps = 1e-10

    entropy_sums: dict[int, np.ndarray] = {
        i: np.zeros(n_heads, dtype=np.float64) for i in range(n_layers)
    }
    total_positions = 0

    for batch_start in range(0, n_seqs, batch_size):
        actual_bs = min(batch_size, n_seqs - batch_start)
        input_ids = _load_batch(eval_tokens, batch_start, actual_bs, seq_len, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.embed_tokens(input_ids)
            rope_cos = model.rope_cos
            rope_sin = model.rope_sin

            if use_attn_res:
                committed: list[torch.Tensor] = []
                partial = x
                boundary_set = model._attn_res_boundary_set

                for i, layer in enumerate(model.layers):
                    sources = committed + [partial]
                    h = model._block_attn_res_from_list(
                        sources, layer.attn_res_query, layer.attn_res_norm
                    )
                    if i in boundary_set:
                        committed.append(partial.clone())

                    x_normed = layer.attn_norm(h)
                    entropy_sums[i] += _head_entropy_batch(
                        layer, x_normed, rope_cos, rope_sin, actual_bs, seq_len, eps
                    )

                    attn_out = layer.attn(x_normed, rope_cos, rope_sin)
                    partial = partial + attn_out
                    sources = committed + [partial]
                    h = model._block_attn_res_from_list(
                        sources, layer.mlp_res_query, layer.mlp_res_norm
                    )
                    partial = partial + layer.ffn(layer.ffn_norm(h))
            else:
                for i, layer in enumerate(model.layers):
                    x_normed = layer.attn_norm(x)
                    entropy_sums[i] += _head_entropy_batch(
                        layer, x_normed, rope_cos, rope_sin, actual_bs, seq_len, eps
                    )
                    x = layer(x, rope_cos, rope_sin)

        total_positions += actual_bs * seq_len
        if (batch_start // batch_size) % 10 == 0:
            logger.info("  Head entropy: %d/%d", batch_start + actual_bs, n_seqs)

    return {k: v / total_positions for k, v in entropy_sums.items()}


def _head_entropy_batch(
    layer: Any, x_normed: torch.Tensor,
    rope_cos: torch.Tensor, rope_sin: torch.Tensor,
    bsz: int, seq_len: int, eps: float,
) -> np.ndarray:
    """Compute per-head entropy sum for one batch. Returns (n_heads,)."""
    from src.model.llama import apply_rope
    import torch.nn.functional as F

    attn = layer.attn
    q = attn.q_proj(x_normed).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x_normed).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    if attn.qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)

    q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
    k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

    if attn.num_kv_groups > 1:
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (attn.head_dim ** -0.5)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(causal, float("-inf"))
    weights = F.softmax(scores, dim=-1)

    log_w = torch.log(weights + eps)
    per_pos_entropy = -(weights * log_w).sum(dim=-1)  # (bsz, n_heads, seq_len)
    return per_pos_entropy.float().sum(dim=(0, 2)).cpu().numpy()


def compute_effective_rank(eigenspectra_path: Path) -> dict[str, np.ndarray]:
    """Participation ratio from saved eigenspectra. CPU-only."""
    data = np.load(eigenspectra_path, allow_pickle=True)
    run_names = list(data["run_names"])
    weight_types = list(data["weight_types"])
    n_layers = int(data["n_layers"][0])
    result: dict[str, np.ndarray] = {}

    for run_name in run_names:
        for wtype in weight_types:
            pr = np.zeros(n_layers, dtype=np.float64)
            for layer_idx in range(n_layers):
                key = f"{run_name}_{wtype}_layer_{layer_idx}"
                try:
                    sigma = data[key].astype(np.float64)
                    s_sum = sigma.sum()
                    s_sq_sum = (sigma ** 2).sum()
                    pr[layer_idx] = (s_sum ** 2) / s_sq_sum if s_sq_sum > 0 else 0.0
                except KeyError:
                    pr[layer_idx] = np.nan
            result[f"{run_name}_{wtype}"] = pr
    return result


def compute_procrustes(
    point_clouds_path: Path, n_components: int = 50,
) -> dict[str, np.ndarray]:
    """PCA-aligned point clouds for cross-run comparison. CPU-only."""
    data = np.load(point_clouds_path, allow_pickle=True)
    run_names = list(data["run_names"])
    n_layers = int(data["n_layers"][0])
    result: dict[str, np.ndarray] = {}

    for layer_idx in range(-1, n_layers):
        clouds = []
        for run_name in run_names:
            key = f"{run_name}_layer_{layer_idx}"
            try:
                clouds.append((run_name, data[key].astype(np.float64)))
            except KeyError:
                break
        else:
            # Mean-center and scale-normalize
            centered = []
            for run_name, cloud in clouds:
                c = cloud - cloud.mean(axis=0, keepdims=True)
                fnorm = np.linalg.norm(c, ord="fro")
                if fnorm > 0:
                    c = c / fnorm
                centered.append((run_name, c))

            # Joint PCA
            all_c = np.concatenate([c for _, c in centered], axis=0)
            cov = np.cov(all_c, rowvar=False)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                sort_idx = np.argsort(eigvals)[::-1]
                top = eigvecs[:, sort_idx[:n_components]]
                for run_name, c in centered:
                    result[f"{run_name}_layer_{layer_idx}"] = (c @ top).astype(np.float32)
            except np.linalg.LinAlgError:
                logger.warning("PCA failed for layer %d", layer_idx)

    return result


# =============================================================================
# Helpers
# =============================================================================


def _load_batch(
    eval_tokens: np.ndarray, batch_start: int, batch_size: int,
    seq_len: int, device: str,
) -> torch.Tensor:
    """Load a batch of eval sequences into a tensor."""
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for j in range(batch_size):
        offset = (batch_start + j) * seq_len
        chunk = eval_tokens[offset: offset + seq_len].astype(np.int64)
        input_ids[j] = torch.from_numpy(chunk)
    return input_ids


# =============================================================================
# Save functions
# =============================================================================


def _save_npz(
    output_dir: Path, filename: str,
    per_run_data: dict[str, dict],
    extra_meta: dict[str, np.ndarray],
    key_fn,
) -> None:
    """Save per-run data to a compressed NPZ file."""
    out: dict[str, np.ndarray] = {}
    for run_name, data in per_run_data.items():
        for k, arr in data.items():
            out[key_fn(run_name, k)] = arr
    out["run_names"] = np.array(list(per_run_data.keys()))
    out.update(extra_meta)
    np.savez_compressed(output_dir / filename, **out)
    logger.info("  Saved %s", filename)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Unified activation geometry extraction (Track 5a)",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("-o", "--output-dir", type=Path, default=None)
    p.add_argument("--components", type=str, default="all",
                   help=f"Comma-separated components. Options: {sorted(ALL_COMPONENTS)}")

    # Checkpoint selection
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--checkpoints", type=str, default=None,
                      help="Comma-separated checkpoint names from registry")
    ckpt.add_argument("--checkpoint", type=Path, default=None,
                      help="Single checkpoint path")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--attn-res", type=str, default=None)

    # Model config
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--config-section", type=str, default="proxy")
    p.add_argument("--eval-data", type=Path, default=Path("data/fineweb_edu_eval_5m.bin"))

    args = p.parse_args()

    # Parse components
    if args.components == "all":
        components = ALL_COMPONENTS
    else:
        components = set(args.components.split(","))
        unknown = components - ALL_COMPONENTS
        if unknown:
            logger.error("Unknown components: %s. Valid: %s", unknown, sorted(ALL_COMPONENTS))
            sys.exit(1)

    want_gpu = bool(components & GPU_COMPONENTS)
    want_cpu = bool(components & CPU_COMPONENTS)

    # Resolve checkpoints
    runs: list[tuple[str, Path, dict | None]] = []
    if args.checkpoint:
        name = args.name or args.checkpoint.stem
        ar = None
        if args.attn_res:
            ar = {"attn_res": True}
            for part in args.attn_res.split():
                k, v = part.split("=", 1)
                if k == "n_blocks":
                    ar["attn_res_n_blocks"] = int(v)
                elif k == "boundaries":
                    ar["attn_res_boundaries"] = [int(x) for x in v.split(",")]
        runs.append((name, args.checkpoint, ar))
    else:
        registry = load_checkpoint_registry()
        if args.checkpoints:
            names = [n.strip() for n in args.checkpoints.split(",")]
            for n in names:
                if n in registry:
                    runs.append((registry[n].name, registry[n].path, registry[n].attn_res_config))
                else:
                    logger.warning("Checkpoint '%s' not in registry", n)
        else:
            for info in registry.values():
                runs.append((info.name, info.path, info.attn_res_config))

    # Filter to existing checkpoints
    runs = [(n, p, a) for n, p, a in runs if p.exists()]
    if not runs:
        logger.error("No valid checkpoints found")
        sys.exit(1)

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        if len(runs) == 1:
            output_dir = Path("analysis") / runs[0][0] / "activation_geometry"
        else:
            output_dir = Path("analysis") / "activation_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = yaml.safe_load(open(args.config))
    model_cfg = cfg[args.config_section]
    n_layers = model_cfg["num_layers"]

    t0_total = time.time()

    # ── CPU derivations ──────────────────────────────────────────────────
    if want_cpu:
        if "effective_rank" in components:
            es_path = output_dir / "eigenspectra.npz"
            if es_path.exists():
                t1 = time.time()
                er_data = compute_effective_rank(es_path)
                er_out = dict(er_data)
                run_set = set()
                for key in er_data:
                    for wt in WEIGHT_TYPES:
                        if key.endswith(f"_{wt}"):
                            run_set.add(key[:-len(f"_{wt}")])
                er_out["run_names"] = np.array(sorted(run_set))
                er_out["weight_types"] = np.array(WEIGHT_TYPES)
                er_out["n_layers"] = np.array([n_layers])
                np.savez_compressed(output_dir / "effective_rank.npz", **er_out)
                logger.info("  Saved effective_rank.npz (%.1fs)", time.time() - t1)
            else:
                logger.warning("  Skipping effective_rank: eigenspectra.npz not found")

        if "procrustes" in components:
            pc_path = output_dir / "point_clouds.npz"
            if pc_path.exists():
                t1 = time.time()
                pr_data = compute_procrustes(pc_path, DEFAULTS["procrustes_components"])
                pr_out = dict(pr_data)
                try:
                    pc = np.load(pc_path, allow_pickle=True)
                    pr_out["run_names"] = pc["run_names"]
                    pr_out["n_layers"] = pc["n_layers"]
                except Exception:
                    pr_out["run_names"] = np.array([n for n, _, _ in runs])
                    pr_out["n_layers"] = np.array([n_layers])
                np.savez_compressed(output_dir / "procrustes.npz", **pr_out)
                logger.info("  Saved procrustes.npz (%.1fs)", time.time() - t1)
            else:
                logger.warning("  Skipping procrustes: point_clouds.npz not found")

    if not want_gpu:
        logger.info("Done (CPU-only). Total: %.1fs", time.time() - t0_total)
        return

    # ── GPU extractions ──────────────────────────────────────────────────

    eval_tokens = np.memmap(args.eval_data, dtype=np.uint16, mode="r")

    # Tokenize trajectory prompts
    prompts_config = yaml.safe_load(open("configs/prompts.yaml"))
    traj_prompts = prompts_config["trajectory"]
    attn_indices = prompts_config.get("trajectory_attention_indices", [0, 2, 4])

    tokenizer = None
    prompt_ids_list: list[torch.Tensor] = []
    prompt_token_strs: dict[int, list[str]] = {}

    need_tokenizer = components & {"trajectories", "attention_weights"}
    if need_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        for j, text in enumerate(traj_prompts):
            ids = tokenizer.encode(text, return_tensors="pt")
            prompt_ids_list.append(ids)
            prompt_token_strs[j] = tokenizer.convert_ids_to_tokens(ids[0].tolist())

    # Storage
    all_data: dict[str, dict[str, dict]] = {comp: {} for comp in components if comp in GPU_COMPONENTS}

    for run_name, ckpt_path, ar_config in runs:
        logger.info("=" * 60)
        logger.info("Extracting: %s", run_name)
        t0 = time.time()

        model = load_model(
            ckpt_path, config_path=args.config, config_section=args.config_section,
            attn_res_config=ar_config, device=args.device,
        )

        if "point_clouds" in components:
            t1 = time.time()
            clouds = extract_point_clouds(
                model, eval_tokens, DEFAULTS["n_seqs_clouds"],
                DEFAULTS["seq_len"], DEFAULTS["batch_size"], args.device,
            )
            all_data["point_clouds"][run_name] = clouds
            logger.info("  Point clouds: %.1fs", time.time() - t1)

        if "topo_clouds" in components:
            t1 = time.time()
            topo = extract_topo_clouds(
                model, eval_tokens, DEFAULTS["n_seqs_topo"],
                DEFAULTS["seq_len"], DEFAULTS["batch_size"],
                DEFAULTS["topo_layers"], DEFAULTS["topo_subsample"], args.device,
            )
            all_data["topo_clouds"][run_name] = topo
            logger.info("  Topo clouds: %.1fs", time.time() - t1)

        if "trajectories" in components:
            t1 = time.time()
            trajs = extract_trajectories(model, prompt_ids_list, args.device)
            all_data["trajectories"][run_name] = trajs
            logger.info("  Trajectories: %.1fs", time.time() - t1)

        if "eigenspectra" in components:
            t1 = time.time()
            spectra = extract_eigenspectra(model)
            all_data.setdefault("eigenspectra", {})[run_name] = spectra
            logger.info("  Eigenspectra: %.1fs", time.time() - t1)

        if "attention_weights" in components:
            t1 = time.time()
            attn_prompt_ids = [prompt_ids_list[i] for i in attn_indices]
            attn = extract_attention_weights_all(
                model, attn_prompt_ids, attn_indices,
                DEFAULTS["attn_layers"], args.device,
            )
            all_data["attention_weights"][run_name] = attn
            logger.info("  Attention weights: %.1fs", time.time() - t1)

        if "attention_outputs" in components:
            t1 = time.time()
            ao = extract_attention_outputs(
                model, eval_tokens, DEFAULTS["n_seqs_attn_outputs"],
                DEFAULTS["seq_len"], DEFAULTS["batch_size"], args.device,
            )
            all_data["attention_outputs"][run_name] = ao
            logger.info("  Attention outputs: %.1fs", time.time() - t1)

        if "head_entropy" in components:
            t1 = time.time()
            he = extract_head_entropy(
                model, eval_tokens, DEFAULTS["n_seqs_head_entropy"],
                DEFAULTS["seq_len"], DEFAULTS["batch_size"], args.device,
            )
            all_data["head_entropy"][run_name] = he
            logger.info("  Head entropy: %.1fs", time.time() - t1)

        del model
        torch.cuda.empty_cache()
        logger.info("  Total for %s: %.1fs", run_name, time.time() - t0)

    # ── Save ─────────────────────────────────────────────────────────────
    logger.info("Saving to %s", output_dir)
    n_layers_arr = np.array([n_layers])

    if "point_clouds" in all_data and all_data["point_clouds"]:
        _save_npz(output_dir, "point_clouds.npz", all_data["point_clouds"],
                  {"n_layers": n_layers_arr},
                  lambda run, k: f"{run}_layer_{k}")

    if "topo_clouds" in all_data and all_data["topo_clouds"]:
        _save_npz(output_dir, "topo_clouds.npz", all_data["topo_clouds"],
                  {"n_layers": n_layers_arr, "topo_layers": np.array(DEFAULTS["topo_layers"])},
                  lambda run, k: f"{run}_layer_{k}")

    if "trajectories" in all_data and all_data["trajectories"]:
        tr_out: dict[str, np.ndarray] = {}
        for run, trajs in all_data["trajectories"].items():
            for pidx, arr in trajs.items():
                tr_out[f"{run}_prompt_{pidx}"] = arr
        tr_out["run_names"] = np.array(list(all_data["trajectories"].keys()))
        tr_out["n_prompts"] = np.array([len(traj_prompts)])
        tr_out["prompt_texts"] = np.array(traj_prompts)
        np.savez_compressed(output_dir / "trajectories.npz", **tr_out)
        logger.info("  Saved trajectories.npz")

    if "eigenspectra" in all_data and all_data["eigenspectra"]:
        _save_npz(output_dir, "eigenspectra.npz", all_data["eigenspectra"],
                  {"n_layers": n_layers_arr, "weight_types": np.array(WEIGHT_TYPES)},
                  lambda run, k: f"{run}_{k}")

    if "attention_weights" in all_data and all_data["attention_weights"]:
        at_out: dict[str, np.ndarray] = {}
        for run, attn in all_data["attention_weights"].items():
            for key, arr in attn.items():
                at_out[f"{run}_{key}"] = arr
        at_out["run_names"] = np.array(list(all_data["attention_weights"].keys()))
        at_out["attn_layers"] = np.array(DEFAULTS["attn_layers"])
        at_out["attn_prompt_indices"] = np.array(attn_indices)
        for idx in attn_indices:
            at_out[f"tokens_prompt_{idx}"] = np.array(prompt_token_strs[idx])
        np.savez_compressed(output_dir / "attention_weights.npz", **at_out)
        logger.info("  Saved attention_weights.npz")

    if "attention_outputs" in all_data and all_data["attention_outputs"]:
        _save_npz(output_dir, "attention_outputs.npz", all_data["attention_outputs"],
                  {"n_layers": n_layers_arr},
                  lambda run, k: f"{run}_layer_{k}")

    if "head_entropy" in all_data and all_data["head_entropy"]:
        _save_npz(output_dir, "head_entropy.npz", all_data["head_entropy"],
                  {"n_layers": n_layers_arr, "n_heads": np.array([model_cfg["num_attention_heads"]])},
                  lambda run, k: f"{run}_layer_{k}")

    logger.info("Done. Total: %.1fs (%.1f min)", time.time() - t0_total, (time.time() - t0_total) / 60)


if __name__ == "__main__":
    main()
