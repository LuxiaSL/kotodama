"""
Distributed training script for luxia-base.

Launch with torchrun::

    torchrun --nproc_per_node=8 -m src.training.train \\
        --data_path data/fineweb_sample.bin \\
        --checkpoint_dir checkpoints/proxy_p1 \\
        --model_size proxy \\
        --total_tokens 6_000_000_000

For testing without data::

    torchrun --nproc_per_node=1 -m src.training.train \\
        --random_data --model_size smoke --total_steps 100

Uses DDP for gradient synchronization.  Muon's Newton-Schulz
orthogonalization operates on the full (replicated) weight matrices
after DDP all-reduces the gradients.  This is correct and simple — no
FSDP sharding complications.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Suppress Inductor's verbose "Online softmax is disabled" warnings — fires per
# subgraph and floods logs.  Must be AFTER `import torch` (torch resets filters).
warnings.filterwarnings("ignore", message="Online softmax", module=r"torch\._inductor")

# Project imports — launch from the luxia-base/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.dataset import RandomTokenDataset, TokenizedDataset
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from src.monitoring.geometric import GeometricMonitor, MonitorConfig
from src.monitoring.wandb_callback import WandbLogger
from src.training.checkpoint import (
    AsyncCheckpointManager,
    CheckpointManager,
    SIGTERMHandler,
)
from src.training.muon import HybridScheduler, build_hybrid_optimizer

logger = logging.getLogger(__name__)


# ── Heimdall metrics (lightweight, no dependency) ────────────────────────────

_hm_client = None
_hm_job_id = os.environ.get("HEIMDALL_JOB_ID")


def hm_log(step: int, **metrics: float) -> None:
    """Log metrics to Heimdall. No-op if not running under Heimdall."""
    global _hm_client
    if not _hm_job_id:
        return
    try:
        if _hm_client is None:
            import httpx
            _hm_client = httpx.Client(base_url="http://127.0.0.1:7000", timeout=2)
        # Filter out None values
        clean = {k: v for k, v in metrics.items() if v is not None}
        if clean:
            _hm_client.post(
                f"/api/v1/jobs/{_hm_job_id}/metrics",
                json={"step": step, "metrics": clean},
            )
    except Exception:
        pass  # Never crash the training loop


# =============================================================================
# Model configs — matches configs/model.yaml
# =============================================================================

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "full": dict(
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=8192,
        vocab_size=49152,
        max_position_embeddings=4096,
    ),
    "3b": dict(
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=24,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=8192,
        vocab_size=49152,
        max_position_embeddings=4096,
    ),
    "8b": dict(
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        vocab_size=49152,
        max_position_embeddings=4096,
    ),
    "intermediate": dict(
        hidden_size=1024,
        num_layers=28,
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=2816,
        vocab_size=49152,
        max_position_embeddings=4096,
    ),
    "proxy": dict(
        hidden_size=512,
        num_layers=28,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=128,
        intermediate_size=1408,
        vocab_size=49152,
        max_position_embeddings=4096,
    ),
    "smoke": dict(
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,
        max_position_embeddings=2048,
    ),
}


# =============================================================================
# Distributed setup
# =============================================================================


def setup_distributed() -> tuple[int, int, int]:
    """Initialize NCCL process group.  Returns (rank, world_size, local_rank)."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Logging
# =============================================================================


def setup_logging(rank: int) -> None:
    """Configure logging — only rank 0 logs to stdout."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"[rank {rank}] %(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Training loop
# =============================================================================


def train(args: argparse.Namespace) -> None:
    # -- Distributed setup -----------------------------------------------------
    rank, world_size, local_rank = setup_distributed()
    setup_logging(rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    # -- Tensor parallelism groups ---------------------------------------------
    tp_size = getattr(args, "tp_size", 1)
    tp_group = None
    dp_group = None
    if tp_size > 1:
        from src.training.tensor_parallel import create_process_groups
        tp_group, dp_group = create_process_groups(world_size, tp_size)
    dp_size = world_size // tp_size

    if is_main:
        logger.info("World size: %d, device: %s", world_size, device)
        if tp_size > 1:
            logger.info("TP=%d, DP=%d", tp_size, dp_size)
        job_id = os.environ.get("HEIMDALL_JOB_ID", "local")
        logger.info("Heimdall job ID: %s", job_id)

    # -- Model -----------------------------------------------------------------
    if args.model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{args.model_size}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )

    model_kwargs = MODEL_CONFIGS[args.model_size].copy()
    model_kwargs["qk_norm"] = True
    model_kwargs["z_loss_weight"] = args.z_loss_weight
    model_kwargs["activation_checkpointing"] = args.activation_checkpointing
    model_kwargs["use_liger"] = getattr(args, "use_liger", False)
    model_kwargs["attn_impl"] = getattr(args, "attn_impl", "auto")
    model_kwargs["attn_res"] = args.attn_res
    model_kwargs["attn_res_n_blocks"] = args.attn_res_n_blocks
    if args.attn_res_boundaries:
        model_kwargs["attn_res_boundaries"] = [int(x) for x in args.attn_res_boundaries.split(",")]
    config = LuxiaModelConfig(**model_kwargs)

    model = LuxiaBaseModel(config).to(device=device, dtype=torch.bfloat16)
    param_count = sum(p.numel() for p in model.parameters())

    if is_main:
        logger.info(
            "Model: %s (%.1fM params, config estimate: %s)",
            args.model_size,
            param_count / 1e6,
            f"{config.param_count() / 1e6:.1f}M",
        )

    # -- NCA → language transition ---------------------------------------------
    if args.resume_nca:
        if is_main:
            logger.info("Loading NCA checkpoint: %s", args.resume_nca)
        nca_state = torch.load(args.resume_nca, map_location=device, weights_only=False)
        nca_model_state = nca_state.get("model", nca_state)

        # Strip _orig_mod. prefix from torch.compile'd checkpoints
        cleaned_state: dict[str, torch.Tensor] = {}
        for k, v in nca_model_state.items():
            clean_key = k.replace("_orig_mod.", "")
            cleaned_state[clean_key] = v
        # strict=False: AttnRes params (queries, norms) won't be in NCA checkpoint
        # and stay at their defaults (zero queries = uniform weighting, ones for norms)
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if is_main:
            if missing:
                logger.info("NCA load — missing keys (expected for AttnRes): %d keys", len(missing))
            if unexpected:
                logger.warning("NCA load — unexpected keys: %s", unexpected[:5])
            logger.info("NCA weights loaded")

        # Reinitialize embeddings to language vocab
        if is_main:
            logger.info(
                "Reinitializing embeddings: NCA vocab → language vocab (%d)",
                config.vocab_size,
            )
        model.reinit_embeddings(new_vocab_size=config.vocab_size)

        # Optionally reinitialize MLPs
        if args.reinit_mlps:
            if is_main:
                logger.info("Reinitializing MLPs (--reinit_mlps set)")
            model.reinit_mlps()
        else:
            if is_main:
                logger.info("Keeping NCA-trained MLPs (embed-only reinit)")

        if is_main:
            logger.info("NCA → language transition complete")

    # Enable TF32 for fp32 matmuls outside autocast (grad norm, loss compute)
    torch.set_float32_matmul_precision("high")

    # Tensor parallelism: shard projection weights across TP group
    if tp_group is not None:
        from src.training.tensor_parallel import apply_tensor_parallelism
        apply_tensor_parallelism(model, tp_group)
        if is_main:
            tp_param_count = sum(p.numel() for p in model.parameters())
            logger.info("After TP sharding: %.1fM params per rank", tp_param_count / 1e6)

    # FP8 training: convert Linear layers to Float8Linear (before compile)
    if getattr(args, "fp8", False):
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            fp8_config = Float8LinearConfig()
            convert_to_float8_training(model, config=fp8_config)
            if is_main:
                logger.info("FP8 training enabled via torchao (Float8Linear)")
        except ImportError:
            raise ImportError(
                "--fp8 requires torchao. Install with: uv pip install torchao"
            )

    # torch.compile for throughput (compile before DDP)
    if args.compile:
        if args.attn_impl == "fa4":
            raise ValueError(
                "--compile and --attn_impl fa4 are incompatible. "
                "FA4 CuTeDSL kernels have no custom_op registration and will break torch.compile. "
                "Use --attn_impl sdpa or --attn_impl fa2 with --compile, or drop --compile for FA4."
            )
        if is_main:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        if is_main:
            logger.info("Compilation registered (will compile on first forward)")

    # Wrap in DDP for gradient synchronization across data-parallel ranks.
    # Pure TP (tp_size == world_size): no DDP — all ranks are in the same
    # TP group with identical replicated-param gradients.
    if dp_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            process_group=dp_group,
            gradient_as_bucket_view=True,
        )
    else:
        if is_main:
            logger.info("Pure TP mode (TP=%d = world_size) — skipping DDP", tp_size)

    # -- Optimizer -------------------------------------------------------------
    raw_model = model.module if hasattr(model, "module") else model

    if args.adamw_only:
        # Pure AdamW baseline (no Muon) — all params in one optimizer
        from torch.optim import AdamW as TorchAdamW

        all_params = [p for p in raw_model.parameters() if p.requires_grad]
        adamw_opt = TorchAdamW(
            all_params,
            lr=args.adamw_lr,
            betas=(args.adamw_beta1, args.adamw_beta2),
            weight_decay=args.adamw_weight_decay,
        )
        # Create a no-op optimizer stand-in for Muon
        muon_opt = _NoOpOptimizer()
        if is_main:
            total_p = sum(p.numel() for p in all_params)
            logger.info("AdamW-only mode: %d params (%.1fM)", total_p, total_p / 1e6)
    else:
        muon_opt, adamw_opt = build_hybrid_optimizer(
            raw_model,
            muon_lr=args.muon_lr,
            muon_momentum=args.muon_momentum,
            muon_weight_decay=args.muon_weight_decay,
            muon_ns_iterations=args.muon_ns_iterations,
            muon_ns_coefficients=args.muon_ns_coefficients,
            adamw_lr=args.adamw_lr,
            adamw_betas=(args.adamw_beta1, args.adamw_beta2),
            adamw_weight_decay=args.adamw_weight_decay,
        )

    # -- Compute training geometry ---------------------------------------------
    seq_len = args.sequence_length
    micro_batch = args.micro_batch_size
    tokens_per_micro = micro_batch * seq_len  # per GPU
    # Under TP, all ranks in a TP group process the same data.
    # Only data-parallel replicas contribute unique tokens.
    tokens_per_global_micro = tokens_per_micro * dp_size

    if args.total_steps is not None:
        total_steps = args.total_steps
        # Infer total_tokens from steps (useful for smoke tests)
        grad_accum = max(1, args.global_batch_tokens // tokens_per_global_micro)
        total_tokens = total_steps * grad_accum * tokens_per_global_micro
    else:
        total_tokens = args.total_tokens
        grad_accum = max(1, args.global_batch_tokens // tokens_per_global_micro)
        total_steps = total_tokens // (grad_accum * tokens_per_global_micro)

    tokens_per_step = grad_accum * tokens_per_global_micro

    if is_main:
        logger.info(
            "Training: %d steps, %d grad_accum, %d tokens/step (%.2fM), "
            "%.1fB total tokens",
            total_steps,
            grad_accum,
            tokens_per_step,
            tokens_per_step / 1e6,
            total_tokens / 1e9,
        )

    # -- Scheduler -------------------------------------------------------------
    scheduler = HybridScheduler(
        muon_opt,
        adamw_opt,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        decay_start_pct=args.decay_start_pct,
        decay_type=args.decay_type,
    )

    # -- Data ------------------------------------------------------------------
    # Under TP, all ranks in a TP group must see the same data.
    # Partition by dp_rank so TP peers share batches.
    dp_rank = dist.get_rank(dp_group) if dp_group is not None else 0

    if args.random_data:
        dataset = RandomTokenDataset(
            vocab_size=config.vocab_size,
            seq_len=seq_len,
            seed=args.seed,
            rank=dp_rank,
        )
        if is_main:
            logger.info("Using random data (vocab=%d)", config.vocab_size)
    else:
        if not args.data_path:
            raise ValueError("--data_path required when not using --random_data")
        dataset = TokenizedDataset(
            path=args.data_path,
            seq_len=seq_len,
            rank=dp_rank,
            world_size=dp_size,
            seed=args.seed,
        )

    def _make_data_iter():
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=micro_batch,
            num_workers=0,  # IterableDataset handles its own partitioning
            pin_memory=True,
        )
        return iter(loader)

    data_iter = _make_data_iter()

    # -- Checkpoint manager ----------------------------------------------------
    if getattr(args, "async_checkpoint", False):
        ckpt_mgr = AsyncCheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            rank=rank,
            keep_last_n=args.keep_checkpoints,
            shm_dir=getattr(args, "checkpoint_shm_dir", "/dev/shm/luxia-base-ckpts"),
            compress=getattr(args, "checkpoint_compress", True),
        )
    else:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir,
            rank=rank,
            keep_last_n=args.keep_checkpoints,
        )

    # Resume from checkpoint if one exists
    start_step = 0
    tokens_consumed = 0
    ckpt_state = ckpt_mgr.load_latest(model, muon_opt, adamw_opt, device)
    if ckpt_state is not None:
        start_step = ckpt_state["step"] + 1
        tokens_consumed = ckpt_state.get("tokens_consumed", 0)
        # Restore data loader position
        if "data" in ckpt_state and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(ckpt_state["data"])
            data_iter = _make_data_iter()
        # Restore scheduler position
        scheduler.step(start_step)

    # -- SIGTERM handler -------------------------------------------------------
    sigterm = SIGTERMHandler()

    # -- Geometric monitoring (rank 0 only) ------------------------------------
    monitor: Optional[GeometricMonitor] = None
    if is_main and args.geo_monitor:
        monitor_config = MonitorConfig(
            tier1_every=args.geo_monitor_tier1_every,
            tier2_every=args.geo_monitor_tier2_every,
        )
        monitor = GeometricMonitor(model, monitor_config)

        # Build a fixed probe batch for longitudinal monitoring
        probe_batches = []
        probe_iter = _make_data_iter()
        probe_tokens_needed = min(
            monitor_config.tier1_probe_size, 64
        )  # sequences, not tokens
        for _ in range(probe_tokens_needed):
            try:
                probe_batches.append(next(probe_iter))
            except StopIteration:
                break
        if probe_batches:
            probe_batch = torch.cat(probe_batches, dim=0)[:probe_tokens_needed]
            monitor.set_probe_batch(probe_batch)
            logger.info("Geometric monitor ready (probe batch: %s)", tuple(probe_batch.shape))

    # -- Wandb logger (rank 0 only) -------------------------------------------
    wb: Optional[WandbLogger] = None
    if is_main and args.wandb:
        wb = WandbLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config={
                "model_size": args.model_size,
                "total_tokens": total_tokens,
                "total_steps": total_steps,
                "grad_accum": grad_accum,
                "tokens_per_step": tokens_per_step,
                "micro_batch_size": micro_batch,
                "sequence_length": seq_len,
                "world_size": world_size,
                "muon_lr": args.muon_lr,
                "muon_ns_coefficients": args.muon_ns_coefficients or "original",
                "adamw_lr": args.adamw_lr,
                "warmup_steps": args.warmup_steps,
                "precision": "bf16",
                "parallelism": f"tp{tp_size}" if tp_size > 1 else "ddp",
                "tp_size": tp_size,
                "dp_size": dp_size,
                "activation_checkpointing": args.activation_checkpointing,
                "use_liger": getattr(args, "use_liger", False),
                "attn_impl": getattr(args, "attn_impl", "auto"),
            },
            log_dir=args.checkpoint_dir,
            tags=["pretraining", args.model_size],
        )

    # -- Profiler setup --------------------------------------------------------
    _profiler = None
    _profile_end_step = -1
    if getattr(args, "profile", False) and is_main:
        _profile_dir = Path(args.profile_dir or args.checkpoint_dir) / "profile"
        _profile_dir.mkdir(parents=True, exist_ok=True)
        _profile_start = args.profile_start_step
        _profile_end_step = _profile_start + args.profile_steps
        logger.info(
            "Profiler enabled: steps %d-%d, output: %s",
            _profile_start, _profile_end_step, _profile_dir,
        )

    # -- Training loop ---------------------------------------------------------
    if is_main:
        logger.info("Starting training from step %d", start_step)

    model.train()
    step_t0 = time.time()
    log_loss_accum = 0.0
    log_z_loss_accum = 0.0
    log_steps = 0

    # Pre-allocate loss accumulators on GPU to avoid .item() sync per micro-step
    _loss_accum = torch.zeros(1, device=device)
    _z_loss_accum = torch.zeros(1, device=device)

    for step in range(start_step, total_steps):
        # Start profiler at the configured step
        if _profiler is None and is_main and step == getattr(args, "profile_start_step", -1) and getattr(args, "profile", False):
            _profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            )
            _profiler.__enter__()
            logger.info("Profiler started at step %d", step)
        _loss_accum.zero_()
        _z_loss_accum.zero_()

        # Batch size ramp: dynamically adjust grad_accum
        current_grad_accum = _compute_grad_accum_for_step(
            step, total_steps, grad_accum,
            initial_batch_fraction=args.batch_ramp_initial,
            ramp_pct=args.batch_ramp_pct,
        )

        for micro_step in range(current_grad_accum):
            try:
                input_ids = next(data_iter)
            except StopIteration:
                # Should not happen with infinite datasets, but handle gracefully
                data_iter = _make_data_iter()
                input_ids = next(data_iter)

            input_ids = input_ids.to(device, non_blocking=True)

            # Skip gradient sync on all but the last micro-step.
            # no_sync() only exists on DDP-wrapped models.
            sync_ctx = (
                model.no_sync()
                if hasattr(model, "no_sync") and micro_step < current_grad_accum - 1
                else nullcontext()
            )

            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(input_ids, labels=input_ids)
                    loss = output["loss"] / current_grad_accum

                loss.backward()

            _loss_accum += output["loss"].detach()
            if "z_loss" in output:
                _z_loss_accum += output["z_loss"].detach()

        # Single GPU→CPU sync after all micro-steps
        step_loss = _loss_accum.item() / current_grad_accum
        step_z_loss = _z_loss_accum.item() / current_grad_accum

        # TP: sync QK-norm gradients (they see sharded heads, need SUM)
        if tp_group is not None and dp_size == 1:
            from src.training.tensor_parallel import sync_tp_replicated_grads
            sync_tp_replicated_grads(raw_model, tp_group)

        # Gradient clipping — TP-aware norm when TP is active
        if args.gradient_clip > 0:
            if tp_group is not None:
                from src.training.tensor_parallel import tp_clip_grad_norm
                grad_norm = tp_clip_grad_norm(model, args.gradient_clip, tp_group)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
        else:
            grad_norm = _compute_grad_norm(model.parameters())

        # Per-layer gradient analysis (before optimizer step clears grads)
        if wb is not None and step % 50 == 0:
            grad_metrics = _per_layer_grad_norms(raw_model)
            wb.log_custom(step, grad_metrics)

        # Optimizer step
        muon_opt.step()
        adamw_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)

        # Scheduler step
        scheduler.step(step)

        # Track tokens (use current_grad_accum for accurate counting)
        current_tokens_per_step = current_grad_accum * tokens_per_global_micro
        tokens_consumed += current_tokens_per_step

        # -- Logging -----------------------------------------------------------
        log_loss_accum += step_loss
        log_z_loss_accum += step_z_loss
        log_steps += 1

        if is_main and (step % args.log_every == 0 or step == total_steps - 1):
            elapsed = time.time() - step_t0
            avg_loss = log_loss_accum / max(log_steps, 1)
            avg_z_loss = log_z_loss_accum / max(log_steps, 1)
            tokens_per_sec = (log_steps * current_tokens_per_step) / max(elapsed, 1e-6)
            iters_per_sec = log_steps / max(elapsed, 1e-6)
            lrs = scheduler.get_last_lr()
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            _grad_norm_scalar = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            logger.info(
                "step=%d/%d | loss=%.4f | z_loss=%.6f | grad_norm=%.3f | "
                "muon_lr=%.6f | adamw_lr=%.6f | tok/s=%.0f | it/s=%.2f | "
                "tokens=%.3fB | gpu_mem=%.1fGB",
                step,
                total_steps,
                avg_loss,
                avg_z_loss,
                _grad_norm_scalar,
                lrs["muon_lr"],
                lrs["adamw_lr"],
                tokens_per_sec,
                iters_per_sec,
                tokens_consumed / 1e9,
                gpu_mem,
            )

            if wb is not None:
                wb.log_step(
                    step=step,
                    loss=avg_loss,
                    z_loss=avg_z_loss,
                    grad_norm=_grad_norm_scalar,
                    muon_lr=lrs["muon_lr"],
                    adamw_lr=lrs["adamw_lr"],
                    tokens_per_sec=tokens_per_sec,
                    iters_per_sec=iters_per_sec,
                    tokens_consumed=tokens_consumed,
                    gpu_mem_gb=gpu_mem,
                    step_time_s=elapsed / max(log_steps, 1),
                )

            if is_main:
                hm_log(
                    step,
                    **{"train/loss": avg_loss, "train/grad_norm": _grad_norm_scalar,
                       "train/perplexity": math.exp(min(avg_loss, 20)),
                       "perf/tokens_per_sec": tokens_per_sec},
                )

            # Reset accumulators
            log_loss_accum = 0.0
            log_z_loss_accum = 0.0
            log_steps = 0
            step_t0 = time.time()

        # -- Geometric monitoring (rank 0 only) --------------------------------
        if monitor is not None:
            if args.geo_monitor_tier1_every > 0 and step % args.geo_monitor_tier1_every == 0:
                geo_metrics = monitor.tier1(step)
                if geo_metrics:
                    if wb is not None:
                        wb.log_geo(step, geo_metrics)
                    hm_log(step, **{k: v for k, v in geo_metrics.items() if isinstance(v, (int, float))})
                    if step % args.log_every == 0:
                        rankme = geo_metrics.get("geo/rankme_last", 0)
                        logger.info("  geo: RankMe=%.1f", rankme)

            if args.geo_monitor_tier2_every > 0 and step % args.geo_monitor_tier2_every == 0:
                geo_metrics_t2 = monitor.tier2(step)
                if geo_metrics_t2:
                    if wb is not None:
                        wb.log_geo(step, geo_metrics_t2)
                    hm_log(step, **{k: v for k, v in geo_metrics_t2.items() if isinstance(v, (int, float))})
                    ww_mean = geo_metrics_t2.get("geo/ww_alpha_mean", 0)
                    ww_healthy = geo_metrics_t2.get("geo/ww_alpha_healthy_frac", 0)
                    logger.info(
                        "  geo tier2: WW_alpha=%.2f, healthy=%.0f%%",
                        ww_mean,
                        ww_healthy * 100,
                    )

        # Commit wandb step
        if wb is not None:
            wb.commit(step)

        # -- Profiler step/stop ------------------------------------------------
        if _profiler is not None:
            _profiler.step()
            if step >= _profile_end_step:
                _profiler.__exit__(None, None, None)
                # Write chrome trace
                trace_path = _profile_dir / "chrome_trace.json"
                _profiler.export_chrome_trace(str(trace_path))
                # Write summary table
                summary_path = _profile_dir / "profile_summary.txt"
                summary = _profiler.key_averages().table(
                    sort_by="cuda_time_total", row_limit=60,
                )
                flops = _profiler.key_averages().table(
                    sort_by="flops", row_limit=20,
                )
                with open(summary_path, "w") as f:
                    f.write(summary)
                    f.write("\n\n--- FLOPS ---\n\n")
                    f.write(flops)
                logger.info(
                    "Profiler stopped at step %d. Trace: %s, Summary: %s",
                    step, trace_path, summary_path,
                )
                _profiler = None

        # -- Checkpoint --------------------------------------------------------
        should_save = (
            step > 0
            and args.save_every > 0
            and step % args.save_every == 0
        )

        if should_save:
            data_state = (
                dataset.state_dict() if hasattr(dataset, "state_dict") else None
            )
            ckpt_mgr.save(
                step=step,
                model=model,
                muon_opt=muon_opt,
                adamw_opt=adamw_opt,
                scheduler=scheduler,
                tokens_consumed=tokens_consumed,
                data_state=data_state,
            )

        # -- SIGTERM check -----------------------------------------------------
        if sigterm.received:
            if is_main:
                logger.info("SIGTERM exit — saving checkpoint at step %d", step)
            data_state = (
                dataset.state_dict() if hasattr(dataset, "state_dict") else None
            )
            # Use blocking save for SIGTERM (async may not finish before SIGKILL)
            save_fn = (
                ckpt_mgr.save_blocking
                if hasattr(ckpt_mgr, "save_blocking")
                else ckpt_mgr.save
            )
            save_fn(
                step=step,
                model=model,
                muon_opt=muon_opt,
                adamw_opt=adamw_opt,
                scheduler=scheduler,
                tokens_consumed=tokens_consumed,
                data_state=data_state,
            )
            break

    # -- Final -----------------------------------------------------------------
    # All ranks must participate in save (it contains a barrier)
    if not sigterm.received:
        if is_main:
            logger.info(
                "Training complete: %d steps, %.2fB tokens",
                total_steps,
                tokens_consumed / 1e9,
            )
        data_state = (
            dataset.state_dict() if hasattr(dataset, "state_dict") else None
        )
        ckpt_mgr.save(
            step=total_steps - 1,
            model=model,
            muon_opt=muon_opt,
            adamw_opt=adamw_opt,
            scheduler=scheduler,
            tokens_consumed=tokens_consumed,
            data_state=data_state,
        )

    # Flush async checkpoint queue before shutting down
    if hasattr(ckpt_mgr, "shutdown"):
        ckpt_mgr.shutdown()

    if wb is not None:
        wb.finish()
    sigterm.restore()
    cleanup_distributed()


def _compute_grad_accum_for_step(
    step: int,
    total_steps: int,
    base_grad_accum: int,
    initial_batch_fraction: float,
    ramp_pct: float,
) -> int:
    """
    Compute gradient accumulation steps with batch size ramp.

    Starts at ``base_grad_accum * initial_batch_fraction`` and linearly
    ramps to ``base_grad_accum`` over the first ``ramp_pct`` of training.

    Spec: 512K → 2M tokens over first 5% (initial_batch_fraction=0.25, ramp_pct=0.05).
    """
    if ramp_pct <= 0 or initial_batch_fraction >= 1.0:
        return base_grad_accum

    ramp_steps = int(total_steps * ramp_pct)
    if step >= ramp_steps:
        return base_grad_accum

    # Linear ramp from initial to full
    progress = step / max(ramp_steps, 1)
    fraction = initial_batch_fraction + (1.0 - initial_batch_fraction) * progress
    return max(1, int(base_grad_accum * fraction))


class _NoOpOptimizer:
    """Dummy optimizer that does nothing. Used when --adamw_only is set."""

    param_groups: list[dict[str, Any]] = []
    defaults: dict[str, Any] = {"lr": 0.0}

    def step(self, closure: Any = None) -> None:
        pass

    def zero_grad(self, set_to_none: bool = False) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


def _compute_grad_norm(params: Any) -> torch.Tensor:
    """Compute total gradient norm on GPU (single sync when .item() is called)."""
    norms = [p.grad.data.float().norm() for p in params if p.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.stack(norms).square().sum().sqrt()


def _per_layer_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Compute gradient norms per transformer layer (attn vs MLP split)."""
    metrics: dict[str, float] = {}

    if not hasattr(model, "layers"):
        return metrics

    for i, layer in enumerate(model.layers):
        if i % 4 != 0 and i != len(model.layers) - 1:
            continue

        attn_norms = []
        mlp_norms = []

        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            norm = param.grad.data.float().norm()
            if "attn" in name:
                attn_norms.append(norm)
            elif "ffn" in name:
                mlp_norms.append(norm)

        attn_total = torch.stack(attn_norms).square().sum().sqrt().item() if attn_norms else 0.0
        mlp_total = torch.stack(mlp_norms).square().sum().sqrt().item() if mlp_norms else 0.0
        metrics[f"grad/layer_{i}/attn"] = attn_total
        metrics[f"grad/layer_{i}/mlp"] = mlp_total

        if mlp_total > 1e-10:
            metrics[f"grad/layer_{i}/attn_mlp_ratio"] = attn_total / (mlp_total + 1e-10)

    return metrics


# =============================================================================
# CLI
# =============================================================================


def _load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML config file and normalize keys (hyphens → underscores)."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --config. Install: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {k.replace("-", "_"): v for k, v in raw.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="luxia-base distributed training")

    # Config file (YAML defaults, CLI overrides)
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config file. CLI flags override config values.",
    )

    # Model
    p.add_argument(
        "--model_size",
        type=str,
        default="proxy",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model size preset",
    )
    p.add_argument("--z_loss_weight", type=float, default=1e-5)
    p.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation checkpointing (saves memory, required for full 3B)",
    )

    # Block Attention Residuals (Moonshot, 2026)
    p.add_argument(
        "--attn_res",
        action="store_true",
        help="Enable Block Attention Residuals (learned depth-wise routing)",
    )
    p.add_argument(
        "--attn_res_n_blocks",
        type=int,
        default=7,
        help="Number of AttnRes blocks (default 7 for 28 layers = blocks of 4)",
    )
    p.add_argument(
        "--attn_res_boundaries",
        type=str,
        default=None,
        help="Explicit block boundary layers as comma-separated ints (e.g. '0,3,7,12,21,25'). Overrides --attn_res_n_blocks.",
    )

    # Training scale
    p.add_argument("--total_tokens", type=int, default=6_000_000_000)
    p.add_argument(
        "--total_steps",
        type=int,
        default=None,
        help="Override total_tokens with explicit step count",
    )
    p.add_argument("--sequence_length", type=int, default=2048)
    p.add_argument("--micro_batch_size", type=int, default=8)
    p.add_argument(
        "--global_batch_tokens",
        type=int,
        default=2_097_152,
        help="Global batch size in tokens (default 2M)",
    )
    p.add_argument("--gradient_clip", type=float, default=1.0)

    # Batch size ramp (spec: 512K → 2M over first 5%)
    p.add_argument(
        "--batch_ramp_initial",
        type=float,
        default=1.0,
        help="Initial batch size as fraction of full (0.25 = 1/4, 1.0 = no ramp)",
    )
    p.add_argument(
        "--batch_ramp_pct",
        type=float,
        default=0.0,
        help="Fraction of training to ramp over (0.05 = first 5%%, 0.0 = no ramp)",
    )

    # Optimizer mode
    p.add_argument(
        "--adamw_only",
        action="store_true",
        help="Use pure AdamW (no Muon) — for baseline comparison",
    )

    # Muon optimizer
    p.add_argument("--muon_lr", type=float, default=0.03)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_weight_decay", type=float, default=0.01)
    p.add_argument("--muon_ns_iterations", type=int, default=5)
    p.add_argument(
        "--muon_ns_coefficients", type=str, default=None,
        choices=["original", "gram_ns", "polar_express"],
        help="NS coefficient preset. 'original': fixed MoonshotAI coefficients. "
             "'gram_ns': per-iteration optimized (Dao-AILab Gram-Newton-Schulz). "
             "'polar_express': conservative per-iteration (Dao-AILab). "
             "Default: original.",
    )

    # AdamW optimizer
    p.add_argument("--adamw_lr", type=float, default=6e-4)
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95)
    p.add_argument("--adamw_weight_decay", type=float, default=0.1)

    # Schedule
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--decay_start_pct", type=float, default=0.90)
    p.add_argument("--decay_type", type=str, default="sqrt")

    # Data
    p.add_argument("--data_path", type=str, default="")
    p.add_argument(
        "--random_data",
        action="store_true",
        help="Use random tokens (for testing)",
    )
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--keep_checkpoints", type=int, default=5)
    p.add_argument(
        "--async_checkpoint", action="store_true",
        help="Async checkpointing: CPU clone + background SHM write + zstd",
    )
    p.add_argument(
        "--checkpoint_shm_dir", type=str, default="/dev/shm/luxia-base-ckpts",
        help="SHM directory for async checkpoint temp storage",
    )
    p.add_argument(
        "--checkpoint_compress", action="store_true", default=True,
        help="Compress checkpoints with zstd (default: True)",
    )
    p.add_argument(
        "--no_checkpoint_compress", dest="checkpoint_compress", action="store_false",
        help="Disable zstd compression for async checkpoints",
    )

    # Logging
    p.add_argument("--log_every", type=int, default=10)

    # Profiling
    p.add_argument(
        "--profile", action="store_true",
        help="Enable torch.profiler for kernel-level tracing",
    )
    p.add_argument(
        "--profile_start_step", type=int, default=50,
        help="Step to start profiling (skip JIT warmup)",
    )
    p.add_argument(
        "--profile_steps", type=int, default=20,
        help="Number of steps to profile",
    )
    p.add_argument(
        "--profile_dir", type=str, default=None,
        help="Directory for profiler trace output (default: checkpoint_dir/profile)",
    )

    # Geometric monitoring
    p.add_argument(
        "--geo_monitor",
        action="store_true",
        help="Enable geometric health monitoring (rank 0 only)",
    )
    p.add_argument("--geo_monitor_tier1_every", type=int, default=500)
    p.add_argument("--geo_monitor_tier2_every", type=int, default=5000)

    # NCA → language transition
    p.add_argument(
        "--resume_nca",
        type=str,
        default=None,
        help="Path to NCA checkpoint — loads weights, reinitializes embeddings",
    )
    p.add_argument(
        "--reinit_mlps",
        action="store_true",
        help="Also reinitialize MLPs after NCA (default: keep NCA-trained MLPs)",
    )

    # FP8 training
    p.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 training via torchao (Linear layers only, Muon NS stays fp16)",
    )

    # Liger fused kernels
    p.add_argument(
        "--use_liger",
        action="store_true",
        help="Enable Liger fused kernels (RMSNorm, SwiGLU, CrossEntropy)",
    )

    # Attention implementation
    p.add_argument(
        "--attn_impl", type=str, default="auto",
        choices=["auto", "fa2", "fa4", "sdpa"],
        help="Attention backend. 'auto': FA2 if available, else SDPA. "
             "'fa2': Flash Attention 2. 'fa4': CuTeDSL SM100 (lazy import, breaks compile). "
             "'sdpa': PyTorch SDPA.",
    )

    # Tensor parallelism
    p.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism degree (1 = DDP only, 8 = full TP on 8 GPUs)",
    )

    # torch.compile
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for throughput",
    )

    # Wandb
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging (rank 0 only)",
    )
    p.add_argument("--wandb_project", type=str, default="luxia-base")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # First parse to check for --config
    args = p.parse_args()

    # If config file provided, load YAML as defaults and re-parse
    # so CLI flags take precedence over YAML values
    if args.config is not None:
        config = _load_yaml_config(args.config)
        # Warn about unknown keys (likely typos)
        known_args = {a.dest for a in p._actions}
        unknown = set(config.keys()) - known_args
        if unknown:
            logger.warning(
                "Unknown keys in %s (ignored): %s",
                args.config, ", ".join(sorted(unknown)),
            )
            for k in unknown:
                del config[k]
        p.set_defaults(**config)
        args = p.parse_args()
        args._config_file = args.config

    return args


def main() -> None:
    args = parse_args()
    try:
        train(args)
    except Exception:
        logger.exception("Training failed")
        cleanup_distributed()
        raise


if __name__ == "__main__":
    main()
