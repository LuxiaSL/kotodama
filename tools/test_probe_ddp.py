#!/usr/bin/env python3
"""Test 3: DDP reproduction — does the crash require multi-GPU?

Run with: torchrun --nproc_per_node=8 tools/test_probe_ddp.py

Tests the exact sequence from the training loop:
  1. DDP compiled forward+backward (all ranks)
  2. compute_sharpness (rank 0 only, 3 model forwards with weight perturbation)
  3. Optimizer step + zero_grad (all ranks)
  4. tier1 probe (rank 0 only) — CRASH POINT

Also tests with/without preceding compute_sharpness to check if it leaves
dirty state that triggers the probe crash.
"""

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]


def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def build_model(local_rank, use_fp8=True, use_compile=True, use_liger=True):
    from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

    config = LuxiaModelConfig(
        vocab_size=49152,
        hidden_size=3072,
        num_layers=28,
        num_attention_heads=24,
        num_kv_heads=8,
        intermediate_size=8192,
        max_position_embeddings=4096,
        attn_res=True,
        attn_res_boundaries=BOUNDARIES,
        use_liger=use_liger,
        attn_impl="auto",
        activation_checkpointing=True,
    )

    model = LuxiaBaseModel(config).cuda().to(torch.bfloat16)

    if use_fp8:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        convert_to_float8_training(model, config=Float8LinearConfig())

    if use_compile:
        model = torch.compile(model)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        gradient_as_bucket_view=True,
        find_unused_parameters=True,  # matches training config for AttnRes
    )

    return ddp_model, config


def run_test(test_name, local_rank, rank, world_size,
             use_fp8=True, use_compile=True, use_liger=True,
             run_sharpness_first=True, run_optimizer=True):
    """Run one complete test matching the training loop sequence."""
    is_main = rank == 0
    if is_main:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"  fp8={use_fp8}, compile={use_compile}, liger={use_liger}")
        print(f"  sharpness_first={run_sharpness_first}, optimizer={run_optimizer}")
        print(f"{'='*60}")

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()

        ddp_model, config = build_model(local_rank, use_fp8, use_compile, use_liger)
        if is_main:
            print(f"  Model built, GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

        # Set up geo monitor on rank 0
        monitor = None
        if is_main:
            from src.monitoring.geometric import GeometricMonitor, MonitorConfig
            monitor = GeometricMonitor(ddp_model, MonitorConfig(
                tier1_every=1,
                tier2_every=9999,
            ))
            probe_batch = torch.randint(0, 49152, (16, 4096), device="cuda")
            monitor.set_probe_batch(probe_batch)

        # Set up optimizer (matching training)
        from src.training.muon import Muon
        raw_model = ddp_model.module
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        muon_params = []
        adamw_params = []
        for name, p in raw_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and "embed" not in name and "norm" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)

        muon_opt = Muon(muon_params, lr=0.02, momentum=0.95,
                        weight_decay=0.01, ns_iterations=5, ns_coefficients="gram_ns")
        adamw_opt = torch.optim.AdamW(adamw_params, lr=6e-4,
                                       betas=(0.9, 0.95), weight_decay=0.1)

        dist.barrier()
        if is_main:
            print("  Starting training step...")

        # === Step 1: Training forward+backward (all ranks) ===
        tokens = torch.randint(0, 49152, (10, 4096), device="cuda")  # match MB=10
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = ddp_model(tokens, labels=tokens)
            loss = output["loss"]
        loss.backward()
        torch.cuda.synchronize()

        if is_main:
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            print(f"  Training fwd+bwd OK: loss={loss.item():.4f}, grad_norm={grad_norm:.3f}")
            print(f"  GPU mem: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")

        # === Step 2: EoS sharpness (rank 0 only) ===
        if run_sharpness_first and is_main:
            print("  Running compute_sharpness...")
            eos_metrics = monitor.compute_sharpness(0)
            sharpness = eos_metrics.get("eos/sharpness", "N/A")
            print(f"  Sharpness={sharpness}")

        # === Step 3: Optimizer step (all ranks) ===
        if run_optimizer:
            muon_opt.step()
            adamw_opt.step()
            muon_opt.zero_grad(set_to_none=True)
            adamw_opt.zero_grad(set_to_none=True)
            if is_main:
                print("  Optimizer step OK")

        # === Step 4: tier1 probe (rank 0 only) — THE CRASH POINT ===
        if is_main:
            print("  Running tier1 probe (THIS IS THE CRASH POINT)...")
            geo_metrics = monitor.tier1(0)
            n_metrics = len(geo_metrics)
            rankme = geo_metrics.get("geo/rankme_last", "N/A")
            print(f"  tier1 OK: {n_metrics} metrics, RankMe={rankme}")

        dist.barrier()
        if is_main:
            print(f"  {test_name}: PASSED")
        return True

    except Exception as e:
        if is_main:
            print(f"  {test_name}: FAILED")
            traceback.print_exc()
        # Sync so other ranks don't hang
        try:
            dist.barrier()
        except Exception:
            pass
        return False


def main():
    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("DDP PROBE CRASH REPRODUCTION TEST")
        print(f"Ranks: {world_size}, Local rank: {local_rank}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print("=" * 60)

    results = {}

    # Test A: Full reproduction — exact training loop sequence
    results["A: full repro"] = run_test(
        "Full reproduction (FP8+compile+liger+sharpness+optimizer)",
        local_rank, rank, world_size,
        use_fp8=True, use_compile=True, use_liger=True,
        run_sharpness_first=True, run_optimizer=True,
    )

    # If A fails, run targeted tests to isolate
    if not results["A: full repro"]:
        # Test B: Without compute_sharpness first
        results["B: no sharpness"] = run_test(
            "No sharpness (skip compute_sharpness)",
            local_rank, rank, world_size,
            use_fp8=True, use_compile=True, use_liger=True,
            run_sharpness_first=False, run_optimizer=True,
        )

        # Test C: Without optimizer step
        results["C: no optimizer"] = run_test(
            "No optimizer step (probe right after backward)",
            local_rank, rank, world_size,
            use_fp8=True, use_compile=True, use_liger=True,
            run_sharpness_first=False, run_optimizer=False,
        )

        # Test D: No Liger
        results["D: no liger"] = run_test(
            "No Liger (standard RMSNorm)",
            local_rank, rank, world_size,
            use_fp8=True, use_compile=True, use_liger=False,
            run_sharpness_first=True, run_optimizer=True,
        )

        # Test E: No FP8
        results["E: no fp8"] = run_test(
            "No FP8",
            local_rank, rank, world_size,
            use_fp8=False, use_compile=True, use_liger=True,
            run_sharpness_first=True, run_optimizer=True,
        )

        # Test F: No compile
        results["F: no compile"] = run_test(
            "No compile",
            local_rank, rank, world_size,
            use_fp8=True, use_compile=False, use_liger=True,
            run_sharpness_first=True, run_optimizer=True,
        )
    else:
        if is_main:
            print("\nTest A passed — crash does NOT reproduce under DDP alone.")
            print("The issue may require real NCA data, expandable_segments, or the full training loop.")

    if is_main:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for name, passed in results.items():
            print(f"  [{('PASS' if passed else 'FAIL')}] {name}")

    dist.destroy_process_group()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
