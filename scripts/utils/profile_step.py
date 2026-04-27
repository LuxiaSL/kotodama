"""
Profile a single training step to identify Muon throughput bottlenecks.

Runs a few warmup steps then profiles one step in detail using
torch.profiler. Outputs a breakdown of time spent in each operation.

Usage:
    # Profile Muon on 8 GPUs
    torchrun --nproc_per_node=8 scripts/profile_step.py --muon

    # Profile AdamW for comparison
    torchrun --nproc_per_node=8 scripts/profile_step.py --adamw

    # Profile Muon with 3 NS iterations
    torchrun --nproc_per_node=8 scripts/profile_step.py --muon --ns_iter 3
"""

import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, ".")
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from src.training.muon import Muon, build_hybrid_optimizer


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--muon", action="store_true")
    p.add_argument("--adamw", action="store_true")
    p.add_argument("--ns_iter", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--profile_steps", type=int, default=3)
    args = p.parse_args()

    if not args.muon and not args.adamw:
        args.muon = True  # default

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Proxy model
    config = LuxiaModelConfig(
        hidden_size=512, num_layers=28, num_attention_heads=4,
        num_kv_heads=2, head_dim=128, intermediate_size=1408,
        vocab_size=49152, max_position_embeddings=4096,
    )
    model = LuxiaBaseModel(config).to(device)

    if args.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank])

    # Optimizer
    if args.adamw:
        from torch.optim import AdamW
        opt = AdamW(model.parameters(), lr=8e-4)
        muon_opt = None
        adamw_opt = opt
        mode = "AdamW"
    else:
        muon_opt, adamw_opt = build_hybrid_optimizer(
            model.module if not args.compile else (model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module),
            muon_lr=0.03, muon_ns_iterations=args.ns_iter,
        )
        mode = f"Muon (NS={args.ns_iter})"

    if rank == 0:
        print(f"\nProfiling: {mode}, compile={args.compile}, 8 GPUs")
        print(f"Warmup: {args.warmup_steps} steps, Profile: {args.profile_steps} steps\n")

    # Random data
    batch_size = 4
    seq_len = 2048
    grad_accum = 2

    # Warmup (including torch.compile JIT)
    model.train()
    for step in range(args.warmup_steps):
        for _ in range(grad_accum):
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, labels=x)
                loss = out["loss"] / grad_accum
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if muon_opt:
            muon_opt.step()
            muon_opt.zero_grad(set_to_none=True)
        adamw_opt.step()
        adamw_opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    if rank == 0:
        print("Warmup done. Profiling...\n")

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for step in range(args.profile_steps):
            with record_function("FULL_STEP"):
                with record_function("FORWARD_BACKWARD"):
                    for micro in range(grad_accum):
                        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
                        with record_function("forward"):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                out = model(x, labels=x)
                                loss = out["loss"] / grad_accum
                        with record_function("backward"):
                            loss.backward()

                with record_function("GRAD_CLIP"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                with record_function("OPTIMIZER_STEP"):
                    if muon_opt:
                        with record_function("muon_step"):
                            muon_opt.step()
                        with record_function("muon_zero_grad"):
                            muon_opt.zero_grad(set_to_none=True)
                    with record_function("adamw_step"):
                        adamw_opt.step()
                    with record_function("adamw_zero_grad"):
                        adamw_opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    if rank == 0:
        # Print summary
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print("\n" + "=" * 60)
        print("CUSTOM REGIONS:")
        print("=" * 60)
        for evt in prof.key_averages():
            if evt.key in [
                "FULL_STEP", "FORWARD_BACKWARD", "forward", "backward",
                "GRAD_CLIP", "OPTIMIZER_STEP", "muon_step", "adamw_step",
                "muon_zero_grad", "adamw_zero_grad",
            ]:
                cuda_ms = evt.cuda_time_total / 1000
                cpu_ms = evt.cpu_time_total / 1000
                count = evt.count
                print(f"  {evt.key:<25} CUDA: {cuda_ms:>8.1f}ms  CPU: {cpu_ms:>8.1f}ms  (×{count})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
