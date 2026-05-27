#!/usr/bin/env python3
"""Test 2: Full 28-layer probe loop mimicking _probe_forward_attn_res.

The isolated kernel test passed — this tests whether the crash requires:
- Iterative layer execution (28 layers, growing committed buffer)
- The actual DD-3B boundary config [0,1,3,7,15,19,24]
- Full-scale tensors (D=3072, actual seq_len, realistic batch)
- Prior compiled training forward on the full model

Escalation levels:
  Level 1: Loop with standalone norms (no full model)
  Level 2: Full 3B model, compile, one training forward, then probe
  Level 3: Full 3B model + FP8 + compile + one training forward, then probe
"""

import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = 3072
EPS = 1e-5
NUM_LAYERS = 28
BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]
BOUNDARY_SET = set(BOUNDARIES)
B = 16       # closer to real probe batch
T = 4096     # actual seq_len


def compute_max_sources(boundaries, n_layers):
    return len(boundaries) + 2  # committed blocks + partial + padding


MAX_S = compute_max_sources(BOUNDARIES, NUM_LAYERS)


def precompute_masks(boundaries, n_layers, max_s, device="cuda"):
    """Replicate the mask precomputation from LuxiaBaseModel._setup_attn_res."""
    masks = []
    active_counts = []
    n_committed = 0
    for i in range(n_layers):
        # pre-attn: n_committed blocks + partial
        n_active = n_committed + 1
        m = torch.zeros(max_s, dtype=torch.bool, device=device)
        m[:n_active] = True
        masks.append(m)
        active_counts.append(n_active)

        if i in set(boundaries):
            n_committed += 1

        # pre-mlp: after potential commit
        n_active = n_committed + 1
        m = torch.zeros(max_s, dtype=torch.bool, device=device)
        m[:n_active] = True
        masks.append(m)
        active_counts.append(n_active)

    # final aggregation
    n_active = n_committed + 1
    m = torch.zeros(max_s, dtype=torch.bool, device=device)
    m[:n_active] = True
    masks.append(m)
    active_counts.append(n_active)

    return masks, active_counts


def route_static_triton(buf, qw_unsqueezed, eps, num_active):
    from src.model.flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
    out, _lse = phase_1_batched_attention_triton_op(buf, qw_unsqueezed, eps, num_active=num_active)
    return out[0]


def test_level1():
    """Level 1: Full 28-layer loop with standalone norms."""
    print(f"\n{'='*60}")
    print("LEVEL 1: 28-layer probe loop (standalone norms, no full model)")
    print(f"D={D}, B={B}, T={T}, boundaries={BOUNDARIES}")
    print(f"{'='*60}")

    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNorm

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

        masks, active_counts = precompute_masks(BOUNDARIES, NUM_LAYERS, MAX_S)

        # Create per-layer norms and queries (mimicking TransformerBlock)
        attn_norms = [LigerRMSNorm(D, eps=EPS).cuda() for _ in range(NUM_LAYERS)]
        ffn_norms = [LigerRMSNorm(D, eps=EPS).cuda() for _ in range(NUM_LAYERS)]
        attn_res_queries = [torch.randn(D, device="cuda") for _ in range(NUM_LAYERS)]
        mlp_res_queries = [torch.randn(D, device="cuda") for _ in range(NUM_LAYERS)]
        attn_res_norm_weights = [torch.ones(D, device="cuda") for _ in range(NUM_LAYERS)]
        mlp_res_norm_weights = [torch.ones(D, device="cuda") for _ in range(NUM_LAYERS)]
        final_query = torch.randn(D, device="cuda")
        final_norm_weight = torch.ones(D, device="cuda")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            embed = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
            committed = []
            partial = embed
            zero = torch.zeros_like(embed)

            def pad_and_stack(committed_list, partial_tensor):
                sources = committed_list + [partial_tensor]
                while len(sources) < MAX_S:
                    sources.append(zero)
                return torch.stack(sources, dim=0)

            for i in range(NUM_LAYERS):
                # Pre-attn routing
                buf = pad_and_stack(committed, partial)
                qw = (attn_res_queries[i] * attn_res_norm_weights[i]).unsqueeze(0)
                na = active_counts[2 * i]
                h = route_static_triton(buf, qw, EPS, na)

                if i in BOUNDARY_SET:
                    committed.append(partial.clone())
                    partial = torch.zeros_like(embed)

                # This is the crash line in _probe_forward_attn_res:
                normed = attn_norms[i](h)
                # Simulate attention (just a noop matmul to exercise the path)
                attn_out = normed * 0.1  # placeholder
                partial = partial + attn_out

                # Pre-MLP routing
                buf = pad_and_stack(committed, partial)
                qw = (mlp_res_queries[i] * mlp_res_norm_weights[i]).unsqueeze(0)
                na = active_counts[2 * i + 1]
                h = route_static_triton(buf, qw, EPS, na)

                normed = ffn_norms[i](h)
                mlp_out = normed * 0.1  # placeholder
                partial = partial + mlp_out

                if (i + 1) % 7 == 0 or i == NUM_LAYERS - 1:
                    torch.cuda.synchronize()
                    print(f"  Layer {i}: committed={len(committed)}, partial norm={partial.float().norm().item():.2f}, OK")

            # Final aggregation
            buf = pad_and_stack(committed, partial)
            qw = (final_query * final_norm_weight).unsqueeze(0)
            na = active_counts[2 * NUM_LAYERS]
            final_h = route_static_triton(buf, qw, EPS, na)
            torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        print(f"\n  Level 1 PASSED ({elapsed:.0f}ms)")
        return True

    except Exception as e:
        print(f"\n  Level 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_level2():
    """Level 2: Full 3B model, compile, train forward, then probe."""
    print(f"\n{'='*60}")
    print("LEVEL 2: Full 3B model + compile → training forward → probe")
    print(f"{'='*60}")

    try:
        from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

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
            use_liger=True,
            attn_impl="sdpa",
        )

        model = LuxiaBaseModel(config).cuda().to(torch.bfloat16)
        print(f"  Model created: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

        # Compile
        compiled_model = torch.compile(model)
        print("  Compiled")

        # Training forward
        tokens = torch.randint(0, 49152, (2, 512), device="cuda")  # small batch for test
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled_model(tokens, labels=tokens)
            loss = out["loss"]
            loss.backward()
        torch.cuda.synchronize()
        print(f"  Training forward+backward OK (loss={loss.item():.4f})")

        # Now run the probe on the UNWRAPPED model (same as geo monitor)
        from src.monitoring.geometric import GeometricMonitor, MonitorConfig
        monitor = GeometricMonitor(compiled_model, MonitorConfig(
            tier1_every=1,
            tier2_every=9999,
        ))
        probe_batch = torch.randint(0, 49152, (16, 4096), device="cuda")
        monitor.set_probe_batch(probe_batch)

        print("  Running tier1 probe...")
        geo_metrics = monitor.tier1(0)
        torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        n_metrics = len(geo_metrics)
        rankme = geo_metrics.get("geo/rankme_last", "N/A")
        print(f"  tier1 returned {n_metrics} metrics, RankMe={rankme}")
        print(f"\n  Level 2 PASSED ({elapsed:.0f}ms)")
        return True

    except Exception as e:
        print(f"\n  Level 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_level3():
    """Level 3: Full 3B model + FP8 + compile → train forward → probe."""
    print(f"\n{'='*60}")
    print("LEVEL 3: Full 3B model + FP8 + compile → training forward → probe")
    print(f"{'='*60}")

    try:
        from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

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
            use_liger=True,
            attn_impl="sdpa",
        )

        model = LuxiaBaseModel(config).cuda().to(torch.bfloat16)
        print(f"  Model created: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

        # FP8
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            convert_to_float8_training(model, config=Float8LinearConfig())
            print("  FP8 conversion OK")
        except ImportError:
            print("  FP8 not available, skipping")
            return True

        # Compile
        compiled_model = torch.compile(model)
        print("  Compiled")

        # Training forward
        tokens = torch.randint(0, 49152, (2, 512), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = compiled_model(tokens, labels=tokens)
            loss = out["loss"]
            loss.backward()
        torch.cuda.synchronize()
        print(f"  Training forward+backward OK (loss={loss.item():.4f})")

        # Probe
        from src.monitoring.geometric import GeometricMonitor, MonitorConfig
        monitor = GeometricMonitor(compiled_model, MonitorConfig(
            tier1_every=1,
            tier2_every=9999,
        ))
        probe_batch = torch.randint(0, 49152, (16, 4096), device="cuda")
        monitor.set_probe_batch(probe_batch)

        print("  Running tier1 probe...")
        geo_metrics = monitor.tier1(0)
        torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        n_metrics = len(geo_metrics)
        rankme = geo_metrics.get("geo/rankme_last", "N/A")
        print(f"  tier1 returned {n_metrics} metrics, RankMe={rankme}")
        print(f"\n  Level 3 PASSED ({elapsed:.0f}ms)")
        return True

    except Exception as e:
        print(f"\n  Level 3 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("FULL LOOP PROBE TEST — ESCALATING LEVELS")
    print("=" * 60)

    results = {}

    results["L1"] = test_level1()

    if results["L1"]:
        results["L2"] = test_level2()
    else:
        print("\nLevel 1 failed — the loop itself causes the crash (no full model needed)")
        results["L2"] = None

    if results.get("L2"):
        results["L3"] = test_level3()
    elif results.get("L2") is False:
        print("\nLevel 2 failed — crash requires full model + compile, but NOT FP8")
        results["L3"] = None
    else:
        results["L3"] = None

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for level, passed in results.items():
        status = "PASS" if passed else ("FAIL" if passed is False else "SKIPPED")
        print(f"  {level}: {status}")

    if all(v for v in results.values() if v is not None):
        print("\nAll levels passed — crash requires DDP or full training loop context")
        print("Next: test with torchrun --nproc_per_node=8 and real NCA data")

    return 0 if all(v is not False for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
