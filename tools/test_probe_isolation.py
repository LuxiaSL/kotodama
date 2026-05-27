#!/usr/bin/env python3
"""Isolate the geo monitor CUDA crash: AttnRes Triton kernel output → LigerRMSNorm.

Tests the exact interaction that crashes _probe_forward_attn_res at line 746:
    attn_out = layer.attn(layer.attn_norm(h), rope_cos, rope_sin)
where h comes from _route_static (Phase 1 Triton kernel) and attn_norm is LigerRMSNorm.

Each test isolates one variable. Run on gpu-host with venv activated.
"""

import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# Match production: D=3072, non-power-of-2
D = 3072
B = 4          # small batch for testing
T = 512        # short seq to stay in memory
EPS = 1e-5
NUM_SOURCES = 8  # max_sources for DD-3B (7 boundaries + partial)
NUM_ACTIVE = 3   # typical mid-training active count


def make_test_tensors(device="cuda", dtype=torch.bfloat16):
    """Create tensors matching the probe forward shapes."""
    buf = torch.randn(NUM_SOURCES, B, T, D, device=device, dtype=dtype)
    qw = torch.randn(1, D, device=device, dtype=torch.float32)  # query * norm.weight, always fp32
    return buf, qw


def route_static_triton(buf, qw, eps, num_active):
    """_route_static Triton path — the exact code from llama.py."""
    from src.model.flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
    out, _lse = phase_1_batched_attention_triton_op(buf, qw, eps, num_active=num_active)
    return out[0]


def route_static_pytorch(buf, qw, eps, num_active):
    """_route_static PyTorch fallback path."""
    active_mask = torch.zeros(buf.shape[0], dtype=torch.bool, device=buf.device)
    active_mask[:num_active] = True
    rsqrt = torch.rsqrt(buf.pow(2).mean(-1) + eps)
    logits = (buf * qw).sum(-1) * rsqrt
    logits = logits.masked_fill(~active_mask.view(-1, 1, 1), float("-inf"))
    weights = F.softmax(logits, dim=0)
    return (weights.unsqueeze(-1) * buf).sum(0)


def make_liger_norm(dim, eps):
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    return LigerRMSNorm(dim, eps=eps).cuda()


def make_custom_norm(dim, eps):
    """Custom RMSNorm (pure PyTorch, no Triton)."""
    class RMSNorm(nn.Module):
        def __init__(self, d, e):
            super().__init__()
            self.eps = e
            self.weight = nn.Parameter(torch.ones(d))
        def forward(self, x):
            norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * norm * self.weight
    return RMSNorm(dim, eps).cuda()


def run_test(name, route_fn, norm_fn, pre_norm_transform=None, sync_before_norm=False):
    """Run one isolation test. Returns (passed: bool, error: str|None, time_ms: float)."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

        buf, qw = make_test_tensors()
        norm = norm_fn(D, EPS)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = route_fn(buf, qw, EPS, NUM_ACTIVE)

            if sync_before_norm:
                torch.cuda.synchronize()

            if pre_norm_transform is not None:
                h = pre_norm_transform(h)

            print(f"  h shape={h.shape}, dtype={h.dtype}, contiguous={h.is_contiguous()}, "
                  f"stride={h.stride()}, device={h.device}")
            print(f"  h range: [{h.float().min().item():.4f}, {h.float().max().item():.4f}], "
                  f"has_nan={h.isnan().any().item()}, has_inf={h.isinf().any().item()}")

            result = norm(h)
            torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        print(f"  result shape={result.shape}, dtype={result.dtype}")
        print(f"  PASSED ({elapsed:.1f}ms)")
        return True, None, elapsed

    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        err = traceback.format_exc()
        print(f"  FAILED ({elapsed:.1f}ms): {e}")
        print(f"  {err}")
        return False, str(e), elapsed


def run_test_compiled_then_probe(name, route_fn, norm_fn):
    """Simulate training: compile + forward, then probe (uncompiled)."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

        norm = norm_fn(D, EPS)

        # Phase A: simulate compiled training forward
        # Create a simple model that uses the norm, compile it, run a forward
        class TinyBlock(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.norm = n
                self.proj = nn.Linear(D, D, bias=False)

            def forward(self, x):
                return self.proj(self.norm(x))

        block = TinyBlock(norm).cuda()
        compiled_block = torch.compile(block)

        x_train = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = compiled_block(x_train)
            loss = compiled_block(x_train).sum()
            loss.backward()
        torch.cuda.synchronize()
        print("  Phase A (compiled forward+backward): OK")

        # Phase B: simulate probe forward (uncompiled, no_grad)
        buf, qw = make_test_tensors()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = route_fn(buf, qw, EPS, NUM_ACTIVE)
            # Use the ORIGINAL norm (unwrapped from compile), same as geo monitor
            result = norm(h)
            torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        print(f"  Phase B (probe): result shape={result.shape}")
        print(f"  PASSED ({elapsed:.1f}ms)")
        return True, None, elapsed

    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        err = traceback.format_exc()
        print(f"  FAILED ({elapsed:.1f}ms): {e}")
        print(f"  {err}")
        return False, str(e), elapsed


def run_test_fp8_then_probe(name, route_fn, norm_fn):
    """Simulate FP8 training: convert to Float8Linear, compile, forward, then probe."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.time()

        norm = norm_fn(D, EPS)

        class TinyBlock(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.norm = n
                self.proj = nn.Linear(D, D, bias=False)

            def forward(self, x):
                return self.proj(self.norm(x))

        block = TinyBlock(norm).cuda()

        # FP8 conversion
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            convert_to_float8_training(block, config=Float8LinearConfig())
            print("  FP8 conversion: OK")
        except ImportError:
            print("  FP8 not available, skipping")
            return True, "skipped (no torchao)", 0

        compiled_block = torch.compile(block)

        x_train = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _ = compiled_block(x_train)
            loss = compiled_block(x_train).sum()
            loss.backward()
        torch.cuda.synchronize()
        print("  Phase A (FP8 compiled forward+backward): OK")

        # Phase B: probe
        buf, qw = make_test_tensors()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = route_fn(buf, qw, EPS, NUM_ACTIVE)
            result = norm(h)
            torch.cuda.synchronize()

        elapsed = (time.time() - t0) * 1000
        print(f"  Phase B (probe after FP8): result shape={result.shape}")
        print(f"  PASSED ({elapsed:.1f}ms)")
        return True, None, elapsed

    except Exception as e:
        elapsed = (time.time() - t0) * 1000
        err = traceback.format_exc()
        print(f"  FAILED ({elapsed:.1f}ms): {e}")
        print(f"  {err}")
        return False, str(e), elapsed


def run_test_num_active_sweep(route_fn, norm_fn):
    """Test all num_active values (1 through NUM_SOURCES) — num_active=1 is the layer-0 case."""
    print(f"\n{'='*60}")
    print(f"TEST: num_active sweep (1..{NUM_SOURCES})")
    print(f"{'='*60}")

    results = {}
    for na in range(1, NUM_SOURCES + 1):
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            buf, qw = make_test_tensors()
            norm = norm_fn(D, EPS)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = route_fn(buf, qw, EPS, na)
                result = norm(h)
                torch.cuda.synchronize()
            results[na] = "PASS"
            print(f"  num_active={na}: PASS")
        except Exception as e:
            results[na] = f"FAIL: {e}"
            print(f"  num_active={na}: FAIL — {e}")

    all_pass = all(v == "PASS" for v in results.values())
    return all_pass, results, 0


def main():
    print("=" * 60)
    print("PROBE ISOLATION TEST SUITE")
    print(f"D={D}, B={B}, T={T}, NUM_SOURCES={NUM_SOURCES}, NUM_ACTIVE={NUM_ACTIVE}")
    print(f"CUDA: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    # Ensure we can import our kernels
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results = []

    # ── Test 1: Exact crash path — Triton AttnRes → LigerRMSNorm ──
    results.append(("T1: Triton→Liger (crash path)",
                     *run_test("Triton AttnRes output → LigerRMSNorm",
                               route_static_triton, make_liger_norm)))

    # ── Test 2: PyTorch routing → LigerRMSNorm ──
    results.append(("T2: PyTorch→Liger",
                     *run_test("PyTorch routing → LigerRMSNorm",
                               route_static_pytorch, make_liger_norm)))

    # ── Test 3: Triton AttnRes → Custom RMSNorm (no Liger) ──
    results.append(("T3: Triton→Custom",
                     *run_test("Triton AttnRes output → Custom RMSNorm",
                               route_static_triton, make_custom_norm)))

    # ── Test 4: Triton AttnRes → cuda.synchronize() → LigerRMSNorm ──
    results.append(("T4: Triton→sync→Liger",
                     *run_test("Triton AttnRes → sync → LigerRMSNorm",
                               route_static_triton, make_liger_norm,
                               sync_before_norm=True)))

    # ── Test 5: Triton AttnRes → .contiguous().clone() → LigerRMSNorm ──
    results.append(("T5: Triton→clone→Liger",
                     *run_test("Triton AttnRes → clone → LigerRMSNorm",
                               route_static_triton, make_liger_norm,
                               pre_norm_transform=lambda h: h.contiguous().clone())))

    # ── Test 6: Compiled training forward then probe (Triton→Liger) ──
    results.append(("T6: compile+fwd→probe(Triton→Liger)",
                     *run_test_compiled_then_probe(
                         "Compiled training → Triton probe → LigerRMSNorm",
                         route_static_triton, make_liger_norm)))

    # ── Test 7: FP8 + compiled training then probe ──
    results.append(("T7: FP8+compile→probe(Triton→Liger)",
                     *run_test_fp8_then_probe(
                         "FP8+Compiled training → Triton probe → LigerRMSNorm",
                         route_static_triton, make_liger_norm)))

    # ── Test 8: num_active sweep (1..8) with Triton→Liger ──
    results.append(("T8: num_active sweep",
                     *run_test_num_active_sweep(route_static_triton, make_liger_norm)))

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed, error, elapsed in results:
        status = "PASS" if passed else "FAIL"
        detail = f" — {error}" if error and not passed else ""
        print(f"  [{status}] {name}{detail}")

    failures = [r for r in results if not r[1]]
    if failures:
        print(f"\n{len(failures)} FAILED — see details above")
        print("\nDiagnostic interpretation:")
        t1 = next((r for r in results if r[0].startswith("T1")), None)
        t2 = next((r for r in results if r[0].startswith("T2")), None)
        t3 = next((r for r in results if r[0].startswith("T3")), None)
        t4 = next((r for r in results if r[0].startswith("T4")), None)
        t5 = next((r for r in results if r[0].startswith("T5")), None)
        t6 = next((r for r in results if r[0].startswith("T6")), None)
        t7 = next((r for r in results if r[0].startswith("T7")), None)

        if t1 and not t1[1]:  # T1 fails
            if t2 and t2[1]:
                print("  → PyTorch routing works, Triton routing doesn't: issue is in AttnRes kernel output")
            if t3 and t3[1]:
                print("  → Custom RMSNorm works: issue is specifically Liger's Triton kernel reading AttnRes output")
            if t4 and t4[1]:
                print("  → sync doesn't help: NOT a stream ordering issue")
            elif t4 and not t4[1]:
                print("  → sync FIXES it: stream ordering issue between the two Triton kernels")
            if t5 and t5[1]:
                print("  → clone FIXES it: AttnRes output tensor has bad metadata/layout")
            elif t5 and not t5[1]:
                print("  → clone doesn't fix it: tensor data itself may be corrupted")
            if t6 and not t6[1] and t1 and t1[1]:
                print("  → only fails after compiled training: torch.compile leaves stale state")
            if t7 and not t7[1] and t6 and t6[1]:
                print("  → only fails with FP8: Float8Linear state interferes with probe")
    else:
        print("\nAll tests PASSED — crash may require full model context to reproduce")
        print("Next steps: test with full 3B model + DDP + real data")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
