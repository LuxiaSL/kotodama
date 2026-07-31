"""MXFP8Linear smoke: numerics vs bf16 reference, compile check, micro-bench.

Tier-C gate stage 1 (SPEC-7B §5.1) for the C2 lever. Tolerances are
DOCUMENTED expectations for an 8-bit format, not exactness claims: fwd
output and input grads should land within a few percent relative error of
bf16 on well-conditioned random inputs; weight grads slightly looser.
The real gates are the canary loss overlay and the proxy soak — this
script only proves the kernels run, differentiate, compile, and are in
the right numerical ballpark on B200.

Run on gpu-host (throwaway venv!):
  KOTODAMA_VENV=/models/kotodama-data/venv-mxfp8 tools/run_py.sh \
      scripts/utils/mxfp8_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.mxfp8 import MXFP8_AVAILABLE, MXFP8Linear, convert_to_mxfp8_training  # noqa: E402

DEVICE = "cuda"
# Production 7B GEMM shapes: attn qkv/o (4096x4096-class) and MLP (4096<->14336)
SHAPES = [(4096, 4096), (4096, 14336), (14336, 4096)]
TOKENS = 16384  # MB=4 x T=4096


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def main() -> int:
    if not MXFP8_AVAILABLE:
        print("FAIL: torchao mx_formats not importable in this venv")
        return 1
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}")
    all_ok = True

    for din, dout in SHAPES:
        g = torch.Generator(device=DEVICE).manual_seed(0)
        lin = torch.nn.Linear(din, dout, bias=False, device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(dout, din, device=DEVICE, generator=g) * din**-0.5)
        x = torch.randn(TOKENS, din, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)

        # bf16 reference fwd/bwd
        r_loss = torch.randn(TOKENS, dout, device=DEVICE, generator=g, dtype=torch.bfloat16)
        xr = x.clone().requires_grad_(True)
        ref = lin(xr)
        (ref * r_loss).sum().backward()
        gref_x, gref_w = xr.grad.clone(), lin.weight.grad.clone()
        lin.weight.grad = None

        mx = MXFP8Linear.from_float(lin)
        xm = x.clone().requires_grad_(True)
        out = mx(xm)
        (out * r_loss).sum().backward()

        e_out = rel_err(out, ref)
        e_gx = rel_err(xm.grad, gref_x)
        e_gw = rel_err(mx.weight.grad, gref_w)
        all_ok &= check(f"{din}x{dout} fwd", e_out < 5e-2, f"relnorm {e_out:.3e}")
        all_ok &= check(f"{din}x{dout} dgrad", e_gx < 5e-2, f"relnorm {e_gx:.3e}")
        all_ok &= check(f"{din}x{dout} wgrad", e_gw < 8e-2, f"relnorm {e_gw:.3e}")
        mx.weight.grad = None

    # converter scope check on a toy transformer-ish module
    toy = torch.nn.Sequential(
        torch.nn.Linear(4096, 14336, bias=False),
        torch.nn.Linear(14336, 4096, bias=False),
        torch.nn.Linear(100, 64, bias=False),  # not 32-divisible -> skipped
    ).to(DEVICE, torch.bfloat16)
    n = convert_to_mxfp8_training(toy)
    all_ok &= check("converter swaps eligible only", n == 2,
                    f"swapped {n}, [2] is {type(toy[2]).__name__}")

    # compile check: MXFP8Linear inside a compiled fn, fwd+bwd
    lin = torch.nn.Linear(4096, 4096, bias=False, device=DEVICE, dtype=torch.bfloat16)
    mx = MXFP8Linear.from_float(lin)

    r_c = torch.randn(TOKENS, 4096, device=DEVICE, dtype=torch.bfloat16)

    def f(x):
        return (mx(x) * r_c).mean()

    # VALUES under compile, not just execution (fa4 NaN lesson 2026-07-09):
    # compiled loss/grads must match eager-MXFP8 and be finite.
    res = {}
    for mode, fn in [("eager", f), ("compiled", torch.compile(f))]:
        x = torch.randn(TOKENS, 4096, device=DEVICE, dtype=torch.bfloat16)
        torch.manual_seed(3)
        xc = x.zero_().normal_().requires_grad_(True)
        try:
            loss = fn(xc)
            loss.backward()
            res[mode] = (loss.detach(), xc.grad, mx.weight.grad.clone())
            mx.weight.grad = None
        except Exception as exc:  # noqa: BLE001
            all_ok &= check(f"torch.compile fwd+bwd ({mode})", False, repr(exc)[:200])
            res[mode] = None
    if res.get("eager") is not None and res.get("compiled") is not None:
        finite = all(t.isfinite().all() for t in res["compiled"])
        e_l = rel_err(res["compiled"][0], res["eager"][0])
        e_gx = rel_err(res["compiled"][1], res["eager"][1])
        e_gw = rel_err(res["compiled"][2], res["eager"][2])
        all_ok &= check("compiled finite + ~eager", finite and max(e_l, e_gx, e_gw) < 2e-2,
                        f"finite={finite} loss {e_l:.2e} dgrad {e_gx:.2e} wgrad {e_gw:.2e}")

    # micro-bench vs bf16 and vs tensorwise Float8Linear (fwd+bwd walltime)
    from torchao.float8 import convert_to_float8_training

    print("\n=== micro-bench (fwd+bwd, 50 iters, 4096x14336 @ 16K tokens) ===")
    for name, factory in [
        ("bf16", lambda m: m),
        ("tensorwise", lambda m: (convert_to_float8_training(m), m)[1]),
        ("mxfp8", lambda m: (convert_to_mxfp8_training(m), m)[1]),
    ]:
        base = torch.nn.Sequential(
            torch.nn.Linear(4096, 14336, bias=False),
            torch.nn.Linear(14336, 4096, bias=False),
        ).to(DEVICE, torch.bfloat16)
        mod = factory(base)
        x = torch.randn(TOKENS, 4096, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        r_b = torch.randn(TOKENS, 4096, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(10):
            (mod(x) * r_b).sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            (mod(x) * r_b).sum().backward()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1e3
        print(f"  {name:11s}: {ms:7.3f} ms/iter")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
