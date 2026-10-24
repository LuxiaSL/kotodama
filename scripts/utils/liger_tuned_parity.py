"""LigerRMSNormTuned vs LigerRMSNorm at the training shapes (hidden, q-norm, k-norm), eager and under torch.compile.

Gate (exit 1 on failure): Y within 2 bf16 ulps of Liger's and differing on < 1e-4 of the elements; dX within 64 ulps
and differing on < 1e-4 of the elements (reduction order only); dW error vs an fp64 reference at most 1.05x Liger's.

  tools/run_py.sh scripts/utils/liger_tuned_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.liger_tuned import LigerRMSNormTuned  # noqa: E402

DEVICE = "cuda"


def ulps(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    d = (a.contiguous().view(torch.int16).int() - b.contiguous().view(torch.int16).int()).abs()
    return int(d.max().item()), (d > 0).float().mean().item()


def case(name: str, shape: tuple[int, ...], eps: float = 1e-5) -> bool:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    D = shape[-1]
    g = torch.Generator(device=DEVICE).manual_seed(sum(shape))
    x0 = torch.randn(*shape, device=DEVICE, generator=g).bfloat16()
    gy = torch.randn(*shape, device=DEVICE, generator=g).bfloat16()
    w0 = (1 + 0.1 * torch.randn(D, device=DEVICE, generator=g)).bfloat16()
    xd = x0.double().reshape(-1, D)
    dw_ref = (gy.double().reshape(-1, D) * xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)).sum(0)
    ok = True
    for mode in ("eager", "compiled"):
        # A fresh dynamo cache per case: recompiling Liger's autograd Function for a different-rank input in the same
        # process trips a dynamo assertion (plain LigerRMSNorm too; the whole-model compile is unaffected).
        torch._dynamo.reset()
        outs = {}
        for tag, cls in (("liger", LigerRMSNorm), ("tuned", LigerRMSNormTuned)):
            m = cls(D, eps=eps).to(DEVICE).bfloat16()
            with torch.no_grad():
                m.weight.copy_(w0)
            fn = torch.compile(m) if mode == "compiled" else m
            x = x0.clone().requires_grad_(True)
            y = fn(x)
            gx, gw = torch.autograd.grad(y, (x, m.weight), gy.clone())
            outs[tag] = (y.detach(), gx, gw)
        (y0, gx0, gw0), (y1, gx1, gw1) = outs["liger"], outs["tuned"]
        uy, fy = ulps(y1, y0)
        ux, fx = ulps(gx1, gx0)
        e0 = ((gw0.double() - dw_ref).norm() / dw_ref.norm()).item()
        e1 = ((gw1.double() - dw_ref).norm() / dw_ref.norm()).item()
        good = uy <= 2 and fy < 1e-4 and ux <= 64 and fx < 1e-4 and e1 <= 1.05 * e0
        ok &= good
        print(f"{name:7s} {str(shape):18s} {mode:8s} Y max {uy} ulp ({fy:.1e} differ)  dX max {ux} ulp ({fx:.1e} "
              f"differ)  dW err tuned {e1:.3e} vs liger {e0:.3e}  {'ok' if good else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    ok = case("hidden", (8, 4096, 1536))
    ok &= case("q_norm", (8, 4096, 12, 128))
    ok &= case("k_norm", (8, 4096, 6, 128))
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
