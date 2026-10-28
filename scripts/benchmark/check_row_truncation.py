#!/usr/bin/env python3
"""The row decoders' top-p / min-p truncation compiled at batched decode shapes (GPU): it must compile (a whole-row
cumsum over the vocabulary does not at these shapes) and match the eager truncation, which keeps exactly the tokens
an eager top-p + min-p filter keeps.

    tools/run_py.sh scripts/benchmark/check_row_truncation.py [--rows 16,128] [--mode max-autotune]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.multi_decode import MultiSampleDecoder  # noqa: E402


def reference(scaled: torch.Tensor, top_p: torch.Tensor, log_min_p: torch.Tensor) -> torch.Tensor:
    """The filter written plainly (whole-row cumsum, eager)."""
    sl, si = torch.sort(scaled, dim=-1, descending=True)
    probs = sl.softmax(-1)
    sl = torch.where(probs.cumsum(-1) - probs >= top_p, torch.full_like(sl, float("-inf")), sl)
    sl = torch.where(sl < sl[:, :1] + log_min_p, torch.full_like(sl, float("-inf")), sl)
    return torch.full_like(scaled, float("-inf")).scatter(1, si, sl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="16,128")
    ap.add_argument("--vocab", type=int, default=49152)
    ap.add_argument("--mode", default="max-autotune")
    a = ap.parse_args()
    ok = True
    fn = torch.compile(MultiSampleDecoder._truncate, mode=a.mode, fullgraph=True, dynamic=False)
    for B in (int(x) for x in a.rows.split(",")):
        g = torch.Generator(device="cuda").manual_seed(B)
        scaled = 4 * torch.randn((B, a.vocab), device="cuda", generator=g)
        top_p = torch.tensor([[0.9], [2.0], [0.5], [0.99]], device="cuda").repeat(B // 4 + 1, 1)[:B]
        log_min_p = torch.log(torch.tensor([[0.05], [1e-9], [0.2], [0.01]], device="cuda")).repeat(B // 4 + 1, 1)[:B]
        t0 = time.perf_counter()
        got = fn(scaled, top_p, log_min_p)
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        want = reference(scaled, top_p, log_min_p)
        kept_got, kept_want = torch.isfinite(got), torch.isfinite(want)
        agree = (kept_got == kept_want).float().mean().item()
        values = torch.equal(torch.where(kept_got, got, 0), torch.where(kept_want, want, 0))
        t1 = time.perf_counter()
        for _ in range(20):
            fn(scaled, top_p, log_min_p)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t1) / 20 * 1000
        row_ok = agree > 0.99999 and kept_got.sum(-1).min().item() >= 1
        ok &= row_ok
        print(f"rows={B}: compiled in {compile_s:.1f}s, kept-set agreement {agree:.6f}, kept values equal {values}, "
              f"{ms:.2f} ms/call, kept per row {kept_got.sum(-1).tolist()[:4]}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
