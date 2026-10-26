"""AttnRes Phase-1 micro-bench + parity at a model's per-sublayer routing shapes (default: the tsumugi).

The per-sublayer path calls Phase 1 with one query over the committed block stack; the stack depth S at each call
follows the AttnRes block layout. This times the forward and each backward variant at every S the model meets
(B x T tokens, bf16 sources), weights them by how many calls per forward use that S, and reports ms per
micro-batch (comparable to a profile's per-micro-batch kernel time), effective bandwidth, and parity of each
backward variant against the default (v1) — max |diff| and relative norm for source and query grads next to v1's
own rerun floor.

Forward: the batched kernel (default) and the single-query online-softmax kernel (KOTODAMA_P1_FWD_ONLINE), with
output / lse / rms / logits parity of online vs default next to the default's rerun floor.
Backward variants: v1 (default), v2 (KOTODAMA_P1_BWD_V2: D-chunked, single bf16 round like v1), torch
(KOTODAMA_P1_BWD_TORCH: einsum; sums its terms in bf16).

Run on a GPU host: tools/run_py.sh scripts/benchmark/attnres_p1_bench.py [--model_size 1b-tsumugi] [--mb 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama import presets  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def call_counts(num_layers: int, boundaries: list[int]) -> Counter[int]:
    """Phase-1 calls per forward by stack depth S (per-sublayer path: layer 0's pre-attn and S=0 skip)."""
    starts = list(boundaries) + [num_layers]
    c: Counter[int] = Counter()
    for b, (s, e) in enumerate(zip(starts[:-1], starts[1:])):
        if b > 0:
            c[b] += 1                    # boundary layer's pre-attn routes over the previous block's stack
        c[b + 1] += (e - s - 1) + (e - s)  # in-block pre-attn (non-boundary) + every pre-mlp
    c[len(boundaries)] += 1              # final routing
    return c


def set_variant(name: str) -> None:
    p1ops._P1_BWD_V2 = name == "v2"
    p1ops._P1_BWD_TORCH = name == "torch"
    p1ops._P1_BWD_TORCH_MIN_S = 1


def timed(fn, iters: int) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--variants", default="v1,v2,torch")
    a = ap.parse_args()
    t_start = time.perf_counter()

    def note(msg: str) -> None:
        print(f"[{time.perf_counter() - t_start:7.1f}s] {msg}", flush=True)

    shape = presets.shape(a.model_size)
    D = shape["hidden_size"]
    counts = call_counts(shape["num_layers"], list(presets.DEFAULT_BOUNDARIES[a.model_size]))
    note(f"{a.model_size}: D={D} B={a.mb} T={a.seq}; calls/forward by S: {dict(sorted(counts.items()))} "
         f"(total {sum(counts.values())})")
    variants = a.variants.split(",")
    g = torch.Generator(device=DEVICE).manual_seed(0)
    tot_fwd = 0.0
    tot_fwd_online = 0.0
    tot_bwd: dict[str, float] = {v: 0.0 for v in variants}
    for S in sorted(counts):
        n = counts[S]
        src0 = torch.randn(S, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        q0 = (torch.randn(1, D, device=DEVICE, generator=g) * 0.5).bfloat16()
        gout = torch.randn(1, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        src_bytes = src0.numel() * 2

        def fwd():
            return p1ops.phase_1_batched_attention_triton_op(src0, q0, EPS, num_active=S)

        def fwd_aux():
            return p1ops._phase_1_batched_attention_forward_with_aux_triton_op(src0, q0, EPS, S)

        t_f = timed(fwd, a.iters)
        tot_fwd += n * t_f
        ref_aux, rerun_aux = fwd_aux(), fwd_aux()
        p1ops._P1_FWD_ONLINE = True
        t_on = timed(fwd, a.iters)
        on_aux = fwd_aux()
        p1ops._P1_FWD_ONLINE = False
        tot_fwd_online += n * t_on
        grads: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        line = [f"S={S} x{n:<3d} fwd {t_f:6.3f} ms ({src_bytes / t_f / 1e9:5.2f} TB/s)",
                f"online {t_on:6.3f} ms ({src_bytes / t_on / 1e9:5.2f} TB/s)"]
        fmsg = []
        for i, nm in ((0, "out"), (1, "lse"), (2, "irms"), (3, "logit")):
            r, o, f = ref_aux[i][..., :S] if i >= 2 else ref_aux[i], on_aux[i][..., :S] if i >= 2 else on_aux[i], \
                rerun_aux[i][..., :S] if i >= 2 else rerun_aux[i]
            fmsg.append(f"{nm} max|Δ| {(o.float() - r.float()).abs().max().item():.1e} "
                        f"(floor {(f.float() - r.float()).abs().max().item():.1e}, "
                        f"bitwise {torch.equal(o, r)})")
        for v in variants:
            set_variant(v)
            src = src0.clone().requires_grad_(True)
            q = q0.clone().requires_grad_(True)

            def fb():
                out, _lse = p1ops.phase_1_batched_attention_triton_op(src, q, EPS, num_active=S)
                return torch.autograd.grad(out, (src, q), grad_outputs=gout)

            t_fb = timed(fb, a.iters)
            t_b = t_fb - t_f
            tot_bwd[v] += n * t_b
            grads[v] = fb()
            if v == "v1":
                grads["v1-rerun"] = fb()
            line.append(f"{v} bwd {t_b:6.3f}")
        set_variant("v1")
        note(" | ".join(line))
        note("    fwd online vs batched: " + "; ".join(fmsg))
        ref = grads["v1"]
        for v in ["v1-rerun", *[x for x in variants if x != "v1"]]:
            if v not in grads:
                continue
            msgs = []
            for i, nm in ((0, "dsrc"), (1, "dq")):
                d = (grads[v][i].float() - ref[i].float())
                msgs.append(f"{nm} max|Δ| {d.abs().max().item():.2e} rel {d.norm().item() / ref[i].float().norm().clamp_min(1e-30).item():.2e}")
            note(f"    parity {v:9s} vs v1: " + ", ".join(msgs))
        del src0, gout, grads, ref_aux, rerun_aux, on_aux
        torch.cuda.empty_cache()
    note(f"PER MICRO-BATCH (weighted by calls/forward): fwd {tot_fwd:.1f} ms, fwd online {tot_fwd_online:.1f} ms; "
         + "; ".join(f"bwd[{v}] {t:.1f} ms" for v, t in tot_bwd.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
