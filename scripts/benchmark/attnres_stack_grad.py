"""One AttnRes block's routing, fwd + bwd: k per-sublayer routes (Phase 1 over a shared committed stack + Phase 2
merge with a partial) reading one stack, as in `llama.py _route_p12`. Times the backward (where the stack gradient
is produced per call and summed across the k consumers) eagerly and inside torch.compile, and saves the gradients
so two source trees can be compared bitwise.

  tools/run_py.sh scripts/benchmark/attnres_stack_grad.py --save /dev/shm/x/new.pt          # time + save grads
  tools/run_py.sh scripts/benchmark/attnres_stack_grad.py --compare /dev/shm/x/old.pt /dev/shm/x/new.pt

`--src <tree>/src` imports the kernels from another checkout (the reference). Default (S, k) pairs are the moonshot
Shape A blocks: stack depth S with k routes per block.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

DEVICE = "cuda"
EPS = torch.finfo(torch.float32).eps
DEFAULT_CASES = "2:12,4:16,6:40,8:26"


def parse_cases(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in text.split(","):
        s, k = item.split(":")
        out.append((int(s), int(k)))
    return out


def make_inputs(S: int, k: int, mb: int, seq: int, D: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return {
        "stack": torch.randn(S, mb, seq, D, device=DEVICE, generator=g).bfloat16(),
        "queries": (torch.randn(k, D, device=DEVICE, generator=g) * 0.5).bfloat16(),
        "partials": torch.randn(k, mb, seq, D, device=DEVICE, generator=g).bfloat16(),
        "gouts": torch.randn(k, mb, seq, D, device=DEVICE, generator=g).bfloat16(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[2] / "src"))
    ap.add_argument("--cases", default=DEFAULT_CASES, help="S:k pairs")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--modes", default="eager,compiled")
    ap.add_argument("--save", default="")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--ref64", action="store_true",
                    help="also an fp64 torch reference; report each mode's grad_stack error vs it")
    ap.add_argument("--kernels", action="store_true", help="per-kernel device time of one backward (torch.profiler)")
    ap.add_argument("--pin", action="store_true", help="first autotune config only (bitwise comparisons)")
    ap.add_argument("--exact", default="grad_stack", help="key substring that must match bitwise; others report only")
    a = ap.parse_args()

    if a.compare:
        ref, new = (torch.load(p, map_location="cpu") for p in a.compare)
        ok = True
        for key in sorted(ref):
            if key not in new:
                print(f"MISSING {key}", flush=True)
                ok = False
                continue
            same = torch.equal(ref[key], new[key])
            r, n = ref[key].float(), new[key].float()
            diff = (r - n).abs().max().item()
            rel = ((r - n).norm() / r.norm().clamp_min(1e-30)).item()
            gate = a.exact in key
            print(f"{'BITWISE' if same else 'DIFFER '} {key:40s} max|d|={diff:.3e} rel={rel:.2e}"
                  f"{'' if gate else '  (not gated)'}", flush=True)
            ok &= same or not gate
        print("PASS (bitwise)" if ok else "FAIL", flush=True)
        return 0 if ok else 1

    sys.path.insert(0, a.src)
    from kotodama.model.flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
    from kotodama.model.flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op
    import kotodama.model.flash_attn_res.ops.phase_1 as p1mod

    print(f"kernels from {p1mod.__file__}", flush=True)
    if a.pin:
        # One config per autotuned kernel: num_warps changes the tl.sum reduction tree, so a run-to-run autotune pick
        # difference alone breaks bitwise equality between two trees.
        import triton
        from kotodama.model.flash_attn_res.kernels import phase_1 as k1, phase_2 as k2, reduce as kr
        for mod in (k1, k2, kr):
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, triton.runtime.autotuner.Autotuner):
                    obj.configs = obj.configs[:1]
                    obj.cache.clear()
                    print(f"pinned {name}: {obj.configs[0]}", flush=True)

    def block_loss(stack, queries, partials, gouts):
        loss = stack.new_zeros((), dtype=torch.float32)
        S = stack.shape[0]
        for i in range(queries.shape[0]):
            qw = queries[i]
            p1_out, p1_lse = phase_1_batched_attention_triton_op(stack, qw.unsqueeze(0), EPS, num_active=S)
            h = phase_2_online_softmax_merge_triton_op(partials[i], qw, p1_out[0], p1_lse[0], EPS)
            loss = loss + (h.float() * gouts[i].float()).sum()
        return loss

    def block_loss_ref64(stack, queries, partials, gouts):
        # llama.py _route_p12's torch fallback, in fp64.
        loss = stack.new_zeros(())
        rsqrt_c = torch.rsqrt(stack.pow(2).mean(-1) + EPS)
        for i in range(queries.shape[0]):
            qw, partial = queries[i], partials[i]
            logits_c = (stack * qw).sum(-1) * rsqrt_c
            rsqrt_p = torch.rsqrt(partial.pow(2).mean(-1, keepdim=True) + EPS)
            logit_p = ((partial * qw).sum(-1, keepdim=True) * rsqrt_p).squeeze(-1)
            weights = torch.softmax(torch.cat([logits_c, logit_p.unsqueeze(0)], dim=0), dim=0)
            h = (weights.unsqueeze(-1) * torch.cat([stack, partial.unsqueeze(0)], dim=0)).sum(0)
            loss = loss + (h * gouts[i]).sum()
        return loss

    compiled_loss = torch.compile(block_loss, fullgraph=True, dynamic=False)
    saved: dict[str, torch.Tensor] = {}
    modes = a.modes.split(",")
    for S, k in parse_cases(a.cases):
        x = make_inputs(S, k, a.mb, a.seq, a.hidden, seed=1000 * S + k)
        ref_gs = None
        if a.ref64:
            st64 = x["stack"].double().requires_grad_(True)
            l64 = block_loss_ref64(st64, x["queries"].double(), x["partials"].double(), x["gouts"].double())
            (ref_gs,) = torch.autograd.grad(l64, (st64,))
            del st64, l64
            torch.cuda.empty_cache()
        for mode in modes:
            fn = compiled_loss if mode == "compiled" else block_loss

            def run():
                stack = x["stack"].clone().requires_grad_(True)
                queries = x["queries"].clone().requires_grad_(True)
                loss = fn(stack, queries, x["partials"], x["gouts"])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gs, gq = torch.autograd.grad(loss, (stack, queries))
                torch.cuda.synchronize()
                return gs, gq, time.perf_counter() - t0

            t_c = time.perf_counter()
            gs, gq, _ = run()  # warm-up (autotune / compile)
            warm = time.perf_counter() - t_c
            times = [run()[2] for _ in range(a.iters)]
            if a.kernels:
                from collections import defaultdict
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    run()
                per: dict[str, list[float]] = defaultdict(list)
                for ev in prof.events():
                    if ev.device_type == torch.autograd.DeviceType.CUDA:
                        per[ev.name].append(ev.device_time)
                for name, ds in sorted(per.items(), key=lambda kv: -sum(kv[1]))[:8]:
                    print(f"    {sum(ds) / 1e3:8.2f} ms  x{len(ds):<4d} {sum(ds) / len(ds):8.1f} us/call  {name[:90]}",
                          flush=True)
            times.sort()
            med = times[len(times) // 2] * 1e3
            err = ""
            if ref_gs is not None:
                d = gs.double() - ref_gs
                err = (f"  grad_stack vs fp64: rel {(d.norm() / ref_gs.norm()).item():.3e}"
                       f" max {d.abs().max().item():.3e}")
            print(f"S={S} k={k:3d} {mode:8s} bwd median {med:8.2f} ms "
                  f"({med / k:6.3f} ms/route)  warm-up {warm:6.1f} s{err}", flush=True)
            saved[f"S{S}_k{k}_{mode}_grad_stack"] = gs.cpu()
            saved[f"S{S}_k{k}_{mode}_grad_queries"] = gq.cpu()
            del gs, gq
        del x
        torch.cuda.empty_cache()
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, a.save)
        print(f"saved {len(saved)} tensors -> {a.save}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
