"""KOTODAMA_ATTNRES_EPS_FP64_COMPAT parity: the forward route kernels under torch.compile, three ways.

The forward kernels take the rms epsilon as a constexpr (fp32 under any launcher) unless EPS_RUNTIME selects the
runtime `eps`, which inductor types as fp64 inside a compiled graph — the behaviour of these kernels before the
constexpr existed. Gates, at the given stack depths on the given shape:
  (1) compat (EPS_RUNTIME) compiled == the pre-constexpr kernels compiled, bitwise — loaded from a directory holding
      that revision's kernels/{phase_1,phase_2,configs}.py and launched directly inside torch.compile;
  (2) default compiled == default eager: Phase-1 outputs bitwise, the Phase-2 bf16 merge within one bf16 ulp at
      the tensor's scale (the two launchers may round its fp32 value differently, e.g. FMA contraction or config choice);
  (3) compat compiled != default compiled (the fp64 path is really taken), reported with max |diff|.
Phase-1 forward outputs (out, lse, rms, logits) and the Phase-2 merged output are all compared; times reported.

Run on a GPU host: tools/run_py.sh scripts/utils/attnres_eps_compat_parity.py --old_kernels DIR [--depths 4,8] [--mb 4]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_2 as p2ops  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def load_old_kernels(directory: Path) -> tuple[types.ModuleType, types.ModuleType]:
    pkg = types.ModuleType("attnres_old_kernels")
    pkg.__path__ = [str(directory)]
    sys.modules["attnres_old_kernels"] = pkg
    mods = {}
    for name in ("configs", "phase_1", "phase_2"):
        spec = importlib.util.spec_from_file_location(f"attnres_old_kernels.{name}", directory / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["phase_1"], mods["phase_2"]


def old_route(k1: types.ModuleType, k2: types.ModuleType, S: int):
    def route(src: torch.Tensor, q: torch.Tensor, partial: torch.Tensor):
        nsb, B, T, D = src.shape
        BT = B * T
        out = torch.empty((1, B, T, D), device=src.device, dtype=torch.bfloat16)
        lse = torch.empty((1, B, T), device=src.device, dtype=torch.float32)
        irms = torch.empty((B, T, nsb), device=src.device, dtype=torch.float32)
        logits = torch.empty((1, B, T, nsb), device=src.device, dtype=torch.float32)
        k1.phase_1_batched_attention_forward_kernel[(BT,)](
            src, q, out, lse, irms, logits, EPS, S, nsb, BT, D, 1,
            triton.next_power_of_2(nsb), triton.next_power_of_2(D))
        merged = torch.empty_like(partial)
        p2_logit = torch.empty((B, T), device=src.device, dtype=torch.float32)
        p2_irms = torch.empty((B, T), device=src.device, dtype=torch.float32)
        k2.phase_2_online_softmax_merge_forward_kernel[(BT,)](
            partial, q[0], out[0], lse[0], merged, p2_logit, p2_irms, EPS, D, triton.next_power_of_2(D))
        return out, lse, irms[..., :S], logits[..., :S], merged
    return route


def new_route(S: int):
    def route(src: torch.Tensor, q: torch.Tensor, partial: torch.Tensor):
        out, lse, irms, logits = p1ops._phase_1_batched_attention_forward_with_aux_triton_op(src, q, EPS, S, p1ops.EPS_FP64_COMPAT)
        merged = p2ops.phase_2_online_softmax_merge_triton_op(partial, q[0], out[0], lse[0], EPS)
        return out, lse, irms[..., :S], logits[..., :S], merged
    return route


def set_compat(on: bool) -> None:
    p1ops.EPS_FP64_COMPAT = on
    p2ops.EPS_FP64_COMPAT = on


def timed(fn, *args, iters: int = 10) -> float:
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn(*args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def compare(x: tuple, y: tuple) -> tuple[bool, float]:
    same = all(torch.equal(a, b) for a, b in zip(x, y))
    diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(x, y))
    return same, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_kernels", type=Path, required=True)
    ap.add_argument("--depths", default="4,8")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1536)
    a = ap.parse_args()
    k1, k2 = load_old_kernels(a.old_kernels)
    ok = True
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for S in (int(x) for x in a.depths.split(",")):
        src = torch.randn(S, a.mb, a.seq, a.hidden, device=DEVICE, generator=g).bfloat16()
        q = (torch.randn(1, a.hidden, device=DEVICE, generator=g) * 0.5).bfloat16()
        partial = torch.randn(a.mb, a.seq, a.hidden, device=DEVICE, generator=g).bfloat16()
        args = (src, q, partial)
        torch._dynamo.reset()
        old_c = torch.compile(old_route(k1, k2, S), fullgraph=True)
        set_compat(True)
        compat_c = torch.compile(new_route(S), fullgraph=True)
        r_old, r_compat = old_c(*args), compat_c(*args)
        t_old, t_compat = timed(old_c, *args), timed(compat_c, *args)
        torch._dynamo.reset()
        set_compat(False)
        default_c = torch.compile(new_route(S), fullgraph=True)
        r_def, r_eager = default_c(*args), new_route(S)(*args)
        t_def, t_eager = timed(default_c, *args), timed(new_route(S), *args)
        names = ("p1_out", "lse", "rms", "logits", "p2_merged")
        print("    default compiled vs eager per output: " + ", ".join(
            f"{n} {'=' if torch.equal(x, y) else f'max|Δ| {(x.float() - y.float()).abs().max().item():.1e}'}"
            for n, x, y in zip(names, r_def, r_eager)), flush=True)
        from kotodama.model.flash_attn_res.kernels import phase_1 as nk1
        print(f"    eager autotune pick (phase-1 fwd): {getattr(nk1.phase_1_batched_attention_forward_kernel, 'best_config', None)}",
              flush=True)
        g1, d1 = compare(r_compat, r_old)
        g2, d2 = compare(r_def, r_eager)
        same3, d3 = compare(r_compat, r_def)
        print(f"S={S} mb={a.mb}: (1) compat == old compiled: bitwise {g1} (max|Δ| {d1:.1e}) | "
              f"(2) default compiled == eager: bitwise {g2} (max|Δ| {d2:.1e}) | "
              f"(3) compat vs default differ: {not same3} (max|Δ| {d3:.1e})", flush=True)
        print(f"    route fwd ms: old-compiled {t_old:.3f}  compat-compiled {t_compat:.3f}  "
              f"default-compiled {t_def:.3f}  eager {t_eager:.3f}", flush=True)
        bits_a, bits_b = r_def[4].view(torch.int16).int(), r_eager[4].view(torch.int16).int()
        same_sign = (bits_a < 0) == (bits_b < 0)
        ulp = torch.where(same_sign, (bits_a - bits_b).abs(), torch.full_like(bits_a, 1 << 15))
        n_diff = int((ulp > 0).sum())
        print(f"    phase-2 merge ulp distance: max {int(ulp.max())}, elements differing {n_diff} / {ulp.numel()}",
              flush=True)
        # One bf16 ulp at the tensor's scale: |diff| <= 2^-7 * max(|v|, rms) per element. Near-zero outputs (where
        # the merge cancels) show many ulps of their own tiny magnitude for an fp32-reassociation-sized difference.
        a_f, b_f = r_def[4].float(), r_eager[4].float()
        scale = torch.maximum(b_f.abs(), b_f.pow(2).mean().sqrt())
        ulp_ok = bool(((a_f - b_f).abs() <= 2.0 ** -7 * scale).all())
        g2_gate = all(torch.equal(x, y) for x, y in zip(r_def[:4], r_eager[:4])) and ulp_ok
        print(f"    gate (2): phase-1 bitwise + phase-2 within 1 bf16 ulp: {g2_gate}", flush=True)
        ok &= g1 and g2_gate and not same3
        del src, partial
        torch.cuda.empty_cache()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
