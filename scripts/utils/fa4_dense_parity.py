"""FA4 dense custom op (kotodama::fa4_dense) parity + compile check. GPU (SM100).

  (a) op fwd+bwd vs the raw flash_attn.cute.flash_attn_func(q, k, v, causal=True)
      the op wraps — same kernel + args, so fwd should be bitwise;
      grads bitwise or at the raw-vs-raw noise floor (the bwd dQ accumulates
      with atomics). Also vs torch SDPA (enable_gqa) as a semantics check.
  (b) torch.compile(model-like attention block using the op, fullgraph=True)
      vs the same block eager — loss + every grad finite and within tolerance.
      (Values, not just execution: a run-only compile check passes on a
      silently-NaN traced backward.)
  (c) LuxiaBaseModel(attn_impl='fa4', no doc masking) compiled vs eager — the
      combination train.py runs for fa4 without doc masking.

Default shape = the 1B ladder attention: hidden 2048, 16 q / 8 kv heads,
head_dim 128, seq 4096, batch 2 (or --preset NAME to take it from presets.py).

Run on the GPU node: tools/run_py.sh scripts/utils/fa4_dense_parity.py
(FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 recommended — first call JITs.)
Exit code 0 = ALL PASS.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama import presets

DEVICE = "cuda"


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def fwd_bwd(fn: Callable, q0, k0, v0, w) -> tuple[torch.Tensor, ...]:
    q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
    out = fn(q, k, v)
    (out.float() * w).sum().backward()
    return out.detach(), q.grad, k.grad, v.grad


def op_parity(B: int, S: int, Hq: int, Hkv: int, D: int) -> bool:
    from flash_attn.cute import flash_attn_func

    from kotodama.model.fa4_varlen import fa4_dense_op

    print(f"\n=== (a) op vs raw FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D} bf16 ===")
    g = torch.Generator(device=DEVICE).manual_seed(0)
    q0 = torch.randn(B, S, Hq, D, device=DEVICE, generator=g).bfloat16()
    k0 = torch.randn(B, S, Hkv, D, device=DEVICE, generator=g).bfloat16()
    v0 = torch.randn(B, S, Hkv, D, device=DEVICE, generator=g).bfloat16()
    w = torch.randn(B, S, Hq, D, device=DEVICE, generator=g)

    def raw(q, k, v):
        res = flash_attn_func(q, k, v, causal=True)
        return res[0] if isinstance(res, tuple) else res

    def op(q, k, v):
        return fa4_dense_op(q, k, v)[0]

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, enable_gqa=True,
        ).transpose(1, 2)

    r1 = fwd_bwd(raw, q0, k0, v0, w)
    r2 = fwd_bwd(raw, q0, k0, v0, w)
    o = fwd_bwd(op, q0, k0, v0, w)
    s = fwd_bwd(sdpa, q0, k0, v0, w)

    ok = True
    names = ["out", "dq", "dk", "dv"]
    for i, nm in enumerate(names):
        floor = max_abs(r2[i], r1[i])
        err = max_abs(o[i], r1[i])
        bitwise = torch.equal(o[i], r1[i])
        finite = bool(o[i].isfinite().all())
        if nm == "out":
            good = finite and bitwise
        else:
            good = finite and (bitwise or err <= 2 * floor or rel_err(o[i], r1[i]) < 1e-3)
        ok &= check(f"op~raw {nm}", good,
                    f"bitwise={bitwise} max|d|={err:.3e} raw-vs-raw floor={floor:.3e} "
                    f"relnorm={rel_err(o[i], r1[i]):.3e}")
    for i, nm in enumerate(names):
        e = rel_err(o[i], s[i])
        ok &= check(f"op~sdpa {nm}", e < 2e-2, f"max|d|={max_abs(o[i], s[i]):.3e} relnorm={e:.3e}")
    return ok


def compile_block(B: int, S: int, hidden: int, Hq: int, Hkv: int, D: int) -> bool:
    from kotodama.model.fa4_varlen import fa4_dense_op

    print(f"\n=== (b) compiled vs eager attention block (hidden={hidden}) ===")
    torch.manual_seed(1)
    wq = torch.nn.Linear(hidden, Hq * D, bias=False, device=DEVICE, dtype=torch.bfloat16)
    wk = torch.nn.Linear(hidden, Hkv * D, bias=False, device=DEVICE, dtype=torch.bfloat16)
    wv = torch.nn.Linear(hidden, Hkv * D, bias=False, device=DEVICE, dtype=torch.bfloat16)
    wo = torch.nn.Linear(Hq * D, hidden, bias=False, device=DEVICE, dtype=torch.bfloat16)
    params = [wq.weight, wk.weight, wv.weight, wo.weight]
    x0 = torch.randn(B, S, hidden, device=DEVICE, dtype=torch.bfloat16)
    r = torch.randn(B, S, hidden, device=DEVICE)

    def block(x):
        q = wq(x).view(B, S, Hq, D)
        k = wk(x).view(B, S, Hkv, D)
        v = wv(x).view(B, S, Hkv, D)
        o, _lse = fa4_dense_op(q, k, v)
        y = wo(o.contiguous().view(B, S, -1))
        return (y.float() * r).sum() / r.numel()

    results = {}
    torch._dynamo.reset()
    for mode, fn in [("eager", block), ("compiled", torch.compile(block, fullgraph=True))]:
        for p in params:
            p.grad = None
        x = x0.clone().requires_grad_(True)
        loss = fn(x)
        loss.backward()
        results[mode] = [loss.detach(), x.grad] + [p.grad.clone() for p in params]
    ok = True
    for nm, a, b in zip(["loss", "dx", "dWq", "dWk", "dWv", "dWo"],
                        results["compiled"], results["eager"]):
        finite = bool(a.isfinite().all())
        e = rel_err(a, b)
        ok &= check(f"compiled~eager {nm}", finite and e < 1e-2,
                    f"finite={finite} max|d|={max_abs(a, b):.3e} relnorm={e:.3e}")
    return ok


def compile_model() -> bool:
    from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

    print("\n=== (c) LuxiaBaseModel fa4 (no doc masking): compiled vs eager ===")
    torch.manual_seed(17)
    cfg = LuxiaModelConfig(
        **presets.shape("smoke"), attn_res=True, attn_res_n_blocks=2,
        attn_impl="fa4", use_liger=False, activation_checkpointing=False,
    )
    ref = LuxiaBaseModel(cfg).to(DEVICE)
    cm = LuxiaBaseModel(cfg).to(DEVICE)
    cm.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (2, 512), device=DEVICE,
                        generator=torch.Generator(device=DEVICE).manual_seed(3))
    torch._dynamo.reset()
    losses = {}
    for name, model, fn in [("eager", ref, ref), ("compiled", cm, torch.compile(cm))]:
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fn(ids, labels=ids)["loss"]
        loss.backward()
        losses[name] = loss.item()
    ok = check("loss finite", all(math.isfinite(v) for v in losses.values()))
    dl = abs(losses["compiled"] - losses["eager"])
    ok &= check("loss compiled~eager", dl < 5e-3,
                f"{losses['eager']:.6f} vs {losses['compiled']:.6f} (|d|={dl:.2e})")
    worst = ("", 0.0)
    all_finite = True
    for (n, p1), (_, p2) in zip(ref.named_parameters(), cm.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        all_finite &= bool(p2.grad.isfinite().all())
        e = rel_err(p2.grad, p1.grad)
        if e > worst[1]:
            worst = (n, e)
    ok &= check("grads compiled~eager", all_finite and worst[1] < 5e-2,
                f"finite={all_finite} worst {worst[0]} relnorm {worst[1]:.2e}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=None, help="take the attention shape from presets.py")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv_heads", type=int, default=8)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--skip_model", action="store_true", help="skip (c)")
    args = ap.parse_args()
    if args.preset:
        s = presets.shape(args.preset)
        args.hidden, args.heads = s["hidden_size"], s["num_attention_heads"]
        args.kv_heads, args.head_dim = s["num_kv_heads"], s["head_dim"]

    if not torch.cuda.is_available():
        print("CUDA not available — this script needs an SM100 GPU.")
        return 2
    torch.cuda.init()
    from kotodama.model.fa4_varlen import fa4_varlen_available

    if not fa4_varlen_available():
        print("flash_attn.cute not importable.")
        return 2

    all_ok = True
    all_ok &= op_parity(args.batch, args.seq, args.heads, args.kv_heads, args.head_dim)
    all_ok &= compile_block(args.batch, args.seq, args.hidden, args.heads,
                            args.kv_heads, args.head_dim)
    if not args.skip_model:
        all_ok &= compile_model()

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
