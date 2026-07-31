"""FA4-varlen custom op vs FA2-varlen parity smoke (C1 lever, SPEC-7B §5.1).

Same structure as flex_parity.py: identical-weight small AttnRes models,
same doc-masked batch, FA2 reference vs FA4 varlen (kotodama::fa4_varlen).
Also checks op-level parity + micro-bench at 7B attention shapes, and that
the custom op runs under torch.compile.

Run on gpu-host: tools/run_py.sh scripts/utils/fa4_varlen_parity.py
(FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 recommended — first call JITs.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig  # noqa: E402

DEVICE = "cuda"
B, T = 2, 512


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def make_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 1024, (B, T), generator=g)
    doc_lens = [[100, 300, 112], [400, 112]]
    position_ids = torch.zeros(B, T, dtype=torch.long)
    cu = [0]
    for b, lens in enumerate(doc_lens):
        off = 0
        for ln in lens:
            position_ids[b, off:off + ln] = torch.arange(ln)
            cu.append(cu[-1] + ln)
            off += ln
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    max_seqlen = max(max(lens) for lens in doc_lens)
    return (input_ids.to(DEVICE), position_ids.to(DEVICE),
            cu_seqlens.to(DEVICE), max_seqlen)


def build_model(attn_impl: str, state=None) -> LuxiaBaseModel:
    torch.manual_seed(17)
    cfg = LuxiaModelConfig(
        hidden_size=256, num_layers=4, num_attention_heads=4, num_kv_heads=2,
        head_dim=64, intermediate_size=512, vocab_size=1024,
        max_position_embeddings=2048, attn_res=True, attn_res_n_blocks=2,
        attn_impl=attn_impl, use_liger=False, activation_checkpointing=False,
    )
    m = LuxiaBaseModel(cfg).to(DEVICE)
    if state is not None:
        m.load_state_dict(state)
    return m


def main() -> int:
    torch.cuda.init()
    all_ok = True

    # ── op-level parity at 7B attention shapes ────────────────────────────
    from flash_attn import flash_attn_varlen_func
    from src.model.fa4_varlen import fa4_varlen_op

    g = torch.Generator(device=DEVICE).manual_seed(0)
    total, nq, nkv, hd = 16384, 32, 8, 128
    q = torch.randn(total, nq, hd, device=DEVICE, generator=g).bfloat16()
    k = torch.randn(total, nkv, hd, device=DEVICE, generator=g).bfloat16()
    v = torch.randn(total, nkv, hd, device=DEVICE, generator=g).bfloat16()
    cu = torch.arange(0, total + 1, 2048, device=DEVICE, dtype=torch.int32)
    ms = 2048

    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    ref = flash_attn_varlen_func(q1, k1, v1, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                 max_seqlen_q=ms, max_seqlen_k=ms, causal=True)
    ref.sum().backward()

    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)
    out, _lse = fa4_varlen_op(q2, k2, v2, cu, ms)
    out.sum().backward()

    e = rel_err(out, ref)
    all_ok &= check("op fwd fa4~fa2", e < 2e-2, f"relnorm {e:.3e}")
    for nm, a, b in [("dq", q2.grad, q1.grad), ("dk", k2.grad, k1.grad),
                     ("dv", v2.grad, v1.grad)]:
        e = rel_err(a, b)
        all_ok &= check(f"op {nm} fa4~fa2", e < 2e-2, f"relnorm {e:.3e}")

    # ── compile check: VALUES, not just execution. The 2026-07-09 NaN canary
    # passed a run-only compile check — the traced backward was silently
    # garbage. Compiled fwd/grads must match eager and be finite.
    r_w = torch.randn_like(q)  # elementwise weight -> contiguous, non-trivial grads

    def f(q_, k_, v_):
        o, _ = fa4_varlen_op(q_, k_, v_, cu, ms)
        return (o * r_w).sum()

    grads = {}
    outs = {}
    for mode, fn in [("eager", f), ("compiled", torch.compile(f))]:
        qc = q.clone().requires_grad_(True)
        kc = k.clone().requires_grad_(True)
        vc = v.clone().requires_grad_(True)
        loss = fn(qc, kc, vc)
        loss.backward()
        outs[mode] = loss.detach()
        grads[mode] = (qc.grad, kc.grad, vc.grad)
    all_ok &= check("compiled loss finite", bool(outs["compiled"].isfinite().all()))
    e = rel_err(outs["compiled"], outs["eager"])
    all_ok &= check("compiled~eager fwd", e < 1e-2, f"relnorm {e:.3e}")
    for i, nm in enumerate(["dq", "dk", "dv"]):
        finite = bool(grads["compiled"][i].isfinite().all())
        e = rel_err(grads["compiled"][i], grads["eager"][i])
        all_ok &= check(f"compiled~eager {nm}", finite and e < 1e-2,
                        f"finite={finite} relnorm {e:.3e}")

    # ── model-level parity (dispatch through GQAttention) ─────────────────
    input_ids, position_ids, cu_seqlens, max_seqlen = make_batch()
    refm = build_model("fa2")
    fa4m = build_model("fa4", state=refm.state_dict())

    losses = {}
    for name, model in [("fa2", refm), ("fa4", fa4m)]:
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=input_ids, position_ids=position_ids,
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out["loss"].backward()
        losses[name] = out["loss"].item()

    dl = abs(losses["fa4"] - losses["fa2"])
    all_ok &= check("loss fa2 vs fa4", dl < 5e-3,
                    f"{losses['fa2']:.6f} vs {losses['fa4']:.6f} (|d|={dl:.2e})")
    worst = ("", 0.0)
    for (n1, p1), (_n2, p2) in zip(refm.named_parameters(), fa4m.named_parameters()):
        if p1.grad is None or p2.grad is None:
            continue
        e = rel_err(p2.grad, p1.grad)
        if e > worst[1]:
            worst = (n1, e)
    all_ok &= check("grads fa2 vs fa4", worst[1] < 5e-2,
                    f"worst {worst[0]} relnorm {worst[1]:.2e}")

    # ── micro-bench fwd+bwd at 7B shapes ──────────────────────────────────
    print("\n=== micro-bench (fwd+bwd, 50 iters, 16K tokens, 32Q/8KV hd128) ===")
    import time
    for name, fn in [
        ("fa2", lambda a, b, c: flash_attn_varlen_func(
            a, b, c, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=ms, max_seqlen_k=ms, causal=True)),
        ("fa4", lambda a, b, c: fa4_varlen_op(a, b, c, cu, ms)[0]),
    ]:
        qb = q.clone().requires_grad_(True)
        kb = k.clone().requires_grad_(True)
        vb = v.clone().requires_grad_(True)
        for _ in range(10):
            fn(qb, kb, vb).sum().backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            fn(qb, kb, vb).sum().backward()
        torch.cuda.synchronize()
        print(f"  {name}: {(time.perf_counter() - t0) / 50 * 1e3:7.3f} ms/iter")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
