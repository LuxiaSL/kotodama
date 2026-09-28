"""FA4 custom ops (kotodama::fa4_dense / fa4_varlen): registration, fake kernels,
autograd wiring, and torch.compile-safety — all on CPU.

The CuTe kernels need an SM100 GPU, so the compile tests register a CPU
*reference* kernel (fp32 causal GQA attention + its analytic backward) for the
dense ops. That exercises exactly what compile-safety depends on — the op
schema, the fake kernels, the autograd registration, and dynamo/AOTAutograd
treating the op as opaque — while the GPU numerics are covered by
scripts/utils/fa4_dense_parity.py.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from kotodama.model import fa4_varlen as fa4

# (B, S, Hq, Hkv, D) — MHA, GQA 2:1 (1B ladder ratio), GQA 4:1 (7B ratio)
DENSE_SHAPES = [(2, 16, 4, 4, 8), (2, 16, 4, 2, 8), (1, 32, 8, 2, 16)]


# ── registration ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name", ["fa4_dense", "fa4_dense_bwd", "fa4_varlen", "fa4_varlen_bwd"]
)
def test_ops_registered(name: str) -> None:
    op = getattr(torch.ops.kotodama, name)
    assert op.default._schema.name == f"kotodama::{name}"


def test_autograd_registered() -> None:
    # register_autograd populates the CustomOpDef's backward slot.
    assert fa4.fa4_dense_op._backward_fn is not None
    assert fa4.fa4_varlen_op._backward_fn is not None
    # the backward ops themselves are leaves (no autograd of their own)
    assert fa4.fa4_dense_bwd_op._backward_fn is None


def test_import_without_flash_attn() -> None:
    """The module must import (and register its ops) on a machine with no
    flash_attn.cute — the lazy loader in llama.py relies on the guard."""
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; sys.modules['flash_attn'] = None; "
        "sys.modules['flash_attn.cute'] = None; "
        "sys.modules['flash_attn.cute.interface'] = None\n"
        "from kotodama.model import fa4_varlen as m\n"
        "import torch\n"
        "assert not m.fa4_varlen_available()\n"
        "torch.ops.kotodama.fa4_dense.default\n"
        "print('ok')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(src), "PATH": ""},
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")


# ── fake kernels ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", DENSE_SHAPES)
@pytest.mark.parametrize("strided", [False, True])
def test_dense_fake_shapes(shape: tuple[int, ...], strided: bool) -> None:
    B, S, Hq, Hkv, D = shape
    with FakeTensorMode():
        if strided:
            # q as the view the model produces from a fused/transposed buffer
            q = torch.empty(B, Hq, S, D, dtype=torch.bfloat16).transpose(1, 2)
            assert not q.is_contiguous()
        else:
            q = torch.empty(B, S, Hq, D, dtype=torch.bfloat16)
        k = torch.empty(B, S, Hkv, D, dtype=torch.bfloat16)
        v = torch.empty(B, S, Hkv, D, dtype=torch.bfloat16)

        out, lse = fa4.fa4_dense_op(q, k, v)
        assert out.shape == (B, S, Hq, D) and out.dtype == torch.bfloat16
        assert out.is_contiguous()
        # LSE layout of _flash_attn_fwd dense path: (batch, num_head, seqlen_q)
        assert lse.shape == (B, Hq, S) and lse.dtype == torch.float32
        assert lse.is_contiguous()

        g = torch.empty(B, S, Hq, D, dtype=torch.bfloat16)
        dq, dk, dv = fa4.fa4_dense_bwd_op(q, k, v, out, g, lse)
        assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
        assert dq.dtype == dk.dtype == dv.dtype == torch.bfloat16
        assert dq.is_contiguous() and dk.is_contiguous() and dv.is_contiguous()


def test_varlen_fake_shapes() -> None:
    total, Hq, Hkv, D = 48, 4, 2, 8
    with FakeTensorMode():
        q = torch.empty(Hq, total, D, dtype=torch.bfloat16).transpose(0, 1)
        k = torch.empty(total, Hkv, D, dtype=torch.bfloat16)
        v = torch.empty(total, Hkv, D, dtype=torch.bfloat16)
        cu = torch.tensor([0, 16, 48], dtype=torch.int32)
        out, lse = fa4.fa4_varlen_op(q, k, v, cu, 32)
        assert out.shape == (total, Hq, D) and out.is_contiguous()
        assert lse.shape == (Hq, total) and lse.dtype == torch.float32
        dq, dk, dv = fa4.fa4_varlen_bwd_op(q, k, v, out, out, lse, cu, 32)
        assert (dq.shape, dk.shape, dv.shape) == (q.shape, k.shape, v.shape)
        assert dq.is_contiguous()


@pytest.mark.parametrize("shape", DENSE_SHAPES)
def test_dense_autograd_traces_bwd_op(shape: tuple[int, ...]) -> None:
    """Forward + autograd.grad traced with fake tensors: the backward must go
    through the opaque kotodama::fa4_dense_bwd op (never raw kernels), with
    grads shaped like their inputs."""
    B, S, Hq, Hkv, D = shape

    def fwd_bwd(q, k, v, g):
        out, _lse = fa4.fa4_dense_op(q, k, v)
        return torch.autograd.grad(out, (q, k, v), g)

    def mk(*s):
        return torch.randn(*s, dtype=torch.bfloat16, requires_grad=True)

    q, k, v = mk(B, S, Hq, D), mk(B, S, Hkv, D), mk(B, S, Hkv, D)
    g = torch.randn(B, S, Hq, D, dtype=torch.bfloat16)
    gm = make_fx(fwd_bwd, tracing_mode="fake")(q, k, v, g)
    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    assert "kotodama.fa4_dense.default" in targets
    assert "kotodama.fa4_dense_bwd.default" in targets
    outs = next(n for n in gm.graph.nodes if n.op == "output").args[0]
    shapes = [tuple(o.meta["val"].shape) for o in outs]
    assert shapes == [(B, S, Hq, D), (B, S, Hkv, D), (B, S, Hkv, D)]


# ── compile-safety with a CPU reference kernel ───────────────────────────────


def _ref_scores(q, k):
    """fp32 causal scores (B, Hq, S, S) with GQA expansion; default scale."""
    groups = q.shape[2] // k.shape[2]
    qh = q.float().transpose(1, 2)
    kh = k.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    s = qh @ kh.transpose(-1, -2) / math.sqrt(q.shape[-1])
    S = q.shape[1]
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    return s.masked_fill(~causal, float("-inf")), groups


_REF_CALLS = {"fwd": 0, "bwd": 0}


def _ref_fwd(q, k, v):
    _REF_CALLS["fwd"] += 1
    s, groups = _ref_scores(q, k)
    lse = torch.logsumexp(s, dim=-1)  # (B, Hq, S)
    p = torch.exp(s - lse.unsqueeze(-1))
    vh = v.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    out = (p @ vh).transpose(1, 2).contiguous().to(q.dtype)
    return out, lse.contiguous()


def _ref_bwd(q, k, v, out, g, lse):
    _REF_CALLS["bwd"] += 1
    s, groups = _ref_scores(q, k)
    p = torch.exp(s - lse.unsqueeze(-1))
    B, S, Hkv, D = k.shape
    qh = q.float().transpose(1, 2)
    kh = k.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    vh = v.float().repeat_interleave(groups, dim=2).transpose(1, 2)
    gh = g.float().transpose(1, 2)
    oh = out.float().transpose(1, 2)
    dvh = p.transpose(-1, -2) @ gh
    dp = gh @ vh.transpose(-1, -2)
    ds = p * (dp - (gh * oh).sum(-1, keepdim=True))
    scale = 1.0 / math.sqrt(D)
    dqh = ds @ kh * scale
    dkh = ds.transpose(-1, -2) @ qh * scale

    def fold(t):  # (B, Hq, S, D) -> (B, S, Hkv, D), summing each GQA group
        return t.view(B, Hkv, groups, S, D).sum(2).transpose(1, 2).contiguous()

    return dqh.transpose(1, 2).contiguous().to(q.dtype), fold(dkh).to(k.dtype), fold(dvh).to(v.dtype)


@pytest.fixture
def cpu_reference_kernels() -> Iterator[None]:
    """Install the CPU reference as the ops' CPU kernel for this test only."""
    ops = [(fa4.fa4_dense_op, _ref_fwd), (fa4.fa4_dense_bwd_op, _ref_bwd)]
    for op, fn in ops:
        if "cpu" not in op._backend_fns:
            op.register_kernel("cpu", fn)  # enabled on registration
        else:
            op.set_kernel_enabled("cpu", True)
    try:
        yield
    finally:
        for op, _ in ops:
            op.set_kernel_enabled("cpu", False)


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=True, enable_gqa=True,
    ).transpose(1, 2)


@pytest.mark.parametrize("shape", DENSE_SHAPES[1:])
def test_dense_reference_matches_sdpa(cpu_reference_kernels, shape) -> None:
    """Sanity: the CPU reference kernel (and so the compile test below) is
    real causal GQA attention, fwd + bwd."""
    B, S, Hq, Hkv, D = shape
    torch.manual_seed(0)
    q0, k0, v0 = torch.randn(B, S, Hq, D), torch.randn(B, S, Hkv, D), torch.randn(B, S, Hkv, D)
    w = torch.randn(B, S, Hq, D)
    grads = []
    for fn in (lambda a, b, c: fa4.fa4_dense_op(a, b, c)[0], _sdpa):
        q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        (fn(q, k, v) * w).sum().backward()
        grads.append((fn(q, k, v).detach(), q.grad, k.grad, v.grad))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_dense_compile_fullgraph(cpu_reference_kernels, backend: str) -> None:
    """fullgraph=True: any graph break (raw CuTe call, untraceable wrapper)
    raises. Compiled fwd + grads must equal eager."""
    B, S, Hq, Hkv, D = 2, 16, 4, 2, 8
    torch.manual_seed(1)
    q0, k0, v0 = torch.randn(B, S, Hq, D), torch.randn(B, S, Hkv, D), torch.randn(B, S, Hkv, D)
    w = torch.randn(B, S * Hq * D)

    def attn(q, k, v):
        # model-like: producer op before, flattened output consumed after
        o, _lse = fa4.fa4_dense_op(q * 1.0, k, v)
        return (o.reshape(B, S, -1) * w.view(B, S, -1)).sum()

    torch._dynamo.reset()
    results = {}
    for mode, fn in [("eager", attn), ("compiled", torch.compile(attn, backend=backend, fullgraph=True))]:
        q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
        loss = fn(q, k, v)
        loss.backward()
        results[mode] = (loss.detach(), q.grad, k.grad, v.grad)
    for a, b in zip(results["compiled"], results["eager"]):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    torch._dynamo.reset()


def test_model_fa4_dense_compiles(cpu_reference_kernels) -> None:
    """Model dispatch: attn_impl='fa4' without doc masking routes through
    kotodama::fa4_dense and compiles (the restriction train.py used to
    enforce). Loss/grads vs the SDPA model within bf16 noise (the fa4 path
    casts q/k/v to bf16)."""
    from kotodama.model import llama

    if not llama._init_fa4():
        pytest.skip("flash_attn.cute not importable — model can't select fa4")
    from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

    def build(impl: str, state=None) -> LuxiaBaseModel:
        torch.manual_seed(17)
        cfg = LuxiaModelConfig(
            hidden_size=64, num_layers=2, num_attention_heads=4, num_kv_heads=2,
            head_dim=16, intermediate_size=128, vocab_size=128,
            max_position_embeddings=64, attn_res=True, attn_res_n_blocks=2,
            attn_impl=impl, use_liger=False, activation_checkpointing=False,
        )
        m = LuxiaBaseModel(cfg)
        if state is not None:
            m.load_state_dict(state)
        return m

    ref = build("sdpa")
    m4 = build("fa4", ref.state_dict())
    assert all(layer.attn._attn_impl == "fa4" for layer in m4.layers)
    ids = torch.randint(0, 128, (2, 32), generator=torch.Generator().manual_seed(3))

    torch._dynamo.reset()
    calls0 = dict(_REF_CALLS)
    losses = {}
    for name, model, fn in [("sdpa", ref, ref), ("fa4", m4, torch.compile(m4, backend="aot_eager"))]:
        with torch.autocast("cpu", dtype=torch.bfloat16):  # as in training
            loss = fn(ids, labels=ids)["loss"]
        loss.backward()
        losses[name] = loss.detach()
    torch._dynamo.reset()
    # every layer's attention went through the op, forward and backward
    assert _REF_CALLS["fwd"] - calls0["fwd"] == len(m4.layers)
    assert _REF_CALLS["bwd"] - calls0["bwd"] == len(m4.layers)
    torch.testing.assert_close(losses["fa4"], losses["sdpa"], atol=2e-2, rtol=2e-2)
    for (n, p1), (_, p2) in zip(ref.named_parameters(), m4.named_parameters()):
        if p1.grad is None:
            continue
        err = ((p2.grad - p1.grad).norm() / p1.grad.norm().clamp_min(1e-12)).item()
        assert err < 5e-2, (n, err)
