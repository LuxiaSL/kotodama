"""rope_fp32: RoPE tables survive the bf16 cast in fp32 and relative-position scores
become more shift-invariant; the default path is bit-identical to before."""
from __future__ import annotations

import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig, apply_rope


def _model(rope_fp32: bool) -> KotodamaModel:
    torch.manual_seed(0)
    return KotodamaModel(KotodamaModelConfig(
        hidden_size=64, num_layers=1, num_attention_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=64, vocab_size=64, max_position_embeddings=4096, attn_impl="sdpa",
        rope_fp32=rope_fp32,
    )).to(torch.bfloat16)


def test_tables_stay_fp32_through_bf16_cast():
    assert _model(True).rope_cos.dtype == torch.float32
    assert _model(False).rope_cos.dtype == torch.bfloat16


def test_rotation_keeps_activation_dtype():
    m = _model(True)
    x = torch.randn(1, 2, 16, 32).bfloat16()
    assert apply_rope(x, m.rope_cos[:16], m.rope_sin[:16]).dtype == torch.bfloat16


def _shift_error(m: KotodamaModel) -> float:
    """|score(q@a, k@b) - score(q@a+s, k@b+s)| averaged: 0 for exact RoPE."""
    torch.manual_seed(1)
    q = torch.randn(64, 32).bfloat16()
    k = torch.randn(64, 32).bfloat16()
    errs = []
    for a, b, s in [(10, 3, 1000), (500, 20, 2500), (3000, 2990, 700)]:
        def sc(p1, p2):
            qa = apply_rope(q[None, None, :, :].expand(1, 1, 64, 32).clone(),
                            m.rope_cos[p1:p1 + 1].expand(64, 32), m.rope_sin[p1:p1 + 1].expand(64, 32))
            kb = apply_rope(k[None, None, :, :].expand(1, 1, 64, 32).clone(),
                            m.rope_cos[p2:p2 + 1].expand(64, 32), m.rope_sin[p2:p2 + 1].expand(64, 32))
            return (qa.float() * kb.float()).sum(-1)
        errs.append((sc(a, b) - sc(a + s, b + s)).abs().mean().item())
    return sum(errs) / len(errs)


def test_fp32_tables_are_more_shift_invariant():
    # ~10% better: the rotated q/k are still rounded to bf16 before the dot product,
    # which dominates what remains.
    assert _shift_error(_model(True)) < 0.95 * _shift_error(_model(False))
