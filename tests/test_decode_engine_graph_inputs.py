"""Inputs the DecodeEngine hands its compiled graphs must have one layout per shape.

A (1, n) slice of a (1, S) prompt keeps stride (S, 1) and a nonzero storage offset, and `.contiguous()` returns it
unchanged because a size-1 row makes it count as contiguous already. The compiled block and step treat both values
as dynamic, and inductor's cudagraph trees record a separate graph, with its own memory pool, for every distinct
value: every new prompt length then costs fresh graph memory until the card runs out. The block prefill, the
prefix-cache block extend and the prefix-cache one-token step must all pass offset-0, canonically strided inputs.

CPU, fp32: the compiled entry points are stood in for by recorders around the eager forwards they compile.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=256, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)


@pytest.fixture()
def recorded() -> tuple[DecodeEngine, list[torch.Tensor]]:
    """An engine whose compiled block and step are recorders around the eager forwards."""
    torch.manual_seed(5)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    engine = DecodeEngine(model, max_seq_len=200, dtype=torch.float32, prefill_backends=[SDPBackend.MATH])
    seen: list[torch.Tensor] = []

    def block(tokens: torch.Tensor) -> torch.Tensor:
        seen.append(tokens)
        return engine._forward_block(tokens)

    def step(token: torch.Tensor) -> torch.Tensor:
        seen.append(token)
        return engine._forward_step(token)

    engine._compiled_block = block
    engine._compiled_forward = step
    return engine, seen


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


def _canonical(t: torch.Tensor) -> bool:
    return t.storage_offset() == 0 and t.stride() == (t.shape[1], 1)


@torch.inference_mode()
def test_block_prefill_chunks_are_canonical(recorded) -> None:
    engine, seen = recorded
    engine.prefill(_ids(83, 1))  # 64 + 16 + a right-aligned overlap chunk: three slices at three offsets
    assert len(seen) == 3
    assert all(_canonical(t) for t in seen), [(t.storage_offset(), t.stride()) for t in seen]


@torch.inference_mode()
def test_prefix_cache_extend_and_step_inputs_are_canonical(recorded) -> None:
    engine, seen = recorded
    prompt = _ids(40, 2)
    engine.prefill_cached(prompt)
    seen.clear()

    engine.prefill_cached(torch.cat([prompt, _ids(1, 3)], dim=1))  # one-token extend: the compiled step
    engine.prefill_cached(torch.cat([prompt, _ids(1, 3), _ids(20, 4)], dim=1))  # block extend
    assert len(seen) >= 2
    assert all(_canonical(t) for t in seen), [(t.storage_offset(), t.stride()) for t in seen]


@torch.inference_mode()
def test_canonical_inputs_leave_logits_unchanged(recorded) -> None:
    engine, _ = recorded
    prompt = _ids(83, 6)
    chunked = engine.prefill(prompt)
    reference = engine.prefill_reference(prompt)
    assert torch.allclose(chunked, reference, atol=1e-4)
