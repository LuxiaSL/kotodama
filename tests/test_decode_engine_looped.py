"""DecodeEngine on a LOOPED model (1b-tsumugi shape, tiny): the static-cache decode step, the eager
reference prefill and the block prefill must reproduce the full no-cache forward.

Looped repeats alias their core's attn/FFN, but each effective position keeps its own KV cache, pre-norms and
AttnRes routing params — so a cache keyed by unique layer (or routing read from the core) would diverge here.
CPU, fp32, MATH SDPA: exact enough that logits must agree to ~1e-4 and every greedy argmax must match.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

# tsumugi geometry in miniature: prelude 0-2, a 3-layer core run 2x (3-8), coda 9-10; a boundary at each pass
# start and at the coda (cf. presets.TSUMUGI_LOOP).
CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)


def _perturb_per_position(model: KotodamaModel) -> None:
    """Fresh init leaves per-position norms at 1 and routing queries at 0, which would hide a bug that reads
    them from the core position. Make every per-position (non-shared) parameter distinct."""
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            for name, p in layer.named_parameters():
                if name.startswith(("attn.", "ffn.")):
                    continue  # shared with the core on loop repeats
                p.add_(0.3 * torch.randn(p.shape, generator=g))


@pytest.fixture(scope="module")
def engine() -> DecodeEngine:
    torch.manual_seed(20261022)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    _perturb_per_position(model)
    assert model.layers[6].attn is model.layers[3].attn  # the loop really aliases
    assert model.layers[6].attn_norm is not model.layers[3].attn_norm
    return DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH])


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def test_cached_greedy_decode_matches_full_forward(engine: DecodeEngine) -> None:
    prompt = _ids(13, 1)
    steps = 24
    toks = engine.generate_greedy(prompt, steps)
    assert len(toks) == steps
    full = engine.model(torch.cat([prompt, torch.tensor([toks])], dim=1))["logits"][0].float()
    want = full[prompt.shape[1] - 1:-1].argmax(-1).tolist()
    assert toks == want


@torch.inference_mode()
def test_step_logits_match_full_forward(engine: DecodeEngine) -> None:
    prompt, forced = _ids(9, 2), _ids(20, 3)[0].tolist()
    seq = torch.cat([prompt, torch.tensor([forced])], dim=1)
    full = engine.model(seq)["logits"][0].float()
    got = [engine.prefill_reference(prompt)[0].float()]
    for t in forced[:-1]:
        got.append(engine.step_logits(torch.tensor([[t]]))[0, -1].float())
    want = full[prompt.shape[1] - 1:-1]
    assert torch.allclose(torch.stack(got), want, atol=2e-4, rtol=1e-4), (torch.stack(got) - want).abs().max()


@torch.inference_mode()
def test_block_prefill_matches_full_forward(engine: DecodeEngine) -> None:
    # 37 tokens -> chunks 16+16 + a right-aligned overlap 16: the block path's per-position cache writes. The
    # served engine compiles _forward_block; running it eagerly here tests the same math on CPU.
    prompt = _ids(37, 4)
    want = engine.model(prompt)["logits"][0, -1].float()
    assert engine._chunk_plan(0, 37) == [(0, 16), (16, 16), (21, 16)]
    engine._compiled_block = engine._forward_block
    try:
        got = engine.prefill(prompt)[0].float()
    finally:
        engine._compiled_block = None
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), (got - want).abs().max()
    # and decoding on from the block-filled cache agrees with a full forward
    nxt = int(got.argmax())
    step = engine.step_logits(torch.tensor([[nxt]]))[0, -1].float()
    full = engine.model(torch.cat([prompt, torch.tensor([[nxt]])], dim=1))["logits"][0, -1].float()
    assert torch.allclose(step, full, atol=2e-4, rtol=1e-4), (step - full).abs().max()
