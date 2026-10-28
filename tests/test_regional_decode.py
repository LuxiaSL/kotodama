"""The regional decode step (``compile_step(regional=True)``) on a tiny looped model, CPU fp32.

What is pinned here:
  * the regional step computes what the eager step computes: logits along a forced continuation within the parity
    tolerance of the lever suite, and the same tokens for seeded sampled rows (plain and truncated laws), greedy rows
    and scores, with lever slots, broadcast steering, readouts and MLP-input captures on, for both row decoders;
  * one compiled graph per layer signature, not per layer: a step compiles exactly the distinct signatures plus the
    tail (final routing + LM head) and its sampler, and a second step compiles nothing;
  * the tsumugi geometry's signature count, from its boundary table;
  * whole-step CUDA graph capture is refused on a CPU engine, and the regional-only options are refused without
    ``regional``.
"""

from __future__ import annotations

import os
import shutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch.nn.attention import SDPBackend

from kotodama import presets
from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.levers import LeverPlan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder, layer_signatures

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
# A deeper loop whose layers repeat signatures more often (16 layers, 9 signatures).
DEEP = dict(CFG, num_layers=16, loop_start=3, loop_len=5, loop_times=2)
ROWS = 4
H = CFG["hidden_size"]
TOL = dict(atol=2e-4, rtol=1e-4)


def _model(cfg: dict) -> KotodamaModel:
    torch.manual_seed(20261026)
    model = KotodamaModel(KotodamaModelConfig(**cfg)).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    return _model(CFG)


def _engine(model: KotodamaModel, levers: int = 3, steer: bool = True) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer, levers=levers)


def _dec(cls, eng: DecodeEngine, **kw) -> MultiSampleDecoder:
    return cls(eng, ROWS, tail_tokens=32, **kw) if cls is CascadeDecoder else cls(eng, ROWS, **kw)


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


def _vec(seed: int, scale: float = 1.0) -> np.ndarray:
    return (scale * np.random.default_rng(seed).standard_normal(H)).astype(np.float32)


def _steered_engine(model: KotodamaModel) -> tuple[DecodeEngine, list[int]]:
    """An engine with two decode-span levers bound and a broadcast steering row engaged; returns it and the row
    slots (two lever rows, two identity rows)."""
    eng = _engine(model)
    s1 = eng.levers.bind(LeverPlan(attn={4: _vec(1, 0.8)}, span="decode"))
    s2 = eng.levers.bind(LeverPlan(mlp={7: _vec(2, 0.8)}, span="decode"))
    eng.set_steer([(2, torch.from_numpy(_vec(3)), 0.5)])
    return eng, [s1, 0, s2, 0]


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_regional_step_matches_the_eager_step(model: KotodamaModel, cls) -> None:
    eng, slots = _steered_engine(model)
    kw = dict(seeded=True, readout_k=3, capture_layers=(2, 5))
    ref, reg = _dec(cls, eng, **kw), _dec(cls, eng, **kw)
    reg.compile_step(regional=True, backend="aot_eager")
    prompt = _ids(10, 1)
    seeds = [11, 12, 13, 14]
    laws = [SamplingParams(temperature=0.8, repetition_penalty=1.2),
            SamplingParams(temperature=0.9, repetition_penalty=1.1, top_p=0.8, min_p=0.05),
            SamplingParams(temperature=0.0, repetition_penalty=1.0)]
    for params in laws:
        logits = eng.prefill(prompt)
        for dec in (ref, reg):
            dec.release(range(ROWS))
            dec.admit(list(range(ROWS)), logits, params, slots=slots, seeds=seeds)
        for _ in range(6):
            assert ref.step(params).view(-1).tolist() == reg.step(params).view(-1).tolist(), params
            torch.testing.assert_close(reg.step_logprob, ref.step_logprob, **TOL)
            assert torch.equal(reg.step_top_ids, ref.step_top_ids)
            torch.testing.assert_close(reg.mlp_in, ref.mlp_in, **TOL)

    forced = torch.cat([_ids(7, 20 + r) for r in range(ROWS)])
    logits = eng.prefill(prompt)
    for dec in (ref, reg):
        dec.release(range(ROWS))
        dec.admit(list(range(ROWS)), logits, laws[0], slots=slots, seeds=seeds)
    for t in range(forced.shape[1]):
        want = ref.step_logits(forced[:, t:t + 1])
        got = reg._regional_forward(forced[:, t:t + 1].clone())
        reg._advance_host()
        torch.testing.assert_close(got, want, **TOL)

    logits = eng.prefill(prompt)
    for dec in (ref, reg):
        dec.release(range(ROWS))
        dec.admit(list(range(ROWS)), logits, laws[0], slots=slots, seeds=seeds)
    torch.testing.assert_close(reg.score(forced), ref.score(forced), **TOL)


@pytest.mark.skipif(shutil.which("c++") is None and shutil.which("g++") is None,
                    reason="inductor's CPU backend needs a C++ compiler")
@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_inductor_regions_match_the_eager_step(model: KotodamaModel, cls) -> None:
    """The regions as the server builds them (inductor), on CPU: same seeded and greedy tokens, logits within the
    parity tolerance."""
    eng, slots = _steered_engine(model)
    ref, reg = _dec(cls, eng, seeded=True), _dec(cls, eng, seeded=True)
    reg.compile_step(regional=True)  # inductor, the default mode's no-cudagraphs form
    prompt = _ids(9, 4)
    for params in (SamplingParams(temperature=0.8, repetition_penalty=1.2),
                   SamplingParams(temperature=0.0, repetition_penalty=1.0)):
        logits = eng.prefill(prompt)
        for dec in (ref, reg):
            dec.release(range(ROWS))
            dec.admit(list(range(ROWS)), logits, params, slots=slots, seeds=[5, 6, 7, 8])
        for _ in range(5):
            assert ref.step(params).view(-1).tolist() == reg.step(params).view(-1).tolist(), params
    forced = torch.cat([_ids(5, 30 + r) for r in range(ROWS)])
    for t in range(forced.shape[1]):
        want = ref.step_logits(forced[:, t:t + 1])
        got = reg._regional_forward(forced[:, t:t + 1].clone())
        reg._advance_host()
        torch.testing.assert_close(got, want, **TOL)


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_one_compiled_graph_per_layer_signature(cls) -> None:
    deep = _model(DEEP)
    eng = _engine(deep)
    dec = _dec(cls, eng, seeded=True)
    sigs = dec.regional_signatures()
    assert len(set(sigs)) == 9 < DEEP["num_layers"]
    counter = CompileCounterWithBackend("aot_eager")
    dec.compile_step(regional=True, backend=counter)
    assert dec.regional_variants == len(set(sigs))
    params = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    dec.admit(list(range(ROWS)), eng.prefill(_ids(8, 2)), params)
    dec.step(params)
    assert counter.frame_count == len(set(sigs)) + 2  # the layer signatures, the tail and the sampler
    dec.step(params)
    assert counter.frame_count == len(set(sigs)) + 2  # nothing recompiles
    dec.step(SamplingParams(temperature=0.0, repetition_penalty=1.2))
    assert counter.frame_count == len(set(sigs)) + 3  # the greedy sampler only


def test_tsumugi_layer_signatures() -> None:
    """Shape A: 72 effective layers over 8 boundaries fall into 15 signatures (committed counts 0..8, boundary
    layers at counts 0..7, interior layers at counts 2..8)."""
    shape = presets.shape("1b-tsumugi")
    sigs = layer_signatures(presets.TSUMUGI_LOOP, shape["num_layers"])
    assert len(sigs) == 72
    assert len(set(sigs)) == 15
    assert {c for c, b in set(sigs) if b} == set(range(8))
    assert {c for c, b in set(sigs) if not b} == set(range(2, 9))


@torch.inference_mode()
def test_regional_options_are_refused_where_they_do_not_apply(model: KotodamaModel) -> None:
    dec = MultiSampleDecoder(_engine(model), ROWS)
    with pytest.raises(ValueError, match="CUDA engine"):
        dec.compile_step(regional=True, backend="aot_eager", cuda_graph=True)
    with pytest.raises(ValueError, match="regional step only"):
        dec.compile_step(backend="aot_eager")


# ── DecodeEngine.compile_step(regional=True) ─────────────────────────────────

BUCKETS = (32, 8)  # block buckets that fit the test cache


def _regional_engine(model: KotodamaModel, backend="aot_eager", **kw) -> DecodeEngine:
    eng = _engine(model, **kw)
    eng.block_buckets = BUCKETS
    eng.compile_step(regional=True, backend=backend)
    return eng


def _bind_levers(eng: DecodeEngine) -> tuple[int, int]:
    """A uniform lever (it writes the prompt too) and a decode-span one; returns their slots."""
    uniform = eng.levers.bind(LeverPlan(attn={4: _vec(1, 0.8)}, mlp={8: _vec(4, 0.5)}, span="uniform"))
    decode = eng.levers.bind(LeverPlan(mlp={7: _vec(2, 0.8)}, span="decode"))
    return uniform, decode


def _eager_step(eng: DecodeEngine, token: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """What a compiled engine's step runs, uncompiled: the in-graph Gumbel-max samplers at temperature > 0 (an
    uncompiled engine's own step samples with multinomial instead)."""
    if params.temperature == 0.0:
        return eng.step(token, params)
    eng._sync_sampling_buffers(params)
    out = (eng._step_sampled_truncated if params.truncated else eng._step_sampled)(token)
    eng._pos_cpu += 1
    return out


def _same_engine_run(ref: DecodeEngine, reg: DecodeEngine, prompt: torch.Tensor, slot: int,
                     laws: list[SamplingParams], cached: bool = False) -> None:
    """Prefill ``prompt`` on both engines at lever ``slot`` (``cached``: through prefill_cached), then step each law
    a few times from the same RNG state: the same tokens, and logits within the parity tolerance."""
    for params in laws:
        if cached:
            (want, _), (got, info) = ref.prefill_cached(prompt, slot=slot), reg.prefill_cached(prompt, slot=slot)
        else:
            want, got = ref.prefill(prompt, slot=slot), reg.prefill(prompt, slot=slot)
        torch.testing.assert_close(got, want, **TOL)
        tok_ref = ref.sample_first(want, params).clone()
        reg.cur_token.copy_(tok_ref)
        tok_reg = reg.cur_token
        for t in range(5):
            torch.manual_seed(100 + t)
            a = _eager_step(ref, tok_ref, params).clone()
            torch.manual_seed(100 + t)
            b = reg.step(tok_reg, params)
            assert a.item() == b.item(), (params, t)
            tok_ref = a


@torch.inference_mode()
def test_regional_engine_matches_the_eager_engine(model: KotodamaModel) -> None:
    ref, reg = _engine(model), _regional_engine(model)
    ref.block_buckets = BUCKETS
    for eng in (ref, reg):
        eng.set_steer([(2, torch.from_numpy(_vec(3)), 0.5)])
        uniform, decode = _bind_levers(eng)
    laws = [SamplingParams(temperature=0.8, repetition_penalty=1.2),
            SamplingParams(temperature=0.9, repetition_penalty=1.1, top_k=20, top_p=0.8, min_p=0.05),
            SamplingParams(temperature=0.0, repetition_penalty=1.0)]
    # an exact bucket, a bucket plus an overlap chunk, and a two-bucket cover; each at no lever and both spans
    for n in (8, 13, 41):
        for slot in (0, uniform, decode):
            _same_engine_run(ref, reg, _ids(n, n + slot), slot, laws)

    # forced continuation: the regional forward's logits along the eager engine's
    prompt, forced = _ids(13, 7), _ids(1, 8).repeat(1, 1)
    ref.prefill(prompt, slot=decode), reg.prefill(prompt, slot=decode)
    for t in range(6):
        tok = _ids(1, 50 + t)
        torch.testing.assert_close(reg.step_logits(tok).clone(), ref.step_logits(tok), **TOL)

    # prefix-cached extends (the block regions extend a suffix; a 1-token suffix runs the step graph)
    base = _ids(24, 9)
    for eng in (ref, reg):
        eng.reset()
        eng.prefill(base)
    for extra in (_ids(11, 10), _ids(1, 11)):
        prompt = torch.cat([base, extra], dim=1)
        _same_engine_run(ref, reg, prompt, 0, laws[:1], cached=True)
        for eng in (ref, reg):
            eng.reset()
            eng.prefill(base)


@torch.inference_mode()
def test_regional_engine_compiles_one_graph_per_signature() -> None:
    deep = _model(DEEP)
    eng = _engine(deep)
    eng.block_buckets = BUCKETS
    counter = CompileCounterWithBackend("aot_eager")
    eng.compile_step(regional=True, backend=counter)
    n_sigs = len(set(layer_signatures(deep._attn_res_boundary_set, DEEP["num_layers"])))
    assert eng.regional_variants == n_sigs == 9
    params = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    logits = eng.prefill(_ids(8, 2))  # one 8-token block
    assert counter.frame_count == n_sigs + 1  # that bucket's layer signatures and the block tail
    tok = eng.sample_first(logits, params)
    eng.step(tok, params)
    assert counter.frame_count == 2 * n_sigs + 3  # + the step's signatures, the step tail and the sampler
    eng.step(eng.cur_token, params)
    eng.step(eng.cur_token, SamplingParams(temperature=0.7, repetition_penalty=1.2, top_p=0.9))
    assert counter.frame_count == 2 * n_sigs + 4  # the truncated sampler only
    eng.prefill(_ids(40, 3))  # 32 + 8: the 32-token bucket compiles its own regions, the tail recompiles for S=32
    assert counter.frame_count == 3 * n_sigs + 5


@pytest.mark.skipif(shutil.which("c++") is None and shutil.which("g++") is None,
                    reason="inductor's CPU backend needs a C++ compiler")
@torch.inference_mode()
def test_inductor_regional_engine_matches_the_eager_engine(model: KotodamaModel) -> None:
    ref, reg = _engine(model, levers=0, steer=False), _regional_engine(model, backend="inductor", levers=0,
                                                                       steer=False)
    ref.block_buckets = BUCKETS
    # inductor draws its sampling noise from its own generator, so sampled tokens are not comparable: greedy tokens
    # and forced logits are
    _same_engine_run(ref, reg, _ids(13, 3), 0, [SamplingParams(temperature=0.0, repetition_penalty=1.0)])
    ref.prefill(_ids(41, 4)), reg.prefill(_ids(41, 4))
    for t in range(4):
        tok = _ids(1, 60 + t)
        torch.testing.assert_close(reg.step_logits(tok).clone(), ref.step_logits(tok), **TOL)


@torch.inference_mode()
def test_regional_engine_refuses_what_does_not_apply(model: KotodamaModel) -> None:
    eng = _engine(model)
    with pytest.raises(ValueError, match="CUDA engine"):
        eng.compile_step(regional=True, backend="aot_eager", cuda_graph=True)
    with pytest.raises(ValueError, match="regional step only"):
        eng.compile_step(backend="aot_eager")
