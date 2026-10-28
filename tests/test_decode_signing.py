"""Signing inside the decode step (kotodama.model.signing) on a tiny looped model, CPU fp32.

What is pinned here:
  * every surface a signed row decoder captures over a row's generated span (hidden, pre-RoPE q / k, v, pre-SiLU
    gate, both routings, and the depth path) equals the replay capture's (kotodama.model.capture.cached_forward over
    the decoded sequence) at the same positions, for the eager step and the regional step, for both row decoders;
  * signing does not change generation: a signed and an unsigned regional decoder give the same tokens and readouts,
    bit for bit;
  * the span bookkeeping: a row's span starts at its admission, a re-admitted row starts over, a row past the tail
    is reported incomplete, and the row's lever slot and steering are recorded;
  * signing compiles only the sampled layers' signatures on top of the unsigned ones.
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

from kotodama.model.capture import cached_forward, hidden_rows
from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.levers import LeverPlan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder
from kotodama.model.signing import SignSpec, sign_bytes

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
ROWS = 4
# a boundary layer with no committed block (0), a boundary layer (3), interior layers (5, 8) and the last layer (10)
LAYERS = (0, 3, 5, 8, 10)
TOL = dict(atol=2e-4, rtol=1e-4)
STEPS = 6
PARAMS = SamplingParams(temperature=0.8, repetition_penalty=1.2)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261028)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


def _engine(model: KotodamaModel, levers: int = 0, steer: bool = False) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer, levers=levers)


def _dec(cls, eng: DecodeEngine, sign, **kw) -> MultiSampleDecoder:
    if cls is CascadeDecoder:
        return cls(eng, ROWS, tail_tokens=32, seeded=True, sign=sign, **kw)
    return cls(eng, ROWS, seeded=True, sign=sign, **kw)


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


def _decode(eng: DecodeEngine, dec: MultiSampleDecoder, prompt: torch.Tensor, steps: int = STEPS,
            slots=None) -> list[list[int]]:
    """Admit every row from ``prompt`` and decode ``steps`` steps; each row's whole sequence."""
    dec.release(range(ROWS))
    first = dec.admit(list(range(ROWS)), eng.prefill(prompt), PARAMS, slots=slots, seeds=[3, 4, 5, 6])
    seqs = [prompt[0].tolist() + [t] for t in first]
    for _ in range(steps):
        toks = dec.step(PARAMS).view(-1).tolist()
        for r in range(ROWS):
            seqs[r].append(toks[r])
    return seqs


def _oracle(model: KotodamaModel, seq: list[int], P: int, n: int) -> dict[str, torch.Tensor]:
    """The replay capture's surfaces at span positions P..P+n-1, per layer, in the export's layouts."""
    _, cap, _ = cached_forward(model, torch.tensor([seq]), past=None, position_offset=0, light=False,
                               capture_hidden=True)
    hid = hidden_rows(cap, CFG["num_layers"])
    span = slice(P, P + n)
    pre_attn = {i: w for kind, i, w in cap["routing"] if kind == "pre_attn"}
    pre_mlp = {i: w for kind, i, w in cap["routing"] if kind == "pre_mlp"}
    out: dict[str, list] = {k: [] for k in ("hidden", "depth", "q", "k", "v", "gate", "route_attn", "route_mlp")}
    for li in range(CFG["num_layers"]):
        out["depth"].append(hid[li + 1][0, span])
    for li in LAYERS:
        out["hidden"].append(hid[li + 1][0, span])
        for name, key in (("q", "queries"), ("k", "keys"), ("v", "values")):
            out[name].append(cap[key][li][0, :, span].transpose(0, 1))  # (n, heads, head_dim)
        out["gate"].append(cap["gate"][li][0, span])
        w = pre_attn.get(li)
        out["route_attn"].append(None if w is None else w[:, 0, span].transpose(0, 1))  # (n, sources)
        out["route_mlp"].append(pre_mlp[li][:, 0, span].transpose(0, 1))
    return {k: (v if k.startswith("route") else torch.stack(v, dim=1)) for k, v in out.items()}


def _check_row(model: KotodamaModel, signed, seq: list[int], P: int, depth: bool) -> None:
    assert signed.start == P and signed.positions == STEPS and signed.complete
    want = _oracle(model, seq, P, STEPS)
    s = signed.surfaces
    for name in ("hidden", "q", "k", "v", "gate") + (("depth",) if depth else ()):
        torch.testing.assert_close(s[name], want[name], **TOL, msg=name)
    for j, li in enumerate(LAYERS):
        n_attn, n_mlp = signed.route_sources[j]
        wa = want["route_attn"][j]
        assert n_attn == (0 if wa is None else wa.shape[1]), li
        if wa is not None:
            torch.testing.assert_close(s["route"][:, j, 0, :n_attn], wa, **TOL)
        torch.testing.assert_close(s["route"][:, j, 0, n_attn:], torch.zeros_like(s["route"][:, j, 0, n_attn:]))
        torch.testing.assert_close(s["route"][:, j, 1, :n_mlp], want["route_mlp"][j], **TOL)


@pytest.mark.parametrize("depth", [False, True])
@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_signed_surfaces_match_the_replay_capture(model: KotodamaModel, cls, regional: bool, depth: bool) -> None:
    eng = _engine(model)
    dec = _dec(cls, eng, SignSpec(LAYERS, tail=16, depth=depth))
    if regional:
        dec.compile_step(regional=True, backend="aot_eager")
    P = 10
    seqs = _decode(eng, dec, _ids(P, 1))
    for r in range(ROWS):
        _check_row(model, dec.signed(r), seqs[r], P, depth)


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_signing_does_not_change_generation(model: KotodamaModel, cls) -> None:
    eng = _engine(model, levers=3, steer=True)
    s1 = eng.levers.bind(LeverPlan(attn={4: np.full(CFG["hidden_size"], 0.3, np.float32)}, span="decode"))
    eng.set_steer([(2, torch.full((CFG["hidden_size"],), 0.2), 0.5)])
    off = _dec(cls, eng, None, readout_k=3)
    on = _dec(cls, eng, SignSpec(LAYERS, tail=16, depth=True), readout_k=3)
    for dec in (off, on):
        dec.compile_step(regional=True, backend="aot_eager")
    prompt = _ids(9, 2)
    for params in (PARAMS, SamplingParams(temperature=0.9, repetition_penalty=1.1, top_p=0.8, min_p=0.05)):
        logits = eng.prefill(prompt)
        for dec in (off, on):
            dec.release(range(ROWS))
            dec.admit(list(range(ROWS)), logits, params, slots=[s1, 0, s1, 0], seeds=[7, 8, 9, 10])
        for _ in range(8):
            assert off.step(params).view(-1).tolist() == on.step(params).view(-1).tolist()
            assert torch.equal(off.step_logprob, on.step_logprob)
            assert torch.equal(off.step_top_lps, on.step_top_lps)
    row = on.signed(0)
    assert row.lever_slot == s1 and row.steered and on.signed(1).lever_slot == 0


@pytest.mark.skipif(shutil.which("c++") is None and shutil.which("g++") is None,
                    reason="inductor's CPU backend needs a C++ compiler")
@torch.inference_mode()
def test_inductor_signed_regions(model: KotodamaModel) -> None:
    """The regions as the server builds them (inductor): signed vs unsigned generation bitwise equal, and the
    surfaces within the parity tolerance of the replay capture."""
    eng = _engine(model)
    off, on = _dec(CascadeDecoder, eng, None), _dec(CascadeDecoder, eng, SignSpec(LAYERS, tail=16))
    for dec in (off, on):
        dec.compile_step(regional=True)
    P = 8
    prompt = _ids(P, 3)
    a, b = _decode(eng, off, prompt), _decode(eng, on, prompt)
    assert a == b
    for r in range(ROWS):
        _check_row(model, on.signed(r), b[r], P, depth=False)


@torch.inference_mode()
def test_span_bookkeeping(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = _dec(MultiSampleDecoder, eng, SignSpec(LAYERS, tail=4))
    _decode(eng, dec, _ids(10, 4), steps=3)
    row = dec.signed(2)
    assert (row.start, row.positions, row.kept, row.complete) == (10, 3, 3, True)
    assert row.surfaces["hidden"].shape == (3, len(LAYERS), CFG["hidden_size"])
    assert row.surfaces["q"].shape == (3, len(LAYERS), 4, 16) and row.surfaces["route"].shape[:3] == (3, 5, 2)
    _decode(eng, dec, _ids(7, 5), steps=6)  # re-admitted from a shorter prompt, past the tail
    row = dec.signed(2)
    assert (row.start, row.positions, row.kept, row.complete) == (7, 6, 4, False)
    with pytest.raises(RuntimeError, match="without signing"):
        _dec(MultiSampleDecoder, eng, None).signed(0)
    with pytest.raises(ValueError, match="outside"):
        _dec(MultiSampleDecoder, eng, SignSpec((3, 11), tail=4))
    with pytest.raises(ValueError, match="sorted"):
        SignSpec((5, 3), tail=4)
    spec = SignSpec(LAYERS, tail=4, depth=True)
    sized = sign_bytes(spec, ROWS, hidden=64, n_heads=4, n_kv_heads=2, head_dim=16, intermediate=128,
                       n_layers=11, route=dec._sign_route, dtype=torch.float32)
    built = _dec(MultiSampleDecoder, eng, spec)
    assert sized == sum(getattr(built, f"sign_{n}").numel() * 4 for n in built.sign_names)


@torch.inference_mode()
def test_signing_compiles_only_the_sampled_signatures(model: KotodamaModel) -> None:
    eng = _engine(model)
    plain = _dec(MultiSampleDecoder, eng, None)
    signed = _dec(MultiSampleDecoder, eng, SignSpec(LAYERS, tail=8))
    base = len(set(plain.regional_signatures()))
    extra = len({s for s in signed.regional_signatures() if s[3]})
    assert extra == len({s[:3] for s, li in zip(signed.regional_signatures(), range(11)) if li in LAYERS})
    counter = CompileCounterWithBackend("aot_eager")
    signed.compile_step(regional=True, backend=counter)
    assert signed.regional_variants == len(set(signed.regional_signatures()))
    unsigned_left = {s for li, s in enumerate(signed.regional_signatures()) if li not in LAYERS}
    assert signed.regional_variants == len(unsigned_left) + extra <= base + extra
    signed.admit(list(range(ROWS)), eng.prefill(_ids(8, 6)), PARAMS)
    signed.step(PARAMS)
    assert counter.frame_count == signed.regional_variants + 2  # the signatures, the tail and the sampler
