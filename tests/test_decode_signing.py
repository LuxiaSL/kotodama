"""Signing inside the decode step (kotodama.model.signing) on a tiny looped model, CPU fp32.

What is pinned here:
  * every surface a signed row decoder captures over a row's generated span (hidden, pre-RoPE q / k, v, pre-SiLU
    gate; both routings at every layer and the final routing; the embedding and the committed blocks; the depth
    path; the next-token logit statistics against the chosen token) equals the replay capture's (kotodama.model.capture.cached_forward over
    the decoded sequence) at the same positions, for the eager step and the regional step, for both row decoders;
  * so do the attention statistics (per-head and head-mean entropy at every layer; per-head prompt, recency and sink
    mass and the head-mean rows at the sampled layers), computed from the replay's attention weights;
  * signing does not change generation: a signed and an unsigned regional decoder give the same tokens and readouts,
    bit for bit;
  * replay through the engine: scoring a generated row's own tokens from the same prompt signs it bit for bit as its
    generation did (every surface, the logit statistics included);
  * the span bookkeeping: a row's span starts at its admission, a re-admitted row starts over, a row past the tail
    is reported incomplete, and the row's lever slot and steering are recorded;
  * signing compiles at most the sampled layers' signatures on top of the unsigned ones' count.
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
from kotodama.model.signing import SignSpec, sign_bytes, weights_digest

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


def _logit_stats(logits: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """entropy, top-1, top-5 mass, rank, chosen prob, logprob of each row (n, V) at its chosen token (n,), fp64."""
    x = logits.double()
    x = x - x.max(-1, keepdim=True).values
    p = torch.softmax(x, -1)
    lp = torch.log_softmax(x, -1)
    xc = x.gather(1, chosen.view(-1, 1))
    return torch.cat([-(p * lp).sum(-1, keepdim=True), p.max(-1, keepdim=True).values,
                      p.topk(5, -1).values.sum(-1, keepdim=True), (x > xc).sum(-1, keepdim=True).double(),
                      p.gather(1, chosen.view(-1, 1)), lp.gather(1, chosen.view(-1, 1))], dim=-1)


def _oracle(model: KotodamaModel, seq: list[int], P: int, n: int) -> dict:
    """The replay capture's surfaces at span positions P..P+n-1, per layer, in the export's layouts."""
    logits, cap, _ = cached_forward(model, torch.tensor([seq]), past=None, position_offset=0, light=False,
                                    capture_hidden=True)
    hid = hidden_rows(cap, CFG["num_layers"])
    span = slice(P, P + n)
    pre_attn = {i: w for kind, i, w in cap["routing"] if kind == "pre_attn"}
    pre_mlp = {i: w for kind, i, w in cap["routing"] if kind == "pre_mlp"}
    out: dict[str, list] = {k: [] for k in ("hidden", "depth", "q", "k", "v", "gate", "route_attn", "route_mlp")}
    for li in range(CFG["num_layers"]):
        out["depth"].append(hid[li + 1][0, span])
        w = pre_attn.get(li)
        out["route_attn"].append(None if w is None else w[:, 0, span].transpose(0, 1))  # (n, sources)
        out["route_mlp"].append(pre_mlp[li][:, 0, span].transpose(0, 1))
    for li in LAYERS:
        out["hidden"].append(hid[li + 1][0, span])
        for name, key in (("q", "queries"), ("k", "keys"), ("v", "values")):
            out[name].append(cap[key][li][0, :, span].transpose(0, 1))  # (n, heads, head_dim)
        out["gate"].append(cap["gate"][li][0, span])
    res = {k: (v if k.startswith("route") else torch.stack(v, dim=1)) for k, v in out.items()}
    final = [w for kind, _, w in cap["routing"] if kind == "final"][0]
    res["route_final"] = final[:, 0, span].transpose(0, 1)
    res["committed"] = [c[0, span] for c in cap["committed"]]
    res["embed"] = res["committed"][0]
    res["logit_stats"] = _logit_stats(logits[0, span], torch.tensor(seq[P + 1:P + 1 + n]))
    return res


def _check_row(model: KotodamaModel, signed, seq: list[int], P: int, depth: bool) -> None:
    assert signed.start == P and signed.positions == STEPS and signed.complete
    want = _oracle(model, seq, P, STEPS)
    s = signed.surfaces
    for name in ("hidden", "q", "k", "v", "gate", "embed") + (("depth",) if depth else ()):
        torch.testing.assert_close(s[name], want[name], **TOL, msg=name)
    for li in range(CFG["num_layers"]):
        n_attn, n_mlp = signed.route_sources[li]
        wa = want["route_attn"][li]
        assert n_attn == (0 if wa is None else wa.shape[1]), li
        if wa is not None:
            torch.testing.assert_close(s["route"][:, li, 0, :n_attn], wa, **TOL)
        torch.testing.assert_close(s["route"][:, li, 0, n_attn:], torch.zeros_like(s["route"][:, li, 0, n_attn:]))
        torch.testing.assert_close(s["route"][:, li, 1, :n_mlp], want["route_mlp"][li], **TOL)
    nf = signed.final_sources
    assert nf == want["route_final"].shape[1]
    torch.testing.assert_close(s["route_final"][:, :nf], want["route_final"], **TOL)
    got, ref = s["logit_stats"], want["logit_stats"]
    assert torch.equal(got[:, 3], ref[:, 3])  # the chosen token's rank
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
    if depth:
        assert len(s["committed"]) == len(want["committed"]) == len(CFG["attn_res_boundaries"])
        for a, b in zip(s["committed"], want["committed"]):
            torch.testing.assert_close(a, b, **TOL)


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


def _xlogx(p: torch.Tensor) -> torch.Tensor:
    return -torch.special.xlogy(p, p).sum(-1)


@pytest.mark.parametrize("regional", [False, True])
@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_attention_statistics_match_the_replay_weights(model: KotodamaModel, cls, regional: bool) -> None:
    eng = _engine(model)
    width = 40
    dec = _dec(cls, eng, SignSpec(LAYERS, tail=16, attn=True, width=width))
    if regional:
        dec.compile_step(regional=True, backend="aot_eager")
    P = 11
    seqs = _decode(eng, dec, _ids(P, 7))
    L = CFG["num_layers"]
    for r in range(ROWS):
        row = dec.signed(r)
        _, cap, _ = cached_forward(model, torch.tensor([seqs[r]]), past=None, position_offset=0, light=False,
                                   capture_hidden=True)
        aw = [cap["attn"][li][0].float() for li in range(L)]  # (heads, S, S)
        p = torch.stack([w[:, P:P + STEPS] for w in aw], dim=0).permute(2, 0, 1, 3)  # (n, L, heads, S)
        torch.testing.assert_close(row.surfaces["head_entropy"], _xlogx(p), **TOL)
        torch.testing.assert_close(row.surfaces["mean_entropy"], _xlogx(p.mean(2)), **TOL)
        ps = p[:, list(LAYERS)]  # (n, layers, heads, S)
        S = ps.shape[-1]
        cols = torch.arange(S)
        lens = torch.arange(P, P + STEPS) + 1
        cut = torch.clamp((lens * 0.8).long(), min=1).view(-1, 1, 1, 1)
        want = torch.stack([(ps * (cols < P)).sum(-1), (ps * (cols >= cut)).sum(-1), ps[..., 0]], dim=2)
        torch.testing.assert_close(row.surfaces["head_scalars"], want, **TOL)
        rows = torch.zeros(STEPS, len(LAYERS), width)
        rows[..., :S] = ps.mean(2)
        torch.testing.assert_close(row.surfaces["mean_rows"], rows, **TOL)


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_signing_does_not_change_generation(model: KotodamaModel, cls) -> None:
    eng = _engine(model, levers=3, steer=True)
    s1 = eng.levers.bind(LeverPlan(attn={4: np.full(CFG["hidden_size"], 0.3, np.float32)}, span="decode"))
    eng.set_steer([(2, torch.full((CFG["hidden_size"],), 0.2), 0.5)])
    off = _dec(cls, eng, None, readout_k=3)
    on = _dec(cls, eng, SignSpec(LAYERS, tail=16, depth=True, attn=True, width=48), readout_k=3)
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
    assert row.surfaces["q"].shape == (3, len(LAYERS), 4, 16) and row.surfaces["route"].shape[:3] == (3, 11, 2)
    assert row.surfaces["logit_stats"].shape == (3, 6) and row.surfaces["logit_stats"].dtype == torch.float64
    _decode(eng, dec, _ids(7, 5), steps=6)  # re-admitted from a shorter prompt, past the tail
    row = dec.signed(2)
    assert (row.start, row.positions, row.kept, row.complete) == (7, 6, 4, False)
    with pytest.raises(RuntimeError, match="without signing"):
        _dec(MultiSampleDecoder, eng, None).signed(0)
    with pytest.raises(ValueError, match="outside"):
        _dec(MultiSampleDecoder, eng, SignSpec((3, 11), tail=4))
    with pytest.raises(ValueError, match="sorted"):
        SignSpec((5, 3), tail=4)
    spec = SignSpec(LAYERS, tail=4, depth=True, attn=True, width=24)
    sized = sign_bytes(spec, ROWS, hidden=64, n_heads=4, n_kv_heads=2, head_dim=16, intermediate=128,
                       n_layers=11, route=dec._sign_route, dtype=torch.float32)
    built = _dec(MultiSampleDecoder, eng, spec)
    assert sized == sum(getattr(built, f"sign_{n}").numel() * getattr(built, f"sign_{n}").element_size()
                        for n in built.sign_names)
    with pytest.raises(ValueError, match="width"):
        SignSpec(LAYERS, tail=4, attn=True)


@torch.inference_mode()
def test_signing_compiles_a_bounded_number_of_signatures(model: KotodamaModel) -> None:
    eng = _engine(model)
    plain = _dec(MultiSampleDecoder, eng, None)
    signed = _dec(MultiSampleDecoder, eng, SignSpec(LAYERS, tail=8))
    base = len(set(plain.regional_signatures()))
    sigs = signed.regional_signatures()
    assert {s[3] for li, s in enumerate(sigs) if li in LAYERS} == {2}
    assert {s[3] for li, s in enumerate(sigs) if li not in LAYERS} == {1}
    # every unsigned signature becomes one routing-only signature; the sampled layers add theirs
    extra = len({s for s in sigs if s[3] == 2})
    counter = CompileCounterWithBackend("aot_eager")
    signed.compile_step(regional=True, backend=counter)
    assert signed.regional_variants == len(set(sigs))
    assert signed.regional_variants == len({s for li, s in enumerate(sigs) if li not in LAYERS}) + extra
    assert signed.regional_variants <= base + extra
    signed.admit(list(range(ROWS)), eng.prefill(_ids(8, 6)), PARAMS)
    signed.step(PARAMS)
    assert counter.frame_count == signed.regional_variants + 2  # the signatures, the tail and the sampler


def _digest_by_rule(model: torch.nn.Module) -> str:
    """anamnesis's lane weights rule (v1) written out independently: sorted state_dict keys; name, dtype, shape,
    raw loaded bytes."""
    import hashlib

    h = hashlib.sha256()
    for name in sorted(model.state_dict()):
        t = model.state_dict()[name].detach().contiguous().cpu()
        h.update(f"{name}\0{t.dtype}\0{','.join(map(str, t.shape))}\0".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.inference_mode()
def test_signed_rows_name_their_weights(model: KotodamaModel) -> None:
    digest = weights_digest(model)
    assert digest == _digest_by_rule(model)
    eng = _engine(model)
    dec = _dec(MultiSampleDecoder, eng, SignSpec(LAYERS, tail=4, weights=digest))
    _decode(eng, dec, _ids(6, 8), steps=2)
    assert dec.signed(0).weights == digest
    with pytest.raises(ValueError, match="declares weights"):
        _dec(MultiSampleDecoder, eng, SignSpec(LAYERS, tail=4, weights="0" * 64))
    other = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    assert weights_digest(other) != digest
    bf16 = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    bf16.load_state_dict(model.state_dict())
    assert weights_digest(bf16) == digest and weights_digest(bf16.to(torch.bfloat16)) != digest  # loaded dtype


def test_weights_digest_matches_anamnesis() -> None:
    """The same rule as anamnesis's own ``weights_digest``, where anamnesis is installed."""
    anamnesis_lane = pytest.importorskip("anamnesis.lane_weights", reason="anamnesis with weights_digest not installed")
    torch.manual_seed(3)
    tiny = KotodamaModel(KotodamaModelConfig(**dict(CFG, num_layers=4, loop_start=1, loop_len=2,
                                                    loop_times=1, attn_res_boundaries=[0, 2])))
    assert weights_digest(tiny) == anamnesis_lane.weights_digest(tiny)


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_scoring_a_generated_row_replays_its_signature(model: KotodamaModel, cls) -> None:
    eng = _engine(model)
    dec = _dec(cls, eng, SignSpec(LAYERS, tail=16, depth=True, attn=True, width=40))
    dec.compile_step(regional=True, backend="aot_eager")
    P = 9
    prompt = _ids(P, 11)
    seqs = _decode(eng, dec, prompt)
    generated = [dec.signed(r) for r in range(ROWS)]
    dec.release(range(ROWS))
    dec.admit(list(range(ROWS)), eng.prefill(prompt), PARAMS, seeds=[1, 2, 3, 4])
    dec.score(torch.tensor([s[P:] for s in seqs]))  # the first token, then every generated token
    for r in range(ROWS):
        replayed = dec.signed(r)
        assert (replayed.positions, replayed.start) == (generated[r].positions, generated[r].start)
        for name, x in generated[r].surfaces.items():
            ys = replayed.surfaces[name]
            if isinstance(x, list):
                assert all(torch.equal(a, b) for a, b in zip(x, ys)), name
            else:
                assert torch.equal(x, ys), name

