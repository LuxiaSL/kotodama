"""Per-request lever tables (kotodama.model.levers) on a tiny looped model, CPU fp32.

What is pinned here:
  * the identity slot is exact: an engine with lever tables computes the same bits as one without, single-stream and
    batched, while other slots are bound;
  * ``write_plan`` is the block-persistent law (rules K/N/S) and agrees with the broadcast ``set_steer`` rows;
  * a lever's write lands where an eager hook on the reference forward lands it (``attn``: the attention output,
    ``mlp``: the MLP output, both keyed on the effective layer, so the two passes of a loop are distinct sites), for
    both spans, in the single-stream engine and in the batched decoders;
  * rows of one batch never mix: each row equals its solo run, and identity rows equal an engine without tables;
  * prompt KV written under a uniform lever is reused only by the same lever.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams, compute_steer_points
from kotodama.model.levers import LeverError, LeverPlan, LeverTables, neuron_row, write_plan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
ROWS = 4
H = CFG["hidden_size"]
TOL = dict(atol=2e-4, rtol=1e-4)


def _model() -> KotodamaModel:
    torch.manual_seed(20261026)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    return _model()


def _engine(model: KotodamaModel, levers: int = 0, steer: bool = False) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer, levers=levers)


def _dec(cls, eng: DecodeEngine) -> MultiSampleDecoder:
    return cls(eng, ROWS, tail_tokens=32) if cls is CascadeDecoder else cls(eng, ROWS)


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


def _vec(seed: int, scale: float = 1.0) -> np.ndarray:
    return (scale * np.random.default_rng(seed).standard_normal(H)).astype(np.float32)


@contextmanager
def _oracle(model: KotodamaModel, plan: LeverPlan, prompt_len: int):
    """The eager reference write: forward hooks on the attention output projection and the MLP of the reference
    model, armed per effective layer by that layer's own pre-norm (the looped repeats share attn/ffn but not their
    norms). ``decode`` span rows write positions >= prompt_len only (a full no-cache forward sees every position)."""
    state = {"attn": None, "mlp": None}
    handles = []
    rows = {k: {L: torch.from_numpy(r) for L, r in plan.rows(k).items()} for k in ("attn", "mlp")}

    def arm(kind, layer):
        def hook(_m, _a, _o):
            state[kind] = layer
        return hook

    def add(kind):
        def hook(_m, _a, out):
            layer, state[kind] = state[kind], None
            row = rows[kind].get(layer)
            if row is None:
                return out
            w = torch.ones(out.shape[1], dtype=out.dtype)
            if plan.span == "decode":
                w[:prompt_len] = 0.0
            return out + w.view(1, -1, 1) * row.to(out.dtype)
        return hook

    seen: set[tuple[str, int]] = set()
    for p, layer in enumerate(model.layers):
        handles.append(layer.attn_norm.register_forward_hook(arm("attn", p)))
        handles.append(layer.ffn_norm.register_forward_hook(arm("mlp", p)))
        for kind, mod in (("attn", layer.attn.o_proj), ("mlp", layer.ffn)):
            if (kind, id(mod)) not in seen:
                seen.add((kind, id(mod)))
                handles.append(mod.register_forward_hook(add(kind)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _oracle_logits(model: KotodamaModel, plan: LeverPlan, prompt: torch.Tensor, cont: torch.Tensor) -> torch.Tensor:
    """Next-token logits at the last prompt position and after each continuation token but the last."""
    with _oracle(model, plan, prompt.shape[1]):
        full = model(torch.cat([prompt, cont], dim=1))["logits"][0].float()
    return full[prompt.shape[1] - 1:-1]


# ── the identity slot ───────────────────────────────────────────────────────


def test_tables_start_identity_and_release_restores_it() -> None:
    t = LeverTables(5, 8, 3, torch.float32, torch.device("cpu"))
    assert t.is_identity() and t.free == 3
    s = t.bind(LeverPlan(attn={2: np.ones(8, np.float32)}))
    assert s == 1 and t.is_identity() and t.tables["attn"][2, 1].sum() == 8
    t.release(s)
    assert t.is_identity() and t.free == 3 and t.bound == ()
    with pytest.raises(LeverError):
        t.release(s)
    with pytest.raises(LeverError):
        t.bind(LeverPlan())
    with pytest.raises(LeverError):
        t.bind(LeverPlan(attn={5: np.ones(8, np.float32)}))
    with pytest.raises(LeverError):
        LeverPlan(mlp={1: np.full(8, np.nan, np.float32)})
    for _ in range(3):
        t.bind(LeverPlan(attn={0: np.ones(8, np.float32)}))
    with pytest.raises(LeverError):
        t.bind(LeverPlan(attn={0: np.ones(8, np.float32)}))


@torch.inference_mode()
def test_identity_slot_is_bit_exact(model: KotodamaModel) -> None:
    plain, lev = _engine(model), _engine(model, levers=3)
    lev.levers.bind(LeverPlan(attn={4: _vec(1)}, mlp={7: _vec(2)}))  # bound, not read: rows read slot 0
    prompt, cont = _ids(12, 1), _ids(6, 2)
    a, b = plain.prefill(prompt), lev.prefill(prompt)
    assert torch.equal(a, b)
    for t in range(cont.shape[1]):
        assert torch.equal(plain.step_logits(cont[:, t:t + 1]), lev.step_logits(cont[:, t:t + 1]))
    for cls in (MultiSampleDecoder, CascadeDecoder):
        plain.prefill(prompt), lev.prefill(prompt)
        dp, dl = _dec(cls, plain), _dec(cls, lev)
        params = SamplingParams(temperature=0.8, repetition_penalty=1.0)
        dp.start(a, params), dl.start(b, params)
        forced = torch.cat([_ids(5, 10 + r) for r in range(ROWS)])
        for t in range(forced.shape[1]):
            assert torch.equal(dp.step_logits(forced[:, t:t + 1]), dl.step_logits(forced[:, t:t + 1])), (cls, t)


# ── the write plan ──────────────────────────────────────────────────────────


def test_write_plan_rules() -> None:
    bounds, n = CFG["attn_res_boundaries"], CFG["num_layers"]
    rows = np.stack([_vec(1), _vec(2)])
    k = write_plan([2, 4], rows, bounds, n, "K")
    pts2, pts4 = compute_steer_points(2, bounds, n), compute_steer_points(4, bounds, n)
    assert sorted(k) == sorted(set(pts2) | set(pts4))
    for layer in k:
        want = np.zeros(H, np.float32)
        if layer in pts2:
            want = want + rows[0]
        if layer in pts4:
            want = want + rows[1]
        assert np.array_equal(k[layer], want)
    s = write_plan([2, 4], rows, bounds, n, "S")
    assert sorted(s) == [2, 4] and np.array_equal(s[4], rows[1])
    norms = {p: 1.0 + 0.5 * p for p in range(n)}
    nn_ = write_plan([4], rows[1:], bounds, n, "N", norms)
    for layer in pts4:
        assert np.array_equal(nn_[layer], rows[1] * np.float32(norms[layer] / norms[4]))
    with pytest.raises(LeverError):
        write_plan([4], rows[1:], bounds, n, "N")
    with pytest.raises(LeverError):
        write_plan([4, 4], rows, bounds, n, "K")


@torch.inference_mode()
def test_write_plan_matches_the_broadcast_steer_rows(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    unit = torch.from_numpy(_vec(3))
    norms = {p: 1.0 + 0.25 * p for p in range(CFG["num_layers"])}
    for rule in ("K", "N", "S"):
        eng.set_steer([(4, unit, 0.7)], rule=rule, norms=norms)
        plan = write_plan([4], (0.7 * unit.numpy())[None], CFG["attn_res_boundaries"], CFG["num_layers"], rule,
                          norms if rule == "N" else None)
        for layer in range(CFG["num_layers"]):
            want = torch.from_numpy(plan[layer]) if layer in plan else torch.zeros(H)
            torch.testing.assert_close(eng.steer_buf[layer], want, atol=1e-6, rtol=1e-6)
    eng.set_steer(None)


# ── writes land where the eager hooks land them ─────────────────────────────


def _plans() -> dict[str, LeverPlan]:
    attn = write_plan([4], _vec(5, 0.8)[None], CFG["attn_res_boundaries"], CFG["num_layers"], "K")
    return {
        "attn-K-uniform": LeverPlan(attn=attn, span="uniform"),
        "attn-K-decode": LeverPlan(attn=attn, span="decode"),
        "attn-S-pass2": LeverPlan(attn={7: _vec(6, 0.8)}, span="uniform"),  # effective 7 runs module 4's attention
        "mlp-pass1": LeverPlan(mlp={4: _vec(7, 0.8)}, span="uniform"),
        "mlp-pass2-decode": LeverPlan(mlp={7: _vec(7, 0.8)}, span="decode"),
    }


@pytest.mark.parametrize("name", list(_plans()))
@torch.inference_mode()
def test_single_stream_matches_the_eager_hooks(model: KotodamaModel, name: str) -> None:
    plan = _plans()[name]
    eng = _engine(model, levers=2)
    slot = eng.levers.bind(plan)
    prompt, cont = _ids(11, 3), _ids(6, 4)
    got = [eng.prefill(prompt, slot=slot)[0]]
    for t in range(cont.shape[1] - 1):
        got.append(eng.step_logits(cont[:, t:t + 1])[0, -1].float())
    want = _oracle_logits(model, plan, prompt, cont)
    torch.testing.assert_close(torch.stack(got), want, **TOL)
    plain = _oracle_logits(model, LeverPlan(attn={0: np.zeros(H, np.float32)}), prompt, cont)
    assert (want - plain).abs().max() > 1e-3  # the write changes the forward


@torch.inference_mode()
def test_looped_passes_are_distinct_sites(model: KotodamaModel) -> None:
    """Effective 4 and 7 run one attention module (and one MLP); a write at one pass is not a write at the other."""
    assert model.layers[4].attn is model.layers[7].attn and model.layers[4].ffn is model.layers[7].ffn
    prompt, cont = _ids(10, 5), _ids(5, 6)
    v = _vec(8, 0.8)
    for kind in ("attn", "mlp"):
        out = []
        for layer in (4, 7):
            plan = LeverPlan(**{kind: {layer: v}}, span="uniform")
            eng = _engine(model, levers=1)
            slot = eng.levers.bind(plan)
            got = [eng.prefill(prompt, slot=slot)[0]]
            for t in range(cont.shape[1] - 1):
                got.append(eng.step_logits(cont[:, t:t + 1])[0, -1].float())
            got_t = torch.stack(got)
            torch.testing.assert_close(got_t, _oracle_logits(model, plan, prompt, cont), **TOL)
            out.append(got_t)
        assert (out[0] - out[1]).abs().max() > 1e-3, kind


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_mixed_rows_match_their_solo_runs(model: KotodamaModel, cls) -> None:
    """Decode-span levers row by row in one batch (slot 0 rows included): each row matches the eager hooks of its
    own lever, and identity rows match an engine without tables bit for bit."""
    plans = _plans()
    eng = _engine(model, levers=3)
    s_attn = eng.levers.bind(plans["attn-K-decode"])
    s_mlp = eng.levers.bind(plans["mlp-pass2-decode"])
    slots = [s_attn, 0, s_mlp, 0]
    prompt = _ids(9, 7)
    forced = torch.cat([_ids(6, 30 + r) for r in range(ROWS)])
    params = SamplingParams(temperature=0.8, repetition_penalty=1.0)
    dec = _dec(cls, eng)
    logits0 = eng.prefill(prompt)
    dec.admit(list(range(ROWS)), logits0, params, slots=slots)
    got = [dec.step_logits(forced[:, t:t + 1]).float() for t in range(forced.shape[1] - 1)]
    got_t = torch.stack(got, dim=1)  # (ROWS, steps, V)
    for r, s in enumerate(slots):
        plan = eng.levers.plan(s) or LeverPlan(attn={0: np.zeros(H, np.float32)}, span="decode")
        want = _oracle_logits(model, plan, prompt, forced[r:r + 1])[1:]
        torch.testing.assert_close(got_t[r], want, **TOL)
    plain = _engine(model)
    ref = _dec(cls, plain)
    ref.admit(list(range(ROWS)), plain.prefill(prompt), params)
    want = torch.stack([ref.step_logits(forced[:, t:t + 1]).float() for t in range(forced.shape[1] - 1)], dim=1)
    for r in (1, 3):
        assert torch.equal(got_t[r], want[r])


@torch.inference_mode()
def test_rows_must_match_the_prompt_lever(model: KotodamaModel) -> None:
    plans = _plans()
    eng = _engine(model, levers=3)
    s_uni = eng.levers.bind(plans["attn-K-uniform"])
    s_dec = eng.levers.bind(plans["attn-K-decode"])
    dec = MultiSampleDecoder(eng, ROWS)
    params = SamplingParams(temperature=0.8, repetition_penalty=1.0)
    prompt = _ids(9, 8)
    logits = eng.prefill(prompt)  # unsteered prompt
    dec.admit([0, 1], logits, params, slots=[0, s_dec])
    with pytest.raises(ValueError):
        dec.admit([2], logits, params, slots=s_uni)
    logits = eng.prefill(prompt, slot=s_uni)
    dec.admit([2, 3], logits, params, slots=s_uni)
    with pytest.raises(ValueError):
        dec.admit([0], logits, params, slots=0)


@torch.inference_mode()
def test_cached_prefix_is_reused_only_under_its_lever(model: KotodamaModel) -> None:
    plans = _plans()
    eng = _engine(model, levers=3)
    s_uni = eng.levers.bind(plans["attn-K-uniform"])
    s_dec = eng.levers.bind(plans["attn-K-decode"])
    base = _ids(30, 9)
    longer = torch.cat([base, _ids(4, 10)], dim=1)
    eng.prefill(base)
    _, info = eng.prefill_cached(longer, slot=s_uni)  # unsteered KV, uniform lever: no reuse
    assert not info["prefix_hit"] and eng.prompt_kv_len == 0
    _, info = eng.prefill_cached(longer, slot=s_dec)  # uniform KV, decode lever (unsteered prompt): no reuse
    assert not info["prefix_hit"] and eng.prompt_kv_len == longer.shape[1]
    _, info = eng.prefill_cached(longer, slot=0)  # unsteered KV, unsteered prompt: reuse
    assert info["prefix_hit"]


def test_neuron_row_is_a_down_projection_column(model: KotodamaModel) -> None:
    row = neuron_row(model, 7, 5, 2.5)
    want = 2.5 * model.layers[7].ffn.down_proj.weight[:, 5].detach().numpy()
    np.testing.assert_allclose(row, want, rtol=1e-6)
    with pytest.raises(LeverError):
        neuron_row(model, 7, CFG["intermediate_size"], 1.0)


def test_engine_without_tables_refuses_slots(model: KotodamaModel) -> None:
    eng = _engine(model)
    with pytest.raises(RuntimeError):
        eng.prefill(_ids(8, 11), slot=1)
    dec = MultiSampleDecoder(eng, ROWS)
    with pytest.raises(RuntimeError):
        dec.set_slots([0], [1])


# ── forced prompts, greedy rows, readouts, captures ─────────────────────────


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_forced_prompts_write_uniform_levers_per_row(model: KotodamaModel, cls) -> None:
    """admit_empty + force: every row writes its own prompt under its own lever; the decode that follows matches
    the eager hooks of that lever over prompt + continuation."""
    plans = _plans()
    eng = _engine(model, levers=3)
    s1 = eng.levers.bind(plans["attn-K-uniform"])
    s2 = eng.levers.bind(plans["mlp-pass1"])
    slots = [s1, 0, s2, 0]
    prompt = _ids(9, 12)
    forced = torch.cat([_ids(6, 40 + r) for r in range(ROWS)])
    dec = _dec(cls, eng)
    dec.admit_empty(slots)
    dec.force(prompt, SamplingParams(temperature=0.0, repetition_penalty=1.0))
    got = torch.stack([dec.step_logits(forced[:, t:t + 1]).float() for t in range(forced.shape[1] - 1)], dim=1)
    for r, s in enumerate(slots):
        plan = eng.levers.plan(s) or LeverPlan(attn={0: np.zeros(H, np.float32)})
        want = _oracle_logits(model, plan, prompt, forced[r:r + 1])[1:]
        torch.testing.assert_close(got[r], want, **TOL)


@pytest.mark.parametrize("cls", [MultiSampleDecoder, CascadeDecoder])
@torch.inference_mode()
def test_greedy_rows_and_readouts(model: KotodamaModel, cls) -> None:
    """Temperature 0 decodes the argmax; the readouts are the chosen token's logprob and the top-k under the
    model's own distribution, for the first token (from the prefill) and every step after."""
    eng = _engine(model, levers=2)
    s = eng.levers.bind(_plans()["attn-K-decode"])
    dec = cls(eng, ROWS, tail_tokens=32, readout_k=3) if cls is CascadeDecoder else cls(eng, ROWS, readout_k=3)
    prompt = _ids(10, 13)
    params = SamplingParams(temperature=0.0, repetition_penalty=1.0)
    slots = [s, 0, s, 0]
    dec.admit(list(range(ROWS)), eng.prefill(prompt), params, slots=slots)
    toks = [dec.cur_tokens.view(-1).tolist()]
    lps = [dec.step_logprob.clone()]
    tops = [dec.step_top_ids.clone()]
    for _ in range(5):
        toks.append(dec.step(params).view(-1).tolist())
        lps.append(dec.step_logprob.clone())
        tops.append(dec.step_top_ids.clone())
    for r in range(ROWS):
        gen = torch.tensor([[t[r] for t in toks]])
        plan = eng.levers.plan(slots[r]) or LeverPlan(attn={0: np.zeros(H, np.float32)}, span="decode")
        want = _oracle_logits(model, plan, prompt, gen)  # logits for each generated token
        want = torch.cat([want, want[-1:]])[: gen.shape[1]]
        lp = torch.log_softmax(want, -1)
        assert gen[0].tolist() == want.argmax(-1).tolist(), r
        torch.testing.assert_close(torch.stack([x[r] for x in lps]), lp.gather(1, gen.view(-1, 1)).view(-1),
                                   atol=1e-4, rtol=1e-4)
        assert torch.stack([x[r] for x in tops]).tolist() == lp.topk(3, -1).indices.tolist()


@torch.inference_mode()
def test_capture_matches_the_instrumented_forward(model: KotodamaModel) -> None:
    from kotodama.model.capture import instrumented_forward
    from kotodama.model.levers import capture

    eng = _engine(model, levers=1)
    prompt, cont = _ids(10, 14), _ids(4, 15)
    layers = [2, 4, 7]
    with capture(eng, layers=layers) as cap:
        eng.prefill(prompt)
        for t in range(cont.shape[1]):
            eng.step_logits(cont[:, t:t + 1])
    ref = instrumented_forward(model, torch.cat([prompt, cont], dim=1), light=True)[1]["residual_partial"]
    for L in layers:
        got = cap.sequence("partial_post_attn", L)
        torch.testing.assert_close(got, ref[2 * L][0].float(), **TOL)
        assert cap.sequence("mlp_in", L).shape == got.shape
    assert eng._taps is None


@torch.inference_mode()
def test_decoder_capture_buffer_holds_the_last_prompt_mlp_input(model: KotodamaModel) -> None:
    from kotodama.model.levers import capture

    eng = _engine(model, levers=2)
    s = eng.levers.bind(_plans()["attn-K-uniform"])
    dec = MultiSampleDecoder(eng, ROWS, capture_layers=(4, 7))
    prompt = _ids(8, 16)
    with capture(eng, dec, layers=[4, 7], what=("mlp_in",)) as cap:
        dec.admit_empty([s, 0, s, 0])
        dec.force(prompt, SamplingParams(temperature=0.0, repetition_penalty=1.0))
    rows = cap.rows(dec)["mlp_in"]
    for i, L in enumerate((4, 7)):
        torch.testing.assert_close(dec.mlp_in[:, i].float(), rows[L][:, -1], atol=0, rtol=0)
    assert not torch.allclose(dec.mlp_in[0], dec.mlp_in[1])  # the uniform lever changed the prompt's last state
    # identity rows at different batch positions: equal up to how the batched GEMM rounds by row
    torch.testing.assert_close(dec.mlp_in[1], dec.mlp_in[3], **TOL)
