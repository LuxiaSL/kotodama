"""Per-request levers through the batcher and the server (--levers), tiny looped model, CPU fp32.

The batcher: steered and unsteered requests share one batch, each request's rows decode under its own lever exactly
as the single-stream engine decodes that lever, a request waits while every lever slot is bound, and no slot outlives
its request. The server: bank vectors, raw injections and the uniform span map onto levers; a uniform request
without --levers, or an injection the plan refuses, is a 400 before anything is engaged.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.levers import LeverPlan, write_plan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import MultiSampleDecoder
from kotodama.serve.xbatch import XBatchJob, XBatchRunner, XBatchScheduler

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
H = CFG["hidden_size"]
GREEDY = SamplingParams(temperature=0.0, repetition_penalty=1.0)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261029)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for layer in m.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return m


def _engine(model: KotodamaModel, levers: int) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        levers=levers)


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(4, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _plan(seed: int, span: str = "decode", scale: float = 2.0) -> LeverPlan:
    v = (scale * np.random.default_rng(seed).standard_normal(H)).astype(np.float32)
    return LeverPlan(attn=write_plan([4], v[None], CFG["attn_res_boundaries"], CFG["num_layers"], "K"), span=span)


@torch.inference_mode()
def _solo_greedy(eng: DecodeEngine, prompt: list[int], plan: LeverPlan | None, n_tok: int) -> list[int]:
    slot = eng.levers.bind(plan) if plan is not None else 0
    try:
        logits = eng.prefill(torch.tensor([prompt]), slot=slot)
        t = logits.argmax(-1).view(1, 1)
        out = [int(t)]
        for _ in range(n_tok - 1):
            t = eng.step_logits(t.clone())[0, -1].argmax().view(1, 1)
            out.append(int(t))
        return out
    finally:
        if slot:
            eng.levers.release(slot)


def _runner(eng: DecodeEngine, width: int = 6) -> XBatchRunner:
    dec = MultiSampleDecoder(eng, width)

    def prefill(ids, slot=0):
        return eng.prefill(ids, slot=slot), None

    return XBatchRunner({width: dec}, prefill=prefill, set_steer=eng.set_steer, decode=lambda ids: "",
                        stop_ids=frozenset(), max_seq_len=MAX_SEQ, device=torch.device("cpu"), refill=True,
                        levers=eng.levers)


def _job(out: dict, name: str, prompt: list[int], n: int, lever: LeverPlan | None, max_new: int = 6) -> XBatchJob:
    return XBatchJob(key="law", prompt_ids=prompt, n=n, max_new_tokens=max_new, params=GREEDY,
                     done=lambda o: out.__setitem__(name, o), lever=lever)


@torch.inference_mode()
def test_steered_and_unsteered_requests_share_a_batch(model: KotodamaModel) -> None:
    eng = _engine(model, levers=4)
    runner = _runner(eng)
    out: dict = {}
    jobs = {"plain": (_ids(10, 1), None), "a": (_ids(12, 2), _plan(3)), "b": (_ids(9, 4), _plan(5, "uniform")),
            "a2": (_ids(12, 2), _plan(6))}
    for name, (prompt, lever) in jobs.items():
        runner.queue.put(_job(out, name, prompt, 1 if name != "plain" else 2, lever))
    served = runner.run_batch()
    assert served == 4 and runner.stats["batches"] == 1  # one batch: the levers are not part of the key
    for name, (prompt, lever) in jobs.items():
        want = _solo_greedy(eng, prompt, lever, 6)
        assert all(row == want for row in out[name].rows), name
    assert out["a"].rows[0] != out["a2"].rows[0]  # same prompt, other lever
    assert eng.levers.bound == () and eng.levers.is_identity()  # no slot outlives its request


@torch.inference_mode()
def test_a_request_waits_while_every_slot_is_bound(model: KotodamaModel) -> None:
    eng = _engine(model, levers=1)
    runner = _runner(eng)
    out: dict = {}
    runner.queue.put(_job(out, "a", _ids(10, 7), 1, _plan(8)))
    runner.queue.put(_job(out, "b", _ids(10, 9), 1, _plan(10)))
    runner.queue.put(_job(out, "plain", _ids(10, 11), 1, None))
    while len(runner.queue):
        runner.run_batch()
    assert set(out) == {"a", "b", "plain"} and all(not isinstance(o, BaseException) for o in out.values())
    assert out["b"].rows[0] == _solo_greedy(eng, _ids(10, 9), _plan(10), 6)
    assert eng.levers.bound == ()


class _CharTok:
    def encode(self, text, return_tensors=None, **_):
        ids = [4 + ord(c) % 200 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)


def _serve(monkeypatch, model: KotodamaModel, levers: int):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    eng = _engine(model, levers=levers)
    runner = _runner(eng, width=4)
    runner.decode = lambda ids: _CharTok().decode(ids)
    used: list[str] = []

    async def run_inline(fn, *args, **kwargs):
        used.append(getattr(fn, "__name__", "?"))
        return fn(*args, **kwargs)

    unit = torch.from_numpy(np.random.default_rng(12).standard_normal(H).astype(np.float32))
    unit = unit / unit.norm()
    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_prefix_pool": None, "_stop_token_ids": frozenset({0}), "_run_generation": run_inline,
                 "_levers": levers, "_steer_loaded": True, "_steer_bank": {"v": (4, unit)},
                 "_steer_meds": {4: 3.0}, "_steer_aliases": {}, "_steer_partial_norms": {}}.items():
        monkeypatch.setattr(server, k, v)
    return TestClient(server.app), eng, unit, used


def test_server_maps_bank_vectors_injections_and_spans(monkeypatch, model: KotodamaModel) -> None:
    client, eng, unit, used = _serve(monkeypatch, model, levers=4)
    base = {"prompt": "two friends talk", "max_tokens": 5, "temperature": 0, "top_p": 0, "return_token_ids": True,
            "frequency_penalty": 0}
    prompt = _CharTok().encode(base["prompt"])

    def completion(**kw) -> list[int]:
        r = client.post("/v1/completions", json={**base, **kw})
        assert r.status_code == 200, r.text
        return r.json()["choices"][0]["token_ids"]["completion"]

    # a bank vector: alpha * the site's median norm * unit, block-persistent (rule K), generated positions only
    got = completion(vectors=[{"name": "v", "alpha": 0.5}])
    rows = (0.5 * 3.0 * unit.numpy())[None].astype(np.float32)
    plan = LeverPlan(attn=write_plan([4], rows, CFG["attn_res_boundaries"], CFG["num_layers"], "K"), span="decode")
    assert got == _solo_greedy(eng, prompt, plan, 5)
    assert got != completion()  # the write changed the continuation
    # the same vector as a raw injection, uniform span: the prompt is written too
    inj = {"sites": [4], "vectors": [(1.5 * unit).tolist()], "alpha": 1.0, "rule": "K"}
    got_u = completion(injection=inj, steer_span="uniform")
    assert got_u == _solo_greedy(eng, prompt, LeverPlan(attn=plan.attn, span="uniform"), 5)
    # an MLP-site injection (a neuron-style write) at the loop's second pass
    got_m = completion(injection={"sites": [7], "vectors": [(4.0 * unit).tolist()], "rule": "S", "kind": "mlp"})
    assert got_m == _solo_greedy(eng, prompt, LeverPlan(mlp={7: (4.0 * unit).numpy()}, span="decode"), 5)
    assert eng.levers.bound == ()
    # refusals, before anything is engaged
    for bad in ({"injection": {"sites": [4], "vectors": [[1.0] * 3]}},
                {"injection": {"sites": [7], "vectors": [unit.tolist()], "kind": "mlp", "rule": "K"}},
                {"injection": inj, "vectors": [{"name": "v"}]}):
        r = client.post("/v1/completions", json={**base, **bad})
        assert r.status_code == 400, (bad, r.text)
    assert eng.levers.bound == ()


def test_uniform_span_needs_levers(monkeypatch, model: KotodamaModel) -> None:
    client, eng, _, used = _serve(monkeypatch, model, levers=0)
    from kotodama.serve import server

    monkeypatch.setattr(server, "_engine", _engine(model, levers=0))
    r = client.post("/v1/completions", json={"prompt": "hey", "max_tokens": 3, "temperature": 0, "top_p": 0,
                                             "vectors": [{"name": "v"}], "steer_span": "uniform"})
    assert r.status_code == 400 and "--levers" in r.text
