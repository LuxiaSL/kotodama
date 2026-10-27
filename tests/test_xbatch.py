"""Cross-request batched decoding on a looped model (tiny moonshot geometry, CPU fp32).

The decoder: rows filled from DIFFERENT prompts of different lengths must each reproduce the full no-cache forward
of their own prompt plus their own continuation, including a row reused after a longer prompt held it. The runner:
key grouping, bucket choice, per-row stops and budgets, per-request n, greedy equivalence with the single-stream
engine, refill (continuous batching), steering installed for the batch and cleared after. The server: compatible
requests go through the batcher, anything else falls back to the existing paths.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import MultiSampleDecoder
from kotodama.serve.xbatch import XBatchJob, XBatchQueue, XBatchResult, XBatchRunner, XBatchScheduler

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
GREEDY = SamplingParams(temperature=1e-4, repetition_penalty=1.0)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261027)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for layer in m.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return m


def _engine(model: KotodamaModel, steer: bool = False) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=MAX_SEQ, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer)


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(4, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


@torch.inference_mode()
def test_rows_from_different_prompts_match_their_own_full_forward(model: KotodamaModel) -> None:
    eng = _engine(model)
    wide = MultiSampleDecoder(eng, 6)
    dec = MultiSampleDecoder(eng, 4, kv_from=wide)  # a narrower bucket over the wide one's KV memory
    assert dec.k_caches[0].data_ptr() == wide.k_caches[0].data_ptr()
    params = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    # a long prompt first fills every row, so the shorter prompts below land on rows holding stale, longer state
    dec.admit(range(4), eng.prefill_reference(torch.tensor([_ids(40, 9)])), params)
    for _ in range(3):
        dec.step(params)
    prompts = {0: _ids(11, 1), 1: _ids(23, 2), 2: _ids(11, 1), 3: _ids(5, 3)}  # rows 0 and 2: one prompt, n=2
    first: dict[int, torch.Tensor] = {}
    for rows, key in (([3], 3), ([0, 2], 0), ([1], 1)):  # admitted out of row order, row lists non-contiguous
        logits = eng.prefill_reference(torch.tensor([prompts[key]]))
        dec.admit(rows, logits, params)
        for r in rows:
            first[r] = logits[0]
    assert [dec.row_pos(r) for r in range(4)] == [11, 23, 11, 5]
    forced = torch.stack([torch.tensor(_ids(7, 30 + r)) for r in range(4)])  # (4, 7): its own continuation per row
    got = [torch.stack([first[r] for r in range(4)])]
    for t in range(forced.shape[1] - 1):
        got.append(dec.step_logits(forced[:, t:t + 1]).float())
    got_t = torch.stack(got, dim=1)  # (4, 7, V)
    for r in range(4):
        p = torch.tensor([prompts[r]])
        full = model(torch.cat([p, forced[r:r + 1]], dim=1))["logits"][0].float()
        want = full[p.shape[1] - 1:-1]
        assert torch.allclose(got_t[r], want, atol=2e-4, rtol=1e-4), (r, (got_t[r] - want).abs().max())
    assert [dec.row_pos(r) for r in range(4)] == [17, 29, 17, 11]


@torch.inference_mode()
def test_idle_rows_park_at_the_last_slot(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, 2)
    params = SamplingParams(temperature=0.9, repetition_penalty=1.0)
    dec.admit([0, 1], eng.prefill_reference(torch.tensor([_ids(MAX_SEQ - 3, 4)])), params)
    for _ in range(8):  # past capacity: the device position must stop at the last slot instead of indexing past it
        dec.step(params)
    assert dec.pos.tolist() == [MAX_SEQ - 1] * 2 and dec.row_pos(0) == MAX_SEQ - 3 + 8
    with pytest.raises(ValueError):
        dec.admit([0, 0], eng.prefill_reference(torch.tensor([_ids(8, 1)])), params)
    with pytest.raises(ValueError):
        dec.admit([2], eng.prefill_reference(torch.tensor([_ids(8, 1)])), params)


class _Rec:
    """Collects job outcomes by name."""

    def __init__(self) -> None:
        self.out: dict[str, XBatchResult | BaseException] = {}

    def job(self, name: str, prompt: list[int], n: int, key="k", max_new: int = 6, stops: tuple[str, ...] = (),
            params: SamplingParams = GREEDY, steer=None) -> XBatchJob:
        return XBatchJob(key=key, prompt_ids=prompt, n=n, max_new_tokens=max_new, params=params,
                         done=lambda o: self.out.__setitem__(name, o), stop_strings=stops, steer_writes=steer,
                         steer_kwargs={"rule": "K"} if steer else {})


def _runner(eng: DecodeEngine, widths=(2, 4), refill: bool = False, stop_ids=frozenset(), decode=None,
            prefix_cache: bool = False) -> XBatchRunner:
    wide = MultiSampleDecoder(eng, max(widths))
    decs = {w: (wide if w == max(widths) else MultiSampleDecoder(eng, w, kv_from=wide)) for w in widths}
    if prefix_cache:
        def prefill(ids):
            return eng.prefill_cached(ids)
    else:
        def prefill(ids):
            return eng.prefill_reference(ids), None
    return XBatchRunner(decs, prefill=prefill, set_steer=eng.set_steer,
                        decode=decode or (lambda ids: " ".join(map(str, ids))), stop_ids=stop_ids,
                        max_seq_len=MAX_SEQ, device=torch.device("cpu"), refill=refill)


def test_queue_groups_by_key_fifo_and_fit() -> None:
    q, rec = XBatchQueue(), _Rec()
    for name, key, n in (("a1", "A", 2), ("b1", "B", 1), ("a2", "A", 3), ("a3", "A", 2)):
        q.put(rec.job(name, [5], n, key=key))
    got = q.take_group(max_rows=4)  # a1 (2) + a3 (2): a2 (3) does not fit after a1, b1 is another key
    assert [j.n for j in got] == [2, 2] and len(q) == 2
    assert q.take_fitting("A", free_rows=8) is None  # B is now the head: a running A batch stops growing
    assert [j.key for j in q.take_group(max_rows=4)] == ["B"]
    assert q.take_fitting("A", free_rows=2) is None and q.take_fitting("A", free_rows=3).n == 3
    assert len(q.drain()) == 0


@torch.inference_mode()
def test_batch_matches_single_stream_greedy_per_request(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(2, 4, 8)), _Rec()
    prompts = {"p": _ids(9, 11), "q": _ids(26, 12), "r": _ids(17, 13)}
    for name, n in (("p", 3), ("q", 1), ("r", 2)):
        runner.queue.put(rec.job(name, prompts[name], n, max_new=8))
    assert runner.run_batch() == 3 and len(runner.queue) == 0
    ref = _engine(model)
    for name, n in (("p", 3), ("q", 1), ("r", 2)):
        res = rec.out[name]
        assert isinstance(res, XBatchResult) and len(res.rows) == n
        want = ref.generate_greedy(torch.tensor([prompts[name]]), 8)
        assert all(row == want for row in res.rows), (name, res.rows, want)
        assert res.timing["batched_rows"] == 8.0  # 6 rows -> the narrowest bucket that holds them
    assert runner.stats["batches"] == 1 and runner.stats["useful_row_steps"] == 6 * 8


@torch.inference_mode()
def test_per_row_stops_budgets_and_capacity(model: KotodamaModel) -> None:
    eng = _engine(model)
    stop = frozenset(range(4, 50))  # a broad stop set: rows end at different steps
    runner, rec = _runner(eng, widths=(4,), stop_ids=stop), _Rec()
    torch.manual_seed(2)
    hot = SamplingParams(temperature=1.3, repetition_penalty=1.1)
    runner.queue.put(rec.job("a", _ids(10, 1), 3, max_new=12, params=hot))
    runner.queue.put(rec.job("edge", _ids(MAX_SEQ - 4, 2), 1, max_new=50, params=hot))  # 4 slots of room
    runner.run_batch()
    a, edge = rec.out["a"], rec.out["edge"]
    assert len(a.rows) == 3 and all(len(r) <= 12 and not set(r) & stop for r in a.rows)
    assert len(edge.rows[0]) <= 4
    # stop strings end a row on the decoded text, and the matching token is kept (as on the single-stream path)
    runner2, rec2 = _runner(_engine(model), widths=(4,), decode=lambda ids: "x" * len(ids)), _Rec()
    runner2.queue.put(rec2.job("s", _ids(8, 3), 4, max_new=20, stops=("xxx",), params=hot))
    runner2.run_batch()
    assert [len(r) for r in rec2.out["s"].rows] == [3, 3, 3, 3]
    # the tail-window check ends each row exactly where a full decode first contains a stop
    letters = lambda ids: "".join(chr(97 + i % 3) for i in ids)  # noqa: E731 - three letters: stops recur
    runner3, rec3 = _runner(_engine(model), widths=(4,), decode=letters), _Rec()
    runner3.queue.put(rec3.job("t", _ids(8, 4), 4, max_new=60, stops=("abca", "ccb"), params=hot))
    runner3.run_batch()
    for row in rec3.out["t"].rows:
        hit = [k for k in range(1, len(row) + 1) if any(s in letters(row[:k]) for s in ("abca", "ccb"))]
        assert (hit[0] if hit else 60) == len(row), row


@torch.inference_mode()
def test_keys_never_share_a_batch_and_steering_is_scoped(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    calls: list = []

    def set_steer(writes, **kw):
        calls.append(None if writes is None else (len(writes), kw.get("rule")))
        eng.set_steer(writes, **kw)

    runner, rec = _runner(eng, widths=(4,)), _Rec()
    runner.set_steer = set_steer
    vec = torch.randn(CFG["hidden_size"], generator=torch.Generator().manual_seed(3))
    runner.queue.put(rec.job("steered", _ids(9, 1), 2, key="S", steer=[(4, vec, 0.7)]))
    runner.queue.put(rec.job("plain", _ids(9, 1), 2, key="P"))
    assert runner.run_batch() == 1 and calls == [(1, "K"), None] and eng.steer_buf.abs().sum() == 0
    assert runner.run_batch() == 1 and calls[-1] is None and runner.run_batch() == 0
    ref = _engine(model, steer=True)
    want = ref.generate_greedy(torch.tensor([_ids(9, 1)]), 6)
    assert rec.out["plain"].rows == [want, want]
    ref.set_steer([(4, vec, 0.7)])  # steered decode differs from plain decode, as on the single-stream engine
    want_s = ref.generate_greedy(torch.tensor([_ids(9, 1)]), 6)
    assert rec.out["steered"].rows == [want_s, want_s] and want_s != want


@torch.inference_mode()
def test_refill_serves_the_queue_in_one_batch_and_keeps_parity(model: KotodamaModel) -> None:
    prompts = [_ids(6 + 5 * i, 40 + i) for i in range(5)]
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(4,), refill=True, prefix_cache=True), _Rec()
    for i, p in enumerate(prompts):
        runner.queue.put(rec.job(f"j{i}", p, 2, max_new=3 + i))
    assert runner.run_batch() == 5 and runner.stats["refilled"] == 3
    ref = _engine(model)
    for i, p in enumerate(prompts):
        want = ref.generate_greedy(torch.tensor([p]), 3 + i)
        assert rec.out[f"j{i}"].rows == [want, want], i
    # without refill the same queue needs a batch per bucketful
    runner2, rec2 = _runner(_engine(model), widths=(4,)), _Rec()
    for i, p in enumerate(prompts):
        runner2.queue.put(rec2.job(f"j{i}", p, 2, max_new=3 + i))
    assert [runner2.run_batch() for _ in range(4)] == [2, 2, 1, 0]
    # another engine user waiting: the batch stops growing and drains
    runner3, rec3 = _runner(_engine(model), widths=(4,), refill=True), _Rec()
    runner3.should_yield = lambda: True
    for i, p in enumerate(prompts):
        runner3.queue.put(rec3.job(f"j{i}", p, 2, max_new=3 + i))
    assert runner3.run_batch() == 2 and runner3.stats["refilled"] == 0


def test_engine_lock_contention_is_visible(monkeypatch) -> None:
    from kotodama.serve import server

    async def main() -> list[bool]:
        lock = asyncio.Lock()
        monkeypatch.setattr(server, "_engine_lock", lock)
        seen = [server._engine_lock_contended()]
        async with lock:
            seen.append(server._engine_lock_contended())
            waiter = asyncio.create_task(lock.acquire())
            await asyncio.sleep(0)
            seen.append(server._engine_lock_contended())
        await waiter
        lock.release()
        seen.append(server._engine_lock_contended())
        return seen

    assert asyncio.run(main()) == [False, False, True, False]


@torch.inference_mode()
def test_a_failing_prefill_fails_only_its_request(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner, rec = _runner(eng, widths=(4,)), _Rec()
    runner.queue.put(rec.job("bad", _ids(MAX_SEQ + 5, 1), 2))  # longer than the cache: the engine refuses it
    runner.queue.put(rec.job("ok", _ids(9, 2), 2))
    assert runner.run_batch() == 2
    assert isinstance(rec.out["bad"], ValueError) and isinstance(rec.out["ok"], XBatchResult)


def test_scheduler_runs_concurrent_submissions_and_refuses_wide_jobs(model: KotodamaModel) -> None:
    eng = _engine(model)
    runner = _runner(eng, widths=(4,))
    batches: list[int] = []

    async def run_on_worker(fn):
        batches.append(len(runner.queue))
        return fn()

    async def main() -> list[XBatchResult]:
        sched = XBatchScheduler(runner, asyncio.Lock(), run_on_worker)
        rec = _Rec()
        with pytest.raises(ValueError):
            await sched.submit(rec.job("wide", _ids(5, 1), 5))
        jobs = [rec.job(f"c{i}", _ids(7 + i, i), 2, max_new=4) for i in range(3)]
        return await asyncio.gather(*(sched.submit(j) for j in jobs))

    with torch.inference_mode():
        results = asyncio.run(main())
    assert [len(r.rows) for r in results] == [2, 2, 2]
    assert batches == [3, 1]  # the first batch took two requests (4 rows); the third waited for the next


class _CharTok:
    """Characters <-> ids inside the tiny vocab (4 + ord % 200)."""

    def encode(self, text, return_tensors=None, **_):
        ids = [4 + ord(c) % 200 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)


def test_server_routes_compatible_requests_through_the_batcher(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    eng = _engine(model, steer=True)
    runner = _runner(eng, widths=(4, 8))
    runner.decode = lambda ids: _CharTok().decode(ids)
    runner.stop_ids = frozenset({0})
    used: list[str] = []

    async def run_inline(fn, *args, **kwargs):
        used.append(getattr(fn, "__name__", "?"))
        return fn(*args, **kwargs)

    sched = XBatchScheduler(runner, asyncio.Lock(), run_inline)
    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {}, "_xbatch": sched,
                 "_device": torch.device("cpu"), "_max_seq_len": MAX_SEQ, "_prefix_cache": False,
                 "_stop_token_ids": frozenset({0}), "_run_generation": run_inline}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 6, "n": 3,
                                             "temperature": 0.9, "top_p": 0, "return_token_ids": True,
                                             "stop": ["q"]})
    assert r.status_code == 200, r.text
    choices = r.json()["choices"]
    assert used == ["run_batch"] and len(choices) == 3
    assert all(1 <= len(c["token_ids"]["completion"]) <= 6 for c in choices)
    assert all("q" not in c["text"][:-1] for c in choices)  # a stop string ends the row where it appears
    used.clear()
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 4, "n": 2,
                                             "temperature": 0.9, "top_p": 0.9})  # truncated law: serial fallback
    assert r.status_code == 200 and used == ["engine_generate", "engine_generate"]
    used.clear()
    r = client.post("/v1/completions", json={"prompt": "x" * MAX_SEQ, "max_tokens": 4, "n": 2, "temperature": 0.9})
    assert r.status_code == 400 and used == []  # refused before it queues
    r = client.post("/v1/completions", json={"prompt": "hey", "max_tokens": 4, "n": 2, "temperature": 0.9,
                                             "vectors": [{"name": "nope", "alpha": 1.0}]})
    assert r.status_code == 400 and used == []  # steering without a bank: refused before it queues
    assert runner.stats["batches"] == 1 and eng.steer_buf.abs().sum() == 0
