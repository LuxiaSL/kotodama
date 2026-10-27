"""Cascade (shared-prefix) decode attention, CPU fp32.

The merge: attention over two disjoint key sets, merged by log-sum-exp, equals attention over their union; grouped
queries packed group-major give each row its own attention. The decoder: rows that share a prompt through one group
reproduce the full no-cache forward of prompt + their own continuation and the per-row decoder's logits; groups are
taken and freed as rows are admitted and released; tails bound a row's tokens; steering matches the single-stream
engine. The batcher: greedy parity per request with cascade rows, admission waiting for free groups, and refill.
The server: the cascade flag builds cascade decoders and keeps requests longer than a tail off the batched paths.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from kotodama.model.cascade_attn import (attend_for, attend_reference, merge, pack_group_queries,
                                         unpack_group_outputs)
from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder, decode_rows
from kotodama.serve.xbatch import XBatchJob, XBatchResult, XBatchRunner

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
GREEDY = SamplingParams(temperature=1e-4, repetition_penalty=1.0)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261028)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(11)
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


# ── the merge ───────────────────────────────────────────────────────────────


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain attention, q (N, Lq, Hq, D) over every key of k/v (N, S, Hk, D) -> (N, Lq, Hq, D)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True)
    return out.transpose(1, 2)


@pytest.mark.parametrize("hq,hk", [(4, 2), (6, 6), (12, 6)])
def test_lse_merge_of_two_key_sets_is_attention_over_their_union(hq: int, hk: int) -> None:
    g = torch.Generator().manual_seed(hq * 10 + hk)
    n, lq, d, s1, s2 = 3, 2, 16, 13, 7
    q = torch.randn(n, lq, hq, d, generator=g, dtype=torch.float64)
    k1, v1 = (torch.randn(n, s1, hk, d, generator=g, dtype=torch.float64) for _ in range(2))
    k2, v2 = (torch.randn(n, s2, hk, d, generator=g, dtype=torch.float64) for _ in range(2))
    lens1 = torch.tensor([13, 5, 1], dtype=torch.int32)  # slots past lens are garbage, never attended
    lens2 = torch.tensor([7, 2, 4], dtype=torch.int32)
    o1, l1 = attend_reference(q, k1, v1, lens1)
    o2, l2 = attend_reference(q, k2, v2, lens2)
    assert l1.shape == (n, hq, lq) and l1.dtype == torch.float32
    got = merge(o1, l1.transpose(1, 2), o2, l2.transpose(1, 2))  # (N, Lq, Hq, D)
    for i in range(n):
        a, b = int(lens1[i]), int(lens2[i])
        want = _sdpa(q[i:i + 1], torch.cat([k1[i:i + 1, :a], k2[i:i + 1, :b]], 1),
                     torch.cat([v1[i:i + 1, :a], v2[i:i + 1, :b]], 1))
        torch.testing.assert_close(got[i:i + 1], want, atol=1e-5, rtol=1e-5)


def test_an_empty_first_part_gets_no_weight() -> None:
    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, 1, 4, 8, generator=g)
    k, v = torch.randn(2, 5, 2, 8, generator=g), torch.randn(2, 5, 2, 8, generator=g)
    o1, l1 = attend_reference(q, k, v, torch.tensor([0, 0], dtype=torch.int32))
    assert torch.isinf(l1).all() and (o1 == 0).all()
    o2, l2 = attend_reference(q, k, v, torch.tensor([3, 5], dtype=torch.int32))
    torch.testing.assert_close(merge(o1, l1.transpose(1, 2), o2, l2.transpose(1, 2)), o2)


def test_group_packed_queries_give_each_row_its_own_attention() -> None:
    g = torch.Generator().manual_seed(2)
    rows, hq, hk, d, s, m = 7, 6, 2, 8, 11, 3
    q_rows = torch.randn(rows, hq, d, generator=g)
    k, v = torch.randn(2, s, hk, d, generator=g), torch.randn(2, s, hk, d, generator=g)
    lens = torch.tensor([11, 6], dtype=torch.int32)
    members = torch.tensor([[4, 0, 2], [1, 6, 1]])  # group 1 has two rows; its third entry repeats a member
    out, lse = attend_reference(pack_group_queries(q_rows, members, hk), k, v, lens)
    o, lo = unpack_group_outputs(out, lse, m)
    assert o.shape == (2 * m, hq, d) and lo.shape == (2 * m, hq)
    for slot, row in [(0, 4), (1, 0), (2, 2), (3, 1), (4, 6)]:
        grp = slot // m
        want_o, want_l = attend_reference(q_rows[row].view(1, 1, hq, d), k[grp:grp + 1], v[grp:grp + 1],
                                          lens[grp:grp + 1])
        torch.testing.assert_close(o[slot], want_o[0, 0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lo[slot], want_l[0, :, 0], atol=1e-5, rtol=1e-5)


def test_cpu_and_fp32_use_the_reference_kernel() -> None:
    assert attend_for(torch.device("cpu"), torch.float32)[0] == "reference"
    assert attend_for(torch.device("cpu"), torch.bfloat16)[0] == "reference"


# ── the decoder ─────────────────────────────────────────────────────────────


@torch.inference_mode()
def test_grouped_rows_match_their_own_full_forward_and_the_per_row_decoder(model: KotodamaModel) -> None:
    eng = _engine(model)
    wide = CascadeDecoder(eng, 6, groups=4, group_rows=2, tail_tokens=24)
    dec = CascadeDecoder(eng, 5, groups=4, group_rows=2, tail_tokens=24, kv_from=wide)  # a narrower bucket
    assert dec.prompt_ks[0].data_ptr() == wide.prompt_ks[0].data_ptr()
    assert dec.tail_ks[0].data_ptr() == wide.tail_ks[0].data_ptr()
    ref = MultiSampleDecoder(eng, 5)
    params = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    # a long prompt fills every row first, so the prompts below land on groups and tails holding stale state
    logits = eng.prefill_reference(torch.tensor([_ids(40, 9)]))
    dec.admit(range(4), logits, params)
    for _ in range(3):
        dec.step(params)
    dec.release(range(5))  # as the batcher does when a batch starts
    # rows 0, 2 and 3 share one prompt (n=3: two groups of up to 2 rows), row 1 its own, row 4 a third
    prompts = {0: _ids(11, 1), 1: _ids(23, 2), 4: _ids(5, 3)}
    first: dict[int, torch.Tensor] = {}
    for rows, key in (([4], 4), ([0, 2, 3], 0), ([1], 1)):
        logits = eng.prefill_reference(torch.tensor([prompts[key]]))
        dec.admit(rows, logits, params)
        ref.admit(rows, logits, params)
        for r in rows:
            first[r] = logits[0]
    assert [dec.row_pos(r) for r in range(5)] == [11, 23, 11, 11, 5]
    assert len({dec._row_group[r] for r in (0, 2, 3)}) == 2 and dec.admit_capacity() == 0
    plen = {0: 11, 1: 23, 2: 11, 3: 11, 4: 5}
    forced = torch.stack([torch.tensor(_ids(8, 30 + r)) for r in range(5)])  # its own continuation per row
    got = [torch.stack([first[r] for r in range(5)])]
    want_rows = [got[0]]
    for t in range(forced.shape[1] - 1):
        got.append(dec.step_logits(forced[:, t:t + 1]).float())
        want_rows.append(ref.step_logits(forced[:, t:t + 1]).float())
    got_t, ref_t = torch.stack(got, dim=1), torch.stack(want_rows, dim=1)  # (5, 8, V)
    torch.testing.assert_close(got_t, ref_t, atol=5e-5, rtol=1e-5)
    for r in range(5):
        p = prompts[0 if r in (0, 2, 3) else r]
        full = model(torch.cat([torch.tensor([p]), forced[r:r + 1]], dim=1))["logits"][0].float()
        want = full[plen[r] - 1:-1]
        assert torch.allclose(got_t[r], want, atol=2e-4, rtol=1e-4), (r, (got_t[r] - want).abs().max())


@torch.inference_mode()
def test_groups_are_taken_on_admit_and_freed_on_release(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = CascadeDecoder(eng, 6, groups=2, group_rows=3, tail_tokens=8)
    params = SamplingParams(temperature=0.9, repetition_penalty=1.0)
    assert dec.admit_capacity() == 6 and dec.token_room(10) == 8 and dec.token_room(90) == MAX_SEQ - 90
    la = eng.prefill_reference(torch.tensor([_ids(9, 4)]))
    dec.admit([0, 1, 2, 3], la, params)  # four rows of one prompt: both groups (3 + 1 rows)
    assert dec.admit_capacity() == 0 and dec.prompt_lens.tolist() == [9, 9]
    lb = eng.prefill_reference(torch.tensor([_ids(6, 5)]))
    with pytest.raises(RuntimeError):
        dec.admit([4], lb, params)
    dec.release([3])  # the second group's only row
    assert dec.admit_capacity() == 3 and dec.prompt_lens.tolist() == [9, 0]
    dec.admit([4, 5], lb, params)
    assert dec.prompt_lens.tolist() == [9, 6]
    dec.restart([1], 9, la, params)  # a row of the first group restarts from its own prompt
    with pytest.raises(ValueError):
        dec.restart([4], 9, la, params)  # row 4 holds the 6-token prompt
    dec.release([0, 1, 2])
    with pytest.raises(ValueError):
        dec.restart([1], 9, la, params)  # released: nothing to restart from
    assert dec.admit_capacity() == 3
    # a row's tail bounds its tokens: decode_rows stops when the tail is full
    la = eng.prefill_reference(torch.tensor([_ids(9, 4)]))
    out = decode_rows(dec, la, params, n=2, max_new_tokens=50, stop_ids=frozenset())
    assert [len(r) for r in out] == [9, 9]  # 8 fed tokens fill the tail; the last sampled one is kept
    with pytest.raises(ValueError):
        CascadeDecoder(eng, 4, tail_tokens=16, kv_from=dec)  # a different tail length cannot view this storage
    with pytest.raises(ValueError):
        CascadeDecoder(eng, 4, groups=5)


@torch.inference_mode()
def test_steered_cascade_rows_match_the_single_stream_engine(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    dec = CascadeDecoder(eng, 4, groups=1, tail_tokens=16)
    vec = torch.randn(CFG["hidden_size"], generator=torch.Generator().manual_seed(3))
    eng.set_steer([(4, vec, 0.7)])
    prompt, forced = torch.tensor([_ids(10, 2)]), torch.tensor([_ids(7, 20)])
    logits0 = eng.prefill_reference(prompt)
    dec.start(logits0, SamplingParams(temperature=0.8, repetition_penalty=1.0))
    batched = [dec.step_logits(forced[:, t:t + 1].expand(4, 1).contiguous()).float() for t in range(6)]
    eng.prefill_reference(prompt)
    single = [eng.step_logits(forced[:, t:t + 1])[0, -1].float() for t in range(6)]
    for b, s in zip(batched, single):
        assert torch.allclose(b, s.expand_as(b), atol=2e-4, rtol=1e-4)
    eng.set_steer(None)


# ── the batcher ─────────────────────────────────────────────────────────────


def _runner(eng: DecodeEngine, decoders: dict, refill: bool = True) -> XBatchRunner:
    return XBatchRunner(decoders, prefill=lambda ids: (eng.prefill_reference(ids), None), set_steer=eng.set_steer,
                        decode=lambda ids: " ".join(map(str, ids)), stop_ids=frozenset(), max_seq_len=MAX_SEQ,
                        device=torch.device("cpu"), refill=refill)


def _greedy(eng: DecodeEngine, prompt: list[int], n: int) -> list[int]:
    logits = eng.prefill_reference(torch.tensor([prompt]))[0]
    out = [int(logits.argmax())]
    for _ in range(n - 1):
        out.append(int(eng.step_logits(torch.tensor([[out[-1]]]))[0, -1].argmax()))
    return out


@torch.inference_mode()
def test_batcher_on_cascade_rows_matches_single_stream_greedy_and_waits_for_groups(model: KotodamaModel) -> None:
    eng = _engine(model)
    # 8 rows but only 2 groups: of the five 3-candidate requests at most two decode at once, the rest wait for
    # a group; candidates past a request's admitted rows restart in its own rows
    dec = CascadeDecoder(eng, 8, groups=2, group_rows=4, tail_tokens=12)
    runner = _runner(eng, {8: dec})
    prompts = [_ids(6 + 5 * i, 50 + i) for i in range(5)]
    got: dict[int, XBatchResult] = {}
    for i, p in enumerate(prompts):
        runner.queue.put(XBatchJob(key="g", prompt_ids=p, n=3, max_new_tokens=12 if i % 2 else 6, params=GREEDY,
                                   done=lambda o, i=i: got.__setitem__(i, o)))
    assert runner.run_batch() == 5 and not len(runner.queue)
    assert runner.stats["refilled"] >= 3  # the requests past the first two were admitted as groups freed
    for i, p in enumerate(prompts):
        want = _greedy(eng, p, 12 if i % 2 else 6)
        assert isinstance(got[i], XBatchResult) and got[i].rows == [want] * 3, i
    # a request whose budget is longer than a tail stops at the tail
    runner.queue.put(XBatchJob(key="g", prompt_ids=prompts[0], n=1, max_new_tokens=40, params=GREEDY,
                               done=lambda o: got.__setitem__(9, o)))
    runner.run_batch()
    assert got[9].rows == [_greedy(eng, prompts[0], 12)]


@torch.inference_mode()
def test_batcher_without_refill_requeues_what_finds_no_group(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = CascadeDecoder(eng, 8, groups=1, group_rows=8, tail_tokens=8)
    runner = _runner(eng, {8: dec}, refill=False)
    done: list[int] = []
    for i in range(3):
        runner.queue.put(XBatchJob(key="g", prompt_ids=_ids(7, 70 + i), n=2, max_new_tokens=4, params=GREEDY,
                                   done=lambda o, i=i: done.append(i)))
    assert runner.run_batch() == 1 and len(runner.queue) == 2  # one group: one request per batch
    assert runner.run_batch() == 1 and runner.run_batch() == 1 and done == [0, 1, 2]


# ── the server ──────────────────────────────────────────────────────────────


def test_server_keeps_requests_longer_than_a_tail_off_the_batched_paths(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from kotodama.serve import server

    req = server.GenerateRequest(prompt="x", max_new_tokens=64, temperature=0.9)
    eng = _engine(model)
    dec = CascadeDecoder(eng, 4, tail_tokens=32)
    monkeypatch.setattr(server, "_multi", {4: dec})
    monkeypatch.setattr(server, "_row_token_cap", 32)
    assert not server._multi_compatible(req, 3)
    assert server._multi_compatible(server.GenerateRequest(prompt="x", max_new_tokens=32, temperature=0.9), 3)
    monkeypatch.setattr(server, "_row_token_cap", 0)  # per-row decoders: no cap
    assert server._multi_compatible(req, 3)
