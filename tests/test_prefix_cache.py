"""Multi-entry prefix KV cache (kotodama.model.prefix_cache).

The index: block-granular matching, shared trunks share nodes, least-recently-used leaves go first, a node outlives
its children, the chain being inserted is never evicted, and a randomized run against a brute-force reference. The
GPU-side pool, on a tiny looped model on CPU (fp32): a pool hit (the engine holding an unrelated prompt) extends from
the pooled prefix and matches the full forward of the prompt and of its continuation; only prefill-written KV is
pooled (decode steps may be steered); a non-finite block is dropped instead of pooled; the cross-request batcher
with the pool in front of its prefill keeps greedy equivalence; the server's prefill goes through the pool when one
is configured.
"""

from __future__ import annotations

import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import MultiSampleDecoder
from kotodama.model.prefix_cache import BlockIndex, PrefixKVCache
from kotodama.serve.xbatch import XBatchJob, XBatchResult, XBatchRunner

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
MAX_SEQ = 96
BLOCK = 16
GREEDY = SamplingParams(temperature=1e-4, repetition_penalty=1.0)


# ── the index ────────────────────────────────────────────────────────────────

def test_match_is_block_granular_and_shared_trunks_share_nodes() -> None:
    ix = BlockIndex(n_blocks=8, block_size=4)
    a = list(range(10, 23))  # 13 tokens: 3 full blocks + 1
    assert ix.match(a) == []
    new = ix.insert(a, 3)
    assert [i for i, _ in new] == [0, 1, 2] and len(ix) == 3
    assert ix.match(a) == [s for _, s in new]
    b = a[:9] + [99, 98, 97, 96]  # diverges inside block 2
    assert ix.match(b) == [s for _, s in new[:2]]
    nb = ix.insert(b, 3)
    assert [i for i, _ in nb] == [2] and len(ix) == 4  # only the diverging block is new
    assert ix.match(a, max_blocks=1) == [new[0][1]]
    assert ix.match(a[:3]) == []  # a partial block never matches
    assert ix.insert(a[:7], 5) == []  # n_full is capped at the full blocks present (already cached)


def test_lru_leaf_eviction_and_chain_protection() -> None:
    ix = BlockIndex(n_blocks=4, block_size=2)
    a = [1, 1, 2, 2, 3, 3]
    b = [1, 1, 2, 2, 4, 4]
    ix.insert(a, 3)  # slots for [1,1] [2,2] [3,3]
    ix.insert(b, 3)  # + [4,4] under the shared trunk: pool full
    ix.match(a)  # a's leaf is now more recent than b's
    c = [7, 7, 8, 8]
    new = ix.insert(c, 2)  # needs 2 slots: b's leaf goes first, then a's leaf; the trunk stays
    assert len(new) == 2 and ix.evicted == 2
    assert len(ix.match(a)) == 2 and len(ix.match(b)) == 2 and len(ix.match(c)) == 2
    # extending a chain longer than the pool keeps the chain's own head and leaves the tail uncached
    ix2 = BlockIndex(n_blocks=3, block_size=1)
    got = ix2.insert([5, 6, 7, 8, 9], 5)
    assert [i for i, _ in got] == [0, 1, 2] and len(ix2.match([5, 6, 7, 8, 9])) == 3
    got = ix2.insert([5, 6, 7, 8, 9], 5)  # the whole pool is this chain: nothing can be evicted for it
    assert got == [] and ix2.evicted == 0


def test_discard_drops_fresh_blocks_deepest_first() -> None:
    ix = BlockIndex(n_blocks=6, block_size=2)
    ix.insert([1, 1, 2, 2], 2)
    new = ix.insert([1, 1, 2, 2, 3, 3, 4, 4], 4)
    ix.discard([s for _, s in new])
    assert len(ix) == 2 and len(ix.free) == 4 and len(ix.match([1, 1, 2, 2, 3, 3])) == 2
    with pytest.raises(RuntimeError):
        ix.discard([ix.match([1, 1])[0]])  # a node with cached children cannot go


def test_index_against_a_brute_force_reference() -> None:
    rng = random.Random(4)
    b = 3
    ix = BlockIndex(n_blocks=40, block_size=b)
    trunks = [[rng.randrange(5) for _ in range(rng.randrange(3, 30))] for _ in range(6)]
    for step in range(600):
        t = list(rng.choice(trunks))
        cut = rng.randrange(len(t) + 1)
        ids = t[:cut] + [rng.randrange(5) for _ in range(rng.randrange(0, 12))]
        if rng.random() < 0.5:
            got = ix.match(ids)
            # the reference: the cached prefixes are exactly the root paths of the index's nodes
            cached = set()
            for node in ix.nodes.values():
                path, n = [], node
                while True:
                    path = list(n.key) + path
                    if n.parent == -1:
                        break
                    n = ix.nodes[n.parent]
                cached.add(tuple(path))
            k = 0
            while (k + 1) * b <= len(ids) and tuple(ids[:(k + 1) * b]) in cached:
                k += 1
            assert len(got) == k, step
        else:
            ix.insert(ids, len(ids) // b)
        # invariants: every node's parent is cached and counts it; slots are partitioned into used and free
        for node in ix.nodes.values():
            assert node.parent == -1 or node.parent in ix.nodes
        for slot, node in ix.nodes.items():
            assert node.children == sum(1 for m in ix.nodes.values() if m.parent == slot)
        assert sorted(list(ix.nodes) + ix.free) == list(range(ix.n_blocks))
    assert ix.evicted > 0


# ── the pool in front of the engine ─────────────────────────────────────────

@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261028)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(9)
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
def test_a_pool_hit_matches_the_full_forward(model: KotodamaModel) -> None:
    eng = _engine(model)
    pool = PrefixKVCache(eng, n_tokens=8 * BLOCK, block_size=BLOCK)
    trunk = _ids(37, 1)
    a, b = trunk + _ids(6, 2), trunk + _ids(9, 3)
    _, info = pool.prefill(torch.tensor([a]))
    assert info["pool_tokens"] == 0 and info["stored_blocks"] == 2  # 43 tokens: 2 full blocks pooled
    pool.prefill(torch.tensor([_ids(30, 4)]))  # an unrelated prompt: the engine's own cache now holds nothing of b
    logits, info = pool.prefill(torch.tensor([b]))
    assert info["prefix_hit"] and info["pool_tokens"] == 2 * BLOCK and info["common_prefix"] == 2 * BLOCK
    full = model(torch.tensor([b]))["logits"][0, -1].float()
    assert torch.allclose(logits[0], full, atol=2e-4, rtol=1e-4), (logits[0] - full).abs().max()
    # the engine's cache now holds b: decoding on from the hit matches the full forward of b + continuation
    cont = _ids(5, 5)
    steps = [eng.step_logits(torch.tensor([[t]]))[0, -1].float() for t in cont]
    full = model(torch.tensor([b + cont]))["logits"][0].float()
    for i, s in enumerate(steps):
        assert torch.allclose(s, full[len(b) + i], atol=2e-4, rtol=1e-4), i
    st = pool.stats
    assert st["requests"] == 3 and st["hits"] == 1 and st["pool_tokens"] == 2 * BLOCK


@torch.inference_mode()
def test_engine_prefix_wins_when_longer_and_partial_load_tops_it_up(model: KotodamaModel) -> None:
    eng = _engine(model)
    pool = PrefixKVCache(eng, n_tokens=8 * BLOCK, block_size=BLOCK)
    trunk = _ids(50, 6)
    pool.prefill(torch.tensor([trunk + _ids(3, 7)]))  # pools 3 blocks
    short = trunk[:20] + _ids(14, 8)
    pool.prefill(torch.tensor([short]))  # the engine now holds 20 tokens of the trunk
    want = trunk + _ids(4, 9)
    logits, info = pool.prefill(torch.tensor([want]))
    # the engine keeps its first block, the pool supplies blocks 1-2
    assert info["pool_tokens"] == 2 * BLOCK and info["common_prefix"] == 3 * BLOCK
    full = model(torch.tensor([want]))["logits"][0, -1].float()
    assert torch.allclose(logits[0], full, atol=2e-4, rtol=1e-4)
    # a prompt the engine itself already holds further than the pool: plain prefill_cached, nothing loaded
    longer = want + _ids(3, 10)
    _, info = pool.prefill(torch.tensor([longer]))
    assert info["pool_tokens"] == 0 and info["common_prefix"] == len(want)


@torch.inference_mode()
def test_only_prefill_written_kv_is_pooled(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    pool = PrefixKVCache(eng, n_tokens=8 * BLOCK, block_size=BLOCK)
    eng.set_steer([(4, torch.randn(CFG["hidden_size"], generator=torch.Generator().manual_seed(2)), 2.0)])
    prompt = _ids(20, 11)
    logits, _ = pool.prefill(torch.tensor([prompt]))
    assert len(pool.index) == 1 and eng.prompt_kv_len == 20
    tok = eng.sample_first(logits, GREEDY)
    gen = []
    for _ in range(20):  # steered decode steps write positions 20..39
        gen.append(int(tok.item()))
        tok = eng.step(tok, GREEDY)
    eng.set_steer(None)
    nxt = prompt + gen + _ids(10, 12)
    _, info = pool.prefill(torch.tensor([nxt]))
    # the engine reuses only the prefill-written positions (the steered decode KV after them is re-prefilled
    # unsteered), so the whole new prompt is prompt KV and the pool takes its full blocks
    assert info["common_prefix"] == 20 and eng.prompt_kv_len == len(nxt)
    assert len(pool.index) == len(nxt) // BLOCK
    # a fresh prefill of the same prompt is all prompt KV again, and its blocks are pooled
    eng.reset()
    pool.prefill(torch.tensor([nxt]))
    assert eng.prompt_kv_len == len(nxt) and len(pool.index) == len(nxt) // BLOCK


@torch.inference_mode()
def test_non_finite_blocks_are_not_pooled(model: KotodamaModel) -> None:
    eng = _engine(model)
    pool = PrefixKVCache(eng, n_tokens=8 * BLOCK, block_size=BLOCK)
    prompt = _ids(40, 13)
    eng.prefill_reference(torch.tensor([prompt]))
    eng.k_caches[3][0, 1, 21] = float("nan")
    new = pool.index.insert(prompt, 2)
    assert pool._store(new) == 0 and len(pool.index) == 0 and pool.stats["dropped_blocks"] == 2
    assert pool.index.match(prompt) == []


def test_pool_refuses_blocks_below_the_engine_minimum(model: KotodamaModel) -> None:
    eng = _engine(model)
    with pytest.raises(ValueError):
        PrefixKVCache(eng, n_tokens=256, block_size=8)
    with pytest.raises(ValueError):
        PrefixKVCache(eng, n_tokens=10, block_size=16)


@torch.inference_mode()
def test_batcher_with_the_pool_keeps_greedy_equivalence(model: KotodamaModel) -> None:
    eng = _engine(model)
    pool = PrefixKVCache(eng, n_tokens=16 * BLOCK, block_size=BLOCK)
    wide = MultiSampleDecoder(eng, 8)
    seen: list[tuple[list[int], torch.Tensor]] = []

    def prefill(ids: torch.Tensor):
        logits, info = pool.prefill(ids)
        seen.append((ids[0].tolist(), logits[0].clone()))
        return logits, info

    runner = XBatchRunner({8: wide}, prefill=prefill, set_steer=eng.set_steer,
                          decode=lambda ids: " ".join(map(str, ids)), stop_ids=frozenset(), max_seq_len=MAX_SEQ,
                          device=torch.device("cpu"), refill=True)
    trunk_a, trunk_b = _ids(40, 20), _ids(35, 21)
    prompts = {}
    for i in range(3):  # two families interleaved: every admit after the first of a family hits the pool
        prompts[f"a{i}"] = trunk_a + _ids(5 + i, 30 + i)
        prompts[f"b{i}"] = trunk_b + _ids(4 + i, 40 + i)
    order = ["a0", "b0", "a1", "b1", "a2", "b2"]
    out: dict[str, XBatchResult | BaseException] = {}
    for name in order:
        runner.queue.put(XBatchJob(key="k", prompt_ids=prompts[name], n=2, max_new_tokens=6, params=GREEDY,
                                   done=lambda o, name=name: out.__setitem__(name, o)))
    assert runner.run_batch() == 6
    ref = _engine(model)
    for name in order:
        res = out[name]
        assert isinstance(res, XBatchResult)
        want = ref.generate_greedy(torch.tensor([prompts[name]]), 6)
        assert all(row == want for row in res.rows), (name, res.rows, want)
    assert pool.stats["hits"] == 4  # a1, b1, a2, b2
    # this tiny model's greedy continuations barely depend on the trunk, so the prefill logits are the sharp check
    assert len(seen) == 6
    for ids, logits in seen:
        full = model(torch.tensor([ids]))["logits"][0, -1].float()
        assert torch.allclose(logits, full, atol=2e-4, rtol=1e-4), (logits - full).abs().max()
    assert all(out[n].timing["prefix_pool_tokens"] == 32.0 for n in ("a1", "b1", "a2", "b2"))


@torch.inference_mode()
def test_server_prefill_goes_through_the_pool(monkeypatch, model: KotodamaModel) -> None:
    from kotodama.serve import server

    eng = _engine(model)
    pool = PrefixKVCache(eng, n_tokens=8 * BLOCK, block_size=BLOCK)
    monkeypatch.setattr(server, "_engine", eng)
    monkeypatch.setattr(server, "_prefix_cache", True)
    monkeypatch.setattr(server, "_prefix_pool", pool)
    ids = torch.tensor([_ids(33, 50)])
    _, info = server._engine_prefill(ids)
    assert "pool_tokens" in info and pool.stats["requests"] == 1
    monkeypatch.setattr(server, "_prefix_pool", None)
    _, info = server._engine_prefill(ids)
    assert "pool_tokens" not in info and pool.stats["requests"] == 1  # off: the engine's own prefix cache only


def test_one_chunk_extend_plan(model: KotodamaModel) -> None:
    eng = _engine(model)
    assert eng.block_buckets == (256, 64, 16)
    greedy = {(100, 140): eng._chunk_plan(100, 140), (0, 40): eng._chunk_plan(0, 40)}
    assert greedy[(100, 140)] == [(100, 16), (116, 16), (124, 16)]
    eng.extend_single_chunk = True
    assert eng._chunk_plan(100, 140) == [(76, 64)]  # one right-aligned chunk re-forwarding 24 cached tokens
    assert eng._chunk_plan(100, 110) == [(94, 16)]
    assert eng._chunk_plan(2000, 2200) == [(1944, 256)]
    assert eng._chunk_plan(30, 40) == [(24, 16)]
    assert eng._chunk_plan(10, 40) == [(10, 16), (24, 16)]  # no bucket of 30..40 tokens: the greedy cover
    assert eng._chunk_plan(0, 40) == greedy[(0, 40)]  # a full prefill keeps the greedy cover
    assert eng._chunk_plan(100, 400) == [(100, 256), (356, 16), (372, 16), (384, 16)]  # wider than any bucket
