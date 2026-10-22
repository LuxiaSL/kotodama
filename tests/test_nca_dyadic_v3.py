"""NCA-v3 generator tests.

CPU-only, tiny geometries. Covers: the v1 path staying byte-identical
(offset=0 / default rng), the v3 alphabet shift + EOS reservation, the
last-write-wins ledger (well-posedness: a query's gold is the last write to its
tag), anti-shortcut structure, marked/unmarked serialization,
token-level mix accounting + sidecar integrity in the writer, and full-run
determinism with no global-RNG consumption.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from kotodama.nca.generator import (NCAConfig, NCARule, sample_rule_config,
                                    simulate_trajectory, tokenize_trajectory)
from kotodama.nca.dyadic import (DyadicConfig, N_VALS, queries_in_stream_order,
                                 serialize_dyadic, simulate_dyadic, token_map)
from kotodama.nca import gen_dyadic_data as gen


def _mk_rule(cfg: NCAConfig, seed: int) -> NCARule:
    torch.manual_seed(seed)
    rc = sample_rule_config(cfg, rng=random.Random(seed))
    return NCARule(d_state=cfg.d_state, n_groups=cfg.n_groups,
                   kernel_size=rc["kernel_size"], hidden_dim=rc["hidden_dim"],
                   num_hidden_layers=rc["num_hidden_layers"])


def _tiny_dyadic(reserve_eos=True, p_tag_reuse=1.0, mark_turns=True):
    return DyadicConfig(
        nca=NCAConfig(grid_size=8, n_groups=2, num_steps=32, burn_in=2,
                      filter_enabled=False, reserve_eos=reserve_eos),
        commits_per_traj=(5, 8), queries_per_traj=(4, 8),
        p_tag_reuse=p_tag_reuse, mark_turns=mark_turns,
        min_query_lag_steps=3, turn_len_min=2, turn_len_max=4)


# ── v1 path regression: defaults unchanged ────────────────────────────────────

def test_tokenize_offset_zero_matches_v1_layout():
    torch.manual_seed(0)
    traj = torch.randint(0, 10, (1, 3, 2, 8, 8))
    t0 = tokenize_trajectory(traj, 10, 2)
    t1 = tokenize_trajectory(traj, 10, 2, offset=0)
    assert np.array_equal(t0, t1)
    assert t0.max() == 10001 and (t0 == 10000).sum() == 6  # START per frame
    shifted = tokenize_trajectory(traj, 10, 2, offset=1)
    assert np.array_equal(shifted.astype(np.int64),
                          t0.astype(np.int64) + 1)
    assert shifted.min() >= 1  # id 0 reserved


def test_sample_rule_config_default_uses_global_stream():
    cfg = NCAConfig()
    random.seed(123)
    a = sample_rule_config(cfg)
    random.seed(123)
    b = sample_rule_config(cfg)
    assert a == b
    # dedicated rng leaves the global stream untouched
    random.seed(123)
    _ = sample_rule_config(cfg, rng=random.Random(7))
    c = sample_rule_config(cfg)
    assert c == b  # global stream advanced exactly as if the rng call absent


def test_v1_token_map_unchanged():
    tm = token_map(NCAConfig())  # reserve_eos defaults False
    assert tm["START"] == 10000 and tm["VAL0"] == 10023
    assert "EOS" not in tm
    tm3 = token_map(NCAConfig(reserve_eos=True))
    assert tm3["EOS"] == 0 and tm3["START"] == 10001 and tm3["VAL0"] == 10024


# ── v3 ledger semantics ───────────────────────────────────────────────────────

@pytest.mark.parametrize("p_reuse,marked", [(0.0, True), (1.0, True),
                                            (0.7, False)])
def test_last_write_wins_gold(p_reuse, marked):
    cfg = _tiny_dyadic(p_tag_reuse=p_reuse, mark_turns=marked)
    tm = token_map(cfg.nca)
    saw_query = saw_overwrite = False
    for seed in range(6):
        rs, ro = _mk_rule(cfg.nca, seed * 2), _mk_rule(cfg.nca, seed * 2 + 1)
        traj = simulate_dyadic(cfg, rs, ro, seed=100 + seed)
        toks = serialize_dyadic(traj, cfg)
        q_pos = np.where(toks == tm["QUERY"])[0]
        c_pos = np.where(toks == tm["COMMIT"])[0]
        for qp in q_pos:
            saw_query = True
            prior = [cp for cp in c_pos
                     if cp < qp and toks[cp + 1] == toks[qp + 1]]
            assert prior, "query with no prior write to its tag"
            assert toks[max(prior) + 2] == toks[qp + 2], \
                "gold is not the last write (ill-posed ledger regression)"
        evs = queries_in_stream_order(traj)
        assert len(evs) == len(q_pos)
        saw_overwrite |= any(e["post_overwrite"] for e in evs)
        # vals distinct per episode → anti-shortcut is structural
        commit_vals = [int(toks[cp + 2]) for cp in c_pos]
        assert len(set(commit_vals)) == len(commit_vals)
        # VAL tokens only at commit/query answer slots
        val_pos = set(np.where((toks >= tm["VAL0"])
                               & (toks < tm["VAL0"] + N_VALS))[0].tolist())
        assert val_pos <= set((c_pos + 2).tolist()) | set((q_pos + 2).tolist())
        # alphabet safety
        if cfg.nca.reserve_eos:
            assert (toks == 0).sum() == 0
        # marking
        turns = ((toks == tm["TURN_SELF"]) | (toks == tm["TURN_OTHER"])).sum()
        assert (turns > 0) == marked
    assert saw_query
    if p_reuse >= 1.0:
        assert saw_overwrite, "reuse=1.0 never produced a post-overwrite query"


def test_min_lag_respected():
    cfg = _tiny_dyadic(p_tag_reuse=0.5)
    for seed in range(4):
        rs, ro = _mk_rule(cfg.nca, seed), _mk_rule(cfg.nca, seed + 50)
        traj = simulate_dyadic(cfg, rs, ro, seed=300 + seed)
        for e in queries_in_stream_order(traj):
            assert e["lag_steps"] >= cfg.min_query_lag_steps


def test_counterfactual_twin_shares_prefix():
    cfg = _tiny_dyadic()
    rs, ro = _mk_rule(cfg.nca, 1), _mk_rule(cfg.nca, 2)
    pert = simulate_dyadic(cfg, rs, ro, seed=42, with_perturb=True)
    base = simulate_dyadic(cfg, rs, ro, seed=42, with_perturb=False)
    ps = [e for e in pert.events if e["kind"] == "perturb"][0]["step"]
    assert torch.equal(pert.frames[:ps], base.frames[:ps])
    assert not torch.equal(pert.frames[ps:], base.frames[ps:])


# ── The writer ──────────────────────────────────────────────────────────────

def _writer_args(tmp_path: Path, **over):
    argv = ["--out", str(tmp_path), "--target-tokens", "60000",
            "--seed", "77", "--orig-frac", "0.3",
            "--bands", "0.0:1.0",           # accept-first: keep the test fast
            "--geometries", "8x8x2:1.0", "--orig-steps", "16",
            "--dyad-grid", "8", "--dyad-steps", "24", "--dyad-burn-in", "2",
            "--commits", "4:6", "--queries", "4:8", "--min-lag", "3",
            "--filter-attempts", "3"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return gen.build_parser().parse_args(argv)


def test_writer_token_mix_eos_and_sidecar(tmp_path):
    gen.write_shard(_writer_args(tmp_path))
    data = np.memmap(tmp_path / "shard_0.bin", dtype=np.uint16, mode="r")
    recs = [json.loads(l) for l in
            (tmp_path / "shard_0.sidecar.jsonl").read_text().splitlines()]
    assert len(recs) > 20
    # every episode followed by exactly one EOS at rec["end"]
    for r in recs:
        assert data[r["end"]] == gen.EOS_ID
        seg = data[r["start"]:r["end"]]
        assert (seg == gen.EOS_ID).sum() == 0
    assert int((data == gen.EOS_ID).sum()) == len(recs)
    # offsets tile the file exactly
    assert recs[0]["start"] == 0
    for a, b in zip(recs, recs[1:]):
        assert b["start"] == a["end"] + 1
    assert recs[-1]["end"] + 1 == len(data)
    # token-level mix lands near target (debt scheduler)
    orig_tok = sum(r["end"] - r["start"] + 1 for r in recs
                   if r["family"] == "orig")
    frac = orig_tok / len(data)
    assert 0.2 < frac < 0.4, f"orig token fraction {frac:.2%} off target 30%"
    # sidecar carries the design fields
    dy = [r for r in recs if r["family"] == "dyadic"]
    assert dy and all("p_tag_reuse" in r and "marked" in r
                      and r["rules"][0]["gzip"] > 0 for r in dy)
    assert any(r["marked"] for r in dy) and any(not r["marked"] for r in dy)
    # ramp: reuse grows across the stream
    assert dy[0]["p_tag_reuse"] < dy[-1]["p_tag_reuse"]


def test_writer_deterministic_and_global_rng_untouched(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    random.seed(999)
    state_before = random.getstate()
    gen.write_shard(_writer_args(a_dir, target_tokens=30000))
    assert random.getstate() == state_before, "writer consumed global random"
    gen.write_shard(_writer_args(b_dir, target_tokens=30000))
    assert (a_dir / "shard_0.bin").read_bytes() == \
           (b_dir / "shard_0.bin").read_bytes()
    assert (a_dir / "shard_0.sidecar.jsonl").read_text() == \
           (b_dir / "shard_0.sidecar.jsonl").read_text()


def test_finalize_merges_and_emits_evals(tmp_path):
    args = _writer_args(tmp_path, target_tokens=30000)
    gen.write_shard(args)
    fin = _writer_args(tmp_path, target_tokens=30000)
    fin.finalize = True
    fin.n_eval_rule_pairs = 2
    fin.n_eval_ledger = 6
    fin.n_eval_smoothing = 2
    gen.finalize(fin)
    meta = json.loads((tmp_path / "meta.json").read_text())
    data = np.memmap(tmp_path / "nca_v3.bin", dtype=np.uint16, mode="r")
    assert meta["total_tokens"] == len(data)
    recs = [json.loads(l) for l in
            (tmp_path / "sidecar.jsonl").read_text().splitlines()]
    for r in recs:
        assert data[r["end"]] == gen.EOS_ID
    uniq = [json.loads(l) for l in
            (tmp_path / "eval_ledger_unique.jsonl").read_text().splitlines()]
    over = [json.loads(l) for l in
            (tmp_path / "eval_ledger_overwrite.jsonl").read_text().splitlines()]
    assert len(uniq) == 6 and len(over) == 6
    assert all(not it["post_overwrite"] for it in uniq)
    assert any(it["post_overwrite"] for it in over)
    assert all(0 <= it["gold_val"] < N_VALS for it in uniq + over)
    smo = [json.loads(l) for l in
           (tmp_path / "eval_smoothing.jsonl").read_text().splitlines()]
    assert len(smo) == 2
    assert all(p["true_frame"] != p["counterfactual_frame"] for p in smo)
