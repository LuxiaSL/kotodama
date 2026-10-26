"""--attnres_res_gain: the AttnRes routing gain (res_norm.weight) only ever multiplies the routing query, so folding it
into the query (q <- q*g, g <- 1) must leave every forward bitwise unchanged — including a checkpoint whose queries and
gains drifted along the (q*c, g/c) symmetry under the learned mode — and the frozen / folded modes must keep the gains
out of training."""

from __future__ import annotations

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training import train


def _cfg(mode: str = "learned") -> KotodamaModelConfig:
    return KotodamaModelConfig(
        hidden_size=64, num_layers=6, num_attention_heads=2, num_kv_heads=1, head_dim=32, intermediate_size=128,
        vocab_size=97, max_position_embeddings=32, attn_res=True, attn_res_boundaries=[0, 1, 3, 5],
        loop_start=1, loop_len=2, loop_times=2, attn_impl="sdpa", attn_res_gain=mode,
    )


def _drifted_learned_state(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """A 'learned' model whose routing queries and gains moved away from init (as on the trunk: |q| up, g down)."""
    torch.manual_seed(0)
    m = KotodamaModel(_cfg("learned")).to(dtype)
    with torch.no_grad():
        for query, norm in m._attn_res_query_norm_pairs():
            query.copy_(torch.randn_like(query) * 0.7)
            norm.weight.copy_(0.3 + torch.rand_like(norm.weight))
    return m.state_dict()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_folded_loads_learned_checkpoint_with_identical_outputs(dtype):
    state = _drifted_learned_state(dtype)
    ids = torch.randint(0, 97, (2, 32), generator=torch.Generator().manual_seed(1))

    learned = KotodamaModel(_cfg("learned")).to(dtype).eval()
    learned.load_state_dict(state)
    folded = KotodamaModel(_cfg("folded")).to(dtype).eval()
    folded.load_state_dict(state)
    n = folded.fold_attn_res_gains()
    assert n > 0
    with torch.no_grad():
        ref = learned(ids)["logits"]
        out = folded(ids)["logits"]
    assert torch.equal(ref, out)
    for query, norm in folded._attn_res_query_norm_pairs():
        assert bool((norm.weight == 1).all())
    assert folded.fold_attn_res_gains() == 0  # idempotent
    with torch.no_grad():
        assert torch.equal(folded(ids)["logits"], ref)


@pytest.mark.parametrize("mode", ["frozen", "folded"])
def test_gains_not_trained(mode):
    m = KotodamaModel(_cfg(mode))
    for _query, norm in m._attn_res_query_norm_pairs():
        assert norm.weight.requires_grad is False
    trained = {id(p) for p in m.parameters() if p.requires_grad}
    assert any(id(q) in trained for q, _n in m._attn_res_query_norm_pairs())


def test_learned_default_trains_gains():
    m = KotodamaModel(_cfg())
    sites = m._attn_res_query_norm_pairs()
    assert sum(n.weight.requires_grad for _q, n in sites) >= len(sites) - 1  # the first boundary's pre-attn is frozen


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        _cfg("tied")


def _args(*extra: str):
    import sys

    old = sys.argv
    try:
        sys.argv = ["train.py", "--model_size", "proxy", *extra]
        return train.parse_args()
    finally:
        sys.argv = old


def test_flag_parses_and_defaults():
    assert _args().attnres_res_gain == "learned"
    assert _args("--attnres_res_gain", "folded").attnres_res_gain == "folded"
    assert _args("--config", "configs/tsumugi-1b.yaml").attnres_res_gain == "learned"  # trunk lineage
