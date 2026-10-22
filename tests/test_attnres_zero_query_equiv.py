"""AttnRes at zero queries == a plain residual stream (up to RMSNorm eps): uniform routing
averages the sources, the sources sum to the residual stream, and every consumer is
pre-normed, so the 1/n_sources scale vanishes. Hence a no-AttnRes checkpoint (e.g. an NCA
phase run with attn_res=False) loads into an AttnRes model function-preservingly."""

from __future__ import annotations

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig


def _cfg(attn_res: bool, bounds: list[int] | None) -> KotodamaModelConfig:
    return KotodamaModelConfig(
        hidden_size=64, num_layers=8, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
        attn_res=attn_res, attn_res_boundaries=bounds, use_liger=False, norm_eps=1e-12,
    )


def _logits(m: KotodamaModel, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = m(ids)
    return out["logits"] if isinstance(out, dict) else out


@pytest.mark.parametrize("bounds", [[0, 1, 3, 5], [0, 2, 4, 6], [0]])
def test_plain_checkpoint_into_attnres_is_function_preserving(bounds):
    torch.manual_seed(0)
    plain = KotodamaModel(_cfg(False, None)).double().eval()
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(0.05 * torch.randn_like(p))
    ids = torch.randint(3, 199, (2, 40))
    ref = _logits(plain, ids)

    ar = KotodamaModel(_cfg(True, bounds)).double().eval()
    missing, unexpected = ar.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all("res_" in k for k in missing)
    assert (_logits(ar, ids) - ref).abs().max().item() < 1e-8

    with torch.no_grad():  # sanity: trained (non-zero) queries do change the function
        for n, p in ar.named_parameters():
            if n.endswith("res_query"):
                p.normal_(0, 0.5)
    assert (_logits(ar, ids) - ref).abs().max().item() > 1e-2
