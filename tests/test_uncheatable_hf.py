"""The HF Uncheatable runner: document-start token and chunk-size rules, and
static-shape padding leaving an HF causal LM's document NLL unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from kotodama.eval.uncheatable import doc_nll  # noqa: E402
from scripts.eval.run_uncheatable_hf import chunk_size_for, document_start_token  # noqa: E402


def test_document_start_prefers_bos_then_eos():
    assert document_start_token(SimpleNamespace(bos_token_id=128000, eos_token_id=128001)) == (
        "tokenizer_bos", 128000)
    assert document_start_token(SimpleNamespace(bos_token_id=None, eos_token_id=151643)) == (
        "eos_as_bos", 151643)
    with pytest.raises(ValueError):
        document_start_token(SimpleNamespace(bos_token_id=None, eos_token_id=None))


def test_chunk_size_capped_by_context():
    assert chunk_size_for(4096) == 4000
    assert chunk_size_for(4001) == 4000
    assert chunk_size_for(4000) == 3999
    assert chunk_size_for(2048) == 2047


def test_static_shape_padding_matches_unpadded_on_hf_model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()

    @torch.no_grad()
    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids).logits[0].float()

    ids = torch.randint(1, 64, (70,)).tolist()  # 3 chunks of 30: 30, 30, 10
    plain = doc_nll(logits_fn, ids, bos_id=0, chunk_size=30, loss_precision="fp32")
    padded = doc_nll(logits_fn, ids, bos_id=0, chunk_size=30, loss_precision="fp32", pad_to=31)
    assert padded == pytest.approx(plain, abs=1e-4)
