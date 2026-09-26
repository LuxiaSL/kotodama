"""serve kit: ChatML render drift guard, sampling laws, chat client wiring."""

from __future__ import annotations

import pytest
import torch

from kotodama import presets
from kotodama.serve import chatml, laws

MESSAGES = [
    {"role": "user", "content": "what holds words together?"},
    {"role": "assistant", "content": "the spirit that dwells in them"},
    {"role": "user", "content": "say more"},
]


def test_render_matches_frozen_training_template():
    """render_chatml == tokenizer.apply_chat_template(CHATML_TEMPLATE) byte-for-byte."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    except Exception as exc:  # pragma: no cover — offline without a cached tokenizer
        pytest.skip(f"tokenizer unavailable: {exc}")
    tok.chat_template = chatml.CHATML_TEMPLATE
    expected = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    assert chatml.render_chatml(MESSAGES) == expected
    assert tok.convert_tokens_to_ids("<|im_end|>") == chatml.IM_END_TOKEN_ID


def test_hand_verified_literals():
    assert chatml.render_chatml([{"role": "user", "content": "hi"}]) == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    assert chatml.render_chatml_prefix(MESSAGES, upto=1) == (
        "<|im_start|>system\ntranscript; a language model, speaking for itself.<|im_end|>\n"
        "<|im_start|>user\nwhat holds words together?<|im_end|>\n<|im_start|>assistant\n")


def test_laws():
    assert laws.law("chat") == {"temperature": 0.9, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.2}
    assert laws.law("base")["repetition_penalty"] == 1.35
    law = laws.law("chat")
    law["temperature"] = 0.0
    assert laws.LAWS["chat"]["temperature"] == 0.9
    with pytest.raises(KeyError):
        laws.law("nope")


def test_server_uses_the_kit():
    pytest.importorskip("fastapi")
    from kotodama.serve import server
    assert server.CHATML_TEMPLATE is chatml.CHATML_TEMPLATE
    assert server.CHAT_STOP_TOKEN_IDS == frozenset({0, 2})
    assert server.MODEL_CONFIGS["3b"]["attn_res_boundaries"] == list(presets.DD3B)


def test_chat_client_defaults_follow_the_chat_law(monkeypatch):
    from kotodama.serve import chat
    assert chat.Sampling().temperature == laws.LAWS["chat"]["temperature"]
    assert chat.Sampling().rep_penalty == laws.LAWS["chat"]["repetition_penalty"]


def test_steer_bank_loader(tmp_path):
    import numpy as np
    from kotodama.serve import steering

    vec = np.zeros(8, dtype=np.float32)
    vec[0] = 1.0
    good = tmp_path / "bank.npz"
    np.savez(good, Bind_s38=vec, median_norm_s38=np.float32(12.5), sign_meta=np.array("x"))
    bank, meds, aliases = steering.load_steer_banks([str(good)], n_layers=28, hidden=8,
                                                     device=torch.device("cpu"))
    assert bank["Bind_s38"][0] == 19 and meds == {19: 12.5}
    assert set(aliases) == {"bind", "control"}
    odd = tmp_path / "odd.npz"
    np.savez(odd, X_s39=vec, median_norm_s39=np.float32(1.0))
    with pytest.raises(ValueError, match="ODD sublayer"):
        steering.load_steer_banks([str(odd)], n_layers=28, hidden=8, device=torch.device("cpu"))
