"""OpenAI-compatible request models tolerate client quirks: null = "not sent" (default applies), a one-element prompt
list is unwrapped, and ranges still validate (Loom sends unchecked fields as null)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from kotodama.serve.server import OAIChatRequest, OAICompletionRequest


def test_nulls_take_defaults() -> None:
    r = OAICompletionRequest.model_validate({"prompt": "there", "n": 5, "temperature": 0.7, "frequency_penalty": 0.2,
                                             "top_p": None, "presence_penalty": None, "best_of": None, "stop": None})
    d = OAICompletionRequest(prompt="x")
    assert r.top_p == d.top_p and r.presence_penalty == d.presence_penalty and r.n == 5 and r.temperature == 0.7


def test_prompt_list() -> None:
    assert OAICompletionRequest.model_validate({"prompt": ["there"]}).prompt == "there"
    with pytest.raises(ValidationError):
        OAICompletionRequest.model_validate({"prompt": ["a", "b"]})


def test_ranges_still_enforced() -> None:
    with pytest.raises(ValidationError):
        OAICompletionRequest.model_validate({"prompt": "x", "n": 9})
    with pytest.raises(ValidationError):
        OAICompletionRequest.model_validate({"prompt": "x", "frequency_penalty": -0.5})


def test_chat_nulls() -> None:
    r = OAIChatRequest.model_validate({"messages": [{"role": "user", "content": "hi"}], "top_p": None,
                                       "presence_penalty": None})
    assert r.top_p == OAIChatRequest(messages=[{"role": "user", "content": "hi"}]).top_p
