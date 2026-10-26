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


def test_completion_maps_to_generate_with_steering() -> None:
    from kotodama.serve.server import oai_completion_to_generate

    r = OAICompletionRequest.model_validate({
        "prompt": "Mara:", "max_tokens": 64, "temperature": 0.75, "top_p": 0.0, "frequency_penalty": 0.2,
        "stop": "\n\nTeo:", "vectors": [{"name": "vrep", "alpha": -0.05}, {"name": "V7_s58", "alpha": 0.03}]})
    g = oai_completion_to_generate(r)
    assert g.prompt == "Mara:" and g.max_new_tokens == 64 and g.temperature == 0.75
    assert g.repetition_penalty == 1.2 and g.stop_strings == ["\n\nTeo:"]
    assert [(v.name, v.alpha) for v in g.vectors] == [("vrep", -0.05), ("V7_s58", 0.03)]
    single = oai_completion_to_generate(OAICompletionRequest.model_validate(
        {"prompt": "x", "steer_model": "vrep", "steer_alpha": None}))
    assert single.steer_model == "vrep" and single.steer_alpha is None and single.vectors is None
