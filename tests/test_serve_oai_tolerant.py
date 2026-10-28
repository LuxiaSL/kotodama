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
        OAICompletionRequest.model_validate({"prompt": "x", "n": 65})
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


def test_token_id_prompts_are_accepted() -> None:
    assert OAICompletionRequest.model_validate({"prompt": [5, 6, 7]}).prompt == [5, 6, 7]
    assert OAICompletionRequest.model_validate({"prompt": [[5, 6, 7]]}).prompt == [5, 6, 7]
    with pytest.raises(ValidationError):
        OAICompletionRequest.model_validate({"prompt": ["a", "b"]})


def test_fields_the_server_would_ignore_are_refused() -> None:
    """A field the server would accept and ignore is a 400: a nonzero presence_penalty, best_of above 1, an unknown
    field that is on (a sampling law must never silently differ from what the client sent), and a completions-only
    field on chat. Off values (null, false, 0, empty) and the gateway's affinity key pass."""
    from fastapi import HTTPException

    from kotodama.serve.server import GenerateRequest, _refuse_unserved

    def refused(model, body: dict) -> bool:
        try:
            _refuse_unserved(model.model_validate(body))
        except HTTPException as exc:
            assert exc.status_code == 400
            return True
        return False

    chat = {"messages": [{"role": "user", "content": "hi"}]}
    for body in ({"presence_penalty": 0.5}, {"best_of": 3}, {"logit_bias": {"5": 2.0}}, {"logprobs": 5},
                 {"echo": True}, {"suffix": "end"}, {"repetition_penalty": 1.3}, {"top_k": 40}, {"stream": True}):
        assert refused(OAICompletionRequest, {"prompt": "x", **body}), body
    for body in ({"presence_penalty": 0.0}, {"best_of": 1}, {"logprobs": None}, {"logprobs": 0}, {"echo": False},
                 {"logit_bias": {}}, {"user": "pin/tree-3"}, {"frequency_penalty": 0.2, "min_p": 0.02, "seed": 4}):
        assert not refused(OAICompletionRequest, {"prompt": "x", **body}), body
    assert refused(OAIChatRequest, {**chat, "presence_penalty": 0.3})
    assert refused(OAIChatRequest, {**chat, "min_p": 0.05})  # served on /v1/completions, not on chat
    assert not refused(OAIChatRequest, {**chat, "stream": True, "user": "u"})
    assert refused(GenerateRequest, {"prompt": "x", "frequency_penalty": 0.2})  # /generate takes repetition_penalty
    assert not refused(GenerateRequest, {"prompt": "x", "repetition_penalty": 1.2, "top_k": 40})
