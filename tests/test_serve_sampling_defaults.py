"""Server sampling defaults (--default_repetition_penalty) and the /score_tokens NLL read."""

from __future__ import annotations

import math

import pytest
import torch

from kotodama.serve.logprobs import LogprobLimitError, score_token_nll_batch


def test_resolve_oai_repetition_penalty(monkeypatch):
    pytest.importorskip("fastapi")
    from kotodama.serve import server
    assert server.resolve_oai_repetition_penalty(None) == server.DEFAULT_REPETITION_PENALTY
    monkeypatch.setattr(server, "_default_repetition_penalty", 1.05)
    assert server.resolve_oai_repetition_penalty(None) == 1.05
    assert server.resolve_oai_repetition_penalty(0.0) == 1.0
    assert server.resolve_oai_repetition_penalty(0.3) == pytest.approx(1.3)
    # the native request's default follows the flag too
    assert server.GenerateRequest(prompt="x").repetition_penalty == 1.05
    assert server.GenerateRequest(prompt="x", repetition_penalty=1.4).repetition_penalty == 1.4


class _FakeTok:
    def encode(self, text, return_tensors=None, **_):
        ids = list(range(1, len(text.split()) + 2))
        return torch.tensor([ids]) if return_tensors == "pt" else ids


@pytest.mark.parametrize("payload,expected", [
    ({}, 1.07),                          # absent -> the flag's value
    ({"frequency_penalty": 0.0}, 1.0),   # explicit 0 -> off
    ({"frequency_penalty": 0.25}, 1.25),
])
def test_oai_completions_route_resolves_penalty(monkeypatch, payload, expected):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    seen: list[float] = []

    async def fake_run_generation(fn, req, **_):
        seen.append(req.repetition_penalty)
        return server.GenerateResponse(text=" ok", prompt_tokens=2, completion_tokens=1,
                                       tokens_per_second=1.0)

    monkeypatch.setattr(server, "_default_repetition_penalty", 1.07)
    monkeypatch.setattr(server, "_model", object())
    monkeypatch.setattr(server, "_tokenizer", _FakeTok())
    monkeypatch.setattr(server, "_engine_compatible", lambda req: True)
    monkeypatch.setattr(server, "_run_generation", fake_run_generation)
    client = TestClient(server.app)  # no context manager: lifespan (model load) does not run
    r = client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 4, **payload})
    assert r.status_code == 200, r.text
    assert seen == [pytest.approx(expected)]


def test_score_token_nll_batch_matches_manual():
    torch.manual_seed(0)
    V = 11
    table = torch.randn(V, V)            # bigram "model": logits at t depend on token t

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return table[ids]

    items = [
        {"id": "a", "prompt_ids": [1, 2], "completion_ids": [3, 4, 5]},
        {"id": "b", "prompt_ids": [7], "completion_ids": [8]},
        {"id": "c", "prompt_ids": [2, 2, 2, 2], "completion_ids": []},
    ]
    out = score_token_nll_batch(logits_fn, items, "cpu", vocab_size=V)
    lsm = torch.log_softmax(table, dim=-1)
    want_a = -(lsm[2, 3] + lsm[3, 4] + lsm[4, 5]).item()
    assert out[0]["id"] == "a" and out[0]["n_tokens"] == 3
    assert math.isclose(out[0]["sum_nll"], want_a, rel_tol=1e-5)
    assert math.isclose(out[0]["mean_nll"], want_a / 3, rel_tol=1e-5)
    assert math.isclose(out[1]["sum_nll"], -lsm[7, 8].item(), rel_tol=1e-5)
    assert out[2]["n_tokens"] == 0 and out[2]["mean_nll"] is None


def test_score_token_nll_batch_limits():
    f = lambda ids: torch.zeros(*ids.shape, 5)  # noqa: E731
    with pytest.raises(LogprobLimitError):
        score_token_nll_batch(f, [{"id": "x", "prompt_ids": [1], "completion_ids": [9]}], "cpu",
                              vocab_size=5)
    with pytest.raises(LogprobLimitError):
        score_token_nll_batch(f, [{"id": "x", "prompt_ids": [1, 2], "completion_ids": [3]}], "cpu",
                              max_seq_len=2)
