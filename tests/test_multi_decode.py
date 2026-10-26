"""MultiSampleDecoder on a looped model (tiny moonshot geometry, CPU fp32): each row must reproduce the full no-cache
forward of the prompt plus that row's own continuation, steering must match the single-stream engine's, and the
per-row sampler must keep its presence sets and stops separate."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import MultiSampleDecoder, decode_rows

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
ROWS = 4


def _model() -> KotodamaModel:
    torch.manual_seed(20261026)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    return _model()


def _engine(model: KotodamaModel, steer: bool = False) -> DecodeEngine:
    return DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=steer)


def _ids(n: int, seed: int) -> torch.Tensor:
    return torch.randint(4, CFG["vocab_size"], (1, n), generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def test_rows_match_full_forward_of_their_own_continuations(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, ROWS)
    prompt = _ids(11, 1)
    forced = torch.cat([_ids(9, 10 + r) for r in range(ROWS)])  # (ROWS, 9): a different continuation per row
    logits0 = eng.prefill_reference(prompt)
    dec.start(logits0, SamplingParams(temperature=0.8, repetition_penalty=1.2))
    got = [logits0.expand(ROWS, -1)]
    for t in range(forced.shape[1] - 1):
        got.append(dec.step_logits(forced[:, t:t + 1]).float())
    got_t = torch.stack(got, dim=1)  # (ROWS, 9, V)
    for r in range(ROWS):
        full = model(torch.cat([prompt, forced[r:r + 1]], dim=1))["logits"][0].float()
        want = full[prompt.shape[1] - 1:-1]
        assert torch.allclose(got_t[r], want, atol=2e-4, rtol=1e-4), (r, (got_t[r] - want).abs().max())


@torch.inference_mode()
def test_steered_rows_match_the_single_stream_engine(model: KotodamaModel) -> None:
    eng = _engine(model, steer=True)
    dec = MultiSampleDecoder(eng, ROWS)
    vec = torch.randn(CFG["hidden_size"], generator=torch.Generator().manual_seed(3))
    eng.set_steer([(4, vec, 0.7)])
    prompt, forced = _ids(10, 2), _ids(7, 20)
    logits0 = eng.prefill_reference(prompt)
    dec.start(logits0, SamplingParams(temperature=0.8, repetition_penalty=1.0))
    batched = [dec.step_logits(forced[:, t:t + 1].expand(ROWS, 1).contiguous()).float() for t in range(6)]
    eng.prefill_reference(prompt)
    single = [eng.step_logits(forced[:, t:t + 1])[0, -1].float() for t in range(6)]
    for b, s in zip(batched, single):
        assert torch.allclose(b, s.expand_as(b), atol=2e-4, rtol=1e-4)
    eng.set_steer(None)
    unsteered = _engine(model, steer=False)
    unsteered.prefill_reference(prompt)
    plain = unsteered.step_logits(forced[:, :1])[0, -1].float()
    assert (single[0] - plain).abs().max() > 1e-3  # the write really changed the step


@torch.inference_mode()
def test_presence_is_per_row_and_rows_diverge(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, ROWS)
    torch.manual_seed(0)
    logits0 = eng.prefill_reference(_ids(8, 4))
    params = SamplingParams(temperature=1.5, repetition_penalty=1.3)
    toks = dec.start(logits0, params).clone()
    seen = [{int(toks[r])} for r in range(ROWS)]
    for _ in range(10):
        toks = dec.step(params).clone()
        for r in range(ROWS):
            seen[r].add(int(toks[r]))
    for r in range(ROWS):
        assert set(dec.presence[r].nonzero().view(-1).tolist()) == seen[r]
    assert len({frozenset(s) for s in seen}) > 1  # independent noise per row


@torch.inference_mode()
def test_decode_rows_stops_per_row(model: KotodamaModel) -> None:
    eng = _engine(model)
    dec = MultiSampleDecoder(eng, ROWS)
    torch.manual_seed(1)
    logits0 = eng.prefill_reference(_ids(8, 5))
    params = SamplingParams(temperature=1.0, repetition_penalty=1.0)
    stop = frozenset(range(4, 60))  # a broad stop set: rows end at different steps
    rows = decode_rows(dec, logits0, params, n=3, max_new_tokens=12, stop_ids=stop)
    assert len(rows) == 3
    assert all(len(r) <= 12 and not (set(r) & stop) for r in rows)
    capped = decode_rows(dec, logits0, params, n=2, max_new_tokens=5, stop_ids=frozenset(),
                         stop_check=lambda ids: len(ids) >= 3)
    assert [len(r) for r in capped] == [3, 3]
    with pytest.raises(ValueError):
        decode_rows(dec, logits0, params, n=ROWS + 1, max_new_tokens=4, stop_ids=frozenset())
    with pytest.raises(ValueError):
        dec.start(logits0, SamplingParams(temperature=0.8, top_p=0.9))


class _CharTok:
    """Characters <-> ids inside the tiny vocab (4 + ord % 200)."""

    def encode(self, text, return_tensors=None, **_):
        ids = [4 + ord(c) % 200 for c in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i - 4) % 26) for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"{m['role']}: {m['content']} " for m in msgs) + "assistant:"


def test_server_routes_n_candidates_through_the_batched_decoder(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    eng = _engine(model, steer=True)
    dec = MultiSampleDecoder(eng, ROWS)
    calls: list[str] = []

    async def run_inline(fn, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": {ROWS: dec, 2: MultiSampleDecoder(eng, 2)}, "_device": torch.device("cpu"),
                 "_max_seq_len": 96, "_prefix_cache": False, "_stop_token_ids": frozenset({0}),
                 "_run_generation": run_inline}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 6, "n": 3,
                                             "temperature": 0.9, "top_p": 0, "return_token_ids": True})
    assert r.status_code == 200, r.text
    choices = r.json()["choices"]
    assert calls == ["engine_generate_multi"] and len(choices) == 3
    assert all(1 <= len(c["token_ids"]["completion"]) <= 6 for c in choices)
    assert eng.steer_buf.abs().sum() == 0  # steering state cleared after the request
    calls.clear()
    r = client.post("/v1/completions", json={"prompt": "two friends talk", "max_tokens": 4, "n": 2,
                                             "temperature": 0.9, "top_p": 0.9})  # truncated law: serial path
    assert r.status_code == 200 and calls == ["engine_generate", "engine_generate"]


def test_bucket_choice_and_chat_route(monkeypatch, model: KotodamaModel) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from kotodama.serve import server

    eng = _engine(model)
    buckets = {2: MultiSampleDecoder(eng, 2), ROWS: MultiSampleDecoder(eng, ROWS)}
    used: list[int] = []
    real = server.decode_rows

    def spy(dec, *a, **k):
        used.append(dec.rows)
        return real(dec, *a, **k)

    async def run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    for k, v in {"_model": model, "_tokenizer": _CharTok(), "_engine": eng, "_multi": buckets, "_device": torch.device("cpu"),
                 "_max_seq_len": 96, "_prefix_cache": False, "_stop_token_ids": frozenset({0}), "_serve_mode": "chat",
                 "_run_generation": run_inline, "decode_rows": spy}.items():
        monkeypatch.setattr(server, k, v)
    assert server._multi_bucket(2).rows == 2 and server._multi_bucket(3).rows == ROWS
    assert server._multi_bucket(ROWS + 1) is None
    client = TestClient(server.app)
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
                                                  "n": 3, "temperature": 0.9, "top_p": 0})
    assert r.status_code == 200, r.text
    assert len(r.json()["choices"]) == 3 and used == [ROWS]
    r = client.post("/v1/completions", json={"prompt": "hey", "max_tokens": 5, "n": 2, "temperature": 0.9, "top_p": 0})
    assert r.status_code == 200 and used == [ROWS, 2]
