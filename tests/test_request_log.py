"""The request log (--request_log, kotodama.serve.request_log): rows by day, compression on rollover, and the server's
rows (request as sent, response, exact candidate ids; opt-out; no client address)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import torch

from kotodama.serve.request_log import RequestLog


def _read(path: Path) -> list[dict]:
    if path.suffix == ".zst":
        import zstandard

        raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb")).read()
    elif path.suffix == ".gz":
        raw = gzip.open(path).read()
    else:
        raw = path.read_bytes()
    return [json.loads(x) for x in raw.decode().splitlines()]


def test_rows_split_by_day_and_a_finished_day_is_compressed(tmp_path) -> None:
    log = RequestLog(tmp_path, writer="w")
    day1, day2 = 1_700_000_000.0, 1_700_000_000.0 + 86_400
    log.write({"ts": day1, "n": 1})
    log.write({"ts": day1 + 5, "n": 2})
    log.write({"ts": day2, "n": 3})  # the next day: the first one is compressed in the background
    log.close()
    d1, d2 = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    for _ in range(100):  # the rollover's compression runs on its own thread
        done = [p for p in d1.iterdir() if p.suffix in (".zst", ".gz")]
        if done and not (d1 / "w.jsonl").exists():
            break
        import time

        time.sleep(0.05)
    assert [r["n"] for r in _read(done[0])] == [1, 2] and not (d1 / "w.jsonl").exists()
    # the last day's file is compressed at close only once its day is over (day2 is long past here)
    files = list(d2.iterdir())
    assert len(files) == 1 and [r["n"] for r in _read(files[0])] == [3]
    assert log.stats["rows"] == 3 and log.stats["errors"] == 0


def test_todays_file_stays_open_at_close(tmp_path) -> None:
    import time

    log = RequestLog(tmp_path, writer="w")
    log.write({"ts": time.time(), "n": 1})
    log.close()
    (day,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert [p.name for p in day.iterdir()] == ["w.jsonl"]


class _FakeTok:
    def encode(self, text, return_tensors=None, **_):
        ids = list(range(1, len(text.split()) + 2))
        return torch.tensor([ids]) if return_tensors == "pt" else ids


def test_the_server_logs_each_request_with_its_candidates(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from kotodama.serve import server

    async def fake_run_generation(fn, req, **_):
        return server.GenerateResponse(text=" ok", prompt_tokens=2, completion_tokens=2, tokens_per_second=1.0,
                                       prompt_ids=[5, 6], completion_ids=[7, 8], stop_id=0)

    log = RequestLog(tmp_path, writer="replica-a")
    for k, v in {"_model": object(), "_tokenizer": _FakeTok(), "_engine_compatible": lambda req: True,
                 "_run_generation": fake_run_generation, "_request_log": log,
                 "_request_log_weights": "w" * 64}.items():
        monkeypatch.setattr(server, k, v)
    client = TestClient(server.app)  # no context manager: lifespan (model load) does not run
    r = client.post("/v1/completions", json={"prompt": "hello there", "max_tokens": 4, "n": 2, "user": "mgr-1"})
    assert r.status_code == 200, r.text
    r = client.post("/v1/completions", json={"prompt": "private", "max_tokens": 4, "log": False})
    assert r.status_code == 200, r.text
    r = client.post("/v1/completions", json={"prompt": "kept", "max_tokens": 4, "log": True})
    assert r.status_code == 200, r.text  # "log" is accepted, whichever way it is set
    client.get("/health")  # not a logged endpoint
    log.close()
    (path,) = [p for d in tmp_path.iterdir() for p in d.iterdir()]
    rows = _read(path)
    assert [row["request"]["prompt"] for row in rows] == ["hello there", "kept"]
    row = rows[0]
    assert row["endpoint"] == "/v1/completions" and row["status"] == 200 and row["writer"] == "replica-a"
    assert row["weights"] == "w" * 64 and row["request"]["user"] == "mgr-1" and not row["stream"]
    assert [c["text"] for c in row["response"]["choices"]] == [" ok", " ok"]
    assert [(c["prompt_ids"], c["completion_ids"], c["stop_id"]) for c in row["candidates"]] == [([5, 6], [7, 8], 0)] * 2
    assert not {"client", "ip", "host", "remote"} & set(row) and "mgr-1" not in json.dumps(row["response"])
    assert log.stats["opted_out"] == 1


def test_a_streamed_response_is_logged_as_its_text() -> None:
    pytest.importorskip("fastapi")
    from kotodama.serve import server

    engine = b'data: {"token": "Hel"}\n\ndata: {"token": "lo"}\n\ndata: {"done": true, "sign_id": "s-0"}\n\n'
    out = server._stream_text(engine)
    assert out["text"] == "Hello" and out["events"] == 3 and out["last"]["sign_id"] == "s-0"
    chat = (b'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]}\n\n'
            b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n')
    out = server._stream_text(chat)
    assert out["text"] == "Hi" and out["events"] == 2
