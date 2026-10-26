"""Transcript UI: candidates are rendered, cut and vetted like self-play; the page and API serve end to end."""
from __future__ import annotations

import json
import threading
import urllib.request

import pytest

httpx = pytest.importorskip("httpx")

from kotodama.posttrain import transcript_ui as ui  # noqa: E402


def _mock_client(seen):
    def handler(request):
        if request.url.path == "/info":
            return httpx.Response(200, json={"name": "tsumugi-test", "checkpoint": "/x/step.pt"})
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [
            {"text": " The kettle's on.\n\nUser:", "finish_reason": "stop", "token_ids": {"completion": [1, 2]}},
            {"text": " As an AI language model, I can't.", "finish_reason": "stop", "token_ids": {"completion": [1]}},
            {"text": " Fine.\n\nInez: who?", "finish_reason": "stop", "token_ids": {"completion": [1]}},
        ][: body["n"]]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_candidates_render_cut_and_vet() -> None:
    seen = []
    req = {"header": "Two friends.", "turns": [{"speaker": "User", "text": "Morning."}], "speaker": "Night Owl",
           "cast": ["User", "Night Owl"], "n": 3, "temperature": 0.8, "repetition_penalty": 1.2,
           "steer": [{"name": "vrep", "alpha": -0.05}, {"name": "v7-entropy", "alpha": 0}]}
    out = ui.candidates(_mock_client(seen), "http://b:1", req)
    body = seen[0]
    assert body["prompt"] == "Two friends.\n\nUser: Morning.\n\nNight Owl:"
    assert body["stop"] == ["\n\nUser:", "\n\nNight Owl:"] and body["frequency_penalty"] == pytest.approx(0.2)
    assert body["vectors"] == [{"name": "vrep", "alpha": -0.05}]  # zero doses are not sent
    assert out[0]["text"] == "The kettle's on." and out[0]["vetoes"] == [] and out[0]["ended_by"] == "stop"
    assert "meta_leak" in out[1]["vetoes"]
    assert out[2]["text"] == "Fine." and out[2]["cut_label"] == "Inez"


def test_parse_backends() -> None:
    assert ui.parse_backends(["a=http://h:1", "b=http://h:2"]) == {"a": "http://h:1", "b": "http://h:2"}
    with pytest.raises(SystemExit):
        ui.parse_backends(["nourl"])


def test_server_end_to_end() -> None:
    import http.server

    seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ui.make_handler({"t": "http://b:1"}, _mock_client(seen)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"Transcript Loom" in urllib.request.urlopen(base + "/").read()
        bs = json.loads(urllib.request.urlopen(base + "/api/backends").read())
        assert bs == [{"key": "t", "name": "tsumugi-test", "checkpoint": "/x/step.pt", "ok": True}]
        req = urllib.request.Request(base + "/api/candidates", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"backend": "t", "turns": [], "speaker": "A", "cast": ["A", "B"],
                                                      "n": 1}).encode())
        cands = json.loads(urllib.request.urlopen(req).read())["candidates"]
        assert cands[0]["text"] == "The kettle's on."
        bad = urllib.request.Request(base + "/api/candidates", method="POST", data=b'{"backend": "nope"}')
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        srv.shutdown()
