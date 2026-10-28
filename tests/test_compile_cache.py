"""The compiled-artifact store (kotodama.serve.compile_cache) and the gateway's compile_first startup: the key moves
with the model code and the engine flags (and not with anything that leaves the compiled code alone), artifacts
round-trip through the store, a missing or broken blob is a cold compile rather than a failed start, and the
gateway starts a pool's other replicas only once its first is ready."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kotodama.serve import compile_cache


def _fields(monkeypatch, src: Path, **flags):
    monkeypatch.setattr(compile_cache, "SOURCE_PACKAGE", src)
    return compile_cache.key_fields({"model_size": "1b-tsumugi", "levers": 16, **flags})


def test_the_key_moves_with_the_code_and_the_engine_flags(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "model"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    monkeypatch.delenv("KOTODAMA_ATTNRES_EPS_FP64_COMPAT", raising=False)
    base = compile_cache.cache_key(_fields(monkeypatch, src))
    assert base == compile_cache.cache_key(_fields(monkeypatch, src))  # stable
    assert base == compile_cache.cache_key(_fields(monkeypatch, src, checkpoint="/other.pt"))  # not an engine flag
    assert base != compile_cache.cache_key(_fields(monkeypatch, src, levers=0))
    monkeypatch.setenv("KOTODAMA_ATTNRES_EPS_FP64_COMPAT", "1")
    with_env = compile_cache.cache_key(_fields(monkeypatch, src))
    assert with_env != base
    (src / "a.py").write_text("x = 2\n")
    assert compile_cache.cache_key(_fields(monkeypatch, src)) != with_env


def test_artifacts_round_trip_and_bad_blobs_compile_cold(monkeypatch, tmp_path: Path) -> None:
    import torch

    saved = {}
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: (b"blob-bytes", "info"))
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda b: saved.setdefault("loaded", b))
    fields = {"f": 1}
    assert compile_cache.load(tmp_path, "k1") is False  # nothing stored: cold
    path = compile_cache.save(tmp_path, "k1", fields)
    assert path is not None and path.read_bytes() == b"blob-bytes"
    assert (tmp_path / "k1.json").is_file() and not list(tmp_path.glob(".*.tmp"))
    assert compile_cache.load(tmp_path, "k1") is True and saved["loaded"] == b"blob-bytes"

    def broken(_b):
        raise RuntimeError("incompatible artifacts")

    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", broken)
    assert compile_cache.load(tmp_path, "k1") is False  # logged, then compiled cold
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    assert compile_cache.save(tmp_path, "k2", fields) is None


def test_compile_first_starts_the_rest_after_the_first_is_ready(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import httpx

    from kotodama.serve import gateway

    cfg = gateway.load_config(str(Path(__file__).resolve().parents[1] / "configs" / "gateway-tsumugi.example.yaml"),
                              [0, 1, 2])
    cfg.workdir, cfg.spawn_stagger_s = str(tmp_path), 0.0
    assert cfg.compile_first
    fleet = gateway.Fleet(cfg)
    events = []
    monkeypatch.setattr(fleet, "spawn", lambda r: events.append(("spawn", r.gpu)))

    async def ready(r):
        events.append(("ready", r.gpu))
        return True

    monkeypatch.setattr(gateway, "_cfg", cfg, raising=False)
    monkeypatch.setattr(gateway, "_fleet", fleet, raising=False)
    monkeypatch.setattr(gateway, "_wait_ready", ready)
    monkeypatch.setattr(gateway, "_http", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(503))), raising=False)  # nothing to adopt
    asyncio.run(gateway._adopt_or_spawn())
    assert events == [("spawn", 0), ("ready", 0), ("spawn", 1), ("spawn", 2)]
