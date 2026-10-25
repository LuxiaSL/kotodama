"""Gateway pieces that need no fleet: the example config loads and validates its routing targets, the GPU choice
respects a scheduler's CUDA_VISIBLE_DEVICES grant, and cache-affinity keys follow a conversation's stable head."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from kotodama.serve import gateway  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "gateway.example.yaml"


def test_example_config_loads() -> None:
    cfg = gateway.load_config(str(EXAMPLE), None)
    assert cfg.completion_pool in cfg.pool_keys()
    assert Path(cfg.workdir).is_absolute()
    assert gateway.load_config(str(EXAMPLE), [3]).gpus == [3]


def test_bad_routing_target_refused(tmp_path) -> None:
    bad = tmp_path / "g.yaml"
    bad.write_text(EXAMPLE.read_text().replace("completion_pool: base", "completion_pool: nope", 1))
    with pytest.raises(SystemExit, match="routing target"):
        gateway.load_config(str(bad), None)


def test_effective_gpus_intersects_the_grant(monkeypatch) -> None:
    cfg = gateway.load_config(str(EXAMPLE), [0, 1, 2])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gateway.effective_gpus(cfg) == [0, 1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,5")
    assert gateway.effective_gpus(cfg) == [1, 2]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
    assert gateway.effective_gpus(cfg) == [6, 7]  # nothing preferred is granted: use the grant


def test_affinity_key_is_the_conversation_head() -> None:
    sys_msg = {"role": "system", "content": "s"}
    turn1 = {"messages": [sys_msg, {"role": "user", "content": "hello"}]}
    turn2 = {"messages": [sys_msg, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"},
                          {"role": "user", "content": "more"}]}
    other = {"messages": [sys_msg, {"role": "user", "content": "different"}]}
    assert gateway._affinity_key(turn1) == gateway._affinity_key(turn2) != gateway._affinity_key(other)
    assert gateway._affinity_key({"user": "u1", "prompt": "x"}) == "user:u1"
    assert gateway._affinity_key({"prompt": "p" * 300}) == "p" * 256
    assert gateway._affinity_key({}) is None
