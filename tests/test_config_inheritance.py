"""Training-config inheritance (`extends:`) + the no-drift guard for configs/*.yaml.

tests/data/config_snapshot.json was captured from the configs BEFORE they were
refactored onto configs/base/*.yaml (the loader with no `extends` in play is
byte-for-byte the old loader). Every config must keep resolving to exactly its
snapshot — both the flat dict the loader returns and the effective argparse
namespace train.py ends up with. A diff here is a change to a run's config.

Regenerate (only when a config change is INTENDED):
    KOTODAMA_WRITE_CONFIG_SNAPSHOT=1 python -m pytest tests/test_config_inheritance.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from kotodama.training import train as train_mod
from kotodama.training.train import ConfigError, _load_yaml_config

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"
SNAPSHOT = Path(__file__).resolve().parent / "data" / "config_snapshot.json"
NOT_TRAINING = {"gateway.example.yaml", "checkpoints.yaml"}


def _training_configs() -> list[Path]:
    return sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name not in NOT_TRAINING)


def _effective(path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The namespace train.py actually runs with for `--config path` (no CLI overrides)."""
    rel = str(path.relative_to(REPO))
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["train", "--config", rel])
    ns = vars(train_mod.parse_args())
    return {k: v for k, v in ns.items() if k not in ("config", "_config_file")}


def _resolve_all(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in _training_configs():
        resolved = _load_yaml_config(p)
        effective = _effective(p, monkeypatch)
        out[p.name] = {
            "resolved": resolved,
            "dropped_unknown": sorted(set(resolved) - set(effective)),
            "effective": effective,
        }
    # JSON round trip so comparison is against exactly what is stored on disk
    return json.loads(json.dumps(out, sort_keys=True))


def test_configs_match_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _resolve_all(monkeypatch)
    if os.environ.get("KOTODAMA_WRITE_CONFIG_SNAPSHOT") == "1":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"snapshot written to {SNAPSHOT}")
    assert SNAPSHOT.is_file(), f"missing {SNAPSHOT}"
    expected = json.loads(SNAPSHOT.read_text())
    assert sorted(current) == sorted(expected), "set of training configs changed"
    for name in expected:
        for part in ("resolved", "effective", "dropped_unknown"):
            assert current[name][part] == expected[name][part], f"{name}: {part} drifted"


def test_extends_key_never_reaches_argparse() -> None:
    for p in _training_configs():
        assert "extends" not in _load_yaml_config(p), p.name


# --- extends semantics -----------------------------------------------------


def _w(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_child_overrides_parent(tmp_path: Path) -> None:
    _w(tmp_path / "base" / "a.yaml", "x: 1\ny: 2\nhyphen-key: 3\n")
    child = _w(tmp_path / "child.yaml", "extends: base/a.yaml\ny: 20\nz: 30\n")
    assert _load_yaml_config(child) == {"x": 1, "y": 20, "hyphen_key": 3, "z": 30}


def test_list_parents_later_wins(tmp_path: Path) -> None:
    _w(tmp_path / "a.yaml", "x: a\ny: a\n")
    _w(tmp_path / "b.yaml", "y: b\nz: b\n")
    child = _w(tmp_path / "c.yaml", "extends: [a.yaml, b.yaml]\nz: c\n")
    assert _load_yaml_config(child) == {"x": "a", "y": "b", "z": "c"}


def test_parent_resolved_relative_to_its_own_file(tmp_path: Path) -> None:
    # grandparent path is relative to the PARENT's directory, not the child's
    _w(tmp_path / "base" / "root.yaml", "x: root\n")
    _w(tmp_path / "base" / "mid.yaml", "extends: root.yaml\ny: mid\n")
    child = _w(tmp_path / "run" / "child.yaml", "extends: ../base/mid.yaml\n")
    assert _load_yaml_config(child) == {"x": "root", "y": "mid"}


def test_diamond_is_not_a_cycle(tmp_path: Path) -> None:
    _w(tmp_path / "root.yaml", "x: 0\n")
    _w(tmp_path / "l.yaml", "extends: root.yaml\nl: 1\n")
    _w(tmp_path / "r.yaml", "extends: root.yaml\nr: 1\n")
    child = _w(tmp_path / "c.yaml", "extends: [l.yaml, r.yaml]\n")
    assert _load_yaml_config(child) == {"x": 0, "l": 1, "r": 1}


def test_cycle_raises(tmp_path: Path) -> None:
    _w(tmp_path / "a.yaml", "extends: b.yaml\n")
    _w(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        _load_yaml_config(tmp_path / "a.yaml")


def test_self_cycle_raises(tmp_path: Path) -> None:
    _w(tmp_path / "a.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        _load_yaml_config(tmp_path / "a.yaml")


def test_missing_parent_raises(tmp_path: Path) -> None:
    child = _w(tmp_path / "c.yaml", "extends: nope.yaml\n")
    with pytest.raises(ConfigError, match="Parent config not found"):
        _load_yaml_config(child)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        _load_yaml_config(tmp_path / "absent.yaml")


@pytest.mark.parametrize("bad", ["extends: 3\n", "extends: [a.yaml, 4]\n", "extends: {a: 1}\n"])
def test_bad_extends_type_raises(tmp_path: Path, bad: str) -> None:
    child = _w(tmp_path / "c.yaml", bad)
    with pytest.raises(ConfigError, match="extends"):
        _load_yaml_config(child)


def test_non_mapping_raises(tmp_path: Path) -> None:
    child = _w(tmp_path / "c.yaml", "- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        _load_yaml_config(child)


def test_empty_file_is_empty_config(tmp_path: Path) -> None:
    assert _load_yaml_config(_w(tmp_path / "c.yaml", "")) == {}
