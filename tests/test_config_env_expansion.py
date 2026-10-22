"""Environment-variable expansion in training YAML configs.

String values (and strings inside list values) go through ``os.path.expandvars``
after ``extends`` merging, so configs carry ``${KOTODAMA_DATA_ROOT}/...`` and
the machine path lives in the site profile. An unset variable is left as
written and warned about.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from kotodama.training.train import _load_yaml_config

REPO = Path(__file__).resolve().parents[1]


def _w(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_braced_and_bare_refs_expand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOTODAMA_DATA_ROOT", "/data/root")
    monkeypatch.setenv("RUN", "a0")
    cfg = _w(tmp_path / "c.yaml",
             "data_path: ${KOTODAMA_DATA_ROOT}/train.bin\n"
             "checkpoint_dir: $KOTODAMA_DATA_ROOT/ckpt/$RUN\n")
    assert _load_yaml_config(cfg) == {
        "data_path": "/data/root/train.bin",
        "checkpoint_dir": "/data/root/ckpt/a0",
    }


def test_parent_values_expand_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOTODAMA_DATA_ROOT", "/d")
    _w(tmp_path / "base.yaml", "data_path: ${KOTODAMA_DATA_ROOT}/p.bin\n")
    child = _w(tmp_path / "child.yaml", "extends: base.yaml\nseed: 1\n")
    assert _load_yaml_config(child) == {"data_path": "/d/p.bin", "seed": 1}


def test_list_strings_expand_and_non_strings_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("X", "v")
    cfg = _w(tmp_path / "c.yaml", "paths: [\"$X/a\", 3]\nn: 7\nflag: true\nf: 6e-4\n")
    assert _load_yaml_config(cfg) == {"paths": ["v/a", 3], "n": 7, "flag": True, "f": "6e-4"}


def test_unset_variable_is_left_and_warned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("KOTODAMA_UNSET_FOR_TEST", raising=False)
    cfg = _w(tmp_path / "c.yaml", "data_path: ${KOTODAMA_UNSET_FOR_TEST}/train.bin\n")
    with caplog.at_level(logging.WARNING, logger="kotodama.training.train"):
        out = _load_yaml_config(cfg)
    assert out == {"data_path": "${KOTODAMA_UNSET_FOR_TEST}/train.bin"}
    assert "KOTODAMA_UNSET_FOR_TEST" in caplog.text


def test_no_tracked_config_carries_an_absolute_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shipped configs name the data root through the variable, never a machine path."""
    monkeypatch.setenv("KOTODAMA_DATA_ROOT", "/site/data")
    for p in sorted((REPO / "configs").glob("*.yaml")):
        if p.name in {"gateway.example.yaml", "checkpoints.yaml"}:
            continue
        assert "/models/" not in p.read_text(), p.name
        for k, v in _load_yaml_config(p).items():
            if isinstance(v, str):
                assert "KOTODAMA_DATA_ROOT" not in v, (p.name, k, v)
