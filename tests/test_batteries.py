"""Run the serving correctness batteries under pytest.

The batteries live in scripts/benchmark/check_*.py (runnable standalone, with
--device / checkpoint options); here they run in their CPU unit mode, and the
CUDA-only ones run when a GPU is present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"battery_{name}", BENCH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(mod, argv: list[str]) -> int:
    old = sys.argv
    sys.argv = [mod.__file__, *argv]
    try:
        return mod.main()
    finally:
        sys.argv = old


def test_steering_engine_cpu():
    mod = _load("check_steering_engine")
    assert _run_main(mod, ["--device", "cpu"]) == 0, mod._FAILURES


def test_prefix_cache_unit_cpu():
    mod = _load("check_prefix_cache")
    rc = _run_main(mod, ["--unit-only", "--device", "cpu"])
    assert rc == 0, getattr(mod, "_FAILURES", rc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA battery")
def test_sampler_truncated_cuda():
    assert _run_main(_load("check_sampler_truncated"), []) == 0
