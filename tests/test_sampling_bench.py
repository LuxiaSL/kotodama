"""sampling_bench metric functions: distinct-n, loop detection, first-candidate reads, pairing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "sampling_bench.py"
_spec = importlib.util.spec_from_file_location("sampling_bench", _path)
sb = importlib.util.module_from_spec(_spec)
sys.modules["sampling_bench"] = sb
_spec.loader.exec_module(sb)


def test_words_and_distinct():
    w = sb.words("The cat, the CAT.")
    assert w == ["the", "cat", ",", "the", "cat", "."]
    assert sb.distinct_n(w, 1) == pytest.approx(4 / 6)
    assert sb.distinct_n(["a", "b", "a", "b"], 2) == pytest.approx(2 / 3)
    assert sb.distinct_n(["a"], 2) is None


def test_loop_detection():
    looped = sb.words("I went home. " * 3)
    assert sb.has_loop(looped)
    assert not sb.has_loop(sb.words("I went home. I went home. Then I slept."))
    assert not sb.has_loop([])


def test_longest_repeated_ngram():
    assert sb.longest_repeated_ngram(["a", "b", "c"]) == 0
    assert sb.longest_repeated_ngram(["a", "b", "x", "a", "b"]) == 2
    assert sb.longest_repeated_ngram(sb.words("one two three four. one two three four.")) == 5


def test_first_candidate():
    cands = [" photo", " onion", " marble"]
    assert sb.first_candidate(" onion, not the photo", cands) == "onion"
    assert sb.first_candidate(" Photo.", cands) == "photo"
    assert sb.first_candidate(" photographs and onions", cands) is None   # whole words only
    assert sb.first_candidate("", cands) is None
    assert sb.first_candidate(" red cup", [" cup", " red cup"]) == "red cup"   # tie -> longer


def test_pair_key_and_gold():
    assert sb.pair_key("jl0000-self-orig", "orig") == "jl0000-self"
    assert sb.pair_key("jl0000-self-swap", "swap") == "jl0000-self"
    assert sb.pair_key("d-v2_000|0shot|swap", "swap") == "d-v2_000|0shot"
    assert sb.pair_key("jl0001-third-orig3", "orig3") is None
    assert sb.item_gold({"gold": "onion"}) == "onion"
    assert sb.item_gold({"arm": "orig", "gold_orig": "switch"}) == "switch"
    assert sb.item_gold({"arm": "swap", "gold_orig": "switch"}) is None


def test_summarize_bind_shift():
    items = {
        "a-orig": {"id": "a-orig", "arm": "orig", "gold": "cup", "candidates": [" cup", " pen"]},
        "a-swap": {"id": "a-swap", "arm": "swap", "gold": "pen", "candidates": [" cup", " pen"]},
    }
    rows = [{"kind": "bind", "item": "a-orig", "sample": 0, "file": "f", "read": "cup"},
            {"kind": "bind", "item": "a-swap", "sample": 0, "file": "f", "read": "pen"}]
    s = sb.summarize_bind(rows, items)
    assert s["acc"] == 1.0 and s["shift"] == 1.0 and s["none"] == 0.0
    rows[1]["read"] = "cup"   # speaker ignored
    assert sb.summarize_bind(rows, items)["shift"] == 0.0


def test_grid_core_first():
    g = sb.build_grid([0.7, 0.8, 0.9, 1.0, 1.1], [0.0, 0.95], [1.0, 1.05, 1.1, 1.2, 1.35])
    assert len(g) == 50
    first_edge = next(i for i, s in enumerate(g) if s.temperature in (0.7, 1.1) or s.repetition_penalty in (1.0, 1.35))
    assert all(s.temperature not in (0.7, 1.1) and s.repetition_penalty not in (1.0, 1.35) for s in g[:first_edge])
    assert g[0].tag == "t0.8_poff_r1.05"
