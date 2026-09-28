"""Formation-ladder checkpoint cadence: token-count parsing, piecewise schedule,
resume-safe threshold firing, and the weights-only file format."""
from __future__ import annotations

import pytest
import torch

from kotodama.ckpt import load, model_state
from kotodama.training.checkpoint import (
    TokenThresholds,
    parse_save_schedule,
    parse_token_count,
    save_weights_only,
)

B = 10**9
M = 10**6


def test_parse_token_count():
    assert parse_token_count("250M") == 250 * M
    assert parse_token_count("1.5B") == 1_500_000_000
    assert parse_token_count("4096") == 4096
    for bad in ("", "abcB", "-1B", "0"):
        with pytest.raises(ValueError):
            parse_token_count(bad)


def test_ladder_schedule_matches_spec_cadence():
    t = parse_save_schedule("250M:4B,1B:10B,5B:end", 30 * B)
    assert t[:4] == [250 * M, 500 * M, 750 * M, 1 * B]
    assert [x for x in t if x <= 4 * B] == [i * 250 * M for i in range(1, 17)]
    assert [x for x in t if 4 * B < x <= 10 * B] == [i * B for i in range(5, 11)]
    assert [x for x in t if x > 10 * B] == [15 * B, 20 * B, 25 * B, 30 * B]
    assert len(t) == 26


def test_schedule_truncates_at_total_and_rejects_bad_segments():
    assert parse_save_schedule("1B:end", 2_500_000_000) == [B, 2 * B]
    assert parse_save_schedule("1B:10B", 3 * B) == [B, 2 * B, 3 * B]
    with pytest.raises(ValueError):
        parse_save_schedule("1B:4B,1B:2B", 30 * B)  # goes backwards
    with pytest.raises(ValueError):
        parse_save_schedule("1B", 30 * B)


def test_thresholds_fire_once_per_crossing_and_skip_on_resume():
    th = TokenThresholds([100, 200, 300])
    fired = [tok for tok in range(0, 450, 60) if th.due(tok)]
    # 120 crosses 100, 240 crosses 200, 300 exactly, never again
    assert fired == [120, 240, 300]
    assert not th.due(10_000)
    resumed = TokenThresholds([100, 200, 300])
    resumed.skip_through(200)  # resumed at 200 tokens: 100/200 already saved
    assert not resumed.due(250)
    assert resumed.due(310)


def test_one_step_crossing_two_thresholds_saves_once():
    th = TokenThresholds([100, 150])
    assert th.due(160)
    assert not th.due(170)


def test_weights_only_format_loads_with_ckpt(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = save_weights_only(tmp_path / "weights", 41, model, tokens_consumed=12345)
    assert path is not None and path.name == "step_00000041.pt"
    ck = load(path)
    assert ck["step"] == 41 and ck["tokens_consumed"] == 12345
    assert set(model_state(ck)) == {"weight", "bias"}
    assert "muon_optimizer" not in ck
