"""Tests for scripts/eval/hm_backfill.py (telemetry backfill glue)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "hm_backfill",
    Path(__file__).parent.parent / "scripts" / "eval" / "hm_backfill.py")
hm = importlib.util.module_from_spec(_spec)
sys.modules["hm_backfill"] = hm
_spec.loader.exec_module(hm)


SUMMARY = {
    "chance_acc": 0.1,
    "rows": [
        {"step": 0, "tokens": 0, "is_init": True,
         "ledger_unique_acc": 0.075, "ledger_unique_medrank": 6,
         "ledger_unique_n": 400, "si": 0.515},
        {"step": 340, "tokens": 50282496, "is_init": False,
         "ledger_unique_acc": 0.055, "ledger_overwrite_acc": 0.13,
         "si": 0.613},
    ],
}


def test_gates_rows_to_points():
    pts = hm.gates_rows_to_points(SUMMARY, prefix="gates")
    assert [p["step"] for p in pts] == [0, 340]
    m0, m1 = pts[0]["metrics"], pts[1]["metrics"]
    assert m0["gates/ledger_unique_acc"] == pytest.approx(0.075)
    assert m0["tokens_seen"] == 0.0                 # tokens -> the counter
    assert m1["tokens_seen"] == pytest.approx(50282496.0)
    assert m1["gates/chance_acc"] == pytest.approx(0.1)
    assert "gates/is_init" not in m0                # bools excluded
    assert "gates/step" not in m0


def test_jsonl_and_merge(tmp_path):
    j = tmp_path / "extra.jsonl"
    j.write_text(json.dumps({"step": 340, "tokens": 50282496,
                             "metrics": {"induction/gain": 0.02,
                                         "note": "skipped-not-numeric"}})
                 + "\n\n"
                 + json.dumps({"step": 680, "metrics": {"induction/gain": 0.5}}))
    pts = hm.jsonl_to_points(j)
    assert pts[0]["metrics"] == {"induction/gain": 0.02,
                                 "tokens_seen": 50282496.0}
    merged = hm.merge_points(hm.gates_rows_to_points(SUMMARY), pts)
    assert [p["step"] for p in merged] == [0, 340, 680]
    at340 = merged[1]["metrics"]
    assert "gates/ledger_unique_acc" in at340 and "induction/gain" in at340


def test_parse_config_args():
    cfg = hm.parse_config_args(["arm=p05-armd-v3", "seed=42", "lr=0.02",
                                "flags={\"a\": 1}"])
    assert cfg["arm"] == "p05-armd-v3"
    assert cfg["seed"] == 42 and cfg["lr"] == 0.02
    assert cfg["flags"] == {"a": 1}
    with pytest.raises(ValueError):
        hm.parse_config_args(["oops"])


def test_post_all_batches_and_config():
    calls = []
    n = hm.post_all(lambda path, body: calls.append((path, body)),
                    "jobX",
                    [{"step": i, "metrics": {"a": float(i)}} for i in range(7)],
                    config={"arm": "test"}, chunk=3)
    assert n == 7
    assert calls[0] == ("/api/v1/jobs/jobX/config", {"config": {"arm": "test"}})
    batch_paths = [c[0] for c in calls[1:]]
    assert batch_paths == ["/api/v1/jobs/jobX/metrics/batch"] * 3
    assert [len(c[1]["points"]) for c in calls[1:]] == [3, 3, 1]
