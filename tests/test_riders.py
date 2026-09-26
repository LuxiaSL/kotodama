"""Rider grid (P0.6 battery): golden generation vs the frozen instrument, aggregate
tables on a synthetic grid, and /logprobs payloads against a mocked server."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from kotodama import presets
from kotodama.eval.riders import aggregate as agg
from kotodama.eval.riders import grid, items, smoothing
from kotodama.eval.riders.__main__ import main as cli_main

FROZEN_DIR = Path.home() / "projects/kotodama-frozen/posttraining/taste/probes/p06"
BANKED_DIR = Path.home() / "projects/kotodama-frozen/_node1-snapshot-2026-09-26/kotodama/probes/p06"


def _frozen_gen_items() -> ModuleType:
    path = FROZEN_DIR / "gen_items.py"
    if not path.exists():
        pytest.skip(f"frozen instrument not present: {path}")
    spec = importlib.util.spec_from_file_location("frozen_p06_gen_items", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations via sys.modules
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def _names_file(tmp_path: Path) -> Path:
    p = tmp_path / "verified_names.json"
    p.write_text(json.dumps({"verified": list(items.BANKED_VERIFIED_NAMES)}))
    return p


# ── (a) golden: byte-identical generation ──────────────────────────────────────

@pytest.mark.parametrize("seed", [items.DEFAULT_SEED, 5])
def test_emit_byte_identical_to_frozen(tmp_path, seed, capsys):
    frozen = _frozen_gen_items()
    names = _names_file(tmp_path)
    frozen.emit(names, tmp_path / "frozen", 200, 200, seed)
    items.emit(names, tmp_path / "new", 200, 200, seed)
    for fname in ("items.jsonl", "shots.json", "emit_meta.json"):
        assert (tmp_path / "new" / fname).read_bytes() == (tmp_path / "frozen" / fname).read_bytes(), fname
    assert sum(1 for _ in (tmp_path / "new" / "items.jsonl").open()) == 3400


def test_banked_pool_default_matches_names_file(tmp_path, capsys):
    items.emit(None, tmp_path / "a", 20, 20, 1002)
    items.emit(_names_file(tmp_path), tmp_path / "b", 20, 20, 1002)
    for fname in ("items.jsonl", "shots.json"):
        assert (tmp_path / "a" / fname).read_bytes() == (tmp_path / "b" / fname).read_bytes()


def test_banked_pool_reproduces_banked_artifacts(tmp_path, capsys):
    """The embedded pool == the banked verified_names.json, and regenerates the banked shots."""
    names = BANKED_DIR / "verified_names.json"
    if not names.exists():
        pytest.skip(f"banked p06 artifacts not present: {BANKED_DIR}")
    assert tuple(json.loads(names.read_text())["verified"]) == items.BANKED_VERIFIED_NAMES
    items.emit(None, tmp_path, 200, 200, 1002)
    assert (tmp_path / "shots.json").read_bytes() == (BANKED_DIR / "items/shots.json").read_bytes()
    meta = json.loads((BANKED_DIR / "items/emit_meta.json").read_text())
    ours = json.loads((tmp_path / "emit_meta.json").read_text())
    assert {k: v for k, v in ours.items() if k != "names_file"} == \
        {k: v for k, v in meta.items() if k != "names_file"}


def test_prompts_byte_identical_to_frozen():
    frozen = _frozen_gen_items()
    pool = list(items.BANKED_VERIFIED_NAMES)
    its, shots = items.generate(pool, 3, 5, 1002)
    f_shots = [frozen.Item(**s.__dict__) for s in shots]
    for it in its:
        f_it = frozen.Item(**it.__dict__)
        for rd in items.RENDERERS:
            for ns in (0, 3):
                for fam in ("raw", "chatml", "hf-chat"):
                    assert items.build_prompt(it, rd, shots[:ns], fam) == \
                        frozen.build_prompt(f_it, rd, f_shots[:ns], fam)
        msgs = items.chat_messages(it, False)
        for open_final in (True, False):
            assert items.chatml_render(msgs, open_final) == frozen.chatml_render(msgs, open_final)


def test_grid_tokenizers_and_selftest(capsys):
    assert items.GRID_TOKENIZERS["smollm2"] == presets.TOKENIZER_NAME == "HuggingFaceTB/SmolLM2-135M"
    assert items.selftest() == 0


# ── (b) aggregate on a synthetic grid ──────────────────────────────────────────

def _row(model: str, rd: str, nshots: int, tier: str, rank: int, *, n: int = 2,
         m: int = 1, rc: bool = False, n_cand: int = 2, margin: float = 1.0,
         item_id: str = "x") -> dict[str, Any]:
    return {"model": model, "item_id": item_id, "renderer": rd, "nshots": nshots,
            "tier": tier, "n": n, "m": m, "recency_control": rc, "gold": "cup",
            "n_candidates": n_cand, "gold_lp": -0.1, "margin": margin,
            "gold_rank_cand": rank, "gold_rank_full": rank, "entropy": 1.0,
            "prompt_tokens": 10, "cand_lps": {}, "top": [],
            "multi_token_candidates": [], "chat_format": None}


def _synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for k in range(20):  # code: d0 passes (100%), std perfect
        rows.append(_row("koto", "code", 3, "d0", 1, n=1, m=0, n_cand=8))
        rows.append(_row("koto", "code", 0, "d0", 1, n=1, m=0, n_cand=8))
        rows.append(_row("koto", "code", 3, "std", 1, rc=k < 4, margin=2.0))
        rows.append(_row("koto", "code", 0, "std", 1 + k % 2, margin=0.0))
    for k in range(10):  # narrative: d0 fails (50%) -> HARNESS-BUG, std at chance
        rows.append(_row("koto", "narrative", 3, "d0", 1 + k % 2, n=1, m=0, n_cand=8))
        rows.append(_row("koto", "narrative", 3, "std", 1 + k % 2, n=3, m=0))
    return rows


def test_aggregate_tables_and_summary():
    md, summary = agg.aggregate(_synthetic_rows())
    assert summary["d0_gate"] == {
        "koto|code": {"rate_3shot": 1.0, "rate_0shot": 1.0, "pass": True},
        "koto|narrative": {"rate_3shot": 0.5, "rate_0shot": summary["d0_gate"]["koto|narrative"]["rate_0shot"],
                           "pass": False},
    }
    cells = summary["cells"]
    assert set(cells) == {"koto|code|3shot", "koto|code|0shot", "koto|narrative|3shot"}
    c3 = cells["koto|code|3shot"]
    assert c3["cc_acc"] == 1.0 and list(c3["ci"]) == [1.0, 1.0] and c3["n"] == 20
    assert not c3["gate_failed"]
    assert cells["koto|code|0shot"]["cc_acc"] == pytest.approx(0.0)
    assert cells["koto|narrative|3shot"]["cc_acc"] == pytest.approx(0.0)
    assert cells["koto|narrative|3shot"]["gate_failed"]

    lines = md.splitlines()
    assert lines[0] == "# P0.6 Stage-1 tables"
    gate_row = next(ln for ln in lines if ln.startswith("| koto | 100.0%"))
    assert gate_row == "| koto | 100.0% (0s 100.0%) | 50.0% (0s nan%) **HARNESS-BUG** | — | — |"
    assert "| koto | +1.00 [+1.00,+1.00] r1 m+2.00 | ~~" in md  # struck-through narrative
    # N×M grid: code N2 M1 (recency-control excluded), narrative N3 M0
    assert "| code | M1:+1.00 | — |" in md and "| narrative | — | M0:+0.00 |" in md
    # recency delta 0 (all correct), shot gap 1.00 - 0.00
    assert "| koto | code | +0.00 | +1.00 |" in md
    # table shape: 4 renderer columns everywhere
    assert "| model | code | narrative | screenplay | chat |" in md


def test_aggregate_write_and_hard_fail(tmp_path, capsys):
    rows = _synthetic_rows()
    (tmp_path / "koto.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    md_path, js_path = agg.write_tables(tmp_path)
    assert md_path.read_text().startswith("# P0.6 Stage-1 tables")
    assert json.loads(js_path.read_text())["d0_gate"]["koto|code"]["pass"] is True
    bad = dict(rows[0], multi_token_candidates=[" teabag"])
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(agg.MultiTokenCandidateError):
        agg.load(tmp_path)
    assert cli_main(["aggregate", "--grids", str(tmp_path)]) == 2


def test_smoothing_index(tmp_path, capsys):
    it = {"item_id": "i1", "tier": "std", "m": 1, "query_box": "F",
          "boxes0": {"F": "cup", "G": "pen"}, "gold": "pen",
          "ops": [{"kind": "swap", "i": "F", "j": "G", "obj": None}]}
    named = dict(it, item_id="i2", gold="map",
                 ops=[{"kind": "overwrite", "i": "F", "j": None, "obj": "map"}])
    (tmp_path / "items.jsonl").write_text(json.dumps(it) + "\n" + json.dumps(named) + "\n")
    assert smoothing.qualifying_items(tmp_path / "items.jsonl") == {"i1": "cup"}
    grids = tmp_path / "grids"
    grids.mkdir()
    row = _row("koto", "code", 3, "std", 1, item_id="i1")
    row.update(gold="pen", cand_lps={"cup": -2.0, "pen": -2.0})
    (grids / "koto.jsonl").write_text(json.dumps(row) + "\n")
    [si] = smoothing.smoothing_index(tmp_path / "items.jsonl", grids)
    assert (si.model, si.renderer, si.n) == ("koto", "code", 1) and si.si == pytest.approx(0.5)


# ── (c) grid runner: /logprobs payloads against a mocked server ────────────────

class _Resp:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


class FakeServer:
    """Answers /logprobs like kotodama.serve.server._score_logprobs: first candidate
    in each item scores highest unless it is told the gold."""

    def __init__(self, golds: dict[str, str]):
        self.golds = golds
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def post(self, url: str, json: dict[str, Any], timeout: float) -> _Resp:
        self.urls.append(url)
        self.bodies.append(json)
        results = []
        for it in json["items"]:
            gold = " " + self.golds[it["id"].split("|")[0]]
            cands = [{"text": c, "id": k, "n_tokens": 1,
                      "logprob": -0.1 if c == gold else -3.0, "rank_full": 1 if c == gold else k + 2}
                     for k, c in enumerate(it["candidates"])]
            results.append({"id": it["id"], "prompt_tokens": len(it.get("prompt", "")),
                            "entropy": 1.5, "candidates": cands, "top": []})
        return _Resp({"results": results, "model": "fake"})

    def get(self, url: str, timeout: float) -> _Resp:
        self.urls.append(url)
        return _Resp({"status": "ok"})


@pytest.fixture()
def grid_items() -> tuple[list[items.Item], list[items.Item]]:
    its, shots = items.generate(list(items.BANKED_VERIFIED_NAMES), 1, 1, 1002)
    return [its[0], its[2]], shots  # one d0 item, one std item (n2m1)


def test_run_grid_payloads(tmp_path, monkeypatch, grid_items, capsys):
    its, shots = grid_items
    fake = FakeServer({it.item_id: it.gold for it in its})
    monkeypatch.setattr(grid.requests, "post", fake.post)
    monkeypatch.setattr(grid.requests, "get", fake.get)
    monkeypatch.setenv("KOTODAMA_ENDPOINT", "http://riders.test:9999/")
    out = tmp_path / "grids" / "koto.jsonl"

    n = grid.run_grid("koto", "koto", its, shots, out, renderers=["code", "chat"], batch=3, top_n=7)
    assert n == 2 * 2 * 2  # items x renderers x shot conditions
    assert fake.urls[0] == "http://riders.test:9999/health"
    assert all(u == "http://riders.test:9999/logprobs" for u in fake.urls[1:])
    assert [len(b["items"]) for b in fake.bodies] == [3, 3, 2]

    sent = {p["id"]: p for b in fake.bodies for p in b["items"]}
    assert all(b["top_n"] == 7 for b in fake.bodies)
    assert set(sent) == {f"{it.item_id}|{rd}|{ns}" for it in its for rd in ("code", "chat") for ns in (0, 3)}
    for it in its:
        for rd in ("code", "chat"):
            for ns in (0, 3):
                p = sent[f"{it.item_id}|{rd}|{ns}"]
                assert set(p) == {"id", "prompt", "candidates"}
                assert p["candidates"] == [" " + c for c in it.candidates]
                fam = "chatml" if rd == "chat" else "raw"
                assert p["prompt"] == items.build_prompt(it, rd, shots[:ns], fam)["prompt"]
                if rd == "chat":
                    assert p["prompt"].startswith("<|im_start|>user\n")
                    assert p["prompt"].endswith(f"<|im_start|>assistant\nSo Box {it.query_box} now contains the")

    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(rows) == 8
    for r in rows:
        assert r["gold_rank_cand"] == 1 and r["margin"] == pytest.approx(2.9)
        assert r["multi_token_candidates"] == []
        assert r["chat_format"] == ("chatml" if r["renderer"] == "chat" else None)

    # resume: nothing re-scored
    assert grid.run_grid("koto", "koto", its, shots, out, renderers=["code", "chat"]) == 0
    assert len(out.read_text().splitlines()) == 8


def test_payload_kinds(grid_items):
    its, shots = grid_items
    it = its[1]
    p, fmt = grid.build_payload_item(it, "chat", 3, shots, "hf-chat")
    assert fmt == "hf-chat" and set(p) == {"id", "messages", "answer_prefix", "candidates"}
    assert p["answer_prefix"] == f"So Box {it.query_box} now contains the"
    p, fmt = grid.build_payload_item(it, "chat", 0, shots, "hf-base")
    assert fmt == "chatml-literal" and p["prompt"].startswith("<|im_start|>")
    p, fmt = grid.build_payload_item(it, "narrative", 3, shots, "hf-chat")
    assert fmt is None and p["prompt"].count("\n\n###\n\n") == 3
    with pytest.raises(ValueError):
        grid.build_payload_item(it, "chat", 0, shots, "nope")


def test_payload_validates_against_server_models(grid_items):
    pytest.importorskip("fastapi")
    try:
        from kotodama.serve.server import LogprobRequest
    except Exception as exc:  # pragma: no cover — server deps unavailable
        pytest.skip(f"server import unavailable: {exc}")
    its, shots = grid_items
    payload = [grid.build_payload_item(it, rd, 3, shots, "koto")[0]
               for it in its for rd in items.RENDERERS]
    req = LogprobRequest(items=payload, top_n=5)
    assert [x.id for x in req.items] == [p["id"] for p in payload]


def test_score_row_ties_and_errors(grid_items):
    it = grid_items[0][1]
    cands = [{"text": " " + c, "id": k, "n_tokens": 1, "logprob": -1.0, "rank_full": 1}
             for k, c in enumerate(it.candidates)]
    res = {"candidates": cands, "entropy": 0.5, "prompt_tokens": 9}
    row = grid.score_row(it, res, "code", 0, "m", None)
    assert row["gold_rank_cand"] == len(cands) and row["margin"] == 0.0  # ties count against gold
    with pytest.raises(grid.LogprobsError):
        grid.score_row(it, dict(res, candidates=[c for c in cands if c["text"] != " " + it.gold]),
                       "code", 0, "m", None)


def test_request_batch_retries_then_fails(monkeypatch):
    calls = []

    def boom(*a: Any, **k: Any) -> None:
        calls.append(1)
        raise ConnectionError("down")

    monkeypatch.setattr(grid.requests, "post", boom)
    monkeypatch.setattr(grid.time, "sleep", lambda s: None)
    with pytest.raises(grid.LogprobsError, match="after 4 attempts"):
        grid.request_batch("http://x", [{"id": "a"}], 5)
    assert len(calls) == 4


def test_default_endpoint(monkeypatch):
    monkeypatch.delenv("KOTODAMA_ENDPOINT", raising=False)
    assert grid.default_endpoint() == "http://localhost:2222"
    monkeypatch.setenv("KOTODAMA_ENDPOINT", "http://elsewhere:1")
    assert grid.default_endpoint() == "http://elsewhere:1"


def test_cli_gen(tmp_path, capsys):
    assert cli_main(["gen", "--out", str(tmp_path), "--per-cell", "4", "--d0-count", "2"]) == 0
    loaded = items.load_items(tmp_path / "items.jsonl")
    assert len(loaded) == 2 + 4 * len(items.CELLS)
    assert len(items.load_items(tmp_path / "shots.json")) == items.N_SHOTS
    rng_check = random.Random(0)  # items are plain data: round-trip is lossless
    it = rng_check.choice(loaded)
    assert items.Item(**json.loads(json.dumps(it.__dict__))) == it
