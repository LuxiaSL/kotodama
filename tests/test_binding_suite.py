"""The binding suite (kotodama.eval.binding): spec parsing, scorer parity with the stage-gauge contract, resume,
the bindprobes adapter, the capacity battery's generator and fit, readers, the registry and pins."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from kotodama.eval.binding import batteries as bt
from kotodama.eval.binding import capacity as cap
from kotodama.eval.binding import readers
from kotodama.eval.binding.__main__ import main as cli_main
from kotodama.eval.binding.models import LoadedModel, ModelSpec, SpecError
from kotodama.eval.binding.score import read_jsonl, score_bindprobe_items, score_prompt_rows


# ── specs ──────────────────────────────────────────────────────────────────────

def test_spec_parse():
    s = ModelSpec.parse("koto:/c/step.pt")
    assert (s.kind, s.path, s.size, s.bounds, s.attn_res) == ("koto", "/c/step.pt", "1b-tsumugi", None, "config")
    s = ModelSpec.parse("koto:/c/a.pt?size=3b&bounds=0,1,3,7,15,19,24&attn_res=on")
    assert (s.size, s.bounds, s.attn_res) == ("3b", "0,1,3,7,15,19,24", "on")
    assert ModelSpec.parse("hf:EleutherAI/pythia-1.4b") == ModelSpec(kind="hf", path="EleutherAI/pythia-1.4b")
    for bad in ("pythia", "vllm:/x", "koto:", "koto:/a?boundz=1", "koto:/a?attn_res=maybe", "hf:/x?size=3b"):
        with pytest.raises(SpecError):
            ModelSpec.parse(bad)


# ── scorer: a causal stub model through the real tokenizer ─────────────────────

@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")


_G = torch.Generator().manual_seed(0)
_E = torch.randn(49152, 16, generator=_G)
_U = torch.randn(16, 49152, generator=_G)


def _causal_logits(ids: torch.Tensor) -> torch.Tensor:
    """(B, T, V) logits that depend on the whole prefix (cumulative mean of embeddings) — so right padding is only
    harmless if the read position is right."""
    h = torch.cumsum(_E[ids], dim=1) / torch.arange(1, ids.shape[1] + 1).view(1, -1, 1)
    return h @ _U


def _model(tok: Any) -> LoadedModel:
    return LoadedModel(ModelSpec("hf", "stub"), _causal_logits, tok, torch.device("cpu"), None)


def _rows(n: int = 30) -> list[dict[str, Any]]:
    rng = random.Random(3)
    objs = ["cup", "lamp", "spoon", "key", "book"]
    out = []
    for i in range(n):
        c = rng.sample(objs, 3)
        words = " ".join(rng.choice(["the", "a", "red", "old", "cat", "sat"]) for _ in range(rng.randint(1, 12)))
        out.append({"id": f"r{i}", "prompt": f"{words} Now Ann holds the", "candidates": [" " + x for x in c],
                    "gold": c[0], "K": i % 3})
    return out


def _reference(tok: Any, row: dict[str, Any]) -> dict[str, float]:
    """The stage-gauge read: one unpadded forward, log_softmax at the last position, first candidate token."""
    ids = torch.tensor([tok.encode(row["prompt"])])
    lp = torch.log_softmax(_causal_logits(ids)[0, -1].float(), dim=-1)
    return {c.strip(): float(lp[tok.encode(c, add_special_tokens=False)[0]]) for c in row["candidates"]}


def test_prompt_rows_match_unpadded_reference_and_resume(tmp_path, tok):
    rows = _rows()
    out = tmp_path / "b" / "m.jsonl"
    assert score_prompt_rows(_model(tok), rows[:10], out, "m", max_batch=4) == 10
    with out.open("a") as f:
        f.write('{"id": "torn')                                     # a cancelled job's half line
    assert score_prompt_rows(_model(tok), rows, out, "m", max_batch=4) == 20
    got = read_jsonl(out)
    assert [r["id"] for r in got] == [r["id"] for r in rows]       # no duplicates, torn line dropped
    for r, g in zip(rows, got):
        ref = _reference(tok, r)
        assert set(g["cand_lps"]) == set(ref)
        assert all(abs(g["cand_lps"][k] - ref[k]) < 1e-4 for k in ref)
        assert "prompt" not in g and g["model"] == "m" and g["K"] == r["K"] and g["multi_token"] == []
        assert all(isinstance(v, int) and v >= 1 for v in g["rank_full"].values())
        assert g["prompt_tokens"] == len(tok.encode(r["prompt"]))


def test_prompt_rows_reject_bad_gold(tmp_path, tok):
    bad = _rows(2)
    bad[1]["gold"] = "zebra"
    with pytest.raises(ValueError, match="not a candidate"):
        score_prompt_rows(_model(tok), bad, tmp_path / "x.jsonl", "m")


def test_bindprobe_adapter_rows_match_grid_rows(tmp_path, tok):
    from kotodama.eval.bindprobes import grid, items, pools
    from kotodama.eval.riders.score import score_logprob_items

    its = items.generate(pools.load_pools(None), seed=11, per_cell={f: 1 for f in items.CORE_FAMILY_NAMES})[:12]
    out = tmp_path / "bp.jsonl"
    n = score_bindprobe_items(_model(tok), its, ("0shot", "3shot"), out, "stub", max_batch=8)
    assert n == len(its) * 2
    got = {(r["item_id"], r["condition"]): r for r in read_jsonl(out)}
    for it in its[:4]:
        for c in ("0shot", "3shot"):
            res = score_logprob_items(_causal_logits, tok, [grid.build_payload_item(it, c)], 5, "cpu")[0]
            ref = grid.score_row(it, res, c, "stub")
            row = got[(it.item_id, c)]
            assert row["gold_rank_cand"] == ref["gold_rank_cand"] and row["pick"] == ref["pick"]
            assert abs(row["gold_lp"] - ref["gold_lp"]) < 1e-4
    assert score_bindprobe_items(_model(tok), its, ("0shot", "3shot"), out, "stub") == 0   # resume: nothing left


# ── capacity ───────────────────────────────────────────────────────────────────

def _words(s: str) -> int:
    return len(s.split())


def test_capacity_generate_shape_and_determinism():
    rows = cap.generate(_words, k_grid=(1, 4, 30), d_grid=(0, 40), seeds=2, trials=5)
    assert len(rows) == 3 * 2 * 2 * 5 and len({r["id"] for r in rows}) == len(rows)
    assert rows == cap.generate(_words, k_grid=(1, 4, 30), d_grid=(0, 40), seeds=2, trials=5)
    for r in rows:
        assert r["prompt"].count("\n") >= r["K"] and r["prompt"].endswith(" is:")
        assert r["gold"] in {c.strip() for c in r["candidates"]} and len(r["candidates"]) == 25
        lines = r["prompt"].split("\n")[: r["K"]]
        assert lines[r["qpos"]].endswith(": " + r["gold"])
        assert (r["d_tokens"] >= 40) == (r["D"] == 40)
        if r["K"] <= 25:
            assert r["gold_repeats"] == 0           # distinct tasks while the pool lasts
    assert any(r["gold_repeats"] > 0 for r in rows if r["K"] == 30)


def _synthetic(k50: float, beta: float, ceiling: float = 0.95, n: int = 400) -> list[dict[str, Any]]:
    rng = random.Random(int(k50 * 100))
    chance = 1 / 25
    rows = []
    for k in cap.K_GRID:
        p = chance + (ceiling - chance) / (1 + math.exp(beta * (math.log(k) - math.log(k50))))
        hits = round(p * n)
        for i in range(n):
            gold = "laundry"
            pick = gold if i < hits else "dishes"
            lps = {w: (0.0 if w == pick else -5.0) for w in cap.TASKS}
            rows.append({"K": k, "D": 0, "gold": gold, "cand_lps": lps, "candidates": [" " + w for w in cap.TASKS],
                         "model": "syn"})
    return rows


def test_capacity_fit_recovers_k50_and_law():
    s = cap.summarize(_synthetic(6.0, 2.0))
    assert abs(math.log(s.fit.k50) - math.log(6.0)) < 0.1 and not s.fit.censored and s.usable
    assert cap.summarize(_synthetic(400.0, 2.0)).fit.censored
    # a midpoint fitted past the grid is censored even when r(64) has dipped under .5 (a flat, slow curve)
    ks = list(cap.K_GRID)
    r = [1 / (1 + (k / 100) ** 0.6) for k in ks]
    r[-1] = 0.48                                   # dips under .5 at K = 64 ...
    slow = cap.fit_logistic(ks, r)
    assert slow.k50 > ks[-1] and slow.censored     # ... but the fitted midpoint is past the grid: censored
    pts = [(n, 3.0 * n ** 0.8) for n in (0.1, 0.5, 1.0, 3.0, 8.0)]
    law = cap.fit_law(pts)
    assert abs(law.alpha - 0.8) < 1e-9 and abs(law.n_eff(3.0) - 1.0) < 1e-9
    with pytest.raises(ValueError):
        cap.fit_law(pts[:2])


def test_capacity_generator_reproduces_the_pinned_battery(tmp_path, tok):
    out = tmp_path / "capacity-v1.jsonl"
    rc = cli_main(["gen-capacity", "--out", str(out)])
    assert rc == 0 and bt.sha256(out) == bt.BATTERIES["capacity-v1"].sha256


# ── readers ────────────────────────────────────────────────────────────────────

def _lp(pick: str, cands: list[str]) -> dict[str, float]:
    return {c: (0.0 if c == pick else -3.0) for c in cands}


def test_joinlite_reader_swap_sign():
    cands = ["cup", "lamp", "key"]
    rows = []
    for s in range(40):
        rows.append({"scene": s, "P": 2, "qkind": "third", "arm": "orig", "gold": "cup", "cand_lps": _lp("cup", cands)})
        rows.append({"scene": s, "P": 2, "qkind": "self", "arm": "orig", "gold": "cup", "cand_lps": _lp("cup", cands)})
        rows.append({"scene": s, "P": 2, "qkind": "self", "arm": "swap", "gold": "lamp", "cand_lps": _lp("lamp", cands)})
        # 'you' answered with the speaker's object both ways: orig wrong, swap moves the same way as I → inverts
        rows.append({"scene": s, "P": 2, "qkind": "addressee", "arm": "orig", "gold": "lamp",
                     "cand_lps": _lp("cup", cands)})
        rows.append({"scene": s, "P": 2, "qkind": "addressee", "arm": "swap", "gold": "cup",
                     "cand_lps": _lp("lamp", cands)})
    r = readers.joinlite(rows)["P2"]
    assert r["lookup"] == 1.0 and r["I"] == 1.0 and r["you"] == 0.0 and r["you_to_speaker_obj"] == 1.0
    assert r["swap_self"]["reading"] == "binds" and r["swap_addressee"]["reading"] == "INVERTS"


def test_swapdyn_and_decomp_readers():
    cands = ["cup", "lamp", "key"]
    rows = [{"condition": "0shot", "qkind": "addressee", "base_id": f"b{i}", "arm": arm, "gold_orig": "cup",
             "cand_lps": _lp("cup" if arm == "orig" else "lamp", cands)} for i in range(30) for arm in ("orig", "swap")]
    r = readers.swapdyn(rows)["0shot/addressee"]
    assert r["reading"] == "binds" and r["argmax_orig"] == 1.0 and r["argmax_swap"] == 0.0
    drows = [{"variant": v, "condition": "0shot", "gold": "cup", "candidates": [" cup", " lamp", " key"],
              "cand_lps": _lp("cup" if v == "third" else "key", cands)} for v in ("third", "deictic") for _ in range(9)]
    d = readers.decomp(drows)
    assert d["third/0shot"]["acc"] == 1.0 and d["deictic/0shot"]["acc"] == 0.0
    assert abs(d["third/0shot"]["chance"] - 1 / 3) < 1e-9
    row = readers.scorecard_row({"decomp": d, "swapdyn": readers.swapdyn(rows)})
    assert row["decomp third"] == "1.000" and row["dyn-you 0shot"].endswith("bind") and row["static I"] == "—"


# ── registry, pins, CLI ────────────────────────────────────────────────────────

def test_resolve_groups_and_pins(tmp_path):
    names = [b.name for b in bt.resolve("gauges,capacity-v1,joinlite-v1")]
    assert names == ["joinlite-v1", "swapdyn-v1", "capacity-v1"]
    with pytest.raises(KeyError):
        bt.resolve("nope")
    b = bt.BATTERIES["joinlite-v1"]
    (tmp_path / b.file).write_text('{"id": "x"}\n')
    with pytest.raises(bt.PinError):
        bt.path_for(b, tmp_path)
    assert bt.path_for(b, tmp_path, check=False) == tmp_path / b.file
    with pytest.raises(FileNotFoundError):
        bt.path_for(bt.BATTERIES["swapdyn-v1"], tmp_path)


def test_report_cli(tmp_path, capsys):
    d = tmp_path / "scores" / "stage5-decomp-v1"
    d.mkdir(parents=True)
    rows = [{"variant": "third", "condition": "0shot", "gold": "cup", "candidates": [" cup", " lamp"],
             "cand_lps": {"cup": 0.0, "lamp": -1.0}, "model": "m1"}] * 4
    (d / "m1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cli_main(["report", "--scores", str(tmp_path / "scores"), "--json", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "| m1 |" in out and "1.000" in out
    assert json.loads((tmp_path / "r.json").read_text())["m1"]["decomp"]["third/0shot"]["acc"] == 1.0


def test_prerelease_shape_name_still_resolves():
    from kotodama import presets
    from kotodama.serve import server

    # the pre-release name stays loadable: checkpoints and published config.json files carry it
    assert presets.resolve("1b-moonshot-loop") == "1b-tsumugi"
    assert presets.shape("1b-moonshot-loop") == presets.shape("1b-tsumugi")
    assert presets.DEFAULT_BOUNDARIES["1b-moonshot-loop"] == presets.TSUMUGI_LOOP
    assert server.model_config_for("1b-moonshot-loop") is server.MODEL_CONFIGS["1b-tsumugi"]
    assert server.model_config_for("no-such-shape") is None
