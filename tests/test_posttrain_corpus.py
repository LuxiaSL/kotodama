"""Self-play corpus tools: assembly, meters, the stratified precision sample, rating and scoring."""
from __future__ import annotations

import asyncio
import json
import random

from kotodama.posttrain import corpus
from kotodama.posttrain import selfplay as sp


def _run(tmp_path):
    script = [[(" fine, thanks.\n\nTeo:", "stop", False), (" As an AI language model, no.", "stop", False),
               (" Not bad, the kettle is on.\n\nTeo:", "stop", False)]]

    async def complete(prompt, n, stops):
        return script[0]
    cfg = sp.SelfPlayConfig(candidates=3, min_turns=3, max_turns=5)
    seeds = [sp.SeedSpec("s", ["Mara", "Teo"], "Two friends.")] * 4
    asyncio.run(sp.run(seeds, complete, cfg, tmp_path, concurrency=2, seed=0))
    return tmp_path


def test_assemble_dedupes_and_drops_headers(tmp_path) -> None:
    d = _run(tmp_path)
    rows = list(corpus._read(d / "transcripts.jsonl"))
    rows.append(dict(rows[0]))  # exact duplicate
    rows.append({"turns": [{"speaker": "A", "text": "short"}]})  # too few turns
    out = corpus.assemble(rows, min_turns=4, header_dropout=1.0, seed=0)
    assert 1 <= len(out) <= 4 and all("header" not in r for r in out)
    assert all(len(r["turns"]) == 5 for r in out)
    kept = corpus.assemble(rows, min_turns=4, header_dropout=0.0, seed=0)
    assert all(r["header"] == "Two friends." for r in kept)


def test_meters(tmp_path) -> None:
    d = _run(tmp_path)
    m = corpus.meters(list(corpus._read(d / "transcripts.jsonl")), list(corpus._read(d / "candidates.jsonl")))
    assert m["transcripts"] == 4 and m["finishes"] == {"max_turns": 4}
    # the scripted turns repeat themselves, so selfcopy joins meta_leak once a speaker has said them before
    assert m["veto_rate"] > 1 / 3 and {"meta_leak", "selfcopy"} <= set(m["veto_reasons"])
    assert 0 < m["survivors_per_draw_mean"] <= 2.0


def test_precision_sample_rate_score(tmp_path) -> None:
    d = _run(tmp_path)
    ts, cs = list(corpus._read(d / "transcripts.jsonl")), list(corpus._read(d / "candidates.jsonl"))
    items = corpus.precision_sample(ts, cs, per_stratum=5, seed=1)
    strata = {it["stratum"] for it in items}
    assert strata == {"survivor", "vetoed", "unfiltered"}
    first = next(it for it in items if it["turn"] > 0)
    assert first["context"].startswith("Two friends.\n\nMara:")
    sample = tmp_path / "sample.jsonl"
    sample.write_text("".join(json.dumps(x) + "\n" for x in items))
    answers = iter(["y" if it["stratum"] != "vetoed" else "n" for it in items])
    ratings = tmp_path / "r.jsonl"
    corpus.rate(sample, ratings, inp=lambda _: next(answers), out=lambda *_: None)
    got = {x["item"]: x["rating"] for x in corpus._read(ratings)}
    s = corpus.score(items, got)
    assert s["precision"]["survivor"] == 1.0 and s["precision"]["vetoed"] == 0.0 and s["lift"] == 0.0
    assert s["vetoed_but_good"] == 0.0
    # resumable: a second pass has nothing left to ask
    corpus.rate(sample, ratings, inp=lambda _: "q", out=lambda *_: None)


def test_rollout_and_within_turn_diversity(tmp_path) -> None:
    d = _run(tmp_path)
    m = corpus.meters(list(corpus._read(d / "transcripts.jsonl")), list(corpus._read(d / "candidates.jsonl")))
    assert 0.0 < m["rollout_diversity"] <= 1.0
    same = [{"tid": "t", "turn": 0, "draw": 0, "text": "the kettle is on and the lamp is lit", "vetoes": []}] * 3
    assert corpus.meters([], same)["rollout_diversity"] == 0.0  # identical siblings: no diversity


def test_fluency_items_and_scoring(tmp_path) -> None:
    import pytest
    httpx = pytest.importorskip("httpx")
    transformers = pytest.importorskip("transformers")
    from kotodama.presets import TOKENIZER_NAME

    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    row = {"tid": "x#0", "header": "Two friends.", "n_generated": 2,
           "turns": [{"speaker": "Mara", "text": "Hello."}, {"speaker": "Teo", "text": "Hi there."}]}
    items = corpus.fluency_items([row], tok)
    assert [it["id"] for it in items] == ["x#0:0", "x#0:1"]
    assert tok.decode(items[1]["prompt_ids"]).endswith("Mara: Hello.\n\nTeo:")
    assert tok.decode(items[1]["completion_ids"]) == " Hi there."

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"id": it["id"], "n_tokens": len(it["completion_ids"]), "sum_nll": 2.0 * len(it["completion_ids"]),
             "mean_nll": 2.0} for it in body["items"]]})

    res = corpus.score_fluency(items, "http://s:1", transport=httpx.MockTransport(handler))
    assert res["mean_nll"] == 2.0 and res["turns"] == 2


def test_compare_table(tmp_path) -> None:
    d = _run(tmp_path)
    (d / "run.json").write_text(json.dumps({"sampling": {"temperature": 0.75, "repetition_penalty": 1.2},
                                            "steer": [{"name": "vrep", "alpha": -0.05}]}))
    (d / "fluency.json").write_text(json.dumps({"mean_nll": 2.5}))
    rows = corpus.compare_rows([str(d)])
    assert rows[0]["steer"] == "vrep:-0.05" and rows[0]["nll"] == 2.5 and rows[0]["T"] == 0.75
    table = corpus.markdown_table(rows)
    assert table.startswith("| run | T | rep | steer |") and "vrep:-0.05" in table


def test_name_drift_rate() -> None:
    ts = [{"tid": "t", "cast": ["Ines", "Paul"], "turns": []}]
    cs = [{"tid": "t", "turn": 0, "draw": 0, "text": "a", "vetoes": [], "cut_label": "Inez"},
          {"tid": "t", "turn": 0, "draw": 0, "text": "b", "vetoes": [], "cut_label": "Paul"},
          {"tid": "t", "turn": 0, "draw": 0, "text": "c", "vetoes": [], "cut_label": None},
          {"tid": "t", "turn": 0, "draw": 0, "text": "d", "vetoes": [], "cut_label": "User"}]
    m = corpus.meters(ts, cs)
    assert m["stray_label_rate"] == 0.75 and m["name_drift_rate"] == 0.25
