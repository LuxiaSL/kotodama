"""Self-play generation: completion splitting, stray-label cuts, the mechanical vetoes, uniform choice among
survivors, banking, finish reasons, and the /v1/completions request the completer sends."""
from __future__ import annotations

import asyncio
import json
import random

import pytest

from kotodama.posttrain import selfplay as sp
from kotodama.posttrain.transcript import Turn

CAST = ["Mara", "Teo"]
STOPS = ["\n\nMara:", "\n\nTeo:"]
CFG = sp.SelfPlayConfig(candidates=3, min_turns=3, max_turns=6, rerolls=1)


def test_split_completion() -> None:
    assert sp.split_completion(" hi there\n\nTeo: and", STOPS, "stop", False) == (" hi there", "stop")
    assert sp.split_completion(" the end", STOPS, "stop", False) == (" the end", "eos")
    assert sp.split_completion(" and on and on", STOPS, "length", True) == (" and on and on", "length")


def test_cut_stray_label() -> None:
    assert sp.cut_stray_label(" fine.\nUser: what now", CAST) == (" fine.", "User")
    assert sp.cut_stray_label(" fine.\n  Teo: ok", CAST) == (" fine.", "Teo")
    assert sp.cut_stray_label(" Mara: hello", CAST) == ("", "Mara")
    assert sp.cut_stray_label(" Note: the kettle is on", CAST) == (" Note: the kettle is on", None)
    # a drifted participant name opening a paragraph is still another speaker's turn
    assert sp.cut_stray_label(" I am.\n\nIne: How are you today?", CAST) == (" I am.", "Ine")
    assert sp.cut_stray_label(" I was leaving.\n\nInez: What's wrong?", CAST) == (" I was leaving.", "Inez")
    assert sp.cut_stray_label(" Two things.\nFirst: the lamp. Second: the boat.", CAST)[1] is None  # not a paragraph


def test_vetoes() -> None:
    prior = [Turn("Mara", "The lamp went out over the black rocks of the bay last night, and nobody went to fix it."),
             Turn("Teo", "Then we should walk down there before the tide comes in.")]
    ok = sp.evaluate(" Yes, bring the lantern.", "stop", "Mara", prior, CAST, "", CFG)
    assert ok.ok
    assert "overlong" in sp.evaluate(" words", "length", "Mara", prior, CAST, "", CFG).vetoes
    assert "early_end" in sp.evaluate(" bye", "eos", "Teo", prior[:1], CAST, "", CFG).vetoes
    assert sp.evaluate(" bye", "eos", "Mara", prior, CAST, "", CFG).ok  # 3 turns = min_turns: a fine ending
    assert "meta_leak" in sp.evaluate(" As an AI language model I can't.", "stop", "Mara", prior, CAST, "", CFG).vetoes
    assert "empty" in sp.evaluate(" Teo: hi", "stop", "Mara", prior, CAST, "", CFG).vetoes
    loop = " " + prior[0].text + " " + prior[0].text
    assert "selfcopy" in sp.evaluate(loop, "stop", "Mara", prior, CAST, "", CFG).vetoes
    echo = " Then we should walk down there before the tide comes in."
    assert "echo" in sp.evaluate(echo, "stop", "Mara", prior, CAST, "", CFG).vetoes
    # reusing names and objects is binding, not copying
    assert sp.evaluate(" Teo, the lamp and the rocks again? Fine, the bay it is.", "stop", "Mara", prior, CAST,
                       "", CFG).ok


def test_next_speaker() -> None:
    assert sp.next_speaker(CAST, []) == "Mara"
    assert sp.next_speaker(["A", "B", "C"], [Turn("B", "x")]) == "C"


def _scripted(script):
    """A Complete returning scripted candidate lists in order; records prompts."""
    calls = []

    async def complete(prompt, n, stops):
        calls.append(prompt)
        return script[min(len(calls) - 1, len(script) - 1)]
    return complete, calls


def test_play_picks_only_survivors_and_banks_everything() -> None:
    good = [(" fine, thanks.\n\nTeo:", "stop", False), (" Not bad at all.\n\nTeo:", "stop", False)]
    bad = [(" As an AI language model, no.", "stop", False)]
    complete, calls = _scripted([good + bad])
    banked = []
    seed = sp.SeedSpec("s1", CAST, "Two friends.", [])
    row = asyncio.run(sp.play(seed, complete, CFG, random.Random(0), banked.append))
    assert row["finish"] == "max_turns" and row["n_turns"] == CFG.max_turns
    assert all(t["text"] in ("fine, thanks.", "Not bad at all.") for t in row["turns"])
    assert len(banked) == 3 * CFG.max_turns and sum(b["picked"] for b in banked) == CFG.max_turns
    assert all(not b["picked"] for b in banked if b["vetoes"])
    assert calls[0].endswith("Two friends.\n\nMara:") and calls[1].endswith("\n\nTeo:")


def test_play_ends_on_eos_and_on_no_survivor() -> None:
    eos = [(" goodbye then.", "stop", False)] * 3
    complete, _ = _scripted([[(" hi.\n\nTeo:", "stop", False)]] * 3 + [eos])
    row = asyncio.run(sp.play(sp.SeedSpec("e", CAST), complete, CFG, random.Random(1), lambda r: None))
    assert row["finish"] == "eos" and row["n_turns"] == 4
    dead = [(" As an AI language model.", "stop", False)] * 3
    complete, calls = _scripted([dead])
    row = asyncio.run(sp.play(sp.SeedSpec("d", CAST), complete, CFG, random.Random(2), lambda r: None))
    assert row["finish"] == "no_survivor" and row["n_turns"] == 0 and len(calls) == CFG.rerolls + 1


def test_run_writes_files(tmp_path) -> None:
    complete, _ = _scripted([[(" ok.\n\nTeo:", "stop", False)]])
    seeds = [sp.SeedSpec(str(i), CAST) for i in range(3)]
    summary = asyncio.run(sp.run(seeds, complete, CFG, tmp_path, concurrency=2, seed=0))
    assert summary["transcripts"] == 3
    rows = [json.loads(x) for x in (tmp_path / "transcripts.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(len(r["turns"]) == CFG.max_turns for r in rows)
    assert len((tmp_path / "candidates.jsonl").read_text().splitlines()) == 3 * CFG.max_turns


def test_oai_completer_request_shape() -> None:
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        return httpx.Response(200, json={"choices": [
            {"text": " hey\n\nTeo:", "finish_reason": "stop", "token_ids": {"completion": [1, 2, 3]}}
            for _ in range(body["n"])]})

    from kotodama.serve.laws import law

    complete, close = sp.oai_completer(["http://a:1", "http://b:2/"], law("tsumugi"), max_tokens=3,
                                       transport=httpx.MockTransport(handler))

    async def go():
        try:
            return await complete("Mara:", 10, STOPS)
        finally:
            await close()

    out = asyncio.run(go())
    assert len(out) == 10 and out[0] == (" hey\n\nTeo:", "stop", True)  # 3 ids == max_tokens
    assert [s[1]["n"] for s in seen] == [8, 2]  # the server caps n at 8
    assert seen[0][0] == "http://a:1/v1/completions" and seen[1][0] == "http://b:2/v1/completions"
    b = seen[0][1]
    assert b["stop"] == STOPS and b["top_p"] == 0.0 and b["frequency_penalty"] == pytest.approx(0.2)
    assert b["temperature"] == law("tsumugi")["temperature"] and b["return_token_ids"] is True


def test_parse_seed() -> None:
    s = sp.parse_seed({"cast": ["A", "B"], "header": " h ", "turns": [{"speaker": "A", "text": "hi"}]}, 7)
    assert s.id == "7" and s.header == "h" and s.turns == [Turn("A", "hi")]
    assert sp.parse_seed({"cast": ["A"]}, 0) is None


def test_parse_steer() -> None:
    assert sp.parse_steer("vrep:-0.05, V7_s58:0.03") == [{"name": "vrep", "alpha": -0.05},
                                                          {"name": "V7_s58", "alpha": 0.03}]
    assert sp.parse_steer("vrep") == [{"name": "vrep"}] and sp.parse_steer("") == []
    with pytest.raises(ValueError):
        sp.parse_steer("vrep:lots")


def test_completer_sends_steering() -> None:
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": " a", "finish_reason": "stop"}]})

    from kotodama.serve.laws import law

    complete, close = sp.oai_completer(["http://a:1"], law("tsumugi"), 8, transport=httpx.MockTransport(handler),
                                       steer=[{"name": "vrep", "alpha": -0.05}])

    async def go():
        try:
            await complete("Mara:", 1, STOPS)
        finally:
            await close()

    asyncio.run(go())
    assert seen[0]["vectors"] == [{"name": "vrep", "alpha": -0.05}]
