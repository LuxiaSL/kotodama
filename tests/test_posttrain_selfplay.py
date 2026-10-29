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
    # a whole dialogue written inside one turn hits the cap — but the turn itself ended at the next label
    inline = sp.evaluate(" Fine, the lantern.\nTeo: and the rope?\nMara: yes", "length", "Mara", prior, CAST, "", CFG)
    assert inline.ok and inline.text == "Fine, the lantern." and inline.ended_by == "stop" and inline.cut_label == "Teo"
    assert "early_end" in sp.evaluate(" bye", "eos", "Teo", prior[:1], CAST, "", CFG).vetoes
    assert sp.evaluate(" bye", "eos", "Mara", prior, CAST, "", CFG).ok  # 3 turns = min_turns: a fine ending
    assert "meta_leak" in sp.evaluate(" As an AI language model I can't.", "stop", "Mara", prior, CAST, "", CFG).vetoes
    assert "empty" in sp.evaluate(" Teo: hi", "stop", "Mara", prior, CAST, "", CFG).vetoes
    loop = " " + prior[0].text + " " + prior[0].text
    assert "selfcopy" in sp.evaluate(loop, "stop", "Mara", prior, CAST, "", CFG).vetoes
    echo = " Then we should walk down there before the tide comes in."
    assert "echo" in sp.evaluate(echo, "stop", "Mara", prior, CAST, "", CFG).vetoes
    # a short turn repeating a recent one, whoever said it (the cross-speaker ping-pong)
    sigh = prior + [Turn("Teo", "*Sigh*")]
    assert "repeat_turn" in sp.evaluate(" *sigh*", "stop", "Mara", sigh, CAST, "", CFG).vetoes
    assert sp.evaluate(" *Sigh* fine, let's go.", "stop", "Mara", sigh, CAST, "", CFG).ok
    far = sigh + [Turn("Mara", f"line {k} about the boat") for k in range(4)]
    assert sp.evaluate(" *Sigh*", "stop", "Teo", far, CAST, "", CFG).ok  # outside the window
    # reusing names and objects is binding, not copying
    assert sp.evaluate(" Teo, the lamp and the rocks again? Fine, the bay it is.", "stop", "Mara", prior, CAST,
                       "", CFG).ok


def test_next_speaker() -> None:
    assert sp.next_speaker(CAST, []) == "Mara"
    assert sp.next_speaker(["A", "B", "C"], [Turn("B", "x")]) == "C"


def _scripted(script):
    """A Complete returning scripted candidate lists in order; records prompts. Each call's turns are numbered
    (" hi." -> " hi. 3") so a script that repeats one line is not a repeat-turn loop."""
    calls = []

    async def complete(prompt, n, stops):
        calls.append(prompt)
        k = len(calls)
        return [((t.replace("\n\n", f" {k}\n\n", 1) if "\n\n" in t else t), fr, hit)
                for t, fr, hit in script[min(k - 1, len(script) - 1)]]
    return complete, calls


def test_play_picks_only_survivors_and_banks_everything() -> None:
    good = [(" fine, thanks.\n\nTeo:", "stop", False), (" Not bad at all.\n\nTeo:", "stop", False)]
    bad = [(" As an AI language model, no.", "stop", False)]
    complete, calls = _scripted([good + bad])
    banked = []
    seed = sp.SeedSpec("s1", CAST, "Two friends.", [])
    row = asyncio.run(sp.play(seed, complete, CFG, random.Random(0), banked.append))
    assert row["finish"] == "max_turns" and row["n_turns"] == CFG.max_turns
    assert all(t["text"].rsplit(" ", 1)[0] in ("fine, thanks.", "Not bad at all.") for t in row["turns"])
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



def test_oai_completer_sends_the_dialogue_law_min_p() -> None:
    import httpx

    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"text": " hi", "finish_reason": "stop"}
                                                     for _ in range(body["n"])]})

    from kotodama.serve.laws import law

    complete, close = sp.oai_completer(["http://a:1"], law("tsumugi-dialogue"), max_tokens=4,
                                       transport=httpx.MockTransport(handler))

    async def go():
        try:
            return await complete("Mara:", 2, STOPS)
        finally:
            await close()

    asyncio.run(go())
    assert seen[0]["min_p"] == 0.02 and seen[0]["temperature"] == 1.0

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


def test_completer_route_pins_and_fails_over() -> None:
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "b":
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"text": " a", "finish_reason": "stop"}]})

    from kotodama.serve.laws import law

    complete, close = sp.oai_completer(["http://a:1", "http://b:2", "http://c:3"], law("tsumugi"), 8,
                                       transport=httpx.MockTransport(handler))

    async def go():
        try:
            await complete("Mara:", 1, STOPS, route=2)
            await complete("Mara:", 1, STOPS, route=2)
            await complete("Mara:", 1, STOPS, route=1)  # b fails: retried on c
        finally:
            await close()

    asyncio.run(go())
    assert [u.split("/")[2] for u in seen] == ["c:3", "c:3", "b:2", "c:3"]


def test_completer_does_not_retry_a_refused_request() -> None:
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.host == "a":
            return httpx.Response(400, json={"detail": "Prompt too long: 5000 tokens (max 4096)"})
        return httpx.Response(422, json={"detail": "bad field"})

    from kotodama.serve.laws import law

    complete, close = sp.oai_completer(["http://a:1", "http://b:2"], law("tsumugi"), 8,
                                       transport=httpx.MockTransport(handler))

    async def go():
        try:
            with pytest.raises(sp.PromptTooLong):
                await complete("Mara:", 1, STOPS, route=0)
            with pytest.raises(httpx.HTTPStatusError):
                await complete("Mara:", 1, STOPS, route=1)
        finally:
            await close()

    asyncio.run(go())
    assert [u.split("/")[2] for u in seen] == ["a:1", "b:2"]  # one request each: a 4xx is never retried


def test_cycle_veto_catches_loops_of_any_period_and_narration_tails() -> None:
    from kotodama.posttrain.transcript import Turn

    loop = ["You hate me.", "No.", "Yes.", "You do.", "I don't know.", "You do, Richard. I hate you."]
    prior = [Turn("P" if k % 2 == 0 else "R", loop[k % 6]) for k in range(12)]
    assert sp.is_cycle(loop[0], prior, 30)  # period 6: the pair (last turn, candidate) was said before
    assert not sp.is_cycle("Then tell me why you stayed.", prior, 30)
    assert not sp.is_cycle(loop[0], prior, 4)  # outside the window
    bye = [Turn("A", "Goodbye. *She smiles and waves.*"), Turn("B", "Goodbye. (He nods.)")]
    assert sp.is_cycle("Goodbye. (He drives home.)", bye + [Turn("A", "Goodbye. *She makes coffee.*")], 30)
    assert not sp.is_cycle("Yes.", [Turn("A", "Are you coming?")], 30)  # an isolated short answer stays legal


def test_template_veto_catches_a_refilled_frame() -> None:
    from kotodama.posttrain.transcript import Turn

    prior = []
    for thing in ("color", "food", "city", "song"):
        prior += [Turn("A", f"What is your favorite {thing}?"), Turn("B", f"My favorite {thing} is a secret one.")]
    prior.append(Turn("A", "What is your favorite band?"))
    assert sp.is_template("My favorite band is the loud one.", prior, 30, 3)  # 4 earlier (q, a) pairs > 3
    assert not sp.is_template("My favorite band is the loud one.", prior, 30, 4)
    assert not sp.is_template("Probably the one my brother liked.", prior, 30, 3)
    assert not sp.is_template("My favorite.", prior, 30, 3)  # under 4 spoken words: left to the cycle veto


def test_novelty_and_vetoes_wire_into_evaluate() -> None:
    from kotodama.posttrain.transcript import Turn

    prior = [Turn("A", "The harbour was cold that night."), Turn("B", "Cold and empty, like the harbour always is.")]
    assert sp.novelty("The lighthouse keeper never came back.", prior) == 1.0
    assert sp.novelty("Cold harbour.", prior) == 0.0
    assert sp.novelty("Yes.", prior) == 0.0
    loop = [Turn("A" if k % 2 == 0 else "B", ["I'll go.", "You can't go."][k % 2]) for k in range(6)]
    cfg = sp.SelfPlayConfig(cycle_window=30)
    assert "cycle" in sp.evaluate("I'll go.\n\nB:", "stop", "A", loop[:-1] + [loop[-1]], ["A", "B"], "", cfg).vetoes \
        or "repeat_turn" in sp.evaluate("I'll go.\n\nB:", "stop", "A", loop, ["A", "B"], "", cfg).vetoes
    off = sp.SelfPlayConfig(repeat_window=0)
    assert "cycle" not in sp.evaluate("I'll go.", "stop", "A", loop, ["A", "B"], "", off).vetoes


def test_novelty_speech_only_ignores_narration() -> None:
    from kotodama.posttrain.transcript import Turn

    prior = [Turn("A", "The harbour was cold that night.")]
    prose = "Cold harbour.\n\nMeanwhile the lighthouse keeper rowed toward the distant orchard islands."
    assert sp.novelty(prose, prior) > 0.5  # the narration paragraph brings many new words
    assert sp.novelty(prose, prior, speech_only=True, cast=["A", "B"]) == 0.0  # the spoken part brings none
    assert sp.spoken_text("Hello.\n\nAnd another thing, you never asked me.", ["A", "B"]).count("\n\n") == 1


def test_novelty_count_and_own_opening_repeat() -> None:
    from kotodama.posttrain.transcript import Turn

    prior = [Turn("A", "The harbour was cold that night.")]
    assert sp.novelty("The lighthouse keeper never came back.", prior, count=True) == 5.0  # lighthouse keeper never came back
    assert sp.novelty("Cold harbour.", prior, count=True) == 0.0
    assert sp.novelty_counts("The cold lighthouse.", prior) == (1, 2)  # lighthouse new; cold seen; "the" a stopword
    assert sp.novelty_counts("Oh, the.", prior) == (0, 0)
    turns = [Turn("A", "Well, I think the boat is late."), Turn("B", "It always is on Sundays."),
             Turn("A", "Maybe the tide turned early."), Turn("B", "Or the captain overslept again.")]
    assert sp.own_opening_repeat("Well, I suppose we wait here then.", "A", turns)  # A's own earlier opening
    assert not sp.own_opening_repeat("Well, I suppose we wait here then.", "B", turns)  # B never opened so
    assert not sp.own_opening_repeat("Well, I suppose we wait here then.", "A", turns, back=1)  # only "maybe the"
    assert not sp.own_opening_repeat("Well, I.", "A", turns)  # under 4 spoken words


def test_match_speaker() -> None:
    cast = ["Ann", "Scott", "Bea"]
    assert sp.match_speaker(" Scott", cast) == "Scott"
    assert sp.match_speaker("scott:", cast) == "Scott"  # case-insensitive after an exact miss
    assert sp.match_speaker("Scott", cast, exclude="Scott") is None  # the last speaker is never chosen
    assert sp.match_speaker("(pause)", cast) is None


def test_speaker_tag_prompt_ends_at_the_turn_break() -> None:
    p = sp.speaker_tag_prompt("H", [Turn("Ann", "Hi."), Turn("Scott", "Hey.")])
    assert p == "H\n\nAnn: Hi.\n\nScott: Hey.\n\n"
    assert sp.speaker_tag_prompt("H", []) == "H\n\n"


def _tags(*texts, fail: bool = False):
    calls = []

    async def complete(prompt, n, stops, **kw):
        calls.append((prompt, n, stops, kw))
        if fail:
            raise RuntimeError("down")
        return [(t, "stop", False) for t in texts][:n]

    return complete, calls


def test_model_speaker_takes_the_first_valid_tag() -> None:
    cast, turns = ["Ann", "Scott", "Bea"], [Turn("Ann", "Hi.")]
    complete, calls = _tags("Ann", "narration", "Bea", "Scott")  # Ann just spoke: skipped; "narration" is no one
    assert asyncio.run(sp.model_speaker(cast, turns, "H", complete)) == ("Bea", True)
    assert calls[0][2] == [":", "\n"] and calls[0][3] == {"max_tokens": 8} and calls[0][1] == 4


def test_model_speaker_falls_back_to_rotation() -> None:
    cast, turns = ["Ann", "Scott", "Bea"], [Turn("Ann", "Hi.")]
    assert asyncio.run(sp.model_speaker(cast, turns, "H", _tags("Ann", "Zed")[0])) == ("Scott", False)
    assert asyncio.run(sp.model_speaker(cast, turns, "H", _tags(fail=True)[0])) == ("Scott", False)
    assert asyncio.run(sp.model_speaker(["Solo"], [], "H", _tags("Solo")[0])) == ("Solo", False)
