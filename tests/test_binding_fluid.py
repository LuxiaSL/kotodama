"""The fluid in-context batteries (kotodama.eval.binding.fluid): determinism and pins, row contract, ladder cells,
per-family semantics re-derived from the prompt text, the shortcut audits, the midpoint reader, the F3 stale rate,
the fluid / crystallized scores and the token-pool checker. CPU only; no network, no tokenizer."""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any

import pytest

from kotodama.eval.binding import batteries as bt
from kotodama.eval.binding import fluid as fl
from kotodama.eval.binding import readers
from kotodama.eval.binding.__main__ import main as cli_main

SMALL = {"seeds": 2, "trials": 6}


@pytest.fixture(scope="module")
def full() -> dict[str, list[dict[str, Any]]]:
    return {b: fl.generate(b) for b in fl.BY_BATTERY}


def _bytes(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


# ── determinism, pins, registry ───────────────────────────────────────────────────────────────────────────────────

def test_same_seed_same_bytes():
    for b in fl.BY_BATTERY:
        assert _bytes(fl.generate(b, **SMALL)) == _bytes(fl.generate(b, **SMALL))


def test_gen_fluid_reproduces_the_pinned_batteries(tmp_path, capsys):
    assert cli_main(["gen-fluid", "--out", str(tmp_path)]) == 0
    for b in fl.BY_BATTERY:
        assert bt.sha256(tmp_path / bt.BATTERIES[b].file) == bt.BATTERIES[b].sha256
    assert "AUDIT" not in capsys.readouterr().out


def test_registry_group_and_readers():
    names = [b.name for b in bt.resolve("fluid")]
    assert names == ["fluid-v1-morph", "fluid-v1-algebra", "fluid-v1-rebind", "fluid-v1-seqind", "cryst-v1"]
    for b in bt.resolve("fluid"):
        assert b.kind == "rows" and b.reader == fl.BY_BATTERY[b.name].reader and b.reader in readers.READERS
        assert re.fullmatch(r"[0-9a-f]{64}", b.sha256)
    assert set(names) <= set(bt.GROUPS["all"])


# ── row contract and ladder cells ─────────────────────────────────────────────────────────────────────────────────

def test_rows_contract(full):
    for b, rows in full.items():
        fam = fl.BY_BATTERY[b]
        assert len({r["id"] for r in rows}) == len(rows)
        pool = {"F1": fl.SYLLABLES, "F2": fl.NONCE, "F3": fl.VALUES, "F4": fl.NONCE}.get(fam.key)
        for r in rows:
            cands = [c.strip() for c in r["candidates"]]
            assert len(cands) == fam.n_candidates == len(set(cands))
            assert all(c.startswith(" ") and " " not in c[1:] for c in r["candidates"])
            assert r["gold"] in cands and cands.index(r["gold"]) == r["gold_index"]
            assert set(r["roles"]) == set(cands) and r["roles"][r["gold"]] == "gold"
            if pool is not None:
                assert set(cands) <= set(pool)
            else:
                assert set(cands) <= set(fl.POOLS[f"c0-{r['relation']}"])
            assert r["family"] == fam.key and 1 <= r["rung"] <= len(fam.ladder)
            assert not r["prompt"].endswith((" ", "\n"))     # the candidate's leading space is the next token's
            for k in ("seed", "trial", "cell", "recent_answer", "gold_is_recent", "freq_follower",
                      "novelty_follower", "recency_subset", "neutral_recency", "neutral_frequency"):
                assert k in r


def test_every_ladder_cell_populated(full):
    for b, rows in full.items():
        fam = fl.BY_BATTERY[b]
        per_rung = {i + 1: 0 for i in range(len(fam.ladder))}
        cells: dict[str, int] = {}
        for r in rows:
            per_rung[r["rung"]] += 1
            cells[r["cell"]] = cells.get(r["cell"], 0) + 1
        assert all(v > 0 for v in per_rung.values()), per_rung
        assert len(set(cells.values())) == 1 and next(iter(cells.values())) == fl.SEEDS * fam.trials
    assert {len(v) for v in full.values()} == {1080, 1200, 1024}


def test_size_is_modest(full):
    assert sum(len(v) for v in full.values()) == 5464


# ── per-family semantics, re-derived from the prompt text ─────────────────────────────────────────────────────────

def _body(r: dict[str, Any]) -> tuple[list[str], str]:
    lines = r["prompt"].split("\n")
    return lines[1:-1], lines[-1]


def test_f1_rules_hold_in_the_text(full):
    for r in full["fluid-v1-morph"]:
        ex, q = _body(r)
        src, out = (x.split() for x in q.split(" → "))
        rule = r["rule"]
        if rule == "suffix":
            assert out == src and all(e.split(" → ")[1].split()[-1] == r["gold"] for e in ex)
        elif rule == "infix":
            assert out == src[:1] and all(e.split(" → ")[1].split()[1] == r["gold"] for e in ex)
        elif rule == "vowel":
            vstar = r["gold"][1]
            assert out == src[:1] and r["gold"] == src[1][0] + vstar and src[1] != r["gold"]
            for e in ex:
                a, b = (x.split() for x in e.split(" → "))
                assert b == [a[0], a[1][0] + vstar] and a[1][1] != vstar
            assert r["gold_count"] == 0 and not r["gold_is_recent"]
        else:
            assert out == src and len(src) == r["stem_syllables"]
            same = {e.split(" → ")[1].split()[-1] for e in ex if len(e.split(" → ")[0].split()) == len(src)}
            assert same == {r["gold"]}
        assert len(ex) == r["n_examples"]
        assert r["recency_subset"] == (rule in ("suffix", "infix"))


def _f2_parse(r: dict[str, Any]) -> None:
    ex, q = _body(r)
    rules = r["op_rules"]
    partner = r["partner"]
    for e in ex:
        a, op, b, eq, res = e.split()
        assert eq == "=" and res == fl.f2_apply(rules[op], a, b, partner)       # type: ignore[arg-type]
    toks = q.replace("(", " ").replace(")", " ").split()
    assert toks[-1] == "="
    expr = toks[:-1]
    v = expr[0]
    for i in range(1, len(expr), 2):
        op, right = expr[i], expr[i + 1]
        if rules[op] == "partner":                    # the fact the query needs is in the examples
            assert any(e.split()[0] == v and e.split()[1] == op for e in ex)
        v = fl.f2_apply(rules[op], v, right, partner)  # type: ignore[arg-type]
    assert v == r["gold"] and len(expr) == 2 * r["depth"] + 1
    assert len(set(rules)) == r["k_ops"]
    assert all(" ".join(expr[:3]) != " ".join(e.split()[:3]) for e in ex)    # the first step is new


def test_f2_equations_evaluate_to_gold(full):
    for r in full["fluid-v1-algebra"]:
        _f2_parse(r)
    assert {r["path_steps"] for r in full["fluid-v1-algebra"] if r["depth"] == 3} >= {0, 1, 2, 3}


def _f3_replay(r: dict[str, Any]) -> tuple[str, str, str]:
    if r["register"] == "code":
        *lines, q = r["prompt"].split("\n")
        events = [tuple(x.split(" = ")) for x in lines]
        target = q.removeprefix("assert ").removesuffix(" ==")
    else:
        body, q = r["prompt"].split(" In the end, the ")
        events = []
        seen: set[str] = set()
        for sent in re.findall(r"[^.]+\.", body):
            if m := re.fullmatch(r"\s*The (\w+) holds the (\w+)\.", sent):
                h, v = m.groups()
                assert h not in seen                   # a holder is introduced once, then only emptied + refilled
            else:
                m = re.fullmatch(r"\s*Then the (\w+) is emptied and the (\w+) is put in it\.", sent)
                assert m, sent
                h, v = m.groups()
                assert h in seen
            seen.add(h)
            events.append((h, v))
        target = q.removesuffix(" holds the")
    hist = [v for h, v in events if h == target]
    return hist[-1], hist[-2], events[-1][1]


def test_f3_replay_gives_gold_stale_and_recency(full):
    for r in full["fluid-v1-rebind"]:
        gold, stale, last = _f3_replay(r)
        assert (gold, stale, last) == (r["gold"], r["stale"], r["recent_answer"])
        assert r["roles"][stale] == "stale" and r["gold_count"] == 1
        assert all(fl.count_word(r["prompt"], c.strip()) == 1 for c in r["candidates"])   # no frequency cue
        target_lines = (sum(1 for x in r["prompt"].split("\n") if x.startswith(f"{r['holder']} = "))
                        if r["register"] == "code" else
                        len(re.findall(rf"The {r['holder']} holds the \w+\.|the {r['holder']} is emptied and",
                                       r["prompt"])))
        assert target_lines == r["N"] + 1
        assert r["n_lines"] == r["N"] + 1 + (fl.F3_HOLDERS - 1) + r["R"]


def test_f4_pattern_continues_to_gold(full):
    for r in full["fluid-v1-seqind"]:
        ex, q = _body(r)
        m, g, j = r["m"], r["gap"], r["copy_pos"]
        order = fl.f4_order(r["pattern"], m)
        for e in ex:
            t = e.split()
            assert len(t) == 2 * m + g and t[m + g:] == [t[x] for x in order]
        qt = q.split()
        assert len(qt) == m + g + j and qt[order[j]] == r["gold"]
        assert r["lag"] == m + g + j - order[j]
        words = [w for line in r["prompt"].split("\n")[1:] for w in line.split()]
        assert len(words) == len(set(words)) + m * len(ex) + j       # the copies are the only repeats


def test_c0_facts_and_frames(full):
    for r in full["cryst-v1"]:
        header, facts = fl.RELATIONS[r["relation"]]
        lines = r["prompt"].split("\n")
        assert lines[0] == header and lines[-1] == f"{r['subject']} →" and len(lines) == r["shots"] + 2
        answer = {s: a for s, a, _ in facts}
        assert answer[r["subject"]] == r["gold"]
        assert all(fl.count_word(r["prompt"], c.strip()) == 0 for c in r["candidates"])


# ── shortcut audits ───────────────────────────────────────────────────────────────────────────────────────────────

def test_audits_pass_on_the_generated_batteries(full):
    for b, rows in full.items():
        aud = fl.audit(rows)
        assert fl.audit_failures(aud) == [], b
        assert all(a["position_max_dev"] <= max(fl.BALANCE_TOL, 1 / a["n"]) for k, a in aud.items()
                   if not k.endswith(":*"))
    aud = fl.audit(full["fluid-v1-rebind"])
    assert all(abs(a["recency"] - 0.2) < 1e-9 for k, a in aud.items() if not k.endswith(":*"))
    aud = fl.audit(full["fluid-v1-morph"])
    assert aud["F1:suffix|n8"]["recency_subset"] == 120 and aud["F1:suffix|n8"]["recency"] is None
    assert abs(aud["F1:cond|n4"]["recency"] - 0.5) < 1e-9


def test_audit_catches_planted_shortcuts(full):
    rows = [dict(r) for r in full["fluid-v1-algebra"] if r["cell"] == "k1|d1"]
    for r in rows:
        r["gold_is_recent"] = True                     # a battery where the last example always answers
        r["freq_follower"] = 1.0
        r["gold_index"] = 0
    bad = fl.audit_failures(fl.audit(rows))
    assert any("recency" in x for x in bad) and any("frequency" in x for x in bad)
    assert any("position" in x for x in bad)
    skew = [dict(r, gold="tob") if i % 2 == 0 else r for i, r in enumerate(full["fluid-v1-seqind"])]
    assert any("identity" in x for x in fl.audit_failures(fl.audit(skew)))


def test_controls_count_whole_words():
    c = fl._controls("ba mi → ba mi ta\nbat ta", ["ta", "ba", "mi", "pu"], "pu", "ta")
    assert c["gold_count"] == 0 and c["gold_is_recent"] is False
    assert c["freq_follower"] == 0.0 and c["novelty_follower"] == 1.0
    c = fl._controls("x y y x", ["x", "y"], "x", None)
    assert c["freq_follower"] == 0.5 and c["gold_is_recent"] is False


# ── reader ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _scored(rows: list[dict[str, Any]], p_ok: dict[int, float], pick_wrong: str = "any",
            seed: int = 0) -> list[dict[str, Any]]:
    """Fake scorer output: per rung, a fraction ``p_ok`` of rows pick gold; the rest pick a non-gold candidate
    (``pick_wrong="stale"`` picks the F3 stale value)."""
    rng = random.Random(seed)
    by: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["rung"], []).append(r)
    out = []
    for rung, sub in by.items():
        hits = round(p_ok[rung] * len(sub))
        for i, r in enumerate(sub):
            cands = [c.strip() for c in r["candidates"]]
            if i < hits:
                pick = r["gold"]
            elif pick_wrong == "stale":
                pick = r["stale"]
            else:
                pick = rng.choice([c for c in cands if c != r["gold"]])
            x = {k: v for k, v in r.items() if k != "prompt"}
            x.update(model="m", cand_lps={c: (0.0 if c == pick else -4.0) for c in cands}, multi_token=[])
            out.append(x)
    return out


def _acc_for(fam: fl.Family, mid: float, beta: float = 1.5) -> dict[int, float]:
    c = 1 / fam.n_candidates
    return {k: c + (1 - c) / (1 + math.exp(beta * (k - mid))) for k in range(1, len(fam.ladder) + 1)}


def test_chance_and_normalisation(full):
    for b, rows in full.items():
        fam = fl.BY_BATTERY[b]
        s = fl.summarize(_scored(rows, {k: 0.0 for k in range(1, len(fam.ladder) + 1)}))
        assert s.chance == pytest.approx(1 / fam.n_candidates)
        assert all(v == pytest.approx(-1 / (fam.n_candidates - 1)) for v in s.curve.values())
        s = fl.summarize(_scored(rows, {k: 1.0 for k in range(1, len(fam.ladder) + 1)}))
        assert all(v == pytest.approx(1.0) for v in s.curve.values()) and s.auc == pytest.approx(1.0)


def test_midpoint_recovered_and_censored(full):
    rows = full["fluid-v1-morph"]
    fam = fl.FAMILIES["F1"]
    s = fl.summarize(_scored(rows, _acc_for(fam, 5.0)))
    assert abs(s.midpoint.value - 5.0) < 0.35 and s.midpoint.censored == "" and s.midpoint.score() == s.midpoint.value
    hi = fl.summarize(_scored(rows, _acc_for(fam, 30.0)))           # never crosses: still above half at the top
    assert hi.midpoint.censored == "high" and hi.midpoint.label() == ">9" and hi.midpoint.score() == 10.0
    lo = fl.summarize(_scored(rows, _acc_for(fam, -20.0)))          # below half from the first rung
    assert lo.midpoint.censored == "low" and lo.midpoint.label() == "<1" and lo.midpoint.score() == 0.0
    rising = fl.fit_midpoint([1, 2, 3, 4], [0.1, 0.3, 0.7, 0.9])     # no falling crossing inside the ladder
    assert rising.censored == "both" and rising.score() is None and rising.label() == "n/m"
    with pytest.raises(ValueError):
        fl.fit_midpoint([1, 2], [0.5])


def test_summarize_rejects_mixed_or_empty(full):
    with pytest.raises(ValueError):
        fl.summarize([])
    mixed = _scored(full["fluid-v1-morph"][:5], {k: 1.0 for k in range(1, 10)}) + \
        _scored(full["cryst-v1"][:5], {k: 1.0 for k in range(1, 5)})
    with pytest.raises(ValueError, match="mix"):
        fl.summarize(mixed)


def test_f3_stale_rate(full):
    rows = full["fluid-v1-rebind"]
    s = fl.summarize(_scored(rows, {k: 0.75 for k in range(1, 11)}, pick_wrong="stale"))
    assert s.stale is not None
    assert s.stale["rate"] == pytest.approx(0.25, abs=0.01) and s.stale["of_errors"] == 1.0
    assert set(s.stale["by_register"]) == {"prose", "code"} and s.stale["error_roles"] == {"stale": 300}
    s = fl.summarize(_scored(rows, {k: 0.75 for k in range(1, 11)}))
    assert s.stale["rate"] < 0.25 and s.stale["of_errors"] < 1.0
    assert set(s.conditions) == {"prose", "code"}


def test_c0_by_shots(full):
    rows = full["cryst-v1"]
    s = fl.summarize(_scored(rows, {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0}))
    assert s.by_shots is not None and s.by_shots["all"] == {0: 0.0, 1: 0.5, 3: 1.0, 6: 1.0}
    assert set(s.by_shots) == {"all", *fl.RELATIONS}
    d = readers.READERS["cryst"](_scored(rows, {1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0}))
    assert d["family"] == "C0" and d["by_shots"]["all"][6] == 1.0


def test_fluid_scores_zscore_across_models(full):
    def summ(mid: float, fam_key: str) -> fl.FamilySummary:
        fam = fl.FAMILIES[fam_key]
        return fl.summarize(_scored(full[fam.battery], _acc_for(fam, mid)))

    models = {
        "strong": {k: summ(6.0, k) for k in fl.FLUID_KEYS} | {"C0": summ(1.5, "C0")},
        "weak": {k: summ(3.0, k) for k in fl.FLUID_KEYS} | {"C0": summ(3.0, "C0")},
        "mid": {k: summ(4.5, k) for k in fl.FLUID_KEYS} | {"C0": summ(2.2, "C0")},
        "no-c0": {k: summ(4.5, k) for k in fl.FLUID_KEYS},
    }
    res = fl.fluid_scores(models)
    assert res["strong"]["fluid"] > res["mid"]["fluid"] > res["weak"]["fluid"]
    assert abs(sum(v["fluid"] for v in res.values())) < 1e-9
    assert res["weak"]["crystallized"] > res["strong"]["crystallized"]    # C0's midpoint: higher = later fall
    assert res["no-c0"]["crystallized"] is None and res["no-c0"]["gap"] is None
    assert res["strong"]["gap"] == pytest.approx(res["strong"]["fluid"] - res["strong"]["crystallized"])
    alt = fl.fluid_scores(models, stat="auc", order="z_then_mean")
    assert alt["strong"]["fluid"] > alt["weak"]["fluid"]
    with pytest.raises(ValueError):
        fl.fluid_scores(models, order="median")   # type: ignore[arg-type]


def test_fluid_cli_reads_scores(tmp_path, full, capsys):
    for b, rows in full.items():
        fam = fl.BY_BATTERY[b]
        d = tmp_path / "scores" / b
        d.mkdir(parents=True)
        for name, mid in (("a", 2.0), ("b", 5.0)):
            sc = _scored(rows, _acc_for(fam, mid))
            for x in sc:
                x["model"] = name
            (d / f"{name}.jsonl").write_text(_bytes(sc))
    out = tmp_path / "f.json"
    assert cli_main(["fluid", "--scores", str(tmp_path / "scores"), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "F3 mid" in text and text.count("\n") >= 3
    dump = json.loads(out.read_text())
    assert dump["scores"]["b"]["fluid"] > dump["scores"]["a"]["fluid"]
    assert cli_main(["report", "--scores", str(tmp_path / "scores")]) == 0       # the scorecard tolerates them


# ── token pools ───────────────────────────────────────────────────────────────────────────────────────────────────

def test_check_pools_flags_splits_and_collisions():
    vocab = {" ba": 1, " mi": 2, " tob": 3}

    def enc(s: str) -> list[int]:
        return [vocab[s]] if s in vocab else [9, 9]

    assert fl.check_pools({"t": enc}, {"p": ("ba", "mi", "tob")}) == []
    bad = fl.check_pools({"t": enc}, {"p": ("ba", "zzz")})
    assert len(bad) == 1 and "'zzz' is 2 tokens" in bad[0]
    same = fl.check_pools({"t": lambda s: [7]}, {"p": ("ba", "mi")})
    assert len(same) == 1 and "share token 7" in same[0]


def test_rows_score_single_token_through_the_suite_scorer(tmp_path, full):
    """Through the real scorer and our tokenizer (skipped when it is not cached): no candidate is multi-token."""
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    import torch

    from kotodama.eval.binding.models import LoadedModel, ModelSpec
    from kotodama.eval.binding.score import read_jsonl, score_prompt_rows

    g = torch.Generator().manual_seed(0)
    emb, unemb = torch.randn(len(tok), 8, generator=g), torch.randn(8, len(tok), generator=g)
    model = LoadedModel(ModelSpec("hf", "stub"), lambda ids: emb[ids] @ unemb, tok, torch.device("cpu"), None)
    enc = {"smollm2": lambda s: tok.encode(s, add_special_tokens=False)}
    assert fl.check_pools(enc) == []
    for b, rows in full.items():
        sample = rows[:: max(1, len(rows) // 40)]
        out = tmp_path / f"{b}.jsonl"
        score_prompt_rows(model, sample, out, "stub", max_batch=16)
        scored = read_jsonl(out)
        assert len(scored) == len(sample) and all(r["multi_token"] == [] for r in scored)
        assert fl.summarize(scored).n_rows == len(sample)


def test_pools_are_sound():
    assert len(fl.SYLLABLES) == len(set(fl.SYLLABLES)) == 38
    assert not {"be", "do", "go", "me", "no", "so", "to"} & set(fl.SYLLABLES)
    for c in {s[0] for s in fl.SYLLABLES}:                   # every consonant row offers a vowel change
        assert len([s for s in fl.SYLLABLES if s[0] == c]) >= 4
    for name, pool in fl.POOLS.items():
        assert len(pool) == len(set(pool)), name
    for _, facts in fl.RELATIONS.values():
        assert len(facts) >= max(fl.C0_SHOTS) + fl.FAMILIES["C0"].n_candidates
        assert len({s for s, _, _ in facts}) == len(facts)
    assert not set(fl.VALUES) & set(fl.HOLDERS)
