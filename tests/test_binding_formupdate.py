"""The form-update battery (kotodama.eval.binding.formupdate): determinism and pin, the grid, the row contract, the
A / B / C pairing invariants re-derived from the prompt text, the shortcut audits, the reader on synthetic scores
with known costs, the paired bootstrap, the four predictions and the CLI. CPU only; no network, no tokenizer."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

import pytest

from kotodama.eval.binding import batteries as bt
from kotodama.eval.binding import fluid as fl
from kotodama.eval.binding import formupdate as fu
from kotodama.eval.binding import readers
from kotodama.eval.binding.__main__ import main as cli_main


@pytest.fixture(scope="module")
def full() -> list[dict[str, Any]]:
    return fu.generate()


def _bytes(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["base_id"], {})[r["condition"]] = r
    return out


# ── determinism, pin, registry, grid ──────────────────────────────────────────────────────────────────────────────

def test_same_seed_same_bytes(full):
    assert _bytes(fu.generate(base=5)) == _bytes(fu.generate(base=5))
    assert _bytes(fu.generate()) == _bytes(full)


def test_gen_formupdate_reproduces_the_pin(tmp_path, capsys):
    assert cli_main(["gen-formupdate", "--out", str(tmp_path)]) == 0
    b = bt.BATTERIES[fu.BATTERY]
    assert bt.sha256(tmp_path / b.file) == b.sha256
    out = capsys.readouterr().out
    assert "= pin" in out and "AUDIT" not in out


def test_registry_group_and_reader():
    b = bt.BATTERIES["formupdate-v1"]
    assert b.kind == "rows" and b.reader == fu.READER and b.reader in readers.READERS
    assert re.fullmatch(r"[0-9a-f]{64}", b.sha256)
    assert [x.name for x in bt.resolve("formupdate")] == ["formupdate-v1"]
    assert "formupdate-v1" in bt.GROUPS["all"]


def test_grid(full):
    assert len(full) == 2160 and len({r["id"] for r in full}) == 2160
    groups = _groups(full)
    assert len(groups) == 720
    cells: dict[str, int] = {}
    for r in full:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1
    assert len(cells) == 18 and set(cells.values()) == {fu.BASE_ITEMS * 3}
    assert {(r["K"], r["U"], r["register"]) for r in full} == set(fu.CELLS)


# ── row contract ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_rows_contract(full):
    for r in full:
        cands = [c.strip() for c in r["candidates"]]
        assert len(cands) == fu.N_CANDIDATES == len(set(cands))
        assert all(c.startswith(" ") and " " not in c[1:] for c in r["candidates"])
        assert set(cands) <= set(fl.VALUES)            # a pool the fluid token check already covers
        assert r["gold"] in cands and cands.index(r["gold"]) == r["gold_index"]
        assert set(r["roles"]) == set(cands) and r["roles"][r["gold"]] == "gold"
        assert not r["prompt"].endswith((" ", "\n"))
        want = {"A": {"q_unshown"}, "B": {"q_unshown"}, "C": {"stale"}}[r["condition"]]
        assert {v for v in r["roles"].values() if v in ("q_unshown", "stale")} == want
        assert sum(v == "absent" for v in r["roles"].values()) == (2 if r["K"] == 2 else 0)
        if r["condition"] == "C":
            assert r["stale"] == r["q_initial"] and r["roles"][r["stale"]] == "stale"
        else:
            assert r["stale"] is None and r["roles"][r["q_new"]] == "q_unshown"


def _replay(r: dict[str, Any]) -> tuple[list[tuple[str, Optional[str], Optional[str]]], str]:
    """(kind, holder, value) per line and the queried holder, parsed back out of the prompt text."""
    out: list[tuple[str, Optional[str], Optional[str]]] = []
    if r["register"] == "code":
        *lines, q = r["prompt"].split("\n")
        seen: set[str] = set()
        for x in lines:
            if x in fu.CODE_EVENTS:
                out.append(("filler", None, None))
                continue
            h, v = x.split(" = ")
            out.append(("update" if h in seen else "bind", h, v))
            seen.add(h)
        return out, q.removeprefix("assert ").removesuffix(" ==")
    body, q = r["prompt"].split(" In the end, the ")
    for sent in re.findall(r"[^.]+\.", body):
        if m := re.fullmatch(r"\s*The (\w+) holds the (\w+)\.", sent):
            out.append(("bind", *m.groups()))
        elif m := re.fullmatch(r"\s*Then the (\w+) is emptied and the (\w+) is put in it\.", sent):
            out.append(("update", *m.groups()))
        else:
            assert sent.strip() in fu.PROSE_EVENTS, sent
            out.append(("filler", None, None))
    return out, q.removesuffix(" holds the")


def test_text_gives_gold_stale_roles_and_recency(full):
    for r in full:
        ev, q = _replay(r)
        assert q == r["holder"]
        cur: dict[str, str] = {}
        upd: set[str] = set()
        for kind, h, v in ev:
            if kind == "filler":
                continue
            assert h is not None and v is not None
            cur[h] = v
            if kind == "update":
                upd.add(h)
        assert cur[q] == r["gold"]
        k, u = r["K"], r["U"]
        assert [e[0] for e in ev[:k]] == ["bind"] * k and len(ev) == k + u == r["n_lines"]
        assert ev[r["q_bind_pos"]][1] == q and ev[r["q_bind_pos"]][2] == r["q_initial"]
        kinds = [e[0] for e in ev[k:]]
        assert kinds == (["filler"] * u if r["condition"] == "A" else ["update"] * u)
        q_updates = [i for i, e in enumerate(ev[k:]) if e[1] == q]
        assert q_updates == ([r["q_update_pos"]] if r["condition"] == "C" else [])
        for w, role in r["roles"].items():
            if role in ("other_current", "other_updated"):
                h = next(h for h, v in cur.items() if v == w)
                assert h != q and (role == "other_updated") == (h in upd)
        last = next(v for kind, _, v in reversed(ev) if kind != "filler")
        assert last == r["recent_answer"] and r["gold_is_recent"] == (last == r["gold"])
        assert all(fl.count_word(r["prompt"], c.strip()) <= 1 for c in r["candidates"])
        assert fl.count_word(r["prompt"], r["gold"]) == 1


def test_prose_and_code_follow_the_fluid_f3_templates(full):
    for r in full:
        if r["condition"] == "A":
            continue
        ev, q = _replay(r)
        assert fu.render(r["register"], ev, q) == fl._f3_render(r["register"], [(h, v, k == "update")
                                                                                for k, h, v in ev], q)


# ── pairing ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_pairing_invariants(full):
    assert fu.pairing_failures(full) == []
    for g in _groups(full).values():
        a, b, c = g["A"], g["B"], g["C"]
        ea, eb, ec = (_replay(x)[0] for x in (a, b, c))
        k = a["K"]
        assert ea[:k] == eb[:k] == ec[:k]                                    # entities, initial values, order
        assert _replay(a)[1] == _replay(b)[1] == _replay(c)[1]               # the query
        diff = [i for i in range(len(eb)) if eb[i] != ec[i]]
        assert diff == [k + c["q_update_pos"]] and ec[diff[0]] == ("update", c["holder"], c["q_new"])
        assert a["gold"] == b["gold"] == c["stale"] == a["q_initial"] and c["gold"] == a["q_new"]
        assert fl.count_word(a["prompt"], a["q_new"]) == fl.count_word(b["prompt"], b["q_new"]) == 0
        ca, cb, cc = ([x.strip() for x in r["candidates"]] for r in (a, b, c))
        assert ca.index(a["q_new"]) == cb.index(a["q_new"]) == cc.index(a["q_new"])
        assert ca.index(a["q_initial"]) == cb.index(a["q_initial"]) == cc.index(a["q_initial"])
        filler_words = {w for line in fu.lines_of(a)[k:] for w in line.rstrip(".").split()}
        assert not filler_words & (set(fl.VALUES) | set(fl.HOLDERS))


def test_pairing_failures_catch_breaks(full):
    rows = [dict(r) for r in full[:6]]
    rows[1]["prompt"] = rows[1]["prompt"].replace(rows[1]["holder"], "box" if rows[1]["holder"] != "box" else "jar", 1)
    assert fu.pairing_failures(rows)
    assert any("conditions" in x for x in fu.pairing_failures(full[:5]))
    swapped = [dict(r) for r in full[:3]]
    swapped[2]["gold"] = swapped[2]["q_initial"]
    assert any("gold" in x for x in fu.pairing_failures(swapped))


def test_schedules_balanced(full):
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in full:
        by.setdefault((r["cell"], r["condition"]), []).append(r)
    for (cell, cond), sub in by.items():
        k, u = sub[0]["K"], sub[0]["U"]
        for field, n in (("q_bind_pos", k), ("q_update_pos", u), ("gold_index", fu.N_CANDIDATES)):
            counts = [sum(r[field] == i for r in sub) for i in range(n)]
            assert len(set(counts)) == 1, (cell, cond, field, counts)
    for g in _groups(full).values():
        assert g["A"]["q_initial"] != g["A"]["q_new"]


# ── audits ────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_audits_pass_at_their_designed_rates(full):
    aud = fu.audit(full)
    assert fu.audit_failures(aud) == []
    for key, a in aud.items():
        if key.startswith("identity|"):
            assert a["identities"] == len(fl.VALUES) and a["identity_max_ratio"] <= fu.IDENTITY_MAX
            continue
        assert a["position_max_dev"] == 0.0 and a["max_count"] == 1 and a["gold_count"] == [1]
        if key.startswith("*|"):
            continue
        k, u = (int(x) for x in re.match(r"K(\d+)\|U(\d+)", key).groups())   # type: ignore[union-attr]
        cond = key[-1]
        want = {"A": 1 / k, "B": 0.0, "C": 1 / u}[cond]
        assert a["recency"] == pytest.approx(want)
        present = (2 if k == 2 else 4) if cond in "AB" else (3 if k == 2 else 5)
        assert a["presence"] == pytest.approx(1 / present) and a["novelty"] == 0.0
    assert aud["*|C"]["recency"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)


def test_audit_catches_planted_shortcuts(full):
    rows = [dict(r) for r in full if r["cell"] == "K4|U2|code"]
    for r in rows:
        r["gold_is_recent"] = True
        r["gold_index"] = 0
        r["max_count"] = 2
    bad = fu.audit_failures(fu.audit(rows))
    assert any("recency" in x for x in bad) and any("position" in x for x in bad)
    assert any("mentioned" in x for x in bad)
    skew = [dict(r, gold="lamp") if i % 2 == 0 else r for i, r in enumerate(full)]
    assert any("identity" in x for x in fu.audit_failures(fu.audit(skew)))


def test_filler_vocabulary_is_clean():
    words = {w for p in fu.FILLER_ADJ + fu.FILLER_NOUN + fu.FILLER_VERB + fu.FILLER_PLACE for w in p.split()}
    assert not words & (set(fl.VALUES) | set(fl.HOLDERS))
    assert "the edge of the map" not in fu.FILLER_PLACE and len(fu.FILLER_PLACE) >= 4


# ── reader ────────────────────────────────────────────────────────────────────────────────────────────────────────

Rule = Callable[[dict[str, Any], int], str]


def _scored(rows: list[dict[str, Any]], pick: Rule, model: str = "m") -> list[dict[str, Any]]:
    """Fake scorer output: ``pick(row, base index)`` names the candidate that gets the top logprob."""
    out = []
    for r in rows:
        b = int(r["base_id"].rsplit("|b", 1)[1])
        p = pick(r, b)
        cands = [c.strip() for c in r["candidates"]]
        assert p in cands
        x = {k: v for k, v in r.items() if k != "prompt"}
        x.update(model=model, cand_lps={c: (0.0 if c == p else -4.0) for c in cands}, multi_token=[])
        out.append(x)
    return out


def _known(acc_b: int, acc_c: int, stale_c: int) -> Rule:
    """Per cell: A always right; B right on base items b < acc_b (of 40); C right on b < acc_c, and on the next
    stale_c items it picks the stale value; every other error picks another holder's value."""
    def pick(r: dict[str, Any], b: int) -> str:
        other = next(w for w, role in r["roles"].items() if role in ("other_current", "other_updated"))
        if r["condition"] == "A":
            return r["gold"]
        if r["condition"] == "B":
            return r["gold"] if b < acc_b else other
        if b < acc_c:
            return r["gold"]
        return r["stale"] if b < acc_c + stale_c else other
    return pick


def test_reader_recovers_known_costs(full):
    s = fu.summarize(_scored(full, _known(32, 20, 12)), n_boot=500)
    p = s.pooled
    assert s.n_items == 720 and s.incomplete == 0 and s.n_rows == 2160
    assert p["acc"] == pytest.approx({"A": 1.0, "B": 0.8, "C": 0.5})
    assert p["form"] == 1.0 and p["interference"] == pytest.approx(0.2) and p["update"] == pytest.approx(0.3)
    # pooled: every item, normalized by B's headroom
    assert p["norm_interference"] == pytest.approx(0.2 / 0.6) and p["norm_update"] == pytest.approx(0.3 / 0.6)
    assert p["stale_rate"] == pytest.approx(0.3) and p["stale_of_errors"] == pytest.approx(0.6)
    # point (the predictions' statistics): the recency-matched items, normalized by their B headroom
    acc = s.recency_split["c_gold_not_last"]["acc"]
    head = acc["B"] - fu.CHANCE
    assert s.point["norm_update"] == pytest.approx((acc["B"] - acc["C"]) / head)
    assert s.point["norm_interference"] == pytest.approx((acc["A"] - acc["B"]) / head)
    assert s.point["norm_update_minus_interference"] == pytest.approx(
        (acc["B"] - acc["C"]) / head - (acc["A"] - acc["B"]) / head)
    for part in (s.by_k, s.by_u, s.by_register, s.by_cell):
        for x in part.values():
            assert x["update"] == pytest.approx(0.3) and x["interference"] == pytest.approx(0.2)
    assert len(s.by_cell) == 18 and set(s.by_k) == set(fu.K_GRID) and set(s.by_register) == {"prose", "code"}
    assert s.error_roles["A"] == {} and s.error_roles["C"]["stale"] == 12 * 18
    assert sum(s.error_roles["B"].values()) == 8 * 18
    d = readers.READERS["formupdate"](_scored(full, _known(32, 20, 12)))
    assert d["pooled"]["update"] == pytest.approx(0.3) and d["chance"] == pytest.approx(0.2)


def test_reader_recency_split(full):
    """C right exactly where its gold is the last mention: the recency-matched update cost is B's accuracy."""
    def pick(r: dict[str, Any], b: int) -> str:
        if r["condition"] == "C":
            return r["gold"] if r["gold_is_recent"] else r["stale"]
        return r["gold"]
    s = fu.summarize(_scored(full, pick), n_boot=200)
    assert s.recency_split["c_gold_last"]["acc"]["C"] == 1.0
    assert s.recency_split["c_gold_not_last"]["acc"]["C"] == 0.0
    assert s.point["update_nonrecent"] == pytest.approx(1.0)
    assert s.recency_split["c_gold_not_last"]["n"] == sum(
        1 for r in full if r["condition"] == "C" and not r["gold_is_recent"])
    assert s.pooled["acc"]["C"] == pytest.approx((1 + 1 / 2 + 1 / 4) / 3)


def test_reader_normalized_cost_undefined_without_headroom(full):
    def pick(r: dict[str, Any], b: int) -> str:              # A at chance: one right in five
        return r["gold"] if b % 5 == 0 else next(w for w in r["roles"] if w != r["gold"])
    s = fu.summarize(_scored(full, pick), n_boot=200)
    assert s.pooled["form"] == pytest.approx(0.2) and s.pooled["norm_update"] is None
    assert s.point["norm_update"] is None and s.ci["norm_update"]["defined"] < 200


def test_summarize_refuses_bad_input_and_counts_incomplete(full):
    with pytest.raises(ValueError):
        fu.summarize([])
    sc = _scored(full, _known(40, 40, 0))
    with pytest.raises(ValueError, match="candidate"):
        fu.summarize([dict(sc[0], candidates=sc[0]["candidates"][:4])] + sc[1:])
    with pytest.raises(ValueError, match="two"):
        fu.summarize(sc + sc[:1])
    s = fu.summarize(sc[1:], n_boot=100)
    assert s.incomplete == 1 and s.n_items == 719


# ── bootstrap ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_bootstrap_contains_point_and_narrows_with_items(full):
    import random
    rng = random.Random(3)
    sc = _scored(full, lambda r, b: r["gold"] if rng.random() < 0.7 else
                 next(w for w in r["roles"] if w != r["gold"]))
    s = fu.summarize(sc)
    for stat in fu.BOOT_STATS:
        lo, hi = s.ci[stat]["ci"]
        assert lo <= s.point[stat] <= hi, stat
        assert s.ci[stat]["n_boot"] == fu.BOOT
    items = [fu._Item((2, 1, "prose"), (rng.random() < .8, rng.random() < .6, rng.random() < .4), False, False)
             for _ in range(60)]
    small = fu.bootstrap(items, n_boot=2000)
    big = fu.bootstrap(items * 9, n_boot=2000)
    for stat in ("form", "update", "norm_update"):
        w_small = small[stat]["ci"][1] - small[stat]["ci"][0]
        w_big = big[stat]["ci"][1] - big[stat]["ci"][0]
        assert w_big < 0.5 * w_small, stat
    assert fu.bootstrap(items, n_boot=300, seed=1) == fu.bootstrap(items, n_boot=300, seed=1)
    with pytest.raises(ValueError):
        fu.bootstrap([])


# ── predictions ───────────────────────────────────────────────────────────────────────────────────────────────────

def _toy(form: float, ni: Optional[float], nu: Optional[float]) -> fu.Summary:
    point = {"form": form, "interference": 0.0, "update": 0.0, "norm_interference": ni, "norm_update": nu,
             "norm_update_minus_interference": (nu - ni) if nu is not None and ni is not None else None,
             "update_nonrecent": 0.0}
    ci = {k: {"ci": [v - 0.05, v + 0.05] if v is not None else None, "defined": 1, "n_boot": 1}
          for k, v in point.items()}
    return fu.Summary("toy", 720, 2160, 0, 0, {"form": form}, ci, point)


def test_predictions_on_toy_numbers():
    ref = {"r1": _toy(0.80, 0.10, 0.20), "r2": _toy(0.90, 0.05, 0.30), "r3": _toy(0.70, 0.20, 0.25),
           "r4": _toy(0.85, 0.00, 0.10), "r5": _toy(0.95, 0.10, 0.15)}
    # medians: form .85; norm update .20; (update − interference) over r1..r5 = .10 .25 .05 .10 .05 → .10
    sums = {**ref, "rel": _toy(0.85, 0.05, 0.40), "ann": _toy(0.86, 0.05, 0.30)}
    p = fu.predictions(sums, list(ref), "rel", "ann")
    assert p["P1"]["median"] == pytest.approx(0.85) and p["P1"]["holds"] is True       # ≥ holds at equality
    assert p["P2"]["median"] == pytest.approx(0.20) and p["P2"]["holds"] is True
    assert p["P3"]["median"] == pytest.approx(0.10) and p["P3"]["release"] == pytest.approx(0.35)
    assert p["P3"]["holds"] is True and p["P4"]["holds"] is True
    assert p["P2"]["release_ci"] == pytest.approx([0.35, 0.45])
    sums2 = {**ref, "rel": _toy(0.84, 0.30, 0.20), "ann": _toy(0.9, 0.0, 0.50)}
    q = fu.predictions(sums2, list(ref), "rel", "ann")
    assert (q["P1"]["holds"], q["P2"]["holds"], q["P3"]["holds"], q["P4"]["holds"]) == (False, False, False, False)
    # the reference set is a parameter: a different membership moves the median
    r = fu.predictions(sums, ["r3", "r4"], "rel", "ann")
    assert r["P1"]["median"] == pytest.approx(0.775) and r["reference"] == ["r3", "r4"]
    und = {**sums, "r5": _toy(0.15, None, None)}
    u = fu.predictions(und, list(ref), "rel", "ann")
    assert u["P2"]["undefined"] == ["r5"] and u["P2"]["median"] == pytest.approx(0.225)
    v = fu.predictions({**sums, "rel": _toy(0.18, None, None)}, list(ref), "rel", "ann")
    assert v["P2"]["holds"] is None and v["P4"]["holds"] is None and v["P1"]["holds"] is False
    with pytest.raises(KeyError):
        fu.predictions(sums, ["r1", "missing"], "rel", "ann")
    with pytest.raises(ValueError):
        fu.predictions(sums, ["r1", "rel"], "rel", "ann")
    with pytest.raises(ValueError):
        fu.predictions(sums, [], "rel", "ann")


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_formupdate_cli_reads_scores_and_predictions(tmp_path, full, capsys):
    d = tmp_path / "scores" / fu.BATTERY
    d.mkdir(parents=True)
    models = {"ref-a": (36, 30, 5), "ref-b": (38, 34, 3), "ref-c": (34, 24, 10), "rel": (38, 20, 15),
              "ann": (38, 28, 8)}
    for name, (b, c, st) in models.items():
        (d / f"{name}.jsonl").write_text(_bytes(_scored(full, _known(b, c, st), model=name)))
    out = tmp_path / "fu.json"
    assert cli_main(["formupdate", "--scores", str(tmp_path / "scores"), "--boot", "200", "--json", str(out),
                     "--reference", "ref-a,ref-b,ref-c", "--release", "rel", "--anneal", "ann"]) == 0
    text = capsys.readouterr().out
    assert "norm update" in text and "P4" in text
    dump = json.loads(out.read_text())
    assert dump["summaries"]["rel"]["pooled"]["update"] == pytest.approx(0.45)
    assert dump["predictions"]["P2"]["holds"] is True and dump["predictions"]["P4"]["holds"] is True
    assert cli_main(["formupdate", "--scores", str(tmp_path / "scores"), "--boot", "50", "--release", "rel"]) == 1
    assert cli_main(["formupdate", "--scores", str(tmp_path / "empty")]) == 1
    assert cli_main(["report", "--scores", str(tmp_path / "scores")]) == 0       # the scorecard tolerates them


def test_a_events_name_nothing_and_stay_out_of_b_and_c(full):
    assert len(fu.PROSE_EVENTS) >= max(fu.U_GRID) and len(fu.CODE_EVENTS) >= max(fu.U_GRID)
    events = fu.PROSE_EVENTS + fu.CODE_EVENTS
    for r in full:
        if r["condition"] == "A":
            assert sum(e in r["prompt"] for e in events) == r["U"]
        else:
            assert not any(e in r["prompt"] for e in events)
