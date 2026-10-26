"""From self-play runs to training data and measurements.

Subcommands (each reads the ``transcripts.jsonl`` / ``candidates.jsonl`` a ``kotodama.posttrain.selfplay`` run
writes):

``assemble``  transcript rows for ``kotodama.posttrain.sft`` (``--format transcript``): keeps transcripts with at
              least ``--min_turns`` turns, drops exact duplicates, and blanks the header of a ``--header_dropout``
              fraction so the model learns the genre with and without a stated cast.
``meters``    run health: finish reasons, veto rates by reason, survivors per turn, turn lengths, distinct-n over
              turns and within turns, rollout diversity (how different a turn's sibling candidates are from each
              other), cross-transcript 5-gram reuse (a corpus repeating itself across transcripts is collapsing even
              when no single transcript loops).
``fluency``   the picked turns' per-token NLL under the served model's PLAIN distribution (``/score_tokens``: no
              temperature, penalty or steering), written to ``<run>/fluency.json`` — the cost side of any sampling or
              steering change.
``compare``   one table over several runs (sampling, steering, the meters, fluency): the A/B read.
``precision-sample``  the coherence study. A blind, shuffled sample of candidate continuations in three strata —
              survivors of the vetoes, vetoed, and unfiltered (any candidate) — each with the transcript as it stood
              when the candidate was drawn. Strata are hidden from the rater.
``rate``      a terminal rater over a sample: for each item, is this a good, logical, coherent continuation of the
              transcript? (y / n / p = partial, s = skip, q = quit; resumable).
``score``     precision per stratum from the ratings, the survivors' lift over unfiltered, and the share of good
              continuations the vetoes removed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from kotodama.posttrain.transcript import Transcript, Turn, parse_row, render

logger = logging.getLogger(__name__)


def _read(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _runs(dirs: Iterable[str], name: str) -> Iterator[dict[str, Any]]:
    for d in dirs:
        p = Path(d) / name
        if p.exists():
            yield from _read(p)
        else:
            logger.warning("%s missing", p)


# ── assemble ─────────────────────────────────────────────────────────────────


def assemble(rows: Iterable[dict[str, Any]], min_turns: int, header_dropout: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out = []
    for row in rows:
        tr = parse_row(row)
        if tr is None or len(tr.turns) < min_turns:
            continue
        key = hashlib.sha1(render(tr)[0].encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        rec: dict[str, Any] = {"turns": [{"speaker": t.speaker, "text": t.text} for t in tr.turns]}
        if tr.header and rng.random() >= header_dropout:
            rec["header"] = tr.header
        rec["source"] = row.get("tid", row.get("seed", ""))
        out.append(rec)
    return out


# ── meters ───────────────────────────────────────────────────────────────────


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(ws: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def meters(transcripts: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    finishes = collections.Counter(t.get("finish", "?") for t in transcripts)
    vetoes = collections.Counter(v for c in candidates for v in c.get("vetoes", []))
    n_c = len(candidates)
    turn_lens: list[int] = []
    uni: collections.Counter = collections.Counter()
    bi: collections.Counter = collections.Counter()
    five_by_doc: collections.Counter = collections.Counter()
    for t in transcripts:
        doc5: set[tuple[str, ...]] = set()
        for turn in t.get("turns", []):
            ws = _words(turn.get("text", ""))
            turn_lens.append(len(ws))
            uni.update(ws)
            bi.update(_ngrams(ws, 2))
            doc5.update(_ngrams(ws, 5))
        five_by_doc.update(doc5)
    n_docs = max(len(transcripts), 1)
    shared5 = sum(1 for g, k in five_by_doc.items() if k >= max(3, n_docs // 100))
    lens = sorted(turn_lens) or [0]
    groups = collections.defaultdict(list)
    for c in candidates:
        groups[(c.get("tid"), c.get("turn"), c.get("draw"))].append(c)
    surv = [sum(1 for c in g if not c.get("vetoes")) for g in groups.values()]
    rollout: list[float] = []
    for g in groups.values():
        sets = [set(_ngrams(_words(c.get("text", "")), 3)) for c in g]
        sets = [x for x in sets if x]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                rollout.append(1.0 - len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    cast_of = {t.get("tid"): set(t.get("cast", [])) for t in transcripts}
    from kotodama.posttrain.selfplay import GENERIC_ROLES

    labelled = [c for c in candidates if c.get("cut_label")]
    drifted = [c for c in labelled
               if c["cut_label"] not in GENERIC_ROLES and c["cut_label"] not in cast_of.get(c.get("tid"), set())]
    within = []
    for t in transcripts:
        for turn in t.get("turns", []):
            ws = _words(turn.get("text", ""))
            if len(ws) >= 10:
                within.append(len(set(_ngrams(ws, 2))) / max(len(ws) - 1, 1))
    return {
        "transcripts": len(transcripts),
        "finishes": dict(finishes),
        "turns_per_transcript": round(sum(len(t.get("turns", [])) for t in transcripts) / n_docs, 2),
        "candidates": n_c,
        "veto_rate": round(sum(1 for c in candidates if c.get("vetoes")) / max(n_c, 1), 4),
        "veto_reasons": {k: round(v / max(n_c, 1), 4) for k, v in vetoes.most_common()},
        "survivors_per_draw_mean": round(sum(surv) / max(len(surv), 1), 3),
        "draws_with_no_survivor": round(sum(1 for s in surv if s == 0) / max(len(surv), 1), 4),
        "words_per_turn_p10_p50_p90": [lens[int(.1 * (len(lens) - 1))], lens[len(lens) // 2],
                                       lens[int(.9 * (len(lens) - 1))]],
        "distinct_1": round(len(uni) / max(sum(uni.values()), 1), 4),
        "distinct_2": round(len(bi) / max(sum(bi.values()), 1), 4),
        "within_turn_distinct_2": round(sum(within) / max(len(within), 1), 4),
        "rollout_diversity": round(sum(rollout) / max(len(rollout), 1), 4),
        "stray_label_rate": round(len(labelled) / max(n_c, 1), 4),
        "name_drift_rate": round(len(drifted) / max(n_c, 1), 4),
        "cross_transcript_5gram_share": round(shared5 / max(len(five_by_doc), 1), 5),
    }


# ── fluency ──────────────────────────────────────────────────────────────────


def fluency_items(transcripts: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    """One scoring item per generated turn: the document up to the open tag, then the turn's text."""
    from kotodama.posttrain.transcript import render_prompt

    items = []
    for t in transcripts:
        tr = parse_row(t)
        if tr is None:
            continue
        first = len(tr.turns) - int(t.get("n_generated", len(tr.turns)))
        for k in range(max(first, 0), len(tr.turns)):
            turn = tr.turns[k]
            prompt = render_prompt(Transcript(tr.turns[:k], tr.header), turn.speaker)
            items.append({"id": f"{t.get('tid', t.get('seed'))}:{k}",
                          "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
                          "completion_ids": tokenizer.encode(" " + turn.text, add_special_tokens=False)})
    return [it for it in items if it["prompt_ids"] and it["completion_ids"]]


def score_fluency(items: list[dict[str, Any]], endpoint: str, timeout: float = 600.0,
                  transport: Any = None) -> dict[str, Any]:
    """Token-weighted mean NLL (and per-turn quantiles) from the server's ``/score_tokens``. ``transport``: an
    httpx transport (tests pass a mock)."""
    import httpx

    from kotodama.eval.riders.score import plan_batches

    lens = [len(it["prompt_ids"]) + len(it["completion_ids"]) for it in items]
    per_turn: list[float] = []
    tot_nll = tot_tok = 0.0
    with httpx.Client(timeout=timeout, transport=transport) as client:
        for batch in plan_batches(lens):
            r = client.post(endpoint.rstrip("/") + "/score_tokens", json={"items": [items[i] for i in batch]})
            r.raise_for_status()
            for res in r.json()["results"]:
                if res["n_tokens"]:
                    tot_nll += res["sum_nll"]
                    tot_tok += res["n_tokens"]
                    per_turn.append(res["mean_nll"])
    per_turn.sort()
    q = (lambda p: round(per_turn[min(len(per_turn) - 1, int(p * len(per_turn)))], 4)) if per_turn else (lambda p: None)
    return {"turns": len(per_turn), "tokens": int(tot_tok), "mean_nll": round(tot_nll / max(tot_tok, 1), 4),
            "turn_nll_p10_p50_p90": [q(.1), q(.5), q(.9)]}


def compare_rows(runs: list[str]) -> list[dict[str, Any]]:
    rows = []
    for d in runs:
        p = Path(d)
        meta = json.loads((p / "run.json").read_text()) if (p / "run.json").exists() else {}
        m = meters(list(_read(p / "transcripts.jsonl")), list(_read(p / "candidates.jsonl")))
        fl = json.loads((p / "fluency.json").read_text()) if (p / "fluency.json").exists() else {}
        samp = meta.get("sampling", {})
        steer = ",".join(f"{s['name']}:{s.get('alpha', 'd')}" for s in meta.get("steer", [])) or "-"
        rows.append({"run": p.name, "T": samp.get("temperature"), "rep": samp.get("repetition_penalty"),
                     "steer": steer, "veto": m["veto_rate"], "selfcopy": m["veto_reasons"].get("selfcopy", 0.0),
                     "no_surv": m["draws_with_no_survivor"], "turns": m["turns_per_transcript"],
                     "words/turn": m["words_per_turn_p10_p50_p90"][1], "d2": m["distinct_2"],
                     "d2_in_turn": m["within_turn_distinct_2"], "rollout_div": m["rollout_diversity"],
                     "x5gram": m["cross_transcript_5gram_share"], "drift": m["name_drift_rate"],
                     "nll": fl.get("mean_nll")})
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


# ── precision study ──────────────────────────────────────────────────────────


def context_text(row: dict[str, Any], turn: int) -> str:
    """The transcript as it stood before ``turn`` (header + turns[:turn]) — what the candidate continued."""
    tr = parse_row(row)
    if tr is None:
        return ""
    return render(Transcript(tr.turns[:turn], tr.header))[0]


def precision_sample(transcripts: list[dict[str, Any]], candidates: list[dict[str, Any]], per_stratum: int,
                     seed: int) -> list[dict[str, Any]]:
    by_tid = {t.get("tid", t.get("seed")): t for t in transcripts}
    usable = [c for c in candidates if c.get("tid") in by_tid and c.get("text")]
    rng = random.Random(seed)
    strata = {
        "survivor": [c for c in usable if not c.get("vetoes")],
        "vetoed": [c for c in usable if c.get("vetoes")],
        "unfiltered": usable,
    }
    items = []
    for name, pool in strata.items():
        for c in rng.sample(pool, min(per_stratum, len(pool))):
            items.append({"stratum": name, "vetoes": c.get("vetoes", []), "tid": c["tid"], "turn": c["turn"],
                          "speaker": c["speaker"], "context": context_text(by_tid[c["tid"]], c["turn"]),
                          "continuation": c["text"]})
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["item"] = f"i{i:05d}"
    return items


def score(sample: list[dict[str, Any]], ratings: dict[str, str]) -> dict[str, Any]:
    val = {"y": 1.0, "p": 0.5, "n": 0.0}
    per: dict[str, list[float]] = collections.defaultdict(list)
    for it in sample:
        r = ratings.get(it["item"])
        if r in val:
            per[it["stratum"]].append(val[r])
    prec = {k: round(sum(v) / len(v), 4) for k, v in per.items() if v}
    out: dict[str, Any] = {"rated": {k: len(v) for k, v in per.items()}, "precision": prec}
    if "survivor" in prec and "unfiltered" in prec:
        out["lift"] = round(prec["survivor"] - prec["unfiltered"], 4)
    if per.get("vetoed"):
        # good continuations the vetoes threw away, as a share of all vetoed items rated
        out["vetoed_but_good"] = round(sum(1 for x in per["vetoed"] if x >= 1.0) / len(per["vetoed"]), 4)
    return out


def rate(sample_path: Path, ratings_path: Path, inp=input, out=print) -> int:
    sample = list(_read(sample_path))
    done = {r["item"]: r["rating"] for r in _read(ratings_path)} if ratings_path.exists() else {}
    todo = [it for it in sample if it["item"] not in done]
    out(f"{len(done)} rated, {len(todo)} to go. y = good continuation, n = not, p = partial, s = skip, q = quit")
    with open(ratings_path, "a", encoding="utf-8") as f:
        for it in todo:
            out("\n" + "=" * 78 + f"\n{it['item']}\n" + "-" * 78)
            out(it["context"][-3000:])
            out("-" * 78 + f"\n>>> {it['speaker']}: {it['continuation']}\n" + "=" * 78)
            while True:
                a = inp("good continuation? [y/n/p/s/q] ").strip().lower()
                if a in ("y", "n", "p", "s", "q"):
                    break
            if a == "q":
                break
            if a == "s":
                continue
            f.write(json.dumps({"item": it["item"], "rating": a}) + "\n")
            f.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="self-play runs -> training data and measurements")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("assemble")
    a1.add_argument("runs", nargs="+")
    a1.add_argument("--out", required=True)
    a1.add_argument("--min_turns", type=int, default=4)
    a1.add_argument("--header_dropout", type=float, default=0.15)
    a1.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("meters")
    m.add_argument("runs", nargs="+")
    ps = sub.add_parser("precision-sample")
    ps.add_argument("runs", nargs="+")
    ps.add_argument("--out", required=True)
    ps.add_argument("--per_stratum", type=int, default=60)
    ps.add_argument("--seed", type=int, default=0)
    fl = sub.add_parser("fluency")
    fl.add_argument("run")
    fl.add_argument("--endpoint", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("runs", nargs="+")
    r = sub.add_parser("rate")
    r.add_argument("sample")
    r.add_argument("--ratings", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("sample")
    sc.add_argument("--ratings", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "assemble":
        rows = assemble(_runs(a.runs, "transcripts.jsonl"), a.min_turns, a.header_dropout, a.seed)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
        logger.info("assembled %d transcripts -> %s", len(rows), a.out)
    elif a.cmd == "meters":
        print(json.dumps(meters(list(_runs(a.runs, "transcripts.jsonl")), list(_runs(a.runs, "candidates.jsonl"))),
                         indent=1))
    elif a.cmd == "precision-sample":
        items = precision_sample(list(_runs(a.runs, "transcripts.jsonl")), list(_runs(a.runs, "candidates.jsonl")),
                                 a.per_stratum, a.seed)
        Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        logger.info("wrote %d items -> %s", len(items), a.out)
    elif a.cmd == "fluency":
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME

        items = fluency_items(list(_read(Path(a.run) / "transcripts.jsonl")),
                              AutoTokenizer.from_pretrained(TOKENIZER_NAME))
        res = score_fluency(items, a.endpoint)
        (Path(a.run) / "fluency.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    elif a.cmd == "compare":
        print(markdown_table(compare_rows(a.runs)))
    elif a.cmd == "rate":
        return rate(Path(a.sample), Path(a.ratings))
    elif a.cmd == "score":
        ratings = {x["item"]: x["rating"] for x in _read(Path(a.ratings))}
        print(json.dumps(score(list(_read(Path(a.sample))), ratings), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
