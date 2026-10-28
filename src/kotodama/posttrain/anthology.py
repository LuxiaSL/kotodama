"""Whole-scene assembly of loom runs: clean each scene, then stack scenes from different seeds into documents.

``python -m kotodama.posttrain.loom assemble <runs> --out docs.jsonl --anthology [...]`` takes every ended branch of
a finished tree (eos, depth, degraded or length_cap; never a pruned opening stub) as one scene, from its header to
its last turn, and cleans it in this order:

1. **Loop truncation** (``--truncate_loops``): the scene ends at ``loom.loop_onset``.
2. **Out-of-frame truncation** (``--frame_gate``): the scene ends before its first turn that leaves the scene for
   text about a work or a page (``narration.out_of_frame``) or that is the speaker's own character sheet
   (``narration.self_described``). A natural ending's final turn whose speech is in frame is the exception when
   ``--clean_tail`` is on: its closing paragraphs are cut instead (step 4), and the turn goes only if it is still
   out of frame after that.
3. **Closing cut** (``--cut_closings E``): self-play keeps a scene going after its story has closed. A closing
   episode is two farewell turns within four ("Goodnight." / "Bye." / "I love you too."); when the model's own
   end-of-text pressure at or just after the episode reaches ``E`` (the share of the K candidates drawn there that
   end the document), the scene is cut at the turn where that pressure peaks and one of those end-of-text
   candidates, itself a farewell, becomes its final turn. The ending is therefore model-written and conditioned on
   exactly the turns before it. An episode needs a parting proper ("Bye", "Good night", "See you", "Take care"):
   "I love you" or "I have to go" alone also come in the middle of a scene that is still developing. A sustained
   farewell orbit (from some turn on, at least 40% of the remaining turns are farewells and the turns add almost no
   new content words) is cut at its onset, ending on a farewell candidate when one was drawn there and otherwise
   cut (its end-of-text untrained). Cuts that would remove fewer than 4 turns are not made: a scene that closes and
   then ends is already right.
4. **Tail cleaning** (``--clean_tail``): a final turn that runs past a blank line keeps closing narration but stops
   at the first paragraph that is out of frame, a document marker ("The end.", "--", credits), another speaker's
   label, a cast-list line ("Name, thirty-two, ...") or text in another script; narration after the speech is
   kept up to about 100 words (a longer tail is a synopsis, not a closing).
5. **Scene gates**: ``--max_template`` drops a scene whose own-opening template share
   (``selfplay.own_opening_repeat``) exceeds it; ``--untrain_repeats`` / ``--untrain_format`` turn flawed turns into
   context (``loom.Clean``), except that a model-written final turn repeating a line ("Good night." / "Good
   night.") still trains; compliance (``--drop_minor_explicit``, ``--blocklist``) drops the scene; ``--redact_pii``
   redacts it. A scene left with fewer than ``--min_scene_turns`` turns is dropped.

A scene ends with the end-of-text token. It TRAINS only when the ending is model-written (a natural ending, or a
closing cut with a candidate ending); a scene cut anywhere else, or stopped by the depth cap, keeps its end-of-text
as context (``"ended": false``), so no document teaches stopping where the model did not stop.

Turns that two scenes of one tree share (the branched opening) train once: the second scene sees them as context.

Stacking: scenes are shuffled and packed greedily into documents of up to ``--max_chars`` characters; a document
never holds two scenes from one seed (``<id>~t<k>`` trees share a seed). A document row is
``{"scenes": [{"header", "turns", "ended", "source"}...]}``, which ``kotodama.posttrain.transcript`` renders as
the scenes one after another, each followed by end-of-text, in one attention segment. ``--scenes_out`` also
writes every cleaned scene as its own transcript row, for training scenes as separate packed examples instead: a
model pretrained with ``--doc_masking`` (``kotodama.training.train``) has never attended across an end-of-text, and
separate examples keep that.
"""

from __future__ import annotations

import json
import logging
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from kotodama.posttrain import compliance
from kotodama.posttrain.loom import Clean, _flawed, loop_onset, node_ids
from kotodama.posttrain.narration import out_of_frame, self_described, split_turn
from kotodama.posttrain.selfplay import novelty, own_opening_repeat, spoken_text
from kotodama.posttrain.transcript import Turn

logger = logging.getLogger(__name__)

ENDED_STATES = ("eos", "depth", "degraded", "length_cap")

FAREWELL = re.compile(
    r"\b(good ?-?bye|bye|bye-bye|good ?-?night|nite|(?<!to )see you(?!r)|see ya|farewell|take care(?! of)|love you|"
    r"miss you|"
    r"sweet dreams|sleep well|(?:have|got|need|must) to go(?=\s*(?:now|home|\.|!|,|$))|gotta go|must go|"
    r"i'?m leaving|i'?m off|"
    r"talk to you (?:later|tomorrow|soon)|until (?:tomorrow|next time)|"
    r"have a (?:good|nice|great) (?:night|day|evening|trip|one)|thank you for (?:everything|coming|tonight|today))\b",
    re.I)
# A parting proper. "I love you" or "I have to go" also come in the middle of a scene; an episode needs one of these.
PARTING = re.compile(r"\b(good ?-?bye|bye|bye-bye|good ?-?night|(?<!to )see you(?!r)|see ya|farewell|"
                     r"take care(?! of)|sweet dreams|sleep well|talk to you (?:later|tomorrow|soon))\b", re.I)

_LABEL_LINE = re.compile(r"^\s*([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2}):\s", re.M)
_DOC_MARK = re.compile(
    r"^\s*(?:-{2,}|_{3,}|={3,}|\*\s*\*\s*\*|#|the end\b|end\b|fin\b|\(?end\)?\.?$|posted\b|source\b|category\b|"
    r"tags?\b|share\b|comments?\b|related\b|read more\b|copyright\b|©|chapter\b|episode\b|act [ivx\d]|scene \d|"
    r"written by\b|directed by\b|credits\b|references\b|note\b|disclaimer\b|transcript\b|"
    r"\[\s*(?:\*|footnote|note|source|editor|end\b|the end\b))", re.I)
_CAST_LINE = re.compile(r"^[A-Z][\w'.-]+(?: [A-Z][\w'.-]+){0,3}, (?:an? |the |\w+-\w+|twenty|thirty|forty|fifty|"
                        r"sixty|seventy|eighty|ninety|\d)")
TAIL_NARRATION_WORDS = 100


@dataclass
class SceneConfig:
    truncate_loops: bool = False
    frame_gate: bool = False
    cut_closings: float = 0.0  # 0 = off; else the end-of-text candidate share that confirms a closing episode
    clean_tail: bool = False
    max_template: float = 1.0  # drop a scene whose own-opening template share exceeds this
    untrain_repeats: int = 0
    untrain_format: bool = False
    drop_minor_explicit: bool = False
    blocklist: frozenset[str] = frozenset()
    redact_pii: bool = False
    min_scene_turns: int = 1
    min_removed: int = 4
    cut_unended: int = 0  # > 0: a scene without a model-written ending keeps at most this many turns


@dataclass
class NodeIndex:
    """What assembly reads from a loom run's ``<out>/nodes.jsonl``: per parent node, (candidates drawn, of which
    end-of-text), and the end-of-text candidates themselves."""

    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    eos: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, prefix: str = "") -> "NodeIndex":
        idx = cls()
        idx.extend(path, prefix)
        return idx

    def extend(self, path: Path, prefix: str = "") -> None:
        """Add a run's nodes; ``prefix`` namespaces its node ids as ``loom assemble`` namespaces paths by run."""
        cnt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                par = prefix + r["parent"]
                c = cnt[par]
                c[0] += 1
                if r.get("ended_by") == "eos":
                    c[1] += 1
                    self.eos.setdefault(par, []).append(
                        {k: r.get(k) for k in ("id", "speaker", "text", "vetoes", "status")})
        self.counts.update({k: (v[0], v[1]) for k, v in cnt.items()})

    def eos_share(self, node: str) -> Optional[float]:
        c = self.counts.get(node)
        return c[1] / c[0] if c and c[0] else None


def is_farewell(text: str, cast: list[str]) -> bool:
    """A short closing turn: its speech (narration paragraphs aside) names a farewell or a parting and runs at most
    20 words."""
    sp = spoken_text(text, cast)
    return bool(FAREWELL.search(sp)) and len(sp.split()) <= 20


def closing_cut(body: list[dict[str, Any]], cast: list[str], header: str, eos_share: list[Optional[float]],
                has_ending: list[bool], eos_min: float, min_removed: int = 4, start: int = 5,
                ending_farewell: Optional[list[bool]] = None) -> Optional[tuple[str, int]]:
    """Where a scene's story has closed: ``("episode" | "orbit", j)`` = keep turns ``0..j``; None when it has not.

    ``eos_share[k]`` is the end-of-text share among the candidates drawn after turn ``k`` (None if unknown),
    ``has_ending[k]`` whether a usable end-of-text candidate was drawn there and ``ending_farewell[k]`` whether that
    candidate is itself a farewell. An episode cut needs a farewell ending: a cut scene that stops on an ordinary
    line teaches stopping mid-thought."""
    n = len(body)
    fw = [is_farewell(t["text"], cast) for t in body]
    parting = [f and bool(PARTING.search(spoken_text(t["text"], cast))) for f, t in zip(fw, body)]
    efw = ending_farewell or [True] * n
    for k in range(start, n):
        lo, hi = max(0, k - 3), k + 4
        if not fw[k] or sum(fw[lo:hi]) < 2 or not any(parting[lo:hi]):
            continue
        best: Optional[tuple[int, float]] = None
        for j in range(max(start + 1, k - 1), min(n - 1, k + 4)):
            e = eos_share[j] or 0.0
            if e >= eos_min and has_ending[j] and efw[j] and (best is None or e > best[1]):
                best = (j, e)
        if best is not None and n - (best[0] + 2) >= min_removed:
            return "episode", best[0]
    prior = [Turn("", header)]
    new = []
    for t in body:
        new.append(novelty(t["text"], prior, window=31, speech_only=True, cast=cast, count=True))
        prior.append(Turn(t["speaker"], t["text"]))
    for g in range(start + 1, n - 8):
        rest = range(g, n)
        if fw[g] and sum(fw[i] for i in rest) >= 0.4 * len(rest) and statistics.fmean(new[i] for i in rest) <= 2.0:
            best = None
            for j in range(max(start + 1, g - 1), min(n - 1, g + 4)):
                e = eos_share[j] or 0.0
                if e > 0 and has_ending[j] and efw[j] and (best is None or e > best[1]):
                    best = (j, e)
            return "orbit", best[0] if best else g
    return None


def clean_tail(text: str, cast: list[str]) -> tuple[str, Optional[str]]:
    """A final turn cut at its first paragraph that has left the scene: (text, reason or None)."""
    paras = split_turn(text)
    if len(paras) <= 1:
        return text, None
    keep = paras[:1]
    names = {c.strip() for c in cast}
    narr_words = 0
    for par in paras[1:]:
        letters = [c for c in par if c.isalpha()]
        why = None
        if out_of_frame(par):
            why = "out_of_frame"
        elif _DOC_MARK.match(par):
            why = "doc_marker"
        elif any(m.group(1) not in names for m in _LABEL_LINE.finditer(par)):
            why = "foreign_label"
        elif _CAST_LINE.match(par):
            why = "cast_list"
        elif letters and sum(c.isascii() for c in letters) / len(letters) < 0.9:
            why = "script"
        elif narr_words + len(par.split()) > TAIL_NARRATION_WORDS and narr_words > 0:
            why = "long_narration"
        if why:
            return "\n\n".join(keep), why
        narr_words += len(par.split())
        keep.append(par)
    return "\n\n".join(keep), None


def _speech_in_frame(text: str, speaker: str) -> bool:
    paras = split_turn(text)
    return bool(paras) and not out_of_frame(paras[0]) and not self_described(text, speaker)


def pick_ending(cands: list[dict[str, Any]], speaker: str, cast: list[str], rng: random.Random
                ) -> Optional[dict[str, Any]]:
    """The end-of-text candidate that closes a cut scene: one drawn for the next speaker, not a self-copy, overlong
    or echo veto (repeating the farewell just said is a fine last line), in frame, preferring farewells."""
    ok = [c for c in cands if c.get("speaker") == speaker and c.get("text", "").strip()
          and not set(c.get("vetoes") or []) & {"selfcopy", "overlong", "echo"}
          and not out_of_frame(split_turn(c["text"])[0]) and not self_described(c["text"], speaker)]
    if not ok:
        return None
    fare = [c for c in ok if is_farewell(c["text"], cast)]
    return rng.choice(sorted(fare or ok, key=lambda c: c["id"]))


def _next_speaker(cast: list[str], last: str) -> str:
    return cast[(cast.index(last) + 1) % len(cast)] if last in cast else (cast[0] if cast else "")


def clean_scene(p: dict[str, Any], nodes: Optional[NodeIndex], cfg: SceneConfig,
                uses: dict[str, int]) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """One loom path -> (scene row or None when dropped, a record of what cleaning did). ``uses`` counts how often
    each node id has trained so far; a turn trains only on its first use."""
    cast = list(p.get("cast", []))
    header = p.get("header", "")
    # the seed turns and any history an earlier run generated are context; this run's turns are the scene
    n_ctx = len(p["turns"]) - p["depth"]
    prefix = [dict(t) for t in p["turns"][:n_ctx]]
    body = [dict(t) for t in p["turns"][n_ctx:]]
    ids = node_ids(p["leaf"])
    if len(ids) != len(body):
        raise ValueError(f"path {p['leaf']}: {len(ids)} node ids for {len(body)} generated turns")
    rec: dict[str, Any] = {"leaf": p["leaf"], "seed": p.get("seed", ""), "state": p["state"], "turns_in": len(body)}
    ended = p["state"] == "eos"
    if cfg.truncate_loops:
        o = loop_onset(body)
        if o < len(body):
            body, ids, ended = body[:o], ids[:o], False
            rec["loop_cut"] = o
    if cfg.frame_gate:
        for k, t in enumerate(body):
            if out_of_frame(t["text"]) or self_described(t["text"], t["speaker"]):
                if ended and k == len(body) - 1 and cfg.clean_tail and _speech_in_frame(t["text"], t["speaker"]):
                    break  # a natural ending whose closing paragraphs leave the scene: the tail cut handles it
                body, ids, ended = body[:k], ids[:k], False
                rec["frame_cut"] = k
                break
    if cfg.cut_closings > 0 and nodes is not None and len(body) > 1:
        shares = [nodes.eos_share(i) for i in ids]
        endings = [pick_ending(nodes.eos.get(i, []), _next_speaker(cast, body[k]["speaker"]), cast,
                               random.Random(i)) for k, i in enumerate(ids)]
        cut = closing_cut(body, cast, header, shares, [e is not None for e in endings], cfg.cut_closings,
                          cfg.min_removed,
                          ending_farewell=[e is not None and is_farewell(e["text"], cast) for e in endings])
        if cut is not None:
            kind, j = cut
            end = endings[j] if endings[j] is not None and is_farewell(endings[j]["text"], cast) else None
            rec["closing"] = {"kind": kind, "after": j, "removed": len(body) - j - 1,
                              "eos_share": shares[j], "ending": end is not None}
            body, ids = body[:j + 1], ids[:j + 1]
            if end is not None:
                body.append({"speaker": end["speaker"], "text": end["text"]})
                ids.append(end["id"])
                ended = True
            else:
                ended = False
    if cfg.clean_tail and ended and body:
        text, why = clean_tail(body[-1]["text"], cast)
        if why:
            body[-1] = dict(body[-1], text=text)
            rec["tail_cut"] = why
        if cfg.frame_gate and (out_of_frame(body[-1]["text"]) or self_described(body[-1]["text"], body[-1]["speaker"])):
            body, ids, ended = body[:-1], ids[:-1], False  # still out of frame after the tail cut
            rec["frame_cut"] = len(body)
    if cfg.cut_unended and not ended and len(body) > cfg.cut_unended:
        body, ids = body[:cfg.cut_unended], ids[:cfg.cut_unended]
        rec["unended_cut"] = cfg.cut_unended
    rec["turns_out"] = len(body)
    rec["ended"] = ended
    if len(body) < cfg.min_scene_turns:
        rec["drop"] = "short"
        return None, rec
    prior = [Turn(t["speaker"], t["text"]) for t in prefix]
    tmpl = []
    for t in body:
        tmpl.append(own_opening_repeat(t["text"], t["speaker"], prior))
        prior.append(Turn(t["speaker"], t["text"]))
    rec["template"] = round(sum(tmpl) / len(tmpl), 4)
    if rec["template"] > cfg.max_template:
        rec["drop"] = "template"
        return None, rec
    texts = [t["text"] for t in prefix + body]
    if cfg.drop_minor_explicit and any("minor_explicit" in compliance.classify(x) for x in texts):
        rec["drop"] = "minor_explicit"
        return None, rec
    if cfg.blocklist and any(compliance.turn_hash(x) in cfg.blocklist for x in texts):
        rec["drop"] = "blocklist"
        return None, rec
    flawed = _flawed(body, cast, Clean(untrain_repeats=cfg.untrain_repeats, untrain_format=cfg.untrain_format))
    if ended and body and (len(body) - 1) in flawed and \
            (len(body) - 1) not in _flawed(body, cast, Clean(untrain_format=cfg.untrain_format)):
        # a model-written ending that repeats a line ("Good night." / "Good night.") still closes the scene: it
        # trains, so its end-of-text does
        flawed.discard(len(body) - 1)
    rec["untrained_flawed"] = len(flawed)
    turns = [dict(t, train=False) for t in prefix]
    shared = 0
    for k, (t, nid) in enumerate(zip(body, ids)):
        train = k not in flawed and uses.get(nid, 0) < 1
        if uses.get(nid, 0) >= 1:
            shared += 1
        uses[nid] = uses.get(nid, 0) + 1
        turns.append({"speaker": t["speaker"], "text": t["text"], "train": train})
    rec["untrained_shared"] = shared
    if cfg.redact_pii:
        red = [compliance.redact_pii(t["text"]) for t in turns]
        rec["pii_redacted"] = sum(a != t["text"] for a, t in zip(red, turns))
        turns = [dict(t, text=x) for t, x in zip(turns, red)]
    return {"header": header, "turns": turns, "ended": ended,
            "source": {"tree": p["tree"], "leaf": p["leaf"], "seed": p.get("seed", ""), "state": p["state"]}}, rec


def seed_key(seed: str) -> str:
    """The seed a tree grew from: ``p1-00042~t2`` -> ``p1-00042`` (trees ``~t<k>`` repeat one seed)."""
    return seed.split("~", 1)[0]


def scene_chars(s: dict[str, Any]) -> int:
    return len(s.get("header", "")) + 2 + sum(len(t["speaker"]) + len(t["text"]) + 4 for t in s["turns"])


def stack(scenes: list[dict[str, Any]], max_chars: int, seed: int) -> list[dict[str, Any]]:
    """Documents of whole scenes, up to ``max_chars`` each (a longer scene stands alone), never two scenes of one
    seed in a document. Scenes are shuffled, then each goes into the first open document it fits."""
    order = list(range(len(scenes)))
    random.Random(seed).shuffle(order)
    docs: list[dict[str, Any]] = []
    room: list[int] = []
    seeds: list[set[str]] = []
    for i in order:
        s = scenes[i]
        c, k = scene_chars(s), seed_key(s["source"].get("seed", ""))
        for d in range(max(0, len(docs) - 64), len(docs)):  # a bounded look-back keeps packing linear
            if room[d] >= c and k not in seeds[d]:
                docs[d]["scenes"].append(s)
                room[d] -= c
                seeds[d].add(k)
                break
        else:
            docs.append({"scenes": [s]})
            room.append(max_chars - c)
            seeds.append({k})
    return docs


def assemble_scenes(paths: list[dict[str, Any]], nodes: Optional[NodeIndex], cfg: SceneConfig, max_chars: int,
                    seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """(documents, cleaned scenes, per-scene cleaning records) from loom paths."""
    uses: dict[str, int] = {}
    scenes: list[dict[str, Any]] = []
    recs: list[dict[str, Any]] = []
    for p in sorted((p for p in paths if p["state"] in ENDED_STATES), key=lambda p: p["leaf"]):
        s, rec = clean_scene(p, nodes, cfg, uses)
        recs.append(rec)
        if s is not None:
            scenes.append(s)
    drops = Counter(r.get("drop", "kept") for r in recs)
    logger.info("%d scenes -> %s", len(recs), dict(drops))
    return stack(scenes, max_chars, seed), scenes, recs


def holdout_seeds(seeds: list[str], frac: float, salt: str = "holdout") -> set[str]:
    """The seed keys (``seed_key``) held out for evaluation: those whose sha256(salt + key) falls in the lowest
    ``frac`` of the hash range. Deterministic, independent of order, and a seed's every tree goes the same way."""
    import hashlib

    keys = {seed_key(s) for s in seeds}
    cut = int(frac * 2 ** 32)
    return {k for k in keys if int(hashlib.sha256(f"{salt}:{k}".encode()).hexdigest()[:8], 16) < cut}
