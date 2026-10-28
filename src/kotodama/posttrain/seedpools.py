"""Model-written seed pools, crossed into combinations, rendered as scene headers by the model itself.

The ``python -m kotodama.posttrain.seeds`` subcommands ``pools``, ``combine``, ``headers`` and ``report`` live here:

``pools``     For each axis (characters, themes, needs, relationships, settings, inciting situations, stakes, register,
              names) the base model continues a list document, one entry per line. On a crossed axis every line opens
              with a pair of items, ``(a / b) entry``, and a request's open line carries a pair drawn at random from
              the ``--crossing_dir`` lists (Gunkel's ideonomic method: two unrelated items, and the model writes the
              concrete thing they suggest; https://ideonomy.mit.edu). Lines are normalised; malformed, mojibake, too short, too long and duplicate
              lines are rejected; then a greedy within-pool dissimilarity filter (character n-gram TF-IDF cosine)
              drops near-paraphrases of entries already kept. Every rejection is banked with its reason.
``combine``   Samples combinations with spread: each slot prefers entries used least so far, and a combination that
              shares too many entries with an accepted one, or whose rendered text is too close to one, is rejected.
              Each carries balanced tags: ``friction`` (four kinds), ``secret_arm`` (plain | hinted) and ``cast_arm``
              (pair | cast of 3-4, pair the majority). Every character is a speaker.
``headers``   The model writes, per combination, a want for each character and the secret one character holds (as
              notes), then the opening page: a cast list with short character notes, a relationship line, and a setting
              paragraph carrying the inciting situation and the clock. It continues a few-shot playwright's notebook.
              A coherence gate rejects pages by rule (a missing name or want, a speaker tag inside the page, leaked
              notebook labels, a character described by another's entry, impossible ages, garbling, copying an
              example), then scores the survivors' mean token NLL through ``/score_tokens`` and cuts the worst tail.
``report``    Pool sizes through each filter, rejection reasons, entry reuse, nearest-neighbour similarity of the
              headers, arm balance, the NLL cut; and a markdown reading sample of random seeds.

Every request is plain decoding through ``/v1/completions`` (``kotodama.posttrain.selfplay.oai_completer``).
Output rows of ``headers`` are loom seeds (``kotodama.posttrain.selfplay.parse_seed`` reads them unchanged):
``{id, cast, header, combo, friction, cast_arm, secret_arm, characters: [{name, label, character, need, want}],
secret: {holder, target, text}, nll, ...}``.

    python -m kotodama.posttrain.seeds pools   --endpoint URL --out DIR [--per_pool 300] [--crossing_dir LISTS]
    python -m kotodama.posttrain.seeds combine --out DIR [--n 3000]
    python -m kotodama.posttrain.seeds headers --endpoint URL --out DIR
    python -m kotodama.posttrain.seeds report  --out DIR [--sample 40]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

Complete = Callable[[str, int, list[str]], Awaitable[list[tuple[str, str, bool]]]]
"""(prompt, n, stops) -> [(text, finish_reason, hit max_tokens)], as ``kotodama.posttrain.selfplay.oai_completer``."""
Score = Callable[[list[tuple[str, str]]], Awaitable[list[float]]]
"""[(prompt, completion)] -> mean per-token log-prob of each completion, as ``kotodama.posttrain.loom.http_scorer``."""

DOC_START = "<|endoftext|>"  # a header opens a document, so its NLL is read after the end-of-text token


# ── axes ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Axis:
    """One pool: the list document's opening line, default example entries, the crossing lists (names of files in
    ``--crossing_dir``; two names = one item from each, one name = two distinct items from it), and length bounds.
    A crossed entry is written ``(a / b) entry``: the pair opens every line, examples included, so each line is
    anchored to its own pair. ``prefix``: one item instead (one side of it when it reads ``x and y``) opens the entry
    itself and the model writes the rest ("thresholds" -> "thresholds and the things that live on them")."""
    name: str
    head: str
    examples: tuple[str, ...]
    crossing: tuple[str, ...] = ()
    prefix: bool = False
    min_chars: int = 20
    max_chars: int = 160
    sim_threshold: float = 0.5


AXES: dict[str, Axis] = {a.name: a for a in (
    Axis("character",
         "Characters for short plays: real people, each a role and one defining trait, written in a few words.",
         ("(fickle / dutiful) a hospice volunteer who changes her mind about everything except showing up",
          "(gruff / sentimental) a long-haul trucker who keeps a sketchbook of the birds he sees at rest stops",
          "(ambitious / forgetful) a junior architect who has promised the same weekend to three different people",
          "(pious / mischievous) a retired choir director who still rigs the parish raffle so the children win",
          "(meticulous / lonely) a night-shift baggage handler who catalogues what travellers leave behind",
          "(modest / resentful) a sous-chef who has quietly run the kitchen for years under someone else's name"),
         crossing=("traits", "traits")),
    Axis("theme",
         "Themes for short plays: an idea crossed with a longing or an illusion, written as a phrase.",
         ("small kindnesses and the belief that they end where they land",
          "owing a debt and the illusion of standing alone",
          "inheritance and the hope that children teach their parents",
          "waiting, and the dream of a day when there is finally enough time",
          "confession and the illusion of knowing why we do things",
          "maps and the longing for a place that stays where it is drawn"),
         crossing=("antisyzygies",), prefix=True, min_chars=15, max_chars=120),
    Axis("need",
         "What people need underneath what they ask for: a dream, a fear or a lack, written as a phrase.",
         ("(devotion / aimlessness) to matter to someone without having to be useful to them",
          "(pride / failure) to be forgiven for a failure no one else remembers",
          "(fear / disease, illness, or infirmity) to believe the body can still be trusted",
          "(curiosity / dogmatism) to find one question that does not already have an owner",
          "(gentleness / brutality) to be held without being handled",
          "(anguish / denial of rights) to stop waiting for permission from the dead"),
         crossing=("emotions", "bads"), min_chars=12, max_chars=110),
    Axis("relationship_pair",
         "How two people in a play are bound to each other, written as a phrase.",
         ("(teacher and pupil / reunion) a piano teacher and the former student who now out-earns her",
          "(giving and receiving / purchase of house) two neighbours who share a driveway and have never agreed where it ends",
          "(friend and enemy / first date) best friends since school who have both asked out the same man",
          "(master and slave / retirement) a retiring head chef and the sous-chef who has secretly run the kitchen for years",
          "(old and new / birth of a child) a widowed father and his daughter-in-law, a week after the baby came",
          "(presence and absence / working the night shift) two night-shift workers who have never had a full conversation"),
         crossing=("antisyzygies", "moments")),
    Axis("relationship_cast",
         "How three or four people in a play are bound to each other, written as a phrase.",
         ("(part and whole / getting married) three sisters and the eldest's fiancé, on the night before the wedding",
          "(judge and judged / death of father) a mother, her two grown sons, and the lawyer reading their father's will",
          "(success and failure / being fired) a band's drummer, its singer, and the manager who has just dropped them",
          "(coming and going / moving) four housemates dividing up the furniture of a flat they are all leaving",
          "(old and new / failing an inspection) a dairy farmer, her teenage son, and the inspector who went to school with her",
          "(giving and receiving / wedding reception) a bride's father, the groom's mother, and the caterer they both owe money"),
         crossing=("antisyzygies", "moments"), max_chars=180),
    Axis("setting",
         "Settings for short plays: a place, a time, and one object in it that matters.",
         ("(anticipation / vacationing) a ferry's car deck at dawn, engines idling; a cooler with a broken lid",
          "(bitterness / moving) an emptied apartment in late afternoon; the one chair nobody took",
          "(awe / aurora observed) a weather station on a ridge at midnight in February; a logbook missing three pages",
          "(boredom / drunk on job) a hotel night desk at three in the morning; a master key card that has gone missing",
          "(nostalgia / high school reunion) a school gym after the reunion dance; a framed photo face-down on the bleachers",
          "(anxiety / birth of a child) a hospital car park in November rain; a car seat still in its box"),
         crossing=("emotions", "moments")),
    Axis("inciting",
         "Inciting moments for short plays: something that has just happened, or is about to.",
         ("(reunion / death of a friend) a letter has arrived for someone who died last spring",
          "(purchase of house / being crass) the buyer has just found out the seller still lives in the garden shed",
          "(getting married / cold feet) the groom has walked out of the rehearsal and will not say why",
          "(wedding / power failure) the power has gone out and the freezer full of wedding food is warming",
          "(finding a lost dog / meeting a neighbour) a stranger has knocked, holding a dog that belongs to one of them",
          "(birth of a child / suspicion) the results of the paternity test are in an envelope on the table"),
         crossing=("moments", "moments")),
    Axis("stakes",
         "Stakes for short plays: what must happen before the scene ends, and the clock that runs out.",
         ("(failure / excessive haste) the car must be packed and gone before the landlord arrives at noon",
          "(distrust / disaster) someone has to decide whether to call the ambulance before the snow closes the road",
          "(cheating / fear of exposure) the stolen exam papers must be back in the office before the janitor's ten o'clock round",
          "(family disintegration / denial of rights) one of them must sign the form before the court office closes at five",
          "(complacency / disaster) they have until the tide turns to get the boat off the sandbar",
          "(frustration / excessive haste) the speech has to be finished before the car comes for them at seven"),
         crossing=("bads", "bads")),
    Axis("register",
         "The register of a short play: its genre and the way its people talk, written as a phrase.",
         ("(amusement / anguish) a kitchen-sink comedy that turns, halfway, into a quarrel nobody can laugh off",
          "(calm / caution) a quiet two-hander, slow and plain, the way people talk in hospitals at night",
          "(defiance / gaiety) a screwball farce, fast overlapping talk, everyone lying a little",
          "(dread / curiosity) a radio drama, terse, all sound and implication",
          "(irony / dignity) a dry, deadpan comedy of manners",
          "(fear / boredom) a ghost story told in daylight by people too tired to be scared"),
         crossing=("emotions", "emotions"), min_chars=12, max_chars=130),
    Axis("names",
         "Names from the guest list of an international conference, one full name per line.",
         ("Ines Moreau", "Kwame Asante", "Dolores Vega", "Bram Visser", "Ruth Callaghan", "Sunil Mehta",
          "Agnieszka Nowak", "Haruto Sato", "Thandiwe Dlamini", "Matteo Russo"),
         min_chars=5, max_chars=32),
)}

FRICTIONS = ("collision", "cooperation-with-incompatible-methods", "shared-problem-unequal-information",
             "compatible-wants-competing-needs")
FRICTION_TEXT = {
    "collision": "their wants collide",
    "cooperation-with-incompatible-methods": "they must work together but go about it in incompatible ways",
    "shared-problem-unequal-information": "they share one problem but not the same information",
    "compatible-wants-competing-needs": "their wants fit together but their needs compete",
}
SECRET_ARMS = ("plain", "hinted")
SECRET_TEXT = {"plain": "the secret is stated plainly on the page",
               "hinted": "the secret is only hinted at on the page, never stated"}
CAST_ARMS = ("pair", "cast")


# ── text hygiene ─────────────────────────────────────────────────────────────

_MOJIBAKE = re.compile(r"[\ufffd\u0080-\u009f]|Ã.|â€|Â\s")
_META_LINE = re.compile(r"https?:|www\.|\bone per line\b|\bsparked by\b|[#*_=|<>{}\[\]]", re.I)
_LABEL_COLON = re.compile(r"(?<!\d):|:(?!\d)")
_CLEAN_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NAME_TOKEN = re.compile(r"^(?:[A-Z][a-z]{0,2}\.|[A-ZÀ-Þ][\w'’-]*)$")
_TITLES = frozenset({"Mr.", "Mrs.", "Ms.", "Dr.", "Mx.", "Fr.", "Sr."})


def norm_key(text: str) -> str:
    """Duplicate key: lowercase letters and digits only, single spaces."""
    return " ".join(re.sub(r"[^\w ]+", " ", text.lower()).split())


def mojibake(text: str) -> bool:
    """Replacement characters, control characters other than newline and tab, UTF-8 decoded as a single-byte
    charset, or letters outside Latin script."""
    if _MOJIBAKE.search(text) or any(unicodedata.category(c) in ("Cc", "Co", "Cs") and c not in "\n\t"
                                     for c in text):
        return True
    return any(c.isalpha() and ord(c) > 0x24F and not 0x1E00 <= ord(c) <= 0x1EFF for c in text)


def clean_entry(line: str) -> str:
    """A list line -> its entry: leading bullet or number removed, whitespace collapsed, trailing period dropped."""
    s = " ".join(_CLEAN_LEAD.sub("", line).split())
    s = s.strip(" \"'“”").rstrip(";,").rstrip()
    return s[:-1].rstrip() if s.endswith(".") and not s.endswith("..") else s


def check_entry(text: str, axis: Axis) -> Optional[str]:
    """Rejection reason for one cleaned entry, or None when it may join the pool."""
    if not text or not re.search(r"[A-Za-z]", text):
        return "malformed"
    if mojibake(text):
        return "mojibake"
    if len(text) < axis.min_chars:
        return "too_short"
    if len(text) > axis.max_chars:
        return "too_long"
    if axis.name == "names":
        if re.search(r"\b(?:Doe|Bloggs|Smith Smith)\b", text):
            return "placeholder"
        return None if name_label(text) else "malformed"
    if _META_LINE.search(text) or _LABEL_COLON.search(text) or text.count("(") != text.count(")"):
        return "malformed"
    words = text.lower().split()
    if len(words) < 3:
        return "too_short"
    if len(set(words)) < 0.6 * len(words):
        return "repetitive"
    return None


def name_label(name: str) -> Optional[str]:
    """The speaker label of a full name: the first name, or title + surname ("Mrs. Okafor"). None if ``name`` is not
    2-3 capitalised tokens (a title counts as one)."""
    toks = name.split()
    if not 2 <= len(toks) <= 3 or not all(_NAME_TOKEN.match(t) for t in toks):
        return None
    if toks[0] in _TITLES:
        return f"{toks[0]} {toks[-1]}"
    if toks[0].endswith(".") or len(toks[0]) < 2:
        return None
    return toks[0]


# ── similarity ───────────────────────────────────────────────────────────────


def tfidf(texts: Sequence[str], ngrams: tuple[int, ...] = (3, 4, 5)) -> Any:
    """L2-normalised character n-gram TF-IDF rows (scipy CSR), sublinear tf, smoothed idf; words padded by spaces."""
    import numpy as np
    from scipy import sparse

    vocab: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        s = f" {' '.join(norm_key(t).split())} "
        counts = collections.Counter(s[j:j + n] for n in ngrams for j in range(len(s) - n + 1))
        for g, c in counts.items():
            rows.append(i)
            cols.append(vocab.setdefault(g, len(vocab)))
            vals.append(1.0 + math.log(c))
    m = sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(texts), max(1, len(vocab))))
    df = np.bincount(m.indices, minlength=m.shape[1])
    idf = np.log((1 + len(texts)) / (1 + df)) + 1.0
    m = m.multiply(idf.astype(np.float32)).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms.astype(np.float32)) @ m


def greedy_dissimilar(texts: Sequence[str], threshold: float) -> tuple[list[int], dict[int, tuple[int, float]]]:
    """Walk ``texts`` in order, keeping one when its cosine to every kept text is below ``threshold``.
    Returns (kept indices, {dropped index: (nearest kept index, similarity)})."""
    if not texts:
        return [], {}
    import numpy as np

    x = tfidf(texts)
    sims = (x @ x.T).toarray()
    kept: list[int] = []
    dropped: dict[int, tuple[int, float]] = {}
    for i in range(len(texts)):
        if kept:
            row = sims[i, kept]
            j = int(np.argmax(row))
            if row[j] >= threshold:
                dropped[i] = (kept[j], float(row[j]))
                continue
        kept.append(i)
    return kept, dropped


def nearest_neighbour_sims(texts: Sequence[str]) -> list[float]:
    """Each text's highest cosine to any other text (character n-gram TF-IDF)."""
    if len(texts) < 2:
        return []
    import numpy as np

    x = tfidf(texts)
    s = (x @ x.T).toarray()
    np.fill_diagonal(s, -1.0)
    return s.max(axis=1).tolist()


def quantiles(xs: Sequence[float], qs: Sequence[float] = (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    return {f"q{int(q * 100)}": round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 4) for q in qs}


# ── stage 1: pools ───────────────────────────────────────────────────────────


def load_list(path: Path) -> list[str]:
    """One item per line; blank lines and lines starting with ``===`` or ``#`` are skipped; a leading bullet goes."""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip() and not ln.lstrip().startswith(("===", "#")):
            out.append(clean_entry(ln))
    return [x for x in out if x]


_PAIR = re.compile(r"^\(([^()/\n]{1,80}?) / ([^()/\n]{1,80}?)\)\s*")


def split_pair(line: str) -> tuple[Optional[tuple[str, str]], str]:
    """``"(a / b) entry"`` -> ((a, b), entry); a line without a leading pair -> (None, line)."""
    m = _PAIR.match(line)
    if not m:
        return None, line
    return (m.group(1).strip(), m.group(2).strip()), line[m.end():]


def pool_prompt(axis: Axis, examples: Sequence[str], crossing: Optional[tuple[str, ...]]) -> str:
    """The list document: the axis's opening line, the example entries as a bulleted list, then an open entry. A
    pair crossing opens the entry as ``(a / b)``; a prefix crossing opens it with the item itself."""
    body = "".join(f"- {e}\n" for e in examples)
    if not crossing:
        return f"{axis.head}\n\n{body}-"
    if axis.prefix:
        return f"{axis.head}\n\n{body}- {crossing[0].lower()}"
    return f"{axis.head}\n\n{body}- ({crossing[0].lower()} / {crossing[1].lower()})"


def parse_list_completion(text: str, hit_cap: bool,
                          lead: str = "") -> list[tuple[Optional[tuple[str, str]], str]]:
    """Entries of a completion that continues an open ``-`` bullet: the first line (after ``lead``, the text the
    prompt already opened it with), then each following line that starts with ``-``, up to the first line that does
    not; each as (its own leading pair or None, entry). A final line cut by the token cap is dropped."""
    lines = (lead + text).split("\n")
    if hit_cap and lines:
        lines = lines[:-1]
    out: list[tuple[Optional[tuple[str, str]], str]] = []
    for i, ln in enumerate(lines):
        if i > 0 and not ln.lstrip().startswith("-"):
            break
        pair, rest = split_pair(_CLEAN_LEAD.sub("", ln).strip()) if i > 0 else (None, ln)
        out.append((pair, clean_entry(rest)))
    return out


def draw_crossing(rng: random.Random, axis: Axis, lists: dict[str, list[str]]) -> Optional[tuple[str, ...]]:
    """A fresh crossing for one request: a pair (``Axis.crossing``) or, on a prefix axis, one item (one side of an
    ``x and y`` item). None when a named list is missing."""
    names = [n for n in axis.crossing if lists.get(n)]
    if len(names) != len(axis.crossing) or not names:
        return None
    if axis.prefix:
        item = rng.choice(lists[names[0]])
        sides = item.split(" and ")
        return (rng.choice(sides).strip(),) if len(sides) == 2 else (item,)
    if len(names) == 1 or names[0] == names[1]:
        a, b = rng.sample(lists[names[0]], 2)
        return a, b
    return rng.choice(lists[names[0]]), rng.choice(lists[names[1]])


@dataclass
class PoolState:
    """Every line drawn for one axis, in arrival order, with its crossing and its first rejection reason."""
    axis: str
    lines: list[dict[str, Any]] = field(default_factory=list)
    requests: int = 0


def filter_pool(state: PoolState, axis: Axis, threshold: float, names_extra: Sequence[str] = ()) -> dict[str, Any]:
    """Run the filters over all lines drawn so far (by line index within a completion, then arrival). Returns {kept: [rows], rejects: [rows],
    counts: {stage: n}}; a row is {text, crossing, reason?, nearest?, sim?}."""
    rejects: list[dict[str, Any]] = []
    seen: set[str] = set()
    labels: set[str] = set()
    candidates: list[dict[str, Any]] = []
    counts = {"drawn": len(state.lines)}
    # first lines of completions come first: they sit right after their crossing, and later lines drift
    ordered = sorted(state.lines, key=lambda r: r.get("line", 0))
    rows = [{"text": t, "crossing": None, "source": "list"} for t in names_extra] + ordered
    for row in rows:
        reason = check_entry(row["text"], axis)
        if reason is None:
            key = norm_key(row["text"])
            if key in seen:
                reason = "duplicate"
            elif axis.name == "names":
                label = name_label(row["text"])
                if label is None or label.lower() in labels:
                    reason = "duplicate_label"
                else:
                    labels.add(label.lower())
            seen.add(key)
        if reason is None:
            candidates.append(row)
        else:
            rejects.append({**row, "reason": reason})
    counts["after_rules"] = len(candidates)
    if axis.name == "names":
        kept = candidates
    else:
        keep_idx, dropped = greedy_dissimilar([r["text"] for r in candidates], threshold)
        kept = [candidates[i] for i in keep_idx]
        for i, (j, s) in sorted(dropped.items()):
            rejects.append({**candidates[i], "reason": "similar", "nearest": candidates[j]["text"], "sim": round(s, 4)})
    counts["after_similarity"] = len(kept)
    return {"kept": kept, "rejects": rejects, "counts": counts}


async def grow_pool(complete: Complete, axis: Axis, target: int, lists: dict[str, list[str]],
                    examples: Sequence[str], rng: random.Random, threshold: float, n_per_request: int = 8,
                    parallel: int = 16, max_rounds: int = 40, names_extra: Sequence[str] = ()) -> dict[str, Any]:
    """Draw lines for ``axis`` in rounds of ``parallel`` requests until the filtered pool reaches ``target`` (or
    ``max_rounds``). Each request shows a random subset of the examples (four, shuffled) and, on a crossed axis, a
    fresh crossing. A line after the first carries the crossing the model wrote for it (``model-continued``)."""
    state = PoolState(axis.name)
    res = filter_pool(state, axis, threshold, names_extra)
    stops = ["\n\n"]
    for rnd in range(max_rounds):
        if len(res["kept"]) >= target:
            break
        prompts = []
        for _ in range(parallel):
            ex = rng.sample(list(examples), min(4, len(examples)))
            crossing = draw_crossing(rng, axis, lists) if axis.crossing else None
            prompts.append((pool_prompt(axis, ex, crossing), crossing))
        outs = await asyncio.gather(*(complete(p, n_per_request, stops) for p, _ in prompts))
        for (_p, crossing), raw in zip(prompts, outs):
            state.requests += 1
            for text, _finish, hit in raw:
                lead = crossing[0].lower() if crossing and axis.prefix else ""
                for k, (own, entry) in enumerate(parse_list_completion(text, hit, lead)):
                    cross = list(crossing) if k == 0 and crossing else list(own) if own else None
                    state.lines.append({"text": entry, "crossing": cross,
                                        "source": "model" if k == 0 else "model-continued", "line": k})
        res = filter_pool(state, axis, threshold, names_extra)
        logger.info("pool %s round %d: %d requests, %d lines, %d kept", axis.name, rnd + 1, state.requests,
                    len(state.lines), len(res["kept"]))
    res["kept"] = res["kept"][:target]
    res["counts"]["final"] = len(res["kept"])
    res["counts"]["requests"] = state.requests
    res["counts"]["threshold"] = threshold
    return res


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file ([] when absent). A final line cut mid-write (a client killed while appending) is
    skipped with a warning, so a rerun resumes; a bad line anywhere else raises."""
    if not path.exists():
        return []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for i, x in enumerate(lines):
        try:
            rows.append(json.loads(x))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            logger.warning("%s: last line is incomplete; skipped", path)
    return rows


async def run_pools(complete: Complete, out: Path, axes: Sequence[str], per_pool: int, lists: dict[str, list[str]],
                    examples: dict[str, list[str]], seed: int, thresholds: dict[str, float],
                    names_extra: Sequence[str] = (), parallel: int = 16, max_rounds: int = 40,
                    per_request: int = 8) -> dict[str, Any]:
    """Grow every requested axis; writes ``<out>/pools/<axis>.jsonl`` (kept: {id, axis, text, crossing, source}),
    ``<out>/pools/<axis>.rejects.jsonl`` and ``<out>/pools/stats.json``. An axis whose pool file exists is skipped
    (resume)."""
    pdir = out / "pools"
    pdir.mkdir(parents=True, exist_ok=True)
    stats_path = pdir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for name in axes:
        axis = AXES[name]
        if (pdir / f"{name}.jsonl").exists() and name in stats:
            logger.info("pool %s exists; skipped", name)
            continue
        thr = thresholds.get(name, axis.sim_threshold)
        res = await grow_pool(complete, axis, per_pool, lists, examples.get(name) or list(axis.examples),
                              random.Random(f"{seed}:{name}"), thr, n_per_request=per_request,
                              parallel=parallel, max_rounds=max_rounds,
                              names_extra=names_extra if name == "names" else ())
        kept = [{"id": f"{name}:{i:04d}", "axis": name, "text": r["text"], "crossing": r["crossing"],
                 "source": r.get("source", "model")} for i, r in enumerate(res["kept"])]
        write_jsonl(pdir / f"{name}.rejects.jsonl", res["rejects"])
        write_jsonl(pdir / f"{name}.jsonl", kept)
        reasons = collections.Counter(r["reason"] for r in res["rejects"])
        stats[name] = {**res["counts"], "reasons": dict(reasons)}
        stats_path.write_text(json.dumps(stats, indent=1))
        logger.info("pool %s: %s", name, stats[name])
    return stats


# ── stage 2: combinations ────────────────────────────────────────────────────

SLOTS_SCENE = ("theme", "setting", "inciting", "stakes", "register")


def load_pools(out: Path) -> dict[str, list[dict[str, Any]]]:
    pools = {name: read_jsonl(out / "pools" / f"{name}.jsonl") for name in AXES}
    missing = [n for n, rows in pools.items() if not rows]
    if missing:
        raise SystemExit(f"pools missing or empty under {out / 'pools'}: {missing} (run the pools stage first)")
    return pools


def arm_schedule(n: int, cast_share: float, rng: random.Random) -> list[tuple[str, str, str]]:
    """(friction, secret_arm, cast_arm) per combination: friction x secret cycle through all eight cells in shuffled
    blocks (balanced to within one per cell), cast arms are a shuffled quota (``cast_share`` casts)."""
    cells = [(f, s) for f in FRICTIONS for s in SECRET_ARMS]
    sched: list[tuple[str, str]] = []
    while len(sched) < n:
        block = list(cells)
        rng.shuffle(block)
        sched.extend(block)
    n_cast = round(n * cast_share)
    casts = ["cast"] * n_cast + ["pair"] * (n - n_cast)
    rng.shuffle(casts)
    return [(f, s, c) for (f, s), c in zip(sched[:n], casts)]


def _pick(rng: random.Random, rows: list[dict[str, Any]], uses: collections.Counter, k: int = 1,
          exclude: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """k distinct rows, each drawn with weight 1/(1+uses)^2 (least-used entries first, without starving the rest)."""
    out: list[dict[str, Any]] = []
    taken = set(exclude)
    for _ in range(k):
        cand = [r for r in rows if r["id"] not in taken]
        w = [1.0 / (1 + uses[r["id"]]) ** 2 for r in cand]
        r = rng.choices(cand, weights=w, k=1)[0]
        out.append(r)
        taken.add(r["id"])
    return out


def combo_text(c: dict[str, Any]) -> str:
    parts = [c["relationship"]["text"], *(s["text"] for s in c["scene"].values())]
    parts += [f"{ch['character']['text']} {ch['need']['text']}" for ch in c["characters"]]
    return " | ".join(parts)


def combo_entry_ids(c: dict[str, Any]) -> list[str]:
    ids = [c["relationship"]["id"], *(s["id"] for s in c["scene"].values())]
    for ch in c["characters"]:
        ids += [ch["character"]["id"], ch["need"]["id"]]
    return ids


def _projection(dim_in: int, dim_out: int, seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed).standard_normal((dim_in, dim_out)).astype(np.float32) / math.sqrt(dim_out)


def combine(pools: dict[str, list[dict[str, Any]]], n: int, seed: int, cast_share: float = 0.3,
            max_shared: int = 3, max_sim: float = 0.6, max_tries: int = 50) -> tuple[list[dict[str, Any]], dict]:
    """``n`` combinations with spread and balanced arms (``arm_schedule``). A candidate is redrawn (up to
    ``max_tries`` times per combination) when it shares ``max_shared`` or more non-name entries with an accepted
    combination, or when the cosine of its rendered text to an accepted one reaches ``max_sim`` (character n-gram
    TF-IDF over all pool entries, randomly projected to 256 dimensions). Names never repeat within a combination and
    labels are distinct and not prefixes of one another. Returns (combinations, stats)."""
    import numpy as np

    rng = random.Random(seed)
    uses: collections.Counter = collections.Counter()
    sched = arm_schedule(n, cast_share, rng)
    all_entries = [r for name, rows in pools.items() if name != "names" for r in rows]
    vec_index = {r["id"]: i for i, r in enumerate(all_entries)}
    x = tfidf([r["text"] for r in all_entries])
    proj = _projection(x.shape[1], 256, seed)
    ent_vecs = np.asarray(x @ proj)
    accepted_vecs = np.zeros((n, 256), dtype=np.float32)
    by_entry: dict[str, list[int]] = collections.defaultdict(list)
    out: list[dict[str, Any]] = []
    rejects = collections.Counter()
    for i, (friction, secret_arm, cast_arm) in enumerate(sched):
        k = 2 if cast_arm == "pair" else rng.choice((3, 4))
        for _attempt in range(max_tries):
            rel = _pick(rng, pools["relationship_pair" if cast_arm == "pair" else "relationship_cast"], uses)[0]
            scene = {s: _pick(rng, pools[s], uses)[0] for s in SLOTS_SCENE}
            chars = _pick(rng, pools["character"], uses, k)
            needs = _pick(rng, pools["need"], uses, k)
            names = _pick_names(rng, pools["names"], uses, k)
            holder = rng.randrange(k)
            target = rng.choice([j for j in range(k) if j != holder])
            c = {"id": f"c{i:05d}", "friction": friction, "secret_arm": secret_arm, "cast_arm": cast_arm,
                 "relationship": rel, "scene": scene,
                 "characters": [{"name": nm["text"], "name_id": nm["id"], "label": name_label(nm["text"]),
                                 "character": ch, "need": nd} for nm, ch, nd in zip(names, chars, needs)],
                 "secret_holder": holder, "secret_target": target}
            ids = combo_entry_ids(c)
            v = ent_vecs[[vec_index[e] for e in ids]].sum(axis=0)
            v = v / (np.linalg.norm(v) or 1.0)
            shared = collections.Counter(j for e in ids for j in by_entry.get(e, ()))
            if shared and max(shared.values()) >= max_shared:
                rejects["overlap"] += 1
                continue
            if out and float((accepted_vecs[:len(out)] @ v).max()) >= max_sim:
                rejects["similar"] += 1
                continue
            break
        else:
            rejects["accepted_after_max_tries"] += 1
        accepted_vecs[len(out)] = v
        for e in ids:
            by_entry[e].append(len(out))
        for e in ids + [ch["name_id"] for ch in c["characters"]]:
            uses[e] += 1
        out.append(c)
    stats = {"n": len(out), "rejects": dict(rejects),
             "arms": {"friction": dict(collections.Counter(c["friction"] for c in out)),
                      "secret_arm": dict(collections.Counter(c["secret_arm"] for c in out)),
                      "cast_arm": dict(collections.Counter(c["cast_arm"] for c in out)),
                      "cast_size": dict(collections.Counter(str(len(c["characters"])) for c in out))},
             "max_shared": max_shared, "max_sim": max_sim}
    return out, stats


def _pick_names(rng: random.Random, names: list[dict[str, Any]], uses: collections.Counter,
                k: int) -> list[dict[str, Any]]:
    """k names whose labels are distinct and do not prefix one another; distinct initials are preferred (tried 20
    times first), since labels that look alike blur who is speaking."""
    def ok(picked: list[dict[str, Any]], initials: bool) -> bool:
        low = [(name_label(p["text"]) or "").lower() for p in picked]
        return (all(low) and len(set(low)) == k and not any(a != b and b.startswith(a) for a in low for b in low)
                and (not initials or len({lb[:1] for lb in low}) == k))

    for attempt in range(200):
        picked = _pick(rng, names, uses, k)
        if ok(picked, initials=attempt < 20):
            return picked
    raise ValueError(f"names pool ({len(names)} names) cannot supply {k} distinct speaker labels")


# ── stage 3: headers ─────────────────────────────────────────────────────────

NOTEBOOK_HEAD = ("A playwright's notebook of one-act plays. For each play it lists what was drawn (the theme, the "
                 "register, how the people are bound to each other, the place, what has just happened, the clock, "
                 "and each person with what they need underneath), then the playwright's notes on what each person "
                 "wants today and the secret one of them keeps, then the play's opening page: who is in it and what "
                 "each of them wants, how they are connected, and where and when it happens.\n\n")
END_MARK = "* * *"


@dataclass(frozen=True)
class HeaderExample:
    """One worked notebook entry: the drawn combination, the notes, and the page."""
    friction: str
    secret_arm: str
    theme: str
    register: str
    relationship: str
    setting: str
    inciting: str
    stakes: str
    people: tuple[tuple[str, str, str], ...]  # (full name, character, need)
    holder: int
    target: int
    wants: tuple[str, ...]
    secret: str
    page: str


HEADER_EXAMPLES: tuple[HeaderExample, ...] = (
    HeaderExample(
        "collision", "plain",
        "owing a debt and the illusion of standing alone",
        "a dry comedy of manners that turns, late, into a quarrel",
        "a piano teacher and the former student who now out-earns her",
        "an emptied flat in late afternoon; the one chair nobody took",
        "the former student has turned up unannounced with a van",
        "the movers come back for the last of it at six",
        (("Ines Moreau", "a retired piano teacher who corrects everyone's posture", "to be thanked without asking"),
         ("Kwame Asante", "a session pianist who cannot stand a silence", "to stop being anyone's prodigy")),
        1, 0,
        ("the old upright piano to go to the church hall where she first taught, not to a dealer",
         "to load the piano into his van tonight and drive it away"),
        "he sold the piano to a dealer last week to clear her rent arrears",
        "Ines Moreau, seventy-one, a retired piano teacher. She moves into sheltered housing tomorrow and wants her "
        "upright piano given to the church hall where she first taught.\n"
        "Kwame Asante, thirty-eight, her former pupil, now a session pianist with a van parked outside. He wants the "
        "piano in that van tonight. He has already sold it to a dealer to clear her rent arrears, and has not told "
        "her.\n"
        "Ines taught Kwame from the age of eight. They have not spoken in four years.\n\n"
        "Ines's flat, late afternoon in March. Everything is boxed except one chair and the piano. The movers come "
        "back for the last of it at six."),
    HeaderExample(
        "cooperation-with-incompatible-methods", "hinted",
        "small kindnesses and the belief that they end where they land",
        "a quiet two-hander, slow and plain, the way people talk at night",
        "two night-shift workers who have never had a full conversation",
        "a motorway service station at three in the morning; a child's rucksack with a name tag",
        "a coach has left without one of its passengers, a boy of ten",
        "the coach reaches the ferry port at five and sails half an hour later",
        (("Dolores Vega", "a night cashier who sings along to the till", "to be trusted with something that matters"),
         ("Bram Visser", "a security guard who follows every rule to the letter",
          "to believe that rules protect anyone")),
        0, 1,
        ("to drive the boy to the ferry port herself before the coach sails",
         "to keep the boy in the staff room until the police arrive, as the procedure says"),
        "she saw the boy hide from the coach on purpose, and promised him she would not tell",
        "Dolores Vega, fifty-five, the night cashier. She has her car keys in her hand and wants to drive the boy to "
        "the ferry port herself, now. Whenever anyone asks how he came to be left behind, she finds something to "
        "tidy.\n"
        "Bram Visser, twenty-six, security. He wants the boy kept in the staff room until the police come, as the "
        "procedure says.\n"
        "They have worked the same night shift for two years and have never had a full conversation.\n\n"
        "A motorway service station, three in the morning, January. A child's rucksack with a name tag sits on the "
        "counter. The coach the boy came on reaches the ferry port at five and sails half an hour later."),
    HeaderExample(
        "shared-problem-unequal-information", "plain",
        "inheritance and the hope that children teach their parents",
        "a kitchen-sink comedy, people talking over each other",
        "a dairy farmer, her teenage son, and the inspector who went to school with her",
        "a milking parlour before dawn in August; a clipboard of yesterday's test results",
        "one of the herd has tested positive, and the inspector has arrived an hour early",
        "the milk tanker comes at seven and will take the milk or refuse it",
        (("Ruth Callaghan", "a dairy farmer who never sits down", "to be allowed to fail once"),
         ("Owen Callaghan", "a sixteen-year-old who would rather be anywhere else", "to be asked rather than told"),
         ("Sunil Mehta", "a county inspector who apologises before every sentence",
          "to be forgiven for having left")),
        1, 0,
        ("the inspector to clear the herd before the tanker comes",
         "his mother to sell the farm, and to say so in front of someone who will back him",
         "to get through the visit without closing down a friend"),
        "he labelled last night's sample tubes, and he may have swapped theirs with the neighbour's",
        "Ruth Callaghan, forty-four, a dairy farmer. She wants the herd cleared before the milk tanker comes at "
        "seven.\n"
        "Owen Callaghan, sixteen, her son. He wants her to sell the farm, and has decided to say so this morning, in "
        "front of a witness. He labelled last night's sample tubes, and may have swapped theirs with the "
        "neighbour's.\n"
        "Sunil Mehta, forty-five, the county inspector, who went to school with Ruth. He wants to get through the "
        "visit without closing down a friend.\n\n"
        "A milking parlour before dawn, August. Yesterday's test results are on a clipboard by the door. The tanker "
        "comes at seven and will take the milk or refuse it."),
)


def _first(full: str) -> str:
    return name_label(full) or full.split()[0]


def entry_drawn(no: int, friction: str, secret_arm: str, theme: str, register: str, relationship: str, setting: str,
                inciting: str, stakes: str, people: Sequence[tuple[str, str, str]], holder: int, target: int) -> str:
    """The drawn half of a notebook entry, ending at ``Notes:`` and the first person's open ``wants:`` line."""
    k = len(people)
    count = {2: "two people", 3: "three people", 4: "four people"}.get(k, f"{k} people")
    lines = [f"No. {no}",
             f"Drawn: {count}; {FRICTION_TEXT[friction]}; {SECRET_TEXT[secret_arm]}",
             f"Theme: {theme}", f"Register: {register}", f"Bound: {relationship}", f"Place: {setting}",
             f"Happening: {inciting}", f"Clock: {stakes}"]
    lines += [f"Person: {full} — {ch}; needs {nd}" for full, ch, nd in people]
    lines += [f"Secret: kept by {_first(people[holder][0])}, bearing on what {_first(people[target][0])} wants",
              "Notes:", f"{_first(people[0][0])} wants:"]
    return "\n".join(lines)


def render_example(no: int, ex: HeaderExample) -> str:
    drawn = entry_drawn(no, ex.friction, ex.secret_arm, ex.theme, ex.register, ex.relationship, ex.setting,
                        ex.inciting, ex.stakes, ex.people, ex.holder, ex.target)
    labels = [_first(p[0]) for p in ex.people]
    notes = [f" {ex.wants[0]}"] + [f"{lb} wants: {w}" for lb, w in zip(labels[1:], ex.wants[1:])]
    notes.append(f"{labels[ex.holder]}'s secret: {ex.secret}")
    return f"{drawn}{chr(10).join(notes)}\nPage:\n{ex.page}\n{END_MARK}\n\n"


def header_prompt(c: dict[str, Any], no: int, examples: Sequence[HeaderExample] = HEADER_EXAMPLES) -> str:
    body = "".join(render_example(i + 1, ex) for i, ex in enumerate(examples))
    people = [(ch["name"], ch["character"]["text"], ch["need"]["text"]) for ch in c["characters"]]
    s = c["scene"]
    return NOTEBOOK_HEAD + body + entry_drawn(
        no, c["friction"], c["secret_arm"], s["theme"]["text"], s["register"]["text"], c["relationship"]["text"],
        s["setting"]["text"], s["inciting"]["text"], s["stakes"]["text"], people, c["secret_holder"],
        c["secret_target"])


def parse_header_completion(text: str, labels: Sequence[str], holder: int) -> tuple[Optional[dict[str, Any]], str]:
    """The completion after ``<first label> wants:`` -> ({wants: [..], secret, holder, page}, "") or (None, reason).
    The secret is the ``<label>'s secret:`` line of the drawn holder, else of any cast member (the model often hands
    the secret to someone else, and the page then follows its choice), else a plain notes line about the drawn
    holder; ``holder`` is the index of whoever keeps it."""
    cut = text.find(END_MARK)
    nxt = text.find("\nNo. ")
    ends = [i for i in (cut, nxt) if i >= 0]
    if not ends:
        return None, "unterminated"
    text = text[:min(ends)]
    notes, sep, page = text.partition("\nPage:\n")
    if not sep:
        return None, "no_page"
    lines = [ln.strip() for ln in notes.split("\n")]
    wants = [lines[0]] if lines and lines[0] else []
    secret = ""
    for lb in labels[1:]:
        pre = f"{lb} wants:"
        hit = next((ln[len(pre):].strip() for ln in lines[1:] if ln.startswith(pre)), "")
        wants.append(hit)
    secret = ""
    for h in [holder] + [i for i in range(len(labels)) if i != holder]:
        pre = f"{labels[h]}'s secret:"
        secret = next((ln[len(pre):].strip() for ln in lines if ln.startswith(pre)), "")
        if secret:
            holder = h
            break
    if not secret:  # the secret written as a plain sentence about its holder ("Rosa has never told her ...")
        secret = next((ln for ln in lines[1:] if ln.startswith(labels[holder] + " ")
                       and not ln.startswith(f"{labels[holder]} wants:")), "")
    if len(wants) != len(labels) or not all(len(w) >= 8 for w in wants):
        return None, "want_missing"
    if len(secret) < 8:
        return None, "secret_missing"
    return {"wants": wants, "secret": secret, "holder": holder, "page": page.strip()}, ""


_CONTENT = re.compile(r"[a-z][a-z'-]{2,}")
_STOPWORDS = frozenset("the and for with that this from have has had was were are his her their them they she him "
                       "who what when where which while into onto about after before than then there here its it's "
                       "not but all any one two can could would should will just only also very more most some "
                       "wants want needs need".split())
_LEAK = re.compile(r"^(?:Drawn|Theme|Register|Bound|Place|Happening|Clock|Person|Secret|Notes|Page|No\.)\b|"
                   r"\bsecret:|\bSECRET\b", re.M)
_AGE_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}


def content_words(text: str) -> set[str]:
    return {w for w in _CONTENT.findall(text.lower()) if w not in _STOPWORDS}


def overlap(a: str, b: str) -> float:
    """Share of ``a``'s content words that occur in ``b`` (0 when ``a`` has none)."""
    wa = content_words(a)
    return len(wa & content_words(b)) / len(wa) if wa else 0.0


def _age(token: str) -> Optional[int]:
    t = token.lower().strip(",.;")
    if t.isdigit():
        return int(t)
    if t in _AGE_WORDS:
        return _AGE_WORDS[t]
    if t in _TENS:
        return _TENS[t]
    if "-" in t:
        a, _, b = t.partition("-")
        if a in _TENS and b in _AGE_WORDS:
            return _TENS[a] + _AGE_WORDS[b]
    if t in ("hundred", "a-hundred"):
        return 100
    return None


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


_EXAMPLE_8GRAMS = set().union(*(_ngrams(ex.page, 8) for ex in HEADER_EXAMPLES))


def gate_page(parsed: dict[str, Any], c: dict[str, Any]) -> Optional[str]:
    """Rule-based coherence gate on one parsed header; the rejection reason, or None."""
    page = parsed["page"]
    names = [ch["name"] for ch in c["characters"]]
    labels = [ch["label"] for ch in c["characters"]]
    if not 120 <= len(page) <= 1600:
        return "page_length"
    if mojibake(page) or re.search(r"\S{40,}", page):
        return "garbled"
    plines = [ln.strip() for ln in page.split("\n") if ln.strip()]
    if len(set(plines)) < len(plines):
        return "garbled"
    if _LEAK.search(page):
        return "label_leak"
    if re.search(r"(?m)^\s*(?:" + "|".join(re.escape(x) for x in labels + [n.upper() for n in labels]) + r")\s*[:(]",
                 page):
        return "speaker_tag"
    if any(lb not in page and n not in page for lb, n in zip(labels, names)):
        return "name_missing"
    starts = [next((i for i, ln in enumerate(plines) if ln.startswith(n)), -1) for n in names]
    if any(sum(ln.startswith(n) for ln in plines) > 1 for n in names) or len(set(names)) < len(names):
        return "name_duplicated"
    if len({w.lower() for w in parsed["wants"]}) < len(parsed["wants"]):
        return "wants_identical"
    for w in parsed["wants"]:
        if overlap(w, page) < 0.25:
            return "want_absent"
    for i, s in enumerate(starts):
        if s < 0:
            continue
        line = plines[s]
        m = re.match(re.escape(names[i]) + r",?\s+([\w-]+)", line)
        if m:
            age = _age(m.group(1))
            if age is not None and not 5 <= age <= 105:
                return "impossible_age"
        own = overlap(c["characters"][i]["character"]["text"], line)
        others = [overlap(c["characters"][j]["character"]["text"], line) for j in range(len(names)) if j != i]
        if others and max(others) >= 0.5 and max(others) > own + 0.2:
            return "role_mixup"
    if len(_ngrams(page, 8) & _EXAMPLE_8GRAMS) >= 3:
        return "copied_example"
    page6 = _ngrams(page, 6)
    if any(_ngrams(ch["need"]["text"], 6) & page6 for ch in c["characters"]):
        return "need_pasted"  # the need is the layer under the want; pasted onto the page it reads as a slogan
    if c["secret_arm"] == "plain" and overlap(parsed["secret"], page) < 0.3:
        return "secret_not_stated"  # the plain arm states the secret on the page; hinting is left to the model
    return None


async def run_headers(complete: Complete, score: Optional[Score], combos: list[dict[str, Any]], out: Path,
                      candidates: int = 2, nll_cut: float = 0.05, concurrency: int = 64,
                      rounds: int = 1) -> dict[str, Any]:
    """Write ``<out>/headers.raw.jsonl`` (every candidate with its verdict) and ``<out>/seeds.jsonl`` (one kept
    header per combination that produced a passing candidate, after the NLL tail cut). A combination is drawn
    ``candidates`` at a time, for up to ``rounds`` draws, until one candidate passes. Resumes: draws already in
    ``<out>/headers.raw.jsonl`` count toward the rounds."""
    raw_path = out / "headers.raw.jsonl"
    for rnd in range(rounds):
        raw = read_jsonl(raw_path)
        passed = {r["combo_id"] for r in raw if not r["reason"]}
        draws = collections.Counter(r["combo_id"] for r in raw if r["k"] % candidates == 0)
        todo = [c for c in combos if c["id"] not in passed and draws[c["id"]] < rounds]
        logger.info("headers round %d: %d combinations, %d passed, %d to draw", rnd + 1, len(combos), len(passed),
                    len(todo))
        if not todo:
            break
        await _draw_headers(complete, score, todo, raw_path, candidates, concurrency,
                            {c["id"]: draws[c["id"]] for c in todo})
    return finalize_headers(combos, read_jsonl(raw_path), out, nll_cut)


async def _draw_headers(complete: Complete, score: Optional[Score], todo: list[dict[str, Any]], raw_path: Path,
                        candidates: int, concurrency: int, draws: dict[str, int]) -> None:
    """One draw of ``candidates`` headers for each combination in ``todo``, appended to ``raw_path``; candidate
    index ``k`` continues across draws."""
    sem = asyncio.Semaphore(concurrency)
    stops = [END_MARK, "\nNo. "]
    t0 = time.time()
    counter = {"n": 0}
    raw_f = raw_path.open("a", encoding="utf-8")

    async def one(c: dict[str, Any]) -> None:
        async with sem:
            labels = [ch["label"] for ch in c["characters"]]
            prompt = header_prompt(c, len(HEADER_EXAMPLES) + 1)
            try:
                outs = await complete(prompt, candidates, stops)
            except Exception as e:  # one combination's failure must not stop the run
                logger.error("combination %s failed: %s", c["id"], e)
                return
            rows = []
            for k, (text, finish, hit) in enumerate(outs, start=draws[c["id"]] * candidates):
                if hit:
                    parsed, reason = None, "overlong"
                else:
                    parsed, reason = parse_header_completion(text + ("\n" + END_MARK if finish == "stop" else ""),
                                                             labels, c["secret_holder"])
                if parsed is not None:
                    reason = gate_page(parsed, c) or ""
                rows.append({"combo_id": c["id"], "k": k, "text": text, "finish": finish, "reason": reason,
                             "parsed": parsed})
            if score is not None:
                ok = [r for r in rows if not r["reason"]]
                if ok:
                    try:
                        lps = await score([(DOC_START, r["parsed"]["page"]) for r in ok])
                        for r, lp in zip(ok, lps):
                            r["nll"] = -lp
                    except Exception as e:
                        logger.error("scoring %s failed: %s", c["id"], e)
            for r in rows:
                raw_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            raw_f.flush()
            counter["n"] += 1
            if counter["n"] % 100 == 0:
                logger.info("headers %d/%d (%.0fs)", counter["n"], len(todo), time.time() - t0)

    try:
        await asyncio.gather(*(one(c) for c in todo))
    finally:
        raw_f.close()


def seed_row(c: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """A loom seed from a combination and its chosen header candidate."""
    p = r["parsed"]
    holder = p.get("holder", c["secret_holder"])
    target = c["secret_target"] if c["secret_target"] != holder else c["secret_holder"]
    chars = [{"name": ch["name"], "label": ch["label"], "character": ch["character"]["text"],
              "need": ch["need"]["text"], "want": w} for ch, w in zip(c["characters"], p["wants"])]
    combo = {"relationship": c["relationship"]["id"], **{s: v["id"] for s, v in c["scene"].items()},
             "characters": [ch["character"]["id"] for ch in c["characters"]],
             "needs": [ch["need"]["id"] for ch in c["characters"]],
             "names": [ch["name_id"] for ch in c["characters"]]}
    return {"id": f"sp-{c['id']}", "cast": [ch["label"] for ch in c["characters"]], "header": p["page"],
            "combo": combo, "friction": c["friction"], "cast_arm": c["cast_arm"], "secret_arm": c["secret_arm"],
            "characters": chars,
            "secret": {"holder": c["characters"][holder]["label"], "target": c["characters"][target]["label"],
                       "text": p["secret"], "holder_drawn": c["characters"][c["secret_holder"]]["label"]},
            "theme": c["scene"]["theme"]["text"], "register": c["scene"]["register"]["text"],
            "relationship": c["relationship"]["text"], "setting": c["scene"]["setting"]["text"],
            "inciting": c["scene"]["inciting"]["text"], "stakes": c["scene"]["stakes"]["text"],
            "secret_overlap": round(overlap(p["secret"], p["page"]), 3), "nll": r.get("nll"), "candidate": r["k"]}


def finalize_headers(combos: list[dict[str, Any]], raw: list[dict[str, Any]], out: Path,
                     nll_cut: float) -> dict[str, Any]:
    """Take the first passing candidate per combination (draw order, so NLL never picks between siblings: choosing
    the likeliest sibling would select for the blandest header), cut the worst ``nll_cut`` share of the chosen
    headers by NLL, and write ``<out>/seeds.jsonl`` + ``<out>/headers.stats.json``."""
    by_combo: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in raw:
        by_combo[r["combo_id"]].append(r)
    chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for c in combos:
        ok = sorted((r for r in by_combo.get(c["id"], []) if not r["reason"]), key=lambda r: r["k"])
        if ok:
            chosen.append((c, ok[0]))
    nlls = sorted(r["nll"] for _, r in chosen if r.get("nll") is not None)
    cut = None
    if nlls and nll_cut > 0:
        cut = nlls[max(0, int(math.ceil(len(nlls) * (1 - nll_cut))) - 1)]
    kept = [(c, r) for c, r in chosen if cut is None or r.get("nll") is None or r["nll"] <= cut]
    rows = [seed_row(c, r) for c, r in kept]
    write_jsonl(out / "seeds.jsonl", rows)
    reasons = collections.Counter(r["reason"] or "pass" for r in raw)
    stats = {"combinations": len(combos), "drawn": len(by_combo), "candidates": len(raw),
             "candidate_verdicts": dict(reasons), "combos_with_pass": len(chosen),
             "nll_cut_share": nll_cut, "nll_cut_value": cut, "nll_quantiles": quantiles(nlls),
             "cut_by_nll": len(chosen) - len(kept), "kept": len(rows)}
    (out / "headers.stats.json").write_text(json.dumps(stats, indent=1))
    logger.info("headers: %s", stats)
    return stats


# ── stage 4: report ──────────────────────────────────────────────────────────


def report(out: Path, sample: int = 40, seed: int = 0) -> dict[str, Any]:
    """``<out>/report.json`` + ``<out>/reading-sample.md`` from the stage outputs under ``<out>``."""
    pool_stats = json.loads((out / "pools" / "stats.json").read_text()) if (out / "pools" / "stats.json").exists() \
        else {}
    combo_stats = json.loads((out / "combos.stats.json").read_text()) if (out / "combos.stats.json").exists() else {}
    head_stats = json.loads((out / "headers.stats.json").read_text()) if (out / "headers.stats.json").exists() \
        else {}
    seeds = read_jsonl(out / "seeds.jsonl")
    pools = {name: read_jsonl(out / "pools" / f"{name}.jsonl") for name in AXES}
    reuse: dict[str, Any] = {}
    for name, rows in pools.items():
        if not rows:
            continue
        cnt: collections.Counter = collections.Counter()
        for s in seeds:
            v = s["combo"].get(name)
            if name == "character":
                v = s["combo"]["characters"]
            elif name == "need":
                v = s["combo"]["needs"]
            elif name == "names":
                v = s["combo"]["names"]
            elif name.startswith("relationship"):
                v = s["combo"]["relationship"] if s["combo"]["relationship"].startswith(name) else None
            for x in (v if isinstance(v, list) else [v] if v else []):
                cnt[x] += 1
        used = [cnt[r["id"]] for r in rows]
        reuse[name] = {"pool": len(rows), "used": sum(1 for u in used if u), "max_uses": max(used, default=0),
                       "mean_uses": round(sum(used) / len(used), 2) if used else 0.0}
    nn = nearest_neighbour_sims([s["header"] for s in seeds])
    pool_nn = {name: quantiles(nearest_neighbour_sims([r["text"] for r in rows])) for name, rows in pools.items()
               if len(rows) > 1 and name != "names"}
    arms = {k: dict(collections.Counter(s[k] for s in seeds)) for k in ("friction", "secret_arm", "cast_arm")}
    arms["friction_x_secret"] = dict(collections.Counter(f"{s['friction']}/{s['secret_arm']}" for s in seeds))
    arms["secret_overlap_by_arm"] = {a: quantiles([s["secret_overlap"] for s in seeds if s["secret_arm"] == a],
                                                  (0.1, 0.5, 0.9)) for a in SECRET_ARMS}
    arms["secret_moved_holder"] = sum(s["secret"]["holder"] != s["secret"]["holder_drawn"] for s in seeds)
    rep = {"pools": pool_stats, "combinations": combo_stats, "headers": head_stats, "seeds": len(seeds),
           "entry_reuse": reuse, "header_nn_similarity": quantiles(nn), "pool_nn_similarity": pool_nn,
           "arms": arms}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    rng = random.Random(seed)
    pick = rng.sample(seeds, min(sample, len(seeds)))
    md = [f"# Reading sample: {len(pick)} random seeds of {len(seeds)}\n"]
    for s in pick:
        md.append(f"## {s['id']} · {s['cast_arm']} · {s['friction']} · secret {s['secret_arm']} · "
                  f"nll {s['nll'] if s['nll'] is None else round(s['nll'], 3)}\n")
        md.append(f"*theme:* {s['theme']}  \n*register:* {s['register']}  \n*bound:* {s['relationship']}  \n"
                  f"*place:* {s['setting']}  \n*happening:* {s['inciting']}  \n*clock:* {s['stakes']}\n")
        for ch in s["characters"]:
            md.append(f"- **{ch['name']}** — {ch['character']}; needs {ch['need']}; wants {ch['want']}")
        md.append(f"- secret ({s['secret']['holder']} → {s['secret']['target']}): {s['secret']['text']}\n")
        md.append("```\n" + s["header"] + "\n```\n")
    (out / "reading-sample.md").write_text("\n".join(md), encoding="utf-8")
    return rep


# ── CLI ──────────────────────────────────────────────────────────────────────


def _per_axis(items: Sequence[str]) -> dict[str, str]:
    """``["AXIS=VALUE", ...]`` -> {axis: value}; an unknown axis or a missing ``=`` exits with the reason."""
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected AXIS=VALUE, got {it!r}")
        if k not in AXES:
            raise SystemExit(f"unknown axis {k!r}; axes: {sorted(AXES)}")
        out[k] = v
    return out


def _completer(endpoints: list[str], law_name: str, temperature: Optional[float],
               max_tokens: int) -> tuple[Complete, Callable[[], Awaitable[None]], dict[str, float]]:
    from kotodama.posttrain.selfplay import oai_completer
    from kotodama.serve.laws import law

    sampling = law(law_name)
    if temperature is not None:
        sampling["temperature"] = temperature
    complete, close = oai_completer(endpoints, sampling, max_tokens=max_tokens)
    return complete, close, sampling


def main_pools(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="model-written seed pools (python -m kotodama.posttrain.seeds pools)")
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axes", default=",".join(AXES), help="comma-separated axes")
    ap.add_argument("--per_pool", type=int, default=300)
    ap.add_argument("--crossing_dir", default=None,
                    help="directory of crossing lists, one item per line, named <list>.txt (see Axis.crossing); "
                         "without it every axis is an uncrossed list")
    ap.add_argument("--examples", action="append", default=[], help="AXIS=FILE: example entries replacing the "
                                                                     "defaults (one per line)")
    ap.add_argument("--names_file", default=None, help="names merged into the names pool ahead of model names")
    ap.add_argument("--threshold", action="append", default=[], help="AXIS=COS: dissimilarity threshold override")
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--parallel", type=int, default=32, help="requests per round")
    ap.add_argument("--per_request", type=int, default=2,
                    help="completions per request (they share the request's crossing)")
    ap.add_argument("--max_rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    axes = [x for x in a.axes.split(",") if x]
    unknown = [x for x in axes if x not in AXES]
    if unknown:
        raise SystemExit(f"unknown axes {unknown}; axes: {sorted(AXES)}")
    lists: dict[str, list[str]] = {}
    if a.crossing_dir:
        for p in sorted(Path(a.crossing_dir).glob("*.txt")):
            lists[p.stem] = load_list(p)
        wanted = {n for ax in axes for n in AXES[ax].crossing}
        missing = sorted(wanted - set(lists))
        if missing:
            logger.warning("crossing lists missing from %s: %s (those axes run uncrossed)", a.crossing_dir, missing)
    examples = {k: load_list(Path(p)) for k, p in _per_axis(a.examples).items()}
    thresholds = {k: float(v) for k, v in _per_axis(a.threshold).items()}
    names_extra = load_list(Path(a.names_file)) if a.names_file else []
    complete, close, sampling = _completer(a.endpoint, a.law, a.temperature, a.max_tokens)
    out = Path(a.out)

    async def go() -> dict[str, Any]:
        try:
            return await run_pools(complete, out, axes, a.per_pool, lists, examples, a.seed, thresholds,
                                   names_extra, a.parallel, a.max_rounds, a.per_request)
        finally:
            await close()

    stats = asyncio.run(go())
    (out / "pools" / "run.json").write_text(json.dumps({"args": vars(a), "sampling": sampling,
                                                         "lists": {k: len(v) for k, v in lists.items()}}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_combine(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="pool entries -> combinations (python -m kotodama.posttrain.seeds "
                                             "combine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--cast_share", type=float, default=0.3)
    ap.add_argument("--max_shared", type=int, default=3)
    ap.add_argument("--max_sim", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    combos, stats = combine(load_pools(out), a.n, a.seed, a.cast_share, a.max_shared, a.max_sim)
    write_jsonl(out / "combos.jsonl", combos)
    (out / "combos.stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_headers(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="combinations -> scene headers (python -m kotodama.posttrain.seeds "
                                             "headers)")
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N combinations (0 = all)")
    ap.add_argument("--candidates", type=int, default=4, help="headers drawn per combination")
    ap.add_argument("--nll_cut", type=float, default=0.05, help="share of the chosen headers cut by NLL (worst)")
    ap.add_argument("--rounds", type=int, default=3, help="draws per combination until one candidate passes")
    ap.add_argument("--no_score", action="store_true", help="skip the NLL read (rules only)")
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=None, help="override the law's temperature")
    ap.add_argument("--max_tokens", type=int, default=480)
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args(argv)
    out = Path(a.out)
    combos = read_jsonl(out / "combos.jsonl")
    if not combos:
        raise SystemExit(f"no combinations in {out / 'combos.jsonl'} (run the combine stage first)")
    if a.limit:
        combos = combos[:a.limit]
    complete, close, sampling = _completer(a.endpoint, a.law, a.temperature, a.max_tokens)
    score, sclose = (None, None)
    if not a.no_score:
        from kotodama.posttrain.loom import http_scorer

        score, sclose = http_scorer(a.endpoint, max_completion_tokens=1024)

    async def go() -> dict[str, Any]:
        try:
            return await run_headers(complete, score, combos, out, a.candidates, a.nll_cut, a.concurrency,
                                     a.rounds)
        finally:
            await close()
            if sclose is not None:
                await sclose()

    stats = asyncio.run(go())
    (out / "headers.run.json").write_text(json.dumps({"args": vars(a), "sampling": sampling}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_report(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="pool/combination/header report + reading sample "
                                             "(python -m kotodama.posttrain.seeds report)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rep = report(Path(a.out), a.sample, a.seed)
    print(json.dumps(rep, indent=1))
    return 0


COMMANDS: dict[str, Callable[[list[str]], int]] = {"pools": main_pools, "combine": main_combine,
                                                    "headers": main_headers, "report": main_report}
