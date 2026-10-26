"""Fluid in-context batteries: rules and bindings that exist only in the prompt, beside a crystallized control.

A few-shot trivia prompt measures storage as much as in-context ability. These batteries cannot be memorized: the
inflection, the operators, the bindings and the patterns are invented per prompt, so the context is the only place
the answer exists. A crystallized control (C0) asks real-world facts in the same few-shot frame, so the two abilities
can be told apart across models. Related work: ICLEval (arXiv 2406.14955) separates copying and rule learning from
knowledge; randomly sampled language problems (arXiv 2501.02825) put naive formats at the floor, which is why every
family here climbs a difficulty ladder and is read by its midpoint rather than one accuracy.

Every battery is a "rows" battery for the suite's one scorer (closed candidates, first token after the prompt). Every
candidate is ONE token after a leading space, with distinct tokens, in each tokenizer of ``ZOO_TOKENIZERS``
(``python -m kotodama.eval.binding check-fluid-tokens`` re-checks the frozen pools), so no family needs
whole-continuation scoring.

Families (battery · ladder, easy → hard · candidates · chance):
  F1  fluid-v1-morph    an invented inflection over invented stems of CV syllables, shown in n examples, then a
                        new stem. Rungs: rule tier (suffix ``ko ba → ko ba mi``; infix ``ko ba → ko mi ba`` or vowel
                        change ``ko ba → ko bi``; a suffix conditioned on stem shape, two- vs three-syllable stems)
                        × n ∈ {8, 4, 2}: 9 rungs. The query copies the stem up to the slot the rule acts on, and the
                        read is that slot. 5 candidates: the correct form plus rule-violating forms, per tier, from:
                        a wrong affix, the other class's affix, the input unchanged, the most recent example's
                        output, a stem syllable repeated or restarted (``roles`` names each). Chance 1/5.
  F2  fluid-v1-algebra  invented binary operators (``⊕ ⊗ ⊙``) over 6 invented tokens, 3 example equations per
                        operator, then a new equation. Each operator follows one rule: FIRST (a∘b = a), SECOND
                        (a∘b = b) or PARTNER (a∘b = the partner of a, under a pairing of the 6 tokens shown only
                        through examples). Rungs: operators k ∈ {1, 2, 3} × composition depth d ∈ {1, 2, 3}, left-
                        nested (``(a ⊕ b) ⊗ c =``): 9 rungs. Candidates: the 6-token inventory. Chance 1/6.
  F3  fluid-v1-rebind   one holder of four is reassigned N times, interleaved with the other holders' lines, in
                        prose (``Then the box holds the lamp.``) and in a code-like register (``box = lamp``), then
                        queried. Rungs: N ∈ {1, 2, 4, 8, 16} × distractor reassignments R ∈ {0, N}: 10 rungs; the
                        two registers are a condition inside every rung. 5 candidates: the current value, the STALE
                        value (the one just before it), an older value of the same holder, and the other holders'
                        values. Chance 1/5.
  F4  fluid-v1-seqind   abstract patterns over fresh nonce tokens, 3 example lines then a partial line: a motif of m
                        tokens, g gap tokens, then the motif again either repeated (period-m) or reversed (mirror:
                        ``A B B A``, ``A B C C B A``). Rungs: pattern length m ∈ {2, 3, 4} × gap g ∈ {0, 2, 4} (the
                        lag back to the motif's last occurrence grows with g): 9 rungs. 6 candidates: the pattern's
                        tokens plus in-context distractors (gap tokens, the last example line's token at the same
                        position, other lines' tokens). Chance 1/6.
  C0  cryst-v1          real-world facts in the F1 frame (header, ``subject → answer`` lines, ``subject →``):
                        capitals, chemical symbols, authors → works, currencies. Rungs: shots ∈ {6, 3, 1, 0}; the
                        relation is a condition inside every rung. 6 candidates, all from the relation's answers,
                        none of them shown in the prompt. Chance 1/6.

Choices left open, fixed here:
  - words: CV syllables over the consonants whose five vowel forms are all one token (F1), and short CVC strings (F2,
    F4) that are one token everywhere. "Nonce" here means: not an English word in a hand list. Pieces that are one
    token in seven tokenizers are subword fragments (name stubs, abbreviations), so they are free of meaning as a
    word, not of every prior association.
  - every prompt starts with one header line (F3 has none: its narrative is the frame).
  - F1: all stem syllables of a prompt distinct; a vowel-change stem's consonant row never appears elsewhere in the
    prompt; the middle tier alternates infix and vowel change row by row (``rule``).
  - F2: the query uses min(k, d) distinct operators; no example line repeats the query's first step; every PARTNER
    step the query needs is shown as an example line with another right operand; operator symbols are assigned to
    rules at random per prompt. Projections short-circuit a chain, so rows record ``path_steps`` (the PARTNER steps
    after the last SECOND, the chain's effective depth) beside the written depth.
  - F3: four holders; every value is distinct within a prompt; the R distractor reassignments go to random other
    holders; the other holders' first lines fall anywhere in the chain, so recency is controlled even at R = 0.
  - F4: example and query tokens all distinct within a prompt; within a rung, pattern type and the read position
    in the copy are balanced.
  - C0: the work's answer is the first word of its title (the shots show whole titles); the facts are unambiguous
    inside every candidate set and stable (currencies: none mid-changeover).
  - size: 4 seeds per cell; F1 / F2 / F4 120 rows per rung, F3 60 per rung × register, C0 64 per shots ×
    relation: 5,464 rows (the order of capacity-v1's 5,760).
  - balance: per cell, the gold's position in the candidate list is exactly balanced, and the gold's identity
    cycles through a shuffled copy of its pool.

Shortcut controls (recorded per row; :func:`audit`, :func:`audit_failures`). Each control is the accuracy a model
following that shortcut would get, compared with the row's NEUTRAL rate (the shortcut-neutral law of the bindprobes
families: a shortcut neither helps nor, by being scheduled at zero, becomes an exclusion cue):
  recency    gold = the most recent example's answer (F3: the value in the last line). Scheduled at the neutral rate
             where the answer varies within a prompt (F2, F3 1/C; F1 conditioned 1/2, its two in-context affixes).
             Structurally 0 where the gold is a new form (F1 vowel change, F4, C0). F1 suffix and infix rows are the
             labelled recency subset (``recency_subset``): one affix answers every line of the prompt.
  frequency  gold = the candidate occurring most often in the prompt (ties split). Scheduled at 1/6 in F2; at or
             below neutral by construction elsewhere; exempt in the recency subset.
  novelty    gold = the only kind of candidate absent from the prompt (ties split). Reported, not gated: F1 vowel
             change answers with a form the prompt never shows, so it sits above chance there by design.
  position / identity: exact balance (above), audited.

Read-out (:func:`summarize`), per family, per model:
  r(rung)   normalized accuracy (acc − chance) / (1 − chance): a fixed ceiling of 1, so a model that never reaches
            the top is not normalized up to it
  midpoint  fitted ladder position (rung units, 1 … R) where r̂ = 1 / (1 + exp(β (rung − m))) crosses .5, least
            squares over the rungs; censored HIGH when r(R) > .5 or m > R (scored R + 1), censored LOW when
            r(1) < .5 or m < 1 (scored 0); a curve that is both has no midpoint
  AUC       mean of r clipped to [0, 1] over the rungs (threshold-free)
  F3        the stale-value rate (the pick is the value just before the current one), overall and per rung, and its
            share of the errors; per-register midpoints
  C0        accuracy by shots, per relation
:func:`fluid_scores` turns many models' summaries into the fluid score (the mean of F1–F4 midpoints, z-scored across
the models passed in) and the crystallized score (C0's, z-scored the same way).
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

VERSION = "fluid-v1"
SEEDS = 4
MAX_DRAWS = 5000          # rejection-sampling budget per row
BALANCE_TOL = 0.05        # per cell, as in the bindprobes audits
IDENTITY_MAX = 2.0        # most frequent gold identity ≤ this × the mean count

# The tokenizers every candidate pool is checked against (one per public model family scored with the suite).
ZOO_TOKENIZERS: tuple[str, ...] = (
    "HuggingFaceTB/SmolLM2-135M", "EleutherAI/pythia-160m", "allenai/OLMo-1B-0724-hf", "allenai/OLMo-2-0425-1B",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base", "meta-llama/Llama-3.2-1B",
)

# ── pools (each word one token after a leading space in every ZOO_TOKENIZERS entry; distinct tokens) ──────────────

VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")
# CV syllables over the consonants whose five vowel forms are all single tokens, minus English words
# (be do go me no so to).
SYLLABLES: tuple[str, ...] = (
    "ba", "bi", "bo", "bu", "da", "de", "di", "du", "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "gu",
    "ma", "mi", "mo", "mu", "na", "ne", "ni", "nu", "pa", "pe", "pi", "po", "pu", "sa", "se", "si", "su",
    "ta", "te", "ti", "tu",
)
# CVC strings that are not English words (hand list).
NONCE: tuple[str, ...] = (
    "bif", "bip", "bir", "dil", "dop", "dut", "fif", "fus", "gaz", "gir", "kar", "ker", "ket", "lig",
    "lik", "mim", "mir", "neb", "nex", "nud", "pag", "pav", "pir", "piv", "rav", "raz", "reb", "rif",
    "rud", "sab", "saf", "sut", "tet", "tid", "tob", "vap", "vig", "vir", "vit", "viv", "vul", "jur",
)
# F3 values: everyday objects, none of them a container.
VALUES: tuple[str, ...] = (
    "lamp", "key", "coin", "ring", "pen", "book", "shoe", "hat", "cup", "fork", "spoon", "knife", "plate", "bell",
    "rope", "nail", "brush", "comb", "sock", "glove", "map", "card", "stamp", "shell", "stone", "leaf", "feather",
    "button", "candle", "mirror", "clock", "ball", "doll", "drum", "apple", "pear", "lemon", "egg", "bread",
    "cheese", "soap", "towel", "belt", "badge", "ticket", "pencil", "marble", "blanket", "pillow", "hammer",
    "ladder", "candy", "cookie", "radio", "camera", "phone",
)
HOLDERS: tuple[str, ...] = ("box", "jar", "bag", "tin", "crate", "chest", "sack", "drawer")
OPS: tuple[str, ...] = ("⊕", "⊗", "⊙")

# C0: (subject, answer token, the answer as the shots show it).
Fact = tuple[str, str, str]


def _same(pairs: Sequence[tuple[str, str]]) -> tuple[Fact, ...]:
    return tuple((s, a, a) for s, a in pairs)


RELATIONS: dict[str, tuple[str, tuple[Fact, ...]]] = {
    "capital": ("Country → capital:", _same((
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"),
        ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Norway", "Oslo"), ("Greece", "Athens"), ("Ireland", "Dublin"),
        ("United Kingdom", "London"), ("Austria", "Vienna"), ("Canada", "Ottawa"), ("China", "Beijing"),
        ("South Korea", "Seoul"), ("Iraq", "Baghdad"), ("Iran", "Tehran"), ("Portugal", "Lisbon"),
        ("Poland", "Warsaw"), ("Czech Republic", "Prague"), ("Hungary", "Budapest"), ("Sweden", "Stockholm"),
        ("Finland", "Helsinki"), ("Belgium", "Brussels"), ("Philippines", "Manila"), ("Syria", "Damascus"),
        ("Denmark", "Copenhagen"), ("United States", "Washington"), ("Chile", "Santiago"),
    ))),
    "symbol": ("Element → chemical symbol:", _same((
        ("Iron", "Fe"), ("Gold", "Au"), ("Silver", "Ag"), ("Sodium", "Na"), ("Potassium", "K"), ("Copper", "Cu"),
        ("Lead", "Pb"), ("Tin", "Sn"), ("Neon", "Ne"), ("Calcium", "Ca"), ("Magnesium", "Mg"), ("Chlorine", "Cl"),
        ("Oxygen", "O"), ("Hydrogen", "H"), ("Nitrogen", "N"), ("Carbon", "C"), ("Sulfur", "S"),
        ("Silicon", "Si"), ("Aluminum", "Al"), ("Lithium", "Li"), ("Nickel", "Ni"), ("Platinum", "Pt"),
        ("Uranium", "U"), ("Titanium", "Ti"), ("Argon", "Ar"), ("Fluorine", "F"), ("Phosphorus", "P"),
    ))),
    "author": ("Author → book:", (
        ("Jane Austen", "Pride", "Pride and Prejudice"), ("Charles Dickens", "Oliver", "Oliver Twist"),
        ("Miguel de Cervantes", "Don", "Don Quixote"), ("Mark Twain", "Tom", "Tom Sawyer"),
        ("George Orwell", "Animal", "Animal Farm"), ("Victor Hugo", "Les", "Les Misérables"),
        ("Leo Tolstoy", "War", "War and Peace"), ("Gustave Flaubert", "Madame", "Madame Bovary"),
        ("Dante Alighieri", "Divine", "Divine Comedy"), ("Fyodor Dostoevsky", "Crime", "Crime and Punishment"),
        ("Jules Verne", "Twenty", "Twenty Thousand Leagues Under the Seas"),
        ("John Milton", "Paradise", "Paradise Lost"), ("Roald Dahl", "Charlie", "Charlie and the Chocolate Factory"),
        ("J. K. Rowling", "Harry", "Harry Potter and the Philosopher's Stone"), ("Isaac Asimov", "Foundation",
                                                                                 "Foundation"),
        ("Daniel Defoe", "Robinson", "Robinson Crusoe"), ("Robert Louis Stevenson", "Treasure", "Treasure Island"),
        ("J. M. Barrie", "Peter", "Peter Pan"), ("Lewis Carroll", "Alice", "Alice's Adventures in Wonderland"),
        ("Rudyard Kipling", "Kim", "Kim"), ("Daphne du Maurier", "Rebecca", "Rebecca"),
        ("Louisa May Alcott", "Little", "Little Women"), ("Anna Sewell", "Black", "Black Beauty"),
        ("Joseph Conrad", "Heart", "Heart of Darkness"), ("Boris Pasternak", "Doctor", "Doctor Zhivago"),
        ("Virginia Woolf", "Mrs", "Mrs Dalloway"),
    )),
    "currency": ("Country → currency:", _same((
        ("Germany", "euro"), ("United States", "dollar"), ("United Kingdom", "pound"), ("Switzerland", "franc"),
        ("South Korea", "won"), ("South Africa", "rand"), ("Brazil", "real"), ("Armenia", "dram"),
        ("Kyrgyzstan", "som"), ("Costa Rica", "colon"), ("Peru", "sol"), ("Bosnia and Herzegovina", "mark"),
    ))),
}

POOLS: dict[str, tuple[str, ...]] = {
    "syllables": SYLLABLES, "nonce": NONCE, "values": VALUES,
    **{f"c0-{r}": tuple(a for _, a, _ in facts) for r, (_, facts) in RELATIONS.items()},
}


# ── families and ladders ─────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Family:
    key: str                          # F1 … F4, C0
    battery: str                      # batteries.BATTERIES key and prompts-file stem
    reader: str                       # readers.READERS key
    ladder: tuple[str, ...]           # rung labels, easy → hard (rung r = index + 1)
    n_candidates: int
    trials: int                       # per seed per cell
    condition: Optional[str]          # a row field pooled inside every rung (reported separately)


F1_TIERS: tuple[str, ...] = ("suffix", "mid", "cond")
F1_N: tuple[int, ...] = (8, 4, 2)
F2_K: tuple[int, ...] = (1, 2, 3)
F2_DEPTH: tuple[int, ...] = (1, 2, 3)
F3_N: tuple[int, ...] = (1, 2, 4, 8, 16)
F3_REGISTERS: tuple[str, ...] = ("prose", "code")
F4_M: tuple[int, ...] = (2, 3, 4)
F4_GAP: tuple[int, ...] = (0, 2, 4)
F4_EXAMPLES = 3
F2_PER_OP = 3
F2_INVENTORY = 6
F3_HOLDERS = 4
C0_SHOTS: tuple[int, ...] = (6, 3, 1, 0)

FAMILIES: dict[str, Family] = {f.key: f for f in (
    Family("F1", "fluid-v1-morph", "fluid-morph", tuple(f"{t}|n{n}" for t in F1_TIERS for n in F1_N), 5, 30, "rule"),
    Family("F2", "fluid-v1-algebra", "fluid-algebra", tuple(f"k{k}|d{d}" for k in F2_K for d in F2_DEPTH), 6, 30,
           None),
    Family("F3", "fluid-v1-rebind", "fluid-rebind", tuple(f"N{n}|R{r}" for n in F3_N for r in ("0", "N")), 5, 15,
           "register"),
    Family("F4", "fluid-v1-seqind", "fluid-seqind", tuple(f"m{m}|g{g}" for m in F4_M for g in F4_GAP), 6, 30,
           "pattern"),
    Family("C0", "cryst-v1", "cryst", tuple(f"shots{s}" for s in C0_SHOTS), 6, 16, "relation"),
)}
FLUID_KEYS: tuple[str, ...] = ("F1", "F2", "F3", "F4")
BY_BATTERY: dict[str, Family] = {f.battery: f for f in FAMILIES.values()}

Row = dict[str, Any]


# ── shared machinery ─────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class _Schedule:
    """Per-cell balanced schedules, indexed by the row's position k in the cell."""
    gold_index: list[int]
    ident: list[Any]
    recent: list[bool]
    freq: list[bool]
    aux: list[Any]


def _balanced(rng: random.Random, n: int, values: Sequence[Any]) -> list[Any]:
    out = [values[i % len(values)] for i in range(n)]
    rng.shuffle(out)
    return out


def _quota(rng: random.Random, n: int, rate: float) -> list[bool]:
    hits = round(n * rate)
    out = [i < hits for i in range(n)]
    rng.shuffle(out)
    return out


def _cycle(rng: random.Random, pool: Sequence[Any], n: int) -> list[Any]:
    """``pool`` in shuffled rounds: every identity is gold ⌊n/|pool|⌋ or ⌈n/|pool|⌉ times."""
    out: list[Any] = []
    while len(out) < n:
        r = list(pool)
        rng.shuffle(r)
        out.extend(r)
    return out[:n]


def count_word(text: str, word: str) -> int:
    """Occurrences of ``word`` as a whole word in ``text``."""
    return len(re.findall(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text))


def _place(rng: random.Random, gold: str, distractors: Sequence[str], gold_index: int) -> list[str]:
    rest = list(distractors)
    rng.shuffle(rest)
    rest.insert(gold_index, gold)
    return [" " + w for w in rest]


def _controls(prompt: str, cands: Sequence[str], gold: str, recent: Optional[str]) -> dict[str, Any]:
    """The shortcut-follower accuracies of one row: recency, frequency (ties split), novelty (ties split)."""
    counts = {c: count_word(prompt, c) for c in cands}
    top = max(counts.values())
    tied = [c for c in cands if counts[c] == top]
    novel = [c for c in cands if counts[c] == 0]
    return {
        "recent_answer": recent,
        "gold_is_recent": recent == gold if recent is not None else False,
        "gold_count": counts[gold],
        "freq_follower": (1.0 / len(tied)) if gold in tied else 0.0,
        "novelty_follower": (1.0 / len(novel)) if gold in novel else 0.0,
    }


def _finish(fam: Family, prompt: str, gold: str, roles: Mapping[str, str], gold_index: int, rng: random.Random,
            recent: Optional[str], neutral_recency: Optional[float], neutral_frequency: Optional[float],
            extra: Mapping[str, Any]) -> Row:
    distract = [w for w in roles if w != gold]
    if len(roles) != fam.n_candidates or gold not in roles or len(set(roles)) != len(roles):
        raise AssertionError(f"{fam.key}: candidate roles {dict(roles)} (want {fam.n_candidates}, gold {gold!r})")
    cands = _place(rng, gold, distract, gold_index)
    row: Row = {"prompt": prompt, "candidates": cands, "gold": gold}
    row.update(extra)
    row["roles"] = dict(roles)
    row["gold_index"] = gold_index
    row["recency_subset"] = neutral_recency is None
    row["neutral_recency"] = neutral_recency
    row["neutral_frequency"] = neutral_frequency
    row.update(_controls(prompt, [c.strip() for c in cands], gold, recent))
    return row


def _draw(make: Callable[[], Optional[Row]], what: str) -> Row:
    for _ in range(MAX_DRAWS):
        row = make()
        if row is not None:
            return row
    raise RuntimeError(f"{what}: no row satisfied the constraints in {MAX_DRAWS} draws")


def _run_cells(fam: Family, cells: Sequence[tuple[str, int, Any]], seeds: int, trials: int,
               sched: Callable[[random.Random, int, Any], _Schedule],
               make: Callable[[random.Random, Any, _Schedule, int], Optional[Row]]) -> list[Row]:
    rows: list[Row] = []
    for cell, rung, params in cells:
        n = seeds * trials
        s = sched(random.Random(f"{VERSION}:{fam.key}:{cell}:schedule"), n, params)
        for seed in range(seeds):
            rng = random.Random(f"{VERSION}:{fam.key}:{cell}:{seed}")
            for t in range(trials):
                k = seed * trials + t
                body = _draw(lambda: make(rng, params, s, k), f"{fam.battery} {cell} s{seed} t{t}")
                row: Row = {"id": f"{fam.battery}|{cell}|s{seed}|t{t}", "family": fam.key, "cell": cell,
                            "rung": rung, "seed": seed, "trial": t}
                row.update(body)
                rows.append(row)
    return rows


def _lines(header: str, lines: Sequence[str], tail: str) -> str:
    return "\n".join([header, *lines, tail])


# ── F1 nonce morphology ──────────────────────────────────────────────────────────────────────────────────────────

F1_HEADER = "Word → plural:"


def _cons_row(c: str) -> list[str]:
    return [s for s in SYLLABLES if s[0] == c]


def _f1_sched(rng: random.Random, n: int, params: tuple[str, int]) -> _Schedule:
    tier, _ = params
    c = FAMILIES["F1"].n_candidates
    aux: list[Any] = ([None] * n if tier == "suffix" else
                      _balanced(rng, n, ("infix", "vowel")) if tier == "mid" else _balanced(rng, n, (2, 3)))
    return _Schedule(_balanced(rng, n, range(c)), _cycle(rng, SYLLABLES, n),
                     _quota(rng, n, 0.5) if tier == "cond" else [False] * n, [False] * n, aux)


def _f1_make(rng: random.Random, params: tuple[str, int], s: _Schedule, k: int) -> Optional[Row]:
    fam = FAMILIES["F1"]
    tier, n = params
    gold: str = s.ident[k]
    c = fam.n_candidates
    if tier == "suffix" or (tier == "mid" and s.aux[k] == "infix"):
        rule = "suffix" if tier == "suffix" else "infix"
        pool = [x for x in SYLLABLES if x != gold]
        syl = rng.sample(pool, 2 * (n + 1))
        stems = [(syl[2 * i], syl[2 * i + 1]) for i in range(n + 1)]
        (q1, q2), last = stems[-1], stems[-2]
        wrong = rng.choice([x for x in pool if x not in syl])
        if rule == "suffix":
            lines = [f"{a} {b} → {a} {b} {gold}" for a, b in stems[:-1]]
            tail = f"{q1} {q2} → {q1} {q2}"
            roles = {gold: "gold", q2: "reduplicate", q1: "restart", last[1]: "recent_stem", wrong: "wrong_affix"}
        else:
            lines = [f"{a} {b} → {a} {gold} {b}" for a, b in stems[:-1]]
            tail = f"{q1} {q2} → {q1}"
            roles = {gold: "gold", q2: "unchanged", q1: "reduplicate", last[1]: "recent_stem", wrong: "wrong_affix"}
        return _finish(fam, _lines(F1_HEADER, lines, tail), gold, roles, s.gold_index[k], rng, gold, None, None,
                       {"tier": tier, "n_examples": n, "rule": rule, "stem": [q1, q2]})
    if tier == "mid":                                   # vowel change: the stem's last vowel becomes gold's vowel
        cons, vstar = gold[0], gold[1]
        q2 = rng.choice([x for x in _cons_row(cons) if x != gold])
        free = [x for x in SYLLABLES if x[0] != cons]
        s2_ok = [x for x in free if x[1] != vstar and x[0] + vstar in SYLLABLES]
        stems: list[tuple[str, str]] = []
        used: set[str] = set()
        for _ in range(n):
            bs = [x for x in s2_ok if x not in used]
            if not bs:
                return None
            b = rng.choice(bs)
            used.add(b)
            a = rng.choice([x for x in free if x not in used])
            used.add(a)
            stems.append((a, b))
        q1 = rng.choice([x for x in free if x not in used])
        recent = stems[-1][1][0] + vstar
        wrongs = [cons + v for v in VOWELS if v not in (vstar, q2[1]) and cons + v in SYLLABLES]
        if len(wrongs) < 2:
            return None
        w1, w2 = rng.sample(wrongs, 2)
        lines = [f"{a} {b} → {a} {b[0] + vstar}" for a, b in stems]
        roles = {gold: "gold", q2: "unchanged", w1: "wrong_vowel", w2: "wrong_vowel", recent: "recent_answer"}
        return _finish(fam, _lines(F1_HEADER, lines, f"{q1} {q2} → {q1}"), gold, roles, s.gold_index[k], rng,
                       recent, 1 / c, 1 / c, {"tier": tier, "n_examples": n, "rule": "vowel", "stem": [q1, q2]})
    # conditioned: two-syllable stems take one suffix, three-syllable stems the other
    qclass: int = s.aux[k]
    other = rng.choice([x for x in SYLLABLES if x != gold])
    affix = {qclass: gold, 5 - qclass: other}
    pool = [x for x in SYLLABLES if x not in (gold, other)]
    shapes = [2] * (n // 2) + [3] * (n // 2) + [qclass]
    need = sum(shapes)
    if need + 1 > len(pool):
        return None
    syl = rng.sample(pool, need + 1)
    wrong = syl.pop()
    stems3: list[list[str]] = []
    i = 0
    for size in shapes:
        stems3.append(syl[i:i + size])
        i += size
    query, ex = stems3[-1], stems3[:-1]
    rng.shuffle(ex)
    want = s.recent[k]
    pick = [j for j, st in enumerate(ex) if (len(st) == qclass) == want]
    j = rng.choice(pick)
    ex.append(ex.pop(j))
    lines = [f"{' '.join(st)} → {' '.join(st)} {affix[len(st)]}" for st in ex]
    q = " ".join(query)
    roles = {gold: "gold", other: "wrong_class", query[-1]: "reduplicate", query[0]: "restart", wrong: "wrong_affix"}
    return _finish(fam, _lines(F1_HEADER, lines, f"{q} → {q}"), gold, roles, s.gold_index[k], rng,
                   affix[len(ex[-1])], 0.5, 0.5,
                   {"tier": tier, "n_examples": n, "rule": "cond", "stem": query, "stem_syllables": qclass})


def _gen_morph(seeds: int, trials: int) -> list[Row]:
    fam = FAMILIES["F1"]
    cells = [(f"{t}|n{n}", fam.ladder.index(f"{t}|n{n}") + 1, (t, n)) for t in F1_TIERS for n in F1_N]
    return _run_cells(fam, cells, seeds, trials, _f1_sched, _f1_make)


# ── F2 symbol algebra ────────────────────────────────────────────────────────────────────────────────────────────

F2_HEADER = "Equations:"
F2_RULES: tuple[str, ...] = ("first", "second", "partner")
F2_RULE_SETS: dict[int, tuple[tuple[str, ...], ...]] = {
    1: (("first",), ("second",), ("partner",)),
    2: (("first", "second"), ("first", "partner"), ("second", "partner")),
    3: (("first", "second", "partner"),),
}


def _f2_sched(rng: random.Random, n: int, params: tuple[int, int]) -> _Schedule:
    kops, _ = params
    c = FAMILIES["F2"].n_candidates
    return _Schedule(_balanced(rng, n, range(c)), _cycle(rng, NONCE, n), _quota(rng, n, 1 / c),
                     _quota(rng, n, 1 / c), _balanced(rng, n, F2_RULE_SETS[kops]))


def f2_apply(rule: str, left: int, right: int, partner: Mapping[int, int]) -> int:
    if rule == "first":
        return left
    if rule == "second":
        return right
    if rule == "partner":
        return partner[left]
    raise ValueError(f"unknown F2 rule {rule!r}")


def _f2_query(t: Sequence[str], leaves: Sequence[int], ops: Sequence[str]) -> str:
    expr = t[leaves[0]]
    for i, op in enumerate(ops):
        nxt = f"{expr} {op} {t[leaves[i + 1]]}"
        expr = nxt if i == len(ops) - 1 else f"({nxt})"
    return expr + " ="


def _f2_make(rng: random.Random, params: tuple[int, int], s: _Schedule, k: int) -> Optional[Row]:
    fam = FAMILIES["F2"]
    kops, depth = params
    rules: tuple[str, ...] = s.aux[k]
    syms = rng.sample(OPS, len(rules))
    op_rule = dict(zip(syms, rules))
    order = list(range(F2_INVENTORY))
    rng.shuffle(order)
    partner = {}
    for i in range(0, F2_INVENTORY, 2):
        partner[order[i]], partner[order[i + 1]] = order[i + 1], order[i]
    leaves = rng.sample(range(F2_INVENTORY), depth + 1)
    ops = [rng.choice(syms) for _ in range(depth)]
    if len(set(ops)) != min(kops, depth):
        return None
    v = leaves[0]
    needs: list[tuple[str, int, int]] = []
    for i, op in enumerate(ops):
        if op_rule[op] == "partner":
            needs.append((op, v, leaves[i + 1]))
        v = f2_apply(op_rule[op], v, leaves[i + 1], partner)
    gold_slot = v
    # SECOND discards everything before it and FIRST keeps the value, so only the PARTNER steps after the last
    # SECOND change what the answer is: the chain's effective depth, recorded beside its written depth.
    last_second = max((i for i, op in enumerate(ops) if op_rule[op] == "second"), default=-1)
    path_steps = sum(op_rule[op] == "partner" for op in ops[last_second + 1:])
    first_step = (leaves[0], ops[0], leaves[1])
    eqs: list[tuple[int, str, int, int]] = []
    for op in syms:
        rule = op_rule[op]
        mine: list[tuple[int, str, int, int]] = []
        if rule == "partner":
            for _, a, qb in needs:
                if any(e[0] == a for e in mine):
                    continue
                bs = [x for x in range(F2_INVENTORY) if x not in (a, qb, partner[a])]
                mine.append((a, op, rng.choice(bs), partner[a]))
        while len(mine) < F2_PER_OP:
            a, b = rng.sample(range(F2_INVENTORY), 2)
            if rule == "partner" and b == partner[a]:
                continue                                # a ∘ partner(a) = partner(a) also reads as SECOND
            if (a, op, b) == first_step or any(e[:3] == (a, op, b) for e in mine):
                continue
            mine.append((a, op, b, f2_apply(rule, a, b, partner)))
        eqs.extend(mine)
    rng.shuffle(eqs)
    want = s.recent[k]
    pick = [j for j, e in enumerate(eqs) if (e[3] == gold_slot) == want]
    if not pick:
        return None
    eqs.append(eqs.pop(rng.choice(pick)))
    counts = Counter(x for e in eqs for x in (e[0], e[2], e[3]))
    counts.update(leaves)
    top = max(counts[x] for x in range(F2_INVENTORY))
    if (counts[gold_slot] == top) != s.freq[k]:
        return None
    gold: str = s.ident[k]
    rest = rng.sample([w for w in NONCE if w != gold], F2_INVENTORY - 1)
    tok = {}
    j = 0
    for slot in range(F2_INVENTORY):
        if slot == gold_slot:
            tok[slot] = gold
        else:
            tok[slot] = rest[j]
            j += 1
    t = [tok[x] for x in range(F2_INVENTORY)]
    lines = [f"{t[a]} {op} {t[b]} = {t[r]}" for a, op, b, r in eqs]
    roles = {w: "gold" if w == gold else (f"leaf{leaves.index(x)}" if x in leaves else "other")
             for x, w in enumerate(t)}
    gold_role = f"leaf{leaves.index(gold_slot)}" if gold_slot in leaves else "other"
    c = fam.n_candidates
    return _finish(fam, _lines(F2_HEADER, lines, _f2_query(t, leaves, ops)), gold, roles, s.gold_index[k], rng,
                   t[eqs[-1][3]], 1 / c, 1 / c,
                   {"k_ops": kops, "depth": depth, "op_rules": op_rule, "partner": {t[a]: t[b] for a, b in
                                                                                    partner.items()},
                    "leaves": [t[x] for x in leaves], "query_ops": ops, "gold_role": gold_role,
                    "path_steps": path_steps})


def _gen_algebra(seeds: int, trials: int) -> list[Row]:
    fam = FAMILIES["F2"]
    cells = [(f"k{k}|d{d}", fam.ladder.index(f"k{k}|d{d}") + 1, (k, d)) for k in F2_K for d in F2_DEPTH]
    return _run_cells(fam, cells, seeds, trials, _f2_sched, _f2_make)


# ── F3 rebinding chains ──────────────────────────────────────────────────────────────────────────────────────────

def _f3_sched(rng: random.Random, n: int, params: tuple[int, int, str]) -> _Schedule:
    c = FAMILIES["F3"].n_candidates
    return _Schedule(_balanced(rng, n, range(c)), _cycle(rng, VALUES, n), _quota(rng, n, 1 / c), [False] * n,
                     [None] * n)


def _f3_render(register: str, events: Sequence[tuple[str, str, bool]], target: str) -> str:
    if register == "prose":
        body = " ".join(f"Then the {h} holds the {v}." if again else f"The {h} holds the {v}."
                        for h, v, again in events)
        return f"{body} In the end, the {target} holds the"
    if register == "code":
        return "\n".join(f"{h} = {v}" for h, v, _ in events) + f"\nassert {target} =="
    raise ValueError(f"unknown F3 register {register!r}")


def _f3_make(rng: random.Random, params: tuple[int, int, str], s: _Schedule, k: int) -> Optional[Row]:
    fam = FAMILIES["F3"]
    n, r, register = params
    holders = rng.sample(HOLDERS, F3_HOLDERS)
    target, others = holders[0], holders[1:]
    n_values = (n + 1) + len(others) + r
    gold: str = s.ident[k]
    vals = [gold] + rng.sample([w for w in VALUES if w != gold], n_values - 1)
    chain = vals[1:n + 1] + [gold]                      # the target's values, first → current
    rest = vals[n + 1:]
    queues: dict[str, list[str]] = {target: list(chain)}
    for h in others:
        queues[h] = [rest.pop()]
    for _ in range(r):
        queues[rng.choice(others)].append(rest.pop())
    history = {h: list(q) for h, q in queues.items()}
    want = s.recent[k]
    held = target if want else rng.choice(others)
    last = (held, queues[held].pop())
    events: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    while any(queues.values()):
        h = rng.choices(list(queues), weights=[len(q) for q in queues.values()])[0]
        events.append((h, queues[h].pop(0), h in seen))
        seen.add(h)
    events.append((last[0], last[1], last[0] in seen))
    stale = chain[-2]
    roles: dict[str, str] = {gold: "gold", stale: "stale"}
    if n >= 2:
        roles[chain[-3]] = "older"
    cross_now = [history[h][-1] for h in others]
    cross_old = [v for h in others for v in history[h][:-1]]
    rng.shuffle(cross_now)
    rng.shuffle(cross_old)
    for v in cross_now + cross_old + chain[:-3][::-1]:
        if len(roles) >= fam.n_candidates:
            break
        roles.setdefault(v, "cross_current" if v in cross_now else "cross_old" if v in cross_old else "older")
    c = fam.n_candidates
    return _finish(fam, _f3_render(register, events, target), gold, roles, s.gold_index[k], rng, last[1], 1 / c,
                   1 / c, {"N": n, "R": r, "register": register, "holder": target, "stale": stale,
                           "n_lines": len(events)})


def _gen_rebind(seeds: int, trials: int) -> list[Row]:
    fam = FAMILIES["F3"]
    cells = []
    for n in F3_N:
        for rk, r in (("0", 0), ("N", n)):
            for reg in F3_REGISTERS:
                cells.append((f"N{n}|R{rk}|{reg}", fam.ladder.index(f"N{n}|R{rk}") + 1, (n, r, reg)))
    return _run_cells(fam, cells, seeds, trials, _f3_sched, _f3_make)


# ── F4 nonce sequence induction ──────────────────────────────────────────────────────────────────────────────────

F4_HEADER = "Sequences:"
F4_PATTERNS: tuple[str, ...] = ("period", "mirror")


def f4_order(pattern: str, m: int) -> list[int]:
    """Copy position j repeats motif position order[j]."""
    if pattern == "period":
        return list(range(m))
    if pattern == "mirror":
        return list(range(m))[::-1]
    raise ValueError(f"unknown F4 pattern {pattern!r}")


def _f4_sched(rng: random.Random, n: int, params: tuple[int, int]) -> _Schedule:
    m, _ = params
    c = FAMILIES["F4"].n_candidates
    combos = [(p, j) for p in F4_PATTERNS for j in range(m)]
    return _Schedule(_balanced(rng, n, range(c)), _cycle(rng, NONCE, n), [False] * n, [False] * n,
                     _balanced(rng, n, combos))


def _f4_make(rng: random.Random, params: tuple[int, int], s: _Schedule, k: int) -> Optional[Row]:
    fam = FAMILIES["F4"]
    m, g = params
    pattern, j = s.aux[k]
    order = f4_order(pattern, m)
    gold: str = s.ident[k]
    width = m + g
    toks = rng.sample([w for w in NONCE if w != gold], (F4_EXAMPLES + 1) * width - 1)
    qmotif = toks[:m - 1]
    qmotif.insert(order[j], gold)
    qgap = toks[m - 1:width - 1]
    pool = toks[width - 1:]
    lines: list[list[str]] = []
    for e in range(F4_EXAMPLES):
        chunk = pool[e * width:(e + 1) * width]
        motif, gap = chunk[:m], chunk[m:]
        lines.append(motif + gap + [motif[x] for x in order])
    query = qmotif + qgap + [qmotif[x] for x in order[:j]]
    recent = lines[-1][width + j]
    roles: dict[str, str] = {gold: "gold", recent: "recent_answer"}
    for w in qmotif:
        roles.setdefault(w, "motif")
    for w in rng.sample(qgap, len(qgap)):
        if len(roles) < fam.n_candidates:
            roles.setdefault(w, "gap")
    earlier = [w for ln in lines for w in ln if w not in roles]
    rng.shuffle(earlier)
    for w in earlier:
        if len(roles) >= fam.n_candidates:
            break
        roles.setdefault(w, "other_line")
    c = fam.n_candidates
    lag = width + j - order[j]
    return _finish(fam, _lines(F4_HEADER, [" ".join(ln) for ln in lines], " ".join(query)), gold, roles,
                   s.gold_index[k], rng, recent, 1 / c, 1 / c,
                   {"m": m, "gap": g, "pattern": pattern, "copy_pos": j, "lag": lag})


def _gen_seqind(seeds: int, trials: int) -> list[Row]:
    fam = FAMILIES["F4"]
    cells = [(f"m{m}|g{g}", fam.ladder.index(f"m{m}|g{g}") + 1, (m, g)) for m in F4_M for g in F4_GAP]
    return _run_cells(fam, cells, seeds, trials, _f4_sched, _f4_make)


# ── C0 crystallized control ──────────────────────────────────────────────────────────────────────────────────────

def _c0_sched(rng: random.Random, n: int, params: tuple[str, int]) -> _Schedule:
    relation, _ = params
    c = FAMILIES["C0"].n_candidates
    return _Schedule(_balanced(rng, n, range(c)), _cycle(rng, RELATIONS[relation][1], n), [False] * n,
                     [False] * n, [None] * n)


def _c0_make(rng: random.Random, params: tuple[str, int], s: _Schedule, k: int) -> Optional[Row]:
    fam = FAMILIES["C0"]
    relation, shots = params
    header, facts = RELATIONS[relation]
    subj, gold, _ = s.ident[k]
    others = [f for f in facts if f[1] != gold]
    shown = rng.sample(others, shots)
    used = {f[1] for f in shown}
    distract = rng.sample([f[1] for f in others if f[1] not in used], fam.n_candidates - 1)
    prompt = _lines(header, [f"{a} → {d}" for a, _, d in shown], f"{subj} →")
    if any(count_word(prompt, w) for w in [gold, *distract]):
        return None                                     # a candidate named in the prompt would be a cue
    roles = {gold: "gold", **{w: "other" for w in distract}}
    c = fam.n_candidates
    return _finish(fam, prompt, gold, roles, s.gold_index[k], rng, shown[-1][1] if shown else None, 1 / c, 1 / c,
                   {"relation": relation, "shots": shots, "subject": subj})


def _gen_cryst(seeds: int, trials: int) -> list[Row]:
    fam = FAMILIES["C0"]
    cells = [(f"{rel}|shots{sh}", fam.ladder.index(f"shots{sh}") + 1, (rel, sh)) for sh in C0_SHOTS
             for rel in RELATIONS]
    return _run_cells(fam, cells, seeds, trials, _c0_sched, _c0_make)


_GENERATORS: dict[str, Callable[[int, int], list[Row]]] = {
    "F1": _gen_morph, "F2": _gen_algebra, "F3": _gen_rebind, "F4": _gen_seqind, "C0": _gen_cryst,
}


def generate(battery: str, seeds: int = SEEDS, trials: Optional[int] = None) -> list[Row]:
    """The prompt rows of one battery (``fluid-v1-morph`` … ``cryst-v1``). Deterministic: no global RNG state."""
    if battery not in BY_BATTERY:
        raise KeyError(f"unknown fluid battery {battery!r} (have: {', '.join(BY_BATTERY)})")
    fam = BY_BATTERY[battery]
    return _GENERATORS[fam.key](seeds, fam.trials if trials is None else trials)


# ── token checks ─────────────────────────────────────────────────────────────────────────────────────────────────

def check_pools(encoders: Mapping[str, Callable[[str], Sequence[int]]],
                pools: Mapping[str, Sequence[str]] = POOLS) -> list[str]:
    """Every pool word must encode to ONE token after a leading space, and a pool's tokens must be distinct, under
    every encoder (name → ``text → ids`` without special tokens). Returns the failures (empty = pass)."""
    bad: list[str] = []
    for tname, enc in encoders.items():
        for pname, words in pools.items():
            first: dict[int, str] = {}
            for w in words:
                ids = list(enc(" " + w))
                if len(ids) != 1:
                    bad.append(f"{tname}: {pname} {w!r} is {len(ids)} tokens")
                    continue
                if ids[0] in first:
                    bad.append(f"{tname}: {pname} {w!r} and {first[ids[0]]!r} share token {ids[0]}")
                first[ids[0]] = w
    return bad


# ── shortcut audit ───────────────────────────────────────────────────────────────────────────────────────────────

def audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per cell (``family:cell``): n, chance, the shortcut-follower accuracies against their neutral rates, the
    recency-subset size and the gold-position balance; under ``family:*`` (C0: ``C0/relation:*``), the gold-identity
    balance."""
    by: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[f"{r['family']}:{r['cell']}"].append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, sub in by.items():
        n = len(sub)
        c = len(sub[0]["candidates"])
        gated = [r for r in sub if r["neutral_recency"] is not None]
        fgated = [r for r in sub if r["neutral_frequency"] is not None]
        pos = Counter(r["gold_index"] for r in sub)
        out[key] = {
            "n": n, "chance": sum(1 / len(r["candidates"]) for r in sub) / n,
            "recency_subset": n - len(gated),
            "recency": (sum(r["gold_is_recent"] for r in gated) / len(gated)) if gated else None,
            "recency_neutral": (sum(r["neutral_recency"] for r in gated) / len(gated)) if gated else None,
            "recency_n": len(gated),
            "frequency": (sum(r["freq_follower"] for r in fgated) / len(fgated)) if fgated else None,
            "frequency_neutral": (sum(r["neutral_frequency"] for r in fgated) / len(fgated)) if fgated else None,
            "frequency_n": len(fgated),
            "novelty": sum(r["novelty_follower"] for r in sub) / n,
            "position_max_dev": max(abs(pos.get(i, 0) / n - 1 / c) for i in range(c)),
        }
    fams: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:                                      # one identity pool per family (C0: per relation)
        fams[r["family"] + (f"/{r['relation']}" if "relation" in r else "")][r["gold"]] += 1
    for f, cnt in fams.items():
        mean = sum(cnt.values()) / len(cnt)
        out[f"{f}:*"] = {"identities": len(cnt), "identity_max_ratio": max(cnt.values()) / mean,
                         "identity_min": min(cnt.values())}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    """A shortcut may not beat its neutral rate by more than max(tol, 1/n); gold positions sit within the same band
    of uniform; no gold identity is more than IDENTITY_MAX × the mean."""
    bad: list[str] = []
    for key, a in aud.items():
        if key.endswith(":*"):
            if a["identity_max_ratio"] > IDENTITY_MAX:
                bad.append(f"{key}: gold identity max/mean {a['identity_max_ratio']:.2f} > {IDENTITY_MAX}")
            continue
        if a["recency"] is not None:
            band = max(tol, 1 / a["recency_n"]) + 1e-9
            if a["recency"] > a["recency_neutral"] + band:
                bad.append(f"{key}: recency {a['recency']:.3f} vs neutral {a['recency_neutral']:.3f}")
        if a["frequency"] is not None:
            band = max(tol, 1 / a["frequency_n"]) + 1e-9
            if a["frequency"] > a["frequency_neutral"] + band:
                bad.append(f"{key}: frequency {a['frequency']:.3f} vs neutral {a['frequency_neutral']:.3f}")
        if a["position_max_dev"] > max(tol, 1 / a["n"]) + 1e-9:
            bad.append(f"{key}: gold position off uniform by {a['position_max_dev']:.3f}")
    return bad


# ── read-out ─────────────────────────────────────────────────────────────────────────────────────────────────────

Censor = Literal["", "high", "low", "both"]


@dataclass(frozen=True)
class Midpoint:
    value: float          # fitted rung (1 … R inside the ladder)
    beta: float
    sse: float
    censored: Censor
    rungs: int

    def label(self) -> str:
        return {"high": f">{self.rungs}", "low": "<1", "both": "n/m"}.get(self.censored, f"{self.value:.2f}")

    def score(self) -> Optional[float]:
        """The value that enters a fluid / crystallized score: R + 1 past the top, 0 below the bottom."""
        return {"high": self.rungs + 1.0, "low": 0.0, "both": None}.get(self.censored, self.value)


def fit_midpoint(rungs: Sequence[int], r: Sequence[float]) -> Midpoint:
    """Least-squares falling logistic in the rung index (grid: m over [0, R + 1], β over [.1, 8])."""
    if not rungs or len(rungs) != len(r):
        raise ValueError(f"need one r per rung (got {len(rungs)} rungs, {len(r)} values)")
    top = max(rungs)
    best = (math.inf, 0.0, 0.0)
    for i in range(241):
        m = i * (top + 1) / 240
        for j in range(1, 81):
            b = j * 0.1
            sse = sum((1 / (1 + math.exp(b * (x - m))) - y) ** 2 for x, y in zip(rungs, r))
            if sse < best[0]:
                best = (sse, m, b)
    sse, m, b = best
    high = r[-1] > 0.5 or m > top
    low = r[0] < 0.5 or m < 1
    cens: Censor = "both" if high and low else "high" if high else "low" if low else ""
    return Midpoint(m, b, sse, cens, top)


@dataclass
class FamilySummary:
    family: str
    model: str
    n_rows: int
    chance: float
    acc: dict[int, float]
    curve: dict[int, float]          # r(rung)
    midpoint: Midpoint
    auc: float
    multi_token_rows: int
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    stale: Optional[dict[str, Any]] = None          # F3
    by_shots: Optional[dict[str, dict[int, float]]] = None   # C0: relation (and "all") → shots → accuracy

    def as_dict(self) -> dict[str, Any]:
        return {"family": self.family, "model": self.model, "n_rows": self.n_rows, "chance": self.chance,
                "midpoint": self.midpoint.value, "midpoint_label": self.midpoint.label(),
                "midpoint_censored": self.midpoint.censored, "midpoint_score": self.midpoint.score(),
                "beta": self.midpoint.beta, "auc": self.auc, "acc": self.acc, "curve": self.curve,
                "multi_token_rows": self.multi_token_rows, "conditions": self.conditions, "stale": self.stale,
                "by_shots": self.by_shots}


def _pick(r: Mapping[str, Any]) -> str:
    lps = r["cand_lps"]
    return max(lps, key=lps.get)


def _curve(rows: Sequence[Mapping[str, Any]], chance: float) -> tuple[dict[int, float], dict[int, float]]:
    by: defaultdict[int, list[bool]] = defaultdict(list)
    for r in rows:
        by[int(r["rung"])].append(_pick(r) == r["gold"])
    acc = {k: sum(v) / len(v) for k, v in sorted(by.items())}
    return acc, {k: (a - chance) / (1 - chance) for k, a in acc.items()}


def summarize(rows: Sequence[Mapping[str, Any]]) -> FamilySummary:
    """One family's read-out for one model, from scored rows (``score.prompt_row`` shape)."""
    if not rows:
        raise ValueError("no rows")
    fams = {r["family"] for r in rows}
    if len(fams) != 1:
        raise ValueError(f"rows mix families {sorted(fams)}; summarize one battery at a time")
    fam = FAMILIES[fams.pop()]
    sizes = {len(r["candidates"]) for r in rows}
    if sizes != {fam.n_candidates}:
        raise ValueError(f"{fam.key}: candidate counts {sorted(sizes)}, want {fam.n_candidates}")
    chance = 1 / fam.n_candidates
    acc, curve = _curve(rows, chance)
    ks = list(curve)
    s = FamilySummary(family=fam.key, model=str(rows[0].get("model", "?")), n_rows=len(rows), chance=chance,
                      acc=acc, curve=curve, midpoint=fit_midpoint(ks, [curve[k] for k in ks]),
                      auc=sum(min(max(x, 0.0), 1.0) for x in curve.values()) / len(curve),
                      multi_token_rows=sum(bool(r.get("multi_token")) for r in rows))
    if fam.condition:
        groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[str(r[fam.condition])].append(r)
        for name, sub in sorted(groups.items()):
            a, cv = _curve(sub, chance)
            mid = fit_midpoint(list(cv), list(cv.values()))
            s.conditions[name] = {"n": len(sub), "acc": sum(_pick(r) == r["gold"] for r in sub) / len(sub),
                                  "midpoint_label": mid.label(), "midpoint_score": mid.score(), "acc_by_rung": a}
    if fam.key == "F3":
        stale = [_pick(r) == r["stale"] for r in rows]
        wrong = [r for r in rows if _pick(r) != r["gold"]]
        by_rung: defaultdict[int, list[bool]] = defaultdict(list)
        by_reg: defaultdict[str, list[bool]] = defaultdict(list)
        for r, x in zip(rows, stale):
            by_rung[int(r["rung"])].append(x)
            by_reg[str(r["register"])].append(x)
        roles = Counter(r["roles"].get(_pick(r), "?") for r in wrong)
        s.stale = {"rate": sum(stale) / len(stale),
                   "of_errors": (sum(_pick(r) == r["stale"] for r in wrong) / len(wrong)) if wrong else None,
                   "by_rung": {k: sum(v) / len(v) for k, v in sorted(by_rung.items())},
                   "by_register": {k: sum(v) / len(v) for k, v in sorted(by_reg.items())},
                   "error_roles": dict(roles)}
    if fam.key == "C0":
        cells: defaultdict[tuple[str, int], list[bool]] = defaultdict(list)
        for r in rows:
            ok = _pick(r) == r["gold"]
            cells[(str(r["relation"]), int(r["shots"]))].append(ok)
            cells[("all", int(r["shots"]))].append(ok)
        out: defaultdict[str, dict[int, float]] = defaultdict(dict)
        for (rel, sh), v in sorted(cells.items()):
            out[rel][sh] = sum(v) / len(v)
        s.by_shots = dict(out)
    return s


Stat = Literal["midpoint", "auc"]
Order = Literal["mean_then_z", "z_then_mean"]


def _value(s: FamilySummary, stat: Stat) -> Optional[float]:
    return s.midpoint.score() if stat == "midpoint" else s.auc


def _z(xs: Mapping[str, float]) -> dict[str, float]:
    if not xs:
        return {}
    mu = sum(xs.values()) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs.values()) / len(xs))
    return {k: (x - mu) / sd if sd > 0 else 0.0 for k, x in xs.items()}


def fluid_scores(summaries: Mapping[str, Mapping[str, FamilySummary]], stat: Stat = "midpoint",
                 order: Order = "mean_then_z") -> dict[str, dict[str, Any]]:
    """Fluid and crystallized scores over the models passed in (model → family key → summary).

    fluid         the mean over F1–F4 of each family's ``stat`` (censored midpoints at their bounds), z-scored
                  across the models that have all four (``order="z_then_mean"`` z-scores each family first, then
                  averages the z's)
    crystallized  C0's ``stat``, z-scored across the models that have it
    gap           fluid − crystallized (z units), where both exist
    A z-score needs ≥ 2 models; with fewer every z is 0. Models missing a family get None for that score."""
    raw: dict[str, dict[str, Optional[float]]] = {
        m: {f: _value(s, stat) for f, s in fs.items()} for m, fs in summaries.items()}
    full = {m: v for m, v in raw.items() if all(v.get(f) is not None for f in FLUID_KEYS)}
    if order == "mean_then_z":
        fluid = _z({m: sum(v[f] for f in FLUID_KEYS) / len(FLUID_KEYS) for m, v in full.items()})  # type: ignore[misc]
    elif order == "z_then_mean":
        per = {f: _z({m: v[f] for m, v in full.items()}) for f in FLUID_KEYS}  # type: ignore[misc]
        fluid = {m: sum(per[f][m] for f in FLUID_KEYS) / len(FLUID_KEYS) for m in full}
    else:
        raise ValueError(f"unknown order {order!r} (mean_then_z | z_then_mean)")
    cryst = _z({m: v["C0"] for m, v in raw.items() if v.get("C0") is not None})  # type: ignore[misc]
    out: dict[str, dict[str, Any]] = {}
    for m in summaries:
        f, c = fluid.get(m), cryst.get(m)
        out[m] = {"fluid": f, "crystallized": c, "gap": (f - c) if f is not None and c is not None else None,
                  "raw": raw[m]}
    return out
