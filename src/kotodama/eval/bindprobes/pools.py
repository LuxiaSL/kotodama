"""Bindprobe pools: names (gendered), objects, code variables — split shot/item.

Every SCORED string is a single token in its scored form:

* names    — scored as ``" Name"`` (leading space): the ``dialogue`` holder query.
             Gendered, because the ``dialogue`` family resolves he/him/she/her.
* objects  — scored as ``" obj"`` (leading space): every family. The pool is the
             riders' banked single-token pool (``riders.items.BANKED_VERIFIED_NAMES``),
             reused, not restated.
* variables — never scored (the ``codevar`` candidates are objects), so only
             kept short and distinct.

``codevar`` scores the object after ``# output:`` — i.e. in the leading-space form
``" cup"``, the same form as every other family. The REPL form (``>>> print(c)``
then a bare ``cup`` / ``'cup'`` on the next line) was checked and rejected: only
36/78 pool objects are single-token without the leading space under SmolLM2, and
after a quote BPE merges contractions (``'dice`` -> ``'d`` + ``ice``).

Shot/item split: exemplars draw from the SHOT pools, test items from the ITEM
pools, so ``3shot`` exemplars are collision-free by construction (no shared name,
variable or object). ``3shot_collide`` re-labels the exemplars INTO the test
item's names/variables (objects stay shot-pool); see ``items.collide_shots``.

``verify_names`` re-checks the candidates against tokenizers (default: every
riders grid tokenizer) and writes a JSON that ``gen --names`` accepts; the
banked pool below is what it produced over SmolLM2 (+ the GPT-NeoX, Llama-3 and
Qwen2.5 vocabularies, checked on locally cached same-vocab checkpoints).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from kotodama.eval.riders.items import BANKED_VERIFIED_NAMES as RIDER_OBJECTS
from kotodama.eval.riders.items import GRID_TOKENIZERS

Gender = Literal["f", "m"]

# Curated: clearly gendered, not common nouns (no Rose/Grace/Will/Mark/Max), no
# near-duplicates inside the pool (Anna but not Ann/Anne; Peter but not Pete).
FEMALE_CANDIDATES: tuple[str, ...] = (
    "Anna", "Emma", "Lucy", "Kate", "Jane", "Mary", "Sarah", "Ruth", "Alice",
    "Clara", "Julia", "Laura", "Helen", "Amy", "Beth", "Jill", "Maya", "Nina",
    "Olivia", "Lily", "Hannah", "Rachel", "Sophie", "Susan", "Linda", "Diana",
    "Wendy", "Carol", "Ellen", "Molly", "Erin", "Lisa", "Emily", "Karen", "Nancy",
    "Betty", "Sandra", "Martha", "Judy", "Rosa", "Ada", "Jessica", "Amanda",
)
MALE_CANDIDATES: tuple[str, ...] = (
    "Tom", "Ben", "Leo", "Paul", "Adam", "Hugo", "Owen", "Ivan", "Omar", "Carl",
    "Dan", "Eric", "Fred", "Greg", "Jack", "Jake", "Joe", "Kurt", "Luke", "Matt",
    "Mike", "Nick", "Peter", "Rick", "Ryan", "Sean", "Seth", "Ted", "Tim", "Tony",
    "Walt", "John", "James", "David", "Henry", "George", "Frank", "Steve", "Kevin",
    "Brian", "Gary", "Larry", "Jerry", "Simon", "Victor", "Oscar", "Felix", "Bruce",
    "Roger", "Neil", "Ross", "Roy", "Hal", "Jim", "Ken",
)

# The banked pool: ``verify_names`` over the candidates above —
# every candidate passed (" Name" is one token in SmolLM2, GPT-NeoX, Llama-3,
# Qwen2.5). Names that FAILED SmolLM2 and were dropped before curation include
# Mara, Theo, Nora, Lena (two tokens with a leading space).
BANKED_FEMALE: tuple[str, ...] = FEMALE_CANDIDATES
BANKED_MALE: tuple[str, ...] = MALE_CANDIDATES
BANKED_POOL_TAG = "<banked:bindprobes-names-2026-10-12>"

# Variables: single lowercase letters, no i/l/o. Disjoint shot/item halves.
ITEM_VARS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h", "k", "m", "n")
SHOT_VARS: tuple[str, ...] = ("p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z")

N_SHOT_NAMES_PER_GENDER = 8   # attr exemplars need up to 8 distinct names
N_SHOT_OBJECTS = 16           # attr exemplars need up to 8 bound + 2 free objects

# ── Object grammar ─────────────────────────────────────────────────────────────
# Pool objects that are not singular count nouns: mass nouns ("A glue lies on the
# shelf" is ungrammatical) and the plural-form "dice" ("the dice is", "passes it").
# The pool itself stays whole, so the shot/item split and every family that never
# puts an object in such a slot are unchanged; a family that does draws through
# :func:`countable` or :func:`singular`.
NON_COUNTABLE: frozenset[str] = frozenset({"dice", "glue", "chalk", "soap", "thread",
                                           "tape", "wire", "straw"})
PLURAL_FORM: frozenset[str] = frozenset({"dice"})
# words whose spelling misleads the vowel rule (sound, not letter, picks the article)
_AN_BEFORE_CONSONANT = ("hour", "honest", "honour", "heir")
_A_BEFORE_VOWEL = ("uni", "use", "usu", "eu", "one", "once")


def indefinite(noun: str) -> str:
    """``"a"`` or ``"an"`` for a singular count noun; raises ``ValueError`` on a pool
    object that takes no indefinite article (:data:`NON_COUNTABLE`)."""
    w = noun.strip().lower()
    if not w:
        raise ValueError("indefinite article of an empty word")
    if w in NON_COUNTABLE:
        raise ValueError(f"{noun!r} is not a singular count noun: no indefinite article")
    if w.startswith(_AN_BEFORE_CONSONANT):
        return "an"
    if w.startswith(_A_BEFORE_VOWEL):
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def countable(objects: Sequence[str]) -> tuple[str, ...]:
    """The objects that read as singular count nouns ("a ___", "the ___ is", "it")."""
    return tuple(o for o in objects if o not in NON_COUNTABLE)


def singular(objects: Sequence[str]) -> tuple[str, ...]:
    """The objects that agree with a singular verb and "it" (mass nouns kept)."""
    return tuple(o for o in objects if o not in PLURAL_FORM)


@dataclass(frozen=True)
class Pools:
    """One side (shot or item) of the split. Names keep their gender."""
    female: tuple[str, ...]
    male: tuple[str, ...]
    objects: tuple[str, ...]
    variables: tuple[str, ...]
    gender: dict[str, Gender] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        g: dict[str, Gender] = {n: "f" for n in self.female}
        g.update({n: "m" for n in self.male})
        object.__setattr__(self, "gender", g)

    @property
    def names(self) -> tuple[str, ...]:
        return self.female + self.male


@dataclass(frozen=True)
class PoolSplit:
    shot: Pools
    item: Pools
    source: str  # BANKED_POOL_TAG or the names-file path


def split_pools(female: Sequence[str], male: Sequence[str],
                objects: Sequence[str] = RIDER_OBJECTS, source: str = BANKED_POOL_TAG
                ) -> PoolSplit:
    """Deterministic shot/item split (shot = the first slice of each pool)."""
    k, ko = N_SHOT_NAMES_PER_GENDER, N_SHOT_OBJECTS
    if min(len(female), len(male)) < k + 8 or len(objects) < ko + 16:
        raise ValueError(f"pools too small: {len(female)} female, {len(male)} male, "
                         f"{len(objects)} objects (need >= {k + 8}/{k + 8}/{ko + 16})")
    for label, pool in (("female", female), ("male", male), ("objects", objects)):
        if len(set(pool)) != len(pool):
            raise ValueError(f"{label} pool has duplicates")
    if set(female) & set(male):
        raise ValueError(f"names in both genders: {sorted(set(female) & set(male))}")
    shot = Pools(tuple(female[:k]), tuple(male[:k]), tuple(objects[:ko]), SHOT_VARS)
    item = Pools(tuple(female[k:]), tuple(male[k:]), tuple(objects[ko:]), ITEM_VARS)
    return PoolSplit(shot=shot, item=item, source=source)


def load_pools(names_path: Optional[Path] = None) -> PoolSplit:
    """The banked split, or the split of a ``verify_names`` JSON."""
    if names_path is None:
        return split_pools(BANKED_FEMALE, BANKED_MALE)
    try:
        d = json.loads(names_path.read_text())
        female, male = d["verified"]["female"], d["verified"]["male"]
        objects = d["verified"].get("objects", list(RIDER_OBJECTS))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"unreadable names file {names_path}: {exc}") from exc
    for x in (female, male, objects):
        if not isinstance(x, list) or not all(isinstance(s, str) for s in x):
            raise ValueError(f"{names_path}: verified pools must be lists of strings")
    return split_pools(female, male, objects, source=str(names_path))


# ── Tokenizer verification ─────────────────────────────────────────────────────

def multi_token(tokenizer: Any, words: Sequence[str]) -> dict[str, int]:
    """``{word: n_tokens}`` for every word whose scored form ``" word"`` is NOT one token."""
    out: dict[str, int] = {}
    for w in words:
        n = len(tokenizer(" " + w, add_special_tokens=False).input_ids)
        if n != 1:
            out[w] = n
    return out


def check_split(split: PoolSplit, tokenizer: Any) -> None:
    """Raise if any scored string (names, objects) of either side is multi-token."""
    words = [w for p in (split.shot, split.item) for w in p.names + p.objects]
    bad = multi_token(tokenizer, words)
    if bad:
        raise ValueError(f"multi-token scored strings under this tokenizer: {bad}")


def load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer  # deferred: heavy import
    return AutoTokenizer.from_pretrained(name)


def verify_names(out_path: Path, tokenizer_keys: Sequence[str] = tuple(GRID_TOKENIZERS)
                 ) -> dict[str, Any]:
    """Write the names (by gender) + objects that are single-token in every
    selected tokenizer (keys of ``riders.items.GRID_TOKENIZERS``)."""
    unknown = [k for k in tokenizer_keys if k not in GRID_TOKENIZERS]
    if unknown:
        raise ValueError(f"unknown tokenizer keys {unknown}; choose from {list(GRID_TOKENIZERS)}")
    toks = {}
    for k in tokenizer_keys:
        print(f"loading tokenizer {k} = {GRID_TOKENIZERS[k]}", flush=True)
        toks[k] = load_tokenizer(GRID_TOKENIZERS[k])
    rejected: dict[str, dict[str, int]] = {}
    for k, tok in toks.items():
        for w, n in multi_token(tok, FEMALE_CANDIDATES + MALE_CANDIDATES
                                + tuple(RIDER_OBJECTS)).items():
            rejected.setdefault(w, {})[k] = n
    verified = {
        "female": [n for n in FEMALE_CANDIDATES if n not in rejected],
        "male": [n for n in MALE_CANDIDATES if n not in rejected],
        "objects": [o for o in RIDER_OBJECTS if o not in rejected],
    }
    report = {"tokenizers": {k: GRID_TOKENIZERS[k] for k in toks}, "verified": verified,
              "rejected": rejected,
              "n_verified": {k: len(v) for k, v in verified.items()}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"verified {report['n_verified']} -> {out_path}", flush=True)
    try:
        split_pools(verified["female"], verified["male"], verified["objects"])
    except ValueError as exc:
        print(f"WARNING: verified pool cannot be split: {exc}", file=sys.stderr)
    return report
