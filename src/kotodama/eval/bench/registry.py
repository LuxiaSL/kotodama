"""The benchmark registry: one row per eval, with what it measures and how far to trust each reading of it.

:data:`kotodama.eval.binding.batteries.BATTERIES` pins the bytes (file and render digests). This table holds the
validity facts beside them, keyed by eval:

* ``construct``: what the eval claims to measure, in the benchmark's construct vocabulary (:data:`CONSTRUCTS`).
* ``batteries``: the pinned batteries that carry it (none for riders, whose items are generated at scoring time).
* ``recipe``: how its prompts are made again from this repository, or that they cannot be.
* ``clean``: the readings that count when two models are compared; every other reading is a diagnostic.
* ``floor``: the reading that bounds it from below (bare items, no worked examples), when there is one.
* ``flags``: audit findings, each scoped to the panel it was measured on. A flag says a reading is not a clean
  measure of the construct; it never says the model is weak.
* ``audit``: how far the rendering audit has reached this eval.

An eval changes only on validity grounds (rendering invariance, insensitivity to off-construct levers, item
shortcuts), never because a model scores low; a re-rendering has to shrink the spread across renderings, not raise
the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

Audit = Literal["audited", "item-audited", "pending"]

CONSTRUCTS: dict[str, str] = {
    "static-lookup": "retrieve which entity holds which attribute when nothing changes",
    "update": "track an entity's state through changes (transfers, reassignments, mutations)",
    "set-binding": "bind several attributes or members to one entity",
    "deixis": "resolve I / you / my to the right person in a dialogue",
    "stage-gauge": "the swap-shift gauges of the staircase: whether a pronoun's answer moves with the speaker",
    "capacity": "how many bindings fit in context before retrieval fails (K50)",
    "fluid": "learn a new rule from the examples in the prompt",
    "crystallized": "recall a known fact in the same few-shot frame (the control for fluid)",
    "form-vs-update": "the cost of updating a binding against forming it, paired per item",
}


@dataclass(frozen=True)
class Flag:
    reading: str        # the rendering or condition the finding is about
    finding: str
    panel: str          # where it holds, model-relative: "6/6 eligible models", "tsumugi"
    test: str           # the test that found it


@dataclass(frozen=True)
class Eval:
    name: str
    construct: str
    source: Literal["binding", "bindprobes", "riders"]
    what: str
    batteries: tuple[str, ...] = field(default_factory=tuple)
    recipe: str = ""
    clean: tuple[str, ...] = field(default_factory=tuple)
    floor: str | None = None
    flags: tuple[Flag, ...] = field(default_factory=tuple)
    audit: Audit = "pending"


# bp-v3.2's recipe: the v3 families at their current versions, emitted with this seed and per-cell sizes
# (kotodama.eval.bindprobes.items.emit); the emit is byte-identical to the pinned file.
BP_V32_RECIPE: Mapping[str, object] = {"seed": 2031, "per_cell": {"setbind": 64, "answer": 160, "adjacency": 225, "codevarmut": 100}}

_BP_CLEAN = ("0shot", "fair")   # bare items are the floor; `fair` (four exemplars balanced on every cell component) is the few-shot reading
_FROZEN = "frozen file; its generator is not in this repository (the pin is the only definition)"

REGISTRY: dict[str, Eval] = {e.name: e for e in (
    # ---- bindprobes: one family per construct, read under every condition of bindprobes.items.CONDITIONS
    Eval("bp-attr", "static-lookup", "bindprobes", "who holds which attribute, in prose", ("bp-profile",),
         _FROZEN, _BP_CLEAN, "0shot",
         (Flag("0shot", "format-loaded: worked examples lift it by 8 to 52 points", "8-model panel",
               "rendering spread, 0shot vs fair"),),
         "audited"),
    Eval("bp-codevar", "update", "bindprobes", "a variable's value after reassignments, in code", ("bp-profile",),
         _FROZEN, _BP_CLEAN, "0shot",
         (Flag("3shot", "shortcut: exemplars that match the item's cell can be copied", "8-model panel",
               "mixed-shot comparison"),),
         "audited"),
    Eval("bp-dialogue", "update", "bindprobes", "who has what after objects change hands in a dialogue", ("bp-profile",),
         _FROZEN, _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-deixis", "deixis", "bindprobes", "person deixis: deictic, third-person and twin variants",
         ("bp-deixis-v2", "bp-deixis-lite"), _FROZEN, _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-setbind", "set-binding", "bindprobes", "several members bound to one set", ("bp-v3.2", "bp-v3.1"),
         "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _FROZEN, _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-answer", "deixis", "bindprobes", "production deixis: who answers, from the reply's own words",
         ("bp-v3.2", "bp-v3.1"), "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _FROZEN, _BP_CLEAN, "0shot",
         (Flag("3shot", "exemplars mislead: below 0shot", "8/8 models", "condition comparison"),),
         "audited"),
    Eval("bp-adjacency", "deixis", "bindprobes", "structural you: the reply's addressee", ("bp-v3.2", "bp-v3.1"),
         "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _FROZEN, _BP_CLEAN, "0shot", (), "audited"),
    Eval("bp-codevarmut", "update", "bindprobes", "aliased lists: a mutation seen through another name",
         ("bp-v3.2", "bp-v3.1"), "bp-v3.2: BP_V32_RECIPE; bp-v3.1: " + _FROZEN, _BP_CLEAN, "0shot",
         (Flag("3shot", "shortcut: cell-matched exemplars teach the answer's role; flipping their answers collapses "
                        "the score", "6/6 eligible models", "flipped-exemplar test"),),
         "audited"),
    # ---- the rows batteries of the binding suite
    Eval("gauges", "stage-gauge", "binding", "static and dynamic swap shifts: the staircase's gauges",
         ("joinlite-v1", "swapdyn-v1"), _FROZEN, ("0shot",), "0shot",
         (Flag("joinlite third-person query", "off-construct: does not group with the other deixis readings under fair",
               "construct matrix, 8 and 20 models", "construct matrix"),),
         "item-audited"),
    Eval("stage5-decomp", "stage-gauge", "binding", "stage-5 decomposition: third / deictic / named transfer / vocative",
         ("stage5-decomp-v1", "stage5-decomp-core"), _FROZEN, ("0shot",), "0shot",
         (Flag("deictic variant", "byte-identical to swapdyn-v1's original rows: one measurement, not two", "all models",
               "item comparison"),),
         "pending"),
    Eval("capacity", "capacity", "binding", "binding capacity, K50 after arXiv 2609.30634", ("capacity-v1",),
         "python -m kotodama.eval.binding gen-capacity", ("K50",), None, (), "pending"),
    Eval("fluid", "fluid", "binding", "fluid F1-F4: nonce morphology, symbol algebra, rebinding chains, sequence induction",
         ("fluid-v1-morph", "fluid-v1-algebra", "fluid-v2-algebra", "fluid-v1-rebind", "fluid-v2-rebind", "fluid-v1-seqind"),
         "python -m kotodama.eval.binding gen-fluid", ("ladder AUC",), None,
         (Flag("F2-F4 against C0", "C0 shares F1's frame only; F2-F4 need a frame-matched control", "by construction",
               "frame comparison"),),
         "pending"),
    Eval("crystallized", "crystallized", "binding", "C0: known facts in the fluid few-shot frame", ("cryst-v1",),
         "python -m kotodama.eval.binding gen-fluid", ("shots 0 / 1 / 3 / 6",), None, (), "pending"),
    Eval("formupdate", "form-vs-update", "binding", "forming vs updating a binding, paired per base item",
         ("formupdate-v2", "formupdate-v1"), "python -m kotodama.eval.binding gen-formupdate", ("paired cost",), None,
         (), "pending"),
    # ---- riders: generated at scoring time from kotodama.eval.riders (the in-loop slice: riders.lite)
    Eval("riders", "update", "riders", "entity-state tracking: boxes and operations, the same items in four registers",
         (), "kotodama.eval.riders.lite.select_items (seeded); no prompts file", ("0shot, per register", "3shot, code"),
         "0shot",
         (Flag("0shot", "register-dependent: the registers spread by 20 points or more", "5/8 models",
               "four-register spread"),
          Flag("pooled prose (narrative + screenplay + chat)", "pooling hides a format term of up to 40 points", "4/8 models",
               "four-register spread"),
          Flag("3shot, prose registers", "fixed exemplars hurt prose and help code", "chat 8/8, screenplay 6/8, narrative 5/8; "
               "code helped 8/8", "condition comparison")),
         "audited"),
)}


def evals_for(battery: str) -> list[Eval]:
    """The registry rows a pinned battery carries."""
    return [e for e in REGISTRY.values() if battery in e.batteries]
