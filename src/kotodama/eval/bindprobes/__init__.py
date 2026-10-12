"""Bindprobes: three referent-binding probe families shaped UNLIKE the riders.

The riders (``kotodama.eval.riders``) track objects through box operations in
four registers. Bindprobes asks whether a binding improvement survives a change
of SHAPE, with the same judge-free closed-candidate logprob read:

  attr      static attribute binding — N ∈ {2,4,6,8} people each keep one object
            (+ unbound distractor objects); one is queried late. Controls: query
            position (first/middle/last declared), recency (gold is / is not the
            last-mentioned object).
  codevar   variable binding in Python — literals, copies, reassignments, swaps
            via a temp; alias-chain depth ∈ {0..3} × interfering ops ∈ {1,2,4};
            value read after ``print(x)  # output:``. Value semantics simulated.
  dialogue  multi-party prose scene — 3–4 people hand 3 objects around, partly
            through it/them/he/him/she/her that resolve only by tracking;
            transfers ∈ {1,2,4}; query the holder of an object or the object of
            a holder.

Every item carries its own 3 exemplars, so each is scored under three conditions:
``0shot``; ``3shot`` (exemplars from DISJOINT shot pools — collision-free by
construction); ``3shot_collide`` (the same exemplars re-labeled into the item's
own names/variables, different bindings — cross-episode referent collision, the
analogue of the riders' re-lettering contrast).

Modules: ``pools`` (verified single-token names/objects/variables, shot/item
split, ``verify_names``), ``attr`` / ``codevar`` / ``dialogue`` (spec, simulator,
renderer, re-labeler, error typer per family), ``items`` (generation, conditions,
prompts, validation, selftest), ``grid`` (rows against ``/logprobs`` — the riders'
HTTP client and closed-candidate read, reused), ``aggregate`` (tables),
``lite`` (balanced ~200/family slice for later in-loop use; NOT wired into training).

Scoring is ``kotodama.eval.riders.score`` behind the server's ``/logprobs``:
candidates are sent as ``" " + c`` and scored by their FIRST token; a multi-token
candidate is a name-verification failure (aggregate hard-fails).

CLI: ``python -m kotodama.eval.bindprobes {gen,verify-names,selftest,run,aggregate}``.
"""

from kotodama.eval.bindprobes import (aggregate, attr, base, codevar, dialogue, errors, grid,
                                      items, lite, pools)
from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.items import (CONDITIONS, FAMILY_NAMES, build_prompt, emit,
                                            generate, load_items)

__all__ = [
    "CONDITIONS",
    "FAMILY_NAMES",
    "Item",
    "aggregate",
    "attr",
    "base",
    "build_prompt",
    "codevar",
    "dialogue",
    "emit",
    "errors",
    "generate",
    "grid",
    "items",
    "lite",
    "load_items",
    "pools",
]
