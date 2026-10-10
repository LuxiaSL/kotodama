"""Rider error typing: what does a model pick when it is WRONG?

A rider row scores closed-candidate logprobs (no generation), so a wrong row's
"answer" is its argmax candidate. That pick is typed against the item's own
structure. Fine labels (types can overlap; :func:`error_types` returns every
match in this priority order, the first one is the label):

  init_query   the queried box's INITIAL object (the ops were not applied)
  box_F        what Box F holds at the end, when the query is not F (every
               standard exemplar shot queries Box F)
  stale_query  an object that passed through the queried box and was replaced
               (a partial update)
  other_final  another box's FINAL object (right state, wrong box)
  other_init   another box's initial object
  last_ment    the last-mentioned object in the item text
  other        none of the above

Coarse labels (:func:`coarse_error_type`, the in-loop telemetry split):

  init_query   pick == the queried box's initial object
  wrong_box    pick is another box's initial or final object (and not init_query)
  other        anything else
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from kotodama.eval.riders.items import Item

FINE_TYPES: tuple[str, ...] = ("init_query", "box_F", "stale_query", "other_final",
                               "other_init", "last_ment", "other")
CoarseType = Literal["init_query", "wrong_box", "other"]
COARSE_TYPES: tuple[CoarseType, ...] = ("init_query", "wrong_box", "other")


def picked(row: Mapping[str, Any]) -> str:
    """The argmax candidate of a grid row (``cand_lps``: object -> logprob).
    Ties go to the first candidate in ``cand_lps`` order."""
    lps: Mapping[str, float] = row["cand_lps"]
    if not lps:
        raise ValueError(f"row {row.get('item_id')!r} has no cand_lps")
    return max(lps, key=lambda k: lps[k])


def passed_through(item: Item, box: str) -> set[str]:
    """Every object ``box`` held at any point (``items._sim`` semantics), minus its final."""
    state: dict[str, Optional[str]] = dict(item.boxes0)
    held: set[Optional[str]] = {item.boxes0.get(box)}
    for op in item.ops:
        k, i, j = op["kind"], op["i"], op.get("j")
        if k == "swap":
            state[i], state[j] = state[j], state[i]
        elif k == "move":
            state[j] = state[i]
            state[i] = None
        elif k in ("put", "overwrite"):
            state[i] = op["obj"]
        else:
            raise ValueError(f"{item.item_id}: unknown op kind {k!r}")
        held.add(state.get(box))
    final = item.final.get(box)
    return {o for o in held if o and o != final}


def error_types(item: Item, pick: str) -> list[str]:
    """Every fine label that applies to ``pick`` (priority order; never empty)."""
    q = item.query_box
    t: list[str] = []
    if pick == item.boxes0.get(q):
        t.append("init_query")
    if q != "F" and pick == item.final.get("F"):
        t.append("box_F")
    if pick in passed_through(item, q):
        t.append("stale_query")
    if any(pick == v for b, v in item.final.items() if b != q):
        t.append("other_final")
    if any(pick == v for b, v in item.boxes0.items() if b != q):
        t.append("other_init")
    if item.mention_order and pick == item.mention_order[-1]:
        t.append("last_ment")
    return t or ["other"]


def coarse_error_type(item: Item, pick: str) -> CoarseType:
    """init_query > wrong_box > other (see the module doc)."""
    q = item.query_box
    if pick == item.boxes0.get(q):
        return "init_query"
    if (any(pick == v for b, v in item.final.items() if b != q)
            or any(pick == v for b, v in item.boxes0.items() if b != q)):
        return "wrong_box"
    return "other"
