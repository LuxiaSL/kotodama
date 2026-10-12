"""Bindprobe error typing: what a WRONG row picked, typed against the item.

The pick is the row's argmax candidate (``riders.errors.picked``, reused). Each
family's typer returns every label that applies, primary first; the primary
label is the one tabulated. Labels per family (priority order):

  attr      wrong_binding (another entity's object; + ``adjacent`` tag when that
            entity was declared next to the queried one) > distractor (an unbound
            object) > other
  codevar   stale (a value the queried var or its provenance path held before
            the value read) > alias (another var's final value) > init (an
            initial literal) > other
  dialogue  previous (earlier holder / object held earlier) > other_person > other
"""

from __future__ import annotations

from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.items import family
from kotodama.eval.riders.errors import picked

__all__ = ["error_types", "primary", "picked", "types_for"]


def error_types(item: Item, pick: str) -> list[str]:
    """Every label for ``pick`` (primary first). Raises if ``pick`` is the gold."""
    if pick == item.gold:
        raise ValueError(f"{item.item_id}: the pick is the gold — not an error")
    return family(item.family).error_types(item, pick)


def primary(item: Item, pick: str) -> str:
    return error_types(item, pick)[0]


def types_for(family_name: str) -> tuple[str, ...]:
    """The tabulated (primary) labels of a family."""
    return family(family_name).ERROR_TYPES
