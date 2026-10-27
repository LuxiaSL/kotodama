"""The bindprobe item record + the per-family interface.

One :class:`Item` shape for all families; the family-specific structure lives in
``spec`` (a plain dict, see each family module's ``Spec`` TypedDict) and is the
single source the renderer, the simulator, the re-labeler and the error typer
all read. ``gold`` is computed by the family's simulator at construction and is
re-checked by :func:`kotodama.eval.bindprobes.items.validate`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from kotodama.eval.bindprobes.pools import Pools

Cell = tuple[Any, ...]


@dataclass
class Item:
    """One probe item. Field order is the JSONL key order — do not reorder."""
    item_id: str
    family: str
    cell: list[Any]            # difficulty tuple (JSON list)
    spec: dict[str, Any]       # family structure: render + simulate from this
    gold: str                  # from the simulator
    candidates: list[str]      # closed set incl. gold, fixed order; scored as " " + c
    mention_order: list[str]   # candidate-type strings in text-mention order
    meta: dict[str, Any]       # controls (recency, query position/kind, ...)
    shots: list[dict[str, Any]] = field(default_factory=list)   # exemplar Items (asdict)
    # per shot: shot label -> item label (positional, gender-preserving); the
    # ``3shot_collide`` condition renders the shots re-labeled through these
    collide_maps: list[dict[str, str]] = field(default_factory=list)

    def shot_items(self) -> list["Item"]:
        return [Item.from_dict(s) for s in self.shots]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Item":
        return Item(**dict(d))

    @property
    def cell_key(self) -> str:
        return cell_key(self.cell)


def cell_key(cell: Sequence[Any]) -> str:
    """``[4, "first"]`` -> ``"4/first"`` (table/summary key)."""
    return "/".join(str(c) for c in cell)


class FamilyModule(Protocol):
    """What ``items`` needs from a family module."""
    FAMILY: str
    CELLS: list[Cell]
    DEFAULT_PER_CELL: int
    SEPARATOR: str
    ERROR_TYPES: tuple[str, ...]

    def gen_item(self, rng: random.Random, cell: Cell, pools: Pools, item_id: str,
                 k: int, per_cell: int, template: Item | None = None) -> Item:
        """``k``/``per_cell`` balance within-cell controls; ``template`` (the test
        item, when generating its exemplars) pins whatever the collide re-label
        must preserve (e.g. the gender pattern)."""
        ...
    def simulate(self, item: Item) -> str: ...
    def render(self, item: Item, with_answer: bool) -> str: ...
    def labels(self, item: Item) -> list[str]: ...
    def relabel(self, item: Item, mapping: Mapping[str, str],
                target: Item | None = None) -> Item:
        """``item`` with its labels renamed through ``mapping`` (re-simulated);
        ``target`` = the item whose labels these are (dialogue reads genders)."""
        ...
    def error_types(self, item: Item, pick: str) -> list[str]: ...


def the_list(objs: Sequence[str]) -> str:
    """``the cup``, ``the cup and the pen``, ``the cup, the pen and the key``."""
    parts = [f"the {o}" for o in objs]
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def name_list(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def rejection(fn: Callable[[], Item | None], what: str, tries: int = 2000) -> Item:
    """Call ``fn`` until it returns an item (``None`` = rejected draw)."""
    for _ in range(tries):
        it = fn()
        if it is not None:
            return it
    raise RuntimeError(f"could not generate {what} in {tries} draws")
