"""Per-request steering levers for the fast engine: write plans and the lever tables the decoders read.

A LEVER is one request's whole steering write, resolved to the effective layers it touches. Two write sites:

  * ``attn``: the attention sublayer's output at an effective layer, before it joins the running AttnRes partial
    (``partial + (attn_out + row)``). The block-persistent rules K/N/S (``write_plan``) decide which later layers
    re-assert a site's row.
  * ``mlp``: the MLP sublayer's output (``partial + (mlp_out + row)``), written once; a neuron unit ``(L, n)`` writes
    ``dose * W_down[:, n]`` here (``neuron_row``).

Every write is keyed on the EFFECTIVE layer: in a looped model the repeats of the core run the same modules, and a
write at one pass must not land at the other.

``LeverTables`` holds, per site kind, one table ``[n_layers, 1 + max_levers, hidden]`` in the engine dtype. Row 0
of every layer is the identity and holds ``-0.0`` in every element, never written: IEEE addition gives
``x + (-0.0) == x`` bit for bit for every non-NaN ``x``, so a row reading slot 0 computes exactly what an engine
without tables computes at that add. ``bind`` writes a lever's rows into a free slot (one slot index across every
layer; a layer the lever does not write keeps ``-0.0`` in that slot) and returns the slot; a decode row reads its
slot through ``index_select``, so every replay of a captured graph reads the slot and table values current at
replay. Tables are allocated once and only written in place.

The span is per lever: ``uniform`` writes every position (prompt included: a steered prefill reads the slot), and
``decode`` writes generated positions only (the prefill stays unsteered and is shared across rows).
"""

from __future__ import annotations

import hashlib
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence

import numpy as np
import torch

KINDS: tuple[str, ...] = ("attn", "mlp")
SPANS: tuple[str, ...] = ("uniform", "decode")
IDENTITY_SLOT: int = 0
Span = Literal["uniform", "decode"]
Rule = Literal["K", "N", "S"]


class LeverError(ValueError):
    """A plan, slot or table is not in the state the call needs."""


def write_plan(sites: Sequence[int], rows: np.ndarray, boundaries: Iterable[int], n_layers: int, rule: Rule = "K",
               norms: Optional[Mapping[int, float]] = None) -> dict[int, np.ndarray]:
    """Effective layer -> the float32 row added to that layer's attention output.

    ``rows[j]`` (float32, dose already baked in) is site ``sites[j]``'s row. Rule K re-asserts it unchanged at the
    first attention of every later AttnRes block (``compute_steer_points``); rule N re-asserts it scaled by
    ``norms[point] / norms[site]`` (the typical partial norm at each point, so the row keeps its size relative to the
    partial it lands in; norms are needed at those points only); rule S writes at the site only. Rows of several sites sum where their points overlap, in
    site order. The arithmetic is float32 throughout, row by row, so two implementations of this law agree bit for
    bit.
    """
    from kotodama.model.decode_engine import compute_steer_points  # the engine module imports this one

    if rule not in ("K", "N", "S"):
        raise LeverError(f"write rule must be K, N or S, got {rule!r}")
    rows = np.asarray(rows, dtype=np.float32)
    sites = [int(s) for s in sites]
    if rows.ndim != 2 or rows.shape[0] != len(sites):
        raise LeverError(f"rows must be [n_sites={len(sites)}, hidden], got shape {rows.shape}")
    if len(set(sites)) != len(sites):
        raise LeverError(f"duplicate sites {sites}: sum the rows of a double write before planning it")
    bset = sorted({int(b) for b in boundaries})
    if rule == "N" and norms is None:
        raise LeverError("write rule N needs the per-layer partial norms")
    plan: dict[int, np.ndarray] = {}
    for row, site in zip(rows, sites):
        points = (site,) if rule == "S" else compute_steer_points(site, bset, n_layers)
        if rule == "N":
            bad = [p for p in points if not float(norms.get(p, 0.0)) > 0]
            if bad:
                raise LeverError(f"rule N needs a positive partial norm at layers {bad} (site {site})")
        for layer in points:
            add = row if rule != "N" else row * np.float32(float(norms[layer]) / float(norms[site]))
            plan[layer] = plan[layer] + add if layer in plan else add.copy()
    return plan


@dataclass(frozen=True)
class LeverPlan:
    """One lever: ``attn`` / ``mlp`` map effective layers to the float32 row added at that site (dose baked in),
    and ``span`` says which positions it writes."""

    attn: Mapping[int, np.ndarray] = field(default_factory=dict)
    mlp: Mapping[int, np.ndarray] = field(default_factory=dict)
    span: Span = "uniform"

    def __post_init__(self) -> None:
        if self.span not in SPANS:
            raise LeverError(f"span must be one of {SPANS}, got {self.span!r}")
        for kind in KINDS:
            rows = {}
            for layer, row in getattr(self, kind).items():
                if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
                    raise LeverError(f"{kind} plan keys must be effective layer ints, got {layer!r}")
                arr = np.ascontiguousarray(row, dtype=np.float32)
                if arr.ndim != 1:
                    raise LeverError(f"{kind} row at layer {int(layer)} must be 1-D [hidden], got {arr.shape}")
                if not np.isfinite(arr).all():
                    raise LeverError(f"{kind} row at layer {int(layer)} contains NaN/inf")
                rows[int(layer)] = arr
            object.__setattr__(self, kind, rows)

    def rows(self, kind: str) -> Mapping[int, np.ndarray]:
        return getattr(self, kind)

    @property
    def empty(self) -> bool:
        return not self.attn and not self.mlp

    def digest(self) -> str:
        """A content hash of the lever (span + every row's bits): equal digests write identical rows."""
        h = hashlib.sha256(self.span.encode())
        for kind in KINDS:
            for layer in sorted(self.rows(kind)):
                h.update(f"|{kind}{layer}|".encode())
                h.update(self.rows(kind)[layer].tobytes())
        return h.hexdigest()[:16]


def neuron_row(model: torch.nn.Module, layer: int, neuron: int, dose: float) -> np.ndarray:
    """The ``mlp`` row of neuron ``(layer, neuron)`` at ``dose``: ``dose * W_down[:, neuron]`` in float32 (the
    MLP output a unit activation of ``dose`` on that neuron contributes)."""
    w = model.layers[int(layer)].ffn.down_proj.weight
    if not 0 <= int(neuron) < w.shape[1]:
        raise LeverError(f"neuron {neuron} outside [0, {w.shape[1]})")
    col = w[:, int(neuron)].detach().float().cpu().numpy()
    return (np.float32(dose) * col).astype(np.float32)


class LeverTables:
    """The lever tables of one engine: per site kind ``[n_layers, 1 + max_levers, hidden]`` (engine dtype), slot 0
    the ``-0.0`` identity. ``bind`` / ``release`` run between steps (host side, a device sync each); the decoders'
    per-row slot buffers name which slot each row reads."""

    def __init__(self, n_layers: int, hidden: int, max_levers: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        if int(max_levers) < 1:
            raise LeverError(f"max_levers must be >= 1, got {max_levers}")
        if not dtype.is_floating_point:
            raise LeverError(f"lever tables need a floating dtype, got {dtype}")
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.max_levers = int(max_levers)
        self.dtype = dtype
        self.device = torch.device(device)
        self.tables: dict[str, torch.Tensor] = {
            kind: torch.full((self.n_layers, 1 + self.max_levers, self.hidden), -0.0, dtype=dtype,
                             device=self.device)
            for kind in KINDS}
        self._plans: dict[int, LeverPlan] = {}
        # bound slots per digest: binding an identical lever twice is refused, so a digest names one slot
        self._by_digest: dict[str, int] = {}
        self._free: list[int] = list(range(1, 1 + self.max_levers))
        self._lock = threading.Lock()

    def bind(self, plan: LeverPlan) -> int:
        """Write ``plan``'s rows into a free slot and return it (>= 1). An empty plan is refused: an unsteered row
        reads slot 0."""
        if not isinstance(plan, LeverPlan):
            raise TypeError(f"bind takes a LeverPlan, got {type(plan).__name__}")
        if plan.empty:
            raise LeverError("an empty plan writes nothing; unsteered rows read slot 0")
        staged: dict[str, list[tuple[int, torch.Tensor]]] = {}
        for kind in KINDS:
            for layer, row in plan.rows(kind).items():
                if not 0 <= layer < self.n_layers:
                    raise LeverError(f"{kind} row at layer {layer} outside [0, {self.n_layers})")
                if row.shape[0] != self.hidden:
                    raise LeverError(f"{kind} row at layer {layer} has {row.shape[0]} elements, "
                                     f"hidden is {self.hidden}")
                staged.setdefault(kind, []).append((layer, torch.from_numpy(row)))
        with self._lock:
            if not self._free:
                raise LeverError(f"all {self.max_levers} lever slots are bound; release one first")
            slot = self._free.pop(0)
            try:
                for kind, items in staged.items():
                    for layer, row in items:
                        # the row cast to the engine dtype, as an eager hook casts it onto a sublayer output
                        self.tables[kind][layer, slot].copy_(row.to(device=self.device, dtype=self.dtype))
            except BaseException:
                for t in self.tables.values():
                    t[:, slot].fill_(-0.0)
                self._free.insert(0, slot)
                raise
            self._plans[slot] = plan
        return slot

    def release(self, slot: int) -> None:
        """Refill ``slot`` with ``-0.0`` at every layer and free it. Releasing an unbound slot is refused."""
        slot = int(slot)
        with self._lock:
            if slot not in self._plans:
                raise LeverError(f"slot {slot} is not bound")
            for t in self.tables.values():
                t[:, slot].fill_(-0.0)
            del self._plans[slot]
            self._free.append(slot)
            self._free.sort()

    def plan(self, slot: int) -> Optional[LeverPlan]:
        """The plan bound at ``slot`` (None for the identity slot)."""
        if int(slot) == IDENTITY_SLOT:
            return None
        try:
            return self._plans[int(slot)]
        except KeyError:
            raise LeverError(f"slot {slot} is not bound") from None

    def span(self, slot: int) -> Optional[Span]:
        p = self.plan(slot)
        return None if p is None else p.span

    def check_slots(self, slots: Iterable[int]) -> list[int]:
        """``slots`` as ints, each the identity or a bound slot."""
        out = [int(s) for s in slots]
        bad = [s for s in out if s != IDENTITY_SLOT and s not in self._plans]
        if bad:
            raise LeverError(f"slots {bad} are not bound (bound: {sorted(self._plans)})")
        return out

    @property
    def bound(self) -> tuple[int, ...]:
        return tuple(sorted(self._plans))

    @property
    def free(self) -> int:
        return len(self._free)

    def is_identity(self) -> bool:
        """True when slot 0 and every free slot is bitwise ``-0.0`` at every layer (one device sync)."""
        cols = [IDENTITY_SLOT, *self._free]
        return all(bool(torch.all(torch.signbit(t[:, cols]) & (t[:, cols] == 0)).item())
                   for t in self.tables.values())

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tables.values())


def rule_n_norms(norms: Sequence[float] | Mapping[int, float], n_layers: int) -> dict[int, float]:
    """Per-layer partial norms as the ``{layer: norm}`` mapping ``write_plan`` reads, checked finite and positive."""
    items = norms.items() if isinstance(norms, Mapping) else enumerate(norms)
    out = {int(k): float(v) for k, v in items}
    bad = [p for p in range(n_layers) if not (p in out and math.isfinite(out[p]) and out[p] > 0)]
    if bad:
        raise LeverError(f"partial norms missing, non-finite or <= 0 at layers {bad[:5]}")
    return out


CAPTURE_SITES: tuple[str, ...] = ("partial_post_attn", "mlp_in")


class _Taps:
    """What one module's eager forwards recorded at the tapped layers, in call order."""

    def __init__(self, layers: Sequence[int], what: Sequence[str]) -> None:
        self.layers = frozenset(int(x) for x in layers)
        self.what = tuple(what)
        self.records: dict[str, dict[int, list[torch.Tensor]]] = {w: {L: [] for L in self.layers} for w in self.what}

    def record(self, layer: int, partial: torch.Tensor, x_mlp: torch.Tensor) -> None:
        if layer not in self.layers:
            return
        for w, t in (("partial_post_attn", partial), ("mlp_in", x_mlp)):
            if w in self.records:
                self.records[w][layer].append(t.detach().float().clone())

    def stacked(self) -> dict[str, dict[int, torch.Tensor]]:
        """Per site and layer, every recorded call concatenated along the position axis: (B, positions, hidden)."""
        return {w: {L: torch.cat(ts, dim=1) if ts else torch.empty(0) for L, ts in per.items()}
                for w, per in self.records.items()}


class Capture:
    """The result of ``capture``: ``engine[what][layer]`` is (1, positions, hidden) over the engine's prefill and
    steps, ``rows(dec)[what][layer]`` is (B, steps, hidden) over a decoder's steps, all float32, in position order."""

    def __init__(self, engine_taps: _Taps, dec_taps: dict[int, _Taps]) -> None:
        self._engine_taps = engine_taps
        self._dec_taps = dec_taps

    @property
    def engine(self) -> dict[str, dict[int, torch.Tensor]]:
        return self._engine_taps.stacked()

    def rows(self, dec: torch.nn.Module) -> dict[str, dict[int, torch.Tensor]]:
        try:
            return self._dec_taps[id(dec)].stacked()
        except KeyError:
            raise LeverError("that decoder was not captured") from None

    def sequence(self, what: str, layer: int, dec: Optional[torch.nn.Module] = None, row: int = 0) -> torch.Tensor:
        """One row's whole sequence (positions, hidden): the engine's positions (its prefill, when the row was
        admitted from it) followed by the row's decode steps in ``dec``."""
        parts = [self.engine[what][layer][0]] if self._engine_taps.records[what][layer] else []
        if dec is not None:
            parts.append(self.rows(dec)[what][layer][row])
        if not parts:
            raise LeverError(f"nothing recorded for {what} at layer {layer}")
        return torch.cat(parts, dim=0)


@contextmanager
def capture(engine: torch.nn.Module, *decoders: torch.nn.Module, layers: Sequence[int],
            what: Sequence[str] = CAPTURE_SITES):
    """Record the post-attention partial (after any lever or steering write; the partial the MLP routing reads)
    and the MLP input (post-``ffn_norm``) at effective ``layers``, at every position the engine and ``decoders``
    forward inside the block. Inside it every forward runs EAGER (compiled graphs carry no taps): an engine prefill
    is one eager block forward over the prompt, a step the eager step. Capture one engine prefill per block, so
    ``Capture.sequence`` can join it to the rows admitted from it."""
    bad = [w for w in what if w not in CAPTURE_SITES]
    if bad:
        raise LeverError(f"unknown capture sites {bad}; choose from {CAPTURE_SITES}")
    n = engine.n_layers
    if any(not 0 <= int(x) < n for x in layers):
        raise LeverError(f"capture layers {list(layers)} outside [0, {n})")
    mods = [engine, *decoders]
    if any(m._taps is not None for m in mods):
        raise LeverError("a capture is already active on this engine or decoder")
    eng_taps = _Taps(layers, what)
    dec_taps = {id(d): _Taps(layers, what) for d in decoders}
    engine._taps = eng_taps
    for d in decoders:
        d._taps = dec_taps[id(d)]
    try:
        yield Capture(eng_taps, dec_taps)
    finally:
        for m in mods:
            m._taps = None
