"""Signing inside the decode step: the row decoders capture a signature lane's per-position surfaces as they decode,
so a signed generation needs no replay forward.

A row's signature reads its generated span: the positions P..S-2 of a sequence of S tokens whose prompt is P long.
Those are exactly the positions a row decoder feeds after admitting the row (the first generated token is sampled from
the prefill, each step feeds one token and samples the next, and the last sampled token is never fed), so step j of a
row writes slot j of the row's buffers: slot = pos - (the row's position at admission).

Surfaces, at the sampled effective layers (``SignSpec.layers``), per position, in the conventions of
kotodama.model.capture (the replay lane's capture):
  * ``hidden``: the layer's post-MLP partial (the in-progress block sum after the layer's MLP, ``hidden_rows``), post
    lever and steering (the pass as it ran);
  * ``q`` / ``k``: the query and key after the projection and QK-norm, before RoPE; ``v``: the value;
  * ``gate``: the MLP's pre-SiLU gate projection;
  * ``route``: the AttnRes routing weights of the layer's attention input and MLP input over their sources (committed
    blocks, then the partial), each padded with zeros to ``route_width`` (a layer with no committed block reads no
    routing for its attention input: all zeros).
With ``depth``, the post-MLP partial at every effective layer (the depth path; ``hidden`` is then its sampled
columns).

The decode step writes q / k / v / gate / route inside each sampled layer's compiled region (index writes into the
decoder's buffers, as its KV writes are) and the partials after each region returns; every buffer lives at a fixed
address, so a CUDA-graph replay writes them too. Rows past ``tail`` positions overwrite their last slot (the export
says so).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

# The surfaces a sampled layer's region writes, in the order its sign tuple carries their buffers.
REGION_SURFACES = ("q", "k", "v", "gate", "route")


@dataclass(frozen=True)
class SignSpec:
    """What a decoder signs. ``layers``: the sampled effective layers (sorted, distinct); ``tail``: positions kept per
    row; ``depth``: also the post-MLP partial at every effective layer."""

    layers: tuple[int, ...]
    tail: int
    depth: bool = False

    def __post_init__(self) -> None:
        layers = tuple(int(x) for x in self.layers)
        if not layers or list(layers) != sorted(set(layers)):
            raise ValueError(f"sign layers must be a non-empty sorted list of distinct layers, got {self.layers}")
        if int(self.tail) < 1:
            raise ValueError(f"sign tail must be >= 1, got {self.tail}")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "tail", int(self.tail))

    def check(self, n_layers: int) -> None:
        if self.layers[0] < 0 or self.layers[-1] >= n_layers:
            raise ValueError(f"sign layers {self.layers} outside [0, {n_layers})")


def route_width(signatures: Sequence[tuple[int, bool]]) -> int:
    """The widest routing a layer reads: a boundary layer's MLP input routes over its committed blocks, the block it
    commits and its new partial (committed + 2); any other input over committed + 1."""
    return max(c + 2 if boundary else c + 1 for c, boundary in signatures)


def route_sources(committed: int, boundary: bool) -> tuple[int, int]:
    """(sources of the attention input's routing, sources of the MLP input's routing) of a layer; 0 = no routing."""
    return (committed + 1 if committed else 0), (committed + 2 if boundary else committed + 1)


def allocate(spec: SignSpec, rows: int, *, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int,
             intermediate: int, n_layers: int, route: int, device: torch.device,
             dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The decoder's sign buffers by name (each [rows, tail, columns, width])."""
    n, T = len(spec.layers), spec.tail

    def buf(cols: int, width: int) -> torch.Tensor:
        return torch.zeros((rows, T, cols, width), device=device, dtype=dtype)

    out = {"q": buf(n, n_heads * head_dim), "k": buf(n, n_kv_heads * head_dim), "v": buf(n, n_kv_heads * head_dim),
           "gate": buf(n, intermediate), "route": buf(n, 2 * route)}
    if spec.depth:
        out["depth"] = buf(n_layers, hidden)
    else:
        out["hidden"] = buf(n, hidden)
    return out


def sign_bytes(spec: SignSpec, rows: int, *, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int,
               intermediate: int, n_layers: int, route: int, dtype: torch.dtype) -> int:
    """Bytes ``allocate`` takes, without allocating (for sizing a decoder before building it)."""
    n = len(spec.layers)
    per_pos = n * (n_heads * head_dim + 2 * n_kv_heads * head_dim + intermediate + 2 * route)
    per_pos += n_layers * hidden if spec.depth else n * hidden
    return rows * spec.tail * per_pos * torch.empty((), dtype=dtype).element_size()


@dataclass
class SignedRow:
    """One row's captured span. ``positions``: span length (steps decoded since admission); ``kept``: positions
    held (``min(positions, tail)``; when they differ the last slot was overwritten and the span is incomplete);
    ``start``: the row's position at admission (its prompt length); ``lever_slot``: the row's lever slot (0: none
    bound); ``steered``: broadcast steering was engaged. ``surfaces``: name -> tensor [kept, layers, ...] with
    q (heads, head_dim), k / v (kv heads, head_dim), gate, route (2, route width), hidden, depth (every layer).
    ``route_sources``: per sampled layer, (attention-input sources, MLP-input sources)."""

    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    layers: tuple[int, ...]
    route_sources: tuple[tuple[int, int], ...]
    surfaces: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.kept == self.positions


def export(spec: SignSpec, bufs: dict[str, torch.Tensor], row: int, *, start: int, pos: int, lever_slot: int,
           steered: bool, signatures: Sequence[tuple[int, bool]], n_heads: int, n_kv_heads: int,
           head_dim: int, device: Optional[torch.device] = None) -> SignedRow:
    """Row ``row``'s span from the buffers (copies, on ``device``: default the CPU)."""
    positions = max(0, int(pos) - int(start))
    kept = min(positions, spec.tail)
    dev = torch.device("cpu") if device is None else device
    n = len(spec.layers)
    surf = {}
    for name, b in bufs.items():
        x = b[row, :kept].to(dev, copy=True)
        if name == "q":
            x = x.view(kept, n, n_heads, head_dim)
        elif name in ("k", "v"):
            x = x.view(kept, n, n_kv_heads, head_dim)
        elif name == "route":
            x = x.view(kept, n, 2, -1)
        surf[name] = x
    if spec.depth:
        surf["hidden"] = surf["depth"][:, list(spec.layers)]
    return SignedRow(positions=positions, kept=kept, start=int(start), lever_slot=int(lever_slot),
                     steered=bool(steered), layers=spec.layers,
                     route_sources=tuple(route_sources(*signatures[li]) for li in spec.layers), surfaces=surf)
