"""Signing inside the decode step: the row decoders capture a signature lane's per-position surfaces as they decode,
so a signed generation needs no replay forward.

A row's signature reads its generated span: the positions P..S-2 of a sequence of S tokens whose prompt is P long.
Those are exactly the positions a row decoder feeds after admitting the row (the first generated token is sampled from
the prefill, each step feeds one token and samples the next, and the last sampled token is never fed), so step j of a
row writes slot j of the row's buffers: slot = pos - (the row's position at admission).

Surfaces per position, in the conventions of kotodama.model.capture (the replay lane's capture). At the sampled
effective layers (``SignSpec.layers``):
  * ``hidden``: the layer's post-MLP partial (the in-progress block sum after the layer's MLP, ``hidden_rows``), post
    lever and steering (the pass as it ran);
  * ``q`` / ``k``: the query and key after the projection and QK-norm, before RoPE; ``v``: the value;
  * ``gate``: the MLP's pre-SiLU gate projection.
At every layer, ``route``: the AttnRes routing weights of the layer's attention input and MLP input over their
sources (committed blocks, then the partial), each padded with zeros to ``route_width`` (a layer with no committed
block reads no routing for its attention input: all zeros); ``route_final``: the final routing (before the norm and
LM head). ``embed``: the token's embedding (the first committed block). ``logit_stats`` (fp64): of the step's
next-token distribution (the raw logits, before penalty and temperature) and the token the step chose: entropy,
top-1 probability, top-5 mass, the chosen token's rank (logits strictly above it), its probability, its logprob.
With ``depth``, the post-MLP partial at every effective layer (the depth path; ``hidden`` is then its sampled
columns, and the export rebuilds the committed blocks from it).

With ``attn``, attention statistics of each position's query row (columns 0..pos, the prompt then the row's own
tokens, self included; fp32 probabilities from the row's attention scores), written by every layer's region after its
attention:
  * ``head_entropy`` (every layer): per query head, -sum p ln p;
  * ``mean_entropy`` (every layer): the entropy of the head-mean distribution m = mean over heads of p;
  * ``head_scalars`` (sampled layers): per head, the mass on prompt columns (< the row's position at admission), the
    recency mass (columns >= max(1, floor(0.8 len)), len = pos + 1) and the sink (column 0);
  * ``mean_rows`` (sampled layers): m over absolute columns 0..width-1 (zero past pos).

The decode step writes the routing and a sampled layer's q / k / v / gate inside each layer's compiled region (index
writes into the decoder's buffers, as its KV writes are), the partials and the embedding around the regions, and the
logit statistics in its sampler; every buffer lives at a fixed
address, so a CUDA-graph replay writes them too. Rows past ``tail`` positions overwrite their last slot (the export
says so).
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch

# The surfaces a sampled layer's region writes, in the order its sign tuple carries their buffers.
REGION_SURFACES = ("q", "k", "v", "gate")
# The logit statistics, in their column order.
LOGIT_STATS = ("entropy", "top1_prob", "top5_mass", "chosen_rank", "chosen_prob", "logprob")
# The attention statistics every layer's region writes (head_scalars and mean_rows at sampled layers only), in the
# order its stats tuple carries them.
STATS = ("head_entropy", "mean_entropy", "head_scalars", "mean_rows")


@dataclass(frozen=True)
class SignSpec:
    """What a decoder signs. ``layers``: the sampled effective layers (sorted, distinct); ``tail``: positions kept per
    row; ``depth``: also the post-MLP partial at every effective layer; ``attn``: attention statistics at every layer,
    with head-mean rows over ``width`` absolute columns at the sampled layers; ``weights``: the weights digest the
    lane declares (``weights_digest``), checked when the decoder is built (None: not checked)."""

    layers: tuple[int, ...]
    tail: int
    depth: bool = False
    attn: bool = False
    width: int = 0
    weights: Optional[str] = None

    def __post_init__(self) -> None:
        layers = tuple(int(x) for x in self.layers)
        if not layers or list(layers) != sorted(set(layers)):
            raise ValueError(f"sign layers must be a non-empty sorted list of distinct layers, got {self.layers}")
        if int(self.tail) < 1:
            raise ValueError(f"sign tail must be >= 1, got {self.tail}")
        if self.attn and int(self.width) < 1:
            raise ValueError("attention statistics need width >= 1 (the head-mean rows' columns)")
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "tail", int(self.tail))
        object.__setattr__(self, "width", int(self.width))

    def check(self, n_layers: int) -> None:
        if self.layers[0] < 0 or self.layers[-1] >= n_layers:
            raise ValueError(f"sign layers {self.layers} outside [0, {n_layers})")


def weights_digest(model: torch.nn.Module) -> str:
    """sha256 of a model's weights as loaded: every ``state_dict()`` entry in sorted key order, each as its utf-8
    name, NUL, its dtype, NUL, its shape (comma-joined), NUL, then its raw bytes in the loaded dtype. Tied or aliased
    tensors are hashed under each key. This is anamnesis's lane weights rule (v1), byte for byte, so a signature row
    names the computor it was captured from."""
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        x = t.detach().contiguous().cpu()
        h.update(name.encode("utf-8") + b"\0" + str(x.dtype).encode("utf-8") + b"\0"
                 + ",".join(str(int(d)) for d in x.shape).encode("utf-8") + b"\0")
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes() if x.numel() else b"")
    return h.hexdigest()


def route_width(signatures: Sequence[tuple[int, bool]]) -> int:
    """The widest routing a layer reads: a boundary layer's MLP input routes over its committed blocks, the block it
    commits and its new partial (committed + 2); any other input over committed + 1."""
    return max(c + 2 if boundary else c + 1 for c, boundary in signatures)


def committed_sources(signatures: Sequence[tuple[int, bool]]) -> int:
    """Sources of the final routing: every committed block and the last partial."""
    return sum(boundary for _, boundary in signatures) + 1


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
           "gate": buf(n, intermediate), "route": buf(n_layers, 2 * route), "route_final": buf(1, route),
           "embed": buf(1, hidden),
           "logit_stats": torch.zeros((rows, T, 1, len(LOGIT_STATS)), device=device, dtype=torch.float64)}
    if spec.depth:
        out["depth"] = buf(n_layers, hidden)
    else:
        out["hidden"] = buf(n, hidden)
    if spec.attn:
        def f32(cols: int, width: int) -> torch.Tensor:
            return torch.zeros((rows, T, cols, width), device=device, dtype=torch.float32)

        out.update(head_entropy=f32(n_layers, n_heads), mean_entropy=f32(n_layers, 1),
                   head_scalars=f32(n, 3 * n_heads), mean_rows=f32(n, spec.width))
    return out


def sign_bytes(spec: SignSpec, rows: int, *, hidden: int, n_heads: int, n_kv_heads: int, head_dim: int,
               intermediate: int, n_layers: int, route: int, dtype: torch.dtype) -> int:
    """Bytes ``allocate`` takes, without allocating (for sizing a decoder before building it)."""
    n = len(spec.layers)
    per_pos = n * (n_heads * head_dim + 2 * n_kv_heads * head_dim + intermediate) + n_layers * 2 * route + route
    per_pos += hidden + (n_layers * hidden if spec.depth else n * hidden)
    total = rows * spec.tail * per_pos * torch.empty((), dtype=dtype).element_size()
    total += rows * spec.tail * 8 * len(LOGIT_STATS)
    if spec.attn:
        total += rows * spec.tail * 4 * (n_layers * (n_heads + 1) + n * (3 * n_heads + spec.width))
    return total


def logit_stats(logits: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """The LOGIT_STATS of each row's distribution (B, V) and chosen token (B, 1), fp64 after max-subtraction."""
    x = logits.double()
    x = x - x.max(dim=-1, keepdim=True).values
    e = x.exp()
    z = e.sum(-1, keepdim=True)
    log_z = z.log()
    xc = x.gather(1, chosen)
    entropy = log_z - (x * e).sum(-1, keepdim=True) / z
    # the top five by the logits as given: x and e are monotone in them, so these are e's top five, in the same order
    # (ties hold equal values), and the selection runs on half-width keys
    top = logits.topk(min(5, e.shape[-1]), dim=-1).indices
    top5 = e.gather(1, top).sum(-1, keepdim=True) / z
    rank = (x > xc).sum(-1, keepdim=True).double()
    return torch.cat([entropy, 1.0 / z, top5, rank, xc.exp() / z, xc - log_z], dim=-1)


def stats_write(st: tuple, p: torch.Tensor, sampled: bool) -> None:
    """One layer's attention statistics into the stats buffers, from its probabilities p (B, heads, columns) fp32 in
    absolute column order: st = (head_entropy, mean_entropy, head_scalars, mean_rows buffers, row ids, each row's
    slot, the layer's column, the sampled column (0-d) or None, each row's position at admission, each row's
    position)."""
    head_ent, mean_ent, head_sc, mean_rows, row_ids, slot, layer_col, sampled_col, start, pos = st
    B, H, W = p.shape
    idx = (row_ids, slot, layer_col.expand(B))
    head_ent.index_put_(idx, -torch.special.xlogy(p, p).sum(-1))
    m = p.mean(1)
    mean_ent.index_put_(idx, -torch.special.xlogy(m, m).sum(-1, keepdim=True))
    if not sampled:
        return
    cols = torch.arange(W, device=p.device).view(1, 1, W)
    cut = torch.clamp(((pos + 1) * 0.8).long(), min=1).view(B, 1, 1)
    prompt = (p * (cols < start.view(B, 1, 1))).sum(-1)
    recency = (p * (cols >= cut)).sum(-1)
    sidx = (row_ids, slot, sampled_col.expand(B))
    head_sc.index_put_(sidx, torch.cat([prompt, recency, p[:, :, 0]], dim=-1))
    width = mean_rows.shape[-1]
    row = m[:, :width] if W >= width else torch.cat([m, m.new_zeros((B, width - W))], dim=-1)
    mean_rows.index_put_(sidx, row)


@dataclass
class SignedRow:
    """One row's captured span. ``positions``: span length (steps decoded since admission); ``kept``: positions
    held (``min(positions, tail)``; when they differ the last slot was overwritten and the span is incomplete);
    ``weights``: the decoder's weights digest (``weights_digest``);
    ``start``: the row's position at admission (its prompt length); ``lever_slot``: the row's lever slot (0: none
    bound); ``steered``: broadcast steering was engaged. ``surfaces``: name -> tensor [kept, layers, ...] with
    q (heads, head_dim), k / v (kv heads, head_dim), gate, route (2, route width), hidden, depth (every layer).
    ``route_sources``: per layer, (attention-input sources, MLP-input sources); ``final_sources``: the final
    routing's. Surfaces also hold route [kept, every layer, 2, route width], route_final [kept, route width], embed
    [kept, hidden], logit_stats [kept, 6] (fp64, LOGIT_STATS order) and, with depth, ``committed`` (a list: the
    embedding, then each block's last partial for every block but the last). With attention statistics:
    head_entropy [kept, every layer, heads], mean_entropy [kept, every layer], head_scalars [kept, layers, 3, heads]
    (prompt mass, recency mass, sink), mean_rows [kept, layers, width]."""

    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    layers: tuple[int, ...]
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int = 0
    weights: str = ""
    surfaces: dict[str, object] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.kept == self.positions


_LAYER_INDEX: dict[tuple, torch.Tensor] = {}


def _layer_index(layers: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """``layers`` as a long tensor on ``device``, made once per (layers, device)."""
    key = (layers, str(device))
    if key not in _LAYER_INDEX:
        _LAYER_INDEX[key] = torch.tensor(layers, dtype=torch.long, device=device)
    return _LAYER_INDEX[key]


def export(spec: SignSpec, bufs: dict[str, torch.Tensor], row: int, *, start: int, pos: int, lever_slot: int,
           steered: bool, signatures: Sequence[tuple[int, bool]], n_heads: int, n_kv_heads: int,
           head_dim: int, weights: str = "", device: Optional[torch.device] = None) -> SignedRow:
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
            x = x.view(kept, x.shape[1], 2, -1)
        elif name in ("route_final", "embed", "logit_stats"):
            x = x[:, 0]
        elif name == "mean_entropy":
            x = x[..., 0]
        elif name == "head_scalars":
            x = x.view(kept, n, 3, n_heads)
        surf[name] = x
    if spec.depth:
        # an index already on the device: indexing with a list copies it from pageable host memory, which waits
        # for the work queued on the stream (an export while the next decode step runs would stall on it)
        surf["hidden"] = surf["depth"].index_select(1, _layer_index(spec.layers, surf["depth"].device))
        # a block's output is its last layer's partial; layer li ends a block when li + 1 opens one
        ends = [li - 1 for li, (_, boundary) in enumerate(signatures) if boundary and li > 0]
        surf["committed"] = [surf["embed"]] + [surf["depth"][:, li] for li in ends]
    return SignedRow(positions=positions, kept=kept, start=int(start), lever_slot=int(lever_slot),
                     steered=bool(steered), layers=spec.layers,
                     route_sources=tuple(route_sources(c, b) for c, b in signatures),
                     final_sources=committed_sources(signatures), weights=weights, surfaces=surf)


def resid_pre(row: SignedRow) -> torch.Tensor:
    """Each effective layer's attention input over ``row``'s span, [positions, layers, hidden] fp32, from its
    captured surfaces (needs the depth path): layer l mixes the blocks committed before it and the partial entering
    it (the embedding at layer 0, else layer l-1's output partial) with its captured pre-attention routing weights,
    as the decode layer routes them; a layer with no committed block reads the partial unrouted. The decode step
    mixes with unrounded weights, so this matches its attention input to the precision the weights are captured in
    (the engine's dtype), not bit for bit."""
    surf = row.surfaces
    if "depth" not in surf:
        raise ValueError("resid_pre needs a row captured with the depth path (SignSpec.depth)")
    depth = surf["depth"].float()
    embed = surf["embed"].float()
    committed = [c.float() for c in surf["committed"]]
    route = surf["route"].float()
    out = []
    for li, (n_attn, _) in enumerate(row.route_sources):
        entering = embed if li == 0 else depth[:, li - 1]
        if n_attn == 0:
            out.append(entering)
            continue
        sources = torch.stack(committed[:n_attn - 1] + [entering], dim=0)
        weights = route[:, li, 0, :n_attn].transpose(0, 1).unsqueeze(-1)
        out.append((weights * sources).sum(0))
    return torch.stack(out, dim=1)


# Source files whose code is the engine's arithmetic: a change to any of them is another engine lane.
ENGINE_SOURCES = ("regional.py", "multi_decode.py", "decode_engine.py", "cascade_attn.py", "signing.py", "llama.py",
                  "kernel_pins.py", "attn_stats.py")


def device_arch(device: Any) -> str:
    """The architecture a device computes on, as lane ids name it: ``cuda-sm_<major><minor>`` for a CUDA device (its
    kernels, and so the captured bytes, differ across architectures), ``cpu-<machine>-<ISA>`` otherwise (the vector
    instruction set torch dispatches to, which picks the CPU kernels)."""
    import platform

    dev = torch.device(device)
    if dev.type == "cuda":
        major, minor = torch.cuda.get_device_capability(dev)
        return f"cuda-sm_{major}{minor}"
    return f"{dev.type}-{platform.machine() or 'unknown'}-{torch.backends.cpu.get_cpu_capability()}"


def engine_identity(*, compile_mode: str, width: int, weights: str, pins_digest: Optional[str], device: Any,
                    extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """What a decode-engine signature's arithmetic is made of, for its lane id: the engine sources' digests, torch and
    triton versions, the device architecture (``device_arch``), the regions' compile mode, the pinned kernel configs
    (kotodama.model.kernel_pins digest; None: unpinned, so not restart-reproducible), the canonical decoder width and
    the weights digest."""
    import torch as _torch

    here = Path(__file__).parent
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    ident: dict[str, Any] = {
        "producer": "kotodama.model.signing",
        "source_sha256": {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in ENGINE_SOURCES},
        "torch": _torch.__version__, "triton": triton_version, "device_arch": device_arch(device),
        "compile_mode": compile_mode,
        "kernel_pins": pins_digest, "width": int(width), "weights": weights,
    }
    if extra:
        ident.update(extra)
    return ident


@dataclass
class ReducedRow:
    """One row's signature as a reducer emits it: the feature vector (fp32), the digest of its feature names, the
    lane id the reducer assigned, and anything else the reducer records (``meta``)."""

    features: Any
    names_digest: str
    lane_id: str
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SignatureReducer(Protocol):
    """Turns signed rows into signature vectors. kotodama defines the shape and never imports an implementation;
    one is named by a dotted path (``load_reducer``). ``reduce`` receives finished rows, each row's full token ids
    (prompt and span) and the engine identity they were captured under, and returns one ReducedRow per row."""

    reducer_identity: str

    def reduce(self, rows: Sequence[SignedRow], *, token_ids: Sequence[Sequence[int]],
               engine: Mapping[str, Any]) -> list[ReducedRow]: ...


def load_reducer(spec: str, **kwargs: Any) -> SignatureReducer:
    """A reducer by dotted path, ``package.module:attr``: attr is a class or factory, called with ``kwargs``."""
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"reducer spec must be 'package.module:attr', got {spec!r}")
    obj = getattr(importlib.import_module(module), attr)(**kwargs)
    if not isinstance(obj, SignatureReducer):
        raise TypeError(f"{spec} does not implement SignatureReducer (reduce + reducer_identity)")
    return obj

