"""Who reads what: AttnRes routing weights at one site, decomposed by source.

At every routing site (``pre_attn`` / ``pre_mlp`` of layer ``i``, and ``final``)
the model softmaxes over its sources in one fixed order:

    index 0            the token embedding (committed at the first boundary)
    index j (1..k)     the finished output of block j-1 (committed at block j's start)
    index -1 (last)    the partial: the current block's in-progress sum — EXCEPT at a
                       block-start ``pre_attn``, which routes before the commit, so its
                       partial is the just-finished previous block.

This module turns a weight vector in that order into (a) absolute labels per source
and (b) a relative, depth-free split that reads the same at every layer:

    emb    the embedding
    prev   the most recently finished block
    older  every finished block between the embedding and ``prev``
    self   the site's own in-progress block (absent at a block-start pre_attn)

plus ``lookback``: the α-weighted number of layers back each source ends
(self = 0, the block that finished at layer i-1 = 1, the embedding = i+1).
Everything here is linear in the weights, so the split of a token-mean equals
the token-mean of the split. One implementation for telemetry
(``llama._compute_attn_res_diagnostics``) and the offline read
(``scripts/eval/routing_matrix.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

REL_SOURCES = ("emb", "prev", "older", "self")


@dataclass(frozen=True)
class Site:
    """One routing site's source layout."""

    layer: int                   # -1 for the final aggregation
    sublayer: str                # "pre_attn" | "pre_mlp" | "final"
    at: int                      # depth the site reads at (the final aggregation: num_layers)
    labels: tuple[str, ...]      # absolute label per source index
    end_layer: tuple[int, ...]   # last layer each source contains (emb = -1)
    rel: tuple[str, ...]         # relative label per source index (REL_SOURCES)


def block_label(start: int, end: int) -> str:
    """Absolute name of a finished block by the layers it spans (end exclusive)."""
    return f"L{start}" if end - start == 1 else f"L{start}-{end - 1}"


def site_layout(block_ranges: list[tuple[int, int]], layer: int, sublayer: str,
                n_src: int) -> Site:
    """Source layout of one site, from the model's block ranges and the weight count.

    Raises ValueError when ``n_src`` does not match what the block structure implies.
    """
    num_layers = block_ranges[-1][1]
    if sublayer == "final":
        k, at, block_start_attn = len(block_ranges) - 1, num_layers, False
    else:
        k = next(b for b, (s, e) in enumerate(block_ranges) if s <= layer < e)
        at, block_start_attn = layer, sublayer == "pre_attn" and layer == block_ranges[k][0]
    # k = the block this site sits in. Committed before it: emb + finished blocks 0..k-1
    # (k+1 sources) — except a block-start pre_attn, which routes BEFORE the commit:
    # emb + blocks 0..k-2 committed, and block k-1 (just finished) as its partial.
    n_committed = k if block_start_attn else k + 1
    if n_src != n_committed + 1:
        raise ValueError(f"site layer={layer}/{sublayer}: {n_src} sources, block structure "
                         f"implies {n_committed + 1}")
    labels, ends = ["emb"], [-1]
    for j in range(1, n_src - 1 if not block_start_attn else n_src):
        s, e = block_ranges[j - 1]
        labels.append(block_label(s, e)); ends.append(e - 1)
    if not block_start_attn:
        s, e = block_ranges[k]
        labels.append(f"self:{block_label(s, e)}"); ends.append(at)
    # relative labels
    last_finished = n_src - 1 if block_start_attn else n_src - 2
    rel = []
    for j in range(n_src):
        if j == 0:
            rel.append("emb")
        elif j == n_src - 1 and not block_start_attn:
            rel.append("self")
        elif j == last_finished:
            rel.append("prev")
        else:
            rel.append("older")
    return Site(layer, sublayer, at, tuple(labels), tuple(ends), tuple(rel))


def relative_split(weights: torch.Tensor, site: Site) -> dict[str, float]:
    """Sum the weights (n_src,) — or token-mean them first if more dims — per relative source.
    Sources the site does not have are absent from the result (not zero)."""
    w = weights.float()
    if w.dim() > 1:
        w = w.reshape(w.shape[0], -1).mean(1)
    out: dict[str, float] = {}
    for j, r in enumerate(site.rel):
        out[r] = out.get(r, 0.0) + float(w[j])
    return out


def lookback(weights: torch.Tensor, site: Site) -> float:
    """α-weighted layers back each source ends, in layers (self = 0)."""
    w = weights.float()
    if w.dim() > 1:
        w = w.reshape(w.shape[0], -1).mean(1)
    back = torch.tensor([site.at - e for e in site.end_layer], dtype=torch.float32)
    return float((w.cpu() * back).sum())


def token_entropy(weights: torch.Tensor) -> float:
    """Mean over tokens of the per-token routing entropy, normalized by log(n_src).
    ``weights``: (n_src, ...) per-token softmax weights. 0 = every token one-hot,
    1 = every token uniform — unlike the entropy of the token-MEAN weights, which
    cannot tell "each token picks one source" from "each token mixes evenly"."""
    n = weights.shape[0]
    if n < 2:
        return 1.0
    w = weights.float().reshape(n, -1)
    h = -(w * (w + 1e-10).log()).sum(0)
    return float(h.mean() / torch.log(torch.tensor(float(n))))
