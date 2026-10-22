"""Induction / prefix-matching score for any koto checkpoint.

Scored per checkpoint, offline or in-loop (``train.py --inloop_evals
induction``). Two channels:

Behavioral (primary): sequences ``[prepend?][A][A]`` with ``A`` = ``prefix_len``
uniform-random token ids (special ids excluded). Per-token CE on the second
copy vs the first measures in-context copying — random tokens carry no
semantics, so the drop is attributable to copying. Position ``j`` of a half
is the token ``A[j]``; ``j = 0`` of the second half is not inducible (its
predecessor is ``A[-1]``, never followed by ``A[0]`` before), so both halves
are aggregated over ``j >= 1``.

  induction_gain            = mean(loss_first) - mean(loss_second)   [nats]
  induction_gain_normalized = induction_gain / mean(loss_first)      (1 = perfect copy)

plus per-position curves and per-sequence gains (for an SE).

Attention (optional, AttnRes checkpoints only): per-head prefix-matching score
read from ``kotodama.model.capture.instrumented_forward`` (eager softmax
weights, parity-tested against ``llama.py``) — the mean attention mass from the
second-copy token ``A[j]`` onto ``A[j+1]`` in the first copy. The capture
walker mirrors the AttnRes forward only; non-AttnRes models skip this channel.

Usage::

  tools/run_py.sh -m kotodama.eval.induction \\
      --checkpoint <path-or-registry-name> --config-section proxy \\
      --attn-res-boundaries ddv1 --tag p2x2-arm1 --out <json>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# model(input_ids) -> {"logits": (B, S, V)} — the KotodamaModel contract.
ModelFn = Callable[[torch.Tensor], dict[str, torch.Tensor]]


# ── Batch construction ────────────────────────────────────────────────────────

def candidate_ids(vocab_lo: int, vocab_hi: int,
                  exclude: Iterable[int] = ()) -> torch.Tensor:
    """Ids in ``[vocab_lo, vocab_hi)`` minus ``exclude`` (sorted, long)."""
    if not 0 <= vocab_lo < vocab_hi:
        raise ValueError(f"bad vocab range [{vocab_lo}, {vocab_hi})")
    ex = set(int(i) for i in exclude)
    ids = [i for i in range(vocab_lo, vocab_hi) if i not in ex]
    if len(ids) < 2:
        raise ValueError("fewer than 2 candidate ids after exclusions")
    return torch.tensor(ids, dtype=torch.long)


def build_repeat_batch(n_seqs: int, prefix_len: int, vocab_lo: int,
                       vocab_hi: int, exclude: Iterable[int] = (),
                       seed: int = 0,
                       prepend_id: int | None = None) -> torch.Tensor:
    """``(n_seqs, off + 2*prefix_len)`` ids: ``[prepend?][A][A]``, seeded.

    ``A`` is drawn uniformly (with replacement) from :func:`candidate_ids`;
    ``off`` is 1 when ``prepend_id`` is given, else 0.
    """
    if n_seqs < 1 or prefix_len < 2:
        raise ValueError("need n_seqs >= 1 and prefix_len >= 2")
    cand = candidate_ids(vocab_lo, vocab_hi, exclude)
    g = torch.Generator().manual_seed(seed)
    a = cand[torch.randint(len(cand), (n_seqs, prefix_len), generator=g)]
    parts = [a, a]
    if prepend_id is not None:
        parts.insert(0, torch.full((n_seqs, 1), int(prepend_id),
                                   dtype=torch.long))
    return torch.cat(parts, dim=1)


# ── Behavioral channel ────────────────────────────────────────────────────────

def token_losses(model: ModelFn, ids: torch.Tensor, batch_size: int = 16,
                 device: str = "cpu") -> torch.Tensor:
    """Per-token CE, ``(N, S)`` fp32 on CPU: entry ``t`` = -log p(ids[t] | ids[:t]).

    Column 0 has no prediction and is NaN. Logits are cast to fp32 before the
    softmax so bf16 models score on the same ruler as fp32 ones.
    """
    n, s = ids.shape
    out = torch.full((n, s), float("nan"))
    for i in range(0, n, batch_size):
        chunk = ids[i:i + batch_size].to(device)
        with torch.inference_mode():
            logits = model(chunk)["logits"].float()
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                 chunk[:, 1:].reshape(-1), reduction="none")
        out[i:i + len(chunk), 1:] = ce.view(len(chunk), s - 1).cpu()
    return out


def split_halves(losses: torch.Tensor, prefix_len: int,
                 offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(first, second)``, each ``(N, prefix_len)``: loss on ``A[j]`` per copy."""
    first = losses[:, offset:offset + prefix_len]
    second = losses[:, offset + prefix_len:offset + 2 * prefix_len]
    if first.shape[1] != prefix_len or second.shape[1] != prefix_len:
        raise ValueError("loss matrix shorter than offset + 2*prefix_len")
    return first, second


def aggregate_induction(losses: torch.Tensor, prefix_len: int, offset: int,
                        n_candidates: int | None = None) -> dict[str, Any]:
    """Headline numbers + per-position curves from a :func:`token_losses` matrix.

    Aggregates over ``j >= 1`` (``j = 0`` of the second copy is not inducible;
    ``j = 0`` of the first is undefined without a prepended token).
    """
    first, second = split_halves(losses, prefix_len, offset)
    f, s = first[:, 1:].double(), second[:, 1:].double()
    if not (torch.isfinite(f).all() and torch.isfinite(s).all()):
        raise ValueError("non-finite losses in the scored region")
    per_seq_gain = (f.mean(1) - s.mean(1)).numpy()
    lf, ls = float(f.mean()), float(s.mean())
    gain = lf - ls
    n = len(per_seq_gain)

    def curve(x: torch.Tensor) -> list[float | None]:
        m = x.double().mean(0)
        return [None if not math.isfinite(v) else float(v) for v in m.tolist()]

    out: dict[str, Any] = {
        "loss_first": lf,
        "loss_second": ls,
        "induction_gain": gain,
        "induction_gain_normalized": gain / lf if lf > 0 else float("nan"),
        "induction_gain_se": (float(np.std(per_seq_gain, ddof=1) / math.sqrt(n))
                              if n > 1 else float("nan")),
        "per_seq_gain": [float(x) for x in per_seq_gain],
        "curve_first": curve(first),
        "curve_second": curve(second),
        "n_seqs": n,
        "prefix_len": prefix_len,
        "offset": offset,
    }
    if n_candidates is not None:
        out["uniform_loss"] = math.log(n_candidates)  # ln|candidates|, ruler
    return out


# ── Attention channel (AttnRes models, via capture) ───────────────────────────

def prefix_matching_scores(model: Any, ids: torch.Tensor, prefix_len: int,
                           offset: int, batch_size: int = 4,
                           device: str = "cpu") -> np.ndarray:
    """``(num_layers, num_heads)`` mean attention from second-copy ``A[j]``
    onto first-copy ``A[j+1]``, over ``j in [0, prefix_len-2]`` and sequences.

    Uses ``instrumented_forward`` (the parity-tested capture walker), which
    covers AttnRes models only; raises ``ValueError`` otherwise.
    """
    if not getattr(getattr(model, "config", None), "attn_res", False):
        raise ValueError("prefix-matching read needs an AttnRes model "
                         "(kotodama.model.capture mirrors the AttnRes forward)")
    from kotodama.model.capture import instrumented_forward

    j = torch.arange(prefix_len - 1)
    q_pos = offset + prefix_len + j            # A[j], second copy
    k_pos = offset + j + 1                     # A[j+1], first copy
    total: np.ndarray | None = None
    n = ids.shape[0]
    for i in range(0, n, batch_size):
        chunk = ids[i:i + batch_size].to(device)
        _, cap = instrumented_forward(model, chunk)
        # cap["attn"]: per layer (B, nh, S, S), query rows x key cols
        per_layer = []
        for w in cap["attn"]:
            sel = w[:, :, q_pos.to(w.device), k_pos.to(w.device)]  # (B, nh, L-1)
            per_layer.append(sel.float().sum(dim=(0, 2)).cpu().numpy())
        del cap
        block = np.stack(per_layer)            # (layers, nh), summed
        total = block if total is None else total + block
    assert total is not None
    return total / (n * (prefix_len - 1))


def summarize_heads(scores: np.ndarray, top_k: int = 8) -> dict[str, Any]:
    """Top heads + the max; chance for a uniform head at query q is ~1/q."""
    flat = np.argsort(scores, axis=None)[::-1][:top_k]
    top = [{"layer": int(li), "head": int(hi), "score": float(scores[li, hi])}
           for li, hi in (np.unravel_index(f, scores.shape) for f in flat)]
    return {"max": float(scores.max()), "top": top,
            "per_layer_max": [float(x) for x in scores.max(axis=1)],
            "scores": scores.tolist()}


# ── CLI ───────────────────────────────────────────────────────────────────────

def resolve_special_ids() -> tuple[list[int], int | None]:
    """(special ids, eos id) from the presets tokenizer; ``([0], 0)`` fallback."""
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        return sorted(set(tok.all_special_ids)), tok.eos_token_id
    except Exception as e:  # offline / missing cache: stay runnable, say so
        logger.warning("tokenizer unavailable (%s); excluding only id 0 and "
                       "treating it as EOS — pass --exclude-ids to be exact", e)
        return [0], 0


def _force_sdpa(model: torch.nn.Module) -> int:
    """Point every attention module at SDPA (no params; math-equivalent).

    ``load_model`` builds with ``attn_impl="auto"`` (flash-attn when
    installed, which has no CPU kernel); passing ``attn_impl`` through its
    ``attn_res_config`` would switch off AttnRes auto-detection, so the
    backend is flipped after load instead.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "_attn_impl"):
            m._attn_impl = "sdpa"
            n += 1
    model.config.attn_impl = "sdpa"
    return n


def _parse_boundaries(arg: str) -> list[int]:
    from kotodama.presets import BOUNDARIES
    if arg in BOUNDARIES:
        return list(BOUNDARIES[arg])
    return [int(x) for x in arg.split(",")]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Induction / prefix-matching score")
    ap.add_argument("--checkpoint", required=True,
                    help="path or kotodama.ckpt registry name")
    ap.add_argument("--config-section", default="proxy")
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--attn-res", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--attn-res-boundaries", default=None,
                    help="comma-separated layer indices or a presets name "
                         "(ddv1, dd3b)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=256,
                    help="length of the random prefix A (sequence = 2x + prepend)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-lo", type=int, default=0)
    ap.add_argument("--vocab-hi", type=int, default=None,
                    help="exclusive; default = model vocab_size")
    ap.add_argument("--exclude-ids", default=None,
                    help="comma-separated ids to exclude; default = tokenizer "
                         "special ids")
    ap.add_argument("--prepend", choices=["eos", "none"], default="eos",
                    help="token before [A][A] (koto docs follow an EOS)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--attn", choices=["auto", "on", "off"], default="auto",
                    help="prefix-matching head read (AttnRes models only)")
    ap.add_argument("--attn-n-seqs", type=int, default=16)
    ap.add_argument("--attn-batch-size", type=int, default=4)
    ap.add_argument("--rope-fp32", action="store_true",
                    help="model config rope_fp32=True (the 1B ladder arms)")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else cpu")
    ap.add_argument("--out", default=None,
                    help="default: ./induction_<tag>.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    attn_res_config = None
    if args.attn_res == "off":
        attn_res_config = {"attn_res": False}
    elif args.attn_res == "on" or args.attn_res_boundaries:
        attn_res_config = {"attn_res": True}
        if args.attn_res_boundaries:
            attn_res_config["attn_res_boundaries"] = _parse_boundaries(
                args.attn_res_boundaries)

    from kotodama.eval.model_loader import load_model
    model = load_model(args.checkpoint, config_path=args.config_path,
                       config_section=args.config_section,
                       attn_res_config=attn_res_config, device=device,
                       overrides={"rope_fp32": True} if args.rope_fp32 else None)
    if "cuda" not in str(device):
        logger.info("CPU run: attention -> sdpa on %d modules",
                    _force_sdpa(model))

    special, eos = resolve_special_ids()
    exclude = ([int(x) for x in args.exclude_ids.split(",")]
               if args.exclude_ids else special)
    vocab_hi = args.vocab_hi or model.config.vocab_size
    prepend = None
    if args.prepend == "eos":
        if eos is None:
            logger.warning("no EOS id resolved; running without a prepend")
        prepend = eos
    offset = 0 if prepend is None else 1
    total_len = offset + 2 * args.seq_len
    if total_len > model.config.max_position_embeddings:
        raise ValueError(f"sequence length {total_len} exceeds "
                         f"max_position_embeddings "
                         f"{model.config.max_position_embeddings}")

    n_cand = len(candidate_ids(args.vocab_lo, vocab_hi, exclude))
    ids = build_repeat_batch(args.n_seqs, args.seq_len, args.vocab_lo,
                             vocab_hi, exclude, seed=args.seed,
                             prepend_id=prepend)
    losses = token_losses(model, ids, batch_size=args.batch_size,
                          device=device)
    agg = aggregate_induction(losses, args.seq_len, offset, n_candidates=n_cand)
    print(f"[{args.tag}] INDUCTION loss first {agg['loss_first']:.3f} -> "
          f"second {agg['loss_second']:.3f}  gain {agg['induction_gain']:.3f} "
          f"± {agg['induction_gain_se']:.3f} nats  normalized "
          f"{agg['induction_gain_normalized']:.3f}  (uniform ln|V| = "
          f"{agg['uniform_loss']:.2f}, n={agg['n_seqs']})")

    result: dict[str, Any] = {
        "tag": args.tag, "checkpoint": str(args.checkpoint),
        "config": {"n_seqs": args.n_seqs, "seq_len": args.seq_len,
                   "seed": args.seed, "vocab_lo": args.vocab_lo,
                   "vocab_hi": vocab_hi, "n_excluded": len(exclude),
                   "prepend_id": prepend,
                   "attn_res": bool(model.config.attn_res)},
        "behavioral": agg, "prefix_matching": None,
    }

    want_attn = args.attn == "on" or (args.attn == "auto"
                                      and model.config.attn_res)
    if want_attn:
        if not model.config.attn_res:
            print(f"[{args.tag}] PREFIX-MATCHING skipped: not an AttnRes model "
                  f"(capture walker covers AttnRes only)")
            result["prefix_matching"] = {"skipped": "non-AttnRes model"}
        else:
            n_attn = min(args.attn_n_seqs, args.n_seqs)
            scores = prefix_matching_scores(
                model, ids[:n_attn], args.seq_len, offset,
                batch_size=args.attn_batch_size, device=device)
            summ = summarize_heads(scores)
            summ["n_seqs"] = n_attn
            result["prefix_matching"] = summ
            top = ", ".join(f"L{h['layer']}H{h['head']}={h['score']:.2f}"
                            for h in summ["top"][:4])
            print(f"[{args.tag}] PREFIX-MATCHING max {summ['max']:.3f}  "
                  f"top: {top}")

    out = Path(args.out) if args.out else Path(f"induction_{args.tag}.json")
    out.write_text(json.dumps(result))
    print(f"[{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
