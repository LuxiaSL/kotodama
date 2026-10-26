#!/usr/bin/env python3
"""Median norm of the AttnRes running partial after every layer's attention sublayer — the per-position scale the
steering write rules use (N: a later re-add is the site row times m(p)/m(site)).

Measured on generated text the model will be steered on (self-play transcripts), teacher-forced, over every token
after the transcript's header, with the same hook build_egv sizes its vectors with (the engine's steer row).

Writes ``<out>.npz`` with ``partial_norm_l{L}`` for every effective layer L (a sidecar the server loads beside the
steering bank) and ``<out>.json`` (the table + sample counts).

Usage (node, GPU):
  tools/run_py.sh scripts/steer/partial_norms.py --ckpt <step.pt> --config-section 1b-tsumugi --rope-fp32 \\
      --transcripts run1/transcripts.jsonl [run2/transcripts.jsonl ...] --max-docs 200 --out partial_norms.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_egv import _wrap_attn_sublayer, build_config, load_model  # noqa: E402
from kotodama import presets  # noqa: E402
from kotodama.posttrain.transcript import SEP, load_jsonl, render  # noqa: E402

logger = logging.getLogger("partial_norms")


@torch.no_grad()
def partial_norms(model, docs: list[tuple[torch.Tensor, int]]) -> tuple[dict[int, float], int]:
    """Per effective layer, the median L2 norm of the post-attention partial over positions >= each doc's start."""
    n_layers = model.config.num_layers
    acc: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}
    start = {"p": 0}

    def recorder(li: int):
        def post(out: torch.Tensor) -> torch.Tensor:
            acc[li].append(out[0, start["p"]:].float().norm(dim=-1).cpu().numpy())
            return out
        return post

    n_tok = 0
    with ExitStack() as stack:
        for li in range(n_layers):
            stack.enter_context(_wrap_attn_sublayer(model, li, recorder(li)))
        for ids, p in docs:
            start["p"] = p
            model(ids[None].to(next(model.parameters()).device))
            n_tok += len(ids) - p
    return {li: float(np.median(np.concatenate(v))) for li, v in acc.items()}, n_tok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-section", default="1b-tsumugi")
    ap.add_argument("--boundaries", default=None)
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--transcripts", nargs="+", required=True)
    ap.add_argument("--max-docs", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    docs: list[tuple[torch.Tensor, int]] = []
    for path in a.transcripts:
        for tr in load_jsonl(path):
            if len(docs) >= a.max_docs:
                break
            text, _ = render(tr)
            ids = tok.encode(text, add_special_tokens=False)[: a.max_len]
            p = len(tok.encode(tr.header + SEP, add_special_tokens=False)) if tr.header else 0
            if len(ids) - p >= 16:
                docs.append((torch.tensor(ids), p))
    if not docs:
        raise SystemExit("no usable transcripts")
    bnd = [int(x) for x in a.boundaries.split(",")] if a.boundaries else None
    model = load_model(a.ckpt, build_config(a.config_section, bnd, a.rope_fp32), torch.device(a.device))
    norms, n_tok = partial_norms(model, docs)
    out = a.out if a.out.suffix == ".npz" else a.out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"partial_norm_l{li}": np.array(v, dtype=np.float64) for li, v in norms.items()})
    out.with_suffix(".json").write_text(json.dumps({"ckpt": a.ckpt, "transcripts": a.transcripts, "docs": len(docs),
                                                    "tokens": n_tok, "norms": {str(k): round(v, 2) for k, v in norms.items()}},
                                                   indent=1))
    logger.info("%d docs, %d tokens; norms %s", len(docs), n_tok, {k: round(v) for k, v in norms.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
