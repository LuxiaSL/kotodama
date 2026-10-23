"""Uncheatable Eval BPB for a public Hugging Face model, scored with the same
math as ``kotodama.eval.uncheatable`` scores KotodamaModel checkpoints.

Data loading, chunking, bf16 loss summation, the per-domain reduction and the
static-shape padding are the module's own (``load_domains`` / ``evaluate``);
only the network, its tokenizer and two per-model choices differ:

- the document-start token: the tokenizer's BOS when it defines one (Llama 3:
  ``<|begin_of_text|>``), else its EOS, the separator such models saw between
  documents in pretraining (Qwen, Pythia, OLMo: ``<|endoftext|>``). A chunk is
  never scored without one. Recorded as ``bos_mode`` / ``bos_id``.
- the chunk size: the module's 4000 when the model's context holds
  ``chunk + BOS``, else ``context - 1`` (Pythia: 2047). Recorded as
  ``chunk_size``.

Writes the module's result JSON (schema ``kotodama.uncheatable/v1``) with the
model's provenance and weight fingerprint under ``settings.extra``.

Usage:
    tools/run_py.sh -m scripts.eval.run_uncheatable_hf --model EleutherAI/pythia-1.4b \\
        --revision step14000 --out out/pythia-1.4b-step14000/uncheatable.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from kotodama.eval.uncheatable import (
    CHUNK_SIZE,
    DATA_ENV,
    DEFAULT_DATASET,
    LOSS_PRECISIONS,
    Settings,
    evaluate,
    load_domains,
)
from scripts.eval.run_lm_eval_hf import load_hf_model

logger = logging.getLogger(__name__)


def document_start_token(tokenizer: Any) -> tuple[str, int]:
    """(bos_mode, token id) prepended to every chunk; see the module doc."""
    if tokenizer.bos_token_id is not None:
        return "tokenizer_bos", int(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        return "eos_as_bos", int(tokenizer.eos_token_id)
    raise ValueError("tokenizer defines neither a BOS nor an EOS token")


def chunk_size_for(context: int, requested: int = CHUNK_SIZE) -> int:
    """``requested`` when ``requested + 1`` positions fit, else ``context - 1``."""
    if context < 3:
        raise ValueError(f"context {context} too short to score a chunk")
    return requested if requested + 1 <= context else context - 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="HF repo id, or a local model directory")
    p.add_argument("--revision", default="main")
    p.add_argument("--out", required=True, help="result JSON path")
    p.add_argument("--data-dir", default=os.environ.get(DATA_ENV, DEFAULT_DATASET),
                   help=f"domain files / parquet / HF dataset id (default ${DATA_ENV}, "
                        f"else {DEFAULT_DATASET})")
    p.add_argument("--domains", default=None, help="comma-separated subset (default: all)")
    p.add_argument("--max-docs-per-domain", type=int, default=None)
    p.add_argument("--max-tokens-per-domain", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help="upper bound; capped at the model's context - 1")
    p.add_argument("--loss-precision", choices=LOSS_PRECISIONS, default="bf16")
    p.add_argument("--no-static-shapes", dest="static_shapes", action="store_false")
    p.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p.add_argument("--cache-dir", default=None, help="HF hub cache for the model")
    p.add_argument("--weights", choices=("auto", "safetensors", "bin"), default="auto")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    domains = load_domains(args.data_dir)
    if args.domains:
        want = [d.strip() for d in args.domains.split(",") if d.strip()]
        missing = [d for d in want if d not in domains]
        if missing:
            raise SystemExit(f"unknown domains {missing}; have {sorted(domains)}")
        domains = {d: domains[d] for d in want}
    logger.info("data: %d domains, %d docs from %s", len(domains),
                sum(len(v) for v in domains.values()), args.data_dir)

    model, tok, info = load_hf_model(args.model, args.revision, args.cache_dir,
                                     args.weights, device)
    bos_mode, bos_id = document_start_token(tok)
    chunk = chunk_size_for(info["context"], args.chunk_size)
    logger.info("document start %s = %d (%r); chunk %d", bos_mode, bos_id,
                tok.convert_ids_to_tokens(bos_id), chunk)

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids).logits[0]

    settings = Settings(
        checkpoint=args.model, config_section="hf", attn_res=False,
        attn_res_boundaries=None, rope_fp32=False, tokenizer=args.model,
        bos_mode=bos_mode, bos_id=bos_id, chunk_size=chunk,
        loss_precision=args.loss_precision,
        max_docs_per_domain=args.max_docs_per_domain,
        max_tokens_per_domain=args.max_tokens_per_domain,
        data_source=str(args.data_dir), device=device,
        static_shapes=args.static_shapes,
        extra={
            "revision": None if info["local"] else args.revision,
            "bos_token": tok.convert_ids_to_tokens(bos_id),
            "dtype": "bfloat16",
            **info,
        },
    )
    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, leave=False)  # noqa: E731
    except ImportError:
        progress = None

    result = evaluate(logits_fn, encode, domains, settings, progress=progress)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    width = max(len(d) for d in result["domains"])
    for name, d in result["domains"].items():
        print(f"{name:<{width}}  {d['bpb']:.6f}  ({d['n_docs']}/{d['n_docs_total']} docs)")
    print(f"{'MEAN':<{width}}  {result['mean_bpb']:.6f}  over {result['n_domains']} domains -> {out}")


if __name__ == "__main__":
    main()
