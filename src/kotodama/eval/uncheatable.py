"""Uncheatable Eval: bits-per-byte on fresh web text, per domain.

A port of the Uncheatable Eval runner (Jellyfish042/uncheatable_eval,
``Evaluator.eval_hf_model``) that produced the shipped 3B's "Uncheatable BPB
0.852 mean over 15 domains" on the 2026-04 release. The math is kept verbatim
so numbers stay comparable with those logs:

- each document is tokenized with the presets tokenizer, no special tokens;
- the token list is cut into chunks of ``chunk_size`` (4000) tokens and every
  chunk is scored as ``[BOS] + chunk`` (``bos_mode=add_default_bos``; the
  SmolLM2 BOS is ``<|endoftext|>``), so each document token is predicted
  exactly once and the context restarts at each chunk;
- per chunk, ``F.cross_entropy(logits[:-1].to(bfloat16), ids[1:])`` and the
  per-token losses are summed in bf16 (``loss_precision="bf16"``, the
  original's rounding, kept for parity — ``"fp32"`` is the exact variant);
- per domain, ``bpb = (mean_doc_nll / mean_doc_utf8_bytes) / ln 2``, i.e.
  total nats over total UTF-8 bytes of the documents, in bits;
- the headline is the unweighted mean of the per-domain BPBs.

Trimmed mode (``--max-docs-per-domain`` / ``--max-tokens-per-domain``) keeps
whole documents in data order — the first N, and/or documents until the token
budget is reached (the document that crosses it is included) — so bytes and
scored tokens always match exactly and a sweep over checkpoints sees the same
text at every step.

Data (``--data-dir``, default ``$KOTODAMA_UNCHEATABLE_DATA``, else the HF
dataset id): a directory of ``<domain>.jsonl`` (``{"content": ...}`` per line)
or ``<domain>.json`` (list of strings) files; a ``.parquet`` file or a
directory containing them (the HF snapshot: ``category`` + ``content``
columns, grouped in row order); or a Hugging Face dataset id, fetched through
the HF cache.

Usage::

    tools/run_py.sh -m kotodama.eval.uncheatable \\
        --checkpoint kotodama-3b-base-final --out uncheatable.json
    # formation sweep, trimmed, one checkpoint of a 1b arm:
    tools/run_py.sh -m kotodama.eval.uncheatable --checkpoint <run>/step_00004000.pt \\
        --config-section 1b --attn-res-boundaries dd3b --rope-fp32 \\
        --max-docs-per-domain 64 --out bpb_step4000.json

The JSON posts to the scheduler telemetry with ``scripts/eval/hm_backfill.py --uncheatable``
as ``uncheatable/{domain}`` + ``uncheatable/mean``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SCHEMA = "kotodama.uncheatable/v1"
DEFAULT_DATASET = "Jellyfish042/UncheatableEval-2026-04"
DATA_ENV = "KOTODAMA_UNCHEATABLE_DATA"
CHUNK_SIZE = 4000  # the original runner's default; do not change for parity
LOSS_PRECISIONS = ("bf16", "fp32")

LogitsFn = Callable[[torch.Tensor], torch.Tensor]  # (1, T) ids -> (T, V) logits
EncodeFn = Callable[[str], list[int]]


# ── Data ──────────────────────────────────────────────────────────────────────

def _read_json_domain(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line)["content"] for line in f if line.strip()]
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
        raise ValueError(f"{path}: expected a JSON list of strings")
    return data


def _read_parquet(paths: Sequence[Path]) -> dict[str, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - pyarrow ships with datasets
        raise ImportError("reading the Uncheatable parquet needs pyarrow") from e
    out: dict[str, list[str]] = {}
    for p in sorted(paths):
        table = pq.read_table(p, columns=["category", "content"])
        for cat, text in zip(table.column("category").to_pylist(),
                             table.column("content").to_pylist()):
            out.setdefault(str(cat), []).append(text)
    return out


def load_domains(source: str | Path) -> dict[str, list[str]]:
    """Domain name -> documents (data order). See the module doc for layouts."""
    p = Path(source).expanduser()
    if p.is_file():
        if p.suffix == ".parquet":
            return _read_parquet([p])
        if p.suffix in (".json", ".jsonl"):
            return {p.stem: _read_json_domain(p)}
        raise ValueError(f"{p}: unsupported data file (want .parquet/.json/.jsonl)")
    if p.is_dir():
        files = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix in (".json", ".jsonl"))
        if files:
            stems = [f.stem for f in files]
            dup = {s for s in stems if stems.count(s) > 1}
            if dup:
                raise ValueError(f"{p}: both .json and .jsonl for {sorted(dup)}")
            return {f.stem: _read_json_domain(f) for f in files}
        parquets = sorted(p.rglob("*.parquet"))
        if parquets:
            return _read_parquet(parquets)
        raise FileNotFoundError(f"{p}: no <domain>.json[l] files and no .parquet inside")
    if str(source).count("/") == 1 and not str(source).startswith((".", "/", "~")):
        from huggingface_hub import snapshot_download
        logger.info("resolving %s through the Hugging Face cache", source)
        local = snapshot_download(repo_id=str(source), repo_type="dataset")
        return load_domains(local)
    raise FileNotFoundError(f"{source!r}: not a path and not an HF dataset id")


# ── Selection (trimmed mode) ──────────────────────────────────────────────────

def select_docs(token_counts: Sequence[int], max_docs: int | None = None,
                max_tokens: int | None = None) -> int:
    """How many leading documents to keep.

    Whole documents only, in data order: at most ``max_docs``, and stop once
    the kept documents hold ``>= max_tokens`` tokens (the crossing document is
    kept, so the budget is always met when the domain has that many tokens).
    """
    n = len(token_counts)
    if max_docs is not None:
        if max_docs < 1:
            raise ValueError("max_docs must be >= 1")
        n = min(n, max_docs)
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        total = 0
        for i in range(n):
            total += token_counts[i]
            if total >= max_tokens:
                return i + 1
    return n


# ── BPB math (verbatim from Evaluator.eval_hf_model) ──────────────────────────

def chunk_nll(logits: torch.Tensor, ids: torch.Tensor,
              loss_precision: str = "bf16") -> float:
    """Summed next-token NLL (nats) of ``ids[1:]`` given ``logits`` (T, V).

    ``bf16`` is the original ``calculate_log_sum``: logits cast to bf16, the
    per-token losses and their sum stay bf16. ``fp32`` computes in fp32 and
    sums in fp64.
    """
    if loss_precision == "bf16":
        loss = F.cross_entropy(logits[:-1].to(torch.bfloat16), ids[1:],
                               reduction="none")
        return loss.sum().item()
    if loss_precision == "fp32":
        loss = F.cross_entropy(logits[:-1].float(), ids[1:], reduction="none")
        return loss.double().sum().item()
    raise ValueError(f"loss_precision must be one of {LOSS_PRECISIONS}")


def doc_nll(logits_fn: LogitsFn, ids: Sequence[int], bos_id: int,
            chunk_size: int = CHUNK_SIZE, loss_precision: str = "bf16",
            device: str | torch.device = "cpu", pad_to: int | None = None) -> float:
    """NLL of one document: ``[BOS] + chunk`` per ``chunk_size`` tokens.

    ``pad_to`` right-pads every chunk to one fixed length before the forward and
    scores only the real positions. Attention is causal and AttnRes routing mixes
    depth at a fixed position, so padding after a position cannot change its
    logits; a single static shape keeps shape-specialized kernels (the Triton
    AttnRes routing kernels) from recompiling for every document length.
    """
    if len(ids) <= 1:
        # The original asserts this; an empty/1-token doc would skew bytes.
        raise ValueError(f"document too short ({len(ids)} tokens)")
    if chunk_size <= 2:
        raise ValueError("chunk_size must be > 2")
    total = 0.0
    for begin in range(0, len(ids), chunk_size):
        chunk = torch.tensor([bos_id, *ids[begin:begin + chunk_size]],
                             dtype=torch.long, device=device)
        if pad_to is not None and pad_to > chunk.numel():
            padded = torch.cat([chunk, chunk.new_full((pad_to - chunk.numel(),), bos_id)])
            logits = logits_fn(padded.unsqueeze(0))[:chunk.numel()]
        else:
            logits = logits_fn(chunk.unsqueeze(0))
        total += chunk_nll(logits, chunk, loss_precision)
    return total


def bpb_from_sums(nll_per_doc: Sequence[float], bytes_per_doc: Sequence[int]) -> float:
    """The original's reduction: (mean nll / mean bytes) / ln 2."""
    n = len(nll_per_doc)
    mean_nll = sum(nll_per_doc) / n
    mean_bytes = sum(bytes_per_doc) / n
    return (mean_nll / mean_bytes) * (1 / math.log(2))


@dataclass
class DomainResult:
    bpb: float
    bpc: float
    nll_mean: float          # mean summed NLL per document (nats); = the
    n_docs: int              # original log's "neg_log_prob_sum"
    n_docs_total: int
    n_tokens: int
    n_bytes: int
    n_chars: int
    trimmed: bool
    seconds: float = 0.0


def eval_domain(logits_fn: LogitsFn, encode: EncodeFn, texts: Sequence[str],
                bos_id: int, chunk_size: int = CHUNK_SIZE,
                loss_precision: str = "bf16", max_docs: int | None = None,
                max_tokens: int | None = None,
                device: str | torch.device = "cpu",
                progress: Callable[[Iterable[Any]], Iterable[Any]] | None = None,
                pad_to: int | None = None,
                ) -> DomainResult:
    t0 = time.time()
    token_ids = [encode(t) for t in texts]
    n = select_docs([len(x) for x in token_ids], max_docs, max_tokens)
    it: Iterable[int] = range(n)
    if progress is not None:
        it = progress(it)
    nlls: list[float] = []
    for i in it:
        nlls.append(doc_nll(logits_fn, token_ids[i], bos_id, chunk_size,
                            loss_precision, device, pad_to))
    kept = texts[:n]
    n_bytes = [len(t.encode("utf-8")) for t in kept]
    n_chars = [len(t) for t in kept]
    mean_nll = sum(nlls) / n
    return DomainResult(
        bpb=bpb_from_sums(nlls, n_bytes),
        bpc=(mean_nll / (sum(n_chars) / n)) * (1 / math.log(2)),
        nll_mean=mean_nll,
        n_docs=n, n_docs_total=len(texts),
        n_tokens=sum(len(token_ids[i]) for i in range(n)),
        n_bytes=sum(n_bytes), n_chars=sum(n_chars),
        trimmed=n < len(texts),
        seconds=round(time.time() - t0, 2),
    )


@dataclass
class Settings:
    checkpoint: str
    config_section: str
    attn_res: bool
    attn_res_boundaries: list[int] | None
    rope_fp32: bool
    tokenizer: str
    bos_mode: str
    bos_id: int
    chunk_size: int
    loss_precision: str
    max_docs_per_domain: int | None
    max_tokens_per_domain: int | None
    data_source: str
    device: str
    extra: dict[str, Any] = field(default_factory=dict)
    # Pad every chunk to chunk_size + 1 (one static shape; identical NLL, see doc_nll).
    static_shapes: bool = True


def evaluate(logits_fn: LogitsFn, encode: EncodeFn,
             domains: dict[str, list[str]], settings: Settings,
             step: int | None = None, tokens: int | None = None,
             progress: Callable[[Iterable[Any]], Iterable[Any]] | None = None,
             ) -> dict[str, Any]:
    """Score every domain; return the result JSON (schema ``SCHEMA``)."""
    if not domains:
        raise ValueError("no domains to evaluate")
    per: dict[str, dict[str, Any]] = {}
    for name, texts in domains.items():
        r = eval_domain(logits_fn, encode, texts, settings.bos_id,
                        settings.chunk_size, settings.loss_precision,
                        settings.max_docs_per_domain,
                        settings.max_tokens_per_domain,
                        settings.device, progress,
                        settings.chunk_size + 1 if settings.static_shapes else None)
        per[name] = asdict(r)
        logger.info("%-24s bpb %.6f  (%d/%d docs, %d tok, %.1fs)", name, r.bpb,
                    r.n_docs, r.n_docs_total, r.n_tokens, r.seconds)
    bpbs = [d["bpb"] for d in per.values()]
    return {
        "schema": SCHEMA,
        "checkpoint": settings.checkpoint,
        "step": step,
        "tokens": tokens,
        "mean_bpb": sum(bpbs) / len(bpbs),
        "n_domains": len(per),
        "trimmed": any(d["trimmed"] for d in per.values()),
        "domains": per,
        "settings": asdict(settings),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_boundaries(arg: str) -> list[int] | None:
    """A presets name (dd3b, ddv1), a comma list, or ``none`` (no AttnRes)."""
    from kotodama.presets import BOUNDARIES
    if arg.lower() in ("none", "off", "null"):
        return None
    if arg in BOUNDARIES:
        return list(BOUNDARIES[arg])
    try:
        return [int(x) for x in arg.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"--attn-res-boundaries {arg!r}: want one of "
                         f"{sorted(BOUNDARIES)}, 'none', or ints") from e


def resolve_arch(checkpoint: str, config_section: str | None,
                 boundaries: str | None) -> tuple[str, list[int] | None]:
    """Shape + boundaries: explicit args, else the registry entry, else presets."""
    from kotodama import ckpt, presets
    entry = None
    if not Path(checkpoint).exists():
        entry = ckpt.registry().get(checkpoint)
    section = config_section or (entry.size if entry else "3b")
    section = presets.resolve(section)
    if boundaries is not None:
        return section, parse_boundaries(boundaries)
    if entry is not None:
        return section, (parse_boundaries(entry.boundaries)
                         if entry.boundaries else None)
    default = presets.DEFAULT_BOUNDARIES.get(section)
    return section, list(default) if default else None


def _step_from_name(path: str) -> int | None:
    import re
    m = re.search(r"step[_-]?0*(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Uncheatable Eval BPB per domain")
    ap.add_argument("--checkpoint", required=True,
                    help="path or kotodama.ckpt registry name")
    ap.add_argument("--config-section", default=None,
                    help="presets shape (3b, 1b, proxy, ...); default: the "
                         "registry entry's size, else 3b")
    ap.add_argument("--attn-res-boundaries", default=None,
                    help="presets name (dd3b, ddv1), comma list, or 'none'; "
                         "default: registry entry, else the shape's default")
    ap.add_argument("--rope-fp32", action="store_true",
                    help="model config rope_fp32=True (the ladder arms)")
    ap.add_argument("--data-dir", default=os.environ.get(DATA_ENV, DEFAULT_DATASET),
                    help=f"domain files / parquet / HF dataset id "
                         f"(default ${DATA_ENV}, else {DEFAULT_DATASET})")
    ap.add_argument("--domains", default=None,
                    help="comma-separated subset (default: all found)")
    ap.add_argument("--max-docs-per-domain", type=int, default=None)
    ap.add_argument("--max-tokens-per-domain", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--no-static-shapes", dest="static_shapes", action="store_false",
                    help="forward each chunk at its own length instead of padding to "
                         "chunk_size + 1 (recompiles shape-specialized kernels per length)")
    ap.add_argument("--loss-precision", choices=LOSS_PRECISIONS, default="bf16",
                    help="bf16 = the original runner's rounding (parity); "
                         "fp32 = exact")
    ap.add_argument("--attn-impl", default=None,
                    help="model attn_impl (default: auto on cuda, sdpa on cpu)")
    ap.add_argument("--step", type=int, default=None,
                    help="override the checkpoint's step (for hm_backfill)")
    ap.add_argument("--tokens", type=int, default=None,
                    help="override the checkpoint's tokens_consumed")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else cpu")
    ap.add_argument("--out", required=True, help="result JSON path")
    args = ap.parse_args(argv)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    section, bounds = resolve_arch(args.checkpoint, args.config_section,
                                   args.attn_res_boundaries)

    domains = load_domains(args.data_dir)
    if args.domains:
        want = [d.strip() for d in args.domains.split(",") if d.strip()]
        missing = [d for d in want if d not in domains]
        if missing:
            raise SystemExit(f"unknown domains {missing}; have {sorted(domains)}")
        domains = {d: domains[d] for d in want}
    logger.info("data: %d domains, %d docs from %s", len(domains),
                sum(len(v) for v in domains.values()), args.data_dir)

    from transformers import AutoTokenizer

    from kotodama.eval.model_loader import load_model
    from kotodama.presets import TOKENIZER_NAME

    model_cfg: dict[str, Any] = {"attn_res": bounds is not None}
    if bounds is not None:
        model_cfg["attn_res_boundaries"] = bounds
    if args.rope_fp32:
        model_cfg["rope_fp32"] = True
    attn_impl = args.attn_impl or ("sdpa" if "cuda" not in device else None)
    if attn_impl:
        model_cfg["attn_impl"] = attn_impl
    meta: dict[str, Any] = {}
    model = load_model(args.checkpoint, config_section=section,
                       attn_res_config=model_cfg, device=device, meta=meta)
    if args.chunk_size + 1 > model.config.max_position_embeddings:
        raise SystemExit(f"chunk_size {args.chunk_size} + BOS exceeds "
                         f"max_position_embeddings {model.config.max_position_embeddings}")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tok.bos_token_id is None:
        raise SystemExit(f"{TOKENIZER_NAME} has no BOS token")
    if len(tok) > model.config.vocab_size:
        raise SystemExit(f"tokenizer size {len(tok)} > model vocab {model.config.vocab_size}")

    def encode(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids)["logits"][0]

    step = args.step if args.step is not None else meta.get("step")
    if step is None:
        step = _step_from_name(str(args.checkpoint))
    tokens = args.tokens if args.tokens is not None else meta.get("tokens_consumed")

    settings = Settings(
        checkpoint=str(args.checkpoint), config_section=section,
        attn_res=bounds is not None, attn_res_boundaries=bounds,
        rope_fp32=bool(args.rope_fp32), tokenizer=TOKENIZER_NAME,
        bos_mode="add_default_bos", bos_id=int(tok.bos_token_id),
        chunk_size=args.chunk_size, loss_precision=args.loss_precision,
        max_docs_per_domain=args.max_docs_per_domain,
        max_tokens_per_domain=args.max_tokens_per_domain,
        static_shapes=args.static_shapes,
        data_source=str(args.data_dir), device=device,
        extra={"attn_impl": model.config.attn_impl,
               "dtype": str(next(model.parameters()).dtype),
               "n_params": sum(p.numel() for p in model.parameters())},
    )
    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, leave=False)  # noqa: E731
    except ImportError:
        progress = None

    result = evaluate(logits_fn, encode, domains, settings, step=step,
                      tokens=tokens, progress=progress)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    width = max(len(d) for d in result["domains"])
    for name, d in result["domains"].items():
        print(f"{name:<{width}}  {d['bpb']:.6f}  ({d['n_docs']}/{d['n_docs_total']} docs)")
    print(f"{'MEAN':<{width}}  {result['mean_bpb']:.6f}  over {result['n_domains']} "
          f"domains{' (trimmed)' if result['trimmed'] else ''}  -> {out}")


if __name__ == "__main__":
    main()
