#!/usr/bin/env python3
"""Unified evaluation generation pipeline (Track 3).

Generates text samples from one or more checkpoints with held-out perplexity.

Replaces: eval_lang_full.py, eval_lang_full_multisample.py, eval_full_matrix.py,
eval_temperature_matrix.py, eval_nca_vs_p3.py, eval_proxy_sweep.py.

Usage::

    # Single checkpoint
    python -m scripts.eval_generate --checkpoint path/to/ckpt.pt --name my-run

    # Multiple checkpoints from registry
    python -m scripts.eval_generate --checkpoints P3-Muon-002,NCA-002

    # Custom generation config
    python -m scripts.eval_generate --checkpoint ckpt.pt --name test \\
        --temperature 0.8 --top-p 0.9 --max-tokens 512 --n-samples 3

    # Skip perplexity (generation only)
    python -m scripts.eval_generate --checkpoint ckpt.pt --name test --no-ppl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.generate import generate_text
from src.eval.model_loader import CheckpointInfo, load_checkpoint_registry, load_model
from src.eval.perplexity import compute_perplexity

logger = logging.getLogger(__name__)

DEFAULT_EVAL_DATA = "data/fineweb_edu_eval_5m.bin"


def _load_prompts(prompt_set: str = "standard") -> list[str]:
    """Load prompts from configs/prompts.yaml."""
    with open("configs/prompts.yaml") as f:
        all_prompts = yaml.safe_load(f)
    if prompt_set not in all_prompts:
        raise ValueError(f"Unknown prompt set '{prompt_set}'. Available: {list(all_prompts.keys())}")
    prompts = all_prompts[prompt_set]
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt set '{prompt_set}' is not a list")
    return prompts


def _resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path, dict[str, Any] | None]]:
    """Resolve checkpoint specifications to (name, path, attn_res_config) triples."""
    runs: list[tuple[str, Path, dict[str, Any] | None]] = []

    if args.checkpoint:
        runs.append((
            args.name or Path(args.checkpoint).stem,
            Path(args.checkpoint),
            _parse_attn_res(args.attn_res) if args.attn_res else None,
        ))
    elif args.checkpoints:
        registry = load_checkpoint_registry()
        for name in args.checkpoints.split(","):
            name = name.strip()
            if name not in registry:
                logger.warning("Checkpoint '%s' not in registry, skipping", name)
                continue
            info = registry[name]
            runs.append((info.name, info.path, info.attn_res_config))

    return runs


def _parse_attn_res(spec: str) -> dict[str, Any]:
    """Parse AttnRes spec string like 'n_blocks=7' or 'boundaries=0,3,7,12'."""
    result: dict[str, Any] = {"attn_res": True}
    for part in spec.split():
        key, val = part.split("=", 1)
        if key == "n_blocks":
            result["attn_res_n_blocks"] = int(val)
        elif key == "boundaries":
            result["attn_res_boundaries"] = [int(x) for x in val.split(",")]
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Unified evaluation generation pipeline (Track 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint selection (mutually exclusive)
    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--checkpoint", type=Path, help="Single checkpoint path")
    ckpt.add_argument("--checkpoints", type=str,
                      help="Comma-separated checkpoint names from registry")

    # Run identification
    p.add_argument("--name", type=str, default=None, help="Run name (for single checkpoint)")
    p.add_argument("--attn-res", type=str, default=None,
                   help="AttnRes config: 'n_blocks=7' or 'boundaries=0,3,7,12'")

    # Generation config
    p.add_argument("--prompt-set", type=str, default="standard",
                   help="Prompt set from configs/prompts.yaml (default: standard)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--n-samples", type=int, default=1,
                   help="Independent generations per prompt")

    # Perplexity
    p.add_argument("--no-ppl", action="store_true", help="Skip perplexity computation")
    p.add_argument("--eval-data", type=Path, default=Path(DEFAULT_EVAL_DATA))
    p.add_argument("--max-seqs", type=int, default=400)

    # Model config
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--config-section", type=str, default="proxy")
    p.add_argument("--device", type=str, default="cuda:0")

    # Output
    p.add_argument("-o", "--output", type=Path, default=None)

    args = p.parse_args()

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        sys.exit(1)

    # Load prompts
    prompts = _load_prompts(args.prompt_set)
    logger.info("Prompts: %d from set '%s'", len(prompts), args.prompt_set)

    # Resolve checkpoints
    runs = _resolve_checkpoints(args)
    if not runs:
        logger.error("No valid checkpoints specified")
        sys.exit(1)

    # Generation config
    gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n_samples": args.n_samples,
        "prompt_set": args.prompt_set,
    }

    total_gens = len(prompts) * args.n_samples
    all_runs: list[dict[str, Any]] = []

    for run_name, ckpt_path, ar_config in runs:
        if not ckpt_path.exists():
            logger.warning("Skipping %s: %s not found", run_name, ckpt_path)
            continue

        logger.info("=" * 60)
        logger.info("Evaluating: %s (%d generations)", run_name, total_gens)
        logger.info("=" * 60)

        try:
            model = load_model(
                ckpt_path,
                config_path=args.config,
                config_section=args.config_section,
                attn_res_config=ar_config,
                device=args.device,
            )
        except Exception as e:
            logger.error("Failed to load %s: %s", run_name, e)
            continue

        run_result: dict[str, Any] = {
            "name": run_name,
            "checkpoint": str(ckpt_path),
        }

        # Held-out perplexity
        if not args.no_ppl and args.eval_data.exists():
            logger.info("Computing held-out perplexity...")
            ppl_result = compute_perplexity(
                model, args.eval_data,
                max_seqs=args.max_seqs,
                device=args.device,
            )
            run_result["eval_loss"] = ppl_result["loss"]
            run_result["eval_ppl"] = ppl_result["perplexity"]

        # Generate samples
        samples: list[dict[str, Any]] = []
        t0 = time.time()

        with torch.no_grad():
            for pi, prompt in enumerate(prompts):
                for si in range(args.n_samples):
                    result = generate_text(
                        model, tokenizer, prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    samples.append({
                        "prompt_idx": pi,
                        "prompt": result["prompt"],
                        "sample_idx": si,
                        "continuation": result["continuation"],
                        "n_tokens": result["n_tokens"],
                        "stopped_by": result["stopped_by"],
                    })

                    done = pi * args.n_samples + si + 1
                    elapsed = time.time() - t0
                    logger.info(
                        "  [%d/%d] prompt %d sample %d: %d tokens (%s, %.0fs)",
                        done, total_gens, pi, si,
                        result["n_tokens"], result["stopped_by"], elapsed,
                    )

        run_result["samples"] = samples
        all_runs.append(run_result)

        del model
        torch.cuda.empty_cache()

    # Build output
    output = {
        "config": gen_config,
        "runs": all_runs,
    }

    # Determine output path
    output_path = args.output
    if output_path is None:
        if len(all_runs) == 1:
            output_dir = Path("analysis") / all_runs[0]["name"]
        else:
            names = "_vs_".join(r["name"] for r in all_runs[:3])
            output_dir = Path("analysis") / names
        output_path = output_dir / "generations.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d runs (%d total samples) to %s",
                len(all_runs),
                sum(len(r["samples"]) for r in all_runs),
                output_path)

    # Summary
    for r in all_runs:
        ppl_str = f"ppl={r['eval_ppl']:.2f}" if "eval_ppl" in r else "no-ppl"
        print(f"  {r['name']}: {len(r['samples'])} samples, {ppl_str}")


if __name__ == "__main__":
    main()
