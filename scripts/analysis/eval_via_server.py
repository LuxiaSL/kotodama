#!/usr/bin/env python3
"""Generate Track 3 eval outputs by hitting a running serve.py instance.

Usage::

    # Against a local or remote server
    python scripts/analysis/eval_via_server.py --url http://node2:2222 --name owt-ddv1

    # Custom config
    python scripts/analysis/eval_via_server.py --url http://node2:2222 --name owt-ddv1 \
        --prompt-set extended --n-samples 3 --temperature 0.7 --max-tokens 512
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_prompts(prompt_set: str, config_path: str = "configs/prompts.yaml") -> list[str]:
    with open(config_path) as f:
        all_prompts = yaml.safe_load(f)
    if prompt_set not in all_prompts:
        raise ValueError(f"Unknown prompt set '{prompt_set}'. Available: {list(all_prompts.keys())}")
    prompts = all_prompts[prompt_set]
    if not isinstance(prompts, list):
        raise ValueError(f"Prompt set '{prompt_set}' is not a list")
    return prompts


def generate_one(url: str, prompt: str, max_tokens: int, temperature: float,
                 top_p: float, top_k: int) -> dict[str, Any]:
    import httpx
    resp = httpx.post(
        f"{url}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Track 3 eval via serve.py endpoint")
    p.add_argument("--url", type=str, required=True, help="Server URL (e.g. http://node2:2222)")
    p.add_argument("--name", type=str, required=True, help="Run name for output")
    p.add_argument("--prompt-set", type=str, default="extended")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    prompts = load_prompts(args.prompt_set)
    total = len(prompts) * args.n_samples
    logger.info("Generating %d samples (%d prompts x %d samples) from %s",
                total, len(prompts), args.n_samples, args.url)

    # Check server health
    import httpx
    try:
        health = httpx.get(f"{args.url}/health", timeout=10).json()
        if not health.get("model_loaded"):
            logger.error("Server reports model not loaded")
            sys.exit(1)
        info = httpx.get(f"{args.url}/info", timeout=10).json()
        logger.info("Server: %s, %s params, device=%s, fast_attnres=%s",
                     info.get("name"), info.get("params"), info.get("device"),
                     info.get("fast_attnres"))
    except Exception as e:
        logger.error("Cannot reach server at %s: %s", args.url, e)
        sys.exit(1)

    gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p if args.top_p > 0 else None,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n_samples": args.n_samples,
        "prompt_set": args.prompt_set,
    }

    samples: list[dict[str, Any]] = []
    t0 = time.time()
    total_tokens = 0

    for pi, prompt in enumerate(prompts):
        for si in range(args.n_samples):
            try:
                result = generate_one(
                    args.url, prompt, args.max_tokens,
                    args.temperature, args.top_p, args.top_k,
                )
                n_tokens = result.get("completion_tokens", 0)
                tps = result.get("tokens_per_second", 0)
                total_tokens += n_tokens

                samples.append({
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "sample_idx": si,
                    "continuation": result.get("text", ""),
                    "n_tokens": n_tokens,
                    "tokens_per_second": tps,
                    "stopped_by": "max_tokens" if n_tokens >= args.max_tokens else "eos",
                })

                done = pi * args.n_samples + si + 1
                elapsed = time.time() - t0
                logger.info("[%d/%d] prompt %d sample %d: %d tok @ %.1f tok/s (%.0fs elapsed)",
                            done, total, pi, si, n_tokens, tps, elapsed)

            except Exception as e:
                logger.error("[%d/%d] Failed: %s", pi * args.n_samples + si + 1, total, e)
                samples.append({
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "sample_idx": si,
                    "continuation": f"[ERROR: {e}]",
                    "n_tokens": 0,
                    "tokens_per_second": 0,
                    "stopped_by": "error",
                })

    elapsed = time.time() - t0
    avg_tps = total_tokens / elapsed if elapsed > 0 else 0

    output = {
        "config": gen_config,
        "runs": [{
            "name": args.name,
            "checkpoint": info.get("checkpoint", "unknown"),
            "samples": samples,
        }],
    }

    output_path = args.output or Path("analysis") / args.name / "generations.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Done: %d samples, %d total tokens, %.1f avg tok/s, %.0fs total",
                len(samples), total_tokens, avg_tps, elapsed)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    main()
