"""Run lm-evaluation-harness benchmarks on LuxiaBaseModel checkpoints.

Usage:
    cd pretraining
    python -m scripts.eval.run_lm_eval --checkpoint path/to/step.pt.zst
    python -m scripts.eval.run_lm_eval --checkpoint ckpt1.pt,ckpt2.pt --tasks hellaswag,piqa --limit 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TASKS = "hellaswag,piqa,arc_easy,arc_challenge,boolq,lambada_openai,winogrande,wikitext,copa,sciq"

DEFAULT_ATTN_RES_CONFIG = {
    "attn_res": True,
    "attn_res_n_blocks": 7,
    "attn_res_boundaries": [0, 3, 7, 12, 21, 25],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks on LuxiaBaseModel checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path(s), comma-separated for multiple",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=f"Comma-separated task names (default: {DEFAULT_TASKS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Rows per forward pass (batched loglikelihood; 32 is a good "
             "B200 default for the 3B)",
    )
    parser.add_argument(
        "--max-batch-tokens", type=int, default=32768,
        help="Cap on rows x padded-width per forward pass; bounds memory "
             "for long-sequence batches (wikitext rolling windows)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/lm_eval",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit examples per task"
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=None, help="Override few-shot count"
    )
    parser.add_argument(
        "--config", type=str, default="configs/model.yaml",
    )
    parser.add_argument(
        "--config-section", type=str, default="proxy",
    )
    parser.add_argument(
        "--no-attn-res",
        action="store_true",
        help="Disable AttnRes (for baseline checkpoints)",
    )
    parser.add_argument(
        "--attn-res-boundaries",
        type=str,
        default=None,
        help="AttnRes block boundaries as comma-separated ints (e.g. '0,1,3,7,15,19,24'). "
             "Overrides the default DD-v1 boundaries.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for faster inference",
    )
    return parser.parse_args()


def run_eval(args: argparse.Namespace) -> None:
    import lm_eval

    from src.eval.lm_eval_adapter import LuxiaEvalLM

    checkpoints = [p.strip() for p in args.checkpoint.split(",")]
    task_list = [t.strip() for t in args.tasks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_attn_res:
        attn_res_config = None
    elif args.attn_res_boundaries:
        boundaries = [int(x) for x in args.attn_res_boundaries.split(",")]
        attn_res_config = {
            "attn_res": True,
            "attn_res_boundaries": boundaries,
        }
    else:
        attn_res_config = DEFAULT_ATTN_RES_CONFIG
    all_results: dict[str, dict] = {}

    for ckpt_path in checkpoints:
        ckpt_name = Path(ckpt_path).stem
        logger.info("Evaluating checkpoint: %s", ckpt_name)

        lm = None
        try:
            lm = LuxiaEvalLM(
                checkpoint_path=ckpt_path,
                config_path=args.config,
                config_section=args.config_section,
                attn_res_config=attn_res_config,
                device=args.device,
                batch_size=args.batch_size,
                compile=getattr(args, "compile", False),
                max_batch_tokens=args.max_batch_tokens,
            )

            eval_kwargs: dict = {
                "model": lm,
                "tasks": task_list,
                "log_samples": False,
            }
            if args.limit is not None:
                eval_kwargs["limit"] = args.limit
            if args.num_fewshot is not None:
                eval_kwargs["num_fewshot"] = args.num_fewshot

            results = lm_eval.simple_evaluate(**eval_kwargs)

            # Save full results
            ckpt_dir = output_dir / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            results_file = ckpt_dir / "results.json"

            serializable = results.get("results", {})
            with open(results_file, "w") as f:
                json.dump(serializable, f, indent=2, default=str)
            logger.info("Results saved to %s", results_file)

            # Print summary
            print(f"\n{'=' * 60}")
            print(f"  {ckpt_name}")
            print(f"{'=' * 60}")
            for task_name, task_results in serializable.items():
                metrics = []
                for key, val in task_results.items():
                    if key.endswith(",none") and isinstance(val, (int, float)):
                        metric_name = key.replace(",none", "")
                        metrics.append(f"{metric_name}={val:.4f}")
                if metrics:
                    print(f"  {task_name}: {', '.join(metrics)}")

            all_results[ckpt_name] = serializable

        except Exception:
            logger.exception("Failed to evaluate %s", ckpt_path)
            continue
        finally:
            # Free GPU memory between checkpoints
            del lm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary comparison if multiple checkpoints
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("  COMPARISON")
        print(f"{'=' * 60}")
        tasks_seen = set()
        for res in all_results.values():
            tasks_seen.update(res.keys())

        for task in sorted(tasks_seen):
            print(f"\n  {task}:")
            for ckpt_name, res in all_results.items():
                if task in res:
                    metrics = {
                        k.replace(",none", ""): v
                        for k, v in res[task].items()
                        if k.endswith(",none") and isinstance(v, (int, float))
                    }
                    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
                    print(f"    {ckpt_name}: {', '.join(parts)}")

        # Save comparison
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Comparison saved to %s", comparison_file)


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
