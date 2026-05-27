#!/usr/bin/env python3
"""Benchmark harness for luxia-base serve.py.

Runs a fixed prompt pack against the server across decode modes and generation
lengths, capturing latency, throughput, memory, and token-level reproducibility.

Usage:
    # Generate prompt pack first
    python scripts/benchmark/generate_prompt_pack.py

    # Run benchmark against a running server
    python scripts/benchmark/bench.py --url http://localhost:2222

    # Specific prompt lengths and generation lengths
    python scripts/benchmark/bench.py --url http://localhost:2222 \
        --prompt-lengths 128 1024 --gen-lengths 32 256

    # Save results
    python scripts/benchmark/bench.py --url http://localhost:2222 \
        --output scripts/benchmark/results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PACK = Path(__file__).parent / "prompt_pack.json"

DECODE_MODES: dict[str, dict[str, Any]] = {
    "greedy": {"temperature": 0.0, "top_k": 0, "top_p": 0.0},
    "top_k": {"temperature": 0.7, "top_k": 50, "top_p": 0.0},
    "top_p": {"temperature": 0.7, "top_k": 0, "top_p": 0.9},
}

GEN_LENGTHS = [32, 256, 1024]
PROMPT_LENGTHS = [16, 128, 1024, 3072]


@dataclass
class BenchResult:
    prompt_name: str
    prompt_tokens: int
    gen_length_target: int
    decode_mode: str
    stream: bool
    completion_tokens: int
    total_latency_s: float
    tokens_per_second: float
    generated_text: str
    generated_token_ids: list[int] | None = None
    server_timing: dict[str, Any] | None = None
    error: str | None = None


def run_one(
    url: str,
    prompt_text: str,
    prompt_token_ids: list[int] | None,
    gen_length: int,
    decode_mode: str,
    stream: bool,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Send a single generation request and capture timing."""
    mode_params = DECODE_MODES[decode_mode]

    payload: dict[str, Any] = {
        "prompt": prompt_text,
        "max_new_tokens": gen_length,
        "stream": stream,
        **mode_params,
    }

    t0 = time.perf_counter()

    if stream:
        tokens_text: list[str] = []
        completion_tokens = 0
        first_token_time: float | None = None

        with httpx.stream("POST", f"{url}/generate", json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if "token" in data:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    tokens_text.append(data["token"])
                if "done" in data and data["done"]:
                    completion_tokens = data.get("completion_tokens", len(tokens_text))

        elapsed = time.perf_counter() - t0
        text = "".join(tokens_text)
        ttft = (first_token_time - t0) if first_token_time else elapsed

        return {
            "text": text,
            "completion_tokens": completion_tokens,
            "total_latency_s": elapsed,
            "ttft_s": ttft,
            "tokens_per_second": completion_tokens / elapsed if elapsed > 0 else 0,
        }
    else:
        resp = httpx.post(f"{url}/generate", json=payload, timeout=timeout)
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()

        result: dict[str, Any] = {
            "text": data.get("text", ""),
            "completion_tokens": data.get("completion_tokens", 0),
            "total_latency_s": elapsed,
            "tokens_per_second": data.get("tokens_per_second", 0),
        }

        if "timing" in data:
            result["server_timing"] = data["timing"]

        return result


def get_server_info(url: str) -> dict[str, Any]:
    """Fetch server info and verify it's ready."""
    health = httpx.get(f"{url}/health", timeout=10).json()
    if not health.get("model_loaded"):
        raise RuntimeError("Server reports model not loaded")
    info = httpx.get(f"{url}/info", timeout=10).json()
    return info


def get_memory_stats(url: str) -> dict[str, Any] | None:
    """Fetch GPU memory stats if the server exposes them."""
    try:
        resp = httpx.get(f"{url}/memory", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def run_benchmark(
    url: str,
    prompt_pack_path: Path,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    decode_modes: list[str],
    test_stream: bool = True,
    warmup_runs: int = 2,
) -> dict[str, Any]:
    """Run the full benchmark suite."""
    with open(prompt_pack_path) as f:
        pack = json.load(f)

    server_info = get_server_info(url)
    logger.info(
        "Server: %s, %s params, device=%s, triton_attn_res=%s, compiled_decode=%s",
        server_info.get("name"), server_info.get("params"),
        server_info.get("device"), server_info.get("triton_attn_res"),
        server_info.get("compiled_decode"),
    )

    prompts_by_length: dict[int, list[dict]] = {}
    for p in pack["prompts"]:
        length = p["actual_tokens"]
        prompts_by_length.setdefault(length, []).append(p)

    selected_prompts: list[dict] = []
    for target_len in prompt_lengths:
        matching = prompts_by_length.get(target_len, [])
        if not matching:
            logger.warning("No prompts at length %d, skipping", target_len)
            continue
        selected_prompts.extend(matching)

    logger.info(
        "Selected %d prompts across lengths %s",
        len(selected_prompts), sorted(set(p["actual_tokens"] for p in selected_prompts)),
    )

    # Warmup: run a few short generations to JIT-compile / warm caches
    if warmup_runs > 0:
        logger.info("Running %d warmup generations...", warmup_runs)
        warmup_prompt = selected_prompts[0]
        for i in range(warmup_runs):
            try:
                run_one(url, warmup_prompt["text"], warmup_prompt.get("token_ids"), 16, "greedy", False)
            except Exception as e:
                logger.warning("Warmup %d failed: %s", i, e)

    results: list[dict[str, Any]] = []
    total_runs = len(selected_prompts) * len(gen_lengths) * len(decode_modes)
    if test_stream:
        total_runs += len(selected_prompts) * len(gen_lengths)
    run_idx = 0

    for prompt in selected_prompts:
        for gen_len in gen_lengths:
            for mode in decode_modes:
                run_idx += 1
                name = f"{prompt['name']}/{mode}/gen{gen_len}"
                logger.info("[%d/%d] %s", run_idx, total_runs, name)

                try:
                    result = run_one(
                        url, prompt["text"], prompt.get("token_ids"),
                        gen_len, mode, stream=False,
                    )
                    result.update({
                        "prompt_name": prompt["name"],
                        "prompt_tokens": prompt["actual_tokens"],
                        "prompt_domain": prompt["domain"],
                        "gen_length_target": gen_len,
                        "decode_mode": mode,
                        "stream": False,
                    })
                    results.append(result)

                    logger.info(
                        "  -> %d tokens, %.2fs, %.1f tok/s",
                        result["completion_tokens"],
                        result["total_latency_s"],
                        result["tokens_per_second"],
                    )
                except Exception as e:
                    logger.error("  -> FAILED: %s", e)
                    results.append({
                        "prompt_name": prompt["name"],
                        "prompt_tokens": prompt["actual_tokens"],
                        "gen_length_target": gen_len,
                        "decode_mode": mode,
                        "stream": False,
                        "error": str(e),
                    })

            # Streaming test (greedy only to keep suite manageable)
            if test_stream:
                run_idx += 1
                name = f"{prompt['name']}/stream-greedy/gen{gen_len}"
                logger.info("[%d/%d] %s", run_idx, total_runs, name)

                try:
                    result = run_one(
                        url, prompt["text"], prompt.get("token_ids"),
                        gen_len, "greedy", stream=True,
                    )
                    result.update({
                        "prompt_name": prompt["name"],
                        "prompt_tokens": prompt["actual_tokens"],
                        "prompt_domain": prompt.get("domain"),
                        "gen_length_target": gen_len,
                        "decode_mode": "greedy",
                        "stream": True,
                    })
                    results.append(result)

                    logger.info(
                        "  -> %d tokens, %.2fs, TTFT=%.3fs",
                        result["completion_tokens"],
                        result["total_latency_s"],
                        result.get("ttft_s", -1),
                    )
                except Exception as e:
                    logger.error("  -> FAILED: %s", e)
                    results.append({
                        "prompt_name": prompt["name"],
                        "prompt_tokens": prompt["actual_tokens"],
                        "gen_length_target": gen_len,
                        "decode_mode": "greedy",
                        "stream": True,
                        "error": str(e),
                    })

    memory = get_memory_stats(url)

    return {
        "metadata": {
            "server_info": server_info,
            "prompt_pack": str(prompt_pack_path),
            "prompt_lengths": prompt_lengths,
            "gen_lengths": gen_lengths,
            "decode_modes": decode_modes,
            "test_stream": test_stream,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "memory": memory,
        "results": results,
    }


def format_summary(benchmark: dict[str, Any]) -> str:
    """Generate a Markdown summary table from benchmark results."""
    lines = [
        "# Benchmark Summary",
        "",
        f"**Timestamp:** {benchmark['metadata']['timestamp']}",
        f"**Server:** {benchmark['metadata']['server_info'].get('name', 'unknown')}",
        f"**Device:** {benchmark['metadata']['server_info'].get('device', 'unknown')}",
        "",
    ]

    results = [r for r in benchmark["results"] if "error" not in r and not r.get("stream")]
    if not results:
        lines.append("No successful results.")
        return "\n".join(lines)

    # Group by prompt length
    by_prompt_len: dict[int, list[dict]] = {}
    for r in results:
        by_prompt_len.setdefault(r["prompt_tokens"], []).append(r)

    # Non-streaming results table
    lines.extend([
        "## Latency & Throughput (non-streaming)",
        "",
        "| Prompt Tokens | Gen Target | Mode | Comp. Tokens | Latency (s) | Tok/s |",
        "|---:|---:|:---|---:|---:|---:|",
    ])

    for plen in sorted(by_prompt_len.keys()):
        group = by_prompt_len[plen]
        by_gen: dict[int, dict[str, list[dict]]] = {}
        for r in group:
            by_gen.setdefault(r["gen_length_target"], {}).setdefault(r["decode_mode"], []).append(r)

        for gen_len in sorted(by_gen.keys()):
            for mode in sorted(by_gen[gen_len].keys()):
                runs = by_gen[gen_len][mode]
                avg_tokens = statistics.mean(r["completion_tokens"] for r in runs)
                avg_latency = statistics.mean(r["total_latency_s"] for r in runs)
                avg_tps = statistics.mean(r["tokens_per_second"] for r in runs)
                lines.append(
                    f"| {plen} | {gen_len} | {mode} | {avg_tokens:.0f} | {avg_latency:.2f} | {avg_tps:.1f} |"
                )

    # Streaming results
    stream_results = [r for r in benchmark["results"] if r.get("stream") and "error" not in r]
    if stream_results:
        lines.extend([
            "",
            "## Streaming (greedy)",
            "",
            "| Prompt Tokens | Gen Target | TTFT (s) | Total (s) | Comp. Tokens |",
            "|---:|---:|---:|---:|---:|",
        ])
        for r in sorted(stream_results, key=lambda x: (x["prompt_tokens"], x["gen_length_target"])):
            lines.append(
                f"| {r['prompt_tokens']} | {r['gen_length_target']} | "
                f"{r.get('ttft_s', -1):.3f} | {r['total_latency_s']:.2f} | {r['completion_tokens']} |"
            )

    # Server timing breakdown if available
    timed = [r for r in results if r.get("server_timing")]
    if timed:
        lines.extend([
            "",
            "## Server Timing Breakdown (sample)",
            "",
        ])
        sample = timed[0]
        timing = sample["server_timing"]
        lines.append(f"Prompt: {sample['prompt_name']}, Gen: {sample['gen_length_target']}, Mode: {sample['decode_mode']}")
        lines.append("")
        lines.append("| Phase | Time (ms) |")
        lines.append("|:---|---:|")
        for phase, ms in timing.items():
            if isinstance(ms, (int, float)):
                lines.append(f"| {phase} | {ms:.1f} |")

    # Memory
    if benchmark.get("memory"):
        mem = benchmark["memory"]
        lines.extend([
            "",
            "## GPU Memory",
            "",
            f"- Allocated: {mem.get('allocated_gb', '?')} GB",
            f"- Reserved: {mem.get('reserved_gb', '?')} GB",
            f"- Peak: {mem.get('peak_gb', '?')} GB",
        ])

    # Errors
    errors = [r for r in benchmark["results"] if "error" in r]
    if errors:
        lines.extend([
            "",
            f"## Errors ({len(errors)} failures)",
            "",
        ])
        for e in errors[:5]:
            lines.append(f"- {e.get('prompt_name', '?')}/{e.get('decode_mode', '?')}: {e['error']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark luxia-base serve.py")
    parser.add_argument("--url", type=str, default="http://localhost:2222")
    parser.add_argument("--prompt-pack", type=Path, default=DEFAULT_PROMPT_PACK)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=PROMPT_LENGTHS)
    parser.add_argument("--gen-lengths", type=int, nargs="+", default=GEN_LENGTHS)
    parser.add_argument("--decode-modes", nargs="+", default=list(DECODE_MODES.keys()),
                        choices=list(DECODE_MODES.keys()))
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    if not args.prompt_pack.exists():
        logger.error("Prompt pack not found at %s. Run generate_prompt_pack.py first.", args.prompt_pack)
        raise SystemExit(1)

    benchmark = run_benchmark(
        url=args.url,
        prompt_pack_path=args.prompt_pack,
        prompt_lengths=args.prompt_lengths,
        gen_lengths=args.gen_lengths,
        decode_modes=args.decode_modes,
        test_stream=not args.no_stream,
        warmup_runs=args.warmup,
    )

    summary = format_summary(benchmark)
    print("\n" + summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        json_path = args.output.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(benchmark, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", json_path)

        md_path = args.output.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(summary)
        logger.info("Summary saved to %s", md_path)


if __name__ == "__main__":
    main()
