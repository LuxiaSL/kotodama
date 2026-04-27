#!/usr/bin/env python3
"""Extract benchmark results from Heimdall job logs.

Usage: python tools/extract_benchmark_results.py <job_id1> <job_id2> ...
       python tools/extract_benchmark_results.py --prefix bench-
"""
import json
import re
import subprocess
import sys
from typing import Optional


HEIMDALL_API = "http://node1.datasci.ath:7000"


def get_job(job_id: str) -> dict:
    result = subprocess.run(
        ["curl", "-s", f"{HEIMDALL_API}/api/v1/jobs/{job_id}"],
        capture_output=True, text=True, timeout=10,
    )
    return json.loads(result.stdout)


def list_jobs(prefix: str, limit: int = 20) -> list[dict]:
    result = subprocess.run(
        ["curl", "-s", f"{HEIMDALL_API}/api/v1/jobs?limit={limit}"],
        capture_output=True, text=True, timeout=10,
    )
    data = json.loads(result.stdout)
    jobs = data if isinstance(data, list) else data.get("jobs", [])
    return [j for j in jobs if j["spec"]["name"].startswith(prefix)]


def extract_metrics(log_tail: str) -> dict:
    """Extract tok/s, mem, loss from Heimdall log tail."""
    tps_vals = [float(m) for m in re.findall(r'tok/s=(\d+\.?\d*)', log_tail)]
    mem_vals = re.findall(r'gpu_mem=(\d+\.?\d*)GB', log_tail)
    loss_vals = re.findall(r'loss=(\d+\.?\d*)', log_tail)

    # Skip first 5 entries (warmup/compilation)
    steady_tps = tps_vals[5:] if len(tps_vals) > 5 else tps_vals

    return {
        "avg_tok_per_sec": int(sum(steady_tps) / len(steady_tps)) if steady_tps else 0,
        "peak_tok_per_sec": int(max(tps_vals)) if tps_vals else 0,
        "min_tok_per_sec": int(min(steady_tps)) if steady_tps else 0,
        "peak_gpu_mem_gb": float(mem_vals[-1]) if mem_vals else 0,
        "final_loss": float(loss_vals[-1]) if loss_vals else 0,
        "n_samples": len(steady_tps),
    }


def get_full_log(job_id: str) -> str:
    """Get full log from Heimdall."""
    result = subprocess.run(
        ["curl", "-s", f"{HEIMDALL_API}/api/v1/jobs/{job_id}/logs?lines=500"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        data = json.loads(result.stdout)
        return data.get("log", data.get("logs", result.stdout))
    except json.JSONDecodeError:
        return result.stdout


def main():
    if "--prefix" in sys.argv:
        idx = sys.argv.index("--prefix")
        prefix = sys.argv[idx + 1]
        jobs_data = list_jobs(prefix)
        job_ids = [j["id"] for j in jobs_data]
    else:
        job_ids = sys.argv[1:]

    if not job_ids:
        print("Usage: extract_benchmark_results.py <job_id>... or --prefix <prefix>")
        sys.exit(1)

    results = []
    for jid in job_ids:
        job = get_job(jid)
        name = job["spec"]["name"]
        status = job["status"]
        command = job["spec"]["command"]

        # Parse flags from command
        has_liger = "--use_liger" in command
        has_compile = "--compile" in command
        attn_impl = "sdpa"
        if "--attn_impl fa2" in command:
            attn_impl = "fa2"
        elif "--attn_impl fa4" in command:
            attn_impl = "fa4"

        log = job.get("log_tail", "")
        if not log:
            log = get_full_log(jid)

        metrics = extract_metrics(log)

        result = {
            "id": jid[:12],
            "name": name,
            "status": status,
            "liger": has_liger,
            "attn": attn_impl,
            "compile": has_compile,
            **metrics,
        }
        results.append(result)

    # Find baseline for speedup calculation
    baseline_tps: Optional[float] = None
    for r in results:
        if not r["liger"] and not r["compile"] and r["attn"] == "sdpa":
            baseline_tps = r["avg_tok_per_sec"]
            break
    # Fallback: compile+sdpa (closest to past runs)
    if baseline_tps is None:
        for r in results:
            if not r["liger"] and r["compile"] and r["attn"] == "sdpa":
                baseline_tps = r["avg_tok_per_sec"]
                break

    # Print table
    header = f"{'Config':<28s} {'Status':>10s} {'tok/s':>10s} {'vs base':>8s} {'mem(GB)':>8s} {'loss':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        flags = []
        if r["liger"]:
            flags.append("liger")
        if r["compile"]:
            flags.append("compile")
        flags.append(r["attn"])
        config_name = " + ".join(flags) if flags else "bare"

        tps = r["avg_tok_per_sec"]
        if baseline_tps and baseline_tps > 0 and tps > 0:
            speedup = f"{tps / baseline_tps:.2f}x"
        else:
            speedup = "N/A"

        status = "OK" if r["status"] == "completed" else r["status"][:10]
        print(f"{config_name:<28s} {status:>10s} {tps:>10,d} {speedup:>8s} {r['peak_gpu_mem_gb']:>8.1f} {r['final_loss']:>8.3f}")

    # Dump JSON
    print("\n--- JSON ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
