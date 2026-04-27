"""
Submit the proxy validation sweep to Heimdall.

Submits 5 chained jobs (P1-P5) via the Heimdall REST API.
Each job depends on the previous one completing, so they run
sequentially on node1's 8×B200 GPUs.

Usage::

    python scripts/submit_proxy_sweep.py
    python scripts/submit_proxy_sweep.py --dry_run  # preview without submitting
    python scripts/submit_proxy_sweep.py --status    # check sweep status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional

try:
    import httpx
except ImportError:
    import urllib.request
    import urllib.error

    # Minimal httpx-like fallback using urllib
    class _FallbackClient:
        def __init__(self, base_url: str, timeout: float = 30.0):
            self.base_url = base_url
            self.timeout = timeout

        def post(self, path: str, json: Any = None) -> Any:
            url = f"{self.base_url}{path}"
            data = __import__("json").dumps(json).encode() if json else None
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"} if data else {},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = __import__("json").loads(resp.read())
                return type("Response", (), {"json": lambda self=body: body, "status_code": resp.status})()

        def get(self, path: str) -> Any:
            url = f"{self.base_url}{path}"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = __import__("json").loads(resp.read())
                return type("Response", (), {"json": lambda self=body: body, "status_code": resp.status})()

    httpx = None  # type: ignore


HEIMDALL_URL = "http://node1.datasci.ath:7000"
DATA_PATH = "data/fineweb_edu_6b.bin"
WORKING_DIR = "/home/athuser/luxi-files/luxia-base"
VENV_PYTHON = ".venv/bin/torchrun"

# Common training args for all runs
COMMON_ARGS = (
    "--data_path {data_path} "
    "--model_size proxy "
    "--sequence_length 2048 "
    "--micro_batch_size 4 "
    "--global_batch_tokens 131072 "
    "--warmup_steps 2000 "
    "--decay_start_pct 0.90 "
    "--decay_type sqrt "
    "--gradient_clip 1.0 "
    "--total_tokens 6000000000 "
    "--log_every 100 "
    "--save_every 2000 "
    "--keep_checkpoints 3 "
    "--geo_monitor "
    "--geo_monitor_tier1_every 200 "
    "--geo_monitor_tier2_every 2000 "
    "--wandb "
    "--activation_checkpointing "
    "--compile "
)

# Per-run configurations
SWEEP_RUNS = [
    {
        "name": "p1-adamw-baseline",
        "extra_args": "--adamw_only --adamw_lr 8e-4",
        "description": "AdamW baseline (no Muon)",
    },
    {
        "name": "p2-muon-lr001",
        "extra_args": "--muon_lr 0.01 --adamw_lr 6e-4",
        "description": "Muon LR 0.01",
    },
    {
        "name": "p3-muon-lr002",
        "extra_args": "--muon_lr 0.02 --adamw_lr 6e-4",
        "description": "Muon LR 0.02",
    },
    {
        "name": "p4-muon-lr003",
        "extra_args": "--muon_lr 0.03 --adamw_lr 6e-4",
        "description": "Muon LR 0.03 (default)",
    },
    {
        "name": "p5-muon-lr004",
        "extra_args": "--muon_lr 0.04 --adamw_lr 6e-4",
        "description": "Muon LR 0.04",
    },
]


def build_command(run: dict, data_path: str = DATA_PATH) -> str:
    """Build the full torchrun command for a sweep run."""
    args = COMMON_ARGS.format(data_path=data_path)
    checkpoint_dir = f"checkpoints/proxy_sweep/{run['name']}"
    wandb_name = f"proxy-{run['name']}"

    return (
        f"{VENV_PYTHON} --nproc_per_node=8 "
        f"-m src.training.train "
        f"{args} "
        f"--checkpoint_dir {checkpoint_dir} "
        f"--wandb_run_name {wandb_name} "
        f"{run['extra_args']}"
    )


def submit_sweep(
    dry_run: bool = False,
    data_path: str = DATA_PATH,
) -> list[dict[str, str]]:
    """Submit all sweep runs as chained Heimdall jobs."""

    if httpx is not None:
        client = httpx.Client(base_url=HEIMDALL_URL, timeout=30.0)
    else:
        client = _FallbackClient(HEIMDALL_URL)

    # Verify Heimdall is reachable
    try:
        resp = client.get("/api/v1/status")
        if hasattr(resp, 'status_code') and resp.status_code != 200:
            print(f"ERROR: Heimdall returned status {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Cannot reach Heimdall at {HEIMDALL_URL}: {e}")
        sys.exit(1)

    submitted: list[dict[str, str]] = []
    prev_job_id: Optional[str] = None

    for i, run in enumerate(SWEEP_RUNS):
        command = build_command(run, data_path)

        job_spec: dict[str, Any] = {
            "job_type": "custom",
            "name": f"luxia-proxy-{run['name']}",
            "command": command,
            "node": "node1",
            "gpus": 8,
            "working_dir": WORKING_DIR,
            "env": {
                "PYTHONUNBUFFERED": "1",
            },
            "estimated_minutes": 360,  # 6 hours generous estimate
            "max_retries": 1,
            "priority": 80,
            "cancel_grace_seconds": 30,  # enough time for checkpoint save
            "tags": ["luxia-base", "proxy-sweep"],
        }

        # Chain: each job depends on the previous
        if prev_job_id is not None:
            job_spec["depends_on"] = [prev_job_id]

        payload = {
            "spec": job_spec,
            "submitted_by": "luxia",
        }

        print(f"\n{'='*60}")
        print(f"Run {i+1}/5: {run['name']} — {run['description']}")
        print(f"Command: {command[:120]}...")
        if prev_job_id:
            print(f"Depends on: {prev_job_id}")

        if dry_run:
            print("[DRY RUN] Would submit to Heimdall")
            fake_id = f"dry-run-{i}"
            submitted.append({"name": run["name"], "job_id": fake_id})
            prev_job_id = fake_id
            continue

        try:
            resp = client.post("/api/v1/jobs", json=payload)
            result = resp.json()
            job_id = result.get("job", {}).get("id", "unknown")
            print(f"SUBMITTED: job_id={job_id}")
            submitted.append({"name": run["name"], "job_id": job_id})
            prev_job_id = job_id
        except Exception as e:
            print(f"ERROR submitting {run['name']}: {e}")
            print("Stopping sweep submission — fix the error and retry")
            break

    print(f"\n{'='*60}")
    print(f"Submitted {len(submitted)}/5 jobs")
    if submitted:
        print("\nJob IDs:")
        for s in submitted:
            print(f"  {s['name']}: {s['job_id']}")
        print(f"\nMonitor: curl {HEIMDALL_URL}/api/v1/jobs/{{job_id}}")
        print(f"Logs: ssh node1 'tail -f /tmp/heimdall_{{job_id}}.log'")
        print(f"Wandb: https://wandb.ai/g-stratiy-personal-/luxia-base")

    return submitted


def check_status() -> None:
    """Check status of any running luxia-base jobs."""
    if httpx is not None:
        client = httpx.Client(base_url=HEIMDALL_URL, timeout=30.0)
    else:
        client = _FallbackClient(HEIMDALL_URL)

    try:
        resp = client.get("/api/v1/jobs")
        jobs = resp.json()

        # Filter for our jobs
        our_jobs = [
            j for j in jobs.get("jobs", [])
            if "luxia" in j.get("spec", {}).get("name", "").lower()
        ]

        if not our_jobs:
            print("No luxia-base jobs found in Heimdall")
            return

        print(f"{'Name':<35} {'Status':<12} {'ID':<15}")
        print("-" * 62)
        for j in our_jobs:
            name = j.get("spec", {}).get("name", "?")
            status = j.get("status", "?")
            jid = j.get("id", "?")
            print(f"{name:<35} {status:<12} {jid:<15}")

    except Exception as e:
        print(f"Error checking status: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Submit proxy sweep to Heimdall")
    p.add_argument("--dry_run", action="store_true", help="Preview without submitting")
    p.add_argument("--status", action="store_true", help="Check sweep job status")
    p.add_argument("--data_path", type=str, default=DATA_PATH)
    args = p.parse_args()

    if args.status:
        check_status()
        return

    submit_sweep(dry_run=args.dry_run, data_path=args.data_path)


if __name__ == "__main__":
    main()
