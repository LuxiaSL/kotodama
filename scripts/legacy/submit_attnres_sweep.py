#!/usr/bin/env python3
"""
Submit the AttnRes proxy validation sweep to Scheduler.

3 chained jobs: P1-AttnRes (AdamW), P3-AttnRes (Muon 0.02), NCA-AttnRes (NCA + Muon 0.02).
All use Block AttnRes (N=7) on the 108M proxy architecture.

Usage::

    python scripts/submit_attnres_sweep.py
    python scripts/submit_attnres_sweep.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

SCHEDULER_URL = os.environ.get("KOTODAMA_SCHEDULER_URL", "")
SCHEDULER_NODE = os.environ.get("KOTODAMA_SCHEDULER_NODE", "")
DATA_PATH = "data/fineweb_edu_6b.bin"
WORKING_DIR = os.environ.get("KOTODAMA_WORKDIR", ".")
VENV_TORCHRUN = ".venv/bin/python -m torch.distributed.run"

# NCA phase checkpoint for the NCA+AttnRes run
NCA_PHASE_CKPT = "checkpoints/nca_proxy/nca-phase-muon-002/step_00002287.pt"

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

# Common training args (matching proxy sweep exactly, plus --attn_res)
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
    "--attn_res "
    "--attn_res_n_blocks 7 "
)

# The 3 AttnRes runs
SWEEP_RUNS = [
    {
        "name": "p1-adamw-attnres",
        "extra_args": "--adamw_only --adamw_lr 8e-4",
        "description": "AdamW baseline + Block AttnRes",
    },
    {
        "name": "p3-muon-002-attnres",
        "extra_args": "--muon_lr 0.02 --adamw_lr 6e-4",
        "description": "Muon LR 0.02 + Block AttnRes",
    },
    {
        # NCA+AttnRes: load NCA phase checkpoint, embed-only reinit (matching prior runs)
        # The NCA checkpoint was trained WITHOUT AttnRes — missing AttnRes params
        # get zero-init (uniform weighting), which is correct per the paper.
        "name": "nca-002-attnres",
        "extra_args": (
            f"--muon_lr 0.02 --adamw_lr 6e-4 "
            f"--resume_nca {NCA_PHASE_CKPT}"
        ),
        "description": "NCA + Muon LR 0.02 + Block AttnRes (embed-only reinit)",
    },
]


def build_command(run: dict, data_path: str = DATA_PATH) -> str:
    """Build the full torchrun command for an AttnRes run."""
    args = COMMON_ARGS.format(data_path=data_path)
    checkpoint_dir = f"checkpoints/attnres_sweep/{run['name']}"
    wandb_name = f"attnres-{run['name']}"

    return (
        f"{VENV_TORCHRUN} --nproc_per_node=8 "
        f"-m src.training.train "
        f"{args} "
        f"--checkpoint_dir {checkpoint_dir} "
        f"--wandb_run_name {wandb_name} "
        f"{run['extra_args']}"
    )


def submit_sweep(dry_run: bool = False) -> list[dict[str, str]]:
    """Submit all AttnRes runs as chained Scheduler jobs."""
    if not SCHEDULER_URL or not SCHEDULER_NODE:
        raise SystemExit("Set KOTODAMA_SCHEDULER_URL and KOTODAMA_SCHEDULER_NODE before submitting.")
    submitted: list[dict[str, str]] = []
    prev_job_id: str | None = None

    # Verify Scheduler is reachable
    try:
        req = urllib.request.Request(f"{SCHEDULER_URL}/api/v1/status")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                print(f"ERROR: Scheduler returned {resp.status}")
                sys.exit(1)
        print(f"Scheduler OK at {SCHEDULER_URL}")
    except Exception as e:
        print(f"ERROR: Cannot reach Scheduler: {e}")
        sys.exit(1)

    for i, run in enumerate(SWEEP_RUNS):
        command = build_command(run)

        job_spec = {
            "job_type": "custom",
            "name": f"luxia-attnres-{run['name']}",
            "command": command,
            "node": SCHEDULER_NODE,
            "gpus": 8,
            "working_dir": WORKING_DIR,
            "env": {"PYTHONUNBUFFERED": "1", **({"WANDB_API_KEY": WANDB_API_KEY} if WANDB_API_KEY else {})},
            "estimated_minutes": 120,  # ~90 min each, 120 for safety
            "max_retries": 1,
            "priority": 80,
            "cancel_grace_seconds": 30,
            "tags": ["luxia-base", "attnres-sweep"],
        }

        if prev_job_id is not None:
            job_spec["depends_on"] = [prev_job_id]

        payload = {"spec": job_spec, "submitted_by": "luxia"}

        print(f"\n{'='*60}")
        print(f"Run {i+1}/3: {run['name']} — {run['description']}")
        print(f"Command: {command[:150]}...")
        if prev_job_id:
            print(f"Depends on: {prev_job_id}")

        if dry_run:
            print("[DRY RUN] Would submit to Scheduler")
            fake_id = f"dry-run-{i}"
            submitted.append({"name": run["name"], "job_id": fake_id})
            prev_job_id = fake_id
            continue

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{SCHEDULER_URL}/api/v1/jobs",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            job_id = result.get("job", {}).get("id", "unknown")
            print(f"SUBMITTED: job_id={job_id}")
            submitted.append({"name": run["name"], "job_id": job_id})
            prev_job_id = job_id
        except Exception as e:
            print(f"ERROR submitting {run['name']}: {e}")
            print("Stopping — fix the error and retry")
            break

    print(f"\n{'='*60}")
    print(f"Submitted {len(submitted)}/3 jobs")
    if submitted:
        print("\nJob IDs:")
        for s in submitted:
            print(f"  {s['name']}: {s['job_id']}")
        print(f"\nMonitor: curl {SCHEDULER_URL}/api/v1/jobs/{{job_id}}")
        print(f"Wandb: https://wandb.ai/luxia-anima-labs/luxia-base")

    return submitted


def main() -> None:
    p = argparse.ArgumentParser(description="Submit AttnRes sweep to Scheduler")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    submit_sweep(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
