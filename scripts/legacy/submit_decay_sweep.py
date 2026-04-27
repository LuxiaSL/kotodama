"""
Submit decay schedule sensitivity sweep to Heimdall.

Resumes P2, P3, P4 from their step_42000 checkpoints with modified
decay schedules. Tests whether longer decay changes the LR ordering.

Original: decay_start_pct=0.90 (decay starts ~step 41198, 4578 steps of decay)
Variants:
  - 0.85 → decay starts ~step 38910 (6866 steps, 50% more decay)
  - 0.80 → decay starts ~step 36621 (9155 steps, 2x more decay)
  - 0.75 → decay starts ~step 34332 (11444 steps, 2.5x more decay)

Since we resume from step 42000, only variants where decay has already
started (all of them) are meaningful. The model will continue from where
it left off but the LR schedule will be different.

Usage::

    python scripts/submit_decay_sweep.py
    python scripts/submit_decay_sweep.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

HEIMDALL_URL = "http://node1.datasci.ath:7000"
DATA_PATH = "data/fineweb_edu_6b.bin"
WORKING_DIR = "/home/athuser/projects/luxia-base"
VENV_PYTHON = ".venv/bin/torchrun"

# Runs to test (source checkpoint → decay variants)
SOURCE_RUNS = [
    {
        "name": "p2-muon-001",
        "ckpt": "checkpoints/proxy_sweep/p2-muon-001",
        "extra_args": "--muon_lr 0.01 --adamw_lr 6e-4",
    },
    {
        "name": "p3-muon-002",
        "ckpt": "checkpoints/proxy_sweep/p3-muon-002",
        "extra_args": "--muon_lr 0.02 --adamw_lr 6e-4",
    },
    {
        "name": "p4-muon-003",
        "ckpt": "checkpoints/proxy_sweep/p4-muon-003",
        "extra_args": "--muon_lr 0.03 --adamw_lr 6e-4",
    },
]

DECAY_VARIANTS = [0.85, 0.80, 0.75]

COMMON_ARGS = (
    "--data_path {data_path} "
    "--model_size proxy "
    "--sequence_length 2048 "
    "--micro_batch_size 4 "
    "--global_batch_tokens 131072 "
    "--warmup_steps 2000 "
    "--decay_type sqrt "
    "--gradient_clip 1.0 "
    "--total_tokens 6000000000 "
    "--log_every 100 "
    "--save_every 2000 "
    "--keep_checkpoints 2 "
    "--geo_monitor "
    "--geo_monitor_tier1_every 200 "
    "--geo_monitor_tier2_every 2000 "
    "--wandb "
    "--activation_checkpointing "
    "--compile "
)


def build_command(source: dict, decay_pct: float) -> str:
    """Build torchrun command for a decay variant."""
    args = COMMON_ARGS.format(data_path=DATA_PATH)
    decay_name = f"d{int(decay_pct*100)}"
    out_dir = f"checkpoints/decay_sweep/{source['name']}-{decay_name}"
    wandb_name = f"decay-{source['name']}-{decay_name}"

    return (
        f"{VENV_PYTHON} --nproc_per_node=8 "
        f"-m src.training.train "
        f"{args} "
        f"--decay_start_pct {decay_pct} "
        f"--checkpoint_dir {out_dir} "
        f"--wandb_run_name {wandb_name} "
        f"{source['extra_args']}"
    )


def submit(dry_run: bool = False) -> None:
    """Submit all decay sweep jobs."""
    jobs = []
    prev_job_id = None

    for source in SOURCE_RUNS:
        for decay_pct in DECAY_VARIANTS:
            decay_name = f"d{int(decay_pct*100)}"
            job_name = f"decay-{source['name']}-{decay_name}"
            command = build_command(source, decay_pct)

            # Copy the source checkpoint to the output dir so resume works
            out_dir = f"checkpoints/decay_sweep/{source['name']}-{decay_name}"
            setup_cmd = (
                f"mkdir -p {out_dir} && "
                f"cp {source['ckpt']}/step_00042000.pt {out_dir}/ && "
                f"cp {source['ckpt']}/metrics.jsonl {out_dir}/ 2>/dev/null; "
                f"{command}"
            )

            spec = {
                "job_type": "custom",
                "name": job_name,
                "command": f"bash -c '{setup_cmd}'",
                "node": "node1",
                "gpus": 8,
                "working_dir": WORKING_DIR,
                "env": {"PYTHONUNBUFFERED": "1"},
                "estimated_minutes": 60,
                "max_retries": 1,
                "priority": 70,
                "cancel_grace_seconds": 30,
                "tags": ["luxia-base", "decay-sweep"],
            }

            # Chain jobs sequentially
            if prev_job_id is not None:
                spec["depends_on"] = [prev_job_id]

            payload = {"spec": spec, "submitted_by": "luxia"}

            print(f"\n{'='*60}")
            print(f"{job_name}: {source['name']} with decay={decay_pct}")
            print(f"  Source ckpt: {source['ckpt']}/step_00042000.pt")
            print(f"  Output dir: {out_dir}")
            print(f"  Command: {command[:100]}...")

            if dry_run:
                print("  [DRY RUN]")
                prev_job_id = f"dry-{len(jobs)}"
                jobs.append({"name": job_name, "id": prev_job_id})
                continue

            try:
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{HEIMDALL_URL}/api/v1/jobs",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    job_id = result.get("job", {}).get("id", "unknown")
                    print(f"  SUBMITTED: {job_id}")
                    prev_job_id = job_id
                    jobs.append({"name": job_name, "id": job_id})
            except Exception as e:
                print(f"  ERROR: {e}")
                break

    print(f"\n{'='*60}")
    print(f"Submitted {len(jobs)} jobs ({len(SOURCE_RUNS)} runs × {len(DECAY_VARIANTS)} decay variants)")
    if jobs:
        for j in jobs:
            print(f"  {j['name']}: {j['id']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    submit(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
