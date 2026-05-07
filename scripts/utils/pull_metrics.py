#!/usr/bin/env python3
"""
Pull all metrics from a training run — wandb API + remote JSONL.

Usage:
    # Pull from wandb only
    python scripts/utils/pull_metrics.py --wandb aethera/kotodama-ddv1-openwebtext/run_id

    # Pull JSONL from remote node
    python scripts/utils/pull_metrics.py --scp node2:~/luxi-files/kotodama/checkpoints/openwebtext-ddv1/

    # Both
    python scripts/utils/pull_metrics.py \
        --wandb aethera/kotodama-ddv1-openwebtext/run_id \
        --scp node2:~/luxi-files/kotodama/checkpoints/openwebtext-ddv1/ \
        -o data/pulled/openwebtext-ddv1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def pull_wandb(run_path: str, output_dir: Path) -> Optional[Path]:
    """Download all metrics from a wandb run via the API."""
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb not installed. Install with: uv pip install wandb", file=sys.stderr)
        return None

    api = wandb.Api()
    try:
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        print(f"ERROR: Could not fetch run '{run_path}': {e}", file=sys.stderr)
        return None

    print(f"Pulling metrics from wandb run: {run.name} ({run.id})")
    print(f"  State: {run.state}, Steps: {run.lastHistoryStep}")

    output_dir.mkdir(parents=True, exist_ok=True)

    history = run.scan_history()
    rows = list(history)
    print(f"  Downloaded {len(rows)} metric rows")

    if not rows:
        print("  WARNING: No metrics found in run", file=sys.stderr)
        return None

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    print(f"  Unique metric keys: {len(all_keys)}")

    out_path = output_dir / "wandb_metrics.jsonl"
    with open(out_path, "w") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if v is not None}
            f.write(json.dumps(clean) + "\n")

    print(f"  Saved to: {out_path}")

    config_path = output_dir / "wandb_config.json"
    with open(config_path, "w") as f:
        json.dump(dict(run.config), f, indent=2)
    print(f"  Config saved to: {config_path}")

    summary_path = output_dir / "wandb_summary.json"
    with open(summary_path, "w") as f:
        json.dump(dict(run.summary._json_dict), f, indent=2, default=str)
    print(f"  Summary saved to: {summary_path}")

    key_path = output_dir / "wandb_keys.txt"
    with open(key_path, "w") as f:
        for k in sorted(all_keys):
            f.write(k + "\n")
    print(f"  Key manifest saved to: {key_path}")

    return out_path


def pull_scp(remote_path: str, output_dir: Path) -> list[Path]:
    """SCP metrics JSONL files from a remote node."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []

    for filename in ["metrics.jsonl", "geo_metrics.jsonl"]:
        remote = f"{remote_path.rstrip('/')}/{filename}"
        local = output_dir / filename
        print(f"Pulling {remote} -> {local}")
        result = subprocess.run(
            ["scp", remote, str(local)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines = sum(1 for _ in open(local))
            print(f"  OK: {lines} lines")
            pulled.append(local)
        else:
            print(f"  SKIP: {result.stderr.strip()}")

    return pulled


def summarize(output_dir: Path) -> None:
    """Print a summary of what was pulled."""
    print(f"\n{'='*60}")
    print(f"Metrics pulled to: {output_dir}")
    print(f"{'='*60}")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f}MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f}KB"
        else:
            size_str = f"{size}B"
        print(f"  {f.name:30s} {size_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull training metrics from wandb and/or remote nodes")
    parser.add_argument(
        "--wandb", type=str, default=None,
        help="Wandb run path: entity/project/run_id",
    )
    parser.add_argument(
        "--scp", type=str, default=None,
        help="Remote checkpoint dir: node2:~/path/to/checkpoint_dir/",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Local output directory (default: data/pulled/<run_name>)",
    )
    parser.add_argument(
        "--list-runs", type=str, default=None, metavar="ENTITY/PROJECT",
        help="List recent runs in a wandb project and exit",
    )
    args = parser.parse_args()

    if args.list_runs:
        try:
            import wandb
            api = wandb.Api()
            runs = api.runs(args.list_runs, order="-created_at", per_page=20)
            print(f"Recent runs in {args.list_runs}:")
            for run in runs:
                print(f"  {run.id:12s}  {run.state:10s}  {run.name}  (step {run.lastHistoryStep})")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
        return

    if not args.wandb and not args.scp:
        parser.error("At least one of --wandb or --scp is required")

    output_dir = Path(args.output) if args.output else Path("data/pulled/run")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        pull_wandb(args.wandb, output_dir)

    if args.scp:
        pull_scp(args.scp, output_dir)

    summarize(output_dir)


if __name__ == "__main__":
    main()
