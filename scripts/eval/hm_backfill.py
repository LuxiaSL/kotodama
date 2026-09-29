#!/usr/bin/env python3
"""Post per-checkpoint eval results into heimdall's Telemetry workspace.

The formation-viewing glue (SPEC-1B-FORMATION-LADDER §4 gates × heimdall's
Telemetry pane): checkpoint sweeps (nca_gates_sweep, induction, batteries)
produce per-step JSON; this posts them as metrics so formation curves sit
next to loss on the tokens axis, with slices/Δ-heatmaps/comets over them.

Two targets:
- ATTACH: ``--job <training-job-id>`` — gates land at the training run's own
  steps, alongside its live telemetry (the ladder-arm case).
- ARCHAEOLOGY: post-hoc trajectories with no live run (the archived 3B, old
  proxies) get a zero-GPU container job first::

      heimdall submit --gpus 0 --name 3b-formation-archaeology -- true

  then ``--job <that id>``. ``tokens_seen`` is posted with every point, so
  the workspace's tokens axis works even for synthetic runs.

Inputs (repeatable, mixable):
- ``--gates summary.json``   nca_gates_sweep output (rows carry step+tokens).
- ``--uncheatable r.json``   kotodama.eval.uncheatable output (one per
                             checkpoint) -> ``uncheatable/{domain}`` +
                             ``uncheatable/mean`` (BPB) at the ckpt's step.
- ``--points rows.jsonl``    escape hatch: {"step": N, "tokens": N?,
                             "metrics": {...}} per line — any sweep adapts.
- ``--config k=v``           run hyperparameters (arm, shape, seed) for the
                             runs-table columns / group-by / scatter.

Usage::

  python scripts/eval/hm_backfill.py --job e94838f335aa \\
      --gates gates/summary.json --prefix gates \\
      --config arm=p05-armd-v3 --config generator=v3 \\
      --url http://<coordinator>:7000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

PostFn = Callable[[str, dict[str, Any]], Any]  # (path, json_body) -> response


# ── Pure converters ───────────────────────────────────────────────────────────

def gates_rows_to_points(summary: dict[str, Any],
                         prefix: str = "gates") -> list[dict[str, Any]]:
    """nca_gates_sweep summary.json -> metric points.

    Every numeric field of each row is posted under ``<prefix>/``; ``step``
    indexes the point and ``tokens`` doubles as the ``tokens_seen`` counter
    (the workspace's tokens axis). ``is_init`` rows keep step 0.
    """
    points = []
    for row in summary.get("rows", []):
        step = int(row["step"])
        metrics: dict[str, float] = {}
        for k, v in row.items():
            if k in ("step", "is_init") or v is None:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            name = "tokens_seen" if k == "tokens" else f"{prefix}/{k}"
            metrics[name] = float(v)
        if "chance_acc" in summary:
            metrics[f"{prefix}/chance_acc"] = float(summary["chance_acc"])
        if metrics:
            points.append({"step": step, "metrics": metrics})
    return points


def uncheatable_to_points(result: dict[str, Any], prefix: str = "uncheatable",
                          step: int | None = None) -> list[dict[str, Any]]:
    """kotodama.eval.uncheatable result JSON -> one metric point.

    ``<prefix>/<domain>`` = that domain's BPB, ``<prefix>/mean`` = the mean
    over domains; the result's ``tokens`` becomes ``tokens_seen``. ``step``
    overrides the result's own step (which comes from the checkpoint).
    """
    if not str(result.get("schema", "")).startswith("kotodama.uncheatable/"):
        raise ValueError("not a kotodama.eval.uncheatable result "
                         f"(schema={result.get('schema')!r})")
    step = result.get("step") if step is None else step
    if step is None:
        raise ValueError(f"{result.get('checkpoint')}: result has no step; "
                         "pass --uncheatable-step or rerun with --step")
    metrics = {f"{prefix}/{name}": float(d["bpb"])
               for name, d in result["domains"].items()}
    metrics[f"{prefix}/mean"] = float(result["mean_bpb"])
    if result.get("tokens") is not None:
        metrics["tokens_seen"] = float(result["tokens"])
    return [{"step": int(step), "metrics": metrics}]


def jsonl_to_points(path: Path) -> list[dict[str, Any]]:
    """The escape hatch: rows of {"step", "tokens"?, "metrics"} per line."""
    points = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = {k: float(v) for k, v in row["metrics"].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if row.get("tokens") is not None:
            metrics["tokens_seen"] = float(row["tokens"])
        points.append({"step": int(row["step"]), "metrics": metrics})
    return points


def merge_points(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge by step (metrics dicts union; later lists win on key clashes)."""
    by_step: dict[int, dict[str, Any]] = {}
    for lst in lists:
        for p in lst:
            slot = by_step.setdefault(int(p["step"]), {"step": int(p["step"]),
                                                       "metrics": {}})
            slot["metrics"].update(p["metrics"])
    return [by_step[s] for s in sorted(by_step)]


def parse_config_args(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        if not k or not _:
            raise ValueError(f"--config wants k=v, got {pair!r}")
        try:
            out[k] = json.loads(v)  # numbers/bools/json pass through typed
        except (json.JSONDecodeError, ValueError):
            out[k] = v
    return out


# ── Posting ───────────────────────────────────────────────────────────────────

def post_all(post: PostFn, job_id: str, points: list[dict[str, Any]],
             config: dict[str, Any] | None = None,
             chunk: int = 500) -> int:
    """POST config (once) + points in batches. Returns points posted."""
    if config:
        post(f"/api/v1/jobs/{job_id}/config", {"config": config})
    n = 0
    for i in range(0, len(points), chunk):
        part = points[i:i + chunk]
        post(f"/api/v1/jobs/{job_id}/metrics/batch", {"points": part})
        n += len(part)
    return n


def http_poster(base_url: str) -> PostFn:
    import httpx
    client = httpx.Client(base_url=base_url, timeout=15)

    def post(path: str, body: dict[str, Any]) -> Any:
        r = client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    return post


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--job", required=True,
                    help="heimdall job id to attach metrics to")
    ap.add_argument("--gates", action="append", default=[],
                    help="nca_gates_sweep summary.json (repeatable)")
    ap.add_argument("--points", action="append", default=[],
                    help="jsonl of {step, tokens?, metrics} rows (repeatable)")
    ap.add_argument("--prefix", default="gates",
                    help="metric prefix for --gates rows")
    ap.add_argument("--uncheatable", action="append", default=[],
                    help="kotodama.eval.uncheatable result JSON, one per "
                         "checkpoint (repeatable) -> uncheatable/{domain,mean}")
    ap.add_argument("--uncheatable-step", type=int, default=None,
                    help="step for a single --uncheatable result that has none")
    ap.add_argument("--config", action="append", default=[],
                    help="k=v run hyperparameter (repeatable)")
    ap.add_argument("--url", default=os.environ.get(
        "KOTODAMA_SCHEDULER_URL",
        os.environ.get("HEIMDALL_URL", "http://127.0.0.1:7000")))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the points, post nothing")
    args = ap.parse_args()

    lists = []
    for g in args.gates:
        lists.append(gates_rows_to_points(json.loads(Path(g).read_text()),
                                          prefix=args.prefix))
    for p in args.points:
        lists.append(jsonl_to_points(Path(p)))
    if args.uncheatable_step is not None and len(args.uncheatable) != 1:
        raise SystemExit("--uncheatable-step applies to exactly one --uncheatable")
    for u in args.uncheatable:
        lists.append(uncheatable_to_points(json.loads(Path(u).read_text()),
                                           step=args.uncheatable_step))
    points = merge_points(*lists)
    if not points:
        raise SystemExit("nothing to post (no --gates / --points / "
                         "--uncheatable rows)")
    config = parse_config_args(args.config)

    if args.dry_run:
        print(json.dumps({"job": args.job, "config": config,
                          "n_points": len(points), "points": points[:3]},
                         indent=1))
        return
    n = post_all(http_poster(args.url), args.job, points, config)
    keys = sorted({k for p in points for k in p["metrics"]})
    print(f"posted {n} points ({len(keys)} keys: {', '.join(keys[:8])}"
          f"{'…' if len(keys) > 8 else ''}) to job {args.job} at {args.url}")


if __name__ == "__main__":
    main()
