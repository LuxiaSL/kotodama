#!/usr/bin/env python3
"""Per-tensor weight statistics across checkpoints, for fp8/gauge-oscillation analysis.

For each checkpoint: decompress (zstd CLI), torch.load on CPU, walk state['model'],
compute absmax / rms / crest / tail kurtosis per tensor. One JSON per checkpoint.

Run: nice -n 10 python3 weight_stats.py
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

CKPT_DIR = Path("/models/kotodama-data/checkpoints/3b-language")
WORK = Path("/models/kotodama-data/wave-analysis")
TMP = WORK / "_tmp_decompress.pt"

CHECKPOINTS = [
    (32174,  CKPT_DIR / "step_00032174.pt.bak"),
    (104600, CKPT_DIR / "step_00104600.pt.zst.bak"),
    (110000, CKPT_DIR / "step_00110000.pt.zst.bak"),
    (120000, CKPT_DIR / "step_00120000.pt.zst.bak"),
    (142000, CKPT_DIR / "step_00142000.pt.zst.bak"),
    (155000, CKPT_DIR / "step_00155000.pt.zst.bak"),
    (157600, WORK / "step_00157600.pt.zst"),
    (159600, WORK / "step_00159600.pt.zst"),
]


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float()
    n = x.numel()
    flat = x.reshape(-1)
    absmax = flat.abs().max().item()
    rms = flat.pow(2).mean().sqrt().item()
    # tail kurtosis on a subsample (cap 2M elements)
    stride = max(1, n // 2_000_000)
    sub = flat[::stride]
    mu = sub.mean()
    sd = sub.std()
    kurt = ((sub - mu) / (sd + 1e-12)).pow(4).mean().item() if sd > 0 else 0.0
    return {
        "numel": n,
        "absmax": absmax,
        "rms": rms,
        "crest": absmax / (rms + 1e-12),
        "kurtosis": kurt,
    }


def process(step: int, path: Path) -> None:
    out_path = WORK / f"stats_{step:08d}.json"
    if out_path.exists():
        print(f"[{step}] already done, skipping", flush=True)
        return
    t0 = time.time()
    load_path = path

    if path.suffix == ".zst" or path.name.endswith(".pt.zst.bak"):
        print(f"[{step}] decompressing {path.name}...", flush=True)
        if TMP.exists():
            TMP.unlink()
        rc = subprocess.run(
            ["zstd", "-dc", "-T4", str(path), "-o", str(TMP)],
            capture_output=True,
        )
        if rc.returncode != 0:
            print(f"[{step}] DECOMPRESS FAILED: {rc.stderr.decode()[:300]}", flush=True)
            return
        load_path = TMP

    print(f"[{step}] loading...", flush=True)
    try:
        state = torch.load(load_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[{step}] LOAD FAILED: {e}", flush=True)
        if load_path == TMP and TMP.exists():
            TMP.unlink()
        return

    model = state.get("model", state)
    stats: dict[str, dict] = {}
    for name, tensor in model.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        try:
            stats[name] = tensor_stats(tensor)
        except Exception as e:
            stats[name] = {"error": str(e)}

    meta = {
        "step": step,
        "ckpt_step_field": state.get("step"),
        "n_tensors": len(stats),
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "tensors": stats}, f)

    del state, model
    gc.collect()
    if load_path == TMP and TMP.exists():
        TMP.unlink()
    print(f"[{step}] done in {time.time()-t0:.0f}s -> {out_path.name}", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    for step, path in CHECKPOINTS:
        if not path.exists():
            print(f"[{step}] MISSING: {path}", flush=True)
            continue
        process(step, path)
    print("ALL DONE", flush=True)
