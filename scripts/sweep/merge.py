#!/usr/bin/env python3
"""Assemble a sweep's shards (scripts/sweep/steer_sweep.py) into per-layer runs, the layout the neuron explorer
(https://github.com/LuxiaSL/neuron-explorer, branch kotodama) and its sweep ranking read: one directory of runs per
span, so the two spans of a layer never shadow each other.

  <sweep>/runs/<mode>/tsumugi-<mode>-L{L}.jsonl       rollouts, shard 0 (baselines) first; ``row`` = line index
  <sweep>/runs/<mode>/tsumugi-<mode>-L{L}.meta.json   layer (effective), span, design, dose unit + norms sha, prompts
  <sweep>/topk/<mode>/tsumugi-<mode>-L{L}.topk.npz    top_ids int32 / top_lps float16 [rows, max_tokens, k], -1 pad
  <sweep>/captures/<mode>/tsumugi-<mode>-L{L}.mlp_in.npy (+ .json)
      uniform: float16 [rows, capture layers, hidden], the last prompt token of each rollout;
      decode: float16 [prompts, capture layers, hidden] (``per_prompt``: every rollout of a prompt shares it), and
      a second array beside it, suffix ``mlp_in_first``, [rows, first-token layers, hidden]: the state the first
      generated token fed in.

    tools/run_py.sh scripts/sweep/merge.py --sweep <sweep dir> --mode decode
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def merge_layer(shard_dir: Path, sweep: Path, mode: str, layer: int, meta: dict) -> int:
    lines: list[dict] = []
    tops_i, tops_l, caps, firsts = [], [], [], []
    for jsonl in sorted(shard_dir.glob(f"L{layer}.s*.jsonl"), key=lambda p: int(re.search(r"\.s(\d+)\.", p.name)[1])):
        shard = int(re.search(r"\.s(\d+)\.", jsonl.name)[1])
        parts: dict[int, dict] = {}
        for line in jsonl.read_text().splitlines():
            rec = json.loads(line)
            k = rec.pop("part")
            i = rec.pop("part_row")
            if k not in parts:
                parts[k] = dict(np.load(shard_dir / f"L{layer}.s{shard}.p{k}.npz"))
            p = parts[k]
            tops_i.append(p["top_ids"][i])
            tops_l.append(p["top_lps"][i])
            if "mlp_in" in p:
                caps.append(p["mlp_in"][i])
            if "mlp_in_first" in p:
                firsts.append(p["mlp_in_first"][i])
            rec["row"] = len(lines)
            lines.append(rec)
    if not lines:
        return 0
    name = f"tsumugi-{mode}-L{layer}"
    runs = sweep / "runs" / mode
    runs.mkdir(parents=True, exist_ok=True)
    with open(runs / f"{name}.jsonl", "w") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    (runs / f"{name}.meta.json").write_text(json.dumps({**meta, "layer": layer,
                                                        "module_layer": lines[0]["module_layer"]}, indent=1))
    topk = sweep / "topk" / mode
    topk.mkdir(parents=True, exist_ok=True)
    np.savez(topk / f"{name}.topk.npz", top_ids=np.stack(tops_i), top_lps=np.stack(tops_l))
    capd = sweep / "captures" / mode
    capd.mkdir(parents=True, exist_ok=True)
    offsets = meta["capture_offsets"]
    n_layers = meta.get("n_layers", 10 ** 9)
    cap_meta = {"hook": "mlp_in", "layers": [layer + o for o in offsets if 0 <= layer + o < n_layers],
                "rows": len(lines), "dtype": "float16"}
    if mode == "uniform" and caps:
        np.save(capd / f"{name}.mlp_in.npy", np.stack(caps))
        cap_meta.update({"positions": "last_prompt", "per_prompt": False})
    elif mode == "decode":
        pc = np.load(shard_dir / f"L{layer}.prompt_caps.npz")
        np.save(capd / f"{name}.mlp_in.npy", pc["mlp_in"])
        cap_meta.update({"positions": "last_prompt", "per_prompt": True, "prompts": pc["prompts"].tolist(),
                         "layers": pc["layers"].tolist()})
        if firsts:
            np.save(capd / f"{name}.mlp_in_first.npy", np.stack(firsts))
            cap_meta["first_token"] = {"layers": [layer + o for o in meta["first_token_offsets"]
                                                  if 0 <= layer + o < n_layers], "positions": "first_generated"}
    (capd / f"{name}.mlp_in.json").write_text(json.dumps(cap_meta, indent=1))
    return len(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--mode", choices=["decode", "uniform"], required=True)
    a = ap.parse_args()
    sweep = Path(a.sweep)
    shard_dir = sweep / "shards" / a.mode
    metas = sorted(shard_dir.glob("meta.s*.json"))
    if not metas:
        raise SystemExit(f"no shard metadata under {shard_dir}")
    meta = json.loads(metas[0].read_text())
    for m in metas[1:]:  # every shard must have run the same design
        other = json.loads(m.read_text())
        for k in ("dose_norms_sha256", "doses", "prompt_ids", "checkpoint", "span", "max_tokens", "seed"):
            if other.get(k) != meta.get(k):
                raise SystemExit(f"{m.name} disagrees with {metas[0].name} on {k}")
    meta.pop("rows", None)
    layers = sorted({int(re.match(r"L(\d+)\.", p.name)[1]) for p in shard_dir.glob("L*.s*.jsonl")})
    total = 0
    for L in layers:
        n = merge_layer(shard_dir, sweep, a.mode, L, meta)
        print(f"L{L}: {n} rollouts")
        total += n
    print(f"{len(layers)} layers, {total} rollouts -> {sweep / 'runs' / a.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
