#!/usr/bin/env python3
"""Cross-checkpoint analysis of per-tensor weight stats vs the loss-wave phase.

Hypothesis: gauge (weight-magnitude) oscillation -> fp8 quantization-error waves.
Prediction: some tensors' absmax/rms/crest deviate non-monotonically at step
142000 (dip1) and partially 155000 (recovery), relative to the smooth trend
through reference points [104600, 110000, 120000, 157600, 159600].
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

STATS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
STEPS_REF = [104600, 110000, 120000, 157600, 159600]   # on-trend
STEPS_TEST = [142000, 155000]                           # in-wave
ALL = sorted(STEPS_REF + STEPS_TEST)
# EMA-residual wave phase at each step (from train-loss analysis)
WAVE = {104600: +0.001, 110000: +0.005, 120000: +0.006,
        142000: -0.043, 155000: -0.012, 157600: -0.004, 159600: -0.002}

data = {}
for s in ALL + [32174]:
    p = STATS_DIR / f"stats_{s:08d}.json"
    if p.exists():
        data[s] = json.load(open(p))["tensors"]
missing = [s for s in ALL if s not in data]
if missing:
    print(f"missing: {missing}")
avail_ref = [s for s in STEPS_REF if s in data]
avail_test = [s for s in STEPS_TEST if s in data]
if len(avail_ref) < 4 or not avail_test:
    sys.exit("not enough checkpoints yet")

names = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f"{len(data)} checkpoints, {len(names)} common tensors\n")


def linfit(xs, ys):
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    den = sum((x - xm) ** 2 for x in xs)
    b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den if den else 0.0
    a = ym - b * xm
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    return a, b, rms


def tensor_type(name):
    if "embed" in name: return "embed"
    for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if t in name: return t
    if "attn_res" in name or "res_query" in name or "res_norm" in name: return "attnres_route"
    if "norm" in name: return "norm_gain"
    return "other"


def layer_of(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1


# Per-tensor z-scores at test steps, per stat
results = []
for name in names:
    row = {"name": name, "type": tensor_type(name), "layer": layer_of(name)}
    for stat in ["absmax", "rms", "crest", "kurtosis"]:
        try:
            ys_ref = [data[s][name][stat] for s in avail_ref]
        except (KeyError, TypeError):
            continue
        a, b, rms_r = linfit(avail_ref, ys_ref)
        floor = max(rms_r, 1e-9, 0.001 * abs(a + b * 130000))  # noise floor: fit resid or 0.1% of level
        for s in avail_test:
            v = data[s][name].get(stat)
            if v is None: continue
            pred = a + b * s
            row[f"{stat}_z{s}"] = (v - pred) / floor
            row[f"{stat}_rel{s}"] = (v - pred) / (abs(pred) + 1e-12)
    results.append(row)

# Rank by |z| at 142000 for each stat
for stat in ["rms", "absmax", "crest", "kurtosis"]:
    key = f"{stat}_z142000"
    ranked = sorted((r for r in results if key in r), key=lambda r: -abs(r[key]))
    print(f"=== top tensors by |z({stat})| at 142000 (dip1) ===")
    print(f"{'tensor':<48} {'z142k':>8} {'rel142k':>9} {'z155k':>8} {'rel155k':>9}")
    for r in ranked[:14]:
        print(f"{r['name']:<48} {r[key]:>8.1f} {r.get(f'{stat}_rel142000', 0):>9.4f} "
              f"{r.get(f'{stat}_z155000', float('nan')):>8.1f} {r.get(f'{stat}_rel155000', 0):>9.4f}")
    print()

# Aggregate |z| by tensor type
print("=== mean |z(142000)| by tensor type ===")
print(f"{'type':<16}" + "".join(f"{s:>10}" for s in ["rms", "absmax", "crest", "kurt"]) + f"{'n':>5}")
by_type = defaultdict(list)
for r in results:
    by_type[r["type"]].append(r)
for t, rows in sorted(by_type.items()):
    cells = []
    for stat in ["rms", "absmax", "crest", "kurtosis"]:
        zs = [abs(r[f"{stat}_z142000"]) for r in rows if f"{stat}_z142000" in r]
        cells.append(sum(zs) / len(zs) if zs else float("nan"))
    print(f"{t:<16}" + "".join(f"{c:>10.2f}" for c in cells) + f"{len(rows):>5}")

# Aggregate |z(142k)| of rms by layer (gauge oscillation should localize)
print("\n=== mean |z_rms(142000)| by layer ===")
by_layer = defaultdict(list)
for r in results:
    if r["layer"] >= 0 and "rms_z142000" in r:
        by_layer[r["layer"]].append(abs(r["rms_z142000"]))
for li in sorted(by_layer):
    zs = by_layer[li]
    bar = "#" * int(min(sum(zs) / len(zs), 40))
    print(f"  L{li:>2}: {sum(zs)/len(zs):6.2f} {bar}")

# Global aggregates per checkpoint: do means wave with loss phase?
print("\n=== global aggregates per checkpoint (fp8-relevant Linears) ===")
lin_types = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
print(f"{'step':>8} {'wave':>7} {'mean_rms':>10} {'mean_crest':>11} {'mean_kurt':>10} {'mean_absmax':>12}")
for s in sorted(data):
    if s == 32174 and 32174 not in WAVE:
        wave_s = float("nan")
    else:
        wave_s = WAVE.get(s, float("nan"))
    vals = defaultdict(list)
    for name in names:
        if tensor_type(name) in lin_types:
            d = data[s].get(name)
            if d:
                for stat in ["rms", "crest", "kurtosis", "absmax"]:
                    vals[stat].append(d[stat])
    print(f"{s:>8} {wave_s:>7.3f} {sum(vals['rms'])/len(vals['rms']):>10.6f} "
          f"{sum(vals['crest'])/len(vals['crest']):>11.3f} "
          f"{sum(vals['kurtosis'])/len(vals['kurtosis']):>10.3f} "
          f"{sum(vals['absmax'])/len(vals['absmax']):>12.5f}")

# Noise floor reference: relative change 157600 -> 159600 (2k calm steps)
if 157600 in data and 159600 in data:
    rels = []
    for name in names:
        a, b = data[157600].get(name), data[159600].get(name)
        if a and b and a.get("rms"):
            rels.append(abs(b["rms"] - a["rms"]) / (abs(a["rms"]) + 1e-12))
    rels.sort()
    print(f"\ncalm-pair (157.6k->159.6k) rms drift: median={rels[len(rels)//2]:.5f} "
          f"p90={rels[int(0.9*len(rels))]:.5f} max={rels[-1]:.5f}")
