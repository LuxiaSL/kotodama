#!/usr/bin/env python3
"""Reconstruct which train.bin sequences rank 0 consumed in loss-wave vs control
windows, and compare content statistics between windows.

The dataset iteration is fully deterministic:
  indices = np.arange(0, 11729297); RandomState(42).shuffle(indices)
  step s consumes indices[60*s : 60*(s+1)]  (micro_batch 10 x grad_accum 6, rank 0)
"""
import numpy as np

PATH = "/models/kotodama-data/train.bin"
SEQ_LEN = 4096
SEED = 42
WORLD = 8
SEQS_PER_STEP = 60  # rank 0: micro_batch 10 * grad_accum 6

data = np.memmap(PATH, dtype=np.uint16, mode="r")
total_tokens = len(data)
total_seqs = total_tokens // SEQ_LEN
seqs_per_rank = total_seqs // WORLD
print(f"total_tokens={total_tokens:,} total_seqs={total_seqs:,} seqs_per_rank={seqs_per_rank:,}")

# Rank 0 permutation (exactly as TokenizedDataset._shuffled_indices does)
rng = np.random.RandomState(SEED + 0)  # seed + epoch(0)
indices = np.arange(0, seqs_per_rank)
rng.shuffle(indices)
print("permutation built")

WINDOWS = {
    "control_A 135-138k": (135000, 138000),
    "dip1 142-145k":      (142000, 145000),
    "rise 149-152k":      (149000, 152000),
    "dip2 152.3-154k":    (152300, 154000),
    "control_B 156.5-159k": (156500, 159000),
}

SUBSAMPLE = 40  # analyze every 40th consumed sequence

print(f"\n{'window':<22} {'n':>5} {'docs/seq':>8} {'meandoclen':>10} {'uniq_frac':>9} "
      f"{'p_tok<1k':>8} {'p_tok>30k':>9} {'mean_tok':>8} {'filepos_q10':>11} {'filepos_q90':>11}")

results = {}
for name, (a, b) in WINDOWS.items():
    lo, hi = SEQS_PER_STEP * a, SEQS_PER_STEP * b
    consumed = indices[lo:hi:SUBSAMPLE]
    n = len(consumed)

    eos_counts = np.empty(n, dtype=np.int64)
    uniq_frac = np.empty(n)
    frac_low = np.empty(n)   # token id < 1000 (common/punct/special)
    frac_high = np.empty(n)  # token id > 30000 (rare tail)
    mean_tok = np.empty(n)

    for i, sidx in enumerate(consumed):
        seq = np.asarray(data[sidx * SEQ_LEN:(sidx + 1) * SEQ_LEN])
        eos_counts[i] = int((seq == 0).sum())
        uniq_frac[i] = len(np.unique(seq)) / SEQ_LEN
        frac_low[i] = float((seq < 1000).mean())
        frac_high[i] = float((seq > 30000).mean())
        mean_tok[i] = float(seq.mean())

    docs_per_seq = eos_counts.mean()
    mean_doclen = SEQ_LEN / max(docs_per_seq, 1e-9)
    fp = np.sort(consumed)
    results[name] = dict(eos=eos_counts, uniq=uniq_frac)
    print(f"{name:<22} {n:>5} {docs_per_seq:>8.3f} {mean_doclen:>10.0f} {uniq_frac.mean():>9.4f} "
          f"{frac_low.mean():>8.4f} {frac_high.mean():>9.4f} {mean_tok.mean():>8.0f} "
          f"{fp[int(0.1*n)]/seqs_per_rank:>11.3f} {fp[int(0.9*n)]/seqs_per_rank:>11.3f}")

# Distributional comparison: EOS count histogram per window
print("\nEOS-count (docs/seq) histogram, fraction of sequences:")
bins = [0, 1, 2, 3, 5, 9, 17, 33, 100000]
labels = ["0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]
print(f"{'window':<22}" + "".join(f"{l:>8}" for l in labels))
for name in WINDOWS:
    e = results[name]["eos"]
    h, _ = np.histogram(e, bins=bins)
    print(f"{name:<22}" + "".join(f"{v/len(e):>8.3f}" for v in h))

# uniq_frac (repetitiveness) percentiles
print("\nunique-token-fraction percentiles (low = repetitive):")
print(f"{'window':<22} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7}")
for name in WINDOWS:
    u = np.sort(results[name]["uniq"])
    n = len(u)
    print(f"{name:<22} " + " ".join(f"{u[int(q*n)]:>7.4f}" for q in [0.05, 0.25, 0.5, 0.75, 0.95]))
