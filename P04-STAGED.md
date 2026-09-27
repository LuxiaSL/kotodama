# P0.4 staged: induction / prefix-matching sweep over the p2x2 arms

Instrument: `kotodama.eval.induction` (tests: `tests/test_induction.py`).
It has not been run on a real checkpoint yet. Everything below is the plan
for when the sweep fires.

## What it measures

- **Behavioral (all checkpoints):** `n_seqs` sequences of `[EOS][A][A]`, where
  `A` is `seq_len` uniform-random ids with tokenizer special ids excluded.
  - `induction_gain` = mean CE on the first copy − mean CE on the second copy,
    in nats, over positions `j >= 1`.
  - `induction_gain_normalized` = gain / first-copy CE. 0 means no copying,
    1 means perfect copying.
  - The output also includes per-position curves, per-sequence gains (for the
    SE), and `uniform_loss = ln|candidates|` as a reference point.
- **Prefix matching (AttnRes checkpoints only):** a per-head score, read from
  the eager softmax weights of `kotodama.model.capture.instrumented_forward`.
  It is the mass that second-copy `A[j]` puts on first-copy `A[j+1]`.
  A uniform head scores about 1/position, which is about 0.003 at
  `seq_len=256`. An induction head scores toward 1.
- **Positive control (CPU test):** a 2-layer AttnRes model trained for 150
  steps on repeats reaches normalized gain > 0.8. Its prefix-matching mass
  sits in layer 1 (about 0.8), with layer 0 below 0.2, where layer 0 acts as
  the previous-token layer. Random init gives a gain of about 0 on both
  channels.

## Per-checkpoint invocation

```bash
tools/run_py.sh -m kotodama.eval.induction \
    --checkpoint <ckpt.pt | registry name> \
    --config-section proxy \
    --attn-res-boundaries ddv1 \
    --tag p2x2-<arm>-<step> \
    --n-seqs 64 --seq-len 256 --seed 0 \
    --out <results dir>/induction_p2x2-<arm>-<step>.json
```

- **Always pass `--attn-res-boundaries`.** Without it, `load_model`
  auto-detects AttnRes and assumes `n_blocks=7`, which gives the wrong
  boundaries for DD-v1 proxies. `ddv1` / `dd3b` resolve through
  `kotodama.presets.BOUNDARIES`, or pass a comma list. Check each arm's
  training config for what it actually used. For a non-AttnRes arm, pass
  `--attn-res off`; the prefix-matching read is then skipped and marked
  `skipped` in the JSON.
- **Keep `--seed`, `--n-seqs` and `--seq-len` fixed across the whole sweep.**
  Every checkpoint then sees the same token batch. The seed fixes the batch
  exactly, given the same candidate set (same vocab and exclusions).
- The tokenizer (`presets.TOKENIZER_NAME`) is only used to resolve special
  ids and EOS. With no HF cache, it falls back to excluding only id 0 and
  logs a warning. On the node, run with `HF_HOME` from `site.env` so the
  cached tokenizer is used. `--exclude-ids` overrides the special-id set.
- `--attn off` skips the head read. `--attn-n-seqs` (default 16) caps how
  many sequences go through the eager capture walker.

## Where the checkpoints are: NOT VERIFIED

Nobody has looked up the p2x2 checkpoint paths from this worktree (no node
access). They have to be found when the sweep fires. The expected location
is under `$KOTODAMA_DATA_ROOT` on the training node (per `site.env`), next to
the proxy inits (`$KOTODAMA_DATA_ROOT/proxy-inits/`), in whatever run
directory the p2x2 protocol wrote. The lab canon (the SPEC-1B ladder / p2x2
results docs) should name the directories. List them before launching, and
include each arm's step checkpoints (not just the final one) if the
per-checkpoint trajectory is wanted.

## Cost

Trivial. For a proxy (108M) with defaults, one checkpoint is:
- 64 × 513 tokens forward (4 batches of 16);
- 16 sequences through the eager capture walker. At bf16 that is about
  28 layers × 4 heads × 513² ≈ 60 MB of attention weights per sequence,
  processed in batches of 4.

That is seconds of GPU time per checkpoint, dominated by checkpoint load.
A whole p2x2 sweep fits on 1–2 GPUs alongside other work, for example with
`heimdall submit --gpus 1` running a shell loop over the checkpoint list.
For 3B-scale checkpoints, use `--config-section 3b --attn-res-boundaries dd3b`.
The capture read is about 0.4 GB/sequence at bf16, so keep
`--attn-batch-size` at 4 or lower.
