# P0.5 staged: ARM-D proxy rerun on the v3 generator

> STATUS: STAGED 2026-09-27. Not launched. Fire in order, after the soak frees node2.
> Spec: lab `SPEC-1B-FORMATION-LADDER-2026-09-26` §2 P0.5, §5, §3b L-P1.
> **This file names nodes and absolute paths. The repo is public, so `git rm` it (or move
> it to the lab) before this branch's merge is pushed.**

Contents of this branch: `configs/p05-armd-v3.yaml`, `scripts/eval/nca_gates_sweep.py`,
`kotodama.eval.nca_gates.score_eval_dir` (a refactor: the CLI and the sweep now share
one scoring path), `tests/test_p05_staging.py`, and the config snapshot entry.

```
TREE=/home/athuser/luxi-files/kotodama      # node2 mirror. CONFIRM the name (see step 0)
DATA=/models/kotodama-data/p05-dyadic-v3    # v3 bin + sidecar + eval sets
CKPT=/models/kotodama-data/p05/armd-v3      # training checkpoints (+ gates/ subdir)
```

## 0. Pre-flight (the step that saves 5am)

1. **Merge this branch, then re-sync node2 from the merged main.** The current node2
   trees predate the v3 writer (`5335188`), the gate scorer (`6ac595d`) and everything
   on this branch. Run this from the local main checkout, clean tree:
   `tools/sync.sh node2:$TREE` (use `-n` first for a dry run).
2. Verify on node2:
   `ssh node2 "cd $TREE && ls configs/p05-armd-v3.yaml scripts/eval/nca_gates_sweep.py src/kotodama/eval/nca_gates.py && grep -c reserve_eos src/kotodama/nca/gen_dyadic_data.py && cat site.env"`.
   `site.env` must set the venv (`KOTODAMA_VENV`), compile caches on /models
   (`KOTODAMA_CACHE_ROOT`) and `KOTODAMA_SCHEDULER_JOB_ID_VAR=HEIMDALL_JOB_ID`. sync.sh
   protects it but does not create it.
3. `ssh node2 "ls $CKPT 2>/dev/null | head; ls $DATA 2>/dev/null | head; df -h /models /dev/shm; which zstd"`.
   **`$CKPT` must be empty or absent.** train.py auto-resumes from any `step_*.pt*`
   it finds in `checkpoint_dir`. /models is node-local, so `/models/kotodama-data`
   may not exist on node2 yet; the commands below create it. Space needed: about
   3 GB for the data (shards plus the concatenated 1.5 GB bin) and on the order of
   1 GB per checkpoint (×15, bf16 weights plus optimizer state; not measured).

## (a) Generate the v3 bin: 8 shards × 93.75M = 750M tokens, then finalize

The flags are the spec rulings (and also the writer's defaults, written out on
purpose): `--orig-frac 0.30` (a fraction of tokens), a tag-reuse ramp 0.0→0.5,
`--p-unmarked 0.5`, band 0.4:0.7, default geometries
(`16x16x2:.35,16x16x4:.35,32x32x2:.15,32x32x4:.15`), default seed 1021.
`--finalize` has to receive the same flags, because its eval sets are built from them.

```bash
heimdall submit --node node2 --gpus 8 -e 120 --max-retries 0 --name p05-gen \
  -w /home/athuser/luxi-files/kotodama \
  '( set -e; OUT=/models/kotodama-data/p05-dyadic-v3; mkdir -p "$OUT/logs";
     FLAGS="--orig-frac 0.30 --tag-reuse-start 0.0 --tag-reuse-end 0.5 --p-unmarked 0.5 --bands 0.4:0.7";
     IFS=, read -ra G <<< "$CUDA_VISIBLE_DEVICES"; pids=();
     for s in 0 1 2 3 4 5 6 7; do
       CUDA_VISIBLE_DEVICES=${G[$s]} tools/run_py.sh -m kotodama.nca.gen_dyadic_data \
         --shard $s --n-shards 8 --target-tokens 93750000 --out "$OUT" $FLAGS \
         > "$OUT/logs/gen-$s.log" 2>&1 & pids+=($!);
     done;
     for p in "${pids[@]}"; do wait $p; done;
     tools/run_py.sh -m kotodama.nca.gen_dyadic_data --finalize --out "$OUT" $FLAGS \
       > "$OUT/logs/finalize.log" 2>&1 )'
```

- The single quotes are deliberate: heimdall runs the string in a bash wrapper on the
  node, and it has to expand there, not locally. Each shard is pinned to one of the
  job's allocated GPUs (`G[$s]`). One failed shard fails the job (`set -e` + `wait`).
- Progress: `ssh node2 tail -n 2 /models/kotodama-data/p05-dyadic-v3/logs/gen-*.log`
  (the writer prints tokens and the orig% it is hitting against the 30% target every
  100 episodes).
- Done means: `finalize.log` ends in a meta JSON with `total_tokens` ≥ 750,000,000 and
  eval counts `ledger_unique 400 / ledger_overwrite 400 / smoothing 200`. Files:
  `nca_v3.bin`, `sidecar.jsonl`, `eval_*.jsonl`, `meta.json` (plus the shard_* inputs,
  which are safe to delete after that).
- Known cosmetic issue: the `args` block in meta.json holds the finalize call's own
  argv (its target_tokens and shard values are defaults). The per-shard truth is in
  the gen logs and the sidecar.

## (b) Re-sync reminder

If anything on the branch changed after step 0, sync again before (c). The config
and the sweep have to be the merged versions.

## (c) Train: proxy + DD-v1 + doc_masking, 750M tokens, 6 GPUs, 15 checkpoints

```bash
heimdall submit --node node2 --gpus 6 -e 90 --max-retries 0 --name p05-armd-v3 \
  -w /home/athuser/luxi-files/kotodama --env KOTODAMA_NPROC=6 --after <p05-gen id> \
  'tools/run_train.sh --config configs/p05-armd-v3.yaml --checkpoint_dir /models/kotodama-data/p05/armd-v3'
```

- **`--env KOTODAMA_NPROC=6` is load-bearing.** run_train.sh defaults to 8 ranks, and
  8 ranks on 6 visible GPUs crashes. The config's batch is exact only at 6 ranks:
  6 × mb 6 × 4096 = 147,456 tok/step, 5,086 steps. train.py floors grad_accum
  silently, so a different rank count quietly changes the batch.
- Checkpoints every 340 steps (50.1M tokens): steps 340…4760 plus the final 5085 give
  15 points. They are staged through `/dev/shm/luxia-p05-armd-v3` (run-private), and
  that directory keeps its copies afterwards: `rm -rf /dev/shm/luxia-p05-armd-v3`
  once the sweep has read `$CKPT`.
- The first log lines to check: `Training: 5086 steps, 1 grad_accum, 147456 tokens/step`
  and `doc_masking=True`. Wandb is off, as in Aug. Loss is in the job log.
- Throughput expectation: the lab DRAFT-SPEC §6 *estimated* about 1 h per 750M-token
  NCA phase at proxy scale. That figure is not a measured Aug wall-clock. `-e 90` is
  a guess.

## (d) Gate sweep: checkpoints to the formation curve

```bash
heimdall submit --node node2 --gpus 1 -e 60 --max-retries 0 --name p05-gates \
  -w /home/athuser/luxi-files/kotodama --after <p05-armd-v3 id> \
  'mkdir -p /models/kotodama-data/tmp && TMPDIR=/models/kotodama-data/tmp \
   tools/run_py.sh -m scripts.eval.nca_gates_sweep \
     --ckpt-dir /models/kotodama-data/p05/armd-v3 \
     --eval-dir /models/kotodama-data/p05-dyadic-v3 --init-seed 42'
```

- Output goes to `$CKPT/gates/`: one `gates_stepNNNNNNNN.json` per step (every item
  row, all splits), plus `summary.tsv` and `summary.json`. The table is printed to the
  job log as well.
- `--init-seed 42` scores the run's exact untrained init as step 0. That is the
  empirical floor on this eval set (the Aug untrained control read 0.113).
- DD-v1 is passed explicitly from `presets.DEFAULT_BOUNDARIES["proxy"]`. The loader's
  own auto-detect would assume n_blocks=7, and since boundaries do not change the
  parameter keys, that mistake would load without any error.
- `TMPDIR` on /models matters: `.pt.zst` checkpoints are decompressed into
  `$TMPDIR/kotodama_checkpoints/`, and node `/` is tiny. The sweep deletes each
  decompressed copy after scoring it.
- The sweep is resumable and safe to run mid-training for an early look. Persisted
  checkpoints are renamed into place atomically, and `.tmp` files are never matched.
  A second run scores only the new steps.

## (e) Reading L-P1 (pre-registered; spec §3b)

**Criterion as written:** ledger acc > chance (0.100) at ≥1 checkpoint. The Aug
ARM-D-lite arms scored 0.068–0.070 (below chance, median rank ~6), which reads as a
learned anti-copy prior. The untrained control read 0.113.

| read | what it looks like in `summary.tsv` | consequence |
|---|---|---|
| **PASS** | `ledger_unique_acc` clearly above 0.100 at some trained step, and above the step-0 row. The one-sided p<.05 bar for a single checkpoint at n=400 is ≈0.125, and the summary prints it | the v3 fixes let the ledger form: A4 unblocked; the curve's shape (where it lifts) is the dose read |
| **FAIL (Aug reproduced)** | every trained step ≤ 0.100, sitting near 0.07 with median rank ~6 | the curriculum design is still broken: **A4 blocked** until redesign (A0–A3 unaffected) |
| **ambiguous** | trained steps hover at chance ≈ the step-0 row: neither anti-copy nor binding | anti-copy artifact gone, induction not formed at 750M. This is the Aug §5c "token-starved" branch: the next rung is the 3B-token dyadic phase with this same sweep |

Read these before calling it:
- **Pick the primary split before looking.** The spec does not say which ledger file
  L-P1 means. Recommendation: `ledger_unique` is primary (pure retrieval, the
  well-posed analog of Aug's ledger); `ledger_overwrite` is secondary (state update,
  and harder by design).
- **Multiple comparisons:** 15 checkpoints × 2 files. One step barely over 0.100 is
  not a PASS. Look for a sustained lift across adjacent checkpoints.
- The per-step JSONs carry the diagnostic splits (self- vs other-committed, lag ≤8
  vs >8, first- vs re-query, unique vs post-overwrite). The sidecar
  (`$DATA/sidecar.jsonl`) confirms the realized token mix (orig ≈30% by tokens).
- **Not like-for-like with Aug on:** batch (147,456 vs 131,072 tok/step), doc_masking
  (on vs off), generator (v3 vs ARM-D-lite), and eval sets (v3 unique/overwrite vs
  the legacy ill-posed ledger). The comparison is to the Aug *floor pattern*, not a
  matched pair.
