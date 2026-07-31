#!/usr/bin/env bash
# Task-sharded lm-eval battery for one checkpoint across N GPUs.
#
# Splits the standard 10-task battery into 3 groups balanced by request-token
# mass (hellaswag alone is ~half the battery), runs one group per GPU, waits,
# then merges the shard results into the standard analysis/lm_eval/<ckpt>/
# results.json layout.
#
# Usage (on gpu-host, from ~/workspace/kotodama):
#   tools/launch_lmeval_sharded.sh <checkpoint.pt[.zst]> [gpus] [batch_size]
#   tools/launch_lmeval_sharded.sh /models/kotodama-data/ckpt.pt.zst 4,5,6 32
#
# Detached use: nohup tools/launch_lmeval_sharded.sh ... > logs/lmeval_sharded.log 2>&1 &
#
# After any harness change, gate the numbers:
#   python -m scripts.eval.check_lmeval_equivalence <banked>/results.json <new>/results.json
set -euo pipefail

CKPT="${1:?usage: launch_lmeval_sharded.sh <ckpt> [gpus=4,5,6] [batch_size=32]}"
GPUS="${2:-4,5,6}"
BS="${3:-32}"

ROOT=~/workspace/kotodama
VENV=~/workspace/.venv-shared/bin/activate
STEM="$(basename "$CKPT")"; STEM="${STEM%.zst}"; STEM="${STEM%.pt}.pt"
SHARD_ROOT="$ROOT/analysis/lm_eval_shards/$STEM"

# 3 groups balanced by cost; hellaswag dominates the battery.
SHARD_TASKS=(
  "hellaswag"
  "lambada_openai,arc_easy,arc_challenge,wikitext"
  "piqa,boolq,winogrande,sciq,copa"
)

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -ne "${#SHARD_TASKS[@]}" ]; then
  echo "ERROR: need exactly ${#SHARD_TASKS[@]} GPUs (got: $GPUS)" >&2
  exit 1
fi

mkdir -p "$ROOT/logs" "$SHARD_ROOT"

# Pre-decompress .zst once so the three shards don't race on it (the loader
# also handles this atomically now, but doing it here avoids 3x the work).
if [[ "$CKPT" == *.zst ]]; then
  TMPDIR=/models/kotodama-data/tmp
  export TMPDIR
  DECOMP_DIR="$TMPDIR/kotodama_checkpoints"
  DECOMP="$DECOMP_DIR/$(basename "${CKPT%.zst}")"
  if [ ! -f "$DECOMP" ] && [ ! -f "${CKPT%.zst}" ]; then
    mkdir -p "$DECOMP_DIR"
    echo "pre-decompressing $CKPT -> $DECOMP"
    zstd -d "$CKPT" -o "$DECOMP.partial.$$" -f
    mv "$DECOMP.partial.$$" "$DECOMP"
  fi
fi

PIDS=()
for i in "${!SHARD_TASKS[@]}"; do
  GPU="${GPU_ARR[$i]}"
  N=$((i + 1))
  echo "shard$N gpu=$GPU tasks=${SHARD_TASKS[$i]}"
  (
    cd "$ROOT" && source "$VENV" &&
    CUDA_VISIBLE_DEVICES="$GPU" KOTODAMA_NO_TRITON_ATTNRES=1 \
    HF_HOME=/models/huggingface TMPDIR=/models/kotodama-data/tmp \
    python -m scripts.eval.run_lm_eval \
      --checkpoint "$CKPT" --tasks "${SHARD_TASKS[$i]}" \
      --config-section model --attn-res-boundaries 0,1,3,7,15,19,24 \
      --batch-size "$BS" --device cuda:0 \
      --output-dir "$SHARD_ROOT/shard$N"
  ) > "$ROOT/logs/lmeval_${STEM}_shard$N.log" 2>&1 &
  PIDS+=($!)
  sleep 2   # stagger .zst decompression / HF cache access
done

echo "waiting on shards: ${PIDS[*]}"
FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "ERROR: shard$((i + 1)) failed — see logs/lmeval_${STEM}_shard$((i + 1)).log" >&2
    FAIL=1
  fi
done
[ "$FAIL" -eq 0 ] || exit 1

cd "$ROOT" && source "$VENV"
MERGED="$ROOT/analysis/lm_eval/$STEM/results.json"
if [ -f "$MERGED" ]; then
  # Never silently clobber banked numbers — divert and let the operator
  # compare (check_lmeval_equivalence) before promoting.
  echo "WARNING: $MERGED already exists (banked row?) — writing merged"
  echo "         results to $SHARD_ROOT/results.json instead."
  MERGED="$SHARD_ROOT/results.json"
fi
python -m scripts.eval.merge_lmeval_shards \
  --shard-root "$SHARD_ROOT" \
  --output "$MERGED"
echo "DONE: $MERGED"
