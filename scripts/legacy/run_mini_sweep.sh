#!/bin/bash
# Mini proxy sweep: quick directional results on 500M tokens.
# Tests P1 (AdamW) and P4 (Muon 0.03) to verify sweep infrastructure
# and get initial comparison before the full 6B sweep.
#
# Usage: bash scripts/run_mini_sweep.sh

set -euo pipefail

DATA_PATH="data/fineweb_edu_500m.bin"
GPUS=8
CHECKPOINT_BASE="checkpoints/mini_sweep"
TOTAL_STEPS=2000  # ~262M tokens at 131K/step
WARMUP=100

COMMON_ARGS="
  --data_path $DATA_PATH
  --model_size proxy
  --sequence_length 2048
  --micro_batch_size 4
  --global_batch_tokens 131072
  --warmup_steps $WARMUP
  --decay_start_pct 0.90
  --gradient_clip 1.0
  --log_every 50
  --save_every 500
  --keep_checkpoints 3
  --geo_monitor
  --geo_monitor_tier1_every 50
  --geo_monitor_tier2_every 500
  --wandb
  --activation_checkpointing
  --total_steps $TOTAL_STEPS
"

run_experiment() {
  local name=$1
  local extra_args=$2
  local port=$3

  echo ""
  echo "============================================"
  echo "Starting: $name"
  echo "============================================"

  mkdir -p "${CHECKPOINT_BASE}/${name}" logs

  .venv/bin/torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$port \
    -m src.training.train \
    $COMMON_ARGS \
    --checkpoint_dir "${CHECKPOINT_BASE}/${name}" \
    --wandb_run_name "mini-${name}" \
    $extra_args \
    2>&1 | tee "logs/mini_sweep_${name}.log"

  echo "Finished: $name"
}

echo "Mini proxy sweep: AdamW baseline vs Muon (lr=0.03)"
echo "Data: $DATA_PATH ($TOTAL_STEPS steps)"
echo ""

# P1: AdamW baseline
run_experiment "p1-adamw" \
  "--adamw_only --adamw_lr 8e-4" \
  29540

# P4: Muon LR 0.03 (our default)
run_experiment "p4-muon-003" \
  "--muon_lr 0.03 --adamw_lr 6e-4" \
  29541

echo ""
echo "============================================"
echo "Mini sweep complete!"
echo "============================================"
echo ""
echo "Analyze: .venv/bin/python scripts/analyze_metrics.py checkpoints/mini_sweep/*/metrics.jsonl"
echo "Wandb: https://wandb.ai/g-stratiy-personal-/luxia-base"
