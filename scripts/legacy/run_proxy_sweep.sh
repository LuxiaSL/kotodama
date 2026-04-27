#!/bin/bash
# Proxy validation sweep for luxia-base.
# Runs the 7-run validation matrix from the spec.
#
# Usage:
#   bash scripts/run_proxy_sweep.sh [data_path] [gpus]
#
# Each run: 300M proxy model, ~6B tokens (or less if data_path is smaller).
# Results logged to wandb project "luxia-base" with run names like "proxy-p1-adamw-baseline".

set -euo pipefail

DATA_PATH="${1:-data/fineweb_edu_500m.bin}"
GPUS="${2:-8}"
CHECKPOINT_BASE="checkpoints/proxy_sweep"
MASTER_PORT_BASE=29530
TOTAL_TOKENS=6000000000
WARMUP=2000

# Common args
COMMON_ARGS="
  --data_path $DATA_PATH
  --model_size proxy
  --sequence_length 2048
  --micro_batch_size 4
  --global_batch_tokens 131072
  --warmup_steps $WARMUP
  --decay_start_pct 0.90
  --decay_type sqrt
  --gradient_clip 1.0
  --log_every 50
  --save_every 500
  --keep_checkpoints 3
  --geo_monitor
  --geo_monitor_tier1_every 100
  --geo_monitor_tier2_every 500
  --wandb
  --activation_checkpointing
  --total_tokens $TOTAL_TOKENS
"

run_experiment() {
  local name=$1
  local extra_args=$2
  local port=$3

  echo "=== Starting $name ==="
  echo "  Port: $port"
  echo "  Args: $extra_args"

  .venv/bin/torchrun \
    --nproc_per_node=$GPUS \
    --master_port=$port \
    -m src.training.train \
    $COMMON_ARGS \
    --checkpoint_dir "${CHECKPOINT_BASE}/${name}" \
    --wandb_run_name "proxy-${name}" \
    $extra_args \
    2>&1 | tee "logs/proxy_sweep_${name}.log"

  echo "=== Finished $name ==="
  echo ""
}

mkdir -p logs "${CHECKPOINT_BASE}"

# P1: AdamW baseline (no Muon)
echo "Running P1: AdamW baseline"
run_experiment "p1-adamw-baseline" \
  "--adamw_only --adamw_lr 8e-4" \
  $((MASTER_PORT_BASE + 1))

# P2: Muon LR 0.01
echo "Running P2: Muon LR 0.01"
run_experiment "p2-muon-lr001" \
  "--muon_lr 0.01 --adamw_lr 6e-4" \
  $((MASTER_PORT_BASE + 2))

# P3: Muon LR 0.02
echo "Running P3: Muon LR 0.02"
run_experiment "p3-muon-lr002" \
  "--muon_lr 0.02 --adamw_lr 6e-4" \
  $((MASTER_PORT_BASE + 3))

# P4: Muon LR 0.03 (default)
echo "Running P4: Muon LR 0.03"
run_experiment "p4-muon-lr003" \
  "--muon_lr 0.03 --adamw_lr 6e-4" \
  $((MASTER_PORT_BASE + 4))

# P5: Muon LR 0.04
echo "Running P5: Muon LR 0.04"
run_experiment "p5-muon-lr004" \
  "--muon_lr 0.04 --adamw_lr 6e-4" \
  $((MASTER_PORT_BASE + 5))

# P6: Muon best + NCA (requires NCA data — skip for now)
echo "SKIPPING P6 (NCA) — NCA data not yet generated"

# P7: Muon best + FP8 (requires FP8 support — skip for now)
echo "SKIPPING P7 (FP8) — FP8 not yet implemented"

echo ""
echo "=== Proxy sweep complete (P1-P5) ==="
echo "View results at: https://wandb.ai/g-stratiy-personal-/luxia-base"
echo "Compare runs in wandb to identify optimal Muon LR."
