#!/bin/bash
# Canary throughput benchmark: measures steady-state tok/s for 3B and 8B models.
# Usage:
#   tools/run_canary.sh              # run both 3B and 8B
#   tools/run_canary.sh 3b           # run 3B only
#   tools/run_canary.sh 8b           # run 8B only
#   tools/run_canary.sh 8b 1         # run 8B on 1 GPU (for testing)
set -euo pipefail

MODEL="${1:-both}"
NUM_GPUS="${2:-8}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
CANARY_DIR="outputs/canary-${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$BASE_DIR"

# Activate venv — adjust path for your node
if [ -f ~/luxi-files/.venv-shared/bin/activate ]; then
    source ~/luxi-files/.venv-shared/bin/activate
elif [ -f .venv-train/bin/activate ]; then
    source .venv-train/bin/activate
fi

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

mkdir -p "$CANARY_DIR"

# Save hardware metadata
cat > "$CANARY_DIR/metadata.json" <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')",
    "gpu_count": ${NUM_GPUS},
    "torch_version": "$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'unknown')",
    "cuda_version": "$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
}
EOF

run_canary() {
    local size="$1"
    local config="configs/canary-${size}.yaml"
    local log_file="$CANARY_DIR/${size}.log"
    local ckpt_dir="$CANARY_DIR/ckpts-${size}"

    echo ""
    echo "============================================"
    echo "  Canary: ${size} model"
    echo "  Config: ${config}"
    echo "  GPUs: ${NUM_GPUS}"
    echo "  Log: ${log_file}"
    echo "============================================"

    local start_time
    start_time=$(date +%s.%N)

    set +e
    torchrun --nproc_per_node="$NUM_GPUS" -m src.training.train \
        --config "$config" \
        --checkpoint_dir "$ckpt_dir" \
        2>&1 | tee "$log_file"
    local exit_code=$?
    set -e

    local end_time
    end_time=$(date +%s.%N)
    local wall_time
    wall_time=$(python -c "print(f'{$end_time - $start_time:.1f}')")

    # Extract steady-state metrics (skip first 500 steps for warmup/compile)
    local steady_tps peak_mem avg_step_time
    steady_tps=$(grep 'tok/s=' "$log_file" | tail -4000 | \
        sed 's/.*tok\/s=\([0-9.]*\).*/\1/' | \
        python -c "
import sys
vals = [float(l) for l in sys.stdin if l.strip()]
if vals:
    vals.sort()
    n = len(vals)
    p5, med, p95 = vals[n//20], vals[n//2], vals[n*19//20]
    print(f'{sum(vals)/len(vals):.0f} (p5={p5:.0f} med={med:.0f} p95={p95:.0f})')
else:
    print('0')
" 2>/dev/null || echo "0")

    peak_mem=$(grep 'gpu_mem=' "$log_file" | tail -1 | \
        sed 's/.*gpu_mem=\([0-9.]*\).*/\1/' 2>/dev/null || echo "0")

    echo ""
    echo "  ${size} result:"
    echo "    Wall time:      ${wall_time}s"
    echo "    Steady tok/s:   ${steady_tps}"
    echo "    Peak GPU mem:   ${peak_mem} GB"
    echo "    Exit code:      ${exit_code}"

    # Write per-model summary
    cat > "$CANARY_DIR/${size}_summary.json" <<SUMEOF
{
    "model": "${size}",
    "wall_time_s": ${wall_time},
    "steady_state_tok_per_sec": "${steady_tps}",
    "peak_gpu_mem_gb": "${peak_mem}",
    "exit_code": ${exit_code},
    "num_gpus": ${NUM_GPUS},
    "total_steps": 5000,
    "seq_len": 4096,
    "async_checkpoint": true,
    "save_every": 500,
    "compile": true,
    "liger": true,
    "attn_res_blocks": 8
}
SUMEOF

    if [ $exit_code -ne 0 ]; then
        echo "  WARNING: ${size} canary failed (exit ${exit_code}). Check ${log_file}"
    fi
}

echo "============================================"
echo "  luxia-base canary throughput benchmark"
echo "  $(date)"
echo "  Output: $CANARY_DIR"
echo "============================================"

case "$MODEL" in
    3b)
        run_canary "3b"
        ;;
    8b)
        run_canary "8b"
        ;;
    both)
        run_canary "3b"
        run_canary "8b"
        ;;
    *)
        echo "Usage: $0 [3b|8b|both] [num_gpus]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Canary complete"
echo "  Results: $CANARY_DIR/"
echo "============================================"

# Print wall-clock training estimates
python -c "
import json, glob, os

results = {}
for f in sorted(glob.glob('$CANARY_DIR/*_summary.json')):
    with open(f) as fh:
        r = json.load(fh)
    results[r['model']] = r

if not results:
    print('No results found')
    exit()

print()
print('Wall-clock estimates (at steady-state tok/s):')
print('=' * 70)

token_budgets = [50e9, 80e9, 100e9, 150e9, 200e9, 250e9]

for model, r in results.items():
    tps_str = r['steady_state_tok_per_sec']
    try:
        tps = float(tps_str.split()[0])
    except (ValueError, IndexError):
        print(f'  {model}: could not parse tok/s')
        continue

    print(f'\n  {model.upper()} @ {tps:,.0f} tok/s (peak mem: {r[\"peak_gpu_mem_gb\"]} GB):')
    for budget in token_budgets:
        hours = budget / tps / 3600
        days = hours / 24
        label = f'{budget/1e9:.0f}B tokens'
        if days >= 1:
            print(f'    {label:>15s} -> {days:5.1f} days ({hours:6.1f} hours)')
        else:
            print(f'    {label:>15s} -> {hours:5.1f} hours')
" 2>/dev/null || true
