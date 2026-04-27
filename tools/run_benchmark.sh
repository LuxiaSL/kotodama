#!/bin/bash
# Benchmark harness: runs optimization matrix on proxy model with full logging.
# Usage: tools/run_benchmark.sh [output_dir]
#
# Runs 9 configs (Liger × Attn × Compile) with AttnRes DD-v1 baseline.
# Each run: 200 steps, real data, torch.profiler on steps 50-150.
# Full stdout/stderr captured per config. Summary JSON at end.
#
# Example:
#   tools/run_benchmark.sh outputs/benchmark-2026-04-01
set -euo pipefail

BENCH_DIR="${1:-outputs/benchmark-$(date +%Y%m%d-%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cd ~/luxi-files/kotodama
source ~/luxi-files/.venv-shared/bin/activate
export OMP_NUM_THREADS=16
export CPATH=~/luxi-files/python3.12-include
export PYTHONUNBUFFERED=1

mkdir -p "$BENCH_DIR"

# ── Base config (matches past proxy runs with DD-v1 AttnRes) ──
BASE_ARGS=(
    --data_path data/fineweb_edu_6b.bin
    --model_size proxy
    --sequence_length 2048
    --micro_batch_size 4
    --global_batch_tokens 131072
    --warmup_steps 50
    --decay_start_pct 0.90
    --decay_type cosine
    --gradient_clip 1.0
    --total_steps 200
    --muon_lr 0.02
    --muon_ns_coefficients gram_ns
    --adamw_lr 6e-4
    --attn_res
    --attn_res_boundaries 0,3,7,12,21,25
    --activation_checkpointing
    --save_every 999999
    --keep_checkpoints 1
    --log_every 10
    --seed 42
    --checkpoint_dir "$BENCH_DIR/ckpts"
)

# ── Benchmark configs ──
# Format: "name|extra_flags"
CONFIGS=(
    "bare_sdpa|"
    "compile_sdpa|--compile"
    "liger_sdpa|--use_liger"
    "liger_compile_sdpa|--use_liger --compile"
    "bare_fa2|--attn_impl fa2"
    "compile_fa2|--attn_impl fa2 --compile"
    "liger_fa2|--use_liger --attn_impl fa2"
    "liger_compile_fa2|--use_liger --attn_impl fa2 --compile"
)
# FA4 added separately if available (requires flash_attn.cute)
python -c "from flash_attn.cute import flash_attn_func; print('fa4')" 2>/dev/null && \
    CONFIGS+=("bare_fa4|--attn_impl fa4") || \
    echo "[bench] FA4 not available, skipping"

echo "============================================"
echo "  luxia-base optimization benchmark"
echo "  $(date)"
echo "  Output: $BENCH_DIR"
echo "  Configs: ${#CONFIGS[@]}"
echo "============================================"

# Save benchmark metadata
cat > "$BENCH_DIR/metadata.json" <<METAEOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
    "gpu_count": $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l),
    "torch_version": "$(python -c 'import torch; print(torch.__version__)')",
    "cuda_version": "$(python -c 'import torch; print(torch.version.cuda)')",
    "flash_attn_version": "$(python -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null || echo 'N/A')",
    "base_args": "${BASE_ARGS[*]}",
    "configs": [$(printf '"%s",' "${CONFIGS[@]}" | sed 's/,$//')]
}
METAEOF

SUMMARY_FILE="$BENCH_DIR/results_summary.jsonl"
> "$SUMMARY_FILE"

run_config() {
    local name="$1"
    local extra="$2"
    local log_file="$BENCH_DIR/${name}.log"
    local profile_dir="$BENCH_DIR/${name}_profile"

    echo ""
    echo "──────────────────────────────────────────"
    echo "  Config: $name"
    echo "  Extra flags: ${extra:-'(none)'}"
    echo "  Log: $log_file"
    echo "──────────────────────────────────────────"

    # Clean checkpoint dir between runs
    rm -rf "$BENCH_DIR/ckpts"

    local start_time
    start_time=$(date +%s.%N)

    # Run training with profiler enabled
    # The profiler args are passed to train.py which should handle them
    # For now, we capture wall-clock and parse tok/s from logs
    set +e
    torchrun --nproc_per_node=8 -m src.training.train \
        "${BASE_ARGS[@]}" \
        $extra \
        2>&1 | tee "$log_file"
    local exit_code=$?
    set -e

    local end_time
    end_time=$(date +%s.%N)
    local wall_time
    wall_time=$(python -c "print(f'{$end_time - $start_time:.1f}')")

    # Extract metrics from log
    local avg_tps median_tps peak_mem final_loss
    avg_tps=$(grep 'tok/s=' "$log_file" | tail -100 | \
        sed 's/.*tok\/s=\([0-9.]*\).*/\1/' | \
        python -c "import sys; vals=[float(l) for l in sys.stdin if l.strip()]; print(f'{sum(vals)/len(vals):.0f}') if vals else print('0')")
    peak_mem=$(grep 'gpu_mem=' "$log_file" | tail -1 | \
        sed 's/.*gpu_mem=\([0-9.]*\).*/\1/' || echo "0")
    final_loss=$(grep 'loss=' "$log_file" | tail -1 | \
        sed 's/.*loss=\([0-9.]*\).*/\1/' || echo "0")

    echo ""
    echo "  Result: exit=$exit_code wall=${wall_time}s avg_tps=$avg_tps peak_mem=${peak_mem}GB loss=$final_loss"

    # Append to summary
    cat >> "$SUMMARY_FILE" <<SUMEOF
{"name": "$name", "exit_code": $exit_code, "wall_time_s": $wall_time, "avg_tok_per_sec": $avg_tps, "peak_gpu_mem_gb": $peak_mem, "final_loss": $final_loss, "extra_flags": "$extra"}
SUMEOF

    if [ $exit_code -ne 0 ]; then
        echo "  WARNING: config $name failed with exit code $exit_code"
        echo "  Check $log_file for details"
    fi
}

# ── Run all configs ──
for config_spec in "${CONFIGS[@]}"; do
    IFS='|' read -r name extra <<< "$config_spec"
    run_config "$name" "$extra"
done

echo ""
echo "============================================"
echo "  Benchmark complete"
echo "  Results: $SUMMARY_FILE"
echo "============================================"
echo ""

# ── Print comparison table ──
python -c "
import json, sys

results = []
with open('$SUMMARY_FILE') as f:
    for line in f:
        if line.strip():
            results.append(json.loads(line))

if not results:
    print('No results')
    sys.exit(0)

baseline_tps = next((r['avg_tok_per_sec'] for r in results if r['name'] == 'compile_sdpa'), None)

print(f'{'Config':<25s} {'tok/s':>10s} {'vs base':>8s} {'wall(s)':>8s} {'mem(GB)':>8s} {'loss':>8s} {'exit':>5s}')
print('-' * 75)
for r in results:
    tps = r['avg_tok_per_sec']
    speedup = f'{tps/baseline_tps:.2f}x' if baseline_tps and baseline_tps > 0 else 'N/A'
    status = 'OK' if r['exit_code'] == 0 else f'FAIL({r[\"exit_code\"]})'
    print(f'{r[\"name\"]:<25s} {tps:>10.0f} {speedup:>8s} {r[\"wall_time_s\"]:>8.1f} {r[\"peak_gpu_mem_gb\"]:>8s} {r[\"final_loss\"]:>8s} {status:>5s}')
" 2>&1 | tee "$BENCH_DIR/comparison_table.txt"

echo ""
echo "Full logs in: $BENCH_DIR/"
echo "Summary JSONL: $SUMMARY_FILE"
echo "Table: $BENCH_DIR/comparison_table.txt"
