#!/bin/bash
# Generate NCA data and split into train/eval.
# Usage: tools/run_nca_gen_and_split.sh
set -e
cd ~/workspace/kotodama
VENV=~/workspace/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1

echo "=== Phase 1: Generate 3.1B NCA tokens ==="
python -m src.nca.generator \
    --output /models/kotodama-data/nca_seed17_3.1b_full.bin \
    --tokens 3100000000 \
    --num_rules 800 \
    --sims_per_rule 30 \
    --seed 17 \
    --device cuda:0

echo "=== Phase 2: Split into train (3B) + eval (100M) ==="
python tools/split_nca_data.py \
    --input /models/kotodama-data/nca_seed17_3.1b_full.bin \
    --train data/nca_seed17_3b_train.bin \
    --eval data/nca_seed17_50m_eval.bin \
    --train_tokens 3000000000

echo "=== Phase 3: Verify ==="
python -m src.nca.generator --verify data/nca_seed17_3b_train.bin
python -m src.nca.generator --verify data/nca_seed17_50m_eval.bin

echo "=== Cleanup: remove full file ==="
rm /models/kotodama-data/nca_seed17_3.1b_full.bin

echo "=== Done ==="
