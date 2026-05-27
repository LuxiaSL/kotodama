#!/bin/bash
# Generate NCA data for the 3B model run.
# 3500 rules, 20 sims/rule, seed 17 → ~9.2B tokens
# Split into 9B train + ~200M eval
set -e
cd ~/workspace/kotodama
VENV=~/workspace/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1

# Clean old proxy NCA data
echo "=== Cleaning old NCA data ==="
rm -f data/nca_seed17_3b_train.bin data/nca_seed17_50m_eval.bin
rm -f /models/kotodama-data/nca_seed17_3.1b_full.bin

echo "=== Phase 1: Generate 9.2B NCA tokens (3500 rules, 20 sims) ==="
python -m src.nca.generator \
    --output /models/kotodama-data/nca_3b_seed17_full.bin \
    --tokens 9200000000 \
    --num_rules 3500 \
    --sims_per_rule 20 \
    --seed 17 \
    --device cuda:0

echo "=== Phase 2: Split into train (9B) + eval (200M) ==="
python tools/split_nca_data.py \
    --input /models/kotodama-data/nca_3b_seed17_full.bin \
    --train /models/kotodama-data/nca_3b_seed17_train.bin \
    --eval /models/kotodama-data/nca_3b_seed17_eval.bin \
    --train_tokens 9000000000

echo "=== Phase 3: Verify ==="
python -m src.nca.generator --verify /models/kotodama-data/nca_3b_seed17_train.bin
python -m src.nca.generator --verify /models/kotodama-data/nca_3b_seed17_eval.bin

echo "=== Cleanup ==="
rm /models/kotodama-data/nca_3b_seed17_full.bin

echo "=== Done ==="
