#!/bin/bash
# Wrapper for luxia-base training jobs on cluster nodes.
# Activates shared venv, sets env, runs torchrun.
# Usage: tools/run_train.sh --config configs/proxy-benchmark.yaml [extra args...]
set -e
cd ~/luxi-files/kotodama
VENV=~/luxi-files/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export OMP_NUM_THREADS=16
export CPATH=~/luxi-files/python3.12-include
export PYTHONUNBUFFERED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1  # persist FA4 JIT kernels to disk
exec torchrun --nproc_per_node=8 -m src.training.train "$@"
