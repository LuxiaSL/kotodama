#!/bin/bash
# Wrapper for luxia-base eval/analysis jobs on cluster nodes.
# Activates shared venv, sets env, runs a Python script.
# Usage: tools/run_eval.sh scripts/analysis/eval_generate.py [args...]
#        tools/run_eval.sh scripts/analysis/extract_activation_geometry.py [args...]
set -e
cd ~/luxi-files/kotodama
VENV=~/luxi-files/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=16
export CPATH=~/luxi-files/python3.12-include
export PYTHONUNBUFFERED=1
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"
exec python "$@"
