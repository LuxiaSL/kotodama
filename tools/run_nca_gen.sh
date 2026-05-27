#!/bin/bash
# Wrapper for NCA data generation on cluster nodes.
# Activates shared venv, runs generator on single GPU.
# Usage: tools/run_nca_gen.sh --output data/nca_seed17_3b.bin --tokens 3000000000 ...
set -e
cd ~/workspace/kotodama
VENV=~/workspace/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1
exec python -m src.nca.generator "$@"
