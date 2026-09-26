#!/bin/bash
# Any single-process job (eval, parity checks, benches, data tools) with the
# launcher environment: venv, package path, unbuffered output, persistent
# compile caches. Accepts a script path or `-m module`.
# Usage: tools/run_py.sh scripts/utils/smoke_test.py [args...]
#        tools/run_py.sh -m kotodama.nca.generator --verify <file.bin>
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
exec python -u "$@"
