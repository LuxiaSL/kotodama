#!/bin/bash
# The inference gateway: supervises a fleet of server replicas (one per
# (gpu, model) in the config) behind one OpenAI-compatible port.
# Usage: tools/run_gateway.sh --config configs/gateway.example.yaml [--gpus 0,1]
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
exec python -m kotodama.serve.gateway "$@"
