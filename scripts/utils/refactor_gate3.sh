#!/bin/bash
# Order-effect falsifier for the /logprobs old-vs-new gap: NEW first, then OLD twice, then NEW.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../tools/_env.sh"
: "${OLD:?}" "${OUT:?}"; NEW="$ROOT"; mkdir -p "$OUT"
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=8
CKPT="$KOTODAMA_DATA_ROOT/3b-language-FINAL-step195311.pt"
PAYLOAD='{"items":[{"id":"a","prompt":"The capital of France is","candidates":[" Paris"," London"]},{"id":"c","prompt":"She opened the door and","candidates":[" saw"," walked"]},{"id":"e","prompt":"The mitochondria is the","candidates":[" powerhouse"," center"]}],"top_n":3}'
serve() { local tag=$1 port=$2; shift 2
  "$@" --checkpoint "$CKPT" --model_size 3b --mode base --engine reference --port "$port" > "$OUT/serve-$tag.log" 2>&1 &
  local pid=$!; for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 5; done
  for k in 1 2; do curl -s "localhost:$port/logprobs" -H 'content-type: application/json' -d "$PAYLOAD" > "$OUT/lp-$tag-req$k.json"; done
  kill $pid; wait $pid 2>/dev/null; echo "serve $tag done"; }
(cd "$NEW" && serve new1 2301 python -m kotodama.serve.server)
(cd "$OLD" && serve old1 2302 python serve.py)
(cd "$OLD" && serve old2 2303 python serve.py)
(cd "$NEW" && serve new2 2304 python -m kotodama.serve.server)
python - "$OUT" <<'PY' || true
import json,sys,pathlib
o=pathlib.Path(sys.argv[1])
for f in sorted(o.glob("lp-*.json")):
    r=json.load(open(f))["results"]
    print(f.stem.ljust(16), [round(c["logprob"],4) for x in r for c in x["candidates"]])
PY
