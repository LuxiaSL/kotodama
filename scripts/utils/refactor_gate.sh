#!/bin/bash
# One-off equivalence gate for the 2026-09 package refactor: OLD tree (package
# `src`, repo-root serve.py) vs NEW tree (package `kotodama`) on identical
# inputs. Run from the NEW tree via tools/run_py.sh-style env on one GPU.
# Usage: OLD=/path/to/old/tree OUT=/path/to/outdir bash scripts/utils/refactor_gate.sh
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../tools/_env.sh"
: "${OLD:?}" "${OUT:?}"
NEW="$ROOT"; mkdir -p "$OUT"
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=8
CKPT="$KOTODAMA_DATA_ROOT/3b-language-FINAL-step195311.pt"
TRAIN_ARGS=(--model_size proxy --data_path "$KOTODAMA_DATA_ROOT/train.bin"
  --total_steps 20 --warmup_steps 5 --sequence_length 1024 --micro_batch_size 4
  --global_batch_tokens 16384 --attn_res --attn_res_boundaries 0,3,7,12,21,25
  --save_every 1000000 --log_every 1 --seed 42 --keep_checkpoints 1)

train() {  # tree module tag
  (cd "$1" && PYTHONPATH="$1/src:$1" torchrun --nproc_per_node=1 --master_port "$4" -m "$2" \
     "${TRAIN_ARGS[@]}" --checkpoint_dir "$OUT/ckpt-$3" > "$OUT/train-$3.log" 2>&1)
  echo "train $3 exit=$?"
}
train "$OLD" src.training.train old 29511
train "$NEW" kotodama.training.train new 29512
train "$NEW" kotodama.training.train new2 29513

serve() {  # tree launch-args... ; tag port
  local tag=$1 port=$2; shift 2
  "$@" --checkpoint "$CKPT" --model_size 3b --mode base --engine reference \
     --port "$port" > "$OUT/serve-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 5; done
  for i in 0 1 2; do
    P=$(python -c "print(['The old lighthouse keeper','def fibonacci(n):','Dear Margaret,\n\nI'][$i])")
    curl -s "localhost:$port/generate" -H 'content-type: application/json' \
      -d "$(python -c "import json,sys;print(json.dumps({'prompt':sys.argv[1],'max_new_tokens':64,'temperature':0.0,'repetition_penalty':1.0}))" "$P")" \
      | python -c "import json,sys;print(json.load(sys.stdin).get('text'))" > "$OUT/gen-$tag-$i.txt"
  done
  curl -s "localhost:$port/logprobs" -H 'content-type: application/json' \
    -d '{"items":[{"id":"a","prompt":"The capital of France is","candidates":[" Paris"," London"]}],"top_n":5}' > "$OUT/logprobs-$tag.json"
  kill $pid; wait $pid 2>/dev/null
  echo "serve $tag done"
}
(cd "$OLD" && serve old 2291 python serve.py)
(cd "$NEW" && serve new 2292 python -m kotodama.serve.server)

lmeval() {  # tree module tag
  (cd "$1" && PYTHONPATH="$1/src:$1" python -m "$2" --checkpoint "$CKPT" --tasks arc_easy --limit 300 \
     --config-section model --attn-res-boundaries 0,1,3,7,15,19,24 --batch-size 32 --device cuda:0 \
     --output-dir "$OUT/lmeval-$3" > "$OUT/lmeval-$3.log" 2>&1)
  echo "lmeval $3 exit=$?"
}
lmeval "$OLD" scripts.eval.run_lm_eval old
lmeval "$NEW" scripts.eval.run_lm_eval new

python - "$OUT" <<'PY'
import re,sys,json,glob,pathlib
o=pathlib.Path(sys.argv[1]); ok=True
def losses(t): return [float(x) for x in re.findall(r"loss=([0-9.]+)",(o/f"train-{t}.log").read_text())]
a,b,c=losses("old"),losses("new"),losses("new2")
print("train steps", len(a),len(b),len(c))
if not a or len(a)!=len(b): ok=False; print("TRAIN: missing/unequal loss lines")
else:
    d_on=max(abs(x-y) for x,y in zip(a,b)); d_nn=max(abs(x-y) for x,y in zip(b,c)) if c else float('nan')
    print(f"train max|old-new|={d_on:.6g}  max|new-new2|={d_nn:.6g}  final old={a[-1]} new={b[-1]}")
    ok &= d_on <= max(d_nn,1e-6)*1.0001 or d_on==0
for i in range(3):
    g1=(o/f"gen-old-{i}.txt").read_text(); g2=(o/f"gen-new-{i}.txt").read_text()
    print(f"gen[{i}] identical={g1==g2} len={len(g1)}"); ok &= (g1==g2 and len(g1)>10)
l1=json.loads((o/"logprobs-old.json").read_text()); l2=json.loads((o/"logprobs-new.json").read_text())
l1.pop("model",None); l2.pop("model",None)
print("logprobs identical", l1==l2); ok&= l1==l2
def acc(t):
    f=glob.glob(str(o/f"lmeval-{t}"/"**"/"results*.json"),recursive=True)
    return json.load(open(f[0]))["results"]["arc_easy"] if f else None
r1,r2=acc("old"),acc("new"); print("lmeval old",r1); print("lmeval new",r2); ok &= (r1 is not None and r1==r2)
print("GATE", "PASS" if ok else "FAIL")
PY
