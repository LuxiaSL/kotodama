#!/bin/bash
# Refactor gate, round 2: noise floors. OLD vs NEW vs NEW (repeat) on
#  (a) training from IDENTICAL weights (--resume_weights), 10 steps,
#  (b) /logprobs on 6 prompts (reference engine),
#  (c) lm-eval arc_easy limit 300.
# PASS = every old-vs-new difference is within the new-vs-new difference.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../../tools/_env.sh"
: "${OLD:?}" "${OUT:?}" "${INIT:?}"
NEW="$ROOT"; mkdir -p "$OUT"
export CUBLAS_WORKSPACE_CONFIG=:16:8 OMP_NUM_THREADS=8
CKPT="$KOTODAMA_DATA_ROOT/3b-language-FINAL-step195311.pt"
TA=(--model_size proxy --data_path "$KOTODAMA_DATA_ROOT/train.bin" --resume_weights "$INIT"
  --total_steps 10 --warmup_steps 2 --sequence_length 1024 --micro_batch_size 4
  --global_batch_tokens 16384 --attn_res --attn_res_boundaries 0,3,7,12,21,25
  --save_every 1000000 --log_every 1 --seed 42 --keep_checkpoints 1)
train() { (cd "$1" && PYTHONPATH="$1/src:$1" torchrun --nproc_per_node=1 --master_port "$4" -m "$2" "${TA[@]}" \
  --checkpoint_dir "$OUT/ckpt-$3" > "$OUT/train-$3.log" 2>&1); echo "train $3 exit=$?"; }
train "$OLD" src.training.train old 29521
train "$NEW" kotodama.training.train new 29522
train "$NEW" kotodama.training.train new2 29523

PAYLOAD='{"items":[
 {"id":"a","prompt":"The capital of France is","candidates":[" Paris"," London"]},
 {"id":"b","prompt":"def add(a, b):\n    return","candidates":[" a"," b"]},
 {"id":"c","prompt":"She opened the door and","candidates":[" saw"," walked"]},
 {"id":"d","prompt":"1, 2, 3, 4,","candidates":[" 5"," 6"]},
 {"id":"e","prompt":"The mitochondria is the","candidates":[" powerhouse"," center"]},
 {"id":"f","prompt":"Once upon a time, there was a","candidates":[" little"," young"]}],"top_n":5}'
serve() { local tag=$1 port=$2; shift 2
  "$@" --checkpoint "$CKPT" --model_size 3b --mode base --engine reference --port "$port" > "$OUT/serve-$tag.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 180); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 5; done
  curl -s "localhost:$port/logprobs" -H 'content-type: application/json' -d "$PAYLOAD" > "$OUT/logprobs-$tag.json"
  kill $pid; wait $pid 2>/dev/null; echo "serve $tag done"; }
(cd "$OLD" && serve old 2291 python serve.py)
(cd "$NEW" && serve new 2292 python -m kotodama.serve.server)
(cd "$NEW" && serve new2 2293 python -m kotodama.serve.server)

lmeval() { (cd "$1" && PYTHONPATH="$1/src:$1" python -m "$2" --checkpoint "$CKPT" --tasks arc_easy --limit 300 \
  --config-section model --attn-res-boundaries 0,1,3,7,15,19,24 --batch-size 32 --device cuda:0 \
  --output-dir "$OUT/lmeval-$3" > "$OUT/lmeval-$3.log" 2>&1); echo "lmeval $3 exit=$?"; }
lmeval "$OLD" scripts.eval.run_lm_eval old
lmeval "$NEW" scripts.eval.run_lm_eval new
lmeval "$NEW" scripts.eval.run_lm_eval new2

python - "$OUT" <<'PY' || true
import re,sys,json,glob,pathlib
o=pathlib.Path(sys.argv[1])
def L(t):
    try: return [float(x) for x in re.findall(r"step\s*\d+.*?loss=([0-9.]+)",(o/f"train-{t}.log").read_text())] or \
                [float(x) for x in re.findall(r"loss=([0-9.]+)",(o/f"train-{t}.log").read_text())]
    except Exception as e: return str(e)
a,b,c=L("old"),L("new"),L("new2"); print("train old",a); print("train new",b); print("train new2",c)
def lp(t):
    try: return {r["id"]:[x["logprob"] for x in r["candidates"]] for r in json.load(open(o/f"logprobs-{t}.json"))["results"]}
    except Exception as e: return str(e)
p1,p2,p3=lp("old"),lp("new"),lp("new2")
if all(isinstance(p,dict) for p in (p1,p2,p3)):
    d=lambda x,y:max(abs(u-v) for k in x for u,v in zip(x[k],y[k]))
    print(f"logprobs max|old-new|={d(p1,p2):.3g} max|new-new2|={d(p2,p3):.3g} max|old-new2|={d(p1,p3):.3g}")
else: print("logprobs", p1, p2, p3)
for t in ("old","new","new2"):
    f=glob.glob(str(o/f"lmeval-{t}"/"**"/"results.json"),recursive=True)
    try: d=json.load(open(f[0])); r=d.get("results",d)["arc_easy"]; print("lmeval",t,r["acc,none"],r["acc_norm,none"])
    except Exception as e: print("lmeval",t,"ERR",e)
PY
