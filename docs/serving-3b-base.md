# Serving Kotodama 3B Base

The public base checkpoint is [aethera-gp/kotodama-3b-base-final](https://huggingface.co/aethera-gp/kotodama-3b-base-final). It is a full, zstd-compressed PyTorch training checkpoint rather than a Transformers `from_pretrained` export. Use this repository's `serve.py` to load it.

## Golden Path

The supported default is the compiled fast engine on Linux with an NVIDIA CUDA GPU. It uses a static KV cache and CUDA-graph decode, so the first startup compiles and warms kernels; a cold start can take several minutes. Later starts reuse the local Inductor and Triton caches.

```bash
git clone https://github.com/LuxiaSL/kotodama.git
cd kotodama
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Select the PyTorch CUDA wheel appropriate for the installed NVIDIA driver.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-serve.txt

tools/serve_3b_base.sh
```

The launcher downloads `step_00195311.pt.zst` once into `models/kotodama-3b-base-final/`, keeps Hugging Face and compiler caches under `.cache/`, and starts an OpenAI-compatible server at `http://127.0.0.1:2222`.

Check the server after startup:

```bash
curl http://127.0.0.1:2222/health
curl http://127.0.0.1:2222/v1/models
curl http://127.0.0.1:2222/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kotodama-3b-base-final","prompt":"The capital of France is","max_tokens":32,"temperature":0.7}'
```

## Machine Sizing

These are intentionally conservative operating targets, not guarantees for every CUDA/PyTorch combination.

| Resource | Fast engine (golden path) | Reference engine |
| --- | --- | --- |
| GPU | One NVIDIA CUDA GPU with 48 GB+ VRAM recommended. The engine has been exercised on B200 hardware. | Compatibility and debugging fallback only; it avoids the fast engine's extra compiled weight copies and static cache, but has no published hardware target. |
| GPU software | Current NVIDIA driver and a CUDA-enabled PyTorch build. CUDA 12.8 wheels are the documented starting point. | Same PyTorch/CUDA requirements. |
| System RAM | 64 GB recommended for the full optimizer-containing checkpoint. | Same checkpoint-loading requirement. |
| Free local storage | 50 GB recommended: roughly 10 GB for the download plus temporary decompression and compiler caches. | Same checkpoint download and temporary decompression requirement. |

The 2.97B-parameter model is about 5.9 GB in BF16. At 4096 tokens, the fast engine's BF16 KV cache is about 0.47 GB and its fused fast-path weight copies add roughly 3.7 GB before PyTorch allocator, compilation, and runtime workspace overhead. This is why the documented fast-path target is 48 GB rather than a theoretical minimum.

`KOTODAMA_MAX_SEQ_LEN=2048 tools/serve_3b_base.sh` limits requests to 2048 tokens and halves the static KV cache, but it does not remove the compiled weight-copy cost. Treat it as a context limit, not as a way to turn an unsupported GPU into a supported host.

## Fast And Reference Engines

`--engine fast` is the default and the recommended path. It compiles the decode step with `max-autotune`, uses a static cache, and is the only path intended for routine serving. One process handles one generation stream at a time; run multiple replicas or use the optional gateway for parallel traffic.

Use the reference path only to investigate compatibility, correctness, or an unavailable fast engine:

```bash
KOTODAMA_ENGINE=reference tools/serve_3b_base.sh
```

It uses the eager model loop, does not compile or CUDA-graph decode, and is substantially slower. It is useful as a fallback, not as a throughput target.

## Checkpoint And Runtime Controls

`serve.py` accepts both `.pt` and `.pt.zst` checkpoints. For `.pt.zst`, it streams decompression into a temporary file before calling `torch.load`, so the compressed and decompressed byte buffers are not both held in RAM. Set `TMPDIR` to a filesystem with sufficient free space if `/tmp` is small.

The launcher accepts these environment overrides:

| Variable | Default | Purpose |
| --- | --- | --- |
| `KOTODAMA_MODEL_DIR` | `models/kotodama-3b-base-final` | Download location for the checkpoint. |
| `KOTODAMA_ENGINE` | `fast` | `fast` or `reference`. |
| `KOTODAMA_MAX_SEQ_LEN` | `4096` | Serving context cap from 16 to 4096. |
| `KOTODAMA_HOST` / `KOTODAMA_PORT` | `0.0.0.0` / `2222` | Bind address and port. |
| `KOTODAMA_PYTHON` | `python` | Interpreter used by the launcher. |

For a multi-replica setup, start from `configs/gateway.example.yaml` and replace its placeholder checkpoint path. Keep local paths, hostnames, scheduler details, and benchmark profiles out of version control.
