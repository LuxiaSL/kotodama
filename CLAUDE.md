# Kotodama — working in this repo

From-scratch LLM pretraining: NCA pre-pretraining + Block Attention Residuals + Muon.
One package, `src/kotodama`. `README.md` has the golden paths (one command per task) —
use them; do not add a parallel way to do something that already has one.

- **One table per fact.** Model shapes, AttnRes boundaries, tokenizer: `kotodama/presets.py`.
  Checkpoint I/O and names: `kotodama/ckpt.py` + `configs/checkpoints.yaml`. ChatML +
  stop tokens: `kotodama/serve/chatml.py`. Sampling laws: `kotodama/serve/laws.py`.
  Never restate these as literals elsewhere.
- **Machine-specific paths live in `site.env`** (gitignored; template `site.env.example`),
  loaded by `tools/_env.sh`. No hostnames, absolute data paths, or keys in tracked files.
- **Configs**: flat YAML = `train.py` flags, `extends:` for shared blocks (`configs/base/`).
  `tests/data/config_snapshot.json` pins every config's effective values; regenerate only
  on purpose (`KOTODAMA_WRITE_CONFIG_SNAPSHOT=1`).
- **Capture/readout** goes through `kotodama.model.capture` (parity-tested against
  `llama.py`); do not write another AttnRes forward mirror.
- **Tests**: `.venv/bin/python -m pytest` (CPU). A change to model/serve/train code ships
  with a test or a parity check.
- The architecture of the shipped 3B is FROZEN (DD-3B boundaries `0,1,3,7,15,19,24`).
