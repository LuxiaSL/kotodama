"""Calibrate a kotodama model for anamnesis: its positional means and residual basis.

anamnesis's ``run_calibration`` loads a Hugging Face checkpoint, which a kotodama checkpoint is
not (an HF cast would drop the AttnRes routing). This is the same calibration with only the two
model-facing steps swapped for kotodama's:

* **sampling** each ruler prompt with :func:`kotodama.model.capture.generate_light`, seeded by
  the prompt's index, under the decode policy the model's anamnesis registry row names;
* **reading states** by replaying each sequence once through the instrumented forward, in the
  hidden-state convention :func:`kotodama.model.capture.replay_extract_koto` reads features
  in (:func:`kotodama.model.capture.hidden_rows`: the embedding, then each layer's output).

Everything else is anamnesis's own, imported rather than restated: the pinned ruler prompts,
the decode policy, the fit, the files and the build receipt. Its refusals carry over: existing
positional means are reused unless ``--refit_means`` is given, and an existing basis is never
overwritten without ``--refit_basis``.

The registry row comes from :mod:`kotodama.anamnesis_registry`, whose file is added to
``ANAMNESIS_MODELS`` before anamnesis reads its registry. The forward runs in float32 with
deterministic algorithms and TF32 off, the arithmetic of kotodama's capture lane.

    tools/run_py.sh -m kotodama.eval.anamnesis_calibrate --checkpoint CKPT --out_dir DIR
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

from kotodama import anamnesis_registry, presets

logger = logging.getLogger(__name__)


def _register() -> None:
    """Name kotodama's registry file in ANAMNESIS_MODELS (once), before anamnesis reads it."""
    current = [p for p in os.environ.get("ANAMNESIS_MODELS", "").split(os.pathsep) if p]
    path = str(anamnesis_registry.REGISTRY_FILE)
    if path not in current:
        os.environ["ANAMNESIS_MODELS"] = os.pathsep.join([*current, path])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="kotodama-tsumugi-1b", choices=sorted(anamnesis_registry.MODELS))
    p.add_argument("--checkpoint", required=True, help="A .pt/.pt.zst path or a kotodama.ckpt name")
    p.add_argument("--out_dir", type=Path, required=True, help="The calibration directory to write")
    p.add_argument("--num_prompts", type=int, default=None, help="Use only the first N ruler prompts")
    p.add_argument("--max_new_tokens", type=int, default=None, help="Default: the registry row's budget")
    p.add_argument("--suppress_eos", action="store_true",
                   help="Generate past the stop tokens to the full budget")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--refit_means", action="store_true",
                   help="Recompute positional means that already exist (invalidates signatures made against them)")
    p.add_argument("--refit_basis", action="store_true", help="Replace an existing basis")
    p.add_argument("--dry_run", action="store_true", help="Print the configuration and stop")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args(argv)
    _register()

    from anamnesis.config import ExtractionConfig, resolve_preset
    from anamnesis.extraction import calibration_fit as cf
    from anamnesis.extraction.calibration import PCA_MODEL_NAME, POSITIONAL_MEANS_NAME

    key = args.model
    shape = anamnesis_registry.MODELS[key]["shape"]
    preset = resolve_preset(key)
    try:
        ruler = cf.calibration_prompts()
    except cf.CalibrationFitError as exc:
        raise SystemExit(str(exc)) from exc
    prompts = ruler if args.num_prompts is None else ruler[: args.num_prompts]
    settings = cf.generation_settings(preset, max_new_tokens=args.max_new_tokens)
    width = settings.max_new_tokens + cf.PROMPT_HEADROOM
    n_components = ExtractionConfig.from_preset(preset).pca_components
    means_path = args.out_dir / POSITIONAL_MEANS_NAME
    basis_path = args.out_dir / PCA_MODEL_NAME

    print(f"model {key} (kotodama shape {shape}) from {args.checkpoint}")
    print(f"  prompts {len(prompts)} bare (base model)   decode: temperature {settings.temperature} "
          f"top_p {settings.top_p} tokens {settings.max_new_tokens}   stop tokens "
          f"{'suppressed' if args.suppress_eos else list(settings.eos_token_ids)}")
    print(f"  means table {width} positions   basis per layer, corrected, {n_components} components")
    print(f"  -> {means_path}  {basis_path}")
    if args.dry_run:
        return
    if basis_path.exists() and not args.refit_basis:
        raise SystemExit(f"a basis is already at {basis_path}; pass --refit_basis to replace it")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_means = cf.read_existing_means(means_path, args.refit_means)

    import torch
    from transformers import AutoTokenizer

    from kotodama.eval.model_loader import load_model
    from kotodama.model.capture import assert_lane_pinned, cached_forward, generate_light, hidden_rows

    lane = assert_lane_pinned()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta: dict = {}
    model = load_model(args.checkpoint, config_section=shape,
                       attn_res_config={"attn_res": True,
                                        "attn_res_boundaries": list(presets.DEFAULT_BOUNDARIES[shape])},
                       device=args.device, meta=meta,
                       overrides={"rope_fp32": True, "attn_impl": "sdpa", "use_liger": False}).float().eval()
    num_layers = int(model.config.num_layers)
    if num_layers != preset.num_layers:
        raise SystemExit(f"{key}'s registry row has {preset.num_layers} layers; the model has {num_layers}")
    tokenizer = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)

    # 1. sample, nothing captured
    entries = {}
    started = time.perf_counter()
    eos = None if args.suppress_eos else list(settings.eos_token_ids)
    top_p = settings.top_p if settings.top_p < 1.0 else None
    with torch.no_grad():
        for index, text in enumerate(prompts):
            prompt = cf.encode_prompt(tokenizer, text, chat_template=False)
            torch.manual_seed(index)
            out = generate_light(model, torch.tensor([prompt], device=args.device), settings.max_new_tokens,
                                 do_sample=settings.do_sample, temperature=settings.temperature, top_p=top_p,
                                 eos_token_id=eos, output_hidden_states=False)
            entries[index] = cf.entry_from_ids(out.sequences[0].tolist(), len(prompt))
            elapsed = time.perf_counter() - started
            logger.info(f"calibration tokens {index + 1}/{len(prompts)}: {entries[index].n_gen} generated, "
                        f"{elapsed:.0f}s elapsed, about {elapsed / (index + 1) * (len(prompts) - index - 1):.0f}s left")
    tokens = cf.manifest_from_entries(entries)

    # 2. replay each sequence once for its states, in the koto hidden-state convention
    def states() -> Iterator[cf.PromptStates]:
        for gen_id in tokens.gen_ids():
            entry = tokens.entry(gen_id)
            ids = torch.tensor([entry.input_ids[:-1]], device=args.device)
            with torch.no_grad():
                _, cap, _ = cached_forward(model, ids, past=None, position_offset=0, light=False,
                                           capture_hidden=True)
                stacked = torch.stack([h[0] for h in hidden_rows(cap, num_layers)]).float().cpu().numpy()
            del cap
            length = entry.prompt_length
            yield cf.PromptStates(prompt_length=length, prefill=stacked[:, :length],
                                  steps=tuple(stacked[:, p] for p in range(length, stacked.shape[1])))

    try:
        fit = cf.fit_calibration(states(), preset=preset, settings=settings, n_components=n_components,
                                 pooled=False, existing_means=existing_means, max_positions=width)
        cf.write_calibration(fit, means_path, basis_path, required_through=None, tokens=tokens)
    except cf.CalibrationFitError as exc:
        raise SystemExit(str(exc)) from exc
    receipt = cf.build_receipt(
        fit, model=key, model_id=f"{key} {args.checkpoint} step {meta.get('step')}", prompts=prompts,
        settings=settings, suppress_eos=args.suppress_eos, chat_template=False, pooled=False,
        n_components=n_components, means_path=means_path, basis_path=basis_path, required_through=None,
        tokens_written=True)
    cf.write_build_receipt(receipt, basis_path)
    logger.info(f"calibration complete: {means_path}, {basis_path}; lane {lane}")


if __name__ == "__main__":
    main()
