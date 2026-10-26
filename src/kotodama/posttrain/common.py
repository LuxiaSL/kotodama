"""Shared post-training machinery: config parsing, policy loading, optimizers/schedule, completion log-probs,
checkpoints, telemetry.

Numerics follow pretraining: the model is bf16 with fp32 masters held by the optimizers (``Muon(fp32_master=True)`` +
``MasterAdamW``) under bf16 autocast. An fp32 model under autocast computes the same forward at twice the weight
memory, so the masters live in the optimizers instead.

AttnRes routing runs in fp32, which is what the serving decode engine computes, so a post-trained model is trained
on the numerics it is served with. ``--attnres_eps_fp64_compat`` restores the fp64 routing some lineages pretrained
with; it takes effect only inside a compiled graph (inductor types the runtime eps fp64; eager Triton launches are
fp32), so it needs ``--compile``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from kotodama import presets
from kotodama.model.llama import KotodamaModel

logger = logging.getLogger(__name__)


# ── config ───────────────────────────────────────────────────────────────────


def parse_with_config(p: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """train.py's convention: a flat YAML (``--config``, with ``extends:``) supplies defaults, CLI flags win."""
    from kotodama.training.train import _load_yaml_config

    args = p.parse_args(argv)
    if getattr(args, "config", None):
        cfg = _load_yaml_config(args.config)
        known = {a.dest for a in p._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise SystemExit(f"unknown keys in {args.config}: {', '.join(unknown)}")
        p.set_defaults(**cfg)
        args = p.parse_args(argv)
    return args


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--config", default=None, help="flat YAML of these flags (extends: supported)")
    g.add_argument("--checkpoint", required=False, help=".pt/.pt.zst path or a checkpoint registry name")
    g.add_argument("--model_size", default="1b-moonshot-loop", choices=presets.names())
    g.add_argument("--attn_res_boundaries", default=None,
                   help="comma list; default = presets.DEFAULT_BOUNDARIES[model_size]")
    g.add_argument("--no_attn_res", action="store_true", help="plain residual stack (non-AttnRes checkpoints)")
    g.add_argument("--rope_fp32", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--attnres_eps_fp64_compat", action=argparse.BooleanOptionalAction, default=False,
                   help="fp64 AttnRes routing (a pretraining lineage's compiled numerics; needs --compile). Off = the "
                        "fp32 routing the serving engine computes")
    g.add_argument("--attn_impl", default="auto", choices=["auto", "fa2", "fa4", "flex", "sdpa"])
    g.add_argument("--use_liger", action=argparse.BooleanOptionalAction, default=True,
                   help="Liger kernels incl. the fused linear+CE loss (no [B,S,V] logits in SFT)")
    g.add_argument("--activation_checkpointing", action="store_true")
    g.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                   help="torch.compile the training forward (default: on for SFT's fixed-length rows, off for "
                        "preference batches whose length varies). Required by --attnres_eps_fp64_compat")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--seed", type=int, default=42)


def add_optim_args(p: argparse.ArgumentParser, muon_lr: float, adamw_lr: float, warmup_ratio: float,
                   decay_start_pct: float, decay_type: str) -> None:
    g = p.add_argument_group("optimizer (Muon hidden matrices + AdamW the rest; WSD schedule)")
    g.add_argument("--optimizer", default="hybrid", choices=["hybrid", "adamw"],
                   help="hybrid = Muon + AdamW as pretraining; adamw = AdamW (fp32 masters) on every parameter, "
                        "at --adamw_lr")
    g.add_argument("--muon_lr", type=float, default=muon_lr)
    g.add_argument("--muon_momentum", type=float, default=0.95)
    g.add_argument("--muon_weight_decay", type=float, default=0.01)
    g.add_argument("--muon_ns_iterations", type=int, default=5)
    g.add_argument("--muon_ns_coefficients", default="gram_ns")
    g.add_argument("--adamw_lr", type=float, default=adamw_lr)
    g.add_argument("--adamw_beta1", type=float, default=0.9)
    g.add_argument("--adamw_beta2", type=float, default=0.95)
    g.add_argument("--adamw_weight_decay", type=float, default=0.1)
    g.add_argument("--adamw_no_decay_norms", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--gradient_clip", type=float, default=1.0)
    g.add_argument("--warmup_ratio", type=float, default=warmup_ratio)
    g.add_argument("--decay_start_pct", type=float, default=decay_start_pct)
    g.add_argument("--decay_type", default=decay_type, choices=["linear", "cosine", "sqrt"])


# ── model ────────────────────────────────────────────────────────────────────


def boundaries(args: argparse.Namespace) -> Optional[list[int]]:
    if args.no_attn_res:
        return None
    if args.attn_res_boundaries:
        spec = args.attn_res_boundaries
        return [int(x) for x in (spec.split(",") if isinstance(spec, str) else spec)]
    key = presets.resolve(args.model_size)
    if key not in presets.DEFAULT_BOUNDARIES:
        raise SystemExit(f"no default AttnRes boundaries for {key}; pass --attn_res_boundaries or --no_attn_res")
    return list(presets.DEFAULT_BOUNDARIES[key])


def load_policy(args: argparse.Namespace, checkpoint: Optional[str] = None) -> KotodamaModel:
    """The model in train mode on ``args.device`` (bf16 on CUDA, fp32 on CPU), routing numerics applied."""
    from kotodama.eval.model_loader import load_model
    from kotodama.training.train import apply_attnres_eps_compat

    if apply_attnres_eps_compat(args):
        logger.info("AttnRes routing: fp64 eps compat ON (takes effect in a compiled forward only)")
    bnd = boundaries(args)
    attn_cfg = {"attn_res": False} if bnd is None else {"attn_res": True, "attn_res_boundaries": bnd}
    on_cuda = str(args.device).startswith("cuda")
    model = load_model(
        checkpoint or args.checkpoint, config_section=args.model_size, attn_res_config=attn_cfg,
        device=args.device,
        overrides={"rope_fp32": bool(args.rope_fp32), "attn_impl": args.attn_impl,
                   "use_liger": bool(args.use_liger and on_cuda),
                   "activation_checkpointing": bool(args.activation_checkpointing)})
    model.train()
    return model


def compiled(model: KotodamaModel, args: argparse.Namespace, dynamic_lengths: bool = False) -> torch.nn.Module:
    """The forward to train with: torch.compile'd on CUDA when ``--compile`` (attribute access passes through to
    the model). ``dynamic_lengths``: callers whose sequence length varies per batch (preference batches, padded to
    ``--pad_multiple``) get a dynamo cache large enough for one graph per length bucket instead of silently falling
    back to eager after the default 8."""
    on_cuda = str(args.device).startswith("cuda")
    if not (args.compile and on_cuda):
        if on_cuda and args.attnres_eps_fp64_compat and not args.no_attn_res:
            logger.warning("--attnres_eps_fp64_compat without --compile has no effect: eager routing runs fp32")
        return model
    from torch import _dynamo

    if dynamic_lengths:
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
    flce = getattr(model, "fused_linear_ce_loss", None)
    if flce is not None and not getattr(flce.forward, "_torchdynamo_disable", False):
        # The fused linear + cross-entropy runs eagerly between compiled trunk graphs: dynamo breaks inside it (an
        # .item()) and the graph it compiles for the rest of the op faults with an illegal memory access on packed
        # SFT batches. Opaque to dynamo, it is the same kernel pretraining runs.
        flce.forward = _dynamo.disable(flce.forward)
    logger.info("compiling the forward (first step of each new shape compiles; fused loss eager)")
    return torch.compile(model)


# ── optimizers / schedule ────────────────────────────────────────────────────


def build_optim(model: torch.nn.Module, args: argparse.Namespace, total_steps: int) -> tuple[Any, Any, Any]:
    """(muon, adamw, scheduler) — the trunk's builders with fp32 masters on both optimizers. ``--optimizer adamw``
    puts every parameter in one MasterAdamW (norm gains without decay when --adamw_no_decay_norms) and gives Muon a
    no-op stand-in, so the step loop is the same either way."""
    from kotodama.training.muon import HybridScheduler, MasterAdamW, build_hybrid_optimizer

    if getattr(args, "optimizer", "hybrid") == "adamw":
        from kotodama.training.train import _NoOpOptimizer

        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        norms = [p for n, p in params if args.adamw_no_decay_norms and p.ndim == 1 and n.endswith("norm.weight")]
        norm_ids = {id(p) for p in norms}
        groups: list[dict[str, Any]] = [{"params": [p for _, p in params if id(p) not in norm_ids]}]
        if norms:
            groups.append({"params": norms, "weight_decay": 0.0})
        adamw = MasterAdamW(groups, lr=args.adamw_lr, betas=(args.adamw_beta1, args.adamw_beta2),
                            weight_decay=args.adamw_weight_decay)
        muon = _NoOpOptimizer()
        sched = HybridScheduler(muon, adamw, warmup_steps=int(total_steps * args.warmup_ratio),
                                total_steps=total_steps, decay_start_pct=args.decay_start_pct,
                                decay_type=args.decay_type)
        sched.step(0)
        return muon, adamw, sched
    muon, adamw = build_hybrid_optimizer(
        model, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum, muon_weight_decay=args.muon_weight_decay,
        muon_ns_iterations=args.muon_ns_iterations, muon_ns_coefficients=args.muon_ns_coefficients,
        adamw_lr=args.adamw_lr, adamw_betas=(args.adamw_beta1, args.adamw_beta2),
        adamw_weight_decay=args.adamw_weight_decay, adamw_no_decay_norms=args.adamw_no_decay_norms,
        muon_fp32_master=True)
    sched = HybridScheduler(muon, adamw, warmup_steps=int(total_steps * args.warmup_ratio), total_steps=total_steps,
                            decay_start_pct=args.decay_start_pct, decay_type=args.decay_type)
    sched.step(0)
    return muon, adamw, sched


def optimizer_step(model: torch.nn.Module, muon: Any, adamw: Any, sched: Any, step: int, clip: float) -> float:
    """Clip, step both optimizers, zero grads, set the LR for the next step. Returns the pre-clip grad norm."""
    params = [p for p in model.parameters() if p.grad is not None]
    gn = torch.nn.utils.clip_grad_norm_(params, clip) if clip > 0 else torch.linalg.vector_norm(
        torch.stack([p.grad.detach().float().norm() for p in params]))
    muon.step()
    adamw.step()
    muon.zero_grad(set_to_none=True)
    adamw.zero_grad(set_to_none=True)
    sched.step(step + 1)
    return float(gn)


# ── log-probs ────────────────────────────────────────────────────────────────


def completion_logps(model: KotodamaModel, ids: torch.Tensor, prompt_lens: list[int], lens: list[int],
                     chunk: int = 4096) -> torch.Tensor:
    """Per-row sum of log p(token) over completion tokens (positions prompt_len .. len-1), fp32.

    The LM head + fp32 log-softmax run only on the rows that predict completion tokens — the same per-sequence math
    as full logits, without the [B, T, V] tensor."""
    B, T = ids.shape
    h = model(ids, return_hidden=True)["hidden"]
    rows, seg = [], []
    for b, (pl, n) in enumerate(zip(prompt_lens, lens)):
        if not 1 <= pl < n <= T:
            raise ValueError(f"row {b}: need 1 <= prompt_len ({pl}) < len ({n}) <= {T}")
        rows.append(b * T + torch.arange(pl - 1, n - 1))
        seg.append(torch.full((n - pl,), b, dtype=torch.long))
    rows_t = torch.cat(rows).to(ids.device)
    seg_t = torch.cat(seg).to(ids.device)
    tgt = ids.reshape(-1)[rows_t + 1]
    hs = h.reshape(-1, h.shape[-1]).index_select(0, rows_t)
    w = model.get_lm_head_weight()
    parts = []
    for c0 in range(0, hs.shape[0], chunk):
        lsm = F.log_softmax(F.linear(hs[c0:c0 + chunk], w).float(), dim=-1)
        parts.append(lsm.gather(-1, tgt[c0:c0 + chunk].unsqueeze(-1)).squeeze(-1))
    out = torch.zeros(B, device=ids.device, dtype=torch.float32)
    return out.index_add_(0, seg_t, torch.cat(parts))


def pad_batch(seqs: list[list[int]], pad_id: int, multiple: int = 128) -> torch.Tensor:
    """Right-padded (B, T) ids with T rounded up to ``multiple`` (bounds the distinct shapes Triton kernels JIT for;
    trailing pads are causally invisible, so the math is unchanged)."""
    T = max(len(s) for s in seqs)
    if multiple > 1:
        T = -(-T // multiple) * multiple
    out = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ── checkpoints / telemetry / signals ────────────────────────────────────────


def save_weights(path: Path, model: torch.nn.Module, step: int, meta: dict[str, Any]) -> Path:
    """Weights-only checkpoint ``{"step", "model", "posttrain"}`` — loadable by load_model / the server."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"step": step, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "posttrain": meta}, tmp)
    os.replace(tmp, path)
    logger.info("saved %s", path)
    return path


class Telemetry:
    """JSONL log + the scheduler's metrics API (hm_log; a no-op outside a scheduler job)."""

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / "train_log.jsonl"
        self._fh = self.path.open("a")

    def log(self, step: int, **metrics: Any) -> None:
        rec = {"step": step, "time": round(time.time(), 1), **metrics}
        try:
            self._fh.write(json.dumps(rec, default=float) + "\n")
            self._fh.flush()
        except OSError as e:
            logger.warning("train_log write failed: %s", e)
        try:
            from kotodama.training.train import hm_log

            hm_log(step, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        except Exception as e:  # telemetry must never stop a run
            logger.debug("hm_log skipped: %s", e)

    def close(self) -> None:
        try:
            from kotodama.training.train import hm_flush

            hm_flush()
        except Exception:
            pass
        self._fh.close()


class SigtermFlag:
    def __init__(self) -> None:
        self.received = False
        self._orig = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: Any) -> None:
        self.received = True
        logger.warning("SIGTERM — saving weights and exiting at the next step boundary")

    def restore(self) -> None:
        signal.signal(signal.SIGTERM, self._orig)


def autocast(device: str):
    """bf16 autocast on CUDA; a no-op context on CPU (tests)."""
    import contextlib

    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
