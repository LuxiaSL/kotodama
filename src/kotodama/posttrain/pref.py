"""Preference post-training: DPO / DPOP / IPO on pairs, KTO on unpaired good/bad completions.

    python -m kotodama.posttrain.pref --config configs/posttrain/tsumugi-kto.yaml \\
        --checkpoint <policy-init.pt> --data <kto.jsonl> --out_dir <dir>

Data (JSONL). The prompt is one of ``prompt_chatml`` (rendered ChatML ending in the open assistant tag),
``messages`` (rendered here) or ``prompt`` (raw base-model text). Chat completions get ``<|im_end|>\\n`` appended (the
turn's close is part of what is scored); raw completions are scored as written, plus ``<|endoftext|>`` when the row says ``"eos": true``.
  pairs (dpo|dpop|ipo):  {<prompt>, "chosen": str, "rejected": str[, "weight"]}
  kto:                   {<prompt>, "completion": str, "label": bool[, "weight"]}   (selfcopy.py label writes these)

``dpo_ln`` is length-normalized DPO: log-probs divided by completion length before the DPO loss (per-token
currency; pair it with a larger β, e.g. 5).

Reference = the init (or ``--ref_checkpoint``), log-probs precomputed once and cached on disk keyed by (checkpoint
file, data file sha, max_len); cache hits are re-verified on a sample. Sequence log-probs are sums over completion
tokens, LM head applied only at completion positions (common.completion_logps).

KTO's reference point z0 = clamp(mean(π − ref), 0) over MISMATCHED (prompt_i, completion_{i+1}) pairs within each
micro-batch (the paper's estimator; ``--kto_kl batch`` uses the matched pairs instead, ``zero`` sets z0 = 0).

More than one epoch is refused: preference training past one effective epoch overfits a model's style and
collapses its range. Success is judged by evaluations of the trained model (binding batteries, loop and early-stop
rates at the serving sampling law, a human read), not by this loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import torch

from kotodama.posttrain import common, losses
from kotodama.posttrain.chat_data import prompt_text
from kotodama.presets import TOKENIZER_NAME
from kotodama.serve.chatml import EOS_TOKEN_ID

logger = logging.getLogger("kotodama.posttrain.pref")

OBJECTIVES = (*losses.PAIRWISE, "dpo_ln", "kto")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="kotodama preference post-training (DPO family + KTO)")
    common.add_model_args(p)
    common.add_optim_args(p, muon_lr=5e-5, adamw_lr=5e-6, warmup_ratio=0.1, decay_start_pct=0.7,
                          decay_type="linear")
    g = p.add_argument_group("objective / data / loop")
    g.add_argument("--objective", default="dpo", choices=OBJECTIVES)
    g.add_argument("--data", default=None, help="pairs JSONL (DPO family) or labelled JSONL (KTO)")
    g.add_argument("--beta", type=float, default=0.1)
    g.add_argument("--dpop_lambda", type=float, default=50.0)
    g.add_argument("--kto_lambda_d", type=float, default=1.0)
    g.add_argument("--kto_lambda_u", type=float, default=1.0)
    g.add_argument("--kto_kl", default="mismatched", choices=["mismatched", "batch", "zero"])
    g.add_argument("--ref_checkpoint", default=None, help="reference model (default: the init checkpoint)")
    g.add_argument("--tokenizer", default=TOKENIZER_NAME)
    g.add_argument("--max_len", type=int, default=3584)
    g.add_argument("--batch", type=int, default=4, help="pairs (or KTO items) per micro-batch")
    g.add_argument("--grad_accum", type=int, default=4)
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--max_steps", type=int, default=0)
    g.add_argument("--limit", type=int, default=0, help="smoke: cap the number of rows")
    g.add_argument("--pad_multiple", type=int, default=128)
    g.add_argument("--logp_chunk", type=int, default=4096)
    g.add_argument("--ref_cache", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--ref_cache_dir", default=None, help="default: <out_dir>/refcache")
    g.add_argument("--ref_verify_n", type=int, default=16)
    g.add_argument("--ref_tol", type=float, default=0.05)
    g.add_argument("--saves", type=int, default=5, help="intermediate weight saves over the run (a trajectory)")
    g.add_argument("--out_dir", default=None)
    a = common.parse_with_config(p, argv)
    if a.compile is None:
        a.compile = False
    for k in ("checkpoint", "data", "out_dir"):
        if not getattr(a, k):
            p.error(f"--{k} is required (flag or config)")
    if a.epochs > 1.0:
        raise SystemExit("refusing > 1 epoch: preference training past one effective epoch overfits style")
    return a


# ── encoding ─────────────────────────────────────────────────────────────────


def save_steps(n_steps: int, saves: int) -> set[int]:
    """Steps that get an intermediate weight save: at most ``saves`` evenly spaced steps before the last (the final
    save is separate). ``saves`` 0 keeps finals only."""
    if saves <= 0 or n_steps < 2:
        return set()
    return {round(n_steps * k / (saves + 1)) for k in range(1, saves + 1)} - {0, n_steps}


def encode_side(tok: Any, prompt: str, completion: str, chat: bool, eos: bool) -> dict[str, Any]:
    pid = tok.encode(prompt, add_special_tokens=False)
    cid = tok.encode(completion + ("<|im_end|>\n" if chat else ""), add_special_tokens=False)
    if eos and not chat:
        cid = cid + [EOS_TOKEN_ID]
    return {"ids": pid + cid, "prompt_len": len(pid)}


def encode_rows(rows: list[dict[str, Any]], tok: Any, objective: str, max_len: int) -> list[dict[str, Any]]:
    out = []
    skipped = 0
    for r in rows:
        try:
            prompt, chat = prompt_text(r)
        except KeyError:
            skipped += 1
            continue
        eos = bool(r.get("eos", False))
        rec: dict[str, Any] = {"weight": float(r.get("weight", 1.0)), "prompt": prompt, "chat": chat, "eos": eos}
        if objective == "kto":
            if not isinstance(r.get("completion"), str) or not isinstance(r.get("label"), bool):
                skipped += 1
                continue
            rec["side"] = encode_side(tok, prompt, r["completion"], chat, eos)
            rec["label"] = r["label"]
            rec["completion"] = r["completion"]
            sides = [rec["side"]]
        else:
            if not isinstance(r.get("chosen"), str) or not isinstance(r.get("rejected"), str):
                skipped += 1
                continue
            rec["chosen"] = encode_side(tok, prompt, r["chosen"], chat, eos)
            rec["rejected"] = encode_side(tok, prompt, r["rejected"], chat, eos)
            sides = [rec["chosen"], rec["rejected"]]
        if any(len(s["ids"]) > max_len or len(s["ids"]) <= s["prompt_len"] for s in sides):
            skipped += 1
            continue
        out.append(rec)
    logger.info("rows: %d usable, %d skipped (malformed / empty completion / > max_len %d)", len(out), skipped,
                max_len)
    return out


def make_micros(recs: list[dict[str, Any]], bs: int, seed: int, objective: str) -> list[list[int]]:
    """Length-bucketed micro-batches in seeded order, full ones first (padding stays tight; a step cap consumes full
    micro-batches)."""
    def length(i: int) -> int:
        r = recs[i]
        return len(r["side"]["ids"]) if objective == "kto" else max(len(r["chosen"]["ids"]), len(r["rejected"]["ids"]))

    by_len = sorted(range(len(recs)), key=length)
    micros = [by_len[i:i + bs] for i in range(0, len(by_len), bs)]
    random.Random(seed).shuffle(micros)
    return [m for m in micros if len(m) == bs] + [m for m in micros if len(m) != bs]


def mismatched_sides(tok: Any, recs: list[dict[str, Any]], micro: list[int], max_len: int) -> list[dict[str, Any]]:
    """(prompt_i, completion_{i+1}) within a micro — KTO's KL-estimate sequences (completion cut to fit max_len)."""
    n = len(micro)
    if n < 2:
        return []
    out = []
    for k in range(n):
        s = encode_side(tok, recs[micro[k]]["prompt"], recs[micro[(k + 1) % n]]["completion"],
                        recs[micro[k]]["chat"], recs[micro[(k + 1) % n]]["eos"])
        s["ids"] = s["ids"][:max(max_len, s["prompt_len"] + 1)]
        out.append(s)
    return out


def micro_sides(tok: Any, recs: list[dict[str, Any]], micro: list[int], objective: str,
                kto_kl: str, max_len: int) -> list[dict[str, Any]]:
    """The sequences one micro-batch scores: KTO items (+ their mismatched KL sequences), or chosen then rejected."""
    if objective == "kto":
        sides = [recs[k]["side"] for k in micro]
        if kto_kl == "mismatched":
            sides = sides + mismatched_sides(tok, recs, micro, max_len)
        return sides
    return [recs[k]["chosen"] for k in micro] + [recs[k]["rejected"] for k in micro]


# ── reference cache ──────────────────────────────────────────────────────────


def side_key(side: dict[str, Any]) -> str:
    return hashlib.sha1((f"{side['prompt_len']}|" + ",".join(map(str, side["ids"]))).encode()).hexdigest()


def _file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


class RefCache:
    """Reference log-probs per encoded side, in a file keyed by (checkpoint file, data sha, max_len)."""

    def __init__(self, cache_dir: Path, ckpt: str, data: str, max_len: int) -> None:
        from kotodama import ckpt as ckpt_mod

        path = ckpt_mod.resolve(ckpt)
        st = os.stat(path)
        fp = f"{Path(path).resolve()}|{st.st_mtime_ns}|{st.st_size}"
        ck = hashlib.sha256(fp.encode()).hexdigest()[:16]
        self.path = cache_dir / f"ref_{ck}_{_file_sha(data)}_{max_len}.json"
        self.meta = {"checkpoint": fp, "data": str(Path(data).resolve()), "max_len": max_len}
        self.entries: dict[str, float] = {}
        if self.path.exists():
            try:
                self.entries = {k: float(v) for k, v in json.loads(self.path.read_text())["entries"].items()}
                logger.info("ref cache hit: %s (%d sides)", self.path, len(self.entries))
            except (OSError, ValueError, KeyError) as e:
                logger.warning("ref cache unreadable (%s) — recomputing", e)

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"meta": self.meta, "entries": self.entries}))
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning("ref cache save failed (non-fatal): %s", e)


def logps(model: torch.nn.Module, sides: list[dict[str, Any]], pad_id: int, a: argparse.Namespace) -> torch.Tensor:
    ids = common.pad_batch([s["ids"] for s in sides], pad_id, a.pad_multiple).to(a.device)
    return common.completion_logps(model, ids, [s["prompt_len"] for s in sides], [len(s["ids"]) for s in sides],
                                   a.logp_chunk)


def reference_pass(model: torch.nn.Module, batches: list[list[dict[str, Any]]], pad_id: int,
                   a: argparse.Namespace, cache: Optional[RefCache]) -> None:
    """Fill side["ref_lp"] for every side (cache first), then verify a sample of cache hits against fresh values."""
    hits = []
    for sides in batches:
        for s in sides:
            if cache is not None and side_key(s) in cache.entries:
                s["ref_lp"] = cache.entries[side_key(s)]
                hits.append(s)
    model.eval()
    n_new = 0
    with torch.no_grad(), common.autocast(a.device):
        for sides in batches:
            todo = [s for s in sides if "ref_lp" not in s]
            if not todo:
                continue
            for s, v in zip(todo, logps(model, todo, pad_id, a).tolist()):
                s["ref_lp"] = float(v)
                n_new += 1
                if cache is not None:
                    cache.entries[side_key(s)] = s["ref_lp"]
        if hits and a.ref_verify_n > 0:
            sample = random.Random(a.seed).sample(hits, min(a.ref_verify_n, len(hits)))
            fresh = logps(model, sample, pad_id, a).tolist()
            dmax = max(abs(f - s["ref_lp"]) for f, s in zip(fresh, sample))
            logger.info("ref cache verify: %d sides, max |cached - fresh| = %.2e (tol %.1e)", len(sample), dmax,
                        a.ref_tol)
            if dmax > a.ref_tol:
                raise SystemExit(f"ref cache verification failed ({dmax:.3e} > {a.ref_tol}); delete "
                                 f"{cache.path if cache else ''} or run with --no-ref_cache")
    if cache is not None and n_new:
        cache.save()
    logger.info("reference pass: %d computed, %d from cache", n_new, len(hits))


# ── loss over one micro ──────────────────────────────────────────────────────


def micro_loss(lp: torch.Tensor, sides: list[dict[str, Any]], recs: list[dict[str, Any]], micro: list[int],
               a: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    dev = lp.device
    ref = torch.tensor([s["ref_lp"] for s in sides], device=dev, dtype=lp.dtype)
    w = torch.tensor([recs[k]["weight"] for k in micro], device=dev, dtype=lp.dtype)
    n = len(micro)
    if a.objective == "kto":
        des = torch.tensor([recs[k]["label"] for k in micro], device=dev)
        lp_m, ref_m = lp[:n], ref[:n]
        if a.kto_kl == "mismatched" and lp.shape[0] > n:
            kl = losses.kto_kl_estimate(lp[n:], ref[n:])
        elif a.kto_kl == "batch":
            kl = losses.kto_kl_estimate(lp_m, ref_m)
        else:
            kl = torch.zeros((), device=dev)
        loss, reward = losses.kto_loss(lp_m, ref_m, des, kl, a.beta, a.kto_lambda_d, a.kto_lambda_u, w)
        reward = reward.detach()
        agree = torch.where(des, reward > 0, reward < 0).float().mean()
        stats = {"kl": float(kl), "reward_agree": float(agree)}
        if des.any():
            stats["reward_desirable"] = float(reward[des].mean())
        if (~des).any():
            stats["reward_undesirable"] = float(reward[~des].mean())
        return loss, stats
    if a.objective == "dpo_ln":
        # length-normalized DPO: per-token log-probs (sums over completion length), so a longer answer is not
        # preferred or penalized for its length alone; margins are in per-token units
        lens = torch.tensor([len(s["ids"]) - s["prompt_len"] for s in sides], device=dev, dtype=lp.dtype)
        lp, ref = lp / lens, ref / lens
    loss, margin = losses.pairwise_loss(lp[:n], lp[n:], ref[:n], ref[n:], a.beta, w,
                                        "dpo" if a.objective == "dpo_ln" else a.objective, a.dpop_lambda)
    margin = margin.detach()
    return loss, {"pref_acc": float((margin > 0).float().mean()), "margin": float(margin.mean())}


def corpus_eval(model: torch.nn.Module, recs: list[dict[str, Any]], micros: list[list[int]],
                batches: list[list[dict[str, Any]]], pad_id: int, a: argparse.Namespace) -> dict[str, float]:
    """Order-independent outcome at the current weights vs the reference, over every row."""
    was = model.training
    model.eval()
    agg: dict[str, float] = {}
    n_tot = 0
    with torch.no_grad(), common.autocast(a.device):
        for m, sides in zip(micros, batches):
            loss, st = micro_loss(logps(model, sides, pad_id, a), sides, recs, m, a)
            for k, v in {"loss": float(loss), **st}.items():
                agg[k] = agg.get(k, 0.0) + v * len(m)
            n_tot += len(m)
    model.train(was)
    return {f"final/{k}": v / max(n_tot, 1) for k, v in agg.items()}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [pref] %(message)s", datefmt="%H:%M:%S")
    a = parse_args(argv)
    torch.manual_seed(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows = []
    with open(a.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if a.limit:
        rows = rows[:a.limit]
    recs = encode_rows(rows, tok, a.objective, a.max_len)
    if not recs:
        raise SystemExit(f"no usable rows in {a.data}")
    if a.objective == "kto":
        n_d = sum(r["label"] for r in recs)
        bal = losses.kto_balance(n_d, len(recs) - n_d, a.kto_lambda_d, a.kto_lambda_u)
        logger.info("KTO: %d desirable / %d undesirable; lambda_D n_D / lambda_U n_U = %.2f", n_d, len(recs) - n_d, bal)
        if not 1.0 <= bal <= 4 / 3:
            logger.warning("KTO balance %.2f outside the paper's [1, 4/3] — adjust --kto_lambda_d/u", bal)
    micros = make_micros(recs, a.batch, a.seed, a.objective)
    batches = [micro_sides(tok, recs, m, a.objective, a.kto_kl, a.max_len) for m in micros]

    cache = RefCache(Path(a.ref_cache_dir or out_dir / "refcache"), a.ref_checkpoint or a.checkpoint, a.data,
                     a.max_len) if a.ref_cache else None
    if a.ref_checkpoint and a.ref_checkpoint != a.checkpoint:
        ref_model = common.load_policy(a, a.ref_checkpoint)
        reference_pass(common.compiled(ref_model, a, dynamic_lengths=True), batches, pad_id, a, cache)
        del ref_model
        if str(a.device).startswith("cuda"):
            torch.cuda.empty_cache()
        model = common.load_policy(a)
        fwd = common.compiled(model, a, dynamic_lengths=True)
    else:
        model = common.load_policy(a)
        fwd = common.compiled(model, a, dynamic_lengths=True)
        reference_pass(fwd, batches, pad_id, a, cache)

    n_steps = a.max_steps or max(1, int(len(micros) * a.epochs / a.grad_accum))
    muon, adamw, sched = common.build_optim(model, a, n_steps)
    tel = common.Telemetry(out_dir)
    sig = common.SigtermFlag()
    meta = {"engine": "kotodama.posttrain.pref", "objective": a.objective, "beta": a.beta,
            "base": str(a.checkpoint), "data": str(a.data)}
    ckpt_dir = out_dir / "checkpoints"
    save_at = save_steps(n_steps, a.saves)
    logger.info("%s: %d rows, %d micros, %d steps (beta %.3g)", a.objective, len(recs), len(micros), n_steps, a.beta)

    model.train()
    step = 0
    acc: dict[str, float] = {}
    t0 = time.perf_counter()
    for mi, m in enumerate(micros):
        if step >= n_steps:
            break
        sides = batches[mi]
        with common.autocast(a.device):
            lp = logps(fwd, sides, pad_id, a)
        loss, st = micro_loss(lp, sides, recs, m, a)
        (loss / a.grad_accum).backward()
        for k, v in {"loss": float(loss.detach()), **st}.items():
            acc[k] = acc.get(k, 0.0) + v / a.grad_accum
        if (mi + 1) % a.grad_accum:
            continue
        gn = common.optimizer_step(model, muon, adamw, sched, step, a.gradient_clip)
        step += 1
        lrs = sched.get_last_lr()
        rec = {**acc, "grad_norm": gn, "muon_lr": lrs["muon_lr"], "adamw_lr": lrs["adamw_lr"],
               "step_s": time.perf_counter() - t0}
        t0 = time.perf_counter()
        logger.info("step %d/%d %s", step, n_steps, " ".join(f"{k} {v:.4g}" for k, v in rec.items()))
        tel.log(step, **rec)
        acc = {}
        if not math.isfinite(rec["loss"]):
            raise SystemExit(f"non-finite loss at step {step}")
        if step in save_at:
            common.save_weights(ckpt_dir / f"step_{step:08d}.pt", model, step, meta)
        if sig.received:
            break

    final = common.save_weights(ckpt_dir / f"step_{step:08d}.pt", model, step, meta)
    ev = corpus_eval(fwd, recs, micros, batches, pad_id, a)
    logger.info("final corpus eval: %s", ev)
    tel.log(step, **ev)
    tel.close()
    sig.restore()
    (out_dir / "final.json").write_text(json.dumps({"checkpoint": str(final), "steps": step, **ev}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
