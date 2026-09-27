"""In-domain NCA gate scorer — ledger + smoothing, on any koto checkpoint.

Port of the frozen ``probes/p07/gate_score.py`` onto the kotodama package
(SPEC-1B-FORMATION-LADDER P0.6): model loading via
``kotodama.eval.model_loader`` (presets + ckpt registry + AttnRes handling),
the v3 eval-set layout (``eval_ledger_unique`` / ``eval_ledger_overwrite``,
plus the legacy ``eval_ledger``), the v3 labels (post_overwrite /
writes_before / marked) as reporting splits, no hard-coded paths, CPU-capable.

Ledger: next-token logprob over the VAL alphabet at the query answer
position; acc / median rank / mean gold logprob, split by committer side
(bind_self vs bind_ctx in miniature), lag, first-vs-re-query, and
unique-vs-post-overwrite (the ramp ruling's two regimes).

Smoothing: teacher-forced mean logprob of the true (perturbation-
incorporating) next frame vs the counterfactual unperturbed frame;
SI_nca = softmax share on the counterfactual (0.5 = indifferent, high =
the attractor rides over the event).

Usage::

  tools/run_py.sh -m kotodama.eval.nca_gates \
      --checkpoint <path-or-registry-name> --config-section proxy \
      --eval-dir <dir with eval_*.jsonl + meta.json> --tag p05-arm1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from kotodama.nca.dyadic import N_VALS, token_map
from kotodama.nca.generator import NCAConfig

logger = logging.getLogger(__name__)

# A "model" here is anything with the LuxiaBaseModel calling convention:
# model(input_ids) -> {"logits": (B, S, V)}.
ModelFn = Callable[[torch.Tensor], dict[str, torch.Tensor]]


def resolve_token_map(eval_dir: Path | str,
                      reserve_eos: str = "auto") -> dict[str, int]:
    """The token layout the eval set was emitted under.

    ``auto`` reads ``meta.json`` (the v3 writer records ``reserve_eos``);
    a missing meta means the legacy v1 layout. ``true``/``false`` override.
    """
    if reserve_eos == "auto":
        meta = Path(eval_dir) / "meta.json"
        flag = (json.loads(meta.read_text()).get("reserve_eos", False)
                if meta.exists() else False)
    else:
        flag = reserve_eos == "true"
    return token_map(NCAConfig(reserve_eos=flag))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ── Ledger ────────────────────────────────────────────────────────────────────

def score_ledger(model: ModelFn, items: list[dict[str, Any]],
                 tm: dict[str, int],
                 device: str = "cpu") -> list[dict[str, Any]]:
    """Per-item next-token read at the query answer position."""
    val0 = tm["VAL0"]
    rows = []
    for it in items:
        ids = torch.tensor([it["prefix"]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(ids)["logits"][0, -1].float()
        lps = F.log_softmax(logits, -1)
        cand = lps[val0: val0 + N_VALS]
        gold = it["gold_val"]
        rank = 1 + int((cand > cand[gold]).sum())
        rows.append({
            "item_id": it.get("item_id"),
            "acc": int(rank == 1),
            "rank": rank,
            "gold_lp": float(cand[gold]),
            "side": it.get("committer_side"),
            "lag": it.get("lag_steps"),
            "requery": it.get("is_requery"),
            "post_overwrite": it.get("post_overwrite"),
            "writes_before": it.get("writes_before"),
            "marked": it.get("marked"),
        })
    return rows


def _agg(sub: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sub:
        return None
    return {"acc": float(np.mean([r["acc"] for r in sub])),
            "med_rank": int(np.median([r["rank"] for r in sub])),
            "mean_gold_lp": float(np.mean([r["gold_lp"] for r in sub])),
            "n": len(sub)}


def aggregate_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The reporting splits (each is measurement, chance acc = 1/N_VALS)."""
    out: dict[str, Any] = {"all": _agg(rows), "chance_acc": 1.0 / N_VALS}
    out["self_committed"] = _agg([r for r in rows if r["side"] == 0])
    out["other_committed"] = _agg([r for r in rows if r["side"] == 1])
    out["lag_le8"] = _agg([r for r in rows if (r["lag"] or 0) <= 8])
    out["lag_gt8"] = _agg([r for r in rows if (r["lag"] or 0) > 8])
    out["first_query"] = _agg([r for r in rows if r["requery"] is False])
    out["re_query"] = _agg([r for r in rows if r["requery"] is True])
    out["unique_write"] = _agg([r for r in rows
                                if r["post_overwrite"] is False])
    out["post_overwrite"] = _agg([r for r in rows
                                  if r["post_overwrite"] is True])
    return out


# ── Smoothing ─────────────────────────────────────────────────────────────────

def score_smoothing(model: ModelFn, pairs: list[dict[str, Any]],
                    device: str = "cpu") -> list[dict[str, Any]]:
    """SI_nca per pair: softmax share on the counterfactual continuation."""
    out = []
    for it in pairs:
        pre = it["prefix"]

        def frame_lp(frame: list[int]) -> float:
            ids = torch.tensor([pre + frame], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(ids)["logits"][0].float()
            lps = F.log_softmax(logits, -1)
            tgt = torch.tensor(frame, device=device)
            pos = torch.arange(len(pre) - 1, len(pre) - 1 + len(frame),
                               device=device)
            return float(lps[pos, tgt].mean())

        lt = frame_lp(it["true_frame"])
        lc = frame_lp(it["counterfactual_frame"])
        m = max(lt, lc)
        si = float(np.exp(lc - m) / (np.exp(lt - m) + np.exp(lc - m)))
        out.append({"item_id": it.get("item_id"), "si": si,
                    "lp_true": lt, "lp_counterfactual": lc})
    return out


def aggregate_smoothing(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    sis = [r["si"] for r in rows]
    return {"si_mean": float(np.mean(sis)),
            "si_se": float(np.std(sis) / max(len(sis), 1) ** 0.5),
            "n": len(sis)}


# ── CLI ───────────────────────────────────────────────────────────────────────

LEDGER_FILES = ("eval_ledger_unique.jsonl", "eval_ledger_overwrite.jsonl",
                "eval_ledger.jsonl")


def _print_ledger(tag: str, name: str, agg: dict[str, Any]) -> None:
    def fmt(a):
        return (f"acc {a['acc']:.3f} medrank {a['med_rank']} "
                f"gold_lp {a['mean_gold_lp']:.2f} (n={a['n']})"
                if a else "—")
    print(f"[{tag}] LEDGER {name}  all: {fmt(agg['all'])}  "
          f"chance acc={agg['chance_acc']:.3f}")
    for key, label in (("self_committed", "self-committed"),
                       ("other_committed", "other-committed"),
                       ("lag_le8", "lag<=8"), ("lag_gt8", "lag>8"),
                       ("first_query", "first-query"),
                       ("re_query", "re-query"),
                       ("unique_write", "unique-write"),
                       ("post_overwrite", "post-overwrite")):
        if agg[key]:
            print(f"[{tag}]   {label}: {fmt(agg[key])}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="NCA in-domain gates")
    ap.add_argument("--checkpoint", required=True,
                    help="path or kotodama.ckpt registry name")
    ap.add_argument("--config-section", default="proxy")
    ap.add_argument("--config-path", default=None)
    ap.add_argument("--attn-res", choices=["auto", "on", "off"],
                    default="auto")
    ap.add_argument("--attn-res-boundaries", default=None,
                    help="comma-separated layer indices")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reserve-eos", choices=["auto", "true", "false"],
                    default="auto")
    ap.add_argument("--device", default=None,
                    help="default: cuda if available else cpu")
    ap.add_argument("--out", default=None,
                    help="default: <eval-dir>/gates_<tag>.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    attn_res_config = None
    if args.attn_res == "off":
        attn_res_config = {"attn_res": False}
    elif args.attn_res == "on" or args.attn_res_boundaries:
        attn_res_config = {"attn_res": True}
        if args.attn_res_boundaries:
            attn_res_config["attn_res_boundaries"] = [
                int(x) for x in args.attn_res_boundaries.split(",")]

    from kotodama.eval.model_loader import load_model
    model = load_model(args.checkpoint, config_path=args.config_path,
                       config_section=args.config_section,
                       attn_res_config=attn_res_config, device=device)

    ev = Path(args.eval_dir)
    tm = resolve_token_map(ev, args.reserve_eos)
    result: dict[str, Any] = {"tag": args.tag,
                              "checkpoint": str(args.checkpoint),
                              "ledger": {}, "smoothing": None}

    found = False
    for name in LEDGER_FILES:
        path = ev / name
        if not path.exists():
            continue
        found = True
        rows = score_ledger(model, _read_jsonl(path), tm, device=device)
        agg = aggregate_ledger(rows)
        result["ledger"][name] = {"aggregate": agg, "rows": rows}
        _print_ledger(args.tag, name.removeprefix("eval_ledger")
                      .removeprefix("_").removesuffix(".jsonl") or "legacy",
                      agg)

    smo_path = ev / "eval_smoothing.jsonl"
    if smo_path.exists():
        found = True
        rows = score_smoothing(model, _read_jsonl(smo_path), device=device)
        agg = aggregate_smoothing(rows)
        result["smoothing"] = {"aggregate": agg, "rows": rows}
        print(f"[{args.tag}] SMOOTHING SI_nca: {agg['si_mean']:.3f} "
              f"± {agg['si_se']:.3f} (n={agg['n']}; 0.5 = indifferent, "
              f"high = rides over the event)")

    if not found:
        raise FileNotFoundError(
            f"no eval_ledger*.jsonl / eval_smoothing.jsonl under {ev}")

    out = Path(args.out) if args.out else ev / f"gates_{args.tag}.json"
    out.write_text(json.dumps(result))
    print(f"[{args.tag}] wrote {out}")


if __name__ == "__main__":
    main()
