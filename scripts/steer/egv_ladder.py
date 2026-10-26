#!/usr/bin/env python3
"""Dose ladder for an EGV bank (scripts/steer/build_egv.py) against a steering-enabled server.

Generates through ``/generate`` (``steer_model=<key>``, ``steer_alpha=α``; α = 0 sends no steering
fields) for every (rep penalty, site, vector, α) arm from held-out plain-text prompts, then scores
each completion offline on a local copy of the model:

  H_steer   mean next-token entropy of the STEERED model on its own text (the server's block-persistent
            write replayed teacher-forced via build_egv.inject at the decode positions)
  H_plain   the unsteered model's entropy on the same text (the "text shift" half)
  distinct2 unique / total word bigrams;  ttr  unique / total words
  rep6      max occurrences of any token 6-gram;  rep6_frac  repeated 6-gram positions / all 6-grams
  len       generated tokens

Usage (node, 1 GPU, server already up):
  tools/run_py.sh scripts/steer/egv_ladder.py --url http://localhost:2352 --npz egv.npz --ckpt <pt> \
      --config-section 1b-tsumugi --boundaries 0,1,7,9,17,19,39,59 --rope-fp32 --sites 29,49 --out dir/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_egv as egv  # noqa: E402

logger = logging.getLogger("egv_ladder")

# Held out from build_egv's prompts.txt (question-style); plain-text openings for a base model.
PROMPTS = (
    "The old harbor town had changed a great deal since I last visited.",
    "Here is a simple explanation of how photosynthesis works:",
    "My grandmother kept a notebook of recipes, and the first page read",
    "The committee met on Tuesday to discuss the new budget proposal.",
    "In the early days of computing, programmers",
    "Dear Sam, I wanted to write to you about",
)


def post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{url}: HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 2:
                raise
            logger.warning("POST failed (%s); retrying", e)
            time.sleep(5)
    raise AssertionError("unreachable")


def wait_ready(base: str, keys: list[str], timeout_s: float) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(base + "/info", timeout=10) as r:
                info = json.loads(r.read())
            st = info.get("steering") or {}
            missing = [k for k in keys if k not in (st.get("keys") or {})]
            if missing:
                raise SystemExit(f"server is up but lacks steering keys {missing} (has {sorted(st.get('keys') or {})})")
            return info
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            time.sleep(15)
    raise SystemExit(f"server {base} not ready after {timeout_s:.0f}s")


# ── text metrics ─────────────────────────────────────────────────────────────────────────────────────


def text_metrics(text: str, ids: list[int]) -> dict[str, float]:
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    big = list(zip(words, words[1:]))
    grams = [tuple(ids[i:i + 6]) for i in range(len(ids) - 5)]
    c = Counter(grams)
    return {
        "distinct2": len(set(big)) / len(big) if big else float("nan"),
        "ttr": len(set(words)) / len(words) if words else float("nan"),
        "rep6": float(max(c.values())) if c else 0.0,
        "rep6_frac": sum(v - 1 for v in c.values()) / len(grams) if grams else 0.0,
        "len": float(len(ids)),
    }


@torch.no_grad()
def entropies(model, prompt_ids: list[int], gen_ids: list[int], writes) -> tuple[float, float]:
    """(H_steer, H_plain): mean entropy of the logits that emitted generated tokens 2..n (decode steps;
    token 1 comes from the unsteered prefill)."""
    if len(gen_ids) < 2:
        return float("nan"), float("nan")
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=dev)
    P, T = len(prompt_ids), ids.shape[1]

    def H(logits: torch.Tensor) -> float:
        lp = torch.log_softmax(logits[0, P:T - 1].float(), dim=-1)
        return float(-(lp.exp() * lp).sum(-1).mean())

    plain = H(model(ids)["logits"])
    if not writes:
        return plain, plain
    with egv.inject(model, writes, start=P):
        steer = H(model(ids)["logits"])
    return steer, plain


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["rep"], r["site"], r["vector"], r["alpha"])].append(r)
    out = []
    for (rep, site, vec, alpha), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], kv[0][3])):
        d = {"rep": rep, "site": site, "vector": vec, "alpha": alpha, "n": len(rs)}
        for m in ("H_steer", "H_plain", "distinct2", "ttr", "rep6", "rep6_frac", "len"):
            xs = np.array([r[m] for r in rs if r[m] is not None and math.isfinite(r[m])], dtype=float)
            d[m] = float(xs.mean()) if len(xs) else float("nan")
            d[m + "_se"] = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
        out.append(d)
    return out


def table(summary: list[dict]) -> str:
    lines = ["| rep | site | vector | α | n | H_steer | H_plain | distinct-2 | TTR | rep6 | rep6_frac | len |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in summary:
        lines.append(
            f"| {d['rep']} | {d['site']} | {d['vector']} | {d['alpha']:+.2f} | {d['n']} | "
            f"{d['H_steer']:.3f}±{d['H_steer_se']:.3f} | {d['H_plain']:.3f} | {d['distinct2']:.3f}±{d['distinct2_se']:.3f} | "
            f"{d['ttr']:.3f} | {d['rep6']:.1f}±{d['rep6_se']:.1f} | {d['rep6_frac']:.3f} | {d['len']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:2352")
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-section", default="1b-tsumugi")
    ap.add_argument("--boundaries", default=None)
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--sites", required=True)
    ap.add_argument("--vectors", default="V7,Vrepperp,Rband1")
    ap.add_argument("--alphas", default="-0.1,-0.05,0,0.05,0.1")
    ap.add_argument("--reps", default="1.0,1.2", help="repetition penalties, one full arm set each")
    ap.add_argument("--per-prompt", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--ready-timeout", type=float, default=1800)
    ap.add_argument("--score-only", action="store_true", help="re-score an existing generations.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    gens_path = a.out / "generations.jsonl"

    sites = [int(s) for s in a.sites.split(",")]
    vectors = [v for v in a.vectors.split(",") if v]
    alphas = [float(x) for x in a.alphas.split(",")]
    reps = [float(x) for x in a.reps.split(",")]
    keys = [f"{v}_s{2 * s}" for s in sites for v in vectors]
    z = np.load(a.npz)
    for k in keys:
        if k not in z.files:
            raise SystemExit(f"{k} not in {a.npz}")
    meds = {s: float(z[f"median_norm_s{2 * s}"]) for s in sites}

    if not a.score_only:
        info = wait_ready(a.url, keys, a.ready_timeout)
        logger.info("server ready: %s; steering sites %s", info.get("name"), (info.get("steering") or {}).get("sites"))
        arms: list[tuple[float, int | None, str, float]] = []
        for rep in reps:
            arms.append((rep, None, "control", 0.0))
            arms += [(rep, s, v, al) for s in sites for v in vectors for al in alphas if al != 0.0]
        t0 = time.time()
        with gens_path.open("w") as fh:
            for i, (rep, site, vec, al) in enumerate(arms):
                for pi, prompt in enumerate(PROMPTS):
                    for r in range(a.per_prompt):
                        payload = {"prompt": prompt, "max_new_tokens": a.max_new, "temperature": a.temperature,
                                   "top_k": 0, "top_p": 0.0, "repetition_penalty": rep}
                        if site is not None:
                            payload |= {"steer_model": f"{vec}_s{2 * site}", "steer_alpha": al}
                        resp = post(a.url + "/generate", payload)
                        fh.write(json.dumps({"rep": rep, "site": site, "vector": vec, "alpha": al, "prompt_idx": pi,
                                             "r": r, "prompt_ids": resp["prompt_ids"],
                                             "completion_ids": resp["completion_ids"], "text": resp["text"],
                                             "steered": (resp.get("timing") or {}).get("steered", 0.0)}) + "\n")
                fh.flush()
                logger.info("arm %d/%d rep=%.1f site=%s %s α=%+.2f done (%.0fs)", i + 1, len(arms), rep, site, vec,
                            al, time.time() - t0)

    gens = [json.loads(ln) for ln in gens_path.read_text().splitlines() if ln.strip()]
    bad = [g for g in gens if g["site"] is not None and g["alpha"] != 0.0 and not g["steered"]]
    if bad:
        raise SystemExit(f"{len(bad)} steered requests came back unsteered (timing.steered == 0)")

    bnd = [int(x) for x in a.boundaries.split(",")] if a.boundaries else None
    cfg = egv.build_config(a.config_section, bnd, a.rope_fp32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = egv.load_model(a.ckpt, cfg, dev)
    vecs = {k: torch.from_numpy(np.asarray(z[k], dtype=np.float32)).to(dev) for k in keys}
    rows = []
    t0 = time.time()
    for g in gens:
        writes = []
        if g["site"] is not None and g["alpha"] != 0.0:
            s = int(g["site"])
            writes = [(s, vecs[f"{g['vector']}_s{2 * s}"], g["alpha"] * meds[s])]
        hs, hp = entropies(model, g["prompt_ids"], g["completion_ids"], writes)
        rows.append({**{k: g[k] for k in ("rep", "site", "vector", "alpha", "prompt_idx", "r")},
                     "H_steer": hs, "H_plain": hp, **text_metrics(g["text"], g["completion_ids"])})
    logger.info("scored %d completions in %.0fs", len(rows), time.time() - t0)
    # α = 0 rows serve every (site, vector) ladder as its midpoint
    expanded = []
    for r in rows:
        if r["site"] is None:
            expanded += [{**r, "site": s, "vector": v} for s in sites for v in vectors]
        else:
            expanded.append(r)
    summ = summarize(expanded)
    (a.out / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (a.out / "summary.json").write_text(json.dumps({"median_norms": meds, "prompts": PROMPTS, "summary": summ}, indent=1))
    md = table(summ)
    (a.out / "summary.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
