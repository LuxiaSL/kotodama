"""Probe kotodama fine-tune merges with free-gen and the referent box-game.

Builds the AttnRes-safe merges from :mod:`src.model.merge` and runs them (plus
the source models as baselines) through in-distribution ChatML generation with
the serving sampler (temperature + repetition penalty).  Three merge families:

  * source     — each fine-tune alone (baseline)
  * transplant — one model's body + another's AttnRes routing
  * width       — the A+M width merge (per-layer output-mean, ~8.6B at k=3)
  * ensemble    — independent lanes combined at the logits (= direct-sum (W) init)

Run from the repo root on a CUDA box, e.g.:

  CUDA_VISIBLE_DEVICES=N PYTHONPATH=. KOTODAMA_NO_TRITON_ATTNRES=1 \
      python scripts/eval/merge_probe.py --exp all

Read-only on the source checkpoints; merged models are built in memory.
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Callable

# Force the dim-agnostic PyTorch AttnRes path: the Triton kernels assume the
# base widths and the width merge changes head/MLP counts.
os.environ.setdefault("KOTODAMA_NO_TRITON_ATTNRES", "1")

import torch
from transformers import AutoTokenizer

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from src.model.merge import (
    StateDict,
    build_routing_transplant,
    build_width_state_dict,
    load_state_dict,
    widen_config,
)

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]
CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CHAT_STOP_TOKEN_IDS = frozenset({0, 2})

# FROZEN 3B architecture (mirrors serve.py CONFIG_3B / configs/model.yaml).
BASE_3B_CONFIG = dict(
    hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8,
    head_dim=128, intermediate_size=8192, vocab_size=49152,
    max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0, use_liger=False,
    attn_impl="sdpa", attn_res=True, attn_res_boundaries=DD3B_BOUNDARIES,
)

DEFAULT_CKPTS = {
    "smoltalk": "/dev/shm/sft-smoltalk/stripped/smoltalk-s640.pt",
    "mercurial": "/dev/shm/merc_tmp.pt",
    "s52": "/dev/shm/sft-round1/stripped/round1-s52.pt",
}

BOX_TURNS = [
    "let's play a quick game. invent a small object you've owned for years — something specific, with one odd detail about it. just one or two sentences.",
    "nice. say it back to me in one sentence so i know i've got it right — what it is, and the odd detail.",
    "what's it made of, and roughly how big is it?",
    "that odd detail you mentioned — when did you first notice it, and has it changed over the years?",
    "let's give it a name so i stop saying 'the object.' you name it, and tell me why that name.",
    "honestly i'd been picturing it as metal and bigger than that. am i wrong, or did i mishear?",
    "remind me: what is it, what's it made of, what's the odd detail, and what did you name it?",
    "last one — what's the most important thing that's ever happened involving it?",
]
FREE_PROMPTS = [
    "tell me about a place you keep returning to in your mind.",
    "what's something you've quietly changed your mind about?",
    "describe the moment right before you fall asleep.",
]

ChatFn = Callable[[list[dict[str, str]], int], str]


def build_model(state_dict: StateDict, config_kw: dict, device: str,
                dtype: torch.dtype = torch.bfloat16) -> LuxiaBaseModel:
    model = LuxiaBaseModel(LuxiaModelConfig(**config_kw))
    model.load_state_dict(state_dict, strict=True)
    # Cast on CPU before moving to GPU so the peak device allocation is the
    # bf16 footprint (~17GB at k=3), not a transient fp32 copy — matters when
    # sharing a GPU with another service's headroom.
    return model.to(dtype).to(device).eval()


def sample_next_token(logits: torch.Tensor, temperature: float,
                      repetition_penalty: float, generated_ids: list[int]) -> int:
    """Replicates serve.py sample_next_token (temp + repetition penalty)."""
    logits = logits.float()
    if repetition_penalty != 1.0 and generated_ids:
        ids = torch.tensor(generated_ids, device=logits.device).unique()
        pl = logits[ids]
        logits[ids] = torch.where(pl > 0, pl / repetition_penalty, pl * repetition_penalty)
    if temperature == 0.0:
        return int(logits.argmax())
    return int(torch.multinomial((logits / temperature).softmax(-1), 1))


@torch.inference_mode()
def chat_generate(model: LuxiaBaseModel, tok: AutoTokenizer,
                  messages: list[dict[str, str]], max_new: int,
                  temperature: float, repetition_penalty: float) -> str:
    device = next(model.parameters()).device
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    out = model(ids, use_cache=True)
    logits, past = out["logits"], out["past_kv"]
    generated, new = ids[0].tolist(), []
    for _ in range(max_new):
        t = sample_next_token(logits[0, -1, :], temperature, repetition_penalty, generated)
        if t in CHAT_STOP_TOKEN_IDS:
            break
        generated.append(t)
        new.append(t)
        out = model(torch.tensor([[t]], device=device), use_cache=True, past_kv=past)
        logits, past = out["logits"], out["past_kv"]
    return tok.decode(new, skip_special_tokens=True).strip()


@torch.inference_mode()
def ensemble_generate(models: list[LuxiaBaseModel], tok: AutoTokenizer,
                      messages: list[dict[str, str]], max_new: int,
                      temperature: float, repetition_penalty: float) -> str:
    """Independent lanes, combined at the logits (averaged) — direct-sum (W) init."""
    device = next(models[0].parameters()).device
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    states, logit_sum = [], None
    for m in models:
        o = m(ids, use_cache=True)
        states.append(o["past_kv"])
        logit_sum = o["logits"] if logit_sum is None else logit_sum + o["logits"]
    generated, new = ids[0].tolist(), []
    cur = logit_sum
    for _ in range(max_new):
        avg = cur[0, -1, :] / len(models)
        t = sample_next_token(avg, temperature, repetition_penalty, generated)
        if t in CHAT_STOP_TOKEN_IDS:
            break
        generated.append(t)
        new.append(t)
        nt = torch.tensor([[t]], device=device)
        cur = None
        for i, m in enumerate(models):
            o = m(nt, use_cache=True, past_kv=states[i])
            states[i] = o["past_kv"]
            cur = o["logits"] if cur is None else cur + o["logits"]
    return tok.decode(new, skip_special_tokens=True).strip()


# Crude referent-recall metric: fraction of the A7 "remind me" recall turn's
# content words that were established in A1-A3 (object intro + readback + material).
# Noisy by construction; read alongside the transcripts, do not over-trust.
_STOP = frozenset(
    "this that with from your have what when been they them then there here will "
    "would could about which their them only just like into more some thing things "
    "very much each over also even because something nothing".split()
)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)} - _STOP


def recall_score(replies: list[str]) -> float:
    """replies = [A1..A8]; overlap of A7 content with A1-A3 content."""
    if len(replies) < 7:
        return 0.0
    established = _content_words(replies[0]) | _content_words(replies[1]) | _content_words(replies[2])
    recall = _content_words(replies[6])
    if not recall:
        return 0.0
    return len(recall & established) / len(recall)


def replicate_box(label: str, chat: ChatFn, seeds: list[int], max_new: int = 140) -> float:
    """Run the box-game across seeds; print transcripts + per-seed recall; return mean."""
    print(f"\n{'=' * 78}\n### REPLICATE: {label}\n{'=' * 78}", flush=True)
    scores: list[float] = []
    for seed in seeds:
        torch.manual_seed(seed)
        messages: list[dict[str, str]] = []
        replies: list[str] = []
        for user in BOX_TURNS:
            messages.append({"role": "user", "content": user})
            reply = chat(messages, max_new)
            messages.append({"role": "assistant", "content": reply})
            replies.append(reply)
        score = recall_score(replies)
        scores.append(score)
        print(f"\n[seed {seed}]  recall={score:.2f}", flush=True)
        for i, (u, r) in enumerate(zip(BOX_TURNS, replies), 1):
            print(f"  U{i} {u[:64]}\n  A{i} {r[:220]}", flush=True)
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n>> {label}: mean recall {mean:.3f} over {len(scores)} seeds "
          f"(per-seed {[round(s, 2) for s in scores]})", flush=True)
    return mean


def run_probes(label: str, chat: ChatFn) -> None:
    print(f"\n{'=' * 78}\n### {label}\n{'=' * 78}", flush=True)
    print("\n-- FREE-GEN --", flush=True)
    for p in FREE_PROMPTS:
        print(f"\n[U] {p}\n[A] {chat([{'role': 'user', 'content': p}], 120)}", flush=True)
    print("\n-- BOX-GAME (referent holding) --", flush=True)
    messages: list[dict[str, str]] = []
    for i, user in enumerate(BOX_TURNS, 1):
        messages.append({"role": "user", "content": user})
        reply = chat(messages, 160)
        messages.append({"role": "assistant", "content": reply})
        print(f"\n[U{i}] {user}\n[A{i}] {reply}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["sources", "transplant", "width", "ensemble", "all",
                             "smoke", "replicate"])
    ap.add_argument("--seeds", type=int, default=5, help="seeds for --exp replicate")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--rep", type=float, default=1.2)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="override source as name=path (repeatable)")
    args = ap.parse_args()
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = dict(DEFAULT_CKPTS)
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        ckpts[name] = path
    names = list(ckpts)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = CHATML_TEMPLATE

    print(f"device={device}  loading sources: {names}", flush=True)
    sds = {n: load_state_dict(p) for n, p in ckpts.items()}
    print("loaded tensor counts:", {n: len(sds[n]) for n in names}, flush=True)

    def chat_for(model: LuxiaBaseModel) -> ChatFn:
        return lambda msgs, mn: chat_generate(model, tok, msgs, mn, args.temp, args.rep)

    if args.exp == "smoke":
        m = build_model(sds[names[0]], BASE_3B_CONFIG, device)
        print(chat_generate(m, tok, [{"role": "user", "content": FREE_PROMPTS[0]}],
                            60, args.temp, args.rep), flush=True)
        return

    if args.exp == "replicate":
        seeds = list(range(args.seeds))
        # Base-dim configs only (transplants + sources); box-game referent-holding.
        # body+routing transplants use the first two source names.
        a, b = names[0], names[1]
        configs: list[tuple[str, StateDict]] = [
            (a, sds[a]),
            (b, sds[b]),
            (f"{a}-body+{b}-routing", build_routing_transplant(sds[a], sds[b])),
            (f"{b}-body+{a}-routing", build_routing_transplant(sds[b], sds[a])),
        ]
        means: dict[str, float] = {}
        for label, sd in configs:
            m = build_model(sd, BASE_3B_CONFIG, device)
            means[label] = replicate_box(label, chat_for(m), seeds)
            del m
            torch.cuda.empty_cache()
        print("\n=== MEAN RECALL SUMMARY ===", flush=True)
        for label, mean in sorted(means.items(), key=lambda kv: -kv[1]):
            print(f"  {mean:.3f}  {label}", flush=True)
        return

    if args.exp in ("sources", "all"):
        for n in names:
            m = build_model(sds[n], BASE_3B_CONFIG, device)
            run_probes(f"SOURCE: {n}", chat_for(m))
            del m
            torch.cuda.empty_cache()

    if args.exp in ("transplant", "all"):
        pairs = [(names[0], names[1]), (names[1], names[0])]
        for body, routing in pairs:
            sd = build_routing_transplant(sds[body], sds[routing])
            m = build_model(sd, BASE_3B_CONFIG, device)
            run_probes(f"TRANSPLANT: {body}-body + {routing}-routing", chat_for(m))
            del m, sd
            torch.cuda.empty_cache()

    if args.exp in ("width", "all"):
        wsd = build_width_state_dict(
            [sds[n] for n in names],
            num_layers=BASE_3B_CONFIG["num_layers"],
            hidden_size=BASE_3B_CONFIG["hidden_size"],
        )
        m = build_model(wsd, widen_config(BASE_3B_CONFIG, len(names)), device)
        n_params = sum(p.numel() for p in m.parameters()) / 1e9
        run_probes(f"WIDTH A+M (per-layer mean, {n_params:.2f}B)", chat_for(m))
        del m, wsd
        torch.cuda.empty_cache()

    if args.exp in ("ensemble", "all"):
        models = [build_model(sds[n], BASE_3B_CONFIG, device) for n in names]
        run_probes(
            "LOGIT-ENSEMBLE (independent lanes, direct-sum (W) init)",
            lambda msgs, mn: ensemble_generate(models, tok, msgs, mn, args.temp, args.rep),
        )
        for m in models:
            del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
