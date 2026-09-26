"""Rider-grid items: one abstract generator, four renderers.

The P0.6 four-register binding battery (prereg FROZEN 2026-08-02; the bars live
there, not here). Harvested from the frozen ``gen_items.py``; generation is
byte-identical for a given seed + name pool, so banked grids stay comparable.

Renderers over the SAME abstract item:
  code       — Python dict mutations, query as ``print(state["G"])  #`` completion
  narrative  — third-person prose, no dialogue
  screenplay — SAM:/RIVER: prefixed lines, no chat tokens
  chat       — the screenplay lines as a role-tagged conversation (koto ChatML
               built here; HF chat models template server-side)

Invariants (``selftest``):
  * gold is computed ONCE, from the abstract simulation, for all renderers;
  * object-mention order is identical across renderers;
  * standard items: gold != most-recently-mentioned object; recency_control
    items (~15%): gold == most-recently-mentioned object;
  * every candidate " <obj>" is a single token in every grid tokenizer
    (``verify_names``; the banked pool is ``BANKED_VERIFIED_NAMES``).

Ops (single-object boxes; empties arise from ``move`` and are refilled by ``put``):
  swap(i,j)        both non-empty; mentions NO objects
  move(i,j)        i non-empty -> j (j's object, if any, is destroyed); i empties
  put(i,new)       i empty; mentions ``new``
  overwrite(i,new) i non-empty; mentions ``new``

Difficulty-0 tier (N=1, M=0, pure copy): candidates = gold + 7 pool distractors.
It gates HARNESS validity per renderer (bar: 95% rank-1).
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from kotodama.presets import TOKENIZER_NAME
from kotodama.serve.chatml import render_chatml

# Box labels: single capitals, dict-key friendly, no A-E (exemplar-ish), no I/O.
BOX_LABELS = ["F", "G", "H", "K", "L", "M", "P", "R"]

# Candidate object nouns. ``verify_names`` reduces this to the subset whose
# " <noun>" form is a single token in EVERY grid tokenizer.
OBJECT_CANDIDATES = [
    "bell", "glass", "pen", "ring", "map", "key", "coin", "book", "apple",
    "stone", "cup", "brush", "candle", "shell", "thread", "button", "feather",
    "spoon", "hammer", "marble", "lantern", "ribbon", "whistle", "magnet",
    "sponge", "needle", "basket", "kettle", "mirror", "pebble", "crayon",
    "funnel", "napkin", "walnut", "zipper", "goggles", "stapler", "teabag",
    "vase", "lamp", "clock", "knife", "fork", "plate", "bowl", "towel",
    "soap", "comb", "wallet", "ticket", "letter", "photo", "card", "dice",
    "chalk", "eraser", "pencil", "notebook", "envelope", "stamp", "bottle",
    "jar", "lid", "cork", "straw", "banana", "orange", "lemon", "peach",
    "carrot", "onion", "potato", "hat", "glove", "scarf", "sock", "belt",
    "boot", "shoe", "rope", "chain", "nail", "screw", "bolt", "wrench",
    "tape", "glue", "wire", "battery", "bulb", "switch", "fan", "drum",
]

GRID_TOKENIZERS: dict[str, str] = {
    "smollm2": TOKENIZER_NAME,                         # koto
    "pythia": "EleutherAI/pythia-2.8b-deduped",
    "llama32": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen25": "Qwen/Qwen2.5-3B-Instruct",              # base shares the vocab
}

# The pool ``verify_names`` produced on 2026-08-02 over GRID_TOKENIZERS (78/93;
# banked ``verified_names.json``). Every banked P0.6 grid was generated from it.
BANKED_VERIFIED_NAMES: tuple[str, ...] = (
    "bell", "glass", "pen", "ring", "map", "key", "coin", "book", "apple",
    "stone", "cup", "brush", "candle", "shell", "thread", "button", "feather",
    "spoon", "hammer", "marble", "lantern", "ribbon", "whistle", "magnet",
    "sponge", "needle", "basket", "mirror", "funnel", "lamp", "clock", "knife",
    "fork", "plate", "bowl", "towel", "soap", "comb", "wallet", "ticket",
    "letter", "photo", "card", "dice", "chalk", "pencil", "notebook",
    "envelope", "stamp", "bottle", "jar", "lid", "straw", "banana", "orange",
    "lemon", "carrot", "onion", "potato", "hat", "glove", "sock", "belt",
    "boot", "shoe", "rope", "chain", "nail", "screw", "bolt", "tape", "glue",
    "wire", "battery", "bulb", "switch", "fan", "drum",
)
BANKED_NAMES_TAG = "<banked:p06-verified-names-2026-08-02>"

SPEAKERS = ("SAM", "RIVER")  # screenplay/chat; SAM lines -> user, RIVER -> assistant

Renderer = Literal["code", "narrative", "screenplay", "chat"]
Family = Literal["raw", "chatml", "hf-chat"]
RENDERERS: tuple[Renderer, ...] = ("code", "narrative", "screenplay", "chat")

CELLS = [(n, m) for n in (2, 3, 5, 7) for m in (0, 1, 2, 4)]
RECENCY_CONTROL_FRAC = 0.15
N_SHOTS = 3
DEFAULT_SEED = 1002


@dataclass
class Op:
    kind: str                 # swap | move | put | overwrite
    i: str                    # box label
    j: Optional[str] = None   # swap/move target
    obj: Optional[str] = None # put/overwrite object


@dataclass
class Item:
    """One abstract item. Field order is the JSONL key order — do not reorder."""
    item_id: str
    n: int                    # boxes
    m: int                    # ops
    tier: str                 # "d0" | "std"
    boxes0: dict[str, str]    # initial contents (label -> object)
    ops: list[dict[str, Any]]
    final: dict[str, Optional[str]]
    query_box: str
    gold: str
    candidates: list[str]     # closed set, includes gold; order fixed
    mention_order: list[str]  # objects in text-mention order (all renderers)
    recency_control: bool     # True: gold IS the last-mentioned object
    gold_decl_pos: int        # 0-based position of gold in declaration order (audit)


# ── Abstract generation ────────────────────────────────────────────────────────

def _sim(boxes0: dict[str, str], ops: list[Op]) -> dict[str, Optional[str]]:
    state: dict[str, Optional[str]] = dict(boxes0)
    for op in ops:
        if op.kind == "swap":
            assert op.j is not None
            state[op.i], state[op.j] = state[op.j], state[op.i]
        elif op.kind == "move":
            assert op.j is not None
            state[op.j] = state[op.i]
            state[op.i] = None
        elif op.kind in ("put", "overwrite"):
            state[op.i] = op.obj
    return state


def gen_item(rng: random.Random, n: int, m: int, pool: list[str],
             recency_control: bool, item_id: str, max_tries: int = 400) -> Item:
    """One std-tier item with ``n`` boxes and ``m`` ops (rejection-sampled)."""
    for _ in range(max_tries):
        labels = BOX_LABELS[:n]
        objs = rng.sample(pool, n + m)  # declaration objects + headroom for put/overwrite
        decl_objs, spare = objs[:n], objs[n:]
        boxes0 = dict(zip(labels, decl_objs))
        mention = list(decl_objs)

        state: dict[str, Optional[str]] = dict(boxes0)
        ops: list[Op] = []
        tries = 0
        while len(ops) < m and tries < 200:
            tries += 1
            kind = rng.choice(["swap", "move", "put", "overwrite"])
            nonempty = [b for b in labels if state[b] is not None]
            empty = [b for b in labels if state[b] is None]
            if kind == "swap" and len(nonempty) >= 2:
                i, j = rng.sample(nonempty, 2)
                ops.append(Op("swap", i, j))
                state[i], state[j] = state[j], state[i]
            elif kind == "move" and nonempty and n >= 2:
                i = rng.choice(nonempty)
                j = rng.choice([b for b in labels if b != i])
                ops.append(Op("move", i, j))
                state[j] = state[i]
                state[i] = None
            elif kind == "put" and empty and spare:
                i = rng.choice(empty)
                obj = spare.pop()
                ops.append(Op("put", i, obj=obj))
                state[i] = obj
                mention.append(obj)
            elif kind == "overwrite" and nonempty and spare:
                i = rng.choice(nonempty)
                obj = spare.pop()
                ops.append(Op("overwrite", i, obj=obj))
                state[i] = obj
                mention.append(obj)
        if len(ops) != m:
            continue

        final = _sim(boxes0, ops)
        assert final == state, "simulation mismatch"
        last_obj = mention[-1]
        eligible = [b for b in labels if final[b] is not None]
        if recency_control:
            eligible = [b for b in eligible if final[b] == last_obj]
        else:
            eligible = [b for b in eligible if final[b] != last_obj]
        if not eligible:
            continue
        qbox = rng.choice(eligible)
        gold = final[qbox]
        assert gold is not None
        cands = list(dict.fromkeys(mention))  # distinct, mention order
        assert gold in cands
        return Item(
            item_id=item_id, n=n, m=m, tier="std", boxes0=boxes0,
            ops=[asdict(o) for o in ops], final=final, query_box=qbox,
            gold=gold, candidates=cands, mention_order=mention,
            recency_control=recency_control,
            gold_decl_pos=decl_objs.index(gold) if gold in decl_objs else -1,
        )
    raise RuntimeError(f"could not generate item {item_id} (n={n}, m={m}, "
                       f"recency_control={recency_control})")


def gen_d0(rng: random.Random, pool: list[str], item_id: str) -> Item:
    """One difficulty-0 (pure copy) item: gold + 7 pool distractors."""
    label = rng.choice(BOX_LABELS)
    picks = rng.sample(pool, 8)
    gold, distractors = picks[0], picks[1:]
    boxes0 = {label: gold}
    return Item(
        item_id=item_id, n=1, m=0, tier="d0", boxes0=boxes0, ops=[],
        final=dict(boxes0), query_box=label, gold=gold,
        candidates=[gold] + distractors, mention_order=[gold],
        recency_control=True,  # trivially: the only mention
        gold_decl_pos=0,
    )


# ── Renderers ──────────────────────────────────────────────────────────────────
# Each returns the item body; with_answer=True appends the gold (shots). All
# lead-ins end WITHOUT trailing space; the scored candidate carries the leading
# space (" vase") — the single-token form that was verified.

def _decl_sentences(item: Item) -> list[str]:
    return [f"Box {b} contains the {o}." for b, o in item.boxes0.items()]


def _op_sentence(op: dict[str, Any]) -> str:
    k = op["kind"]
    if k == "swap":
        return f"The contents of Box {op['i']} and Box {op['j']} are swapped."
    if k == "move":
        return (f"Everything in Box {op['i']} is moved into Box {op['j']}, "
                f"replacing whatever was there.")
    if k == "put":
        return f"The {op['obj']} is placed into the empty Box {op['i']}."
    if k == "overwrite":
        return (f"The {op['obj']} is placed into Box {op['i']}, "
                f"replacing what was there.")
    raise ValueError(f"unknown op kind {k!r}")


def render_narrative(item: Item, with_answer: bool) -> str:
    body = " ".join(_decl_sentences(item) + [_op_sentence(o) for o in item.ops])
    lead = f"{body} Box {item.query_box} now contains the"
    return f"{lead} {item.gold}." if with_answer else lead


def _code_op(op: dict[str, Any]) -> str:
    k = op["kind"]
    if k == "swap":
        return (f'state["{op["i"]}"], state["{op["j"]}"] = '
                f'state["{op["j"]}"], state["{op["i"]}"]')
    if k == "move":
        return f'state["{op["j"]}"] = state.pop("{op["i"]}")'
    if k in ("put", "overwrite"):
        return f'state["{op["i"]}"] = "{op["obj"]}"'
    raise ValueError(f"unknown op kind {k!r}")


def render_code(item: Item, with_answer: bool) -> str:
    decl = ", ".join(f'"{b}": "{o}"' for b, o in item.boxes0.items())
    lines = [f"state = {{{decl}}}"] + [_code_op(o) for o in item.ops]
    lines.append(f'print(state["{item.query_box}"])  #')
    text = "\n".join(lines)
    return f"{text} {item.gold}" if with_answer else text


def _dialogue_lines(item: Item) -> list[str]:
    """Line contents shared by screenplay + chat. Parity arranged so the query
    line always lands on the SECOND speaker (RIVER -> assistant): with m even
    the declarations are one line; with m odd they are split into two."""
    decls = _decl_sentences(item)
    if item.m % 2 == 0:
        lines = [" ".join(decls)]
    else:
        cut = max(1, (len(decls) + 1) // 2)
        lines = [" ".join(decls[:cut]), " ".join(decls[cut:]) or "Go on."]
    lines += [_op_sentence(o) for o in item.ops]
    lines.append(f"So Box {item.query_box} now contains the")
    assert (len(lines) - 1) % 2 == 1, "query line must land on speaker 2"
    return lines


def render_screenplay(item: Item, with_answer: bool) -> str:
    lines = _dialogue_lines(item)
    out = [f"{SPEAKERS[i % 2]}: {ln}" for i, ln in enumerate(lines)]
    if with_answer:
        out[-1] = f"{out[-1]} {item.gold}."
    return "\n".join(out)


def chat_messages(item: Item, with_answer: bool) -> list[dict[str, str]]:
    """The screenplay lines as alternating user/assistant turns. The final
    assistant turn is the OPEN lead-in unless with_answer (shots)."""
    lines = _dialogue_lines(item)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": ln}
            for i, ln in enumerate(lines)]
    if with_answer:
        msgs[-1]["content"] = f"{msgs[-1]['content']} {item.gold}."
    return msgs


def chatml_render(messages: list[dict[str, str]], open_final: bool) -> str:
    """koto ChatML as a raw prompt (base-mode servers accept it as plain text).

    open_final=True leaves the last (assistant) turn unclosed with its lead-in
    inline: ``render_chatml`` of the closed turns (whose trailing generation
    prompt IS the open assistant header) + the lead-in — byte-identical to the
    frozen battery's hand-built string. open_final=False closes every turn and
    appends no generation prompt, which ``render_chatml`` cannot express.
    """
    if not open_final:
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                       for m in messages)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("open_final needs a final assistant turn")
    return render_chatml(messages[:-1]) + messages[-1]["content"]


_RAW_RENDERERS = {"code": render_code, "narrative": render_narrative,
                  "screenplay": render_screenplay}


def build_prompt(item: Item, renderer: str, shots: list[Item],
                 family: str) -> dict[str, Any]:
    """Assemble the full prompt for one (item, renderer, shot-condition).

    family: 'raw'     -> {"prompt": ...} plain text (every non-chat renderer)
            'chatml'  -> {"prompt": ...} koto ChatML string (chat renderer;
                         koto base AND tuned are both driven raw)
            'hf-chat' -> {"messages": [...], "answer_prefix": ...} for an HF shim
                         to template server-side (chat renderer, HF instruct)
    """
    if renderer != "chat":
        if renderer not in _RAW_RENDERERS:
            raise ValueError(f"unknown renderer {renderer!r}; choose from {RENDERERS}")
        rfn = _RAW_RENDERERS[renderer]
        blocks = [rfn(s, True) for s in shots] + [rfn(item, False)]
        # "###" item delimiter: d0 gate v1 caught llama-3.2 blending narrative
        # shot items into the query under bare "\n\n" (prereg §10 amendment).
        return {"prompt": "\n\n###\n\n".join(blocks)}

    shot_msgs = [m for s in shots for m in chat_messages(s, True)]
    msgs = shot_msgs + chat_messages(item, False)
    if family == "hf-chat":
        return {"messages": msgs[:-1], "answer_prefix": msgs[-1]["content"]}
    return {"prompt": chatml_render(msgs, open_final=True)}


# ── Name verification ──────────────────────────────────────────────────────────

def verify_names(out_path: Path) -> list[str]:
    """Write the single-token-in-every-grid-tokenizer pool to ``out_path``."""
    from transformers import AutoTokenizer  # deferred: heavy import
    toks = {}
    for key, name in GRID_TOKENIZERS.items():
        print(f"loading tokenizer {key} = {name}", flush=True)
        toks[key] = AutoTokenizer.from_pretrained(name)
    verified: list[str] = []
    rejected: dict[str, dict[str, int]] = {}
    for noun in OBJECT_CANDIDATES:
        bad = {}
        for key, tok in toks.items():
            ids = tok(" " + noun, add_special_tokens=False).input_ids
            if len(ids) != 1:
                bad[key] = len(ids)
        if bad:
            rejected[noun] = bad
        else:
            verified.append(noun)
    report = {"tokenizers": GRID_TOKENIZERS, "verified": verified,
              "rejected": rejected, "n_verified": len(verified)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"{len(verified)}/{len(OBJECT_CANDIDATES)} nouns verified "
          f"single-token in all {len(toks)} tokenizers -> {out_path}", flush=True)
    if len(verified) < 40:
        print("WARNING: pool below 40 — N=7/M=4 cells need 11 distinct objects "
              "per item; consider extending OBJECT_CANDIDATES", file=sys.stderr)
    return verified


def load_names(names_path: Optional[Path]) -> list[str]:
    """The verified pool from a ``verify_names`` file, or the banked pool if None."""
    if names_path is None:
        return list(BANKED_VERIFIED_NAMES)
    try:
        names = json.loads(names_path.read_text())["verified"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"unreadable names file {names_path}: {exc}") from exc
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ValueError(f"{names_path}: 'verified' must be a list of strings")
    return names


# ── Emission ───────────────────────────────────────────────────────────────────

def generate(pool_all: list[str], per_cell: int, d0_count: int,
             seed: int) -> tuple[list[Item], list[Item]]:
    """(items, shots) for the full grid. Shots draw from a disjoint name pool."""
    rng = random.Random(seed)
    shot_pool = pool_all[:12]
    item_pool = pool_all[12:]
    if len(item_pool) < 11:
        raise RuntimeError(f"item pool too small ({len(item_pool)})")

    shot_rng = random.Random(seed + 1)
    shots = [gen_item(shot_rng, 3, 2, shot_pool, False, f"shot_{k}")
             for k in range(N_SHOTS)]

    items: list[Item] = [gen_d0(rng, item_pool, f"d0_{k:03d}")
                         for k in range(d0_count)]
    for n, m in CELLS:
        # m=0 recency-control would force gold == last-declared; only tag where
        # ops exist.
        n_rc = round(per_cell * RECENCY_CONTROL_FRAC) if m > 0 else 0
        for k in range(per_cell):
            items.append(gen_item(rng, n, m, item_pool, k < n_rc,
                                  f"n{n}m{m}_{k:03d}"))
    return items, shots


def emit(names_path: Optional[Path], out_dir: Path, per_cell: int = 200,
         d0_count: int = 200, seed: int = DEFAULT_SEED) -> Path:
    """Write items.jsonl + shots.json + emit_meta.json under ``out_dir``."""
    pool_all = load_names(names_path)
    items, shots = generate(pool_all, per_cell, d0_count, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it)) + "\n")
    with (out_dir / "shots.json").open("w") as fh:
        json.dump([asdict(s) for s in shots], fh, indent=2)
    (out_dir / "emit_meta.json").write_text(json.dumps({
        "seed": seed, "per_cell": per_cell, "d0_count": d0_count,
        "cells": CELLS, "recency_control_frac": RECENCY_CONTROL_FRAC,
        "names_file": str(names_path) if names_path is not None else BANKED_NAMES_TAG,
        "n_items": len(items), "shot_pool": pool_all[:12],
    }, indent=2))
    print(f"{len(items)} items -> {out_dir / 'items.jsonl'} "
          f"(+{N_SHOTS} shots, disjoint names)", flush=True)
    return out_dir / "items.jsonl"


def load_items(path: Path) -> list[Item]:
    """Items from an ``items.jsonl`` (or shots from a ``shots.json``)."""
    try:
        text = path.read_text()
        if path.suffix == ".json":
            rows = json.loads(text)
        else:
            rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        return [Item(**r) for r in rows]
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"unreadable items file {path}: {exc}") from exc


# ── Self-test ──────────────────────────────────────────────────────────────────

def selftest() -> int:
    """Invariant checks on a large generated sample; returns the failure count."""
    rng = random.Random(7)
    pool = OBJECT_CANDIDATES[12:]
    failures = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal failures
        if not cond:
            failures += 1
            print(f"FAIL: {msg}", file=sys.stderr)

    n_checked = 0
    pat = re.compile(r"\b(" + "|".join(map(re.escape, pool)) + r")\b")
    for trial in range(600):
        n = rng.choice((2, 3, 5, 7))
        m = rng.choice((0, 1, 2, 4))
        rc = rng.random() < 0.3 and m > 0
        it = gen_item(rng, n, m, pool, rc, f"t{trial}")
        n_checked += 1

        # 1. gold consistency: recompute simulation from scratch
        resim = _sim(it.boxes0, [Op(**o) for o in it.ops])
        check(resim[it.query_box] == it.gold, f"{it.item_id}: gold mismatch")
        check(it.gold in it.candidates, f"{it.item_id}: gold not in candidates")

        # 2. recency invariant on the abstract mention order
        last = it.mention_order[-1]
        if it.recency_control:
            check(it.gold == last, f"{it.item_id}: rc item gold != last mention")
        else:
            check(it.gold != last, f"{it.item_id}: std item gold == last mention")

        # 3. cross-renderer mention order
        for rname, rfn in _RAW_RENDERERS.items():
            text = rfn(it, False)
            found = pat.findall(text)
            check(found == it.mention_order,
                  f"{it.item_id}/{rname}: mention order {found} != {it.mention_order}")
            check(not text.endswith(" "), f"{it.item_id}/{rname}: trailing space")
        msgs = chat_messages(it, False)
        chat_found = pat.findall(" ".join(m["content"] for m in msgs))
        check(chat_found == it.mention_order, f"{it.item_id}/chat: mention order")
        check(msgs[-1]["role"] == "assistant",
              f"{it.item_id}/chat: final turn not assistant")
        roles = [m["role"] for m in msgs]
        check(all(roles[k] != roles[k + 1] for k in range(len(roles) - 1)),
              f"{it.item_id}/chat: roles do not alternate")

        # 4. with_answer renders end with the gold
        for rfn in (render_narrative, render_screenplay):
            check(rfn(it, True).endswith(f" {it.gold}."),
                  f"{it.item_id}: shot render missing gold")
        check(render_code(it, True).endswith(f"# {it.gold}"),
              f"{it.item_id}: code shot missing gold")

    # 5. d0 tier
    for k in range(50):
        d0 = gen_d0(rng, pool, f"d0t{k}")
        check(len(d0.candidates) == 8 and d0.gold == d0.candidates[0],
              f"{d0.item_id}: d0 candidate set malformed")

    # 6. ChatML render shape
    it = gen_item(rng, 3, 2, pool, False, "chatml_check")
    s = chatml_render(chat_messages(it, False), open_final=True)
    check(s.endswith(f"So Box {it.query_box} now contains the"),
          "chatml: open final turn malformed")
    check(s.count("<|im_start|>assistant") >= 1 and not s.endswith("<|im_end|>\n"),
          "chatml: final turn closed but should be open")

    print(f"selftest: {n_checked} items checked, {failures} failures", flush=True)
    return failures
