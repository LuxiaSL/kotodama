"""Self-play transcript generation: the model writes every turn of a plain-text transcript, one speaker at a time,
through an OpenAI-compatible ``/v1/completions`` endpoint (``kotodama.serve.server`` in base mode).

For each turn the generator asks for K candidates for the next speaker (the document so far, ending at an open
``Name:`` tag; stops = any participant's tag opening a new paragraph), runs the mechanical vetoes on each, and picks
UNIFORMLY at random among the survivors. Uniform choice keeps the corpus's entropy: picking the "best" candidate by
any score concentrates the distribution round after round, and the loops that follow are the failure the vetoes
exist to remove. Every candidate is banked with its verdict, so later stages (an outside veto judge, preference
pairs, the precision study) work from the same exhaust.

Mechanical vetoes (pure; ``evaluate``):

* ``overlong``    the turn hit ``max_tokens`` without ending;
* ``early_end``   the document ended (end-of-text) before ``min_turns``;
* ``empty`` / ``no_letters`` / ``meta_leak``  degenerate turns ("as an AI language model", "end of transcript",
  control-token text);
* ``selfcopy``    the turn re-emits the speaker's own earlier text (kotodama.posttrain.selfcopy, the speaker's prior
  turns as its own context);
* ``echo``        the turn copies the other participants' text verbatim (a span of the transcript's other turns);
* a stray speaker label inside a turn (a participant's name or a generic role label at a line start) is not a veto:
  the turn is cut there, and vetoed as ``empty`` only if nothing is left.

A transcript ends at ``max_turns``, at end-of-text after ``min_turns``, when the document would outgrow
``max_prompt_chars``, or when a turn has no survivor after ``rerolls`` extra draws (``no_survivor``; the transcript is
kept up to the last good turn).

Seeds are JSONL rows ``{"id"?, "header"?, "cast": [names], "turns"?: [opening turns]}``.

    python -m kotodama.posttrain.selfplay --seeds seeds.jsonl --endpoint http://HOST:PORT --out run/ \\
        [--law tsumugi] [--candidates 5] [--max_turns 12] [--concurrency 32]

Writes ``run/transcripts.jsonl`` (transcript rows + generation metadata) and ``run/candidates.jsonl`` (every
candidate with its verdict).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from kotodama.posttrain import selfcopy
from kotodama.posttrain.transcript import SEP, Transcript, Turn, render, render_prompt, stop_sequences, to_row
from kotodama.serve.laws import law

logger = logging.getLogger(__name__)

GENERIC_ROLES = ("User", "Assistant", "AI", "Human", "Bot", "System", "Narrator")
_META = re.compile(
    r"as an ai language model|as a language model,|i am an ai\b|end of (?:the )?(?:transcript|conversation)\b|"
    r"\[end\]|\(end\)|<\|[a-z_]+\|>|</?s>|transcript ends", re.I)


@dataclass(frozen=True)
class SelfPlayConfig:
    candidates: int = 5
    max_tokens: int = 256
    min_turns: int = 6
    max_turns: int = 12
    max_prompt_chars: int = 12000
    rerolls: int = 2
    echo_min_frac: float = 0.5
    # a turn identical (case and punctuation aside) to one of the last N turns, whoever spoke it, at any length —
    # the short cross-speaker ping-pong ("*Sigh*" / "*Sigh*") that self-copy and echo spans are too short to see
    repeat_window: int = 4
    # cycles of any period up to N turns: the candidate's speech (stage directions stripped) after the previous
    # turn's speech is an adjacent pair already said within the last N turns ("You do." / "I don't." ...); 0 = off
    cycle_window: int = 0
    # template lock: the (previous turn, candidate) pair of turn openings (first two words of speech) already seen
    # more than template_max times within the last template_window turns ("What is your favorite X?" / "My favorite
    # X is ..."); only candidates of 4+ words; 0 = off
    template_window: int = 0
    template_max: int = 3
    selfcopy: selfcopy.SelfCopyConfig = field(default_factory=selfcopy.SelfCopyConfig)


@dataclass
class Candidate:
    text: str
    ended_by: str  # "stop" (next speaker), "eos" (document end), "length"
    vetoes: list[str] = field(default_factory=list)
    cut_label: Optional[str] = None
    selfcopy_frac: float = 0.0
    echo_frac: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.vetoes


# ── pure pieces ──────────────────────────────────────────────────────────────


def split_completion(raw: str, stops: list[str], finish_reason: str, max_tokens_hit: bool) -> tuple[str, str]:
    """(turn text, ended_by). The server returns the text THROUGH the first stop string it saw, so cut there."""
    cut = min((i for i in (raw.find(s) for s in stops) if i >= 0), default=-1)
    if cut >= 0:
        return raw[:cut], "stop"
    if max_tokens_hit or finish_reason == "length":
        return raw, "length"
    return raw, "eos"


# A paragraph that opens with a short capitalized label and a colon is a new speaker's turn in this genre — including
# a participant whose name the model has drifted ("Ines" -> "Ine:", "Inez:"), which no stop sequence catches.
_PARA_LABEL = re.compile(r"\n[ \t]*\n[ \t]*([A-Z][\w'.-]{0,24}(?: [A-Z][\w'.-]{0,24}){0,2})[ \t]*:(?=\s)")


def cut_stray_label(text: str, cast: list[str]) -> tuple[str, Optional[str]]:
    """Cut a turn where another speaker's turn begins inside it: a line starting with a participant's name or a
    generic role label, or any paragraph opening with a short capitalized ``Label:``. Returns (kept text, label)."""
    names = sorted({*cast, *GENERIC_ROLES}, key=len, reverse=True)
    pat = re.compile(r"(?m)^[ \t]*(" + "|".join(re.escape(n) for n in names) + r")[ \t]*:")
    hits = [m for m in (pat.search(text), _PARA_LABEL.search(text)) if m is not None]
    if not hits:
        return text, None
    m = min(hits, key=lambda x: x.start())
    # a label with nothing before it means the turn itself opened as (another) speaker: nothing survives
    return text[:m.start()].rstrip(), m.group(1)


def evaluate(raw_text: str, ended_by: str, speaker: str, prior: list[Turn], cast: list[str], header: str,
             cfg: SelfPlayConfig) -> Candidate:
    """Run the mechanical vetoes on one candidate turn for ``speaker`` after ``prior`` turns."""
    text, label = cut_stray_label(raw_text, cast)
    text = text.strip()
    if label is not None and ended_by == "length":
        # the turn ended where another speaker's line began (single-newline transcript written inside one turn);
        # the token cap was hit by the text after it, which is cut away
        ended_by = "stop"
    c = Candidate(text=text, ended_by=ended_by, cut_label=label)
    if ended_by == "length":
        c.vetoes.append("overlong")
    if ended_by == "eos" and len(prior) + 1 < cfg.min_turns:
        c.vetoes.append("early_end")
    if not text:
        c.vetoes.append("empty")
        return c
    if not re.search(r"[A-Za-z]", text):
        c.vetoes.append("no_letters")
    if _META.search(text):
        c.vetoes.append("meta_leak")
    key = _norm_turn(text)
    if key and cfg.repeat_window and any(key == _norm_turn(t.text) for t in prior[-cfg.repeat_window:]):
        c.vetoes.append("repeat_turn")
    if cfg.cycle_window and prior and is_cycle(text, prior, cfg.cycle_window):
        c.vetoes.append("cycle")
    if cfg.template_window and prior and is_template(text, prior, cfg.template_window, cfg.template_max):
        c.vetoes.append("template")
    own = "\n\n".join(t.text for t in prior if t.speaker == speaker)
    others = "\n\n".join([header] + [t.text for t in prior if t.speaker != speaker])
    rep = selfcopy.score(text, context=own, prompt=others, cfg=cfg.selfcopy)
    c.selfcopy_frac, c.echo_frac = rep.copied_frac, rep.prompt_copied_frac
    if rep.copied_frac >= cfg.selfcopy.bad_frac or rep.longest_span >= cfg.selfcopy.bad_span:
        c.vetoes.append("selfcopy")
    if rep.prompt_copied_frac >= cfg.echo_min_frac and rep.n_words >= cfg.selfcopy.min_span:
        c.vetoes.append("echo")
    return c


def _norm_turn(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


_STAGE = re.compile(r"\([^)]*\)|\[[^\]]*\]|\*[^*]*\*")


def speech_key(text: str) -> str:
    """A turn's spoken words, normalised: stage directions in (), [] and ** removed, lowercase, letters and digits."""
    return " ".join(_norm_turn(_STAGE.sub(" ", text)).split())


def is_cycle(text: str, prior: list[Turn], window: int) -> bool:
    """The candidate repeats, after the previous turn, an adjacent pair of turns already said in the last ``window``
    turns (speech only): a loop of any period up to the window, whatever narration is appended."""
    cand, prev = speech_key(text), speech_key(prior[-1].text)
    if not cand:
        return False
    keys = [speech_key(t.text) for t in prior[-window - 1:]]
    return any(keys[i] == prev and keys[i + 1] == cand for i in range(len(keys) - 2))


def _opening(key: str) -> str:
    return " ".join(key.split()[:2])


def is_template(text: str, prior: list[Turn], window: int, max_repeats: int) -> bool:
    """The (previous turn, candidate) pair of openings (first two spoken words) already occurred more than
    ``max_repeats`` times as adjacent turns in the last ``window`` turns: a fixed question/answer frame refilled with
    new words. Candidates under 4 spoken words are left to the cycle veto."""
    cand = speech_key(text)
    if len(cand.split()) < 4:
        return False
    pair = (_opening(speech_key(prior[-1].text)), _opening(cand))
    keys = [_opening(speech_key(t.text)) for t in prior[-window - 1:]]
    return sum((keys[i], keys[i + 1]) == pair for i in range(len(keys) - 1)) > max_repeats


_STOP = frozenset("the a an and or but to of in on at for with is are was were be been it its this that you your i "
                  "me my we our he she his her they them their what do does did not no yes so just have has had will "
                  "would can could dont im its youre thats there here well oh okay know like get got go going".split())


def spoken_text(text: str, cast: Optional[list[str]] = None) -> str:
    """A turn's speech only: its first paragraph plus later paragraphs that ``kotodama.posttrain.narration`` does not
    call narration (stage directions are stripped later by ``speech_key``)."""
    from kotodama.posttrain.narration import classify_paragraph, split_turn

    paras = split_turn(text)
    keep = paras[:1] + [b for a, b in zip(paras, paras[1:]) if classify_paragraph(b, a, cast)[0] != "narr"]
    return "\n\n".join(keep)


def novelty(text: str, prior: list[Turn], window: int = 30, speech_only: bool = False,
            cast: Optional[list[str]] = None, count: bool = False) -> float:
    """Share of the candidate's content words (3+ letters, not stopwords) that none of the last ``window`` turns
    used; 0 for a turn without content words. ``speech_only``: narration paragraphs earn nothing (prose is word-rich,
    so counting it rewards narration over speech). ``count``: the number of such new words instead of their share
    (new state per turn, which a share hides: a one-word turn with one new word scores 1.0 as a share)."""
    new, total = novelty_counts(text, prior, window, speech_only, cast)
    if not total:
        return 0.0
    return float(new) if count else new / total


def novelty_counts(text: str, prior: list[Turn], window: int = 30, speech_only: bool = False,
                   cast: Optional[list[str]] = None) -> tuple[int, int]:
    """(new, total): the candidate's distinct content words that none of the last ``window`` turns used, and all of
    its distinct content words (``novelty`` is their ratio, or ``new`` with ``count``)."""
    words = {w for w in speech_key(spoken_text(text, cast) if speech_only else text).split()
             if len(w) >= 3 and w not in _STOP}
    if not words:
        return 0, 0
    seen = {w for t in prior[-window:] for w in speech_key(t.text).split()}
    return len(words - seen), len(words)


def own_opening_repeat(text: str, speaker: str, prior: list[Turn], back: int = 3) -> bool:
    """The turn's opening (first two spoken words) equals one of ``speaker``'s own last ``back`` openings: a
    per-speaker template ("Well, I ..." turn after turn) that the cross-speaker pair check (``is_template``) does not
    see. Turns under 4 spoken words are left to the repeat and cycle checks (False)."""
    key = speech_key(text)
    if len(key.split()) < 4:
        return False
    own = [_opening(speech_key(t.text)) for t in prior if t.speaker == speaker][-back:]
    return _opening(key) in own


def next_speaker(cast: list[str], turns: list[Turn]) -> str:
    """Round-robin over the cast, continuing from the last speaker."""
    if not turns:
        return cast[0]
    try:
        return cast[(cast.index(turns[-1].speaker) + 1) % len(cast)]
    except ValueError:
        return cast[0]


def speaker_tag_prompt(header: str, turns: list[Turn]) -> str:
    """The document up to the start of the next turn's name tag: what the model continues to name the next speaker."""
    if turns:
        return render(Transcript(tuple(turns), header))[0] + SEP
    return header + SEP if header else ""


def match_speaker(text: str, cast: list[str], exclude: Optional[str] = None) -> Optional[str]:
    """The cast member a sampled name tag names (exact, then case-insensitive), or None; ``exclude`` never matches."""
    name = text.strip().rstrip(":").strip()
    options = [c for c in cast if c != exclude]
    for c in options:
        if name == c:
            return c
    for c in options:
        if name.casefold() == c.casefold():
            return c
    return None


async def model_speaker(cast: list[str], turns: list[Turn], header: str, complete: Complete, samples: int = 4,
                        max_tokens: int = 8) -> tuple[str, bool]:
    """(next speaker, chosen by the model). The model writes the next turn's name tag: ``samples`` draws of a short
    continuation of the document up to the turn break, stopped at the colon. The first draw that names a cast member
    other than the last speaker is the choice, so the choice follows the model's own distribution over who speaks
    next. Falls back to round-robin (False) when no draw names one, or when the request fails."""
    rotate = next_speaker(cast, turns)
    if len(cast) < 2:
        return rotate, False
    last = turns[-1].speaker if turns else None
    try:
        raw = await complete(speaker_tag_prompt(header, turns), samples, [":", "\n"], max_tokens=max_tokens)
    except Exception as e:  # a failed tag draw costs this turn the model's choice, never the scene
        logger.warning("speaker tag draw failed (%s); rotating", e)
        return rotate, False
    for text, _finish, _hit in raw:
        got = match_speaker(text, cast, exclude=last)
        if got is not None:
            return got, True
    return rotate, False


# ── generation ───────────────────────────────────────────────────────────────

Complete = Callable[[str, int, list[str]], Awaitable[list[tuple[str, str, bool]]]]
"""(prompt, n, stops) -> [(raw text, finish_reason, hit max_tokens)] — one entry per candidate."""


def parse_steer(spec: str) -> list[dict[str, Any]]:
    """``"vrep:-0.05,V7_s58:0.03"`` -> ``[{"name": "vrep", "alpha": -0.05}, ...]`` (a bare name keeps the alias's
    default dose)."""
    out: list[dict[str, Any]] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        name, sep, alpha = part.rpartition(":")
        if not sep:
            out.append({"name": part})
            continue
        try:
            out.append({"name": name.strip(), "alpha": float(alpha)})
        except ValueError:
            raise ValueError(f"bad steering member {part!r}; expected name or name:alpha") from None
    return out


class PromptTooLong(ValueError):
    """The server refused a prompt longer than its context (a 400 naming the prompt length)."""


def oai_completer(endpoints: list[str], sampling: dict[str, float], max_tokens: int, timeout: float = 600.0,
                  model: str = "", transport: Any = None,
                  steer: Optional[list[dict[str, Any]]] = None,
                  steer_rule: str = "K") -> tuple[Complete, Callable[[], Awaitable[None]]]:
    """A ``Complete`` over one or more ``/v1/completions`` endpoints, and its async closer. Requests rotate over the
    endpoints, or go to ``endpoints[route]`` when the caller passes ``route=`` (one conversation per server, so its
    prompts share that server's prefix cache); a failed request retries on the next endpoint. ``user=`` is sent as the
    OpenAI ``user`` field, which ``kotodama.serve.gateway`` uses to keep one conversation on one replica. Every request has an
    overall deadline of ``timeout`` seconds, so a connection that stalls without closing still fails. A 4xx is
    not retried; a prompt the server finds longer than its context raises ``PromptTooLong``.
    ``steer``: a residual steering stack sent with every request (the server's ``vectors``; it needs
    ``--steer-npz``). ``sampling`` may also carry ``min_p`` and ``penalty_scope``, sent when present. Keyword
    ``fields`` of one call (e.g. ``seed``, ``user``, ``max_tokens``) are added to that call's request body; a seeded
    call split into several requests offsets the seed per request, so its candidates stay distinct.
    ``transport``: an httpx transport (tests pass a mock)."""
    import httpx

    client = httpx.AsyncClient(timeout=timeout, transport=transport)
    counter = {"i": 0}
    rep = float(sampling.get("repetition_penalty", 1.0))
    law_extra = {k: sampling[k] for k in ("min_p", "penalty_scope") if k in sampling}

    async def complete(prompt: str, n: int, stops: list[str], route: Optional[int] = None,
                       **fields: Any) -> list[tuple[str, str, bool]]:
        out: list[tuple[str, str, bool]] = []
        cap = int(fields.get("max_tokens", max_tokens))
        while len(out) < n:  # one request carries at most 8 candidates
            k = min(8, n - len(out))
            body = {"prompt": prompt, "model": model, "n": k, "max_tokens": max_tokens,
                    "temperature": sampling["temperature"], "top_p": sampling.get("top_p", 0.0),
                    "frequency_penalty": max(0.0, rep - 1.0), "stop": stops, "return_token_ids": True,
                    **law_extra, **fields}
            if fields.get("seed") is not None and out:
                body["seed"] = int(fields["seed"]) + len(out)
            if steer:
                body["vectors"] = steer
                body["steer_rule"] = steer_rule
            first = route if route is not None else counter["i"]
            if route is None:
                counter["i"] += 1
            for attempt in range(4):
                url = endpoints[(first + attempt) % len(endpoints)].rstrip("/") + "/v1/completions"
                try:
                    r = await asyncio.wait_for(client.post(url, json=body), timeout)
                    if r.status_code == 400 and "too long" in r.text.lower():
                        raise PromptTooLong(r.text)
                    r.raise_for_status()
                    break
                except (httpx.HTTPError, asyncio.TimeoutError) as e:
                    if attempt == 3 or (isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500):
                        raise
                    logger.warning("completion request to %s failed (%r); retrying", url, e)
                    await asyncio.sleep(2 ** attempt)
            for ch in r.json()["choices"]:
                ids = (ch.get("token_ids") or {}).get("completion") or []
                out.append((ch["text"], ch.get("finish_reason", "stop"), len(ids) >= cap))
        return out

    return complete, client.aclose


@dataclass
class SeedSpec:
    id: str
    cast: list[str]
    header: str = ""
    turns: list[Turn] = field(default_factory=list)
    history: list[Turn] = field(default_factory=list)  # turns an earlier run generated: context, never re-seeded


def parse_seed(row: dict[str, Any], idx: int) -> Optional[SeedSpec]:
    cast = row.get("cast")
    if not isinstance(cast, list) or len(cast) < 2 or not all(isinstance(c, str) and c.strip() for c in cast):
        return None
    def turns_of(key: str) -> list[Turn]:
        return [Turn(t["speaker"], t["text"]) for t in row.get(key, [])
                if isinstance(t, dict) and isinstance(t.get("speaker"), str) and isinstance(t.get("text"), str)]

    return SeedSpec(str(row.get("id", idx)), [c.strip() for c in cast], str(row.get("header", "")).strip(),
                    turns_of("turns"), turns_of("history"))


async def play(seed: SeedSpec, complete: Complete, cfg: SelfPlayConfig, rng: random.Random,
               bank: Callable[[dict[str, Any]], None], tid: str = "") -> dict[str, Any]:
    """Generate one transcript from ``seed``; returns its row (transcript + metadata). ``tid`` identifies the
    transcript (several transcripts may share a seed) in the row and in every banked candidate."""
    tid = tid or seed.id
    turns = list(seed.turns)
    stops = stop_sequences(seed.cast)
    finish = "max_turns"
    stats = {"candidates": 0, "survivors": 0, "rerolls": 0, "vetoes": {}}
    while len(turns) < cfg.max_turns:
        speaker = next_speaker(seed.cast, turns)
        prompt = render_prompt(Transcript(tuple(turns), seed.header), speaker)
        if len(prompt) > cfg.max_prompt_chars:
            finish = "length_cap"
            break
        survivors: list[Candidate] = []
        for draw in range(cfg.rerolls + 1):
            raw = await complete(prompt, cfg.candidates, stops)
            cands = [evaluate(*split_completion(t, stops, fr, hit)[:2], speaker, turns, seed.cast, seed.header, cfg)
                     for t, fr, hit in raw]
            survivors = [c for c in cands if c.ok]
            pick = rng.choice(survivors) if survivors else None
            for i, c in enumerate(cands):
                stats["candidates"] += 1
                for v in c.vetoes:
                    stats["vetoes"][v] = stats["vetoes"].get(v, 0) + 1
                bank({"tid": tid, "seed": seed.id, "turn": len(turns), "speaker": speaker, "draw": draw, "index": i,
                      "picked": c is pick, **asdict(c)})
            stats["survivors"] += len(survivors)
            if survivors:
                break
            stats["rerolls"] += 1
        if not survivors:
            finish = "no_survivor"
            break
        turns.append(Turn(speaker, pick.text))
        if pick.ended_by == "eos":
            finish = "eos"
            break
    tr = Transcript(tuple(turns), seed.header)
    return {**to_row(tr), "tid": tid, "seed": seed.id, "cast": seed.cast, "finish": finish, "n_turns": len(turns),
            "n_generated": len(turns) - len(seed.turns), "stats": stats}


async def run(seeds: list[SeedSpec], complete: Complete, cfg: SelfPlayConfig, out_dir: Path, concurrency: int,
              seed: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tf = (out_dir / "transcripts.jsonl").open("a", encoding="utf-8")
    cf = (out_dir / "candidates.jsonl").open("a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0}
    finishes: dict[str, int] = {}
    t0 = time.time()

    def bank(rec: dict[str, Any]) -> None:
        cf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(i: int, s: SeedSpec) -> None:
        async with sem:
            try:
                row = await play(s, complete, cfg, random.Random(f"{seed}:{s.id}:{i}"), bank, tid=f"{s.id}#{i}")
            except Exception as e:  # one transcript's failure must not stop the run
                logger.error("seed %s failed: %s", s.id, e)
                finishes["error"] = finishes.get("error", 0) + 1
                return
            tf.write(json.dumps(row, ensure_ascii=False) + "\n")
            tf.flush()
            cf.flush()
            finishes[row["finish"]] = finishes.get(row["finish"], 0) + 1
            done["n"] += 1
            if done["n"] % 25 == 0:
                logger.info("%d/%d transcripts (%.0fs) finishes %s", done["n"], len(seeds), time.time() - t0, finishes)

    try:
        await asyncio.gather(*(one(i, s) for i, s in enumerate(seeds)))
    finally:
        tf.close()
        cf.close()
    return {"transcripts": done["n"], "finishes": finishes, "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [selfplay] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="self-play plain-text transcripts through /v1/completions")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--endpoint", action="append", required=True, help="server base URL (repeatable: round-robin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--law", default="tsumugi", help="sampling law name (kotodama.serve.laws)")
    ap.add_argument("--temperature", type=float, default=None, help="override the law's temperature")
    ap.add_argument("--repetition_penalty", type=float, default=None, help="override the law's repetition penalty")
    ap.add_argument("--steer", default="", help='residual steering stack, e.g. "vrep:-0.05,V7_s58:0.05" '
                                                "(server needs --steer-npz)")
    ap.add_argument("--steer_rule", choices=["K", "N", "S"], default="K",
                    help="what lands at later block entries: K the site row unchanged, N scaled to the partial norm "
                         "there (server needs partial_norm_l* in its banks), S a single write")
    ap.add_argument("--repeat", type=int, default=1, help="transcripts per seed")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    for f in ("candidates", "max_tokens", "min_turns", "max_turns", "max_prompt_chars", "rerolls"):
        ap.add_argument(f"--{f}", type=int, default=getattr(SelfPlayConfig(), f))
    a = ap.parse_args(argv)
    cfg = SelfPlayConfig(candidates=a.candidates, max_tokens=a.max_tokens, min_turns=a.min_turns,
                         max_turns=a.max_turns, max_prompt_chars=a.max_prompt_chars, rerolls=a.rerolls)
    seeds: list[SeedSpec] = []
    with open(a.seeds, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                s = parse_seed(json.loads(line), i)
                if s is None:
                    logger.warning("seed row %d skipped (needs a cast of >= 2 names)", i)
                else:
                    seeds.extend([s] * a.repeat)
    if not seeds:
        raise SystemExit(f"no usable seeds in {a.seeds}")
    sampling = law(a.law)
    if a.temperature is not None:
        sampling["temperature"] = a.temperature
    if a.repetition_penalty is not None:
        sampling["repetition_penalty"] = a.repetition_penalty
    complete, close = oai_completer(a.endpoint, sampling, cfg.max_tokens, steer=parse_steer(a.steer) or None,
                                    steer_rule=a.steer_rule)

    async def go() -> dict[str, Any]:
        try:
            return await run(seeds, complete, cfg, Path(a.out), a.concurrency, a.seed)
        finally:
            await close()

    summary = asyncio.run(go())
    (Path(a.out) / "run.json").write_text(json.dumps({"args": vars(a), "config": asdict(cfg), "sampling": sampling,
                                                      "steer": parse_steer(a.steer), "steer_rule": a.steer_rule,
                                                      **summary}, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
