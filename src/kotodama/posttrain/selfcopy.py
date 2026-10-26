"""Span-level self-copy detector: the KTO labeler for repetition.

What it flags is the model re-emitting ITS OWN earlier text — not token overlap. Reusing names, objects, short noun
phrases, list markers or code identifiers is how a model keeps track of who-has-what (binding), so it must never be
penalized. Three signals, all on lowercased words with edge punctuation stripped (code fences are skipped):

  span      a verbatim run of >= ``min_span`` words that already occurred earlier in the own text (LZ77-style greedy
            parse, so a loop that copies itself — "a b a b a b" — counts)
  sentence  a sentence of >= ``min_sentence_words`` words identical to an earlier one
  template  a sentence of >= ``min_template_words`` words whose word alignment with an earlier sentence has
            ratio >= ``template_ratio`` (difflib) — "Who would buy a telescope in the rain? / Who would buy a
            wedding ring in the rain?", "Day 3: I woke up and ..." loops

``copied_frac`` = the fraction of the completion's words covered by any signal. A short shared noun phrase
("the brass telescope") is below every threshold; a re-quoted paragraph or a template loop is not.

The own text is the completion plus any ``context`` the caller marks as the model's own earlier output (e.g. its
previous assistant turns); a user's or document's text is not own text, so copying a prompt is measured separately
(``prompt_copied_frac``) and does not label.

Labelling (``label``): undesirable if copied_frac >= ``bad_frac`` or the longest copied run >= ``bad_span``;
desirable if copied_frac <= ``good_frac`` AND the completion has >= ``min_words`` words (a terse answer is not a
repetition fix — it is the early-stop failure); otherwise unlabelled (dropped from KTO).

CLI:  python -m kotodama.posttrain.selfcopy label --in samples.jsonl --out kto.jsonl [--context-field ...]
      python -m kotodama.posttrain.selfcopy score --in samples.jsonl     (distribution only)
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_EDGE = re.compile(r"^[^\w]+|[^\w]+$", re.U)


@dataclass(frozen=True)
class SelfCopyConfig:
    min_span: int = 8
    min_sentence_words: int = 4
    min_template_words: int = 6
    template_ratio: float = 0.8
    bad_frac: float = 0.20
    bad_span: int = 24
    good_frac: float = 0.03
    min_words: int = 40


@dataclass
class SelfCopyReport:
    n_words: int
    copied_frac: float
    span_words: int
    sentence_repeats: int
    template_repeats: int
    longest_span: int
    prompt_copied_frac: float = 0.0
    examples: list[str] = field(default_factory=list)  # up to 3 flagged snippets, for audit


def _words(text: str) -> list[str]:
    out = []
    for w in text.split():
        w = _EDGE.sub("", w.lower())
        if w:
            out.append(w)
    return out


def _sentences(text: str) -> list[list[str]]:
    return [ws for s in _SENT_END.split(text) if (ws := _words(s))]


def _strip_code(text: str) -> str:
    return _FENCE.sub(" ", text)


def _span_mask(own_prefix: list[str], words: list[str], min_span: int) -> tuple[list[bool], int]:
    """Mark words of ``words`` inside verbatim runs (>= min_span) that start earlier in own_prefix + words."""
    seq = own_prefix + words
    off = len(own_prefix)
    index: dict[tuple[str, ...], list[int]] = {}
    mask = [False] * len(words)
    longest = 0

    def add(j: int) -> None:
        if j + min_span <= len(seq):
            index.setdefault(tuple(seq[j:j + min_span]), []).append(j)

    for j in range(off):  # the context is all "earlier"
        add(j)
    i = off
    n = len(seq)
    while i < n:
        best = 0
        if i + min_span <= n:
            for j in index.get(tuple(seq[i:i + min_span]), ()):
                if j >= i:
                    continue
                L = min_span
                while i + L < n and seq[j + L] == seq[i + L]:  # j + L < i + L: overlap allowed (self-loops)
                    L += 1
                best = max(best, L)
        if best >= min_span:
            for k in range(i, i + best):
                mask[k - off] = True
                add(k)
            longest = max(longest, best)
            i += best
        else:
            add(i)
            i += 1
    return mask, longest


def score(completion: str, context: str = "", prompt: str = "",
          cfg: SelfCopyConfig = SelfCopyConfig()) -> SelfCopyReport:
    comp = _strip_code(completion)
    words = _words(comp)
    n = len(words)
    if n == 0:
        return SelfCopyReport(0, 0.0, 0, 0, 0, 0)
    ctx_text = _strip_code(context)
    span_mask, longest = _span_mask(_words(ctx_text), words, cfg.min_span)
    mask = list(span_mask)
    examples: list[str] = []

    # sentence-level signals: compare each completion sentence with every earlier own sentence
    ctx_sents = _sentences(ctx_text)
    comp_sents = _sentences(comp)
    # locate each completion sentence in the word stream (sentences tile the word list in order)
    starts: list[int] = []
    pos = 0
    for s in comp_sents:
        starts.append(pos)
        pos += len(s)
    seen_exact: set[tuple[str, ...]] = {tuple(s) for s in ctx_sents if len(s) >= cfg.min_sentence_words}
    earlier: list[list[str]] = [s for s in ctx_sents if len(s) >= cfg.min_template_words]
    sent_rep = tmpl_rep = 0
    for s, st in zip(comp_sents, starts):
        key = tuple(s)
        flagged = False
        if len(s) >= cfg.min_sentence_words and key in seen_exact:
            sent_rep += 1
            flagged = True
        elif len(s) >= cfg.min_template_words:
            sm = difflib.SequenceMatcher(autojunk=False)
            sm.set_seq2(s)
            for e in earlier:
                sm.set_seq1(e)
                if sm.real_quick_ratio() >= cfg.template_ratio and sm.quick_ratio() >= cfg.template_ratio \
                        and sm.ratio() >= cfg.template_ratio:
                    tmpl_rep += 1
                    flagged = True
                    break
        if flagged:
            for k in range(st, min(st + len(s), n)):
                mask[k] = True
            if len(examples) < 3:
                examples.append(" ".join(s)[:160])
        if len(s) >= cfg.min_sentence_words:
            seen_exact.add(key)
        if len(s) >= cfg.min_template_words:
            earlier.append(s)

    prompt_frac = 0.0
    if prompt:
        pmask, _ = _span_mask(_words(_strip_code(prompt)), words, cfg.min_span)
        # only runs that start in the prompt count here; runs inside the completion are already in span_mask
        prompt_frac = sum(p and not s for p, s in zip(pmask, span_mask)) / n
    return SelfCopyReport(
        n_words=n, copied_frac=sum(mask) / n, span_words=sum(span_mask), sentence_repeats=sent_rep,
        template_repeats=tmpl_rep, longest_span=longest, prompt_copied_frac=prompt_frac, examples=examples)


def label(report: SelfCopyReport, cfg: SelfCopyConfig = SelfCopyConfig()) -> Optional[bool]:
    """True = desirable, False = undesirable, None = ambiguous (leave out of KTO)."""
    if report.copied_frac >= cfg.bad_frac or report.longest_span >= cfg.bad_span:
        return False
    if report.copied_frac <= cfg.good_frac and report.n_words >= cfg.min_words:
        return True
    return None


# ── CLI ──────────────────────────────────────────────────────────────────────


def _iter_rows(path: Path):
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("%s:%d unparseable (%s) — skipped", path, ln, e)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["label", "score"])
    ap.add_argument("--in", dest="inp", required=True, help="JSONL rows with a completion field")
    ap.add_argument("--out", help="label mode: KTO JSONL (rows + label + selfcopy report)")
    ap.add_argument("--completion-field", default="completion")
    ap.add_argument("--context-field", default=None,
                    help="field holding the model's OWN earlier text (e.g. its previous assistant turns)")
    ap.add_argument("--prompt-field", default="prompt", help="measured (prompt_copied_frac) but never labels")
    for f in SelfCopyConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = ap.parse_args(argv)
    cfg = SelfCopyConfig(**{k: getattr(a, k) for k in SelfCopyConfig.__dataclass_fields__})
    if a.mode == "label" and not a.out:
        ap.error("label mode needs --out")

    counts = {True: 0, False: 0, None: 0}
    fracs: list[float] = []
    out = open(a.out, "w", encoding="utf-8") if a.mode == "label" else None
    try:
        for row in _iter_rows(Path(a.inp)):
            comp = row.get(a.completion_field)
            if not isinstance(comp, str):
                continue
            ctx = row.get(a.context_field, "") if a.context_field else ""
            prm = row.get(a.prompt_field, "")
            rep = score(comp, ctx if isinstance(ctx, str) else "", prm if isinstance(prm, str) else "", cfg)
            lab = label(rep, cfg)
            counts[lab] += 1
            fracs.append(rep.copied_frac)
            if out is not None and lab is not None:
                out.write(json.dumps({**row, "label": lab, "selfcopy": asdict(rep)}, ensure_ascii=False) + "\n")
    finally:
        if out is not None:
            out.close()
    fracs.sort()
    q = (lambda p: fracs[min(len(fracs) - 1, int(p * len(fracs)))]) if fracs else (lambda p: float("nan"))
    print(json.dumps({"n": len(fracs), "desirable": counts[True], "undesirable": counts[False],
                      "unlabelled": counts[None], "copied_frac_p10_p50_p90": [q(.1), q(.5), q(.9)],
                      "config": asdict(cfg)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
