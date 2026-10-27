"""Compliance screen for fine-tuning text: what must not be trained on whatever the model is for.

Self-play transcripts are a distillation of the base model's prior, so they carry what its pretraining text did.
Two kinds of text are screened here, deterministically and cheaply:

  * ``minor_explicit``: a turn that names a minor (child, kid, an age under 18, ...) and an explicit sexual term.
    Documents containing such a turn are dropped (legal compliance). The rule is a co-occurrence test, so it
    over-flags (profanity near "kids") rather than under-flags; at the measured rates (well under 1% of documents)
    dropping the over-flagged ones costs nothing.
  * PII: email addresses, phone numbers, US social-security numbers, payment-card numbers (Luhn-checked) and IPv4
    addresses. Generated dialogue rarely holds real PII, so these are redacted to placeholders rather than dropped.

``turn_hash`` gives a turn a stable content id (sha256 of its whitespace-normalised, lowercased text), so a scan can
write a blocklist of flagged turns WITHOUT their text, and assembly can drop every document that contains one. The
blocklist stays valid across re-assemblies and across later rounds that carry earlier turns forward as history.

    flags = classify(text)                 # {"minor_explicit", "pii_email", ...}
    python -m kotodama.posttrain.compliance scan <run dirs> --out blocklist.tsv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

_MINOR = re.compile(r"\b(child|children|kids?|little (girl|boy)|underage|minor|toddler|preteen|schoolgirl|schoolboy|"
                    r"(1[0-7]|[1-9]|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen)[- ]years?[- ]old)\b",
                    re.I)
_EXPLICIT = re.compile(r"\b(sex|sexual|naked|nude|undress\w*|underwear|panties|breasts?|nipples?|cock|penis|dick|"
                       r"vagina|pussy|orgasm|cum|cums|fuck\w*|masturbat\w*|erection|aroused|horny)\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<![\w-])(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IPV4 = re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")
PLACEHOLDER = {"pii_email": "<email>", "pii_phone": "<phone>", "pii_ssn": "<ssn>", "pii_card": "<card>",
               "pii_ip": "<ip>"}


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _cards(text: str) -> list[re.Match]:
    out = []
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            out.append(m)
    return out


def classify(text: str) -> set[str]:
    """The compliance flags one turn's text raises."""
    flags: set[str] = set()
    if _MINOR.search(text) and _EXPLICIT.search(text):
        flags.add("minor_explicit")
    if _EMAIL.search(text):
        flags.add("pii_email")
    if _SSN.search(text):
        flags.add("pii_ssn")
    if _cards(text):
        flags.add("pii_card")
    if _PHONE.search(text):
        flags.add("pii_phone")
    if _IPV4.search(text):
        flags.add("pii_ip")
    return flags


def redact_pii(text: str) -> str:
    """``text`` with every PII match replaced by its placeholder (cards first, so their digit runs are not read as
    phone numbers)."""
    for m in sorted(_cards(text), key=lambda m: -m.start()):
        text = text[:m.start()] + PLACEHOLDER["pii_card"] + text[m.end():]
    text = _EMAIL.sub(PLACEHOLDER["pii_email"], text)
    text = _SSN.sub(PLACEHOLDER["pii_ssn"], text)
    text = _PHONE.sub(PLACEHOLDER["pii_phone"], text)
    return _IPV4.sub(PLACEHOLDER["pii_ip"], text)


def turn_hash(text: str) -> str:
    """A turn's stable content id: sha256 of its lowercased, whitespace-normalised text."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def load_blocklist(path: str | Path) -> frozenset[str]:
    """Turn hashes from a blocklist file (first tab-separated field per line; '#' lines are comments)."""
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            out.add(line.split("\t", 1)[0].strip())
    return frozenset(out)


def scan(files: list[Path], drop_flags: frozenset[str] = frozenset({"minor_explicit"})) -> tuple[dict[str, str], Counter]:
    """Scan JSONL files of transcript rows (a loom run's ``<out>/paths.jsonl`` or assembled documents: rows with a
    ``turns`` list).
    Returns {turn hash: comma-joined flags} for turns raising any of ``drop_flags``, and a count of every flag over
    distinct turns."""
    block: dict[str, str] = {}
    counts: Counter = Counter()
    seen: set[str] = set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for t in json.loads(line).get("turns", []):
                    h = turn_hash(t["text"])
                    if h in seen:
                        continue
                    seen.add(h)
                    flags = classify(t["text"])
                    counts.update(flags)
                    counts["turns"] += 1
                    if flags & drop_flags:
                        block[h] = ",".join(sorted(flags))
    return block, counts


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="flagged turn hashes (no text) from loom runs or docs files -> a blocklist")
    s.add_argument("inputs", nargs="+", help="loom run directories (their paths.jsonl) or JSONL files")
    s.add_argument("--out", required=True)
    s.add_argument("--drop_flags", default="minor_explicit", help="comma list of flags whose turns are blocklisted")
    a = ap.parse_args(argv)
    files = [Path(x) / "paths.jsonl" if Path(x).is_dir() else Path(x) for x in a.inputs]
    block, counts = scan(files, frozenset(f.strip() for f in a.drop_flags.split(",") if f.strip()))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("# turn sha256\tflags   (kotodama.posttrain.compliance scan; text deliberately not stored)\n")
        for h, flags in sorted(block.items()):
            fh.write(f"{h}\t{flags}\n")
    logger.info("%d distinct turns scanned; flag counts %s; %d turns blocklisted -> %s",
                counts.pop("turns", 0), dict(counts), len(block), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
