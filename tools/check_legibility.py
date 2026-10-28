#!/usr/bin/env python3
"""Legibility report for prose a stranger reads: README, model cards, results pages.

Report only; it never fails a build. It measures the processing-difficulty features of Clark, Martin & Mahowald,
"Why is AI-generated language so hard to understand?" (https://arxiv.org/abs/2610.29162), with the parser they use
(spaCy ``en_core_web_trf``), and compares each document with a human reference: Wikipedia text scored the same way
(``tools/legibility_reference.json``). Every flagged sentence is listed, so the copy can be fixed rather than only scored.

Features, per 1,000 words (the animacy and denominal judgements are lexicon heuristics where the paper uses a classifier):

  dep_len       mean dependency arc length, directly standardized over sentence-length bins 1-5, 6-10, 11-15, 16-20,
                21-30 words with the reference's bin weights (punctuation arcs and the root excluded; sentences over 30
                words are left out of the mean and listed)
  orc           object relative clauses with an overt relativizer ("the memo that the committee wrote")
  orc_reduced   object relatives without one ("the memo the committee wrote")
  stacked       chains of three or more common nouns in compound relations ("cache invalidation policy harness")
  inanimate     a mind or agency verb whose subject is inanimate ("the pipeline decided")
  denominal     a verb whose lemma the reference uses almost only as a noun
  rare          words with a wordfreq Zipf frequency below 3

Usage::

    python tools/check_legibility.py README.md CONTRIBUTING.md [--json report.json]
    python tools/check_legibility.py --build-ref tools/legibility_reference.json [--articles 3000 --parse 120]

Needs ``spacy``, the ``en_core_web_trf`` model and ``wordfreq`` (``--build-ref`` also needs ``datasets``). Without them
the report says so and exits 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "legibility_reference.json"
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30)]
FEATURES = ("orc", "orc_reduced", "stacked", "inanimate", "denominal")
ANIMATE_VERBS = frozenset("""
want decide think believe know feel try choose refuse hope expect prefer intend realize realise remember forget understand
learn insist agree admit claim argue wonder worry care like love hate notice recognize recognise assume imagine plan attempt
pretend promise wish desire fear trust doubt guess dare bother mind struggle manage seek aim strive tend ask say tell speak
listen hear see watch look smell taste consider judge reckon suppose suspect accept reject complain apologize thank praise
blame forgive obey ignore pay lie cheat win lose""".split())
PERSON_NOUNS = frozenset("""
person people man woman men women child children human humans user users reader readers author authors writer speaker
listener researcher researchers team student students teacher friend friends someone anyone everyone nobody somebody everybody
participant participants annotator annotators judge judges rater raters customer client operator developer developers engineer
engineers scientist scientists player character characters agent agents assistant group committee audience colleague partner
owner holder addressee narrator protagonist boy girl mother father parent parents""".split())
ANIMATE_PRON = frozenset("i you we he she they me us him her them who whoever one".split())
INANIMATE_PRON = frozenset("it this that which these those something nothing everything anything".split())
RELATIVIZERS = frozenset("that which who whom whose".split())
# heads of adverbial relatives ("the moment something breaks", "the way it is read"): not object relatives
ADVERBIAL_HEADS = frozenset("time times moment day year way place reason point extent".split())
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def markdown_prose(text: str) -> str:
    """The prose of a Markdown document: code, tables, images, link targets and markup removed."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " CODE ", text)
    text = re.sub(r"^\s*\|.*$", " ", text, flags=re.M)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    return re.sub(r"[*_]{1,3}", "", text)


def paragraphs(text: str, min_words: int) -> list[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9#*\-•>])", text)]
    return [p for p in paras if len(WORD.findall(p)) >= min_words]


def compound_chain(tok) -> list:
    """The compound-noun chain headed by tok (tok included), in text order."""
    out = [tok]
    for c in tok.children:
        if c.dep_ == "compound" and c.pos_ in ("NOUN", "PROPN"):
            out = compound_chain(c) + out
    return sorted(out, key=lambda t: t.i)


def animacy(subj) -> str:
    """animate, inanimate or unknown, by lexicon; a relativizer subject takes its antecedent's animacy."""
    t = subj.head.head if subj.lower_ in RELATIVIZERS and subj.head.dep_ == "relcl" else subj
    word = t.lower_
    if t.pos_ == "PRON":
        return "animate" if word in ANIMATE_PRON else "inanimate" if word in INANIMATE_PRON else "unknown"
    if t.ent_type_ in ("PERSON", "NORP") or t.lemma_.lower() in PERSON_NOUNS or word in PERSON_NOUNS:
        return "animate"
    if t.text == "CODE":
        return "inanimate"
    if t.pos_ in ("NOUN", "PROPN"):
        return "unknown" if t.ent_type_ == "ORG" else "inanimate"
    return "unknown"


def sentence_features(sent, denominal: frozenset) -> dict:
    words = [t for t in sent if not t.is_punct and not t.is_space]
    arcs = [abs(t.i - t.head.i) for t in words if t.dep_ != "ROOT" and t.head is not t]
    f: dict = {"n": len(words), "dep": sum(arcs) / len(arcs) if arcs else 0.0, **{k: [] for k in FEATURES}}
    for t in sent:
        if t.dep_ == "relcl" and t.pos_ in ("VERB", "AUX"):
            kids = list(t.children)
            relz = [c for c in kids if c.lower_ in RELATIVIZERS or c.tag_ in ("WDT", "WP", "WP$")]
            marked = any(x.i < t.i and (x.lower_ in RELATIVIZERS or x.tag_ in ("WDT", "WP", "WP$")) for x in t.subtree)
            has_subj = any(c.dep_ in ("nsubj", "nsubjpass") and c not in relz for c in kids)
            if [c for c in relz if c.dep_ in ("dobj", "pobj", "obj")] and has_subj:
                f["orc"].append(f"{t.head.text} … {t.text}")
            elif not relz and not marked and has_subj and not any(c.dep_ in ("dobj", "obj") for c in kids) \
                    and not any(c.dep_ == "advmod" and c.tag_ == "WRB" for c in kids) \
                    and t.head.lemma_.lower() not in ADVERBIAL_HEADS:
                f["orc_reduced"].append(f"{t.head.text} … {t.text}")
        if t.pos_ == "NOUN" and t.dep_ != "compound":
            chain = compound_chain(t)
            if sum(x.pos_ == "NOUN" for x in chain) >= 3:
                f["stacked"].append(" ".join(x.text for x in chain))
        if t.pos_ == "VERB" and t.lemma_.lower() in ANIMATE_VERBS and t.tag_ != "VBN" \
                and not (t.lemma_.lower() == "look" and any(c.lower_ == "like" for c in t.children)):
            subj = next((c for c in t.children if c.dep_ == "nsubj"), None)
            if subj is not None and animacy(subj) == "inanimate":
                f["inanimate"].append(f"{subj.text} {t.text}")
        if t.pos_ == "VERB" and t.lemma_.lower() in denominal:
            f["denominal"].append(t.text)
    return f


def load_nlp():
    import spacy
    return spacy.load("en_core_web_trf")


def build_reference(out: Path, articles: int, parse: int) -> None:
    """Wikipedia (wikimedia/wikipedia, 20231101.en) scored with the same features: bin weights, rates, denominal lemmas."""
    from datasets import load_dataset
    nlp = load_nlp()
    totals: Counter = Counter()
    bins: Counter = Counter()
    counts: Counter = Counter()
    pos_by_lemma: dict = defaultdict(Counter)
    for k, row in enumerate(load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)):
        if k >= min(articles, parse):
            break
        for doc in nlp.pipe(paragraphs(row["text"], 8)[:25], batch_size=8):
            for sent in doc.sents:
                f = sentence_features(sent, frozenset())
                if not f["n"]:
                    continue
                totals["words"] += f["n"]
                b = next((i for i, (lo, hi) in enumerate(BINS) if lo <= f["n"] <= hi), None)
                if b is not None:
                    bins[b] += 1
                    totals[f"dep_{b}"] += f["dep"]
                for key in ("orc", "orc_reduced", "stacked", "inanimate"):
                    counts[key] += len(f[key])
                for t in sent:
                    if t.pos_ in ("NOUN", "VERB"):
                        pos_by_lemma[t.lemma_.lower()][t.pos_] += 1
    n_bins = sum(bins.values())
    weights = [bins[i] / n_bins for i in range(len(BINS))]
    dep_by_bin = [totals[f"dep_{i}"] / max(1, bins[i]) for i in range(len(BINS))]
    ref = {"source": f"wikimedia/wikipedia 20231101.en, first {parse} articles", "parsed_words": totals["words"],
           "bin_weights": weights, "dep_by_bin": dep_by_bin, "dep_len": sum(w * d for w, d in zip(weights, dep_by_bin)),
           "rates": {k: v * 1000 / totals["words"] for k, v in counts.items()},
           "denominal_lemmas": sorted(l for l, c in pos_by_lemma.items()
                                      if c["NOUN"] + c["VERB"] >= 30 and c["NOUN"] / (c["NOUN"] + c["VERB"]) >= 0.95)}
    out.write_text(json.dumps(ref, indent=1))
    print(f"wrote {out}: dep_len {ref['dep_len']:.3f}, rates {ref['rates']}")


def report(paths: list[str], json_out: str | None, min_words: int) -> None:
    from wordfreq import zipf_frequency
    ref = json.loads(REFERENCE.read_text())
    denominal = frozenset(ref["denominal_lemmas"])
    nlp = load_nlp()
    results = []
    for path in paths:
        text = Path(path).read_text()
        prose = markdown_prose(text) if path.endswith((".md", ".markdown")) else text
        words, rare = 0, 0
        dep_bins: dict = defaultdict(list)
        flags: dict = defaultdict(list)
        long_sentences = []
        for doc in nlp.pipe(paragraphs(prose, min_words), batch_size=8):
            for sent in doc.sents:
                f = sentence_features(sent, denominal)
                if not f["n"]:
                    continue
                words += f["n"]
                b = next((i for i, (lo, hi) in enumerate(BINS) if lo <= f["n"] <= hi), None)
                if b is not None:
                    dep_bins[b].append(f["dep"])
                elif f["n"] > 30:
                    long_sentences.append(sent.text.strip())
                for key in FEATURES:
                    flags[key] += [{"match": m, "sentence": sent.text.strip()} for m in f[key]]
                rare += sum(zipf_frequency(w.lower(), "en") < 3 for w in WORD.findall(sent.text))
        if not words:
            continue
        w = ref["bin_weights"]
        dep = sum(w[i] * (sum(dep_bins[i]) / len(dep_bins[i]) if dep_bins[i] else ref["dep_by_bin"][i])
                  for i in range(len(BINS)))
        rates = {k: len(flags[k]) * 1000 / words for k in FEATURES}
        rates["rare"] = rare * 1000 / words
        res = {"path": path, "words": words, "dep_len": round(dep, 3), "dep_len_ratio": round(dep / ref["dep_len"], 2),
               "rates_per_1000": {k: round(v, 2) for k, v in rates.items()},
               "ratio_to_reference": {k: round(rates[k] / ref["rates"][k], 2) for k in ref["rates"] if ref["rates"][k]},
               "sentences_over_30_words": long_sentences, "flags": dict(flags)}
        results.append(res)
        print(f"\n{path}: {words} words, {len(long_sentences)} sentences over 30 words")
        print(f"  dep_len {dep:.2f}  (reference {ref['dep_len']:.2f}, x{res['dep_len_ratio']})")
        for k in (*FEATURES, "rare"):
            r = res["ratio_to_reference"].get(k)
            print(f"  {k:12s} {rates[k]:6.2f} per 1,000 words" + (f"  (x{r} the reference)" if r is not None else ""))
        for k in FEATURES:
            for item in flags[k][:5]:
                print(f"    [{k}] {item['match']}: {item['sentence'][:140]}")
    if json_out:
        Path(json_out).write_text(json.dumps({"reference": ref["source"], "results": results}, indent=1, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="Markdown or text files")
    ap.add_argument("--json", default=None, help="also write every flag and long sentence here")
    ap.add_argument("--min-words", type=int, default=4, help="skip paragraphs shorter than this (headings, labels)")
    ap.add_argument("--build-ref", type=Path, default=None, help="rebuild the reference into this file")
    ap.add_argument("--articles", type=int, default=3000)
    ap.add_argument("--parse", type=int, default=120)
    a = ap.parse_args()
    try:
        if a.build_ref:
            build_reference(a.build_ref, a.articles, a.parse)
        else:
            report(a.paths or ["README.md"], a.json, a.min_words)
    except (ImportError, OSError) as e:
        print(f"legibility report skipped: {e} (needs spacy, the en_core_web_trf model and wordfreq)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
