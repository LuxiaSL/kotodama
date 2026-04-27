#!/usr/bin/env python3
"""
Semantic text quality analysis for small language model (108M) generations.

Computes multi-layer quantitative metrics across:
  Layer 1: Lexical diversity & structure
  Layer 2: Semantic coherence & progression
  Layer 3: Structural complexity
  Layer 4: Repetition characterization
  Layer 5: Creativity & commitment

Usage:
    python scripts/analyze_text_quality.py
    python scripts/analyze_text_quality.py --input data/confound_check_eval.json --output outputs/text_quality_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── NLTK imports (lazy-downloaded) ──────────────────────────────────────────
try:
    import nltk
    from nltk import pos_tag, sent_tokenize, word_tokenize
    from nltk.tokenize import RegexpTokenizer

    _NLTK_RESOURCES = [
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker_tab",
        "words",
        "stopwords",
    ]
    for res in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{res}" if "punkt" in res else res)
        except LookupError:
            nltk.download(res, quiet=True)
except ImportError:
    print("ERROR: nltk is required. Install with: pip install nltk", file=sys.stderr)
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print(
        "ERROR: scikit-learn is required. Install with: pip install scikit-learn",
        file=sys.stderr,
    )
    sys.exit(1)

# Silence sklearn warnings about empty vocabularies on degenerate text
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Constants ───────────────────────────────────────────────────────────────

DISCOURSE_MARKERS = {
    "however",
    "therefore",
    "because",
    "although",
    "furthermore",
    "moreover",
    "nevertheless",
    "consequently",
    "meanwhile",
    "nonetheless",
    "otherwise",
    "hence",
    "thus",
    "instead",
    "accordingly",
    "similarly",
    "likewise",
    "conversely",
    "specifically",
    "indeed",
    "certainly",
    "evidently",
    "incidentally",
    "additionally",
    "alternatively",
    "subsequently",
    "regardless",
    "apparently",
    "presumably",
    "supposedly",
    "notably",
}

TEMPORAL_MARKERS = {
    "then",
    "after",
    "before",
    "when",
    "while",
    "suddenly",
    "finally",
    "meanwhile",
    "afterwards",
    "later",
    "earlier",
    "next",
    "soon",
    "eventually",
    "immediately",
    "once",
    "during",
    "until",
    "already",
    "since",
    "recently",
    "now",
    "previously",
    "initially",
    "first",
    "last",
}

DIALOGUE_MARKERS = {
    "said",
    "asked",
    "replied",
    "answered",
    "exclaimed",
    "whispered",
    "shouted",
    "muttered",
    "declared",
    "insisted",
    "responded",
    "remarked",
    "suggested",
    "explained",
    "told",
    "cried",
    "called",
    "announced",
}

# POS tag sets (Penn Treebank)
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}
ADV_TAGS = {"RB", "RBR", "RBS"}
PRONOUN_TAGS = {"PRP", "PRP$", "WP", "WP$"}
PROPER_NOUN_TAGS = {"NNP", "NNPS"}

# Regex for fabricated-but-specific claims
RE_DR_NAME = re.compile(r"\bDr\.?\s+[A-Z][a-z]+", re.UNICODE)
RE_PROF_NAME = re.compile(r"\bProf(?:essor)?\.?\s+[A-Z][a-z]+", re.UNICODE)
RE_UNIVERSITY = re.compile(
    r"\b(?:University|Institute|College)\s+of\s+[A-Z][a-z]+", re.UNICODE
)
RE_PERCENTAGE = re.compile(r"\b\d+(?:\.\d+)?%")
RE_YEAR = re.compile(r"\b(?:1[89]\d{2}|20[0-2]\d)\b")
RE_SPECIFIC_NUMBER = re.compile(r"\b\d{2,}\b")  # 2+ digit numbers

_word_tokenizer = RegexpTokenizer(r"\w+")


# ── Dataclasses for results ─────────────────────────────────────────────────


@dataclass
class LexicalMetrics:
    ttr_50: float = 0.0
    ttr_100: float = 0.0
    ttr_200: float = 0.0
    ttr_500: float = 0.0
    ttr_all: float = 0.0
    hapax_ratio: float = 0.0
    yules_k: float = 0.0
    honores_r: float = 0.0
    pos_noun_ratio: float = 0.0
    pos_verb_ratio: float = 0.0
    pos_adj_ratio: float = 0.0
    pos_adv_ratio: float = 0.0
    pos_pronoun_ratio: float = 0.0
    sent_len_mean: float = 0.0
    sent_len_std: float = 0.0
    sent_len_min: float = 0.0
    sent_len_max: float = 0.0
    sent_count: int = 0
    punct_types: int = 0
    punct_ratio: float = 0.0


@dataclass
class CoherenceMetrics:
    topic_drift_mean: float = 0.0
    topic_drift_std: float = 0.0
    topic_drift_trend: float = 0.0  # slope of consecutive similarities
    novelty_mean: float = 0.0
    novelty_std: float = 0.0
    novelty_trend: float = 0.0  # slope of novelty curve
    novelty_final_vs_initial: float = 0.0
    entity_count: int = 0
    entity_intro_rate: float = 0.0  # new entities per 100 tokens
    referential_density: float = 0.0  # pronoun-to-noun ratio


@dataclass
class StructuralMetrics:
    clause_depth_mean: float = 0.0
    clause_depth_std: float = 0.0
    discourse_marker_count: int = 0
    discourse_marker_density: float = 0.0  # per 100 tokens
    structural_variety_entropy: float = 0.0
    pct_declarative: float = 0.0
    pct_interrogative: float = 0.0
    pct_exclamatory: float = 0.0
    pct_imperative: float = 0.0
    register_consistency_mean: float = 0.0
    register_consistency_std: float = 0.0


@dataclass
class RepetitionMetrics:
    repetition_onset_5gram: int = -1  # -1 = no repetition
    repetition_unit_size_mean: float = 0.0
    repetition_unit_size_max: int = 0
    unique_1gram_ratio: float = 0.0
    unique_2gram_ratio: float = 0.0
    unique_3gram_ratio: float = 0.0
    unique_5gram_ratio: float = 0.0
    unique_8gram_ratio: float = 0.0
    post_rep_novel_tokens: int = 0  # novel tokens after first repeat
    post_rep_novel_ratio: float = 0.0


@dataclass
class CreativityMetrics:
    specificity_score: float = 0.0
    temporal_marker_count: int = 0
    temporal_marker_density: float = 0.0
    dialogue_marker_count: int = 0
    dialogue_marker_density: float = 0.0
    quote_count: int = 0
    fabrication_density: float = 0.0  # fabricated claims per 100 tokens
    topical_unexpectedness: float = 0.0  # distance from mean response


@dataclass
class SampleResult:
    run: str = ""
    prompt: str = ""
    lexical: LexicalMetrics = field(default_factory=LexicalMetrics)
    coherence: CoherenceMetrics = field(default_factory=CoherenceMetrics)
    structural: StructuralMetrics = field(default_factory=StructuralMetrics)
    repetition: RepetitionMetrics = field(default_factory=RepetitionMetrics)
    creativity: CreativityMetrics = field(default_factory=CreativityMetrics)


# ── Utility functions ───────────────────────────────────────────────────────


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division with zero-safety."""
    if b == 0:
        return default
    return a / b


def compute_ttr(tokens: list[str], window: int) -> float:
    """Type-token ratio for first `window` tokens."""
    if len(tokens) == 0:
        return 0.0
    subset = tokens[: min(window, len(tokens))]
    if len(subset) == 0:
        return 0.0
    return len(set(subset)) / len(subset)


def compute_yules_k(tokens: list[str]) -> float:
    """
    Yule's K — measures vocabulary richness.
    Lower K = richer vocabulary. Range roughly 50-300 for natural text.
    Formula: K = 10^4 * (M2 - N) / N^2
    where M2 = sum(i^2 * V(i)) over frequency spectrum V.
    """
    if len(tokens) < 2:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    # spectrum: V(i) = number of types with frequency i
    spectrum = Counter(freq.values())
    m2 = sum(i * i * v for i, v in spectrum.items())
    denom = n * n
    if denom == 0:
        return 0.0
    k = 1e4 * (m2 - n) / denom
    return max(k, 0.0)


def compute_honores_r(tokens: list[str]) -> float:
    """
    Honore's R — rewards hapax legomena (words appearing once).
    R = 100 * log(N) / (1 - V1/V)
    where V1 = hapax count, V = vocabulary size, N = total tokens.
    Higher R = richer vocabulary.
    """
    if len(tokens) < 2:
        return 0.0
    n = len(tokens)
    freq = Counter(tokens)
    v = len(freq)
    v1 = sum(1 for c in freq.values() if c == 1)
    if v == 0 or v1 == v:
        return 0.0  # degenerate case: all hapax or no vocab
    denom = 1.0 - (v1 / v)
    if abs(denom) < 1e-10:
        return 0.0
    return 100.0 * math.log(n + 1) / denom


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract all n-grams from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def classify_sentence(sent: str) -> str:
    """Classify a sentence by its terminal punctuation / structure."""
    stripped = sent.strip()
    if not stripped:
        return "declarative"
    if stripped.endswith("?"):
        return "interrogative"
    if stripped.endswith("!"):
        return "exclamatory"
    # Simple imperative detection: starts with a verb-like word
    first_word = stripped.split()[0].lower() if stripped.split() else ""
    imperative_starters = {
        "do",
        "don't",
        "let",
        "go",
        "come",
        "take",
        "give",
        "make",
        "put",
        "get",
        "keep",
        "set",
        "try",
        "look",
        "listen",
        "note",
        "consider",
        "imagine",
        "think",
        "remember",
        "stop",
        "start",
        "begin",
        "notice",
    }
    if first_word in imperative_starters:
        return "imperative"
    return "declarative"


def entropy(probs: list[float]) -> float:
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def linear_slope(values: list[float]) -> float:
    """Slope of a simple linear fit to sequential values."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def estimate_clause_depth(sent: str, tagged: list[tuple[str, str]]) -> int:
    """
    Heuristic clause depth based on subordination cues.
    Counts nested subordinating conjunctions, relative pronouns, commas
    introducing clauses, and parenthetical structures.
    This is a cheap proxy for real dependency parse depth.
    """
    subordinators = {
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "while",
        "although",
        "because",
        "since",
        "if",
        "unless",
        "until",
        "after",
        "before",
        "as",
        "though",
        "whereas",
    }
    depth = 1  # base clause
    for word, tag in tagged:
        lower = word.lower()
        if lower in subordinators and tag in {"IN", "WDT", "WP", "WRB"}:
            depth += 1
    # Parenthetical nesting
    depth += sent.count("(")
    return depth


def detect_fabrications(text: str) -> int:
    """Count patterns suggesting fabricated-but-specific claims."""
    count = 0
    count += len(RE_DR_NAME.findall(text))
    count += len(RE_PROF_NAME.findall(text))
    count += len(RE_UNIVERSITY.findall(text))
    count += len(RE_PERCENTAGE.findall(text))
    # Years — be conservative, only count if they look like historical claims
    year_matches = RE_YEAR.findall(text)
    count += len(year_matches)
    return count


# ── Core analysis functions ─────────────────────────────────────────────────


def analyze_lexical(text: str) -> LexicalMetrics:
    """Layer 1: Lexical diversity and structure."""
    m = LexicalMetrics()

    # Tokenize
    word_tokens = _word_tokenizer.tokenize(text.lower())
    if len(word_tokens) == 0:
        return m

    # TTR at multiple windows
    m.ttr_50 = compute_ttr(word_tokens, 50)
    m.ttr_100 = compute_ttr(word_tokens, 100)
    m.ttr_200 = compute_ttr(word_tokens, 200)
    m.ttr_500 = compute_ttr(word_tokens, 500)
    m.ttr_all = compute_ttr(word_tokens, len(word_tokens))

    # Hapax legomena
    freq = Counter(word_tokens)
    hapax = sum(1 for c in freq.values() if c == 1)
    m.hapax_ratio = safe_div(hapax, len(word_tokens))

    # Vocabulary richness
    m.yules_k = compute_yules_k(word_tokens)
    m.honores_r = compute_honores_r(word_tokens)

    # POS distribution
    try:
        # Use original-case tokens for POS tagging
        orig_tokens = _word_tokenizer.tokenize(text)
        tagged = pos_tag(orig_tokens)
        tag_counts: Counter[str] = Counter(tag for _, tag in tagged)
        total_tags = sum(tag_counts.values())
        m.pos_noun_ratio = safe_div(
            sum(tag_counts.get(t, 0) for t in NOUN_TAGS), total_tags
        )
        m.pos_verb_ratio = safe_div(
            sum(tag_counts.get(t, 0) for t in VERB_TAGS), total_tags
        )
        m.pos_adj_ratio = safe_div(
            sum(tag_counts.get(t, 0) for t in ADJ_TAGS), total_tags
        )
        m.pos_adv_ratio = safe_div(
            sum(tag_counts.get(t, 0) for t in ADV_TAGS), total_tags
        )
        m.pos_pronoun_ratio = safe_div(
            sum(tag_counts.get(t, 0) for t in PRONOUN_TAGS), total_tags
        )
    except Exception:
        pass  # POS tagging can fail on very degenerate text

    # Sentence length distribution
    sentences = sent_tokenize(text)
    m.sent_count = len(sentences)
    if sentences:
        sent_lens = [len(_word_tokenizer.tokenize(s)) for s in sentences]
        sent_lens = [sl for sl in sent_lens if sl > 0]
        if sent_lens:
            m.sent_len_mean = float(np.mean(sent_lens))
            m.sent_len_std = float(np.std(sent_lens))
            m.sent_len_min = float(min(sent_lens))
            m.sent_len_max = float(max(sent_lens))

    # Punctuation diversity
    punct_chars = [c for c in text if c in '.,;:!?—–-()[]{}"\'/…']
    punct_types = set(punct_chars)
    m.punct_types = len(punct_types)
    m.punct_ratio = safe_div(len(punct_chars), len(text))

    return m


def analyze_coherence(text: str) -> CoherenceMetrics:
    """Layer 2: Semantic coherence and progression."""
    m = CoherenceMetrics()

    sentences = sent_tokenize(text)
    word_tokens = _word_tokenizer.tokenize(text)
    n_tokens = len(word_tokens)

    if n_tokens == 0 or len(sentences) < 2:
        return m

    # ── Topic drift: chunk-level TF-IDF cosine similarity ──
    n_chunks = 5
    chunk_size = max(1, len(sentences) // n_chunks)
    chunks: list[str] = []
    for i in range(0, len(sentences), chunk_size):
        chunk_text = " ".join(sentences[i : i + chunk_size])
        if chunk_text.strip():
            chunks.append(chunk_text)

    if len(chunks) >= 2:
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(chunks)
            consecutive_sims: list[float] = []
            for i in range(len(chunks) - 1):
                sim = cosine_similarity(tfidf_matrix[i : i + 1], tfidf_matrix[i + 1 : i + 2])[
                    0, 0
                ]
                consecutive_sims.append(float(sim))
            if consecutive_sims:
                m.topic_drift_mean = float(np.mean(consecutive_sims))
                m.topic_drift_std = float(np.std(consecutive_sims))
                m.topic_drift_trend = linear_slope(consecutive_sims)
        except ValueError:
            pass  # empty vocabulary after stop-word removal

    # ── Semantic novelty curve: per-sentence similarity to all prior ──
    if len(sentences) >= 3:
        try:
            sent_vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
            sent_tfidf = sent_vectorizer.fit_transform(sentences)
            novelty_scores: list[float] = []
            for i in range(1, len(sentences)):
                prior_matrix = sent_tfidf[:i]
                current = sent_tfidf[i : i + 1]
                sims = cosine_similarity(current, prior_matrix)[0]
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                novelty_scores.append(1.0 - max_sim)
            if novelty_scores:
                m.novelty_mean = float(np.mean(novelty_scores))
                m.novelty_std = float(np.std(novelty_scores))
                m.novelty_trend = linear_slope(novelty_scores)
                # Compare first quarter vs last quarter
                q = max(1, len(novelty_scores) // 4)
                initial_mean = float(np.mean(novelty_scores[:q]))
                final_mean = float(np.mean(novelty_scores[-q:]))
                m.novelty_final_vs_initial = final_mean - initial_mean
        except ValueError:
            pass

    # ── Entity tracking (regex-based for robustness) ──
    # Proper nouns, specific numbers, date-like patterns
    entities_seen: set[str] = set()
    entity_positions: list[int] = []

    try:
        orig_tokens = _word_tokenizer.tokenize(text)
        tagged = pos_tag(orig_tokens)
        for idx, (word, tag) in enumerate(tagged):
            if tag in PROPER_NOUN_TAGS and len(word) > 1:
                if word.lower() not in entities_seen:
                    entities_seen.add(word.lower())
                    entity_positions.append(idx)
    except Exception:
        pass

    m.entity_count = len(entities_seen)
    m.entity_intro_rate = safe_div(len(entities_seen) * 100, n_tokens)

    # ── Referential density: pronoun / noun ratio ──
    try:
        orig_tokens = _word_tokenizer.tokenize(text)
        tagged = pos_tag(orig_tokens)
        n_pronouns = sum(1 for _, tag in tagged if tag in PRONOUN_TAGS)
        n_nouns = sum(1 for _, tag in tagged if tag in NOUN_TAGS)
        m.referential_density = safe_div(n_pronouns, n_nouns)
    except Exception:
        pass

    return m


def analyze_structural(text: str) -> StructuralMetrics:
    """Layer 3: Structural complexity."""
    m = StructuralMetrics()

    sentences = sent_tokenize(text)
    word_tokens = _word_tokenizer.tokenize(text)
    n_tokens = len(word_tokens)

    if n_tokens == 0 or len(sentences) == 0:
        return m

    # ── Clause depth (heuristic) ──
    depths: list[int] = []
    for sent in sentences:
        try:
            sent_tokens = _word_tokenizer.tokenize(sent)
            if len(sent_tokens) == 0:
                continue
            tagged = pos_tag(sent_tokens)
            d = estimate_clause_depth(sent, tagged)
            depths.append(d)
        except Exception:
            depths.append(1)

    if depths:
        m.clause_depth_mean = float(np.mean(depths))
        m.clause_depth_std = float(np.std(depths))

    # ── Discourse markers ──
    lower_tokens = [t.lower() for t in word_tokens]
    dm_count = sum(1 for t in lower_tokens if t in DISCOURSE_MARKERS)
    m.discourse_marker_count = dm_count
    m.discourse_marker_density = safe_div(dm_count * 100, n_tokens)

    # ── Structural variety ──
    classifications = [classify_sentence(s) for s in sentences]
    class_counts = Counter(classifications)
    total_sents = len(classifications)
    if total_sents > 0:
        probs = [c / total_sents for c in class_counts.values()]
        m.structural_variety_entropy = entropy(probs)
        m.pct_declarative = safe_div(class_counts.get("declarative", 0), total_sents)
        m.pct_interrogative = safe_div(
            class_counts.get("interrogative", 0), total_sents
        )
        m.pct_exclamatory = safe_div(class_counts.get("exclamatory", 0), total_sents)
        m.pct_imperative = safe_div(class_counts.get("imperative", 0), total_sents)

    # ── Register consistency: vocab overlap between successive chunks ──
    n_chunks = 5
    chunk_size = max(1, len(sentences) // n_chunks)
    chunk_vocabs: list[set[str]] = []
    for i in range(0, len(sentences), chunk_size):
        chunk_words = set()
        for s in sentences[i : i + chunk_size]:
            chunk_words.update(w.lower() for w in _word_tokenizer.tokenize(s))
        if chunk_words:
            chunk_vocabs.append(chunk_words)

    if len(chunk_vocabs) >= 2:
        overlaps: list[float] = []
        for i in range(len(chunk_vocabs) - 1):
            a, b = chunk_vocabs[i], chunk_vocabs[i + 1]
            union = a | b
            if len(union) == 0:
                continue
            overlap = len(a & b) / len(union)
            overlaps.append(overlap)
        if overlaps:
            m.register_consistency_mean = float(np.mean(overlaps))
            m.register_consistency_std = float(np.std(overlaps))

    return m


def analyze_repetition(text: str) -> RepetitionMetrics:
    """Layer 4: Repetition characterization."""
    m = RepetitionMetrics()

    tokens = _word_tokenizer.tokenize(text.lower())
    n = len(tokens)
    if n == 0:
        return m

    # ── Unique n-gram ratios ──
    for ng_size, attr in [(1, "unique_1gram_ratio"), (2, "unique_2gram_ratio"),
                          (3, "unique_3gram_ratio"), (5, "unique_5gram_ratio"),
                          (8, "unique_8gram_ratio")]:
        ngrams = get_ngrams(tokens, ng_size)
        if ngrams:
            setattr(m, attr, safe_div(len(set(ngrams)), len(ngrams)))

    # ── Repetition onset (first position where a 5-gram has been seen before) ──
    seen_5grams: dict[tuple[str, ...], int] = {}
    first_repeat_pos = -1
    first_repeat_gram: Optional[tuple[str, ...]] = None
    for i in range(n - 4):
        gram = tuple(tokens[i : i + 5])
        if gram in seen_5grams:
            first_repeat_pos = i
            first_repeat_gram = gram
            break
        seen_5grams[gram] = i

    m.repetition_onset_5gram = first_repeat_pos

    # ── Repetition unit size: find repeated segments of varying lengths ──
    if first_repeat_pos >= 0:
        # Scan for the longest repeated unit starting at the repeat position
        repeat_unit_sizes: list[int] = []
        for unit_len in range(2, min(50, n // 2)):
            ngrams = get_ngrams(tokens, unit_len)
            ngram_counts = Counter(ngrams)
            repeated = [ng for ng, c in ngram_counts.items() if c > 1]
            if repeated:
                repeat_unit_sizes.append(unit_len)

        if repeat_unit_sizes:
            m.repetition_unit_size_mean = float(np.mean(repeat_unit_sizes))
            m.repetition_unit_size_max = max(repeat_unit_sizes)

        # ── Post-repetition recovery ──
        # After the first 5-gram repeat, measure unique-5gram ratio in the
        # post-repeat zone. This captures whether the model recovers from
        # the loop or stays stuck. Also count tokens participating in
        # truly novel 5-grams (never seen before in the entire text up to
        # that point, tracked with a running set).
        post_tokens = tokens[first_repeat_pos:]
        if len(post_tokens) >= 5:
            post_grams = get_ngrams(post_tokens, 5)
            if post_grams:
                post_unique_ratio = safe_div(
                    len(set(post_grams)), len(post_grams)
                )
            else:
                post_unique_ratio = 0.0

            # Running novelty: track all 5-grams seen so far, count novel ones
            running_seen: set[tuple[str, ...]] = set(
                get_ngrams(tokens[:first_repeat_pos], 5)
            )
            novel_count = 0
            for gram in post_grams:
                if gram not in running_seen:
                    novel_count += 1
                running_seen.add(gram)
            m.post_rep_novel_tokens = novel_count
            m.post_rep_novel_ratio = post_unique_ratio

    return m


def analyze_creativity(
    text: str,
    all_continuations_for_prompt: Optional[list[str]] = None,
) -> CreativityMetrics:
    """Layer 5: Creativity and commitment."""
    m = CreativityMetrics()

    word_tokens = _word_tokenizer.tokenize(text)
    n_tokens = len(word_tokens)
    if n_tokens == 0:
        return m

    lower_tokens = [t.lower() for t in word_tokens]

    # ── Specificity: concrete vs abstract nouns (heuristic via POS + patterns) ──
    try:
        tagged = pos_tag(word_tokens)
        proper_nouns = sum(1 for _, tag in tagged if tag in PROPER_NOUN_TAGS)
        common_nouns = sum(1 for _, tag in tagged if tag in {"NN", "NNS"})
        numbers = sum(1 for _, tag in tagged if tag == "CD")
        # Specificity = (proper nouns + numbers) / (common nouns + proper nouns + numbers + 1)
        concrete = proper_nouns + numbers
        total_noun_like = common_nouns + proper_nouns + numbers
        m.specificity_score = safe_div(concrete, total_noun_like + 1)
    except Exception:
        pass

    # ── Narrative indicators ──
    m.temporal_marker_count = sum(1 for t in lower_tokens if t in TEMPORAL_MARKERS)
    m.temporal_marker_density = safe_div(m.temporal_marker_count * 100, n_tokens)
    m.dialogue_marker_count = sum(1 for t in lower_tokens if t in DIALOGUE_MARKERS)
    m.dialogue_marker_density = safe_div(m.dialogue_marker_count * 100, n_tokens)
    m.quote_count = text.count('"') // 2 + text.count("\u201c")  # pairs of quotes

    # ── Fabrication density ──
    fab_count = detect_fabrications(text)
    m.fabrication_density = safe_div(fab_count * 100, n_tokens)

    # ── Topical unexpectedness: distance from mean response ──
    if all_continuations_for_prompt and len(all_continuations_for_prompt) >= 2:
        try:
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            all_tfidf = vectorizer.fit_transform(all_continuations_for_prompt)
            # Mean vector = centroid of all responses
            mean_vec = np.asarray(all_tfidf.mean(axis=0))
            if mean_vec.ndim == 1:
                mean_vec = mean_vec.reshape(1, -1)
            # Find this text's index
            idx = None
            for i, cont in enumerate(all_continuations_for_prompt):
                if cont == text:
                    idx = i
                    break
            if idx is not None:
                sim = cosine_similarity(all_tfidf[idx : idx + 1], mean_vec)[0, 0]
                m.topical_unexpectedness = 1.0 - float(sim)
        except ValueError:
            pass

    return m


# ── Aggregation and formatting ──────────────────────────────────────────────


def metrics_to_flat_dict(result: SampleResult) -> dict[str, Any]:
    """Flatten all nested metrics into a single dict with prefixed keys."""
    flat: dict[str, Any] = {"run": result.run, "prompt": result.prompt}
    for layer_name in ["lexical", "coherence", "structural", "repetition", "creativity"]:
        layer_obj = getattr(result, layer_name)
        for k, v in asdict(layer_obj).items():
            flat[f"{layer_name}.{k}"] = v
    return flat


def compute_model_profiles(
    results: list[SampleResult],
) -> dict[str, dict[str, float]]:
    """Compute per-model average profile across all prompts."""
    model_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        model_metrics[r.run].append(metrics_to_flat_dict(r))

    profiles: dict[str, dict[str, float]] = {}
    for run, flat_list in model_metrics.items():
        profile: dict[str, float] = {}
        # Get all numeric keys
        numeric_keys = [
            k
            for k in flat_list[0]
            if k not in ("run", "prompt") and isinstance(flat_list[0][k], (int, float))
        ]
        for key in numeric_keys:
            values = [d[key] for d in flat_list]
            profile[key] = float(np.mean(values))
        profiles[run] = profile
    return profiles


def find_most_distinctive(
    profiles: dict[str, dict[str, float]],
) -> dict[str, list[tuple[str, float, float]]]:
    """
    For each model, find the metrics where it differs most from the mean
    of all models (in terms of z-score magnitude).
    Returns top 5 most distinctive metrics per model.
    """
    if len(profiles) < 2:
        return {}

    # Get all metric keys
    all_keys = list(next(iter(profiles.values())).keys())
    runs = list(profiles.keys())

    distinctive: dict[str, list[tuple[str, float, float]]] = {}
    for run in runs:
        z_scores: list[tuple[str, float, float]] = []  # (metric, z_score, value)
        for key in all_keys:
            values = [profiles[r][key] for r in runs]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            if std_val < 1e-10:
                continue  # no variation across models
            z = (profiles[run][key] - mean_val) / std_val
            z_scores.append((key, z, profiles[run][key]))

        # Sort by absolute z-score
        z_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        distinctive[run] = z_scores[:5]

    return distinctive


# ── Pretty printing ─────────────────────────────────────────────────────────


def print_comparison_table(
    profiles: dict[str, dict[str, float]],
    section_name: str,
    keys: list[str],
) -> None:
    """Print a nicely formatted comparison table for a subset of metrics."""
    runs = sorted(profiles.keys())
    if not runs or not keys:
        return

    print(f"\n{'=' * 80}")
    print(f"  {section_name}")
    print(f"{'=' * 80}")

    # Header
    col_width = 14
    metric_width = 36
    header = f"{'Metric':<{metric_width}}"
    for run in runs:
        # Truncate run name if needed
        label = run[:col_width]
        header += f"{label:>{col_width}}"
    print(header)
    print("-" * (metric_width + col_width * len(runs)))

    for key in keys:
        short_key = key.split(".", 1)[-1] if "." in key else key
        row = f"{short_key:<{metric_width}}"
        for run in runs:
            val = profiles[run].get(key, 0.0)
            if isinstance(val, float):
                if abs(val) >= 100:
                    row += f"{val:>{col_width}.1f}"
                elif abs(val) >= 1:
                    row += f"{val:>{col_width}.3f}"
                else:
                    row += f"{val:>{col_width}.4f}"
            else:
                row += f"{val:>{col_width}}"
        print(row)


def print_full_report(
    profiles: dict[str, dict[str, float]],
    distinctive: dict[str, list[tuple[str, float, float]]],
) -> None:
    """Print the complete analysis report."""
    # Group keys by layer
    lexical_keys = [k for k in next(iter(profiles.values())) if k.startswith("lexical.")]
    coherence_keys = [k for k in next(iter(profiles.values())) if k.startswith("coherence.")]
    structural_keys = [k for k in next(iter(profiles.values())) if k.startswith("structural.")]
    repetition_keys = [k for k in next(iter(profiles.values())) if k.startswith("repetition.")]
    creativity_keys = [k for k in next(iter(profiles.values())) if k.startswith("creativity.")]

    print_comparison_table(profiles, "Layer 1: Lexical Diversity & Structure", lexical_keys)
    print_comparison_table(profiles, "Layer 2: Semantic Coherence & Progression", coherence_keys)
    print_comparison_table(profiles, "Layer 3: Structural Complexity", structural_keys)
    print_comparison_table(profiles, "Layer 4: Repetition Characterization", repetition_keys)
    print_comparison_table(profiles, "Layer 5: Creativity & Commitment", creativity_keys)

    # Most distinctive metrics
    runs = sorted(distinctive.keys())
    print(f"\n{'=' * 80}")
    print("  Most Distinctive Metrics Per Model (by z-score)")
    print(f"{'=' * 80}")
    for run in runs:
        print(f"\n  {run}:")
        for metric, z, val in distinctive[run]:
            direction = "HIGH" if z > 0 else "LOW"
            short_metric = metric.split(".", 1)[-1] if "." in metric else metric
            print(f"    {direction:>4}  z={z:+.2f}  {short_metric:<35} = {val:.4f}")


# ── Main pipeline ───────────────────────────────────────────────────────────


def load_data(input_path: Path) -> list[dict[str, Any]]:
    """Load the evaluation data from JSON.

    Supports two formats:
    - Legacy: top-level list of {"run": ..., "samples": [...]}
    - Track 3: {"config": ..., "runs": [{"name": ..., "samples": [...]}]}

    Always returns the legacy-style list for downstream compatibility.
    """
    with open(input_path) as f:
        data = json.load(f)

    # Track 3 format: {"config": ..., "runs": [...]}
    if isinstance(data, dict) and "runs" in data:
        runs = data["runs"]
        # Normalize "name" → "run" for downstream code
        for entry in runs:
            if "name" in entry and "run" not in entry:
                entry["run"] = entry["name"]
        return runs

    # Legacy format: top-level list
    if isinstance(data, list):
        return data

    raise ValueError(f"Unrecognized data format (top-level type: {type(data).__name__})")


def run_analysis(data: list[dict[str, Any]]) -> list[SampleResult]:
    """Run all analysis layers on the data."""
    results: list[SampleResult] = []

    # Pre-index: group continuations by prompt for topical unexpectedness
    prompt_continuations: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for entry in data:
        run_name = entry["run"]
        for sample in entry.get("samples", []):
            prompt = sample["prompt"]
            continuation = sample["continuation"]
            prompt_continuations[prompt].append((run_name, continuation))

    total_samples = sum(len(e.get("samples", [])) for e in data)
    processed = 0

    for entry in data:
        run_name = entry["run"]
        for sample in entry.get("samples", []):
            prompt = sample["prompt"]
            text = sample["continuation"]
            processed += 1

            print(
                f"  [{processed}/{total_samples}] {run_name} | "
                f"{prompt[:40]}...",
                flush=True,
            )

            # Gather all continuations for same prompt (for unexpectedness)
            all_conts = [cont for _, cont in prompt_continuations[prompt]]

            result = SampleResult(run=run_name, prompt=prompt)
            result.lexical = analyze_lexical(text)
            result.coherence = analyze_coherence(text)
            result.structural = analyze_structural(text)
            result.repetition = analyze_repetition(text)
            result.creativity = analyze_creativity(text, all_conts)

            results.append(result)

    return results


def save_results(
    results: list[SampleResult],
    profiles: dict[str, dict[str, float]],
    distinctive: dict[str, list[tuple[str, float, float]]],
    output_path: Path,
) -> None:
    """Save all results to a JSON file."""
    output: dict[str, Any] = {
        "per_sample": [metrics_to_flat_dict(r) for r in results],
        "model_profiles": profiles,
        "most_distinctive": {
            run: [
                {"metric": m, "z_score": round(z, 4), "value": round(v, 6)}
                for m, z, v in items
            ]
            for run, items in distinctive.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic text quality analysis for small LM generations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/confound_check_eval.json"),
        help="Path to evaluation JSON (default: data/confound_check_eval.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/text_quality_results.json"),
        help="Path for JSON output (default: outputs/text_quality_results.json)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from: {args.input}")
    data = load_data(args.input)
    print(f"Found {len(data)} runs, {sum(len(e.get('samples', [])) for e in data)} total samples")

    print("\nRunning analysis...")
    results = run_analysis(data)

    print("\nComputing model profiles...")
    profiles = compute_model_profiles(results)
    distinctive = find_most_distinctive(profiles)

    print_full_report(profiles, distinctive)
    save_results(results, profiles, distinctive, args.output)


if __name__ == "__main__":
    main()
