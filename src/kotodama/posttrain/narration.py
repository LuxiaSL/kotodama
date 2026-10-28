"""Is an unlabelled paragraph narration, or the speaker still talking?

In a ``Name: text`` transcript a turn can run across a blank line. The paragraph after it is either the same
character continuing (a long speech, a letter read aloud, a list) or a narrator stepping out of the dialogue (a stage
direction, a scene change, a time-skip, third-person description). ``classify_paragraph`` decides from surface cues:

  narration cues  — opens with a scene/stage marker ( "(", "[", "*", "Later", "Years later", "Meanwhile", "As …",
                    "When …"), third-person subjects (he / she / they / a cast name + a verb), past or present
                    narrative tense with no first/second person outside quotes;
  speech cues     — opens with a quote mark or lowercase continuation, first/second person (I / you / we) outside
                    quotes, addresses someone (a vocative, an imperative, a question), or continues a list/letter.

Returns ("narr" | "speech" | "unclear", the cues that fired). It is a heuristic gate with a measured error (see the
precision/recall study that calibrates it); callers that need certainty treat "unclear" as speech.

    split_turn(text) -> [paragraphs]; classify_paragraph(paragraph, previous, cast) -> (label, cues)
"""

from __future__ import annotations

import re

_QUOTED = re.compile(r"“[^”]*”|\"[^\"]*\"|‘[^’]*’", re.S)
_STAGE_OPEN = re.compile(r"^\s*(\(|\[|\*(?!\*)|-{2,}\s*$|\*\s*\*\s*\*)")
_SCENE_OPEN = re.compile(r"^\s*(Later|Meanwhile|Years later|Hours later|Days later|Moments later|The next (morning|day|"
                         r"night)|That (night|evening|morning)|Afterwards?|Some time later|Soon|Then|As |When |While |"
                         r"Outside|Inside|Back at|At the|In the)\b")
_THIRD = re.compile(r"\b(he|she|they|his|her|their|him|them)\b", re.I)
_FIRST_SECOND = re.compile(r"\b(I|I'm|I’m|I've|I’ve|I'll|I’ll|I'd|I’d|me|my|mine|myself|we|we're|we’re|us|our|"
                           r"you|you're|you’re|your|yours|yourself)\b")
_ACTION_VERB = re.compile(r"\b(walks|walked|looks|looked|sits|sat|stands|stood|turns|turned|takes|took|says|said|"
                          r"nods|nodded|leaves|left|enters|entered|opens|opened|smiles|smiled|laughs|laughed|sighs|"
                          r"sighed|picks|picked|pauses|paused|watches|watched|steps|stepped|reaches|reached|is|was|are|"
                          r"were)\b")
_LIST = re.compile(r"^\s*(\d+[.)]|[-•]\s)")
_IMPERATIVE_OPEN = re.compile(r"^\s*(Look|Listen|Come|Go|Stop|Wait|Please|Let's|Let’s|Don't|Don’t|Tell|Give|Take|"
                              r"Remember|Hey|Oh|Well|Yes|No|Okay|OK)\b")


def split_turn(text: str) -> list[str]:
    """A turn's paragraphs (split on blank lines), empties dropped."""
    return [p.strip() for p in re.split(r"\n[ \t]*\n", text) if p.strip()]


def classify_paragraph(paragraph: str, previous: str = "", cast: list[str] | None = None) -> tuple[str, list[str]]:
    """Label an unlabelled paragraph that follows ``previous`` inside a turn: "narr", "speech" or "unclear"."""
    p = paragraph.strip()
    cues_n: list[str] = []
    cues_s: list[str] = []
    if not p:
        return "unclear", ["empty"]
    if p[0] in "\"“‘'":
        cues_s.append("opens-quote")
    if p[0].islower() or (previous.rstrip()[-1:] in (",", ";", ":", "-", "—") if previous.strip() else False):
        cues_s.append("continues-sentence")
    if _LIST.match(p) and _LIST.search(previous or ""):
        cues_s.append("list-continues")
    bare = _QUOTED.sub(" ", p)
    if _FIRST_SECOND.search(bare):
        cues_s.append("first/second-person")
    if bare.rstrip().endswith("?") or _IMPERATIVE_OPEN.match(p):
        cues_s.append("address")
    names = [n for n in (cast or []) if n]
    if names and re.search(r"^\s*(" + "|".join(re.escape(n) for n in names) + r")\s*,", p):
        cues_s.append("vocative")
    if _STAGE_OPEN.match(p):
        cues_n.append("stage-marker")
    if _SCENE_OPEN.match(p):
        cues_n.append("scene-opener")
    third_subject = _THIRD.search(bare) or (names and re.search(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\b\s+\w+", bare))
    if third_subject and _ACTION_VERB.search(bare) and "first/second-person" not in cues_s:
        cues_n.append("third-person-narrative")
    strong_n = {"stage-marker", "scene-opener"} & set(cues_n)
    strong_s = {"opens-quote", "first/second-person", "vocative", "list-continues"} & set(cues_s)
    if cues_n and not cues_s:
        return "narr", cues_n
    if cues_s and not cues_n:
        return "speech", cues_s
    if strong_n and not strong_s:
        return "narr", cues_n + cues_s
    if strong_s and not strong_n:
        return "speech", cues_s + cues_n
    return "unclear", cues_n + cues_s


# Out-of-frame text inside a scene: the turn stops being the scene (a character's speech or the scene's narration)
# and becomes text ABOUT a work or a page: a review, a plot summary, an essay, web or transcript furniture. Only
# sentences without first or second person outside quotes are read (a character saying what they think of a book is
# speech). Strong cues decide alone; weak cues (nouns a scene can also use in-world) need two distinct hits in a turn.
_FRAME_STRONG = re.compile(
    r"\b(?:the|this|that|his|her|their|its) (?:novel|novel's|author|author's|narrator|protagonist|plot|reader|"
    r"readers|screenplay|sequel)\b|\bthe book's\b|\bchapter \d+\b|"
    r"\bspoilers?\b|\bout of (?:five|ten|5|10) stars\b|\b\d(?:\.\d)?/(?:5|10)\b|\bin this (?:essay|article|post|"
    r"review|chapter)\b|\bin conclusion\b|\bthis article\b|\bposted (?:by|on)\b|\ball rights reserved\b|"
    r"\bcopyright\b|https?://|\bwww\.|\bclick here\b|\bsubscribe\b|\b(?:the|this) (?:transcript|conversation above)\b|"
    r"\bend of (?:the )?(?:transcript|chapter|episode)\b", re.I)
_FRAME_WEAK = re.compile(r"\b(?:the|this) (story|scene|characters?|play|show|dialogue|writing|genre|audience|series|"
                         r"film|movie|episode|book)\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def out_of_frame(text: str) -> bool:
    """Whether a turn leaves the scene for text about a work or a page (review, summary, essay, page furniture)."""
    weak: set[str] = set()
    for sent in _SENTENCE_END.split(_QUOTED.sub(" ", text)):
        if not sent.strip() or _FIRST_SECOND.search(sent):
            continue
        if _FRAME_STRONG.search(sent):
            return True
        weak |= {m.group(1).lower().rstrip("s") for m in _FRAME_WEAK.finditer(sent)}
    return len(weak) >= 2
