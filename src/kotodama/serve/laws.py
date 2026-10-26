"""Named sampling laws of record (temperature / top_k / top_p / repetition penalty).

top_k 0 and top_p 0 mean "off": kotodama models sample best with pure temperature
(top-p degrades them; temperature is the anti-loop dial — the 3B lineage never goes to 0.7 for chat).
Each law is measured per model family.
"""

from __future__ import annotations

LAWS: dict[str, dict[str, float]] = {
    # Conversation with the post-trained models.
    "chat": {"temperature": 0.9, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.2},
    # kotodama-tsumugi (1b-tsumugi): the sampling bench's best binding (.42 vs .34 at the 3B's 0.9/1.2) sits at
    # T 0.7-0.8 with top-p off; repetition 1.2 (1.35 if it loops).
    "tsumugi": {"temperature": 0.75, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.2},
    # The base model: raw base loops without strong anti-repetition.
    "base": {"temperature": 0.9, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.35},
    # Checkpoint soups: temp 1.0 clears the "i-don't-know" depth collapse.
    "soup": {"temperature": 1.0, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.2},
    # Earlier crowned-model law.
    "crowned": {"temperature": 0.85, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.1},
    # Generation-side sampling core with no penalty (banked capture recipes).
    "pure": {"temperature": 0.9, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.0},
    # A cleaner read of model quality; truncated, so not comparable with the above.
    "tight": {"temperature": 0.7, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.1},
}


def law(name: str) -> dict[str, float]:
    if name not in LAWS:
        raise KeyError(f"unknown sampling law {name!r}; choose from {sorted(LAWS)}")
    return dict(LAWS[name])
