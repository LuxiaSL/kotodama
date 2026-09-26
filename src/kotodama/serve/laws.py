"""Named sampling laws of record (temperature / top_k / top_p / repetition penalty).

top_k 0 and top_p 0 mean "off": kotodama models sample best with pure temperature
(top-p degrades them; temperature is the anti-loop dial — never go to 0.7 for chat).
Measured per model family in 2026-06/07; see the lab's SERVING-LAWS record.
"""

from __future__ import annotations

LAWS: dict[str, dict[str, float]] = {
    # Conversation with the post-trained models (gamut-confirmed 2026-07-09).
    "chat": {"temperature": 0.9, "top_k": 0, "top_p": 0.0, "repetition_penalty": 1.2},
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
