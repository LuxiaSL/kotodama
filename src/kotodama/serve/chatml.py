"""ChatML: the conversation format the post-trained models were trained on.

``CHATML_TEMPLATE`` is FROZEN (it is what SFT rendered); ``render_chatml`` is its
dependency-free string twin, byte-identical with a generation prompt appended
(guarded by tests/test_serve_kit.py against the tokenizer render).
"""

from __future__ import annotations

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

# Token ids in the SmolLM2 vocab: 0 = <|endoftext|>, 2 = <|im_end|>.
EOS_TOKEN_ID = 0
IM_END_TOKEN_ID = 2
BASE_STOP_TOKEN_IDS = frozenset({EOS_TOKEN_ID})
CHAT_STOP_TOKEN_IDS = frozenset({EOS_TOKEN_ID, IM_END_TOKEN_ID})

# The system prompt the self-distilled conversation models were trained under.
SELF_DISTILL_SYSTEM = "transcript; a language model, speaking for itself."


def render_chatml(messages: list[dict[str, str]]) -> str:
    """``CHATML_TEMPLATE`` with a generation prompt appended, as a plain string."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"


def render_chatml_prefix(messages: list[dict[str, str]], upto: int,
                         system: str = SELF_DISTILL_SYSTEM) -> str:
    """System + ``messages[:upto]``, ending in the open assistant tag."""
    return render_chatml([{"role": "system", "content": system}, *messages[:upto]])
