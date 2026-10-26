"""Self-generated seeds for self-play: the model writes the cast and setting of each transcript.

The generator continues a short catalogue document — entries of the form ``Participants: Name, who they are; Name,
who they are`` then ``Setting: ...`` — so the people, places and situations come from the model's own distribution
rather than from a hand-written topic list. The catalogue's opening entries are the only fixed text; they are
context, never training data (a seed becomes a transcript header, and headers never carry loss), and ``--examples``
can replace them with entries an earlier round generated.

    python -m kotodama.posttrain.seeds --endpoint http://HOST:PORT --n 2000 --out seeds.jsonl [--temperature 1.0]

Each output row is a self-play seed: ``{"id", "cast": [names], "header": "Participants: ...\\nSetting: ..."}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FRAME_HEAD = ("Transcripts from a collection of recorded conversations. Each transcript opens with a note on who is "
              "talking and where.\n\n")
DEFAULT_EXAMPLES = (
    "Participants: Ines, a lighthouse keeper's daughter; Paul, the postman on the island route\n"
    "Setting: a cottage kitchen on a winter morning, the radio on low",
    "Participants: Dev, a night-shift nurse; Mrs. Okafor, who runs the pawnshop downstairs\n"
    "Setting: the stairwell of an apartment building, just after midnight",
    "Participants: Kai, a first-year chemistry student; Ren, the teaching assistant\n"
    "Setting: an empty lab after a failed experiment",
)
_NAME = re.compile(r"^[A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2}$")


def catalogue_prompt(examples: list[str], k: int) -> str:
    """The document so far: the frame, the example entries numbered 1..n, and an open entry n+1."""
    body = "".join(f"Transcript {i + 1}\n{e}\n\n" for i, e in enumerate(examples))
    return f"{FRAME_HEAD}{body}Transcript {len(examples) + 1}\nParticipants:"


def parse_entry(completion: str, max_cast: int = 3) -> Optional[dict[str, Any]]:
    """``" Name, desc; Name, desc\\nSetting: ..."`` -> {cast, header}; None if malformed."""
    lines = [ln.strip() for ln in completion.strip("\n").split("\n")]
    if not lines or not lines[0]:
        return None
    parts = [p.strip() for p in lines[0].split(";") if p.strip()]
    cast, people = [], []
    for p in parts:
        name, _, desc = p.partition(",")
        name = name.strip()
        if not _NAME.match(name) or len(name) > 30 or name in cast:
            return None
        cast.append(name)
        people.append(f"{name}, {desc.strip()}" if desc.strip() else name)
    if not 2 <= len(cast) <= max_cast:
        return None
    setting = next((ln for ln in lines[1:] if ln.startswith("Setting:")), "")
    if not setting or len(setting) < 12:
        return None
    return {"cast": cast, "header": f"Participants: {'; '.join(people)}\n{setting}"}


async def generate(complete: Any, n: int, examples: list[str], shuffle_seed: int,
                   max_cast: int = 3) -> list[dict[str, Any]]:
    """Up to ``n`` distinct seeds. Example order is reshuffled per request so the open entry's context varies."""
    rng = random.Random(shuffle_seed)
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, Any]] = []
    stops = ["\n\nTranscript", "\n\n\n"]
    tries = 0
    while len(out) < n and tries < 4 * n:
        ex = list(examples)
        rng.shuffle(ex)
        raw = await complete(catalogue_prompt(ex, len(out)), 8, stops)
        for text, _finish, _hit in raw:
            tries += 1
            cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=len(text))
            e = parse_entry(text[:cut], max_cast)
            if e is None:
                continue
            key = (*sorted(e["cast"]), e["header"].split("\n")[-1].lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"id": f"seed{len(out):06d}", **e})
            if len(out) >= n:
                break
    return out


def main(argv: list[str] | None = None) -> int:
    from kotodama.posttrain.selfplay import oai_completer
    from kotodama.serve.laws import law

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [seeds] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="self-generated self-play seeds")
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=1.0, help="seed diversity: hotter than the dialogue law")
    ap.add_argument("--examples", default=None, help="JSONL of earlier seeds whose headers replace the defaults")
    ap.add_argument("--max_cast", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    examples = list(DEFAULT_EXAMPLES)
    if a.examples:
        rows = [json.loads(x) for x in Path(a.examples).read_text().splitlines() if x.strip()]
        examples = [r["header"] for r in random.Random(a.seed).sample(rows, min(6, len(rows)))]
    sampling = {**law(a.law), "temperature": a.temperature}
    complete, close = oai_completer(a.endpoint, sampling, max_tokens=96)

    async def go() -> list[dict[str, Any]]:
        try:
            return await generate(complete, a.n, examples, a.seed, a.max_cast)
        finally:
            await close()

    seeds = asyncio.run(go())
    Path(a.out).write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds))
    logger.info("wrote %d seeds -> %s", len(seeds), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
