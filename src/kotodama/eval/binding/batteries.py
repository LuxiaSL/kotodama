"""The battery registry: what each battery scores, where its frozen prompts live, and which reader reads it.

Prompt files live under one prompts directory (``--prompts-dir``, else ``$KOTODAMA_BINDING_PROMPTS``; machine paths
belong in site.env). Every file is pinned by sha256, and a run refuses a file that does not match its pin, so a
number in the record always names the exact strings it was scored on. ``capacity-v1`` can be regenerated
(``python -m kotodama.eval.binding gen-capacity``), and the pin then checks that the regeneration is byte-identical;
so can the ``fluid`` group (``gen-fluid``, :mod:`kotodama.eval.binding.fluid`) and ``formupdate-v1``
(``gen-formupdate``, :mod:`kotodama.eval.binding.formupdate`).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

PROMPTS_ENV = "KOTODAMA_BINDING_PROMPTS"


@dataclass(frozen=True)
class Battery:
    name: str
    kind: Literal["rows", "bindprobes"]
    file: str                                   # relative to the prompts dir
    sha256: str
    reader: Optional[str]                       # readers.READERS key (None: read with bindprobes aggregate)
    what: str
    families: tuple[str, ...] = field(default_factory=tuple)     # bindprobes only
    conditions: tuple[str, ...] = field(default_factory=tuple)   # bindprobes only


BATTERIES: dict[str, Battery] = {b.name: b for b in (
    Battery("joinlite-v1", "rows", "joinlite-v1.jsonl",
            "1e20c12961803a8a459b302d7ece30c34c7d4535f871330dba51020699facb90", "joinlite",
            "stage gauges, static: lookup, I / you, P2 speaker swap"),
    Battery("swapdyn-v1", "rows", "swapdyn-v1.jsonl",
            "8ddab1262a2fa340e559fbbcd966b2c20a2acdfade2bb888e55034851810bdd3", "swapdyn",
            "stage gauge, dynamic: speaker swap on transfer dialogues (0shot / 3shot)"),
    Battery("stage5-decomp-v1", "rows", "stage5-decomp-v1.jsonl",
            "c08b40150b84e3e2e6d5e9918a8b7c6b5986230545f730b06b1cd2d8bc3db666", "decomp",
            "stage-5 decomposition, all 6 variants × 0shot / 3shot"),
    Battery("stage5-decomp-core", "rows", "stage5-decomp-core.jsonl",
            "f653040de69157ef6edb75e4702e00473ce229bc0e25dfe8cdc457a09ac3727e", "decomp",
            "stage-5 decomposition, core variants (deictic / third / named_transfer)"),
    Battery("capacity-v1", "rows", "capacity-v1.jsonl",
            "c7241ec026e40054c21bef0ec44465848e3758da40ff7b183a7ada4a89e3c88d", "capacity",
            "binding capacity (K50, after arXiv 2609.30634)"),
    Battery("fluid-v1-morph", "rows", "fluid-v1-morph.jsonl",
            "ec73660da59ba0893f5a65c6a7675105b1dbcc7c17166da255c9f2c6ccfaa341", "fluid-morph",
            "fluid F1: nonce morphology (rule tier × examples)"),
    Battery("fluid-v1-algebra", "rows", "fluid-v1-algebra.jsonl",
            "81442d46aa9d0356d54f0676cef7e912d5f1ace36058f2fe74f41bfe8560ecf4", "fluid-algebra",
            "fluid F2: symbol algebra (operators × composition depth)"),
    Battery("fluid-v1-rebind", "rows", "fluid-v1-rebind.jsonl",
            "7431ebb1dc74dca1ab1c656456a7371bdf0d0300317dbeff0a84149626f3a543", "fluid-rebind",
            "fluid F3: rebinding chains, prose + code (reassignments × distractors; stale-value rate)"),
    Battery("fluid-v1-seqind", "rows", "fluid-v1-seqind.jsonl",
            "b2a1e6b54040500363909cf4d669f84ef968a60df6353127599e9f041a46e15a", "fluid-seqind",
            "fluid F4: nonce sequence induction (pattern length × gap)"),
    Battery("cryst-v1", "rows", "cryst-v1.jsonl",
            "e73e55699384eab1665b0dc7501a5f96f3ab793aeeca475bdc8089946783e5a2", "cryst",
            "crystallized control C0: facts in the fluid few-shot frame (shots 6 / 3 / 1 / 0)"),
    Battery("formupdate-v1", "rows", "formupdate-v1.jsonl",
            "7222046c4c63a4eca769212c6cca327eb64401d4909db58e675dc16a31a17d1f", "formupdate",
            "forming vs updating: static / update-other / update-target, paired per base item (K × U × register)"),
    Battery("bp-deixis-lite", "bindprobes", "bindprobes-deixis-lite/items.jsonl",
            "229e7ac377ff2ec5ef298ecdee5c27667f9c97d86ba2337fb5e411aac9167c3e", None,
            "bindprobes deixis-lite (0shot / 3shot)", ("deixis",), ("0shot", "3shot")),
    Battery("bp-profile", "bindprobes", "bindprobes-items/items.jsonl",
            "7bfab8bff499667846fd9c9522cfbb4147476626bdd37bb7d702a2f55f4f9776", None,
            "bindprobes profile (attr / codevar / dialogue …, all conditions)"),
    Battery("bp-v3.1", "bindprobes", "bindprobes-v3.1/items.jsonl",
            "bc3496d3c68990b8a615f485ca38e43b917deca1315cc6207990454be1330f37", None,
            "bindprobes v3.1: setbind / answer / adjacency / codevarmut",
            ("setbind", "answer", "adjacency", "codevarmut")),
    Battery("bp-deixis-v2", "bindprobes", "bindprobes-deixis-v2-full/items.jsonl",
            "08151e66c4798288c7b1f471aaeb46d07e3d21acd5d279e8d8c137ba3bd35234", None,
            "bindprobes deixis v2 full", ("deixis",)),
)}

# Named sets for the CLI: `--batteries gauges,capacity` etc.
GROUPS: dict[str, tuple[str, ...]] = {
    "gauges": ("joinlite-v1", "swapdyn-v1"),
    "decomp": ("stage5-decomp-v1",),
    "bindprobes": ("bp-deixis-lite", "bp-profile", "bp-v3.1", "bp-deixis-v2"),
    "scorecard": ("joinlite-v1", "swapdyn-v1", "stage5-decomp-v1", "capacity-v1"),
    "fluid": ("fluid-v1-morph", "fluid-v1-algebra", "fluid-v1-rebind", "fluid-v1-seqind", "cryst-v1"),
    "formupdate": ("formupdate-v1",),
    "all": tuple(BATTERIES),
}


class PinError(RuntimeError):
    """A prompts file whose bytes do not match its registry pin."""


def resolve(names: str) -> list[Battery]:
    """``a,b,group`` → batteries, in order, deduplicated."""
    out: list[Battery] = []
    for n in filter(None, (x.strip() for x in names.split(","))):
        for m in GROUPS.get(n, (n,)):
            if m not in BATTERIES:
                raise KeyError(f"unknown battery {m!r} (have: {', '.join(BATTERIES)}; groups: {', '.join(GROUPS)})")
            if BATTERIES[m] not in out:
                out.append(BATTERIES[m])
    return out


def prompts_dir(arg: Optional[str]) -> Path:
    d = arg or os.environ.get(PROMPTS_ENV)
    if not d:
        raise SystemExit(f"no prompts dir: pass --prompts-dir or set ${PROMPTS_ENV} (site.env)")
    return Path(d)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def path_for(b: Battery, root: Path, check: bool = True) -> Path:
    p = Path(root) / b.file
    if not p.exists():
        raise FileNotFoundError(f"{b.name}: prompts file missing: {p}")
    if check and (got := sha256(p)) != b.sha256:
        raise PinError(f"{b.name}: {p} sha256 {got[:16]}… ≠ pin {b.sha256[:16]}… (the strings changed)")
    return p
