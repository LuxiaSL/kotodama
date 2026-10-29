"""Pinned kernel configs: inductor's runtime autotune choices recorded once and replayed, so every compiled instance of
a graph computes the same arithmetic.

An inductor-generated Triton kernel can carry several launch configs (block sizes). On its first launch inductor times
them and keeps the fastest; timings jitter, so two instances of the same compiled graph (two decoders, two processes)
can keep different configs. For a reduction the block size sets the summation order, so the two instances then differ
in the low bits, and a signature lane (kotodama.model.signing) captured on one is not byte-reproducible on the other.

``install(table_path, mode)`` patches inductor's runtime autotuner for the process:
  * ``record``: autotune as usual and record each kernel's choice; ``PinSession.save()`` writes the table;
  * ``replay``: take each kernel's recorded config (exact match), no timing; a kernel missing from the table is
    refused (``KernelPinMiss``), or with ``allow_missing`` takes its first config and is recorded as a miss, which
    changes the table digest;
  * ``first``: every kernel takes its first config (deterministic, often slower).
A kernel is keyed by the sha256 of its Triton source and its size hints; a config by its block sizes, warps and
stages. The table digest (``PinSession.digest``) names the pins in a lane's identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

Mode = Literal["record", "replay", "first"]


class KernelPinMiss(RuntimeError):
    """A kernel the pin table does not hold (replay without allow_missing)."""


def config_key(config: Any) -> str:
    """A launch config's identity: its kwargs (block sizes), warps and stages."""
    kwargs = getattr(config, "kwargs", {}) or {}
    items = ",".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    return f"{items}|w{getattr(config, 'num_warps', '?')}|s{getattr(config, 'num_stages', '?')}"


def kernel_key(source: str, size_hints: Any) -> str:
    """A kernel's identity: the sha256 of its Triton source and its size hints."""
    return hashlib.sha256(f"{source}\0{size_hints!r}".encode("utf-8")).hexdigest()


@dataclass
class PinSession:
    """The process's pin table and what happened to it. ``table``: kernel key -> config key."""

    mode: Mode
    path: Optional[Path]
    table: dict[str, str] = field(default_factory=dict)
    allow_missing: bool = False
    hits: int = 0
    recorded: int = 0
    misses: list[str] = field(default_factory=list)

    def choose(self, key: str, launchers: Sequence[Any]) -> Optional[Any]:
        """The launcher to keep for kernel ``key``, or None to autotune (record mode). Raises KernelPinMiss."""
        if self.mode == "first":
            return launchers[0]
        if self.mode == "record":
            return None
        want = self.table.get(key)
        if want is not None:
            for launcher in launchers:
                if config_key(launcher.config) == want:
                    self.hits += 1
                    return launcher
            raise KernelPinMiss(f"kernel {key[:12]}: pinned config {want} is not among its configs")
        if not self.allow_missing:
            raise KernelPinMiss(f"kernel {key[:12]} is not in the pin table {self.path}")
        self.misses.append(key)
        self.table[key] = config_key(launchers[0].config)
        return launchers[0]

    def record(self, key: str, launcher: Any) -> None:
        self.table[key] = config_key(launcher.config)
        self.recorded += 1

    @property
    def digest(self) -> str:
        """sha256 of the table's canonical JSON (sorted keys): the pins a lane names."""
        return hashlib.sha256(json.dumps(self.table, sort_keys=True).encode("utf-8")).hexdigest()

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path or self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.table, sort_keys=True, indent=0))
        return out


_session: Optional[PinSession] = None


def session() -> Optional[PinSession]:
    """The installed pin session, if any."""
    return _session


def install(table_path: Optional[str | Path], mode: Mode, allow_missing: bool = False) -> PinSession:
    """Patch inductor's runtime autotuner for this process (see the module docstring). ``table_path`` is read in
    replay mode and is where ``save`` writes in record mode. Returns the session."""
    global _session
    from torch._inductor.runtime import triton_heuristics as th

    path = Path(table_path) if table_path is not None else None
    table: dict[str, str] = {}
    if mode == "replay":
        if path is None or not path.exists():
            raise FileNotFoundError(f"replay needs a pin table, got {path}")
        table = json.loads(path.read_text())
    sess = PinSession(mode=mode, path=path, table=table, allow_missing=allow_missing)
    cls = th.CachingAutotuner
    original = getattr(cls, "_kotodama_unpinned_autotune", cls.autotune_to_one_config)
    cls._kotodama_unpinned_autotune = original

    def autotune_to_one_config(self, *args, **kwargs):
        key = kernel_key(getattr(self.fn, "src", getattr(self.fn, "__name__", "?")), getattr(self, "size_hints", None))
        chosen = sess.choose(key, self.launchers)
        if chosen is None:
            original(self, *args, **kwargs)
            sess.record(key, self.launchers[0])
            return
        self.launchers = [chosen]

    cls.autotune_to_one_config = autotune_to_one_config
    _session = sess
    logger.info("kernel pins: %s (%s, %d pinned kernels)", mode, path, len(table))
    return sess
