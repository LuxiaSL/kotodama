"""Pinned kernel configs: inductor's runtime autotune choices recorded once and replayed, so every compiled instance of
a graph computes the same arithmetic.

An inductor-generated Triton kernel can carry several launch configs (block sizes). On its first launch inductor times
them and keeps the fastest; timings jitter, so two instances of the same compiled graph (two decoders, two processes)
can keep different configs. For a reduction the block size sets the summation order, so the two instances then differ
in the low bits, and a signature lane (kotodama.model.signing) captured on one is not byte-reproducible on the other.

``install(table_path, mode)`` patches inductor's runtime autotuner for the process:
  * ``record``: autotune as usual and record each kernel's choice; a later instance of a recorded kernel (same
    source and size hints) takes the recorded config instead of timing again, so the process runs exactly the
    arithmetic the table replays; ``PinSession.save()`` writes the table;
  * ``replay``: take each kernel's recorded config (exact match), no timing; a kernel missing from the table is
    refused (``KernelPinMiss``), or with ``allow_missing`` takes its first config and is recorded as a miss, which
    changes the table digest;
  * ``first``: every kernel takes its first config (deterministic, often slower).
A kernel is keyed by the sha256 of its Triton source and its size hints; a config by its block sizes, warps and
stages.

Hand-written ``@triton.autotune`` kernels (the AttnRes kernels the eager forward runs) choose the same way: once per
process per tuning key, by timing, so two processes can keep different warps and differ in a reduction's order. The
same session pins them in the same table: each (kernel, tuning key) is keyed ``triton:<name>:<source sha256>:<tuning
key>``, a config by every launch option it sets. In replay the pinned config is put in the autotuner's cache before
its first launch, so nothing is timed, and a launch that ends on another config is refused.

Under max-autotune, inductor also picks each GEMM's implementation (an extern kernel such as cuBLAS, or one of its
Triton templates) by timing the candidates when it lowers a graph, once per graph: two graphs with the same GEMM (an
unsigned decoder's layer and a signing decoder's) can pick different implementations, which round differently. The
same session pins those choices: a GEMM is keyed ``gemm:<op>:<sha256 of its input layouts, the fp32 matmul precision
and every candidate's identity>``, a choice by the chosen candidate's ``hash_key``; the first graph's choice stands
for every later graph in the process (record), or the table's (replay).

While pins are installed, inductor's autotune caches (local, remote, bundled) are off: a kernel that loads a cached
choice launches with one config and never reaches the pins, so a cache written by an unpinned process would run
arithmetic the table does not name. (Changing these settings changes inductor's graph-cache keys: the first pinned
start compiles once.)

The table digest (``PinSession.digest``) names the pins in a lane's identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

logger = logging.getLogger(__name__)

Mode = Literal["record", "replay", "first"]


class KernelPinMiss(RuntimeError):
    """A kernel the pin table does not hold (replay without allow_missing)."""


def config_key(config: Any) -> str:
    """A launch config's identity: its kwargs (block sizes), warps and stages."""
    kwargs = getattr(config, "kwargs", {}) or {}
    items = ",".join(f"{k}={kwargs[k]}" for k in sorted(kwargs))
    return f"{items}|w{getattr(config, 'num_warps', '?')}|s{getattr(config, 'num_stages', '?')}"


def triton_config_key(config: Any) -> str:
    """A ``triton.Config``'s identity: every meta-parameter and launch option it sets (warps, stages, CTAs, ...)."""
    opts = config.all_kwargs() if hasattr(config, "all_kwargs") else dict(getattr(config, "kwargs", {}) or {})
    return ",".join(f"{k}={opts[k]}" for k in sorted(opts))


def triton_tuning_key(autotuner: Any, args: Sequence[Any], kwargs: dict) -> Optional[tuple]:
    """The key ``triton.runtime.autotuner.Autotuner.run`` caches its choice under: the ``key`` arguments' values,
    then the dtype of every argument that has one (None: a single config, nothing is tuned)."""
    if len(autotuner.configs) <= 1:
        return None
    named = {**dict(zip(autotuner.arg_names, args)), **kwargs}
    named = {k: v for k, v in named.items() if k in autotuner.arg_names}
    key = [named[k] for k in autotuner.keys if k in named]
    key += [str(v.dtype) for v in named.values() if hasattr(v, "dtype")]
    return tuple(key)


def triton_kernel_key(autotuner: Any, tuning_key: tuple) -> str:
    """An ``@triton.autotune`` kernel's pin key: its name, the sha256 of its source and the tuning key."""
    fn = autotuner.base_fn
    src = getattr(fn, "src", None) or getattr(fn, "__name__", "?")
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    return f"triton:{getattr(fn, '__name__', '?')}:{sha}:{tuning_key!r}"


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
        return self._choose(key, launchers, lambda launcher: config_key(launcher.config))

    def choose_config(self, key: str, configs: Sequence[Any]) -> Optional[Any]:
        """The ``triton.Config`` to keep for an ``@triton.autotune`` kernel's tuning ``key``, or None to autotune
        (record mode). Raises KernelPinMiss."""
        return self._choose(key, configs, triton_config_key)

    def _choose(self, key: str, options: Sequence[Any], name: Callable[[Any], str]) -> Optional[Any]:
        if self.mode == "first":
            return options[0]
        want = self.table.get(key)
        if self.mode == "record" and want is None:
            return None
        if want is not None:
            for option in options:
                if name(option) == want:
                    self.hits += 1
                    return option
            raise KernelPinMiss(f"kernel {key[:40]}: pinned config {want} is not among its configs")
        if not self.allow_missing:
            raise KernelPinMiss(f"kernel {key[:40]} is not in the pin table {self.path}")
        self.misses.append(key)
        self.table[key] = name(options[0])
        return options[0]

    def record(self, key: str, launcher: Any) -> None:
        self._record(key, config_key(launcher.config))

    def record_config(self, key: str, config: Any) -> None:
        self._record(key, triton_config_key(config))

    def _record(self, key: str, config: str) -> None:
        if key in self.table:  # the first instance's choice stands (later instances take it, see _choose)
            return
        self.table[key] = config
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
    """Patch inductor's runtime autotuner and triton's ``Autotuner`` for this process (see the module docstring).
    ``table_path`` is read in replay mode and is where ``save`` writes in record mode. Returns the session."""
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
    _install_triton(sess)
    _install_gemm(sess)
    _cache_flags(off=True)
    _session = sess
    logger.info("kernel pins: %s (%s, %d pinned kernels)", mode, path, len(table))
    return sess


def uninstall() -> None:
    """Undo ``install``: inductor's and triton's autotuners choose by timing again."""
    global _session
    from torch._inductor.runtime import triton_heuristics as th

    cls = th.CachingAutotuner
    if hasattr(cls, "_kotodama_unpinned_autotune"):
        cls.autotune_to_one_config = cls._kotodama_unpinned_autotune
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        Autotuner = None
    if Autotuner is not None and hasattr(Autotuner, "_kotodama_unpinned_run"):
        Autotuner.run = Autotuner._kotodama_unpinned_run
    try:
        from torch._inductor.select_algorithm import AlgorithmSelectorCache
    except ImportError:
        AlgorithmSelectorCache = None
    if AlgorithmSelectorCache is not None and hasattr(AlgorithmSelectorCache, "_kotodama_unpinned_do_autotuning"):
        AlgorithmSelectorCache.do_autotuning = AlgorithmSelectorCache._kotodama_unpinned_do_autotuning
    _cache_flags(off=False)
    _session = None


_CACHE_FLAGS = ("autotune_local_cache", "autotune_remote_cache", "bundled_autotune_remote_cache")
_saved_flags: Optional[dict[str, Any]] = None


def _cache_flags(off: bool) -> None:
    """Turn inductor's autotune caches off (saving their settings), or restore them."""
    global _saved_flags
    import torch._inductor.config as ic

    if off:
        if _saved_flags is None:
            _saved_flags = {name: getattr(ic, name) for name in _CACHE_FLAGS}
        for name in _CACHE_FLAGS:
            setattr(ic, name, False)
    elif _saved_flags is not None:
        for name, value in _saved_flags.items():
            setattr(ic, name, value)
        _saved_flags = None


def _install_triton(sess: PinSession) -> None:
    """Patch ``triton.runtime.autotuner.Autotuner.run`` for ``sess`` (no-op without triton)."""
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return
    original = getattr(Autotuner, "_kotodama_unpinned_run", Autotuner.run)
    Autotuner._kotodama_unpinned_run = original

    def run(self, *args, **kwargs):
        tuning = triton_tuning_key(self, args, kwargs)
        if tuning is None:
            return original(self, *args, **kwargs)
        key = triton_kernel_key(self, tuning)
        fresh = tuning not in self.cache
        if fresh:
            pinned = sess.choose_config(key, self.configs)
            if pinned is not None:
                self.cache[tuning] = pinned
        out = original(self, *args, **kwargs)
        chosen = triton_config_key(self.best_config)
        if fresh and sess.mode == "record":
            sess.record_config(key, self.best_config)
        elif sess.mode == "replay" and sess.table.get(key) not in (None, chosen):
            raise KernelPinMiss(f"kernel {key[:40]} ran {chosen}, its pin is {sess.table[key]}")
        return out

    Autotuner.run = run


def gemm_key(name: str, inputs_key: str, choices: Sequence[Any]) -> str:
    """A GEMM selection's identity: its op, its inputs' layouts, the fp32 matmul precision and its candidates."""
    import torch

    parts = [inputs_key, torch.get_float32_matmul_precision(), *sorted(str(c.hash_key()) for c in choices)]
    return f"gemm:{name}:" + hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def pinned_timings(sess: PinSession, key: str, choices: Sequence[Any]) -> Optional[dict]:
    """The timings that make inductor keep ``key``'s pinned choice (0 for it, inf for the rest), or None to time the
    candidates (record mode, unrecorded). Raises KernelPinMiss."""
    chosen = sess._choose(key, list(choices), lambda c: str(c.hash_key()))
    if chosen is None:
        return None
    return {c: (0.0 if c is chosen else float("inf")) for c in choices}


def _install_gemm(sess: PinSession) -> None:
    """Patch inductor's GEMM selection (``AlgorithmSelectorCache.do_autotuning``) for ``sess`` (no-op when this
    torch has none)."""
    try:
        from torch._inductor.select_algorithm import AlgorithmSelectorCache
    except ImportError:
        return
    cls = AlgorithmSelectorCache
    original = getattr(cls, "_kotodama_unpinned_do_autotuning", cls.do_autotuning)
    cls._kotodama_unpinned_do_autotuning = original

    def do_autotuning(self, name, input_nodes, layout, input_gen_fns, inputs_key, choices, *args, **kwargs):
        key = gemm_key(name, inputs_key, choices)
        timings = pinned_timings(sess, key, choices)
        if timings is not None:
            return timings
        timings = original(self, name, input_nodes, layout, input_gen_fns, inputs_key, choices, *args, **kwargs)
        if timings:
            sess._record(key, str(min(timings, key=timings.__getitem__).hash_key()))
        return timings

    cls.do_autotuning = do_autotuning
