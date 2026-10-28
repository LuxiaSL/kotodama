"""A content-addressed store of compiled engine artifacts, so a server compiles its decode graphs once per code
version instead of once per start.

The fast engine's startup is dominated by torch.compile: every decode-step variant and every row-bucket width is
its own graph. ``torch.compiler.save_cache_artifacts`` packages what a process compiled (inductor FX graphs,
autotuning choices, Triton kernels) into one blob, and ``load_cache_artifacts`` puts a blob back into the local
compile caches, so the next compile of the same graphs is a cache hit.

The blob is stored under a key that hashes everything the compiled code depends on: the sources of
``kotodama.model`` (the forward the graphs trace), the torch / Triton / CUDA versions, the GPU model, the
engine-shaping server flags and the environment variables that change kernels. Checkpoint weights are graph inputs,
not constants, so one blob serves every checkpoint of a shape. A changed key is a cache miss: the server compiles
cold and stores a new blob; nothing is reused across a change that could alter the compiled code.

Layout: ``<store>/<key>.bin`` (the artifacts) and ``<store>/<key>.json`` (what went into the key, its size, when
it was made), written atomically.

Servers that start together share one compile: ``acquire`` loads a stored key, and on a miss the first server takes
``<store>/<key>.lock`` (kept fresh by a heartbeat while it compiles, released once it has saved) while the others wait
for the blob and load it. A lock whose heartbeat stopped (its server died) is stale and is taken over.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Server flags that change which graphs the engine compiles, or what they contain.
ENGINE_FLAGS: tuple[str, ...] = (
    "model_size", "rope_fp32", "attn_res", "attn_res_boundaries", "max_seq_len", "engine", "mode", "prefix_cache",
    "multi_rows", "xbatch_rows", "xbatch_refill", "prefix_pool_tokens", "prefix_pool_block", "cascade_attn",
    "cascade_groups", "cascade_tail_tokens", "levers", "seeded_rows", "steer_npz", "compile_mode", "warm_truncated",
    "regional",
)
# Environment variables that change the kernels a compile emits.
ENGINE_ENV: tuple[str, ...] = ("KOTODAMA_ATTNRES_EPS_FP64_COMPAT",)
SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "model"


def _source_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(SOURCE_PACKAGE.rglob("*.py")):
        h.update(str(p.relative_to(SOURCE_PACKAGE)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def key_fields(flags: dict[str, Any]) -> dict[str, Any]:
    """Everything the compiled code depends on, as a JSON-able dict (the manifest of a key)."""
    import torch

    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "model_source_sha256": _source_digest(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "flags": {k: flags.get(k) for k in ENGINE_FLAGS},
        "env": {k: os.environ.get(k) for k in ENGINE_ENV},
    }


def cache_key(fields: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:32]


def cache_counters() -> str:
    """How this process's compiles went so far: inductor FX-graph and AOT-autograd cache hits and misses (a start
    whose graphs all hit compiled nothing)."""
    try:
        from torch._dynamo.utils import counters
    except ImportError:
        return "cache counters unavailable"
    return ", ".join(f"{name} {counters[group].get(f'{key}_hit', 0)} hit / {counters[group].get(f'{key}_miss', 0)} miss"
                     for name, group, key in (("fx graphs", "inductor", "fxgraph_cache"),
                                              ("aot", "aot_autograd", "autograd_cache")))


def load(store: str | os.PathLike[str], key: str) -> bool:
    """Put the stored artifacts of ``key`` into this process's compile caches; False when the store has none (or
    they cannot be loaded, which is logged and treated as a miss)."""
    path = Path(store) / f"{key}.bin"
    if not path.is_file():
        logger.info("compile store: no artifacts for key %s (cold compile)", key)
        return False
    import torch

    t0 = time.perf_counter()
    try:
        info = torch.compiler.load_cache_artifacts(path.read_bytes())
    except Exception as exc:  # a corrupt or incompatible blob: compile cold rather than fail to start
        logger.warning("compile store: artifacts for key %s did not load (%r); compiling cold", key, exc)
        return False
    logger.info("compile store: loaded key %s (%.0f MB) in %.1fs%s", key, path.stat().st_size / 1e6,
                time.perf_counter() - t0, f": {info}" if info is not None else "")
    return True


class StoreClaim:
    """This process's claim on compiling a key: the lock file exists and its mtime is refreshed every ``beat_s``
    seconds until ``release``."""

    def __init__(self, path: Path, beat_s: float) -> None:
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, args=(beat_s,), name="compile-store-claim", daemon=True)
        self._thread.start()

    def _beat(self, beat_s: float) -> None:
        while not self._stop.wait(beat_s):
            try:
                os.utime(self.path)
            except OSError:
                return

    def release(self) -> None:
        self._stop.set()
        try:
            self.path.unlink()
        except OSError:
            pass


def acquire(store: str | os.PathLike[str], key: str, *, stale_s: float = 300.0, beat_s: float = 30.0,
            poll_s: float = 10.0, wait_s: float = 4 * 3600.0) -> tuple[bool, Optional[StoreClaim]]:
    """Load ``key`` from the store, or claim its compile. Returns (loaded, claim): (True, None) on a hit; (False,
    claim) when this process compiles the key (``release`` the claim after ``save``); (False, None) when it compiles
    without a claim (the store is unwritable, or another server's live claim outlasted ``wait_s``). While another
    server holds a fresh claim (its lock touched within ``stale_s``), this one waits for its blob."""
    root = Path(store)
    blob, lock = root / f"{key}.bin", root / f"{key}.lock"
    t0, last_log = time.monotonic(), 0.0
    while True:
        if blob.is_file():
            if load(store, key):
                return True, None
            # a blob that does not load: compile cold and replace it (claimed, so the others wait for the new one)
        try:
            root.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # released between the two calls: look again
            if age > stale_s:
                logger.warning("compile store: claim on key %s is stale (%.0fs since its heartbeat); taking it over",
                               key, age)
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
                continue
            waited = time.monotonic() - t0
            if waited > wait_s:
                logger.warning("compile store: waited %.0fs for key %s; compiling it here too", waited, key)
                return False, None
            if waited - last_log >= 60.0 or last_log == 0.0:
                logger.info("compile store: another server is compiling key %s (claim %s); waiting for it (%.0fs)",
                            key, lock.read_text(errors="replace").strip() if lock.exists() else "?", waited)
                last_log = waited or 1e-9
            time.sleep(poll_s)
            continue
        except OSError as exc:
            logger.warning("compile store: cannot claim key %s under %s (%r); compiling without a claim", key, root,
                           exc)
            return False, None
        with os.fdopen(fd, "w") as f:
            f.write(f"{socket.gethostname()} pid {os.getpid()} since {time.strftime('%H:%M:%S')}\n")
        logger.info("compile store: no artifacts for key %s; this server compiles it (claimed)", key)
        return False, StoreClaim(lock, beat_s)


def save(store: str | os.PathLike[str], key: str, fields: dict[str, Any]) -> Optional[Path]:
    """Store what this process compiled under ``key`` (atomically; a concurrent writer of the same key wins or
    loses whole). Returns the blob's path, or None when there was nothing to save or the write failed."""
    import torch

    try:
        out = torch.compiler.save_cache_artifacts()
    except Exception as exc:
        logger.warning("compile store: could not package the compiled artifacts (%r)", exc)
        return None
    if out is None:
        logger.info("compile store: nothing compiled to save for key %s", key)
        return None
    blob, info = out
    root = Path(store)
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{key}.bin"
        tmp = root / f".{key}.{os.getpid()}.tmp"
        tmp.write_bytes(blob)
        tmp.replace(path)
        manifest = {"key": key, "fields": fields, "bytes": len(blob), "created_unix": time.time(),
                    "info": str(info)}
        mtmp = root / f".{key}.{os.getpid()}.json.tmp"
        mtmp.write_text(json.dumps(manifest, indent=1, default=str))
        mtmp.replace(root / f"{key}.json")
    except OSError as exc:
        logger.warning("compile store: could not write key %s under %s (%r)", key, root, exc)
        return None
    logger.info("compile store: saved key %s (%.0f MB)", key, len(blob) / 1e6)
    return path
