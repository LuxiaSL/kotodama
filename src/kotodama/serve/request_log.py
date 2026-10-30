"""The request log: one JSON line per request a server answers, for research use (``--request_log DIR``; off unless
the flag is given).

Each row holds what the client sent and what it got back, plus what only the server knows: the weights digest, and
per candidate the exact prompt and generated token ids, the stop token and the sign id. A request that sends
``"log": false`` is answered as usual and not written (counted in /info). Client addresses are never recorded. Layout:

    <dir>/<YYYY-MM-DD>/<writer>.jsonl        today's rows, appended by one server process (UTC days)
    <dir>/<YYYY-MM-DD>/<writer>.jsonl.zst    a finished day, compressed (.gz where zstandard is not installed)

A day's file is compressed in the background when the first row of the next day is written, and at shutdown only
if its day is over. Rows are written by one thread, so a slow disk never holds up a request; a row that cannot be
written is counted, never raised.
"""

from __future__ import annotations

import datetime as _dt
import gzip
import json
import logging
import os
import queue
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# endpoints whose requests are logged (the generation and scoring surfaces); /health, /info and the like are not
LOGGED_PATHS = frozenset({"/generate", "/v1/completions", "/v1/chat/completions", "/logprobs", "/score_tokens"})


def _day(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d")


def compress(path: Path) -> Path:
    """``path`` compressed beside itself (zstd, else gzip) and removed once the compressed copy is complete."""
    try:
        import zstandard

        out = path.with_name(path.name + ".zst")
        tmp = out.with_name(out.name + ".part")
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            zstandard.ZstdCompressor(level=10, threads=-1).copy_stream(src, dst)
    except ImportError:
        out = path.with_name(path.name + ".gz")
        tmp = out.with_name(out.name + ".part")
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
    os.replace(tmp, out)
    path.unlink()
    return out


class RequestLog:
    """Appends request rows under ``root`` from a background thread (``write`` never blocks on the disk)."""

    def __init__(self, root: str | Path, writer: Optional[str] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = writer or f"{socket.gethostname()}-{os.getpid()}"
        self.stats: dict[str, int] = {"rows": 0, "opted_out": 0, "errors": 0, "compressed": 0}
        self._q: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._day: Optional[str] = None
        self._file: Any = None
        self._thread = threading.Thread(target=self._run, name="request-log", daemon=True)
        self._thread.start()

    def path(self, day: str) -> Path:
        return self.root / day / f"{self.writer}.jsonl"

    def write(self, row: dict) -> None:
        """Queue one row (its ``ts`` decides its day)."""
        self._q.put(row)

    def opted_out(self) -> None:
        self.stats["opted_out"] += 1

    def close(self, timeout: float = 30.0) -> None:
        """Write what is queued, then stop; today's file stays uncompressed (the day is not over)."""
        self._q.put(None)
        self._thread.join(timeout)

    def info(self) -> dict:
        return {"dir": str(self.root), "writer": self.writer, "stats": dict(self.stats), "pending": self._q.qsize()}

    def _run(self) -> None:
        while True:
            row = self._q.get()
            if row is None:
                break
            try:
                day = _day(float(row.get("ts", time.time())))
                if day != self._day:
                    self._roll(day)
                self._file.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
                self._file.flush()
                self.stats["rows"] += 1
            except Exception:
                self.stats["errors"] += 1
                logger.exception("request log: a row could not be written")
        if self._file is not None:
            self._file.close()
            if self._day is not None and self._day < _day(time.time()):
                self._compress(self.path(self._day))

    def _roll(self, day: str) -> None:
        if self._file is not None:
            self._file.close()
            done = self.path(self._day)
            threading.Thread(target=self._compress, args=(done,), daemon=True).start()
        self.path(day).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path(day), "a", encoding="utf-8")
        self._day = day

    def _compress(self, path: Path) -> None:
        try:
            if path.exists() and path.stat().st_size:
                compress(path)
                self.stats["compressed"] += 1
        except Exception:
            self.stats["errors"] += 1
            logger.exception("request log: compressing %s failed", path)
