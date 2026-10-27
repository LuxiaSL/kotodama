"""Cross-request batched decoding: concurrent completion requests decoded together in one wide row batch.

A server holds one engine and decodes one request at a time, while a decode step over 64 rows costs about twice a
step over 8 (the deep looped trunk is kernel-count bound, not FLOP bound). This module gathers concurrent requests
that share a sampling law and a steering config (``XBatchJob.key``), prefills each on the engine (prefix cache
included), fills ``n`` rows of a ``MultiSampleDecoder`` from each prefill (``admit``: every row keeps its own
position, mask and presence set) and steps every row together, ending each row at its own stop token, stop string,
token budget or the context limit. A request's result is delivered the moment its last row ends.

Two parts:
  * ``XBatchRunner.run_batch`` runs on the server's generation thread, with the engine lock held. It takes the
    oldest waiting request's key group from the queue (FIFO, as many requests as fit the widest bucket), decodes it
    on the narrowest bucket that holds the group's rows, and returns when every row has ended. With ``refill`` on,
    rows freed by finished requests are refilled from the queue between steps (continuous batching), until a request
    with a different key is waiting or ``should_yield`` says another engine user is (the server passes "a request
    outside the batcher waits on the engine lock"); those then get the engine when the batch drains.
  * ``XBatchScheduler`` is the asyncio side: ``submit`` enqueues a job and awaits its result, and one pump task
    runs batches (each under the engine lock, on the generation thread) while the queue is non-empty.

The runner knows nothing about HTTP: prompts arrive tokenized and validated, steering arrives as the engine's
``set_steer`` arguments, and stop strings are checked through a ``decode`` callable (the tokenizer's).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Optional

import torch

from kotodama.model.decode_engine import SamplingParams
from kotodama.model.multi_decode import MultiSampleDecoder

logger = logging.getLogger(__name__)


@dataclass
class XBatchResult:
    """One request's outcome: ``rows[i]`` is candidate i's generated ids (stop tokens not kept)."""

    rows: list[list[int]]
    timing: dict[str, float]


@dataclass
class XBatchJob:
    """One request: ``n`` candidates of ``prompt_ids``.

    ``key`` groups requests that may share a batch (equal sampling law and steering config); ``params``,
    ``steer_writes`` and ``steer_kwargs`` must be equal across one key. ``done`` is called exactly once, on the
    generation thread, with the result or the exception that ended the request.
    """

    key: Hashable
    prompt_ids: list[int]
    n: int
    max_new_tokens: int
    params: SamplingParams
    done: Callable[[XBatchResult | BaseException], None]
    stop_strings: tuple[str, ...] = ()
    steer_writes: Optional[list[tuple[int, torch.Tensor, float]]] = None
    steer_kwargs: dict[str, Any] = field(default_factory=dict)
    t_enqueue: float = field(default_factory=time.perf_counter)


class XBatchQueue:
    """Thread-safe FIFO of waiting jobs (the event loop puts, the generation thread takes)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: list[XBatchJob] = []

    def put(self, job: XBatchJob) -> None:
        with self._lock:
            self._jobs.append(job)

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def take_group(self, max_rows: int) -> list[XBatchJob]:
        """The oldest job's key group, oldest first, while the rows fit ``max_rows``; later jobs of the key that
        do not fit stay queued, as do jobs of other keys."""
        with self._lock:
            if not self._jobs:
                return []
            key = self._jobs[0].key
            taken, rows = [], 0
            for job in self._jobs:
                if job.key == key and rows + job.n <= max_rows:
                    taken.append(job)
                    rows += job.n
            for job in taken:
                self._jobs.remove(job)
            return taken

    def drain(self) -> list[XBatchJob]:
        """Remove and return every waiting job."""
        with self._lock:
            jobs, self._jobs = self._jobs, []
            return jobs

    def take_fitting(self, key: Hashable, free_rows: int) -> Optional[XBatchJob]:
        """The oldest job of ``key`` with n <= ``free_rows``, or None. Returns None while any job of another key
        waits at the head, so a running batch stops growing once a different request is due."""
        with self._lock:
            if not self._jobs or self._jobs[0].key != key:
                return None
            for job in self._jobs:
                if job.key == key and job.n <= free_rows:
                    self._jobs.remove(job)
                    return job
            return None


@dataclass
class _Live:
    """A job decoding in the batch: its rows, each row's output and liveness, and its token budget."""

    job: XBatchJob
    rows: list[int]
    out: list[list[int]]
    live: list[bool]
    budget: int
    timing: dict[str, float]
    t_admit: float


class XBatchRunner:
    """Runs batches on the generation thread. ``decoders`` maps bucket width -> decoder (all over one engine);
    ``prefill(ids (1, S)) -> (last logits (1, V), prefix-cache info or None)`` prefills the engine;
    ``set_steer(writes, **kwargs)`` installs the engine's steering rows (``None`` clears); ``decode(ids) -> str``
    is the tokenizer's decode, for stop strings; ``should_yield() -> bool``, polled between steps, stops refilling
    so the engine passes to whoever waits for it once the batch drains."""

    def __init__(self, decoders: dict[int, MultiSampleDecoder], prefill: Callable[[torch.Tensor], tuple],
                 set_steer: Callable[..., None], decode: Callable[[list[int]], str], stop_ids: frozenset[int],
                 max_seq_len: int, device: torch.device, refill: bool = False,
                 should_yield: Optional[Callable[[], bool]] = None) -> None:
        if not decoders:
            raise ValueError("XBatchRunner needs at least one decoder bucket")
        self.decoders = dict(sorted(decoders.items()))
        self.max_rows = max(self.decoders)
        self.prefill = prefill
        self.set_steer = set_steer
        self.decode = decode
        self.stop_ids = stop_ids
        self.max_seq_len = max_seq_len
        self.device = device
        self.refill = refill
        self.should_yield = should_yield or (lambda: False)
        self.queue = XBatchQueue()
        self.stats: dict[str, float] = {"batches": 0, "steps": 0, "row_steps": 0, "useful_row_steps": 0,
                                        "admitted": 0, "refilled": 0}

    def bucket_for(self, rows: int) -> MultiSampleDecoder:
        """The narrowest decoder that holds ``rows`` rows (the widest when none does)."""
        return next((d for w, d in self.decoders.items() if w >= rows), self.decoders[self.max_rows])

    @staticmethod
    def _deliver(job: XBatchJob, outcome: XBatchResult | BaseException) -> None:
        try:
            job.done(outcome)
        except Exception:  # a broken callback must not end the batch for every other request
            logger.exception("xbatch: result callback failed")

    def _admit(self, dec: MultiSampleDecoder, job: XBatchJob, free: list[int]) -> Optional[_Live]:
        """Prefill ``job`` and fill ``job.n`` free rows from it; process each row's first token. Returns None (the
        job already failed) when the prefill raises."""
        t0 = time.perf_counter()
        timing: dict[str, float] = {"queue_ms": round((t0 - job.t_enqueue) * 1000, 2)}
        try:
            ids = torch.tensor([job.prompt_ids], dtype=torch.long, device=self.device)
            logits, info = self.prefill(ids)
            rows = free[:job.n]
            first = dec.admit(rows, logits, job.params)
            del free[:job.n]
        except Exception as exc:
            logger.warning("xbatch: prefill/admit failed for a %d-token prompt: %r", len(job.prompt_ids), exc)
            self._deliver(job, exc)
            return None
        timing["prefill_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if info is not None:
            timing["prefix_cache_hit"] = float(bool(info.get("prefix_hit")))
            timing["prefix_common_tokens"] = float(info.get("common_prefix", 0))
        timing["batched_rows"] = float(dec.rows)
        budget = min(job.max_new_tokens, self.max_seq_len - len(job.prompt_ids))
        lv = _Live(job, rows, [[] for _ in rows], [True] * len(rows), budget, timing, t0)
        self._consume(dec, lv, first)
        self.stats["admitted"] += 1
        return lv

    def _consume(self, dec: MultiSampleDecoder, lv: _Live, toks: list[int]) -> None:
        """Process one sampled token per row of ``lv`` (``toks[i]`` for ``lv.rows[i]``), ending rows at a stop
        token (not kept), the token budget, a stop string or the context limit."""
        stops = lv.job.stop_strings
        for i, tok in enumerate(toks):
            if not lv.live[i]:
                continue
            if tok in self.stop_ids:
                lv.live[i] = False
                continue
            out = lv.out[i]
            out.append(tok)
            self.stats["useful_row_steps"] += 1
            if (len(out) >= lv.budget or dec.row_pos(lv.rows[i]) >= self.max_seq_len
                    or (stops and any(s in self.decode(out) for s in stops))):
                lv.live[i] = False

    def _finish(self, lv: _Live) -> None:
        lv.timing["decode_ms"] = round((time.perf_counter() - lv.t_admit) * 1000 - lv.timing["prefill_ms"], 2)
        lv.timing["total_ms"] = round((time.perf_counter() - lv.job.t_enqueue) * 1000, 2)
        self._deliver(lv.job, XBatchResult(rows=lv.out, timing=lv.timing))

    def run_batch(self) -> int:
        """Decode one batch from the queue to completion; returns the number of requests it served (0 when the
        queue was empty). Call on the generation thread with the engine lock held."""
        jobs = self.queue.take_group(self.max_rows)
        if not jobs:
            return 0
        head = jobs[0]
        dec = self.bucket_for(sum(j.n for j in jobs))
        free = list(range(dec.rows))
        active: list[_Live] = []
        served = 0
        try:
            if head.steer_writes:
                self.set_steer(head.steer_writes, **head.steer_kwargs)
            for job in jobs:
                lv = self._admit(dec, job, free)
                served += 1
                if lv is not None:
                    active.append(lv)
            self.stats["batches"] += 1
            while True:
                for lv in [lv for lv in active if not any(lv.live)]:
                    active.remove(lv)
                    free.extend(lv.rows)
                    self._finish(lv)
                if self.refill and not self.should_yield():
                    while free:
                        job = self.queue.take_fitting(head.key, len(free))
                        if job is None:
                            break
                        lv = self._admit(dec, job, free)
                        served += 1
                        self.stats["refilled"] += 1
                        if lv is not None:
                            active.append(lv)
                    for lv in [lv for lv in active if not any(lv.live)]:  # a refill may end at its first token
                        active.remove(lv)
                        free.extend(lv.rows)
                        self._finish(lv)
                if not active:
                    break
                toks = dec.step(head.params).view(-1).tolist()
                self.stats["steps"] += 1
                self.stats["row_steps"] += dec.rows
                for lv in active:
                    self._consume(dec, lv, [toks[r] for r in lv.rows])
        except Exception as exc:
            logger.exception("xbatch: batch failed with %d live requests", len(active))
            for lv in active:
                self._deliver(lv.job, exc)
            active.clear()
        finally:
            self.set_steer(None)  # steering state never outlives the batch
        return served


class XBatchScheduler:
    """The asyncio side: ``submit`` enqueues a job and awaits its result; one pump task runs batches while jobs
    wait. ``run_on_worker(fn)`` runs ``fn`` on the generation thread; ``lock`` is the engine lock every engine
    user holds, so other request paths interleave between batches."""

    def __init__(self, runner: XBatchRunner, lock: asyncio.Lock,
                 run_on_worker: Callable[[Callable[[], int]], Awaitable[int]]) -> None:
        self.runner = runner
        self.lock = lock
        self.run_on_worker = run_on_worker
        self._pump_task: Optional[asyncio.Task] = None

    async def submit(self, job: XBatchJob) -> XBatchResult:
        """Queue ``job`` (its ``done`` is replaced) and wait for its result; re-raises the exception that ended it.
        A job wider than the widest bucket is refused (ValueError): no batch could ever take it."""
        if not 1 <= job.n <= self.runner.max_rows:
            raise ValueError(f"n={job.n} outside [1, {self.runner.max_rows}]")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        def resolve(outcome: XBatchResult | BaseException) -> None:
            if fut.done():  # the awaiting request was cancelled
                return
            if isinstance(outcome, BaseException):
                fut.set_exception(outcome)
            else:
                fut.set_result(outcome)

        job.done = lambda outcome: loop.call_soon_threadsafe(resolve, outcome)
        self.runner.queue.put(job)
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._pump())
        return await fut

    async def _pump(self) -> None:
        while len(self.runner.queue):
            async with self.lock:
                try:
                    await self.run_on_worker(self.runner.run_batch)
                except Exception as exc:
                    # run_batch delivers its own failures; reaching here means the generation thread itself is
                    # gone (e.g. shutdown), so nothing queued can ever run: fail it rather than spin.
                    logger.exception("xbatch: pump failed; failing %d queued requests", len(self.runner.queue))
                    for job in self.runner.queue.drain():
                        XBatchRunner._deliver(job, exc)
                    return
