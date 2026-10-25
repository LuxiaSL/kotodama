"""Per-microbatch gradient statistics and spike capture (``--micro_grad_stats``, ``--spike_capture_factor``).

A loss spike or grad-norm spike is cheap to see and expensive to explain: the step's data can be reconstructed later,
but the model state at that step is gone. These two opt-in tools keep what is needed at the step itself.

``MicroGradStats`` records, per data-parallel rank and per micro-step, the micro-batch loss and the squared norm of the
rank's ACCUMULATED local gradient after that micro-step (S_1 … S_A). Under DDP the last micro-step's all-reduce
overwrites the local gradient, so S_A comes from a comm hook that sums each bucket's squares before the all-reduce.
The jump S_m − S_{m−1} carries the cross term 2⟨G_{m−1}, g_m⟩, so it is a localizer, not an exact per-micro norm: a
micro-batch whose gradient dwarfs the rest shows as the one big jump. One JSONL row per step (rank 0 writes).

``SpikeDetector`` flags a step whose global grad norm exceeds ``factor`` × the running median of the previous
``window`` steps. On a flagged step the trainer saves the weights from BEFORE the optimizer step plus a manifest, so
the spiking batch can be re-scored offline against the exact state that produced the spike.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def grad_sq_norm(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Sum of squared gradient entries over ``params`` (fp32, on device, no host sync)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return torch.zeros(())
    norms = torch._foreach_norm(grads, 2)
    return torch.stack([n.float() for n in norms]).square().sum()


def jumps(sq: list[float]) -> list[float]:
    """S_m − S_{m−1} for the accumulated squared norms S_1..S_A (S_0 = 0)."""
    return [b - a for a, b in zip([0.0] + sq[:-1], sq)]


class MicroGradStats:
    """Collects per-micro losses and accumulated local squared grad norms for one step at a time.

    Usage per step: ``begin(n_micro)``; after each micro-step's backward ``after_micro(i, loss)``; after the step's
    gradient is final ``finish(step, grad_norm)`` → the step's record (identical on every rank) or None if unused."""

    def __init__(self, params: list[torch.nn.Parameter], device: torch.device,
                 group: Optional[dist.ProcessGroup] = None, ddp: bool = False,
                 out_path: Optional[Path] = None) -> None:
        self.params = params
        self.device = device
        self.group = group
        self.ddp = ddp
        self.out_path = out_path
        self._n = 0
        self._loss: Optional[torch.Tensor] = None
        self._sq: Optional[torch.Tensor] = None
        self._hook_sq = torch.zeros((), device=device)

    # -- DDP comm hook: S_A for the syncing (last) micro-step, read before the all-reduce averages it ----------------
    def comm_hook(self, state, bucket):  # noqa: ANN001,ANN201 — DDP validates these annotations by identity, and
        # postponed (string) annotations fail its check; (Any, dist.GradBucket) -> Future[Tensor]
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

        self._hook_sq += bucket.buffer().float().square().sum()
        return allreduce_hook(state, bucket)

    def begin(self, n_micro: int) -> None:
        self._n = n_micro
        self._loss = torch.zeros(n_micro, device=self.device)
        self._sq = torch.zeros(n_micro, device=self.device)
        self._hook_sq.zero_()

    def after_micro(self, i: int, loss: torch.Tensor) -> None:
        assert self._loss is not None and self._sq is not None, "begin() first"
        self._loss[i] = loss.detach().float()
        if not (self.ddp and i == self._n - 1):  # under DDP the last micro's local grad is read by comm_hook
            self._sq[i] = grad_sq_norm(self.params).to(self.device)

    def finish(self, step: int, grad_norm: float) -> dict[str, Any]:
        assert self._loss is not None and self._sq is not None, "begin() first"
        if self.ddp:
            self._sq[self._n - 1] = self._hook_sq
        local = torch.cat([self._loss, self._sq])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size(self.group) > 1:
            parts = [torch.empty_like(local) for _ in range(dist.get_world_size(self.group))]
            dist.all_gather(parts, local, group=self.group)  # list form: gloo has no all_gather_into_tensor
            rows = torch.stack(parts).tolist()
        else:
            rows = [local.tolist()]
        n = self._n
        rec = {"step": step, "grad_norm": grad_norm,
               "loss": [r[:n] for r in rows], "sq": [r[n:] for r in rows]}
        return rec

    def write(self, rec: dict[str, Any]) -> None:
        if self.out_path is None:
            return
        try:
            with open(self.out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:  # telemetry must never stop training
            logger.warning("micro_grad_stats: write failed (%s); disabling the file", e)
            self.out_path = None


def summarize(rec: dict[str, Any]) -> dict[str, float]:
    """Telemetry scalars from one record: the largest per-micro jump as a multiple of the median jump, and the
    largest rank norm as a multiple of the median rank norm (both ≈ 1–3 on ordinary steps)."""
    all_jumps = [j for sq in rec["sq"] for j in jumps(sq)]
    rank_norms = [max(sq[-1], 0.0) ** 0.5 for sq in rec["sq"]]
    med_j = statistics.median(abs(j) for j in all_jumps) or 1e-30
    med_r = statistics.median(rank_norms) or 1e-30
    return {"gradstats/micro_jump_max_over_median": max(all_jumps) / med_j,
            "gradstats/rank_norm_max_over_median": max(rank_norms) / med_r}


class SpikeDetector:
    """``update(gn)`` → True when ``gn > factor × median(previous window)``, once ``min_history`` steps are in, and at
    most ``max_captures`` times per run. Every value (spike or not) joins the window."""

    def __init__(self, factor: float, window: int = 200, min_history: int = 50, max_captures: int = 5) -> None:
        if factor <= 1.0:
            raise ValueError(f"spike factor must be > 1, got {factor}")
        self.factor = factor
        self.min_history = min(min_history, window)
        self.max_captures = max_captures
        self.hist: deque[float] = deque(maxlen=window)
        self.captures = 0
        self.last_median = float("nan")

    def update(self, gn: float) -> bool:
        spike = False
        if len(self.hist) >= self.min_history and self.captures < self.max_captures:
            self.last_median = statistics.median(self.hist)
            spike = gn > self.factor * self.last_median
            self.captures += int(spike)
        self.hist.append(gn)
        return spike
