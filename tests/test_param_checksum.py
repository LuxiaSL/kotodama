"""Cross-rank weight checksum: 0 for identical replicas, nonzero for a desync.

Two real gloo processes on CPU; rank 1 perturbs one weight on purpose."""
from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, desync: bool, out: "mp.Queue") -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from kotodama.training.train import param_checksum_spread

    torch.manual_seed(0)
    model = torch.nn.Linear(16, 8)
    if desync and rank == 1:
        with torch.no_grad():
            model.weight[0, 0] += 1e-3
    out.put((rank, param_checksum_spread(model)))
    dist.destroy_process_group()


def _run(desync: bool) -> list[float]:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, desync, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    return [q.get(timeout=5)[1] for _ in range(2)]


def test_identical_replicas_have_zero_spread():
    assert _run(desync=False) == [0.0, 0.0]


def test_desynced_replica_is_detected():
    spreads = _run(desync=True)
    assert all(s > 0 for s in spreads)
