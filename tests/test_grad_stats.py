"""--micro_grad_stats / --spike_capture_factor helpers (kotodama.training.grad_stats).

The DDP case runs two real gloo processes on CPU: each rank's recorded S_A (read by the comm hook before the
all-reduce) must equal that rank's own local accumulated gradient norm, and the hook must leave DDP's averaged
gradient unchanged."""
from __future__ import annotations

import os
import socket
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.training.grad_stats import MicroGradStats, SpikeDetector, grad_sq_norm, jumps, summarize


def test_jumps_and_sq_norm() -> None:
    assert jumps([1.0, 1.5, 9.0]) == [1.0, 0.5, 7.5]
    lin = torch.nn.Linear(4, 3)
    lin(torch.ones(2, 4)).sum().backward()
    ref = sum(float(p.grad.pow(2).sum()) for p in lin.parameters())
    assert float(grad_sq_norm(lin.parameters())) == pytest.approx(ref, rel=1e-6)


def test_spike_detector() -> None:
    with pytest.raises(ValueError):
        SpikeDetector(1.0)
    d = SpikeDetector(4.0, window=10, min_history=5, max_captures=1)
    assert not any(d.update(1.0) for _ in range(4))  # history too short to judge
    assert not d.update(100.0) or False  # 5th value: still only 4 in history → not judged
    d = SpikeDetector(4.0, window=10, min_history=5, max_captures=1)
    for _ in range(5):
        d.update(1.0)
    assert not d.update(3.9)
    assert d.update(4.1) and d.last_median == 1.0
    assert not d.update(50.0)  # max_captures reached


def test_single_process_records_accumulated_norms() -> None:
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 3)
    st = MicroGradStats(list(lin.parameters()), torch.device("cpu"))
    st.begin(2)
    expected = []
    for i in range(2):
        loss = lin(torch.randn(5, 4)).pow(2).mean()
        (loss / 2).backward()
        st.after_micro(i, loss)
        expected.append(sum(float(p.grad.pow(2).sum()) for p in lin.parameters()))
    rec = st.finish(7, 1.23)
    assert rec["step"] == 7 and rec["grad_norm"] == 1.23 and len(rec["loss"]) == 1
    assert rec["sq"][0] == pytest.approx(expected, rel=1e-5)
    s = summarize(rec)
    assert set(s) == {"gradstats/micro_jump_max_over_median", "gradstats/rank_norm_max_over_median"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, out: "mp.Queue") -> None:
    try:
        _worker_body(rank, port, out)
    except Exception as e:  # surface the failure instead of a queue timeout
        out.put((rank, f"{type(e).__name__}: {e}", None, None))


def _worker_body(rank: int, port: int, out: "mp.Queue") -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(0)
    base = torch.nn.Linear(6, 4)
    ref = torch.nn.Linear(6, 4)
    ref.load_state_dict(base.state_dict())
    model = DDP(base)
    st = MicroGradStats([p for p in base.parameters()], torch.device("cpu"), ddp=True)
    model.register_comm_hook(None, st.comm_hook)
    gen = torch.Generator().manual_seed(100 + rank)  # different data per rank
    xs = [torch.randn(5, 6, generator=gen) for _ in range(3)]
    st.begin(3)
    for i, x in enumerate(xs):
        ctx = model.no_sync() if i < 2 else nullcontext()
        with ctx:
            loss = model(x).pow(2).mean()
            (loss / 3).backward()
        st.after_micro(i, loss)
    rec = st.finish(0, 0.0)
    for x in xs:  # local reference without DDP: this rank's own accumulated gradient
        (ref(x).pow(2).mean() / 3).backward()
    local_sq = sum(float(p.grad.pow(2).sum()) for p in ref.parameters())
    avg = [p.grad.clone() for p in base.parameters()]
    ref_g = [p.grad.clone() for p in ref.parameters()]
    for g in ref_g:
        dist.all_reduce(g)
        g /= 2
    same_avg = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(avg, ref_g))
    out.put((rank, rec, local_sq, same_avg))
    dist.destroy_process_group()


def test_ddp_comm_hook_reads_local_norm_before_allreduce() -> None:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    res = sorted(q.get(timeout=120) for _ in procs)
    for p in procs:
        p.join(timeout=60)
    errors = [r[1] for r in res if isinstance(r[1], str)]
    assert not errors, errors
    recs = [r[1] for r in res]
    assert recs[0] == recs[1]  # all-gathered record is identical on both ranks
    for rank, _rec, local_sq, same_avg in res:
        assert same_avg, "comm hook changed DDP's averaged gradient"
        assert recs[0]["sq"][rank][-1] == pytest.approx(local_sq, rel=1e-5)
        assert recs[0]["sq"][rank][-1] != pytest.approx(recs[0]["sq"][1 - rank][-1], rel=1e-3)  # really local
