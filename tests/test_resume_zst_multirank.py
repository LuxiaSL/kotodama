"""Multi-rank resume from a compressed (.pt.zst) checkpoint: every rank loads the same
state, and the decompressed temp file is gone afterwards. (Each rank used to zstd -f
into one shared path and unlink it — rank 1 died with FileNotFoundError.)"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.training.checkpoint import AsyncCheckpointManager

WORLD = 2


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Linear(8, 4)


def _write_compressed_ckpt(d: Path) -> torch.nn.Module:
    m = _model()
    with torch.no_grad():
        m.weight.add_(1.0)
    muon = torch.optim.SGD(m.parameters(), lr=0.1)
    adamw = torch.optim.AdamW(m.parameters(), lr=0.1)
    p = d / "step_00000599.pt"
    torch.save({"model": m.state_dict(), "muon_optimizer": muon.state_dict(),
                "adamw_optimizer": adamw.state_dict(), "step": 599,
                "tokens_consumed": 123}, p)
    subprocess.run(["zstd", "-q", "--rm", str(p)], check=True)
    return m


def _worker(rank: int, ckpt_dir: str, shm_dir: str, port: int, out_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        import kotodama.training.checkpoint as ck
        real_run, calls = ck.subprocess.run, []
        ck.subprocess.run = lambda cmd, *a, **k: (calls.append("decomp" if "-d" in cmd else cmd[0]), real_run(cmd, *a, **k))[1]
        m = _model()
        muon = torch.optim.SGD(m.parameters(), lr=0.1)
        adamw = torch.optim.AdamW(m.parameters(), lr=0.1)
        mgr = AsyncCheckpointManager(ckpt_dir, rank=rank, keep_last_n=1, shm_dir=shm_dir)
        state = mgr.load_latest(m, muon, adamw, torch.device("cpu"))
        assert state is not None and state["step"] == 599
        torch.save(m.state_dict(), Path(out_dir) / f"rank{rank}.pt")
        (Path(out_dir) / f"zstd_calls{rank}").write_text(str(calls.count("decomp")))
        if hasattr(mgr, "shutdown"):
            mgr.shutdown()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd binary not installed")
def test_multirank_resume_from_zst(tmp_path):
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    ref = _write_compressed_ckpt(ckpt_dir)
    port = 29500 + (os.getpid() % 1000)
    mp.spawn(_worker, args=(str(ckpt_dir), str(tmp_path / "shm"), port, str(tmp_path)),
             nprocs=WORLD, join=True)
    for r in range(WORLD):
        got = torch.load(tmp_path / f"rank{r}.pt")
        assert all(torch.equal(got[k], ref.state_dict()[k]) for k in got)
    # only rank 0 decompresses (concurrent zstd -f into one path = torn file / unlink race)
    assert [(tmp_path / f"zstd_calls{r}").read_text() for r in range(WORLD)] == ["1", "0"]
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["step_00000599.pt.zst"]  # no temp left
