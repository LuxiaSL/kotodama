"""GeometricMonitor memory hygiene: a probe releases the CUDA cache only when the device is short of free memory
(releasing it unconditionally made the following training steps re-cudaMalloc their blocks)."""

from __future__ import annotations

import torch

from kotodama.monitoring import geometric
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])


def _monitor(threshold: float) -> GeometricMonitor:
    return GeometricMonitor(_Tiny(), MonitorConfig(release_cache_below_gb=threshold))


def test_release_only_when_short(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(geometric.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(geometric.torch.cuda, "empty_cache", lambda: calls.append(1))
    m = _monitor(8.0)
    monkeypatch.setattr(m, "_free_gb", lambda: 50.0)
    m._maybe_release_cache()
    assert calls == []
    monkeypatch.setattr(m, "_free_gb", lambda: 2.0)
    m._maybe_release_cache()
    assert calls == [1]


def test_defaults():
    cfg = MonitorConfig()
    assert cfg.release_cache_below_gb == 8.0 and cfg.sharpness_gpu_snapshot is True


def test_stable_rank_matches_svd():
    """_stable_rank's Gram + eigvalsh sigma_max equals the SVD one (the series stays continuous across the change)."""
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 96), (48, 48)]:
        W = torch.randn(*shape) @ torch.diag(torch.linspace(1.0, 0.1, shape[1]))
        exact = (W.double().pow(2).sum() / torch.linalg.svdvals(W.double())[0].pow(2)).item()
        assert abs(geometric._stable_rank(W) - exact) <= 1e-9 * exact
