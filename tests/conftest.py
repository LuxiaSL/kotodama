import os
import sys
from pathlib import Path

# Under pytest-xdist every worker is its own process; left alone, each one's torch and BLAS thread pools claim every
# core and the workers fight over them. One thread per worker lets the workers be the parallelism.
_XDIST_WORKER = "PYTEST_XDIST_WORKER" in os.environ
if _XDIST_WORKER:
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    if _XDIST_WORKER:
        import torch

        torch.set_num_threads(1)
