"""The AttnRes Triton kernels take the token count BT as a runtime argument, never a constexpr.

A constexpr is compiled into the kernel, so each distinct batch shape (rows x length) compiles a fresh set of kernels,
one per source-block count and autotune pick, and every compiled kernel stays loaded for the life of the process. A
training run sees a handful of shapes; an eager caller whose lengths vary (a server scoring prompts, an eval sweeping
documents) sees a new one per request, and host memory grows until the process is killed. This test reads the kernel
signatures, so it runs without a GPU.
"""

from __future__ import annotations

import pytest
import triton

from kotodama.model.flash_attn_res.kernels import phase_1, phase_2


def _kernels():
    for module in (phase_1, phase_2):
        for name in dir(module):
            obj = getattr(module, name)
            fn = getattr(obj, "fn", obj)  # an Autotuner wraps the JITFunction
            if isinstance(fn, triton.runtime.jit.JITFunction):
                yield f"{module.__name__.rsplit('.', 1)[-1]}.{name}", fn


KERNELS = dict(_kernels())


def test_kernels_found():
    assert "phase_1.phase_1_batched_attention_forward_kernel" in KERNELS
    assert "phase_2.phase_2_online_softmax_merge_backward_kernel" in KERNELS


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_token_count_is_a_runtime_argument(name):
    params = {p.name: p for p in KERNELS[name].params}
    if "BT" in params:
        assert not params["BT"].is_constexpr, f"{name} compiles BT in: one kernel set per batch shape"
