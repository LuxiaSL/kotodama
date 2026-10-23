import os

import triton

# KOTODAMA_ATTNRES_EPS_FP64_COMPAT=1: the forward kernels add the runtime eps (typed fp64 by inductor inside a
# compiled graph, so the rms / logits / softmax run in fp64) instead of the constexpr one (fp32 under any launcher).
# For continuing runs trained before the constexpr existed; eager launches are fp32 either way.
EPS_FP64_COMPAT = os.environ.get("KOTODAMA_ATTNRES_EPS_FP64_COMPAT", "0") == "1"

forward_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=1, num_stages=4),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=3),
]

phase1_backward_configs = [
    triton.Config({}, num_warps=1, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=4, num_stages=4),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
]


phase2_backward_configs = [
    triton.Config({"BLOCK_BT": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_BT": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 16}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_BT": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BT": 32}, num_warps=8, num_stages=1),
]

# v2 backward: hidden-dim-CHUNKED two-pass kernel. The v1 kernel holds
# full 4096-wide rows in registers — (BLOCK_BT, 4096)
# fp32 tiles x ~6 live tensors = massive register spill to local memory,
# which is why it runs at <900 GB/s effective on an 8 TB/s part. BLOCK_D
# bounds the live tile
# to (BLOCK_BT, BLOCK_D). BLOCK_BT must stay >= PHASE2_BWD_MIN_BLOCK_BT (the
# wrapper sizes the grad_pseudo_query partials buffer with it).
PHASE2_BWD_MIN_BLOCK_BT = 8

phase2_backward_v2_configs = [
    triton.Config({"BLOCK_BT": 8, "BLOCK_D": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_BT": 8, "BLOCK_D": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 16, "BLOCK_D": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BT": 16, "BLOCK_D": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 16, "BLOCK_D": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BT": 32, "BLOCK_D": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 32, "BLOCK_D": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BT": 64, "BLOCK_D": 256}, num_warps=8, num_stages=2),
]


# v2 phase-1 backward: same register-spill cure as phase-2 v2 (BLOCK_D
# chunking, two-pass), plus per-PROGRAM query-grad partials instead of v1's
# per-token (num_queries, BT, D) fp32 monster + reduce kernel. BLOCK_BT must
# stay >= PHASE1_BWD_MIN_BLOCK_BT (partials buffer sizing).
PHASE1_BWD_MIN_BLOCK_BT = 4

phase1_backward_v2_configs = [
    triton.Config({"BLOCK_BT": 4, "BLOCK_D": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_BT": 4, "BLOCK_D": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 8, "BLOCK_D": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BT": 8, "BLOCK_D": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 8, "BLOCK_D": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_BT": 16, "BLOCK_D": 128}, num_warps=8, num_stages=2),
]


# Single-query online-softmax Phase-1 forward: one (BLOCK_BT, D) fp32 accumulator + the current source row live.
phase1_online_forward_configs = [
    triton.Config({"BLOCK_BT": 1}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_BT": 1}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_BT": 2}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 4}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_BT": 4}, num_warps=16, num_stages=2),
]


# Chained single-query phase-1 backward (phase_1_route_chained_backward_kernel; unmasked 2-D tiles). One token per
# program (measured best on B200 at S 6-8, D 1536; 2+ tokens per program, or several walked serially, are slower), so
# the query-grad partials have one row per token and every row is written. BLOCK_D must divide the hidden size (the
# wrapper checks).
phase1_route_chained_backward_configs = [
    triton.Config({"BLOCK_BT": 1, "BLOCK_D": 512}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_BT": 1, "BLOCK_D": 512}, num_warps=2, num_stages=3),
    triton.Config({"BLOCK_BT": 1, "BLOCK_D": 512}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_BT": 1, "BLOCK_D": 256}, num_warps=1, num_stages=3),
]


def set_autotune_configs(kernel, configs):
    kernel.configs = list(configs)
    kernel.cache.clear()
