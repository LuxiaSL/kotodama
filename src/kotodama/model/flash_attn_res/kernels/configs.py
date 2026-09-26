import triton

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

# v2 backward: hidden-dim-CHUNKED two-pass kernel. The v1 kernel (and the
# first v2 draft) held full 4096-wide rows in registers — (BLOCK_BT, 4096)
# fp32 tiles x ~6 live tensors = massive register spill to local memory,
# which is why it ran at <900 GB/s effective on an 8 TB/s part (same disease
# as the batched-P1 net-negative, 2026-05-17). BLOCK_D bounds the live tile
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


def set_autotune_configs(kernel, configs):
    kernel.configs = list(configs)
    kernel.cache.clear()
