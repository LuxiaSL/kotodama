"""Run-flag helpers: DDP options and the AttnRes eps-compat flag.

DDP defaults reproduce the previous construction exactly (bucket view on, buffers broadcast, DDP's own bucket size);
--ddp_bucket_cap_mb / --ddp_no_broadcast_buffers change only their own key. --attnres_eps_fp64_compat switches both
AttnRes forward ops to the runtime-eps path; the moonshot config turns it on and --no_attnres_eps_fp64_compat off.
"""

from __future__ import annotations

import sys

import pytest

from kotodama.training import train


def _args(*extra: str):
    argv = ["train.py", "--model_size", "proxy", *extra]
    old = sys.argv
    try:
        sys.argv = argv
        return train.parse_args()
    finally:
        sys.argv = old


def test_defaults_match_previous_construction():
    assert train.ddp_options(_args(), find_unused=False) == dict(
        gradient_as_bucket_view=True, find_unused_parameters=False, broadcast_buffers=True)


@pytest.mark.parametrize("flags,key,value", [
    (("--ddp_bucket_cap_mb", "200"), "bucket_cap_mb", 200.0),
    (("--ddp_no_broadcast_buffers",), "broadcast_buffers", False),
])
def test_each_flag_changes_only_its_key(flags, key, value):
    base = train.ddp_options(_args(), find_unused=True)
    got = train.ddp_options(_args(*flags), find_unused=True)
    assert got[key] == value
    assert {k: v for k, v in got.items() if k != key} == {k: v for k, v in base.items() if k != key}


def test_attnres_eps_compat_flag_sets_both_ops():
    from kotodama.model.flash_attn_res.ops import phase_1 as p1ops
    from kotodama.model.flash_attn_res.ops import phase_2 as p2ops

    saved = (p1ops.EPS_FP64_COMPAT, p2ops.EPS_FP64_COMPAT)
    try:
        p1ops.EPS_FP64_COMPAT = p2ops.EPS_FP64_COMPAT = False
        assert train.apply_attnres_eps_compat(_args()) is False
        assert train.apply_attnres_eps_compat(_args("--attnres_eps_fp64_compat")) is True
        assert p1ops.EPS_FP64_COMPAT and p2ops.EPS_FP64_COMPAT
        p1ops.EPS_FP64_COMPAT = p2ops.EPS_FP64_COMPAT = False
        args = _args("--config", "configs/moonshot-1b-shapeA.yaml")
        assert args.attnres_eps_fp64_compat is True  # the trunk lineage keeps its numerics by default
        assert _args("--config", "configs/moonshot-1b-shapeA.yaml",
                     "--no_attnres_eps_fp64_compat").attnres_eps_fp64_compat is False
    finally:
        p1ops.EPS_FP64_COMPAT, p2ops.EPS_FP64_COMPAT = saved
