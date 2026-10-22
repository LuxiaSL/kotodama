"""DDP options from the run flags: defaults reproduce the previous construction exactly (bucket view on, buffers
broadcast, DDP's own bucket size); --ddp_bucket_cap_mb / --ddp_no_broadcast_buffers change only their own key."""

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
