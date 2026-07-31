import sys
from pathlib import Path

import torch
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import _load_checkpoint


def test_load_checkpoint_supports_plain_and_zstd_payloads(tmp_path):
    checkpoint = {"model": {"weight": torch.tensor([1.0, 2.0])}, "step": 7}
    plain_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, plain_path)

    compressed_path = tmp_path / "checkpoint.pt.zst"
    with plain_path.open("rb") as source, compressed_path.open("wb") as target:
        zstd.ZstdCompressor().copy_stream(source, target)

    plain = _load_checkpoint(plain_path)
    compressed = _load_checkpoint(compressed_path)

    assert plain["step"] == compressed["step"] == 7
    assert torch.equal(plain["model"]["weight"], compressed["model"]["weight"])
