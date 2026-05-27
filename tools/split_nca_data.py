"""Split a generated NCA binary into train and eval sets.

Usage:
    python tools/split_nca_data.py \
        --input data/nca_seed17_full.bin \
        --train data/nca_seed17_3b_train.bin \
        --eval data/nca_seed17_50m_eval.bin \
        --train_tokens 3000000000
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--train_tokens", type=int, required=True)
    args = p.parse_args()

    src = Path(args.input)
    total_bytes = src.stat().st_size
    total_tokens = total_bytes // 2
    train_bytes = args.train_tokens * 2
    eval_bytes = total_bytes - train_bytes

    if train_bytes >= total_bytes:
        raise ValueError(
            f"train_tokens ({args.train_tokens:,}) >= total tokens ({total_tokens:,})"
        )

    print(f"Input: {total_tokens:,} tokens ({total_bytes / 1e9:.2f} GB)")
    print(f"Train: {args.train_tokens:,} tokens ({train_bytes / 1e9:.2f} GB)")
    print(f"Eval:  {eval_bytes // 2:,} tokens ({eval_bytes / 1e9:.2f} GB)")

    chunk = 100_000_000  # 100MB chunks

    with open(src, "rb") as f:
        with open(args.train, "wb") as t:
            remaining = train_bytes
            while remaining > 0:
                data = f.read(min(remaining, chunk))
                if not data:
                    break
                t.write(data)
                remaining -= len(data)

        with open(args.eval, "wb") as e:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                e.write(data)

    train_actual = Path(args.train).stat().st_size // 2
    eval_actual = Path(args.eval).stat().st_size // 2
    print(f"Done: train={train_actual:,} eval={eval_actual:,}")


if __name__ == "__main__":
    main()
