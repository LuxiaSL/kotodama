"""
Smoke test for document-boundary attention masking.

Tests:
1. compute_doc_boundaries produces correct cu_seqlens/position_ids
2. PackedSample + collate_packed work correctly
3. Model forward pass with cu_seqlens (SDPA fallback on CPU)
4. Loss decreases over a few steps with doc masking enabled

Run: python scripts/utils/smoke_doc_masking.py
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from src.data.dataset import (
    PackedSample,
    RandomTokenDataset,
    collate_packed,
    compute_doc_boundaries,
)
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig


def test_compute_doc_boundaries() -> None:
    print("=== test_compute_doc_boundaries ===")

    # Case 1: No EOS — entire sequence is one doc
    tokens = np.array([10, 20, 30, 40, 50], dtype=np.uint16)
    cu, pos, max_sl = compute_doc_boundaries(tokens, eos_id=0)
    assert list(cu) == [0, 5], f"Expected [0, 5], got {list(cu)}"
    assert list(pos) == [0, 1, 2, 3, 4], f"Bad position_ids: {list(pos)}"
    assert max_sl == 5
    print("  No EOS: PASS")

    # Case 2: Single EOS in middle
    tokens = np.array([10, 20, 0, 30, 40], dtype=np.uint16)
    cu, pos, max_sl = compute_doc_boundaries(tokens, eos_id=0)
    assert list(cu) == [0, 3, 5], f"Expected [0, 3, 5], got {list(cu)}"
    assert list(pos) == [0, 1, 2, 0, 1], f"Bad position_ids: {list(pos)}"
    assert max_sl == 3
    print("  Single EOS: PASS")

    # Case 3: EOS at end
    tokens = np.array([10, 20, 30, 40, 0], dtype=np.uint16)
    cu, pos, max_sl = compute_doc_boundaries(tokens, eos_id=0)
    assert list(cu) == [0, 5], f"Expected [0, 5], got {list(cu)}"
    assert list(pos) == [0, 1, 2, 3, 4], f"Bad position_ids: {list(pos)}"
    assert max_sl == 5
    print("  EOS at end: PASS")

    # Case 4: Multiple EOS (3 documents)
    tokens = np.array([1, 2, 0, 3, 4, 5, 0, 6, 7, 8], dtype=np.uint16)
    cu, pos, max_sl = compute_doc_boundaries(tokens, eos_id=0)
    assert list(cu) == [0, 3, 7, 10], f"Expected [0, 3, 7, 10], got {list(cu)}"
    assert list(pos) == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2], f"Bad position_ids: {list(pos)}"
    assert max_sl == 4
    print("  Multiple EOS: PASS")

    # Case 5: Consecutive EOS (empty docs filtered)
    tokens = np.array([1, 0, 0, 2, 3, 0], dtype=np.uint16)
    cu, pos, max_sl = compute_doc_boundaries(tokens, eos_id=0)
    # EOS at positions 1, 2, 5. Doc starts: [0, 2, 3, 6(filtered)]
    # After filtering (< seq_len=6): starts [0, 2, 3]
    # cu_seqlens: [0, 2, 3, 6]
    assert cu[-1] == 6, f"Last cu_seqlen should be seq_len=6, got {cu[-1]}"
    # Positions reset at each doc boundary
    print(f"  Consecutive EOS: cu={list(cu)}, pos={list(pos)}")
    # Verify positions are valid (reset at boundaries)
    for i in range(len(cu) - 1):
        doc_pos = pos[cu[i]:cu[i+1]]
        expected = np.arange(cu[i+1] - cu[i], dtype=np.int32)
        assert np.array_equal(doc_pos, expected), f"Doc {i}: expected {list(expected)}, got {list(doc_pos)}"
    print("  Consecutive EOS: PASS")

    print("All boundary tests passed!\n")


def test_collate_packed() -> None:
    print("=== test_collate_packed ===")

    seq_len = 8
    # Sample 1: docs at [0:4], [4:8]
    s1 = PackedSample(
        input_ids=torch.tensor([1, 2, 3, 0, 4, 5, 6, 0], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.int32),
        position_ids=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64),
        max_seqlen=4,
    )
    # Sample 2: docs at [0:6], [6:8]
    s2 = PackedSample(
        input_ids=torch.tensor([7, 8, 9, 10, 11, 0, 12, 0], dtype=torch.int64),
        cu_seqlens=torch.tensor([0, 6, 8], dtype=torch.int32),
        position_ids=torch.tensor([0, 1, 2, 3, 4, 5, 0, 1], dtype=torch.int64),
        max_seqlen=6,
    )

    batch = collate_packed([s1, s2])

    assert batch["input_ids"].shape == (2, 8), f"Bad shape: {batch['input_ids'].shape}"
    assert batch["max_seqlen"] == 6
    # cu_seqlens should be: [0, 4, 8, 14, 16]  (s2 offsets by 8)
    expected_cu = torch.tensor([0, 4, 8, 14, 16], dtype=torch.int32)
    assert torch.equal(batch["cu_seqlens"], expected_cu), f"Bad cu_seqlens: {batch['cu_seqlens']}"
    assert batch["position_ids"].shape == (2, 8)
    print("  Collation: PASS\n")


def test_model_forward_with_doc_masking() -> None:
    print("=== test_model_forward (SDPA with doc masking) ===")
    device = torch.device("cpu")

    config = LuxiaModelConfig(
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=1024,
        max_position_embeddings=64,
        qk_norm=True,
        z_loss_weight=1e-5,
    )

    model = LuxiaBaseModel(config).to(device)
    model.train()

    # Create a batch with doc masking
    seq_len = 16
    batch_size = 2
    input_ids = torch.randint(1, 1024, (batch_size, seq_len))
    # Insert EOS at positions to create doc boundaries
    input_ids[0, 5] = 0
    input_ids[0, 11] = 0
    input_ids[1, 8] = 0

    # Build cu_seqlens + position_ids for each sample
    samples = []
    for b in range(batch_size):
        tokens_np = input_ids[b].numpy().astype(np.uint16)
        cu, pos, max_sl = compute_doc_boundaries(tokens_np, eos_id=0)
        samples.append(PackedSample(
            input_ids=input_ids[b],
            cu_seqlens=torch.from_numpy(cu),
            position_ids=torch.from_numpy(pos).long(),
            max_seqlen=max_sl,
        ))

    batch = collate_packed(samples)

    # Forward pass (will use SDPA fallback on CPU)
    output = model(
        input_ids=batch["input_ids"].to(device),
        labels=batch["input_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        cu_seqlens=batch["cu_seqlens"].to(device),
        max_seqlen=batch["max_seqlen"],
    )

    assert "loss" in output, "No loss in output"
    assert output["loss"].isfinite(), f"Loss is not finite: {output['loss']}"
    print(f"  Loss: {output['loss'].item():.4f}")
    print("  Forward pass: PASS\n")


def test_model_forward_attn_res_with_doc_masking() -> None:
    print("=== test_model_forward (AttnRes + doc masking) ===")
    device = torch.device("cpu")

    config = LuxiaModelConfig(
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=1024,
        max_position_embeddings=64,
        qk_norm=True,
        z_loss_weight=1e-5,
        attn_res=True,
        attn_res_boundaries=[0, 2],
    )

    model = LuxiaBaseModel(config).to(device)
    model.train()

    seq_len = 16
    batch_size = 2
    input_ids = torch.randint(1, 1024, (batch_size, seq_len))
    input_ids[0, 7] = 0
    input_ids[1, 4] = 0
    input_ids[1, 12] = 0

    samples = []
    for b in range(batch_size):
        tokens_np = input_ids[b].numpy().astype(np.uint16)
        cu, pos, max_sl = compute_doc_boundaries(tokens_np, eos_id=0)
        samples.append(PackedSample(
            input_ids=input_ids[b],
            cu_seqlens=torch.from_numpy(cu),
            position_ids=torch.from_numpy(pos).long(),
            max_seqlen=max_sl,
        ))

    batch = collate_packed(samples)

    output = model(
        input_ids=batch["input_ids"].to(device),
        labels=batch["input_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        cu_seqlens=batch["cu_seqlens"].to(device),
        max_seqlen=batch["max_seqlen"],
    )

    assert "loss" in output, "No loss in output"
    assert output["loss"].isfinite(), f"Loss is not finite: {output['loss']}"
    print(f"  Loss: {output['loss'].item():.4f}")
    print("  AttnRes forward pass: PASS\n")


def test_training_steps() -> None:
    print("=== test_training_steps (loss should decrease) ===")
    device = torch.device("cpu")

    config = LuxiaModelConfig(
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=1024,
        max_position_embeddings=64,
        qk_norm=True,
        z_loss_weight=1e-5,
    )

    model = LuxiaBaseModel(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Use RandomTokenDataset with doc_masking
    dataset = RandomTokenDataset(
        vocab_size=1024,
        seq_len=32,
        doc_masking=True,
        seed=42,
    )

    data_iter = iter(torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collate_packed,
    ))

    losses = []
    for step in range(20):
        batch = next(data_iter)
        output = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["input_ids"].to(device),
            position_ids=batch["position_ids"].to(device),
            cu_seqlens=batch["cu_seqlens"].to(device),
            max_seqlen=batch["max_seqlen"],
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    print(f"  Step  0 loss: {losses[0]:.4f}")
    print(f"  Step 19 loss: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    print("  Training convergence: PASS\n")


def main() -> None:
    t0 = time.time()
    test_compute_doc_boundaries()
    test_collate_packed()
    test_model_forward_with_doc_masking()
    test_model_forward_attn_res_with_doc_masking()
    test_training_steps()
    print(f"All smoke tests passed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
