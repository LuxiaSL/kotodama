"""Phase-0 throughput-patch equivalence tests (CPU, no GPU required).

Every optimization in the 7B throughput program's Tier A must be exact-math.
These tests verify bitwise/allclose equivalence of:
  1. RoPE pre-gather (ndim==3 path) vs per-layer gather
  2. Vectorized position_ids vs the reference per-doc loop
  3. attn_res_freeze_unused — frozen params get no grads, loss unchanged
  4. Muon distributed flag — no-op without an initialized process group
  5. _PrefetchIterator — identical batch stream + exact resume state

Run: python tests/test_phase0_equivalence.py   (from pretraining/)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import TokenizedDataset, collate_packed, compute_doc_boundaries
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from src.training.muon import Muon


def _smoke_config(**overrides) -> LuxiaModelConfig:
    base = dict(
        hidden_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
        attn_res=True,
        attn_res_boundaries=[0, 1, 2],
    )
    base.update(overrides)
    return LuxiaModelConfig(**base)


def test_rope_pregather_equivalence() -> None:
    """ndim==3 pre-gathered tables must match per-layer gather exactly."""
    torch.manual_seed(0)
    config = _smoke_config(attn_res=False)
    model = LuxiaBaseModel(config)
    attn = model.layers[0].attn

    B, S = 2, 32
    x = torch.randn(B, S, config.hidden_size)
    position_ids = torch.randint(0, 64, (B, S))

    out_gather, _ = attn._forward_sdpa(
        x, model.rope_cos, model.rope_sin, position_ids=position_ids
    )
    out_pregathered, _ = attn._forward_sdpa(
        x,
        model.rope_cos[position_ids],
        model.rope_sin[position_ids],
        position_ids=None,
    )
    assert torch.equal(out_gather, out_pregathered), "RoPE pre-gather mismatch (sdpa)"
    print("PASS rope_pregather_equivalence")


def test_position_ids_vectorization() -> None:
    """Vectorized position_ids must match the reference per-doc loop."""

    def reference(tokens: np.ndarray, eos_id: int = 0):
        seq_len = len(tokens)
        eos_positions = np.where(tokens == eos_id)[0]
        if len(eos_positions) == 0:
            return np.arange(seq_len, dtype=np.int32)
        doc_starts = np.empty(len(eos_positions) + 1, dtype=np.int32)
        doc_starts[0] = 0
        doc_starts[1:] = eos_positions + 1
        doc_starts = doc_starts[doc_starts < seq_len]
        cu = np.empty(len(doc_starts) + 1, dtype=np.int32)
        cu[:-1] = doc_starts
        cu[-1] = seq_len
        pos = np.empty(seq_len, dtype=np.int32)
        for i in range(len(cu) - 1):
            pos[cu[i]:cu[i + 1]] = np.arange(cu[i + 1] - cu[i], dtype=np.int32)
        return pos

    rng = np.random.RandomState(7)
    cases = []
    for _ in range(50):
        toks = rng.randint(0, 30, size=rng.randint(1, 200)).astype(np.uint16)
        cases.append(toks)
    # Edge cases: no EOS, all EOS, EOS at end, consecutive EOS, single token
    cases += [
        np.array([5, 6, 7], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.uint16),
        np.array([5, 6, 0], dtype=np.uint16),
        np.array([5, 0, 0, 6], dtype=np.uint16),
        np.array([0], dtype=np.uint16),
        np.array([7], dtype=np.uint16),
    ]
    for toks in cases:
        _, pos_new, _ = compute_doc_boundaries(toks, eos_id=0)
        pos_ref = reference(toks, eos_id=0)
        assert np.array_equal(pos_new, pos_ref), f"position_ids mismatch for {toks[:16]}"
    print(f"PASS position_ids_vectorization ({len(cases)} cases)")


def test_freeze_unused_attn_res() -> None:
    """Frozen first-boundary routing params: no grads, identical loss,
    and every OTHER param still receives a gradient (DDP-safety condition)."""
    torch.manual_seed(0)
    B, S = 2, 32
    input_ids = torch.randint(0, 256, (B, S))

    losses = []
    for freeze in (True, False):
        torch.manual_seed(0)
        model = LuxiaBaseModel(_smoke_config(attn_res_freeze_unused=freeze))
        model.train()
        out = model(input_ids, labels=input_ids)
        out["loss"].backward()
        losses.append(out["loss"].item())

        first = model.layers[0]
        if freeze:
            assert not first.attn_res_query.requires_grad
            assert first.attn_res_query.grad is None
            assert first.attn_res_norm.weight.grad is None
            # DDP requires every requires_grad param to receive a gradient
            missing = [
                n for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            assert not missing, f"params without grads (breaks DDP static mode): {missing}"
        else:
            # Unfrozen: the unused params exist, require grad, and get none —
            # this is exactly why find_unused_parameters was needed before
            assert first.attn_res_query.requires_grad
            assert first.attn_res_query.grad is None

    assert losses[0] == losses[1], f"freeze changed the loss: {losses}"
    print(f"PASS freeze_unused_attn_res (loss identical: {losses[0]:.6f})")


def test_muon_distributed_noop_without_dist() -> None:
    """distributed=True without an initialized process group must match
    distributed=False bitwise."""
    results = []
    for flag in (False, True):
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.randn(16, 32))
        opt = Muon([w], lr=0.02, distributed=flag)
        for _ in range(3):
            opt.zero_grad()
            loss = (w * torch.ones_like(w)).sum()
            loss.backward()
            opt.step()
        results.append(w.detach().clone())
    assert torch.equal(results[0], results[1]), "muon distributed flag changed single-proc result"
    print("PASS muon_distributed_noop_without_dist")


def test_prefetcher_stream_and_resume() -> None:
    """Prefetcher must yield the identical batch stream and produce resume
    states that continue with zero skipped/replayed sequences."""
    from src.training.train import _PrefetchIterator

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.RandomState(3)
        tokens = rng.randint(0, 200, size=20_000).astype(np.uint16)
        tokens[rng.choice(20_000, 300, replace=False)] = 0  # EOS markers
        bin_path = Path(td) / "toy.bin"
        tokens.tofile(bin_path)

        def make_dataset():
            return TokenizedDataset(
                path=bin_path, seq_len=128, rank=0, world_size=1,
                seed=11, doc_masking=True,
            )

        def make_loader(ds):
            return iter(torch.utils.data.DataLoader(
                ds, batch_size=4, num_workers=0, collate_fn=collate_packed,
            ))

        # Reference stream: plain loader
        ds_ref = make_dataset()
        it_ref = make_loader(ds_ref)
        ref_batches = [next(it_ref) for _ in range(12)]

        # Prefetched stream
        ds_pf = make_dataset()
        pf = _PrefetchIterator(lambda: make_loader(ds_pf), ds_pf, depth=3)
        pf_batches = [next(pf) for _ in range(12)]
        state_after_8: dict | None = None
        for i, (a, b) in enumerate(zip(ref_batches, pf_batches)):
            assert torch.equal(a["input_ids"], b["input_ids"]), f"batch {i} diverged"
            assert torch.equal(a["cu_seqlens"], b["cu_seqlens"]), f"cu_seqlens {i} diverged"

        # Resume from the state captured after batch 8
        pf.shutdown()

        ds_pf2 = make_dataset()
        pf2 = _PrefetchIterator(lambda: make_loader(ds_pf2), ds_pf2, depth=3)
        for _ in range(8):
            next(pf2)
        state_after_8 = pf2.consumed_state
        pf2.shutdown()

        ds_resume = make_dataset()
        ds_resume.load_state_dict(state_after_8)
        it_resume = make_loader(ds_resume)
        resumed = next(it_resume)
        assert torch.equal(resumed["input_ids"], ref_batches[8]["input_ids"]), (
            "resume state skipped or replayed data"
        )
    print("PASS prefetcher_stream_and_resume")


def test_doc_masked_forward_smoke() -> None:
    """End-to-end: attn_res + AC + doc masking on CPU (SDPA block-causal
    fallback) runs forward+backward and the hoisted RoPE path is exercised."""
    torch.manual_seed(0)
    config = _smoke_config(activation_checkpointing=True)
    model = LuxiaBaseModel(config)
    model.train()

    B, S = 2, 64
    toks = torch.randint(1, 256, (B, S))
    toks[:, 20] = 0
    toks[:, 45] = 0
    samples = []
    from src.data.dataset import PackedSample
    for b in range(B):
        cu, pos, ms = compute_doc_boundaries(toks[b].numpy().astype(np.uint16))
        samples.append(PackedSample(
            input_ids=toks[b], cu_seqlens=torch.from_numpy(cu),
            position_ids=torch.from_numpy(pos).long(), max_seqlen=ms,
        ))
    batch = collate_packed(samples)

    out = model(
        batch["input_ids"], labels=batch["input_ids"],
        cu_seqlens=batch["cu_seqlens"], max_seqlen=batch["max_seqlen"],
        position_ids=batch["position_ids"],
    )
    out["loss"].backward()
    assert torch.isfinite(out["loss"]), "non-finite loss"

    # Non-AttnRes + AC + doc masking: previously dropped position_ids/cu_seqlens
    torch.manual_seed(0)
    model2 = LuxiaBaseModel(_smoke_config(attn_res=False, activation_checkpointing=True))
    model2.train()
    out_ac = model2(
        batch["input_ids"], labels=batch["input_ids"],
        cu_seqlens=batch["cu_seqlens"], max_seqlen=batch["max_seqlen"],
        position_ids=batch["position_ids"],
    )
    model2.eval()  # AC off in eval — same math, no checkpointing
    with torch.no_grad():
        out_noac = model2(
            batch["input_ids"], labels=batch["input_ids"],
            cu_seqlens=batch["cu_seqlens"], max_seqlen=batch["max_seqlen"],
            position_ids=batch["position_ids"],
        )
    assert torch.allclose(out_ac["loss"], out_noac["loss"], atol=1e-6), (
        f"AC path diverges from non-AC: {out_ac['loss'].item()} vs {out_noac['loss'].item()} "
        "(doc-masking args dropped under AC?)"
    )
    print(f"PASS doc_masked_forward_smoke (AC==no-AC loss: {out_ac['loss'].item():.6f})")


if __name__ == "__main__":
    test_rope_pregather_equivalence()
    test_position_ids_vectorization()
    test_freeze_unused_attn_res()
    test_muon_distributed_noop_without_dist()
    test_prefetcher_stream_and_resume()
    test_doc_masked_forward_smoke()
    print("\nALL PHASE-0 EQUIVALENCE TESTS PASSED")
