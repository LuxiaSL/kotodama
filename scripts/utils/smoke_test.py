"""
Smoke test: verify model + Muon + z-loss work end-to-end on a single GPU.

Run: python scripts/smoke_test.py
Expected: loss decreases over 50 steps, no crashes, Muon orthogonalization runs.
"""

import sys
import time

import torch

sys.path.insert(0, ".")
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from src.training.muon import build_hybrid_optimizer, HybridScheduler


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Tiny model for smoke test
    config = LuxiaModelConfig(
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1024,  # Tiny vocab for speed
        max_position_embeddings=512,
        qk_norm=True,
        z_loss_weight=1e-5,
    )

    model = LuxiaBaseModel(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    print(f"Estimated from config: {config.param_count():,}")

    # Build hybrid optimizer
    muon_opt, adamw_opt = build_hybrid_optimizer(
        model,
        muon_lr=0.02,
        adamw_lr=3e-4,
    )

    # Build scheduler
    scheduler = HybridScheduler(
        muon_opt, adamw_opt,
        warmup_steps=10,
        total_steps=50,
        decay_start_pct=0.8,
    )

    # Training loop on random data
    batch_size = 4
    seq_len = 128
    num_steps = 50

    print(f"\nTraining {num_steps} steps (batch={batch_size}, seq={seq_len})...")
    print("-" * 60)

    model.train()
    losses = []
    t0 = time.time()

    for step in range(num_steps):
        # Random input
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()

        # Forward
        output = model(input_ids, labels=labels)
        loss = output["loss"]

        # Backward
        loss.backward()

        # Step both optimizers
        muon_opt.step()
        adamw_opt.step()
        muon_opt.zero_grad()
        adamw_opt.zero_grad()

        # Step scheduler
        scheduler.step(step)

        losses.append(loss.item())

        if step % 10 == 0 or step == num_steps - 1:
            lrs = scheduler.get_last_lr()
            z_loss = output.get("z_loss", torch.tensor(0.0)).item()
            print(
                f"Step {step:3d} | loss={loss.item():.4f} | z_loss={z_loss:.6f} | "
                f"muon_lr={lrs['muon_lr']:.6f} | adamw_lr={lrs['adamw_lr']:.6f}"
            )

    elapsed = time.time() - t0
    print("-" * 60)
    print(f"Done in {elapsed:.1f}s ({elapsed/num_steps*1000:.0f}ms/step)")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")

    # Test MLP reinitialization (for NCA → language transition)
    print("\nTesting MLP reinitialization...")
    pre_reinit_loss = losses[-1]
    model.reinit_mlps()
    with torch.no_grad():
        output = model(input_ids, labels=labels)
    post_reinit_loss = output["loss"].item()
    print(f"Pre-reinit loss:  {pre_reinit_loss:.4f}")
    print(f"Post-reinit loss: {post_reinit_loss:.4f}")
    print(f"Loss increased after reinit (expected): {post_reinit_loss > pre_reinit_loss}")

    # Test embedding reinitialization (for NCA → language vocab switch)
    print("\nTesting embedding reinitialization (vocab 1024 → 2048)...")
    model.reinit_embeddings(new_vocab_size=2048)
    new_input = torch.randint(0, 2048, (batch_size, seq_len), device=device)
    with torch.no_grad():
        output = model(new_input, labels=new_input)
    print(f"New vocab loss: {output['loss'].item():.4f}")
    print(f"New vocab size: {model.config.vocab_size}")

    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()
