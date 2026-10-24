"""Whole-model gradient parity for --attnres_stack_grad_chain: every parameter's gradient, chained vs unchained, against
an fp32 reference of the same model (fp32 weights, torch fallback routing).

A small KotodamaModel with the full AttnRes structure the training model has — several blocks of uneven length, a
looped stretch, routes that continue a stack's chain across the next block's boundary — is run eagerly (bf16 weights,
as in training) with the chain off and on, and once in fp32 (chain off) as the reference. Per parameter: relative
error of each bf16 gradient vs the fp32 one. The chain changes only the stack-gradient sum's order and rounding, so
its error should be no larger than the unchained one anywhere; a parameter whose chained error is far above its
unchained error points at a wiring bug (a missing or doubled route contribution).

Gate (exit 1 on failure): for every parameter, chain_err <= 1.5 x unchained_err + 1e-3; plus the loss values agree.

  tools/run_py.sh scripts/utils/stack_grad_chain_model_parity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model import llama  # noqa: E402
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402

DEVICE = "cuda"


def grads(model: KotodamaModel, ids: torch.Tensor, chain: bool, triton: bool = True, runner=None
          ) -> tuple[float, dict[str, torch.Tensor]]:
    """runner: what to call (the model itself, or a compiled / DDP-wrapped view of it)."""
    llama._STACK_GRAD_CHAIN = chain
    llama._TRITON_ATTN_RES_AVAILABLE = triton
    model.zero_grad(set_to_none=True)
    logits = (runner if runner is not None else model)(ids)["logits"]
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1))
    loss.backward()
    return loss.item(), {n: p.grad.detach().double().clone() for n, p in model.named_parameters() if p.grad is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--boundaries", default="0,1,4,6,11,13,18,21")
    ap.add_argument("--mb", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--compiled_ddp", action="store_true",
                    help="run the bf16 arms as training does: DDP (one rank, tiny buckets so the DDP optimizer splits "
                         "the graph often) + torch.compile; the fp32 reference stays eager")
    a = ap.parse_args()
    cfg = KotodamaModelConfig(
        hidden_size=a.hidden, num_layers=a.layers, num_attention_heads=a.hidden // 128, num_kv_heads=max(1, a.hidden // 256),
        head_dim=128, intermediate_size=4 * a.hidden, vocab_size=8192, max_position_embeddings=a.seq, qk_norm=True,
        attn_impl="sdpa",  # one attention path for all three models (FA2 has no fp32)
        attn_res=True, attn_res_boundaries=[int(x) for x in a.boundaries.split(",")],
        loop_start=6, loop_len=6, loop_times=2,
    )
    torch.manual_seed(0)
    ref_model = KotodamaModel(cfg).to(DEVICE).float()
    bf_model = KotodamaModel(cfg).to(DEVICE)
    bf_model.load_state_dict(ref_model.state_dict())
    bf_model = bf_model.bfloat16()
    ids = torch.randint(0, cfg.vocab_size, (a.mb, a.seq), device=DEVICE, generator=torch.Generator(DEVICE).manual_seed(1))

    if not llama._TRITON_ATTN_RES_AVAILABLE:
        print(f"Triton AttnRes ops unavailable: {llama._TRITON_ATTN_RES_IMPORT_ERROR}", flush=True)
        return 1
    # Reference: fp32 weights through the torch fallback routing (the Triton forward always emits bf16).
    loss_ref, g_ref = grads(ref_model, ids, chain=False, triton=False)
    if a.compiled_ddp:
        import os

        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("nccl", rank=0, world_size=1)
        # One DDP wrapper (two would double-register gradient hooks). The chain flag is a module global read while
        # dynamo traces; dynamo guards on it, so the second arm retraces on its own.
        runner = torch.compile(DDP(bf_model, bucket_cap_mb=1, gradient_as_bucket_view=True))
        loss_un, g_un = grads(bf_model, ids, chain=False, runner=runner)
        loss_ch, g_ch = grads(bf_model, ids, chain=True, runner=runner)
        print(f"compiled + DDP (1 rank, bucket_cap_mb=1)", flush=True)
    else:
        loss_un, g_un = grads(bf_model, ids, chain=False)
        loss_ch, g_ch = grads(bf_model, ids, chain=True)
    print(f"loss fp32 {loss_ref:.5f}  bf16 unchained {loss_un:.5f}  bf16 chain {loss_ch:.5f}", flush=True)
    ok = abs(loss_ch - loss_un) < 1e-3
    worst = []
    for name, r in g_ref.items():
        if name not in g_un or name not in g_ch:
            print(f"MISSING grad {name}", flush=True)
            ok = False
            continue
        denom = r.norm().clamp_min(1e-30)
        e_un = ((g_un[name] - r).norm() / denom).item()
        e_ch = ((g_ch[name] - r).norm() / denom).item()
        bad = e_ch > 1.5 * e_un + 1e-3
        ok &= not bad
        worst.append((e_ch - e_un, name, e_un, e_ch, bad))
    worst.sort(reverse=True)
    print(f"{len(worst)} parameters; largest chain-minus-unchained error first:", flush=True)
    for d, name, e_un, e_ch, bad in worst[:15]:
        print(f"  {'BAD ' if bad else '    '}{name:55s} unchained {e_un:.3e}  chain {e_ch:.3e}", flush=True)
    better = sum(1 for d, *_ in worst if d <= 0)
    print(f"chain error <= unchained for {better}/{len(worst)} parameters", flush=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
