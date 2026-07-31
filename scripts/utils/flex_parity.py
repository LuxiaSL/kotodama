"""FlexAttention vs FA2-varlen parity smoke (C1 lever, SPEC-7B §5.1).

Two copies of a small AttnRes model with identical weights process the same
doc-masked batch — one through the FA2 varlen path, one through FlexAttention
+ BlockMask. Same math up to fp reassociation: loss and gradients must agree
within bf16 attention tolerance. Runs eager (no compile) — the compile-path
check is the canary arm's job.

Run on gpu-host: tools/run_py.sh scripts/utils/flex_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig  # noqa: E402
from src.training.train import _FlexBlockMaskBuilder  # noqa: E402

DEVICE = "cuda"
B, T = 2, 512


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def make_batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 1024, (B, T), generator=g)
    # 3 docs in row 0, 2 in row 1 — uneven boundaries
    doc_lens = [[100, 300, 112], [400, 112]]
    position_ids = torch.zeros(B, T, dtype=torch.long)
    cu = [0]
    for b, lens in enumerate(doc_lens):
        off = 0
        for ln in lens:
            position_ids[b, off:off + ln] = torch.arange(ln)
            cu.append(cu[-1] + ln)
            off += ln
    cu_seqlens = torch.tensor(cu, dtype=torch.int32)
    max_seqlen = max(max(lens) for lens in doc_lens)
    return (input_ids.to(DEVICE), position_ids.to(DEVICE),
            cu_seqlens.to(DEVICE), max_seqlen)


def build_model(attn_impl: str, state=None) -> LuxiaBaseModel:
    torch.manual_seed(17)
    cfg = LuxiaModelConfig(
        hidden_size=256, num_layers=4, num_attention_heads=4, num_kv_heads=2,
        head_dim=64, intermediate_size=512, vocab_size=1024,
        max_position_embeddings=2048, attn_res=True, attn_res_n_blocks=2,
        attn_impl=attn_impl, use_liger=False, activation_checkpointing=False,
    )
    m = LuxiaBaseModel(cfg).to(DEVICE)
    if state is not None:
        m.load_state_dict(state)
    return m


def main() -> int:
    torch.cuda.init()
    all_ok = True
    input_ids, position_ids, cu_seqlens, max_seqlen = make_batch()

    ref = build_model("fa2")
    flex = build_model("flex", state=ref.state_dict())

    builder = _FlexBlockMaskBuilder(torch.device(DEVICE))
    block_mask = builder.build(position_ids)

    results = {}
    for name, model, kwargs in [
        ("fa2", ref, dict(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)),
        ("flex", flex, dict(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                            block_mask=block_mask)),
    ]:
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids, labels=input_ids,
                        position_ids=position_ids, **kwargs)
        out["loss"].backward()
        results[name] = out["loss"].item()

    dl = abs(results["flex"] - results["fa2"])
    all_ok &= check("loss fa2 vs flex", dl < 5e-3,
                    f"{results['fa2']:.6f} vs {results['flex']:.6f} (|d|={dl:.2e})")

    worst = ("", 0.0)
    for (n1, p1), (_n2, p2) in zip(
        ref.named_parameters(), flex.named_parameters()
    ):
        if p1.grad is None or p2.grad is None:
            continue
        e = rel_err(p2.grad, p1.grad)
        if e > worst[1]:
            worst = (n1, e)
    all_ok &= check("grads fa2 vs flex", worst[1] < 5e-2,
                    f"worst {worst[0]} relnorm {worst[1]:.2e}")

    # Mask-correctness probe: flex must NOT attend across doc boundaries.
    # Perturb tokens of row-0 doc-1 and confirm doc-2's logits are unchanged.
    flex.eval()
    with torch.no_grad():
        base = flex(input_ids, position_ids=position_ids,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    block_mask=block_mask)["logits"]
        poked = input_ids.clone()
        poked[0, 100:400] = (poked[0, 100:400] + 7) % 1024
        pert = flex(poked, position_ids=position_ids,
                    cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    block_mask=block_mask)["logits"]
    leak = (base[0, 400:] - pert[0, 400:]).abs().max().item()
    all_ok &= check("no cross-doc leakage", leak == 0.0, f"max |dlogit|={leak:.2e}")
    changed = (base[0, 100:400] - pert[0, 100:400]).abs().max().item()
    all_ok &= check("perturbed doc DID change", changed > 0.0,
                    f"max |dlogit|={changed:.2e}")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
