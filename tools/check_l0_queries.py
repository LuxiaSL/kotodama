import torch, subprocess, tempfile, os

def load_zst(path):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    subprocess.run(["zstd", "-d", path, "-o", tmp, "--force"], check=True, capture_output=True)
    ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
    os.unlink(tmp)
    return ckpt

s800 = load_zst("/models/kotodama-data/checkpoints/nca-3b-phase3/step_00000800.pt.zst")["model"]

print("=== L0 queries (pre-attn vs pre-MLP) ===")
print(f"  attn_res_query norm: {s800['layers.0.attn_res_query'].float().norm().item():.6f}")
print(f"  mlp_res_query  norm: {s800['layers.0.mlp_res_query'].float().norm().item():.6f}")
print()
print("=== L1 queries (both should learn - 2+ sources) ===")
print(f"  attn_res_query norm: {s800['layers.1.attn_res_query'].float().norm().item():.6f}")
print(f"  mlp_res_query  norm: {s800['layers.1.mlp_res_query'].float().norm().item():.6f}")
print()
print("=== All boundary pre-attn queries ===")
for b in [0, 1, 3, 7, 15, 19, 24]:
    aq = s800[f'layers.{b}.attn_res_query'].float().norm().item()
    mq = s800[f'layers.{b}.mlp_res_query'].float().norm().item()
    print(f"  L{b:>2}: attn_q={aq:.4f}  mlp_q={mq:.4f}")
