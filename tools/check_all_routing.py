"""Extract all 57 routing query norms from the Phase 3 checkpoint.

Each layer has attn_res_query (pre-attention) and mlp_res_query (pre-MLP).
Larger norm = the query has learned more = routing is more active at that point.
The final_res_query controls the output aggregation.
"""
import torch, subprocess, tempfile, os

def load_zst(path):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    subprocess.run(["zstd", "-d", path, "-o", tmp, "--force"], check=True, capture_output=True)
    ckpt = torch.load(tmp, map_location="cpu", weights_only=False)
    os.unlink(tmp)
    return ckpt

state = load_zst("/models/kotodama-data/nca-3b-phase3-BASE.pt.zst")["model"]
boundaries = {0, 1, 3, 7, 15, 19, 24}

print("=== Per-Layer Routing Query Norms (step 3000) ===")
print(f"{'Layer':>6} {'Type':>8} {'Norm':>8} {'Block':>6} {'Note'}")
print("-" * 50)

for i in range(28):
    block_label = ""
    if i in boundaries:
        block_idx = sorted(boundaries).index(i)
        block_label = f"B{block_idx}←"

    for qtype, label in [("attn_res_query", "pre-attn"), ("mlp_res_query", "pre-mlp")]:
        key = f"layers.{i}.{qtype}"
        if key in state:
            norm = state[key].float().norm().item()
            note = ""
            if norm == 0:
                note = "DEAD (structural)"
            elif norm > 0.8:
                note = "** HIGH **"
            elif norm > 0.4:
                note = "active"
            elif norm > 0.1:
                note = "moderate"
            else:
                note = "weak"
            print(f"  L{i:>2}  {label:>8}  {norm:>8.4f} {block_label:>6} {note}")

# Final aggregation
fn = state["final_res_query"].float().norm().item()
print(f"\n  Final agg      {fn:>8.4f}        {'active' if fn > 0.1 else 'weak'}")

# Summary: which layers have the most active routing?
print("\n=== Top 10 Most Active Routing Points ===")
all_queries = []
for i in range(28):
    for qtype, label in [("attn_res_query", f"L{i} pre-attn"), ("mlp_res_query", f"L{i} pre-mlp")]:
        key = f"layers.{i}.{qtype}"
        if key in state:
            norm = state[key].float().norm().item()
            all_queries.append((norm, label, i))

all_queries.sort(reverse=True)
for norm, label, layer_idx in all_queries[:10]:
    in_block = None
    for b in sorted(boundaries):
        if layer_idx >= b:
            in_block = sorted(boundaries).index(b)
    print(f"  {norm:.4f}  {label:<20} (block {in_block})")

# Per-block average query activity
print("\n=== Average Query Norm Per Block ===")
block_bounds = sorted(boundaries)
for bi in range(len(block_bounds)):
    start = block_bounds[bi]
    end = block_bounds[bi + 1] if bi + 1 < len(block_bounds) else 28
    norms = []
    for i in range(start, end):
        for qtype in ["attn_res_query", "mlp_res_query"]:
            key = f"layers.{i}.{qtype}"
            if key in state:
                norms.append(state[key].float().norm().item())
    avg = sum(norms) / len(norms) if norms else 0
    print(f"  Block {bi} (L{start}-L{end-1}, {end-start} layers): avg_norm={avg:.4f}")
