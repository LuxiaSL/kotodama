import torch, weightwatcher as ww
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

config = LuxiaModelConfig(attn_res=True, attn_res_boundaries=[0,1,3,7,15,19,24], use_liger=True)
model = LuxiaBaseModel(config).cuda().to(torch.bfloat16)
convert_to_float8_training(model, config=Float8LinearConfig())
for mod in model.modules():
    if hasattr(mod, "weight") and mod.weight is None:
        delattr(mod, "weight")

watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(min_evals=10, plot=False)

print("Columns:", details.columns.tolist())
print(f"\nTotal layers analyzed: {len(details)}")
print("\nFirst 14 rows (layers 0-1):")
for idx, row in details.head(14).iterrows():
    print(f"  layer_id={row.get('layer_id', '?'):>4}  name={row['name']:<12}  D={row.get('D', '?')}")

# 7 weight types per transformer layer + embedding = 7*28 + 1 = 197
# WW layer_id maps sequentially; we need layer_id -> transformer layer index
print(f"\nExpected: 1 embed + 28*7=196 linear weights = 197 total")
print(f"Embed is layer_id 0, then layers go 1..196")
print(f"Transformer layer i has 7 weights at layer_ids {7*0+1}..{7*0+7} (for layer 0)")
