import torch
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
import weightwatcher as ww

config = LuxiaModelConfig(
    attn_res=True, attn_res_boundaries=[0,1,3,7,15,19,24], use_liger=True
)
model = LuxiaBaseModel(config).cuda().to(torch.bfloat16)
convert_to_float8_training(model, config=Float8LinearConfig())

# Patch weight=None modules
for mod in model.modules():
    if hasattr(mod, "weight") and mod.weight is None:
        delattr(mod, "weight")

watcher = ww.WeightWatcher(model=model)
details = watcher.analyze(min_evals=10, plot=False)

print("WW 'name' column samples:")
for idx, row in details.head(20).iterrows():
    print(f"  [{idx}] name='{row.get('name', '')}' alpha={row['alpha']:.4f}")
