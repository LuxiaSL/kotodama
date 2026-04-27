"""Parse a Chrome trace JSON from torch.profiler and print kernel summary."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/profile-compile/profile/chrome_trace.json"

print(f"Loading {path}...")
with open(path) as f:
    data = json.load(f)

events = data.get("traceEvents", data) if isinstance(data, dict) else data

kernel_times = defaultdict(lambda: {"count": 0, "total_us": 0})
cpu_times = defaultdict(lambda: {"count": 0, "total_us": 0})

for e in events:
    dur = e.get("dur", 0)
    if not dur:
        continue
    cat = e.get("cat", "")
    name = e.get("name", "unknown")
    if cat == "kernel":
        kernel_times[name]["count"] += 1
        kernel_times[name]["total_us"] += dur
    elif cat in ("cpu_op", "user_annotation"):
        cpu_times[name]["count"] += 1
        cpu_times[name]["total_us"] += dur

total_cuda = sum(v["total_us"] for v in kernel_times.values())
total_cpu = sum(v["total_us"] for v in cpu_times.values())

print(f"\nTotal CUDA kernel time: {total_cuda/1e6:.1f}s")
print(f"Unique CUDA kernels: {len(kernel_times)}")

sorted_kernels = sorted(kernel_times.items(), key=lambda x: -x[1]["total_us"])

print(f"\n{'Kernel':<90s} {'Total(ms)':>10s} {'Count':>7s} {'Avg(us)':>9s} {'%':>6s}")
print("-" * 125)
cumulative = 0
for name, stats in sorted_kernels[:30]:
    total_ms = stats["total_us"] / 1000
    avg_us = stats["total_us"] / max(stats["count"], 1)
    pct = stats["total_us"] / max(total_cuda, 1) * 100
    cumulative += pct
    short = name[:90]
    print(f"{short:<90s} {total_ms:>10.1f} {stats['count']:>7d} {avg_us:>9.1f} {pct:>5.1f}%")

print(f"\nTop 30 = {cumulative:.1f}% of CUDA time")

# Also show top CPU ops
sorted_cpu = sorted(cpu_times.items(), key=lambda x: -x[1]["total_us"])
print(f"\n{'Top CPU ops':<90s} {'Total(ms)':>10s} {'Count':>7s}")
print("-" * 110)
for name, stats in sorted_cpu[:15]:
    total_ms = stats["total_us"] / 1000
    short = name[:90]
    print(f"{short:<90s} {total_ms:>10.1f} {stats['count']:>7d}")
