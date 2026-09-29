#!/usr/bin/env python3
"""Check that training nodes run the SAME stack (so runs/resumes can hop nodes).

Compares, per host (over ssh): the Python stack (torch, triton, flash_attn, cutlass-dsl,
torchao, liger), a sha256 over the installed flash_attn/cute sources (the FA4 overlay),
a sha256 over this repo's tracked files as mirrored on the host, and driver/CUDA/GPU.
Exit 1 on any mismatch (or unreachable host), 0 when every host agrees.

Usage:
  python scripts/utils/node_parity.py --hosts HOST1 HOST2 --mirror /path/on/hosts/kotodama \
      --python /path/on/hosts/venv/bin/python
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

REMOTE = r'''
import hashlib, json, os, subprocess, sys
from importlib import metadata as md
out = {"py": sys.version.split()[0]}
for name in ["torch", "triton", "flash_attn", "nvidia-cutlass-dsl", "torchao", "liger-kernel"]:
    try: out[name] = md.version(name)
    except Exception: out[name] = None
try:
    import flash_attn.cute as fc
    d = os.path.dirname(fc.__file__); h = hashlib.sha256()
    for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
            h.update(f.encode()); h.update(open(os.path.join(d, f), "rb").read())
    out["fa4_cute_sha"] = h.hexdigest()[:16]
except Exception as e:
    out["fa4_cute_sha"] = f"ERR {type(e).__name__}"
files = json.loads(FILES_JSON); mirror = sys.argv[1]; h = hashlib.sha256(); missing = 0
for f in files:
    p = os.path.join(mirror, f)
    if os.path.exists(p):
        h.update(f.encode()); h.update(open(p, "rb").read())
    else:
        missing += 1
out["mirror_sha"] = h.hexdigest()[:16]; out["mirror_missing"] = missing
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    out["gpus"] = f"{len(q)}x {q[0].split(',')[0].strip()}" if q else None
    out["driver"] = q[0].split(",")[1].strip() if q else None
except Exception:
    out["gpus"] = out["driver"] = None
try:
    import torch; out["cuda"] = torch.version.cuda
except Exception:
    out["cuda"] = None
print(json.dumps(out))
'''


def local_mirror_sha(files: list[str], root: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        p = root / f
        if p.exists():
            h.update(f.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="+", required=True)
    ap.add_argument("--mirror", required=True, help="repo mirror path on the hosts")
    ap.add_argument("--python", required=True, help="venv python on the hosts")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True,
                           check=True).stdout.split()
    files = [f for f in files if not f.endswith((".md", ".json"))]  # code + configs only
    want_mirror = local_mirror_sha(files, root)
    rows: dict[str, dict] = {}
    ok = True
    for host in a.hosts:
        script = f"FILES_JSON = {json.dumps(json.dumps(files))}\n" + REMOTE
        cmd = ["ssh", host, f"{shlex.quote(a.python)} - {shlex.quote(a.mirror)}"]
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=300)
        try:
            rows[host] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            print(f"FAIL {host}: {r.stderr.strip()[-300:]}")
            ok = False
    if not rows:
        return 1
    keys = sorted({k for r in rows.values() for k in r})
    print(f"{'':18s}" + "".join(f"{h:>26s}" for h in rows) + f"{'local git':>20s}")
    for k in keys:
        vals = [rows[h].get(k) for h in rows]
        same = len({json.dumps(v) for v in vals}) == 1
        extra = want_mirror if k == "mirror_sha" else ""
        if k == "mirror_sha" and any(v != want_mirror for v in vals):
            same = False
        ok &= same or k in ("mirror_missing",)
        print(f"{k:18s}" + "".join(f"{str(v):>26s}" for v in vals) + f"{extra:>20s}"
              + ("" if same else "   <-- MISMATCH"))
    print("PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
