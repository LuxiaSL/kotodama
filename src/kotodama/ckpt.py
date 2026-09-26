"""Checkpoint files: load, inspect, strip, and resolve by name.

A kotodama checkpoint is a ``torch.save`` dict: ``{"model": state_dict, "step",
"tokens_consumed"}`` plus ``"optimizer"``/``"scheduler"`` for resumable training
checkpoints. Files may be zstd-compressed (``.pt.zst``). The pickles contain
only torch/collections objects, so they load independent of package layout.

Names: ``resolve("kotodama-3b-base-final")`` looks the name up in the registry
YAMLs — ``configs/checkpoints.yaml`` in the repo, plus every file listed in
``$KOTODAMA_CKPT_REGISTRY`` (colon-separated; a site's private models). Relative
paths resolve against ``$KOTODAMA_DATA_ROOT``.

CLI:
  python -m kotodama.ckpt info  <name-or-path>
  python -m kotodama.ckpt strip <name-or-path> <out.pt[.zst]> [--bf16]
  python -m kotodama.ckpt list
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)

_REPO_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "checkpoints.yaml"


# ── registry ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path
    size: str = "3b"
    boundaries: str | None = "dd3b"
    hf: str | None = None
    note: str = ""


def data_root() -> Path:
    return Path(os.environ.get("KOTODAMA_DATA_ROOT", Path.cwd() / "data"))


def registry_files() -> list[Path]:
    files = [_REPO_REGISTRY]
    extra = os.environ.get("KOTODAMA_CKPT_REGISTRY", "")
    files += [Path(p) for p in extra.split(":") if p]
    return files


def registry() -> dict[str, Entry]:
    out: dict[str, Entry] = {}
    for f in registry_files():
        if not f.exists():
            if f != _REPO_REGISTRY:
                logger.warning("checkpoint registry %s not found", f)
            continue
        raw = yaml.safe_load(f.read_text()) or {}
        for name, d in raw.items():
            p = Path(d["path"])
            out[name] = Entry(
                name=name,
                path=p if p.is_absolute() else data_root() / p,
                size=d.get("size", "3b"),
                boundaries=d.get("boundaries", "dd3b"),
                hf=d.get("hf"),
                note=d.get("note", ""),
            )
    return out


def resolve(name_or_path: str | Path) -> Path:
    """A registry name or a filesystem path -> an existing checkpoint path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    reg = registry()
    if str(name_or_path) in reg:
        path = reg[str(name_or_path)].path
        if not path.exists():
            raise FileNotFoundError(f"{name_or_path}: registry path {path} does not exist")
        return path
    raise FileNotFoundError(
        f"{name_or_path!r} is neither a file nor a registry name "
        f"(registries: {', '.join(map(str, registry_files()))})"
    )


# ── loading ───────────────────────────────────────────────────────────────────


def load(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint dict. ``.zst`` streams through a temp file so the
    compressed bytes, decompressed bytes, and tensors are never all in RAM."""
    path = resolve(path)
    if path.suffix != ".zst":
        return torch.load(path, map_location="cpu", weights_only=False)

    import zstandard as zstd

    tmp_dir = os.environ.get("TMPDIR") or tempfile.gettempdir()
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, dir=tmp_dir) as temp_file:
            temp_path = Path(temp_file.name)
            with path.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as reader:
                shutil.copyfileobj(reader, temp_file, length=16 * 1024 * 1024)
        return torch.load(temp_path, map_location="cpu", weights_only=False)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def decompress_cached(path: str | Path) -> Path:
    """Decompress a ``.zst`` checkpoint ONCE to a shared temp location and return
    the plain path (parallel processes on one checkpoint reuse it). Plain paths
    are returned unchanged."""
    path = resolve(path)
    if path.suffix != ".zst":
        return path
    beside = path.with_suffix("")
    if beside.exists():
        return beside
    tmp_dir = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / "kotodama_checkpoints"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / beside.name
    if out.exists():
        return out
    # Process-unique partial + atomic rename: racing shards never read a half file.
    partial = out.with_name(f"{out.name}.partial.{os.getpid()}")
    logger.info("Decompressing %s -> %s", path.name, out)
    try:
        subprocess.run(["zstd", "-d", str(path), "-o", str(partial), "-f"],
                       check=True, capture_output=True)
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)
    return out


def model_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    """The model state dict with torch.compile / DDP prefixes removed."""
    state = ckpt.get("model", ckpt)
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}


# ── inspection / conversion ───────────────────────────────────────────────────


def info(path: str | Path) -> dict[str, Any]:
    path = resolve(path)
    ckpt = load(path)
    state = model_state(ckpt)
    tensors = [v for v in state.values() if isinstance(v, torch.Tensor)]
    return {
        "path": str(path),
        "file_gb": round(path.stat().st_size / 2**30, 2),
        "keys": sorted(k for k in ckpt if k != "model") if "model" in ckpt else [],
        "step": ckpt.get("step"),
        "tokens_consumed": ckpt.get("tokens_consumed"),
        "params_b": round(sum(t.numel() for t in tensors) / 1e9, 4),
        "dtypes": sorted({str(t.dtype) for t in tensors}),
        "attn_res": any("attn_res" in k for k in state),
        "has_optimizer": "optimizer" in ckpt,
    }


def strip(src: str | Path, dst: str | Path, *, bf16: bool = False) -> Path:
    """Write a model-only checkpoint (drops optimizer/scheduler). ``dst`` ending
    in ``.zst`` is compressed with zstd and the intermediate removed."""
    src, dst = resolve(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt = load(src)
    state = model_state(ckpt)
    if bf16:
        state = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
                 for k, v in state.items()}
    out = {"model": state, "step": ckpt.get("step"), "tokens_consumed": ckpt.get("tokens_consumed")}
    compress = dst.suffix == ".zst"
    pt_path = dst.with_suffix("") if compress else dst
    tmp = pt_path.with_name(pt_path.name + ".tmp")
    torch.save(out, tmp)
    tmp.rename(pt_path)
    if compress:
        if shutil.which("zstd") is None:
            raise RuntimeError("zstd not on PATH")
        subprocess.run(["zstd", "-T0", "-q", "-f", str(pt_path), "-o", str(dst), "--rm"], check=True)
    return dst


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kotodama.ckpt", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_info = sub.add_parser("info", help="step, params, dtypes, optimizer presence")
    p_info.add_argument("ckpt")
    p_strip = sub.add_parser("strip", help="model-only copy (optionally bf16, zstd)")
    p_strip.add_argument("ckpt")
    p_strip.add_argument("out")
    p_strip.add_argument("--bf16", action="store_true")
    sub.add_parser("list", help="registry names")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        for k, v in info(args.ckpt).items():
            print(f"{k:16s} {v}")
    elif args.cmd == "strip":
        print(strip(args.ckpt, args.out, bf16=args.bf16))
    elif args.cmd == "list":
        for name, e in sorted(registry().items()):
            mark = "" if e.path.exists() else "  (missing here)"
            print(f"{name:40s} {e.size:6s} {e.path}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
