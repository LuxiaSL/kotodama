"""The sign spool: records of generated rows waiting to be signed, and the signatures written for them.

A server started with ``--sign_spool DIR`` appends a ``SignRecord`` (kotodama.model.signer) for every candidate of a
request that sets ``sign``; with a reducer configured it also signs the spool itself whenever it has nothing else to
do, one batch at a time. The same drain runs as a job of its own (this module's command line), for a spool whose
server never idles or was started without a reducer. Layout:

    <spool>/records/<writer>.jsonl   one record per line, appended by each writer (a server process)
    <spool>/plans/<digest>.npz       the lever plans records name (kotodama.model.signer.save_plans)
    <spool>/signed/<batch>/          per signed batch: kotodama.model.signer.write_results's files and the engine
                                     identity it was signed under (<batch>/engine.json); batches sort in drain order
    <spool>/index.jsonl              one line per signed batch: its name, directory, host, and row counts
    <spool>/state.json               per records file, the byte offset signed (or skipped) up to

Records are small and belong with the run; signed batches are bulk (~19 KB per row on the 1B), so a drain can write
them elsewhere (``signed_dir``, e.g. node-local storage) and the index says where each batch went.

A drain holds an exclusive flock on ``<spool>/drain.lock`` while it signs a batch, so a server and a job can drain one
spool together. A batch is written (into a temporary directory, then renamed) before the state moves past it, so a
crash re-signs at most the batch in flight, under the same request ids.

    tools/run_py.sh -m kotodama.serve.sign --spool <dir> --checkpoint <ckpt> --model_size 1b-tsumugi \\
        --reducer <package.module:attr> --reducer_kwargs '<json>' --layers 0,18,25,... [--pins <table>]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from kotodama.model.levers import LeverPlan
from kotodama.model.signer import RecordSigner, SignRecord, load_plans, save_plans, write_results

logger = logging.getLogger(__name__)


class SignSpool:
    """A spool directory (see the module docstring)."""

    def __init__(self, root: str | Path, signed_dir: Optional[str | Path] = None) -> None:
        self.root = Path(root)
        self.signed_dir = Path(signed_dir) if signed_dir is not None else self.root / "signed"
        for sub in ("records", "plans"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.signed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._plans: dict[str, LeverPlan] = {}

    # ── writing ──
    def add(self, writer: str, records: list[SignRecord], plans: Optional[list[LeverPlan]] = None) -> None:
        """Append ``records`` to ``writer``'s records file, after saving any plan the spool does not hold yet."""
        for plan in plans or ():
            path = self.root / "plans" / f"{plan.digest()}.npz"
            if not path.exists():
                tmp = path.with_name(f".{path.stem}.{os.getpid()}.npz")
                save_plans(tmp, [plan])
                os.replace(tmp, path)
        lines = "".join(json.dumps(r.to_json(), separators=(",", ":")) + "\n" for r in records)
        with self._lock, open(self.root / "records" / f"{writer}.jsonl", "a", encoding="utf-8") as f:
            f.write(lines)
            f.flush()

    # ── draining ──
    def _state(self) -> dict[str, int]:
        path = self.root / "state.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _save_state(self, state: dict[str, int]) -> None:
        tmp = self.root / f".state.{os.getpid()}.json"
        tmp.write_text(json.dumps(state, sort_keys=True))
        os.replace(tmp, self.root / "state.json")

    def has_pending(self) -> bool:
        """Whether any records file has bytes past its signed offset (a stat per file)."""
        state = self._state()
        return any(path.stat().st_size > state.get(path.name, 0)
                   for path in (self.root / "records").glob("*.jsonl"))

    def pending(self) -> int:
        """Complete record lines past each file's signed offset."""
        state = self._state()
        n = 0
        for path in sorted((self.root / "records").glob("*.jsonl")):
            with open(path, "rb") as f:
                f.seek(state.get(path.name, 0))
                n += sum(1 for line in f if line.endswith(b"\n"))
        return n

    def plans(self) -> dict[str, LeverPlan]:
        """Every plan in the spool, by digest (loaded once each)."""
        for path in (self.root / "plans").glob("*.npz"):
            if path.stem not in self._plans:
                self._plans.update(load_plans(path))
        return dict(self._plans)

    def _take(self, max_rows: int) -> tuple[list[SignRecord], dict[str, int]]:
        """Up to ``max_rows`` unsigned records, oldest file first, and the state after them."""
        state = self._state()
        out: list[SignRecord] = []
        for path in sorted((self.root / "records").glob("*.jsonl")):
            done = state.get(path.name, 0)
            with open(path, "rb") as f:
                f.seek(done)
                for line in f:
                    if len(out) >= max_rows or not line.endswith(b"\n"):  # a line still being written waits
                        break
                    try:
                        out.append(SignRecord.from_json(json.loads(line)))
                    except (ValueError, TypeError, UnicodeDecodeError) as exc:
                        logger.warning("sign spool: %s at byte %d is not a record (%s); skipped", path.name, done,
                                       exc)
                    done += len(line)
            state[path.name] = done
            if len(out) >= max_rows:
                break
        return out, state

    @contextmanager
    def _drain_lock(self) -> Iterator[bool]:
        with open(self.root / "drain.lock", "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def drain_once(self, signer: RecordSigner, max_rows: Optional[int] = None) -> Optional[dict[str, Any]]:
        """Sign one batch of pending records (default: the signer's width) and write it. Returns the batch's summary,
        or None when nothing was pending or another drain holds the spool."""
        with self._drain_lock() as mine:
            if not mine:
                return None
            records, state = self._take(max_rows or signer.dec.rows)
            if not records and state == self._state():
                return None
            signer.plans.update(self.plans())
            t0 = time.perf_counter()
            results = signer.sign(records) if records else []
            name = f"{time.time_ns():020d}-{os.getpid()}"  # sorts in drain order
            tmp = self.signed_dir / f".{name}"
            summary = write_results(tmp, results)
            (tmp / "engine.json").write_text(json.dumps(signer.identity, sort_keys=True, default=str))
            final = self.signed_dir / name
            os.replace(tmp, final)
            with open(self.root / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": name, "dir": str(final.resolve()), "host": socket.gethostname(),
                                    "signed": summary["signed"], "skipped": summary["skipped"]}) + "\n")
            self._save_state(state)
            summary.update(batch=name, seconds=round(time.perf_counter() - t0, 2),
                           signer_timing={k: round(v, 3) for k, v in signer.timing.items()})
            logger.info("sign spool: %s", summary)
            return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Drain a sign spool: sign its records through a pinned signing decoder "
                                             "and a SignatureReducer")
    ap.add_argument("--spool", required=True)
    ap.add_argument("--signed_dir", default=None,
                    help="where signed batches go (default <spool>/signed; the spool's index.jsonl lists each)")
    ap.add_argument("--audit_every", type=int, default=100,
                    help="keep the raw signed surfaces of 1 in N rows for an offline re-reduction check (0: none)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--reducer", required=True, help="package.module:attr (a SignatureReducer class or factory)")
    ap.add_argument("--reducer_kwargs", default="{}", help="JSON keyword arguments for the reducer")
    ap.add_argument("--layers", required=True, help="the sampled effective layers, comma-separated")
    ap.add_argument("--tail", type=int, default=256, help="sign positions per row (longer spans are skipped)")
    ap.add_argument("--width", type=int, default=0, help="head-mean row columns (0: max_seq_len)")
    ap.add_argument("--rows", type=int, default=128, help="the signing decoder's width (the lane's canonical 128)")
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--group_rows", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--levers", type=int, default=4, help="lever slots (records' levers bind them per batch)")
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--pins", default=None, help="kernel pin table: replayed when it exists, else recorded")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--once", action="store_true", help="drain what is pending and exit (default: keep polling)")
    ap.add_argument("--poll_s", type=float, default=30.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch

    from kotodama.model import kernel_pins
    from kotodama.model.decode_engine import DecodeEngine
    from kotodama.model.signing import load_reducer
    from kotodama.serve import server

    pins = None
    if a.pins:
        pins = kernel_pins.install(a.pins, "replay" if Path(a.pins).exists() else "record")
    if a.rope_fp32:
        server._rope_fp32_override = True
    model, _ = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="reference",
                                 max_seq_len=a.max_seq_len)
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
    eng.compile_step(mode=a.compile_mode, regional=True)
    reducer = load_reducer(a.reducer, **json.loads(a.reducer_kwargs))
    signer = RecordSigner.build(eng, reducer, layers=[int(x) for x in a.layers.split(",")], tail=a.tail,
                                width=a.width or a.max_seq_len, rows=a.rows, groups=a.groups,
                                group_rows=a.group_rows, compile_mode=a.compile_mode, pins=pins,
                                audit_every=a.audit_every)
    if pins is not None and pins.mode == "record":
        signer.warm()
        logger.info("kernel pins recorded: %s (%d kernels, %s)", pins.save(), len(pins.table), pins.digest[:12])
    spool = SignSpool(a.spool, signed_dir=a.signed_dir)
    with torch.inference_mode():
        while True:
            summary = spool.drain_once(signer)
            if summary is not None:
                print(json.dumps(summary), flush=True)
                continue
            if a.once:
                return 0
            print(f"sign spool: {spool.pending()} pending; polling every {a.poll_s:.0f}s", flush=True)
            time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
