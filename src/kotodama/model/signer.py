"""Signing finished generations by replaying them through the decode engine: the sign path a server runs in-process
after a batch, and a signing job runs over stored records.

A generated row is described by a small ``SignRecord``: its full prompt ids, its generated ids, the lever it decoded
under (a LeverPlan digest; the plans travel beside the records, ``save_plans`` / ``load_plans``), the weights digest of
the model that generated it, and the serving engine's identity. ``RecordSigner`` re-admits each record's prompt into a
signing ``CascadeDecoder`` (kotodama.model.multi_decode) at its canonical width, teacher-forces the generated ids
through the decode step (``score``: a scored span is signed exactly as a generated one) and hands the signed rows to a
``SignatureReducer`` (kotodama.model.signing). Rows of a batch are independent: a record signs to the same bytes alone
or beside any batchmates, so records are packed by prompt and the unused rows of a batch score filler.

A record is skipped, with the reason logged, when the signer cannot reproduce or fully hold it: other weights, other
broadcast steering or adapter, a lever plan it was not given, fewer than two generated ids (no fed position), a span
longer than the sign tail, or (with attention statistics) a sequence wider than the head-mean rows. A batch the reducer
refuses is retried row by row, and the rows it still refuses are skipped with the reducer's error.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch

from kotodama.model import signing
from kotodama.model.levers import KINDS, LeverPlan

logger = logging.getLogger(__name__)

RECORD_FORMAT = "kotodama-sign-record/1"


@dataclass(frozen=True)
class SignRecord:
    """One generated row, as much as signing it needs. ``prompt_ids``: every token the row was conditioned on (system
    and earlier turns included); ``generated_ids``: the tokens it sampled, in order (a stop token it sampled
    included); ``weights``: ``signing.weights_digest`` of the generating model; ``lever``: the LeverPlan digest it
    decoded under (None: unsteered) and ``dose`` the dose that plan was built at (metadata); ``steering``: the
    broadcast steering bank identity (None: none); ``adapter``: the merged adapter identity (None: none);
    ``engine``: the serving engine's identity (``signing.engine_identity``; None: unknown); ``meta``: anything else
    (seed, law, the caller's row id)."""

    request_id: str
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    weights: str
    lever: Optional[str] = None
    dose: Optional[float] = None
    steering: Optional[str] = None
    adapter: Optional[str] = None
    engine: Optional[Mapping[str, Any]] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("a sign record needs a non-empty request_id")
        for name in ("prompt_ids", "generated_ids"):
            ids = tuple(int(t) for t in getattr(self, name))
            if any(t < 0 for t in ids):
                raise ValueError(f"{self.request_id}: {name} holds a negative token id")
            object.__setattr__(self, name, ids)
        if not self.prompt_ids:
            raise ValueError(f"{self.request_id}: prompt_ids is empty")
        if not isinstance(self.weights, str) or not self.weights:
            raise ValueError(f"{self.request_id}: a sign record names its weights digest")
        if self.dose is not None:
            object.__setattr__(self, "dose", float(self.dose))
        object.__setattr__(self, "meta", dict(self.meta))
        if self.engine is not None:
            object.__setattr__(self, "engine", dict(self.engine))

    def to_json(self) -> dict[str, Any]:
        out = {"format": RECORD_FORMAT}
        out.update({f.name: getattr(self, f.name) for f in dataclasses.fields(self)})
        out["prompt_ids"], out["generated_ids"] = list(self.prompt_ids), list(self.generated_ids)
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "SignRecord":
        fmt = obj.get("format")
        if fmt != RECORD_FORMAT:
            raise ValueError(f"not a {RECORD_FORMAT} record (format {fmt!r})")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(obj) - names - {"format"}
        if unknown:
            raise ValueError(f"unknown sign record fields {sorted(unknown)}")
        return cls(**{k: v for k, v in obj.items() if k in names})


def write_records(path: str | Path, records: Iterable[SignRecord], append: bool = True) -> int:
    """Write records as JSON lines (appending by default). Returns how many were written."""
    n = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> Iterator[SignRecord]:
    """The records of a JSON-lines file, in order; a malformed line raises ValueError naming it."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield SignRecord.from_json(json.loads(line))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{i}: {exc}") from exc


def save_plans(path: str | Path, plans: Iterable[LeverPlan]) -> list[str]:
    """Lever plans as one npz, keyed by digest (``load_plans`` reads them back). Returns the digests."""
    arrays: dict[str, np.ndarray] = {}
    digests = []
    for plan in plans:
        d = plan.digest()
        digests.append(d)
        arrays[f"{d}/span"] = np.array(plan.span)
        for kind in (*KINDS, "act"):
            for layer, row in plan.rows(kind).items():
                arrays[f"{d}/{kind}/{layer}"] = row
    np.savez(path, **arrays)
    return digests


def load_plans(path: str | Path) -> dict[str, LeverPlan]:
    """The plans ``save_plans`` wrote, by digest; a plan whose content no longer hashes to its key raises."""
    rows: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            d, _, rest = key.partition("/")
            entry = rows.setdefault(d, {"span": None, **{k: {} for k in (*KINDS, "act")}})
            if rest == "span":
                entry["span"] = str(z[key])
            else:
                kind, _, layer = rest.partition("/")
                entry[kind][int(layer)] = z[key]
    plans = {}
    for d, entry in rows.items():
        plan = LeverPlan(attn=entry["attn"], mlp=entry["mlp"], act=entry["act"], span=entry["span"])
        if plan.digest() != d:
            raise ValueError(f"{path}: the plan stored under {d} hashes to {plan.digest()}")
        plans[d] = plan
    return plans


@dataclass
class SignResult:
    """One record's outcome: its reduced signature, or why it was skipped. ``same_engine``: the record's serving
    identity equals the signer's (its signature is then the one its generation would have captured)."""

    record: SignRecord
    reduced: Optional[signing.ReducedRow] = None
    skipped: Optional[str] = None
    same_engine: bool = False


def _canonical(x: Any) -> Any:
    """``x`` as it reads back from JSON (tuples become lists), for comparing identities."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def trim(row: signing.SignedRow, n: int) -> signing.SignedRow:
    """``row``'s first ``n`` positions (a row that scored filler past its own span)."""
    if n >= row.positions:
        return row
    kept = min(n, row.kept)
    surf = {name: ([t[:kept] for t in x] if isinstance(x, list) else x[:kept]) for name, x in row.surfaces.items()}
    return dataclasses.replace(row, positions=n, kept=kept, surfaces=surf)


class RecordSigner:
    """Signs records on a signing decoder (a ``CascadeDecoder`` built with ``sign=``, compiled regionally, its
    engine's lever tables holding at least as many slots as a batch has distinct levers) and a reducer.

    ``plans``: the lever plans records may name, by digest; ``steering`` / ``adapter``: the broadcast steering bank
    and merged adapter the signer's engine runs (records must match); ``compile_mode`` / ``pins_digest``: what the
    decoder's regions were compiled under and the kernel pin table they replay (kotodama.model.kernel_pins), for the
    engine identity."""

    def __init__(self, decoder: Any, reducer: signing.SignatureReducer, *, compile_mode: str,
                 pins_digest: Optional[str] = None, plans: Optional[Mapping[str, LeverPlan]] = None,
                 steering: Optional[str] = None, adapter: Optional[str] = None) -> None:
        from kotodama.model.multi_decode import CascadeDecoder

        if not isinstance(decoder, CascadeDecoder) or decoder.sign is None:
            raise TypeError("RecordSigner needs a CascadeDecoder built with sign=")
        if not isinstance(reducer, signing.SignatureReducer):
            raise TypeError("the reducer does not implement SignatureReducer")
        self.dec = decoder
        self.reducer = reducer
        self.spec: signing.SignSpec = decoder.sign
        self.plans = dict(plans or {})
        self.steering, self.adapter = steering, adapter
        self.weights: str = decoder._sign_weights
        self.identity = signing.engine_identity(
            compile_mode=compile_mode, width=decoder.rows, weights=self.weights, pins_digest=pins_digest,
            extra={"decoder": {"class": type(decoder).__name__, "rows": decoder.rows, "groups": decoder.groups,
                               "group_rows": decoder.group_rows, "tail_tokens": decoder.tail_tokens,
                               "max_seq_len": decoder.max_seq_len, "seeded": decoder.seeded},
                   "sign": dataclasses.asdict(self.spec), "steering": steering, "adapter": adapter})

    def why_not(self, rec: SignRecord) -> Optional[str]:
        """Why ``rec`` cannot be signed here (None: it can)."""
        n = len(rec.generated_ids)
        total = len(rec.prompt_ids) + n
        if rec.weights != self.weights:
            return f"weights {rec.weights[:12]}... are not the signer's {self.weights[:12]}..."
        if rec.steering != self.steering:
            return f"steering {rec.steering!r} is not the signer's {self.steering!r}"
        if rec.adapter != self.adapter:
            return f"adapter {rec.adapter!r} is not the signer's {self.adapter!r}"
        if rec.lever is not None and rec.lever not in self.plans:
            return f"lever {rec.lever} has no plan"
        if rec.lever is not None and self.dec.engine.levers is None:
            return "the signer's engine has no lever tables"
        if n < 2:
            return f"{n} generated ids: no fed position to sign"
        if n - 1 > self.spec.tail:
            return f"span {n - 1} is longer than the sign tail {self.spec.tail}"
        if n >= self.dec.tail_tokens:
            return f"{n} scored tokens do not fit the decoder's {self.dec.tail_tokens}-token tails"
        if total >= self.dec.max_seq_len:
            return f"{total} tokens do not fit max_seq_len {self.dec.max_seq_len}"
        if self.spec.attn and total - 1 > self.spec.width:
            return f"{total - 1} attended columns exceed the head-mean rows' width {self.spec.width}"
        return None

    def _batches(self, keyed: list[tuple[tuple, list[int]]],
                 n_scored: Sequence[int]) -> Iterator[list[tuple[tuple, list[int]]]]:
        """Chunks of (key, record indices) per batch: at most ``rows`` rows and, with the filler rows' groups, at
        most ``groups`` groups and ``lever slots`` distinct levers; every row scores the batch's longest span (its
        own, then filler), so the longest prompt plus that span must fit ``max_seq_len``."""
        dec = self.dec
        m, rows, groups = dec.group_rows, dec.rows, dec.groups
        slots = dec.engine.levers.free if dec.engine.levers is not None else 0
        batch: list[tuple[tuple, list[int]]] = []
        used_rows = used_groups = 0
        levers: set[str] = set()
        longest_prompt = longest_span = 0

        def fits(k: int, lever: Optional[str], prompt: int, span: int) -> bool:
            g = used_groups + -(-k // m)
            filler = rows - used_rows - k
            return (used_rows + k <= rows and g + -(-filler // m) <= groups
                    and max(longest_prompt, prompt) + max(longest_span, span) < dec.max_seq_len
                    and (lever is None or lever in levers or len(levers) < slots))

        for key, idx in keyed:
            lever, prompt = key[1], len(key[0])
            while idx:
                k = min(len(idx), rows - used_rows, m * (groups - used_groups))
                while k > 0 and not fits(k, lever, prompt, max(n_scored[i] for i in idx[:k])):
                    k -= 1
                if k == 0:
                    if not batch:
                        raise RuntimeError("a single record does not fit an empty batch (groups/levers too few)")
                    yield batch
                    batch, used_rows, used_groups, levers = [], 0, 0, set()
                    longest_prompt = longest_span = 0
                    continue
                batch.append((key, idx[:k]))
                used_rows += k
                used_groups += -(-k // m)
                longest_prompt = max(longest_prompt, prompt)
                longest_span = max(longest_span, max(n_scored[i] for i in idx[:k]))
                if lever is not None:
                    levers.add(lever)
                idx = idx[k:]
        if batch:
            yield batch

    @torch.inference_mode()
    def sign(self, records: Sequence[SignRecord]) -> list[SignResult]:
        """Sign ``records``; one result per record, in order."""
        from kotodama.model.decode_engine import SamplingParams

        results = [SignResult(record=r) for r in records]
        keyed: dict[tuple, list[int]] = {}
        for i, rec in enumerate(records):
            reason = self.why_not(rec)
            if reason is not None:
                results[i].skipped = reason
                logger.warning("sign: skipped %s: %s", rec.request_id, reason)
                continue
            results[i].same_engine = rec.engine is not None and _canonical(rec.engine) == _canonical(self.identity)
            keyed.setdefault((rec.prompt_ids, rec.lever), []).append(i)
        dec, eng = self.dec, self.dec.engine
        law = SamplingParams(temperature=1.0, repetition_penalty=1.0)
        dev = eng.device
        for batch in self._batches(list(keyed.items()), [len(r.generated_ids) for r in records]):
            bound: dict[str, int] = {}
            row_of: dict[int, int] = {}
            dec.release(range(dec.rows))
            try:
                nxt = 0
                for (prompt, lever), idx in batch:
                    slot = 0
                    if lever is not None:
                        if lever not in bound:
                            bound[lever] = eng.levers.bind(self.plans[lever])
                        slot = bound[lever]
                    rows = list(range(nxt, nxt + len(idx)))
                    nxt += len(idx)
                    logits = eng.prefill(torch.tensor([prompt], device=dev), slot=slot)
                    dec.admit(rows, logits, law, slots=[slot] * len(rows) if eng.levers is not None else None,
                              seeds=[0] * len(rows))
                    row_of.update(zip(idx, rows))
                # the unused rows score filler from the last prompt (they stay in their own groups)
                last_slot = slot
                filler = list(range(nxt, dec.rows))
                if filler:
                    dec.admit(filler, logits, law,
                              slots=[last_slot] * len(filler) if eng.levers is not None else None,
                              seeds=[0] * len(filler))
                S = max(len(records[i].generated_ids) for _, idx in batch for i in idx)
                last = records[batch[-1][1][0]].generated_ids
                tokens = torch.zeros((dec.rows, S), dtype=torch.long)
                for r in range(dec.rows):
                    tokens[r] = torch.tensor((list(last) + [last[-1]] * S)[:S])
                for i, r in row_of.items():
                    g = records[i].generated_ids
                    tokens[r] = torch.tensor(list(g) + [g[-1]] * (S - len(g)))
                dec.score(tokens.to(dev))
                order = [i for _, idx in batch for i in idx]
                signed = [trim(dec.signed(row_of[i], device=dev), len(records[i].generated_ids) - 1)
                          for i in order]
            finally:
                for slot in bound.values():
                    eng.levers.release(slot)
            self._reduce(records, results, order, signed)
        dec.release(range(dec.rows))
        return results

    def _reduce(self, records: Sequence[SignRecord], results: list[SignResult], order: list[int],
                signed: list[signing.SignedRow]) -> None:
        """Reduce a batch's rows together; if the reducer refuses the batch, row by row."""
        def call(ix: list[int], rows: list[signing.SignedRow]) -> list[signing.ReducedRow]:
            out = self.reducer.reduce(rows, token_ids=[records[i].prompt_ids + records[i].generated_ids for i in ix],
                                      engine=self.identity)
            if len(out) != len(rows):
                raise RuntimeError(f"the reducer returned {len(out)} rows for {len(rows)}")
            return out

        try:
            for i, red in zip(order, call(order, signed)):
                results[i].reduced = red
            return
        except Exception as exc:  # noqa: BLE001 - any reducer failure is retried row by row, then reported per row
            logger.warning("sign: the reducer refused a batch of %d rows (%s: %s); retrying row by row",
                           len(order), type(exc).__name__, exc)
        for i, row in zip(order, signed):
            try:
                results[i].reduced = call([i], [row])[0]
            except Exception as exc:  # noqa: BLE001
                results[i].skipped = f"reducer: {type(exc).__name__}: {exc}"
                logger.warning("sign: skipped %s: %s", records[i].request_id, results[i].skipped)


def write_results(out_dir: str | Path, results: Sequence[SignResult]) -> dict[str, Any]:
    """Results into ``out_dir``: ``features.npy`` (the reduced rows' features stacked, fp32, in result order) and
    ``rows.jsonl`` (per record: request id, its row in features.npy or the skip reason, lane id, names digest,
    same_engine, the reducer's meta, the record's meta). Returns a summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = []
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for res in results:
            line: dict[str, Any] = {"request_id": res.record.request_id, "meta": res.record.meta,
                                    "same_engine": res.same_engine}
            if res.reduced is None:
                line["skipped"] = res.skipped or "not signed"
            else:
                x = res.reduced.features
                x = x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, np.float32)
                line.update(index=len(feats), lane_id=res.reduced.lane_id, names_digest=res.reduced.names_digest,
                            reducer_meta=res.reduced.meta)
                feats.append(x.reshape(-1))
            f.write(json.dumps(line, default=str) + "\n")
    if feats:
        widths = {x.shape[0] for x in feats}
        if len(widths) != 1:
            raise ValueError(f"reduced rows have different widths {sorted(widths)}")
        np.save(out / "features.npy", np.stack(feats).astype(np.float32))
    return {"signed": len(feats), "skipped": sum(r.reduced is None for r in results)}
