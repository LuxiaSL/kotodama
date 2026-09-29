"""kotodama.eval.uncheatable: BPB math (known answers), chunking, trimming,
data layouts, result schema, the CLI end to end on a tiny checkpoint, and the
hm_backfill mapping."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from kotodama.eval import uncheatable as U
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

V = 300  # stub vocab: byte b -> id b + 1, id 0 = BOS


def encode(text: str) -> list[int]:
    return [b + 1 for b in text.encode("utf-8")]


def uniform_logits(ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros(ids.shape[-1], V)


DOCS = {
    "alpha": ["hello world, this is a document", "tiny bit", "ümlaut ç 字 ok"],
    "beta": ["x" * 50, "print('hi')\n" * 3],
}


def _settings(**kw) -> U.Settings:
    base = dict(checkpoint="stub", config_section="smoke", attn_res=False,
                attn_res_boundaries=None, rope_fp32=False, tokenizer="stub",
                bos_mode="add_default_bos", bos_id=0, chunk_size=4000,
                loss_precision="fp32", max_docs_per_domain=None,
                max_tokens_per_domain=None, data_source="mem", device="cpu")
    base.update(kw)
    return U.Settings(**base)


# ── known answers ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("chunk", [3, 7, 4000])
def test_uniform_model_bpb_known_answer(chunk):
    texts = DOCS["alpha"]
    r = U.eval_domain(uniform_logits, encode, texts, bos_id=0, chunk_size=chunk,
                      loss_precision="fp32")
    n_tok = sum(len(encode(t)) for t in texts)
    n_bytes = sum(len(t.encode()) for t in texts)
    assert r.n_tokens == n_tok and r.n_bytes == n_bytes  # 1 byte = 1 token here
    expect = n_tok * math.log(V) / (n_bytes * math.log(2))
    assert r.bpb == pytest.approx(expect, rel=1e-6)  # fp32 CE: ln V to ~1e-8
    # every token scored exactly once, whatever the chunking
    assert r.nll_mean == pytest.approx(n_tok * math.log(V) / len(texts), rel=1e-6)


def test_bf16_mode_replicates_original_rounding():
    """bf16: per-token loss AND its per-chunk sum are rounded to bf16."""
    ids = encode("x" * 40)
    chunk = 16
    got = U.doc_nll(uniform_logits, ids, 0, chunk, "bf16")
    lnv_bf16 = torch.tensor(math.log(V)).bfloat16()
    expect = 0.0
    for b in range(0, len(ids), chunk):
        n = len(ids[b:b + chunk])
        expect += torch.full((n,), lnv_bf16.item(), dtype=torch.bfloat16).sum().item()
    assert got == expect
    assert got == pytest.approx(len(ids) * math.log(V), rel=5e-3)


def test_bos_prepended_per_chunk_and_scored_targets():
    seen: list[list[int]] = []

    def spy(ids: torch.Tensor) -> torch.Tensor:
        seen.append(ids[0].tolist())
        return torch.zeros(ids.shape[-1], V)

    ids = list(range(1, 11))
    U.doc_nll(spy, ids, bos_id=0, chunk_size=4, loss_precision="fp32")
    assert seen == [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10]]


def test_peaked_model_matches_direct_cross_entropy():
    torch.manual_seed(0)
    table = torch.randn(V, V)

    def bigram(ids: torch.Tensor) -> torch.Tensor:  # logits depend on current token
        return table[ids[0]]

    ids = encode("the quick brown fox")
    got = U.doc_nll(bigram, ids, 0, 4000, "fp32")
    full = torch.tensor([0, *ids])
    ref = F.cross_entropy(table[full[:-1]], full[1:], reduction="sum").item()
    assert got == pytest.approx(ref, rel=1e-6)


def test_short_doc_rejected():
    with pytest.raises(ValueError):
        U.doc_nll(uniform_logits, [5], 0)


# ── trimming ─────────────────────────────────────────────────────────────────

def test_select_docs():
    counts = [10, 20, 30, 40]
    assert U.select_docs(counts) == 4
    assert U.select_docs(counts, max_docs=2) == 2
    assert U.select_docs(counts, max_tokens=25) == 2      # crossing doc kept
    assert U.select_docs(counts, max_tokens=30) == 2      # exactly met
    assert U.select_docs(counts, max_tokens=10_000) == 4
    assert U.select_docs(counts, max_docs=1, max_tokens=1000) == 1
    with pytest.raises(ValueError):
        U.select_docs(counts, max_docs=0)


def test_trimmed_domain_is_prefix_exact():
    texts = DOCS["alpha"]
    full = U.eval_domain(uniform_logits, encode, texts[:2], 0, loss_precision="fp32")
    trimmed = U.eval_domain(uniform_logits, encode, texts, 0, loss_precision="fp32",
                            max_docs=2)
    assert trimmed.trimmed and not full.trimmed
    assert trimmed.n_docs == 2 and trimmed.n_docs_total == 3
    assert trimmed.bpb == full.bpb and trimmed.n_bytes == full.n_bytes
    by_tok = U.eval_domain(uniform_logits, encode, texts, 0, loss_precision="fp32",
                           max_tokens=len(encode(texts[0])) + 1)
    assert by_tok.n_docs == 2


# ── data layouts ─────────────────────────────────────────────────────────────

def test_load_domains_jsonl_and_json(tmp_path):
    (tmp_path / "alpha.jsonl").write_text(
        "\n".join(json.dumps({"content": t}) for t in DOCS["alpha"]) + "\n")
    (tmp_path / "beta.json").write_text(json.dumps(DOCS["beta"]))
    d = U.load_domains(tmp_path)
    assert d == DOCS
    assert U.load_domains(tmp_path / "beta.json") == {"beta": DOCS["beta"]}


def test_load_domains_parquet_groups_in_row_order(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [("beta", "b0"), ("alpha", "a0"), ("beta", "b1"), ("alpha", "a1")]
    table = pa.table({"category": [r[0] for r in rows],
                      "content": [r[1] for r in rows],
                      "date": ["2026-04-01"] * 4})
    sub = tmp_path / "snap" / "data"
    sub.mkdir(parents=True)
    pq.write_table(table, sub / "test-00000-of-00001.parquet")
    d = U.load_domains(tmp_path / "snap")
    assert list(d) == ["beta", "alpha"]
    assert d == {"beta": ["b0", "b1"], "alpha": ["a0", "a1"]}


def test_load_domains_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        U.load_domains(tmp_path)  # empty dir
    with pytest.raises(FileNotFoundError):
        U.load_domains(tmp_path / "nope" / "deeper")


# ── result schema ────────────────────────────────────────────────────────────

def test_evaluate_schema_and_mean():
    res = U.evaluate(uniform_logits, encode, DOCS, _settings(), step=7, tokens=99)
    json.dumps(res)  # serializable
    assert res["schema"] == U.SCHEMA
    assert res["step"] == 7 and res["tokens"] == 99
    assert set(res["domains"]) == {"alpha", "beta"}
    for d in res["domains"].values():
        assert {"bpb", "bpc", "nll_mean", "n_docs", "n_docs_total", "n_tokens",
                "n_bytes", "n_chars", "trimmed"} <= set(d)
    bpbs = [d["bpb"] for d in res["domains"].values()]
    assert res["mean_bpb"] == pytest.approx(sum(bpbs) / 2)
    assert res["n_domains"] == 2 and res["trimmed"] is False
    assert res["settings"]["bos_mode"] == "add_default_bos"


def test_parse_boundaries_and_resolve_arch():
    assert U.parse_boundaries("dd3b") == [0, 1, 3, 7, 15, 19, 24]
    assert U.parse_boundaries("0,2") == [0, 2]
    assert U.parse_boundaries("none") is None
    with pytest.raises(ValueError):
        U.parse_boundaries("bogus")
    assert U.resolve_arch("/no/such.pt", "1b", None) == ("1b", [0, 1, 3, 7, 15, 19, 24])
    assert U.resolve_arch("/no/such.pt", "model", "none") == ("3b", None)
    assert U.resolve_arch("kotodama-3b-base-final", None, None) == (
        "3b", [0, 1, 3, 7, 15, 19, 24])


# ── CLI end to end on a tiny checkpoint ──────────────────────────────────────

class _StubTok:
    bos_token_id = 0

    def __len__(self) -> int:
        return V

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": encode(text)}


def test_cli_end_to_end(tmp_path, monkeypatch):
    import transformers

    from kotodama import presets

    torch.manual_seed(0)
    shape = presets.shape("smoke")
    cfg = LuxiaModelConfig(**shape, attn_res=True, attn_res_boundaries=[0, 2],
                           rope_fp32=True, attn_impl="sdpa")
    model = LuxiaBaseModel(cfg).eval()
    ck = tmp_path / "run" / "step_00000050.pt"
    ck.parent.mkdir()
    torch.save({"model": model.state_dict(), "step": 123,
                "tokens_consumed": 4567}, ck)
    data = tmp_path / "data"
    data.mkdir()
    for name, texts in DOCS.items():
        (data / f"{name}.json").write_text(json.dumps(texts))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _StubTok()))
    out = tmp_path / "res.json"
    U.main(["--checkpoint", str(ck), "--config-section", "smoke",
            "--attn-res-boundaries", "0,2", "--rope-fp32", "--data-dir", str(data),
            "--domains", "alpha", "--chunk-size", "8", "--loss-precision", "fp32",
            "--device", "cpu", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["step"] == 123 and res["tokens"] == 4567
    assert list(res["domains"]) == ["alpha"]
    s = res["settings"]
    assert s["rope_fp32"] and s["attn_res_boundaries"] == [0, 2]
    assert s["extra"]["attn_impl"] == "sdpa" and s["chunk_size"] == 8

    # independent reference: same weights, chunks of 8 with BOS, fp32 CE
    nll, nbytes = 0.0, 0
    with torch.no_grad():
        for t in DOCS["alpha"]:
            ids = encode(t)
            nbytes += len(t.encode())
            for b in range(0, len(ids), 8):
                x = torch.tensor([[0, *ids[b:b + 8]]])
                lg = model(input_ids=x)["logits"][0]
                nll += F.cross_entropy(lg[:-1], x[0, 1:], reduction="sum").item()
    assert res["domains"]["alpha"]["bpb"] == pytest.approx(
        nll / nbytes / math.log(2), rel=1e-5)


# ── hm_backfill mapping ──────────────────────────────────────────────────────

def _hm():
    spec = importlib.util.spec_from_file_location(
        "hm_backfill", Path(__file__).parent.parent / "scripts" / "eval" / "hm_backfill.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hm_backfill"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_hm_backfill_uncheatable_mapping():
    hm = _hm()
    res = U.evaluate(uniform_logits, encode, DOCS, _settings(), step=340,
                     tokens=50282496)
    pts = hm.uncheatable_to_points(res)
    assert len(pts) == 1 and pts[0]["step"] == 340
    m = pts[0]["metrics"]
    assert set(m) == {"uncheatable/alpha", "uncheatable/beta", "uncheatable/mean",
                      "tokens_seen"}
    assert m["uncheatable/mean"] == pytest.approx(res["mean_bpb"])
    assert m["tokens_seen"] == 50282496.0
    # step override; missing step refused; merge with other sweeps by step
    assert hm.uncheatable_to_points(res, step=9)[0]["step"] == 9
    res["step"] = None
    with pytest.raises(ValueError):
        hm.uncheatable_to_points(res)
    with pytest.raises(ValueError):
        hm.uncheatable_to_points({"schema": "other"})
    merged = hm.merge_points(hm.uncheatable_to_points(res, step=340),
                             [{"step": 340, "metrics": {"gates/x": 1.0}}])
    assert set(merged[0]["metrics"]) >= {"gates/x", "uncheatable/mean"}
