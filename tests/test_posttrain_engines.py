"""Post-training engines end to end on CPU, on a tiny LOOPED AttnRes model with the real tokenizer's vocab:
return_hidden + sliced completion log-probs match the full-logits math; SFT, DPO and KTO run, write loadable
checkpoints, start at the reference (DPO margin 0 / KTO reward 0 at step 1), and the reference cache round-trips."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from kotodama import presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

# tsumugi geometry in miniature (cf. tests/test_decode_engine_looped.py), at the SmolLM2 vocab so the real
# tokenizer's ids fit.
TINY = dict(hidden_size=32, num_layers=7, num_attention_heads=2, num_kv_heads=1, head_dim=16, intermediate_size=64,
            vocab_size=49152, max_position_embeddings=512, loop_start=2, loop_len=2, loop_times=2)
BOUNDS = "0,1,2,4,6"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture()
def tiny_ckpt(tmp_path, monkeypatch):
    monkeypatch.setitem(presets.SHAPES, "tiny-loop", dict(TINY))
    torch.manual_seed(0)
    cfg = KotodamaModelConfig(**TINY, attn_res=True, attn_res_boundaries=[int(x) for x in BOUNDS.split(",")],
                              attn_impl="sdpa", rope_theta=10000.0)
    model = KotodamaModel(cfg)
    with torch.no_grad():  # move routing off its zero init so every block matters
        for n, p in model.named_parameters():
            if "res_query" in n:
                p.normal_(0, 0.5)
    path = tmp_path / "base.pt"
    torch.save({"step": 0, "model": model.state_dict()}, path)
    return path


def _common(ckpt, out) -> list[str]:
    return ["--checkpoint", str(ckpt), "--model_size", "tiny-loop", "--attn_res_boundaries", BOUNDS,
            "--device", "cpu", "--attn_impl", "sdpa", "--no-rope_fp32", "--out_dir", str(out)]


def _log(out) -> list[dict]:
    return [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]


def test_return_hidden_and_sliced_logps_match_full_logits(tiny_ckpt) -> None:
    from kotodama.eval.model_loader import load_model
    from kotodama.posttrain.common import completion_logps

    m = load_model(tiny_ckpt, config_section="tiny-loop", device="cpu",
                   attn_res_config={"attn_res": True, "attn_res_boundaries": [0, 1, 2, 4, 6]},
                   overrides={"attn_impl": "sdpa"})
    ids = torch.randint(5, 49152, (3, 24), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = m(ids)["logits"]
        hidden = m(ids, return_hidden=True)["hidden"]
        assert torch.allclose(F.linear(hidden, m.get_lm_head_weight()), logits, atol=1e-5)
        pls, lens = [5, 10, 1], [24, 20, 12]
        got = completion_logps(m, ids, pls, lens, chunk=7)
        lsm = F.log_softmax(logits.float(), -1)
        want = torch.stack([lsm[b, pl - 1:n - 1].gather(-1, ids[b, pl:n, None]).sum()
                            for b, (pl, n) in enumerate(zip(pls, lens))])
    assert torch.allclose(got, want, atol=1e-4)


def test_sft_runs_and_learns(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.eval.model_loader import load_model
    from kotodama.posttrain import sft

    conv = {"messages": [{"role": "user", "content": "say the word"},
                         {"role": "assistant", "content": "lantern lantern lantern"}]}
    data = tmp_path / "chat.jsonl"
    data.write_text("\n".join(json.dumps(conv) for _ in range(16)) + "\n")
    out = tmp_path / "sft"
    rc = sft.main(_common(tiny_ckpt, out) + [
        "--format", "chatml", "--train_jsonl", str(data), "--eval_jsonl", str(data), "--max_seq_len", "64",
        "--micro_batch_size", "2",
        "--grad_accum", "1", "--max_steps", "12", "--muon_lr", "0.02", "--adamw_lr", "3e-3",
        "--warmup_ratio", "0", "--save_every", "6"])
    assert rc == 0
    log = [r for r in _log(out) if "loss" in r]
    assert len(log) == 12 and log[-1]["loss"] < log[0]["loss"] * 0.9
    evals = [r for r in _log(out) if "eval/loss" in r]
    assert evals[-1]["eval/loss"] < evals[0]["eval/loss"] and "eval/im_end_top1" in evals[-1]
    final = json.loads((out / "final.json").read_text())["checkpoint"]
    assert (out / "checkpoints" / "step_00000006.pt").exists()
    m = load_model(final, config_section="tiny-loop", device="cpu",
                   attn_res_config={"attn_res": True, "attn_res_boundaries": [0, 1, 2, 4, 6]})
    assert m.layers[4].attn is m.layers[2].attn  # the looped aliasing survives the round trip


def _pairs(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}], "chosen": f"good answer {i}",
             "rejected": f"bad bad bad {i}"} for i in range(8)]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_dpo_starts_at_reference_and_cache_roundtrips(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.posttrain import pref

    data = _pairs(tmp_path)
    args = ["--objective", "dpo", "--data", str(data), "--batch", "2", "--grad_accum", "1", "--max_steps", "3",
            "--pad_multiple", "16", "--ref_cache_dir", str(tmp_path / "rc")]
    out1 = tmp_path / "dpo1"
    assert pref.main(_common(tiny_ckpt, out1) + args) == 0
    first = _log(out1)[0]
    assert first["margin"] == pytest.approx(0.0, abs=1e-4)  # policy == reference at the first micro
    assert "final/pref_acc" in json.loads((out1 / "final.json").read_text())
    caches = list((tmp_path / "rc").glob("ref_*.json"))
    assert len(caches) == 1 and len(json.loads(caches[0].read_text())["entries"]) == 16
    out2 = tmp_path / "dpo2"  # second run: all sides from the cache, verified against fresh values
    assert pref.main(_common(tiny_ckpt, out2) + args) == 0


def test_kto_mismatched_runs(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.posttrain import pref

    rows = [{"prompt": f"Once there was a keeper number {i}.", "completion": t, "label": lab}
            for i, (t, lab) in enumerate([("He lit the lamp.", True), ("He he he he he he.", False)] * 4)]
    data = tmp_path / "kto.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "kto"
    assert pref.main(_common(tiny_ckpt, out) + [
        "--objective", "kto", "--data", str(data), "--batch", "4", "--grad_accum", "1", "--max_steps", "2",
        "--pad_multiple", "16", "--kto_kl", "mismatched", "--no-ref_cache"]) == 0
    first = _log(out)[0]
    assert first["kl"] == pytest.approx(0.0, abs=1e-5)
    assert first.get("reward_desirable", 0.0) == pytest.approx(0.0, abs=1e-4)


def test_sft_varlen_inputs_static_bound_when_compiled() -> None:
    from kotodama.posttrain.sft import varlen_inputs
    b = {"labels": torch.zeros(1, 8), "position_ids": torch.zeros(1, 8), "cu_seqlens": torch.tensor([0, 3, 8]),
         "max_seqlen": 5}
    assert varlen_inputs(b, 8, compiled=False)["max_seqlen"] == 5
    kw = varlen_inputs(b, 8, compiled=True)
    assert kw["max_seqlen"] == 8 and kw["cu_seqlens"] is b["cu_seqlens"] and set(kw) == {
        "labels", "position_ids", "cu_seqlens", "max_seqlen"}


def test_pref_save_steps_are_bounded() -> None:
    from kotodama.posttrain.pref import save_steps
    assert save_steps(600, 5) == {100, 200, 300, 400, 500}
    assert len(save_steps(8, 5)) == 5 and max(save_steps(8, 5)) < 8
    assert save_steps(3, 5) == {1, 2}
    assert save_steps(100, 0) == set() and save_steps(1, 5) == set()


def test_epoch_law_enforced(tiny_ckpt, tmp_path) -> None:
    from kotodama.posttrain import pref

    with pytest.raises(SystemExit, match="refusing > 1 epoch"):
        pref.parse_args(_common(tiny_ckpt, tmp_path) + ["--data", "x.jsonl", "--epochs", "2"])


@pytest.mark.parametrize("name,module", [("tsumugi-sft", "sft"), ("tsumugi-dpo", "pref"),
                                         ("tsumugi-kto-repetition", "pref"), ("tsumugi-dialogue-unmasked", "sft"),
                                         ("tsumugi-dialogue-masked", "sft"), ("tsumugi-assistant-sft", "sft"),
                                         ("tsumugi-assistant-dpo", "pref")])
def test_posttrain_configs_parse(name: str, module: str) -> None:
    import importlib
    from pathlib import Path

    mod = importlib.import_module(f"kotodama.posttrain.{module}")
    cfg = Path(__file__).resolve().parents[1] / "configs" / "posttrain" / f"{name}.yaml"
    req = ["--checkpoint", "c.pt", "--out_dir", "o"] + (
        ["--train_jsonl", "t.jsonl"] if module == "sft" else ["--data", "d.jsonl"])
    a = mod.parse_args(["--config", str(cfg)] + req)
    assert a.model_size == "1b-tsumugi" and not a.attnres_eps_fp64_compat
    assert a.attn_res_boundaries == "0,1,7,9,17,19,39,59"


def test_sft_on_plain_transcripts_unmasked(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.posttrain import sft

    row = {"header": "Two friends at a kitchen table.",
           "turns": [{"speaker": "Mara", "text": "Is the kettle on?"}, {"speaker": "Teo", "text": "It is now."},
                     {"speaker": "Mara", "text": "Good."}]}
    data = tmp_path / "t.jsonl"
    data.write_text("\n".join(json.dumps(row) for _ in range(12)) + "\n")
    out = tmp_path / "sft-tx"
    assert sft.main(_common(tiny_ckpt, out) + [
        "--train_jsonl", str(data), "--eval_jsonl", str(data), "--max_seq_len", "64", "--micro_batch_size", "2",
        "--grad_accum", "1", "--max_steps", "6", "--muon_lr", "0.02", "--adamw_lr", "3e-3", "--warmup_ratio", "0",
        "--train_speakers", "all"]) == 0
    log = [r for r in _log(out) if "loss" in r]
    assert log[-1]["loss"] < log[0]["loss"]
    assert "eval/eos_top1" in [r for r in _log(out) if "eval/loss" in r][-1]


def test_dpo_ln_runs_and_starts_at_reference(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.posttrain import pref

    data = _pairs(tmp_path)
    out = tmp_path / "dpo-ln"
    assert pref.main(_common(tiny_ckpt, out) + [
        "--objective", "dpo_ln", "--beta", "5", "--data", str(data), "--batch", "2", "--grad_accum", "1",
        "--max_steps", "2", "--pad_multiple", "16", "--no-ref_cache"]) == 0
    assert _log(out)[0]["margin"] == pytest.approx(0.0, abs=1e-4)


def test_sft_adamw_only_learns(tiny_ckpt, tok, tmp_path) -> None:
    from kotodama.posttrain import sft

    row = {"turns": [{"speaker": "User", "text": "Name a color."}, {"speaker": "Assistant", "text": "Teal."}]}
    data = tmp_path / "a.jsonl"
    data.write_text("\n".join(json.dumps(row) for _ in range(12)) + "\n")
    out = tmp_path / "sft-adamw"
    assert sft.main(_common(tiny_ckpt, out) + [
        "--train_jsonl", str(data), "--max_seq_len", "32", "--micro_batch_size", "2", "--grad_accum", "1",
        "--max_steps", "6", "--optimizer", "adamw", "--adamw_lr", "3e-3", "--warmup_ratio", "0",
        "--train_speakers", "Assistant", "--drop_overlong"]) == 0
    log = [r for r in _log(out) if "loss" in r]
    assert log[-1]["loss"] < log[0]["loss"] and log[0]["muon_lr"] == 0.0


@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_wraps_on_the_cuda_path(dynamic) -> None:
    """The CUDA branch of common.compiled (torch.compile is lazy, so this runs on CPU without compiling)."""
    import argparse

    from kotodama.posttrain import common

    model = KotodamaModel(KotodamaModelConfig(hidden_size=32, num_layers=2, num_attention_heads=2, num_kv_heads=1,
                                              head_dim=16, intermediate_size=64, vocab_size=64,
                                              max_position_embeddings=32, attn_impl="sdpa"))
    args = argparse.Namespace(device="cuda:0", compile=True, attnres_eps_fp64_compat=False, no_attn_res=True)
    model.fused_linear_ce_loss = torch.nn.Identity()  # stands in for Liger's fused loss (CUDA-only)
    wrapped = common.compiled(model, args, dynamic_lengths=dynamic)
    assert wrapped is not model and wrapped.get_lm_head_weight() is model.get_lm_head_weight()
    assert getattr(model.fused_linear_ce_loss.forward, "_torchdynamo_disable", False)  # the loss stays eager
    args.compile = False
    assert common.compiled(model, args) is model
