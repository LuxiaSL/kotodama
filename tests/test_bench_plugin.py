"""kotodama and the binding benchmark (the ``bench`` extra): what kotodama keeps a copy of stays identical to the
benchmark's, the ``koto:`` loader registers, ``python -m kotodama.eval.binding`` runs the benchmark's CLI, and the
in-loop riders suite refuses with the extra's name when the benchmark is absent."""

from __future__ import annotations

import ast
import builtins
import random
from pathlib import Path

import pytest
import torch

bench_score = pytest.importorskip("binding_bench.score", reason="the binding benchmark is not installed (kotodama[bench])")

from binding_bench import chatml as bench_chatml  # noqa: E402
from binding_bench import tokenizer as bench_tokenizer  # noqa: E402
from binding_bench.binding import models as bench_models  # noqa: E402

from kotodama import bench_plugin, presets  # noqa: E402
from kotodama.serve import chatml, logprobs  # noqa: E402


def _code_without_docstring(path: Path) -> str:
    tree = ast.parse(path.read_text())
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
        tree.body = tree.body[1:]
    return ast.dump(tree)


def test_the_servers_scorer_is_the_benchmarks_scorer():
    """Same code below the module docstring, so a /logprobs row and a benchmark row mean the same thing."""
    assert _code_without_docstring(Path(logprobs.__file__)) == _code_without_docstring(Path(bench_score.__file__))


def test_the_two_scorers_agree_on_a_batch():
    class Tok:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [3 + (ord(c) % 40) for c in text]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    gen = torch.Generator().manual_seed(0)
    table = torch.randn(64, 50, generator=gen)

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        return table[ids % 64]

    rng = random.Random(1)
    items = [{"id": f"i{k}", "prompt": "".join(rng.choice("abcdefgh ") for _ in range(rng.randint(3, 30))),
              "candidates": [" x", " yy", " z"]} for k in range(9)]
    ours = logprobs.score_logprob_items(logits_fn, Tok(), items, 4, "cpu", max_batch=4)
    theirs = bench_score.score_logprob_items(logits_fn, Tok(), items, 4, "cpu", max_batch=4)
    assert ours == theirs and len(ours) == 9
    assert logprobs.plan_batches([5, 1, 9, 3], 2, 100) == bench_score.plan_batches([5, 1, 9, 3], 2, 100)


def test_the_chat_frame_and_the_item_tokenizer_match_kotodamas():
    msgs = [{"role": "user", "content": "Box A holds the cup."}, {"role": "assistant", "content": "ok"}]
    assert chatml.render_chatml(msgs) == bench_chatml.render_chatml(msgs)
    assert presets.TOKENIZER_NAME == bench_tokenizer.TOKENIZER_NAME


def test_register_makes_load_koto_the_koto_loader(monkeypatch):
    monkeypatch.setattr(bench_models, "_REGISTERED", {})
    monkeypatch.delenv(bench_models.LOADERS_ENV, raising=False)
    bench_plugin.register()
    bench_plugin.register()   # idempotent
    assert bench_models.loader_for("koto") is bench_plugin.load_koto


def test_the_env_name_resolves_to_the_same_loader(monkeypatch):
    monkeypatch.setattr(bench_models, "_REGISTERED", {})
    monkeypatch.setenv(bench_models.LOADERS_ENV, "koto=kotodama.bench_plugin:load_koto")
    assert bench_models.loader_for("koto") is bench_plugin.load_koto


def test_the_module_cli_runs_the_benchmarks_cli(monkeypatch, capsys):
    from kotodama.eval.binding.__main__ import main

    monkeypatch.setattr(bench_models, "_REGISTERED", {})
    assert main(["list"]) == 0
    assert "capacity-v1" in capsys.readouterr().out
    assert bench_models._REGISTERED["koto"] is bench_plugin.load_koto


def test_inloop_riders_refuse_without_the_benchmark(monkeypatch):
    from kotodama.training import inloop

    real_import = builtins.__import__

    def no_bench(name, *a, **kw):
        if name.startswith("binding_bench"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_bench)
    monkeypatch.setattr("importlib.import_module", lambda name, *a: no_bench(name))
    with pytest.raises(RuntimeError, match=r"kotodama\[bench\]"):
        inloop._bench_riders("lite")


def test_rider_payloads_validate_against_the_servers_request_model():
    pytest.importorskip("fastapi")
    try:
        from kotodama.serve.server import LogprobRequest
    except Exception as exc:  # pragma: no cover — server deps unavailable
        pytest.skip(f"server import unavailable: {exc}")
    from binding_bench.riders import grid, items

    its, shots = items.generate(list(items.BANKED_VERIFIED_NAMES), 1, 1, 1002)
    payload = [grid.build_payload_item(it, rd, 3, shots, "koto")[0] for it in (its[0], its[2]) for rd in items.RENDERERS]
    req = LogprobRequest(items=payload, top_n=5)
    assert [x.id for x in req.items] == [p["id"] for p in payload]
