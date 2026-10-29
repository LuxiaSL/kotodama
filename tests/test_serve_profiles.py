"""Serve profiles (kotodama.serve.profiles): every profile in configs/serve names only server flags and parses to a
complete command line; a flag given on the command line overrides the profile; the profile's environment never
overrides one already set; a key that is not a server flag is refused. A tsumugi-lineage shape started without the
AttnRes eps compatibility it trained with is refused unless explicitly allowed."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from kotodama import presets  # noqa: E402
from kotodama.serve import profiles  # noqa: E402
from kotodama.serve.server import build_arg_parser  # noqa: E402
from kotodama.yaml_config import ConfigError  # noqa: E402

PROFILES = sorted(profiles.PROFILE_DIR.glob("*.yaml"))


def test_there_is_a_tsumugi_profile() -> None:
    assert "tsumugi-1b" in {p.stem for p in PROFILES}


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_profile_is_a_complete_server_command_line(path: Path, monkeypatch) -> None:
    flags, env = profiles.load(str(path))
    parser = build_arg_parser()
    for k in env:
        monkeypatch.setenv(k, "unset-marker")  # recorded, so the test restores the environment it changes
        monkeypatch.delenv(k)
    profiles.apply(parser, str(path))
    args = parser.parse_args([])  # nothing else on the command line
    for k, v in flags.items():
        assert getattr(args, k) == v, k
    presets.resolve(args.model_size)  # the shape is a known preset
    for k, v in env.items():
        import os
        assert os.environ[k] == v


def test_command_line_overrides_the_profile_and_env_is_not_clobbered(monkeypatch) -> None:
    monkeypatch.setenv("KOTODAMA_ATTNRES_EPS_FP64_COMPAT", "0")
    parser = build_arg_parser()
    profiles.apply(parser, "tsumugi-1b")
    args = parser.parse_args(["--levers", "4", "--port", "9123", "--checkpoint", "/x/y.pt"])
    assert args.levers == 4 and args.port == 9123 and args.checkpoint == "/x/y.pt"
    assert args.xbatch_rows == "16,32,64,128" and args.cascade_attn is True
    import os
    assert os.environ["KOTODAMA_ATTNRES_EPS_FP64_COMPAT"] == "0"


def test_unknown_keys_and_names_are_refused(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_size: 1b-tsumugi\nnot_a_flag: 1\n")
    with pytest.raises(ConfigError, match="not_a_flag"):
        profiles.apply(build_arg_parser(), str(bad))
    with pytest.raises(ConfigError, match="no serve profile"):
        profiles.resolve("no-such-profile")


def test_profiles_extend_like_training_configs(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model_size: 1b-tsumugi\nlevers: 16\nenv:\n  A_VAR: one\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nlevers: 8\n")
    flags, env = profiles.load(str(tmp_path / "child.yaml"))
    assert flags == {"model_size": "1b-tsumugi", "levers": 8} and env == {"A_VAR": "one"}


def test_a_profile_switch_can_be_turned_off_on_the_command_line() -> None:
    parser = build_arg_parser()
    profiles.apply(parser, "tsumugi-1b")
    args = parser.parse_args(["--no-seeded_rows", "--no-cascade_attn", "--levers", "0"])
    assert args.seeded_rows is False and args.cascade_attn is False and args.levers == 0
    assert build_arg_parser().parse_args(["--checkpoint", "c", "--prefix-cache"]).prefix_cache is True


def test_narrow_row_buckets_skip_kernel_benchmarking(monkeypatch) -> None:
    from kotodama.serve import server

    monkeypatch.setattr(server, "_compile_mode", "max-autotune")
    monkeypatch.setattr(server, "_autotune_min_rows", 64)
    assert [server._row_compile_mode(r) for r in (16, 32, 64, 128)] == ["reduce-overhead"] * 2 + ["max-autotune"] * 2
    monkeypatch.setattr(server, "_compile_mode", "max-autotune-no-cudagraphs")
    assert server._row_compile_mode(16) == "default"
    monkeypatch.setattr(server, "_compile_mode", "reduce-overhead")
    assert server._row_compile_mode(16) == "reduce-overhead"
    monkeypatch.setattr(server, "_autotune_min_rows", 0)
    monkeypatch.setattr(server, "_compile_mode", "max-autotune")
    assert server._row_compile_mode(16) == "max-autotune"


def test_a_tsumugi_shape_without_the_eps_compat_is_refused() -> None:
    from kotodama.serve.server import attnres_eps_refusal

    assert "KOTODAMA_ATTNRES_EPS_FP64_COMPAT" in attnres_eps_refusal("1b-tsumugi", {}, allowed=False)
    assert attnres_eps_refusal("1b-moonshot-loop", {}, allowed=False) is not None  # the alias resolves to it
    assert attnres_eps_refusal("1b-tsumugi", {"KOTODAMA_ATTNRES_EPS_FP64_COMPAT": "1"}, allowed=False) is None
    assert attnres_eps_refusal("1b-tsumugi", {}, allowed=True) is None
    assert attnres_eps_refusal("1b-standard", {}, allowed=False) is None  # routes in the kernels' own precision
