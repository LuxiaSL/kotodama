"""Serve profiles: one named YAML file per way of serving a model, so a server is one ``--profile <name>`` instead
of a chain of flags.

A profile lives at ``configs/serve/<name>.yaml`` (or any path). Its keys are ``server.py`` flags by their argparse
names (``prefix_cache``, ``xbatch_rows``, ...; hyphens and underscores are interchangeable), loaded through
``kotodama.yaml_config`` like a training config, so ``extends:`` and ``${ENV}`` expansion work the same way. One
extra key, ``env``, maps environment variables the server process needs before the model is built; they are set
only where the environment does not already set them. A profile supplies defaults: any flag given on the command
line overrides it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from kotodama.yaml_config import ConfigError, load_yaml_config

PROFILE_DIR = Path(__file__).resolve().parents[3] / "configs" / "serve"
ENV_KEY = "env"


def resolve(name_or_path: str) -> Path:
    """A profile name (``tsumugi-1b`` -> ``configs/serve/tsumugi-1b.yaml``) or a path to a profile file."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    found = PROFILE_DIR / f"{name_or_path}.yaml"
    if not found.is_file():
        known = sorted(x.stem for x in PROFILE_DIR.glob("*.yaml")) if PROFILE_DIR.is_dir() else []
        raise ConfigError(f"no serve profile {name_or_path!r}; known: {known}")
    return found


def load(name_or_path: str) -> tuple[dict[str, Any], dict[str, str]]:
    """The profile's flag defaults (by argparse dest) and its environment block."""
    path = resolve(name_or_path)
    raw = load_yaml_config(path)
    env = raw.pop(ENV_KEY, None) or {}
    if not isinstance(env, dict) or not all(isinstance(k, str) for k in env):
        raise ConfigError(f"profile {path}: `env` must be a mapping of variable names to values")
    flags = {k: (",".join(str(x) for x in v) if isinstance(v, list) else v) for k, v in raw.items()}
    return flags, {k: str(v) for k, v in env.items()}


def apply(parser: argparse.ArgumentParser, name_or_path: str) -> dict[str, str]:
    """Make the profile's flags ``parser``'s defaults (a required flag the profile sets stops being required) and
    set its environment where unset. Refuses keys that are not flags of ``parser``. Returns the environment block."""
    flags, env = load(name_or_path)
    actions = {a.dest: a for a in parser._actions}
    unknown = sorted(k for k in flags if k not in actions or k in ("help", "profile"))
    if unknown:
        raise ConfigError(f"profile {name_or_path!r}: {unknown} are not server flags")
    for k in flags:
        actions[k].required = False
    parser.set_defaults(**flags)
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env
