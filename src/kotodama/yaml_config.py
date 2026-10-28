"""Flat YAML configs with ``extends`` inheritance and environment expansion: training configs (``train.py
--config``) and serve profiles (``server.py --profile``) load through here, so both read one config grammar."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """A YAML config (or its `extends` chain) is malformed."""


_EXTENDS_KEY = "extends"


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML config file into a flat dict of argparse defaults.

    Keys are normalized (hyphens → underscores). A config may declare
    ``extends: <path>`` or ``extends: [<path>, ...]``; parents are resolved
    relative to the child file's directory and loaded recursively. Configs
    are flat, so merging is a plain key override: parents apply in list
    order (later parents override earlier ones), then the child's own keys
    override all parents. The ``extends`` key is consumed here and never
    reaches argparse. Raises ConfigError on cycles, missing parents, or a
    malformed file.

    After merging, environment variables in string values (and in strings
    inside list values) are expanded with ``os.path.expandvars``, so a config
    can say ``data_path: ${KOTODAMA_DATA_ROOT}/train.bin`` and keep machine
    paths in the site profile. A reference to an unset variable is left as
    written (expandvars semantics) and logged as a warning, so the run then
    fails on a path that names the missing variable.
    """
    merged = _resolve_yaml_config(Path(path).resolve(), chain=())
    return {k: _expand_env(k, v, path) for k, v in merged.items()}


_ENV_REF = re.compile(r"\$(\w+|\{[^}]*\})")


def _expand_env(key: str, value: Any, path: str | os.PathLike[str]) -> Any:
    """``value`` with environment variables expanded in every string it holds."""
    if isinstance(value, list):
        return [_expand_env(key, v, path) for v in value]
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    unresolved = _ENV_REF.findall(expanded)
    if unresolved:
        logger.warning(
            "Config %s: %s=%r references unset environment variable(s): %s",
            path, key, value, ", ".join(sorted({u.strip("{}") for u in unresolved})),
        )
    return expanded


def _resolve_yaml_config(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required to load YAML configs")

    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"Config inheritance cycle: {cycle}")
    if not path.is_file():
        if chain:
            raise ConfigError(f"Parent config not found: {path} (extended by {chain[-1]})")
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config {path} must be a mapping of key: value, got {type(raw).__name__}"
        )
    for k in raw:
        if not isinstance(k, str):
            raise ConfigError(f"Config {path}: non-string key {k!r}")

    own = {k.replace("-", "_"): v for k, v in raw.items()}
    extends = own.pop(_EXTENDS_KEY, None)
    if extends is None:
        parents: list[str] = []
    elif isinstance(extends, str):
        parents = [extends]
    elif isinstance(extends, list) and all(isinstance(p, str) for p in extends):
        parents = list(extends)
    else:
        raise ConfigError(
            f"Config {path}: `extends` must be a path or a list of paths, got {extends!r}"
        )

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged.update(_resolve_yaml_config(parent_path.resolve(), (*chain, path)))
    merged.update(own)
    return merged
