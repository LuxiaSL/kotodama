"""Uvicorn logging for the kotodama servers: every line, request (access) lines included, starts with a timestamp.

uvicorn's default access line is ``INFO:     10.0.0.2:5767 - "POST /v1/completions HTTP/1.1" 200 OK``, with no time,
so a server log cannot say when it was last used. This config puts ``%(asctime)s`` first on uvicorn's own and access
lines, in the format of the servers' ``logging.basicConfig`` lines (date, time, milliseconds).
"""

from __future__ import annotations

import copy
from typing import Any


def uvicorn_log_config() -> dict[str, Any]:
    """uvicorn's ``LOGGING_CONFIG`` with the time leading every formatter; pass as ``uvicorn.run(log_config=...)``."""
    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    for fmt in cfg["formatters"].values():
        if "%(asctime)s" not in fmt["fmt"]:
            fmt["fmt"] = "%(asctime)s " + fmt["fmt"]
    return cfg
