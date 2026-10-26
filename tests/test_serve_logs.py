"""Server request (access) lines carry a timestamp: the uvicorn log config the servers pass leads every formatter
with the time, and it applies as a logging dictConfig."""
from __future__ import annotations

import logging
import logging.config
import re

from kotodama.serve.logs import uvicorn_log_config

STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ")


def test_every_formatter_leads_with_the_time():
    cfg = uvicorn_log_config()
    assert cfg["formatters"], "no formatters"
    for fmt in cfg["formatters"].values():
        assert fmt["fmt"].startswith("%(asctime)s ")
    assert uvicorn_log_config()["formatters"] == cfg["formatters"]          # idempotent: no double stamp


def test_access_line_is_stamped():
    logging.config.dictConfig(uvicorn_log_config())
    handler = logging.getLogger("uvicorn.access").handlers[0]
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ("10.0.0.2:5767", "POST", "/v1/completions", "1.1", 200), None)
    line = handler.formatter.format(record)
    assert STAMP.match(line), line
    assert '"POST /v1/completions HTTP/1.1" 200' in line
