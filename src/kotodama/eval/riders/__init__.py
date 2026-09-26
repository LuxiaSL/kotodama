"""The rider grid: the P0.6 four-register entity-state binding battery.

Judge-free and logprob-only. ``items`` generates N-boxes × M-updates binding
items (plus difficulty-0 gate items) and renders them in four registers (code,
narrative, screenplay, chat); ``grid`` scores them against a live server's
``/logprobs``; ``aggregate`` builds the gate + margin/rank/accuracy tables;
``smoothing`` reads the attractor-pull index off the same grids.

CLI: ``python -m kotodama.eval.riders {gen,verify-names,selftest,run,aggregate,smoothing}``.
"""

# Submodule names (items, grid, aggregate, smoothing) are NOT shadowed by
# re-exported functions: ``from kotodama.eval.riders import aggregate`` is the module.
from kotodama.eval.riders import aggregate, grid, items, smoothing
from kotodama.eval.riders.aggregate import cc_acc, write_tables
from kotodama.eval.riders.grid import default_endpoint, run_grid, score_row
from kotodama.eval.riders.items import (RENDERERS, Item, build_prompt, emit, generate,
                                        load_items)
from kotodama.eval.riders.smoothing import smoothing_index

__all__ = [
    "RENDERERS",
    "Item",
    "aggregate",
    "grid",
    "items",
    "smoothing",
    "build_prompt",
    "cc_acc",
    "default_endpoint",
    "emit",
    "generate",
    "load_items",
    "run_grid",
    "score_row",
    "smoothing_index",
    "write_tables",
]
