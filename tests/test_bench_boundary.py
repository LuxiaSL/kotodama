"""The benchmark's package boundary and its registry.

The benchmark packages import nothing from kotodama outside themselves except through ``kotodama.eval.bench.host``,
so they can leave this repository with only that module replaced. Every pinned battery has a registry row, and every
registry row names real batteries and constructs.
"""

from __future__ import annotations

import ast
from pathlib import Path

from kotodama.eval.bench import registry as rg
from kotodama.eval.binding import batteries as bt
from kotodama.eval.bindprobes.items import CONDITIONS as BP_CONDITIONS

SRC = Path(__file__).resolve().parents[1] / "src" / "kotodama"
BENCH = ("kotodama.eval.bench", "kotodama.eval.bindprobes", "kotodama.eval.binding", "kotodama.eval.riders")
HOST = "kotodama.eval.bench.host"


def _module(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _imports(path: Path) -> list[tuple[int, str]]:
    mod = _module(path)
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    out = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:   # relative: resolve against this module's package
                base = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                out.append((node.lineno, ".".join(base + ([node.module] if node.module else []))))
            elif node.module:
                out.append((node.lineno, node.module))
    return out


def _inside(name: str) -> bool:
    return any(name == b or name.startswith(b + ".") for b in BENCH)


def test_bench_reaches_kotodama_only_through_the_host():
    files = [f for b in BENCH for f in (SRC / Path(*b.split(".")[1:])).rglob("*.py")]
    assert files
    leaks = [f"{_module(f)}:{line} imports {name}" for f in files if _module(f) != HOST
             for line, name in _imports(f)
             if (name == "kotodama" or name.startswith("kotodama.")) and not _inside(name)]
    assert not leaks, "the benchmark imports kotodama outside its host module:\n" + "\n".join(leaks)


def test_every_pinned_battery_has_a_registry_row():
    carried = {b for e in rg.REGISTRY.values() for b in e.batteries}
    assert set(bt.BATTERIES) <= carried, f"no registry row: {sorted(set(bt.BATTERIES) - carried)}"
    assert carried <= set(bt.BATTERIES), f"unknown batteries in the registry: {sorted(carried - set(bt.BATTERIES))}"
    assert all(rg.evals_for(b) for b in bt.BATTERIES)


def test_registry_rows_are_well_formed():
    for e in rg.REGISTRY.values():
        assert e.construct in rg.CONSTRUCTS, e.name
        assert e.recipe and e.clean, e.name
        if e.source == "bindprobes":
            assert set(e.clean) <= set(BP_CONDITIONS), e.name
            assert all(bt.BATTERIES[b].kind == "bindprobes" for b in e.batteries), e.name
        elif e.source == "binding":
            assert all(bt.BATTERIES[b].kind == "rows" for b in e.batteries), e.name
        else:
            assert not e.batteries, e.name
        for f in e.flags:
            assert f.reading and f.finding and f.panel and f.test, e.name
    assert rg.REGISTRY["riders"].source == "riders"


def test_bp_v32_recipe_names_its_battery_families():
    assert set(rg.BP_V32_RECIPE["per_cell"]) == set(bt.BATTERIES["bp-v3.2"].families)
