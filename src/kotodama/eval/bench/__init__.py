"""The binding and fluid-ICL benchmark: its registry and its one door into the rest of kotodama.

The benchmark is four packages: this one (:mod:`.registry`, :mod:`.host`), :mod:`kotodama.eval.bindprobes`,
:mod:`kotodama.eval.binding` and :mod:`kotodama.eval.riders`. They import kotodama only through :mod:`.host`
(``tests/test_bench_boundary.py``). ``README.md`` beside this file maps the pieces.
"""
