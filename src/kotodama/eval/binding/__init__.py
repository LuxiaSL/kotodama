"""``python -m kotodama.eval.binding``: the binding benchmark's suite CLI, with kotodama checkpoints loadable.

The suite lives in the binding benchmark (``binding_bench.binding``, kotodama's ``bench`` extra); this package only
registers the ``koto:`` loader (:mod:`kotodama.bench_plugin`) before handing over, so launchers that name this module
keep working.
"""
