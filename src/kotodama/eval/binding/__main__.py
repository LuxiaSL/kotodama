"""Register kotodama's ``koto:`` loader, then run the binding benchmark's suite CLI with the same arguments."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from binding_bench.binding.__main__ import main as bench_main
    except ImportError as exc:
        raise SystemExit("python -m kotodama.eval.binding needs the binding benchmark: install kotodama[bench], "
                         f"or put the binding-bench checkout on PYTHONPATH ({exc})") from exc
    from kotodama.bench_plugin import register

    register()
    return bench_main(argv)


if __name__ == "__main__":
    sys.exit(main())
