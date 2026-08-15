#!/usr/bin/env python
"""Write the `.seq`/`.bin` fixture corpus under ``tests/python/fixtures/``.

The corpus contents and builders are defined in
``tests/python/fixture_corpus.py``; this script just runs them and reports
what it wrote. Output is deterministic, so a clean tree stays clean when
the generators have not changed.

Usage (any working directory)::

    python tests/utils/generate_fixtures.py
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    tests_python = pathlib.Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(tests_python))

    import fixture_corpus as corpus

    out = corpus.FIXTURES_DIR
    out.mkdir(exist_ok=True)

    for name, build in corpus.CORPUS.items():
        seq = build()
        seq.write(out / f"{name}.seq")
        print(f"wrote {name}.seq")
        if name in corpus.BINARY_TWINS:
            seq.write_binary(out / f"{name}.bin")
            print(f"wrote {name}.bin")

    for path in corpus.write_epi_collections(out):
        print(f"wrote {path.name}")

    for name, seq in corpus.build_parser_edges().items():
        seq.write(out / f"{name}.seq")
        print(f"wrote {name}.seq")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
