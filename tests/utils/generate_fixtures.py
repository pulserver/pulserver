#!/usr/bin/env python
"""Write every checked-in fixture: the zoo corpus and the synthetic set.

Two output trees, both fully regenerable from this one script:

* ``tests/python/fixtures/`` -- the zoo corpus (one entry per zoo slot,
  plus edges and NextSequence chains), from ``tests/python/fixture_corpus.py``.
* ``tests/utils/expected/`` -- the synthetic C-test corpus (hand-specified
  safety/structure cases plus builder-produced specials), from
  ``tests/utils/synthetic_fixtures.py``.

Output is deterministic, so a clean tree stays clean when the generators
have not changed.  The paired text/binary fixtures under
``tests/utils/expected/binary/`` are rebuilt by
``tests/utils/make_binary_fixtures.py`` (run it after this script when the
writers change).

Usage (any working directory)::

    python tests/utils/generate_fixtures.py
"""

from __future__ import annotations

import pathlib
import sys


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[0] / "python"))
    sys.path.insert(0, str(here))

    import fixture_corpus as corpus
    import synthetic_fixtures as synthetic

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

    expected = here / "expected"
    for name in synthetic.write_all(expected, out):
        print(f"wrote expected/{name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
