#!/usr/bin/env python
"""Regenerate the paired text/binary fixtures used by test_pulseq_binary.c.

The test asks one question: does the C binary reader build the same
``pulseq_file`` the C text reader builds?  Answering it needs the *same*
sequence stored both ways, which is why the fixtures come in pairs rather
than the test reusing an existing ``.seq``.  Reading a ``.seq`` and writing
it back is not byte-stable (ids are reassigned, ``TotalDuration`` is
recomputed), so comparing a stock fixture against a binary derived from it
would be comparing two different sequences.

Usage (from the repository root)::

    python tests/utils/make_binary_fixtures.py

Fixtures land in ``tests/utils/expected/binary/`` and are checked in; this
script only needs re-running when the writers change.
"""

from __future__ import annotations

import pathlib
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: One fixture per feature the binary format has its own encoding for.
#: Sources are the synthetic corpus (tests/utils/expected) and, for the
#: label-heavy pairs, the zoo corpus (tests/python/fixtures).
SOURCES = [
    ("basic", "tests/utils/expected/00_basic_rfstat.seq"),  # rf, shapes
    (
        "arbgrad",
        "tests/utils/expected/05_ok_extended_with_delay.seq",
    ),  # arbitrary gradients
    (
        "rotations",
        "tests/utils/expected/09_fail_rot_identity.seq",
    ),  # rotation quaternions
    ("rfshims", "tests/utils/expected/07_rfstat_cp_8ch_180.seq"),  # pTx shim vectors
    ("labelset", "tests/python/fixtures/gre_2d.seq"),  # LABELSET + custom names
    ("labelinc", "tests/python/fixtures/epi_2d_main.seq"),  # LABELINC + triggers
]

OUT_DIR = ROOT / "tests" / "utils" / "expected" / "binary"


def main() -> int:
    import pulserver.pypulseq as pp

    OUT_DIR.mkdir(exist_ok=True)
    for name, source in SOURCES:
        path = ROOT / source
        if not path.exists():
            print(f"SKIP {name}: {source} not found")
            continue
        seq = pp.Sequence()
        seq.read(str(path))
        seq.write(OUT_DIR / f"{name}.seq")
        seq.write_binary(OUT_DIR / f"{name}.bin")
        text_size = (OUT_DIR / f"{name}.seq").stat().st_size
        bin_size = (OUT_DIR / f"{name}.bin").stat().st_size
        print(f"{name:12s} from {source:44s} text {text_size:8d}  binary {bin_size:8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
