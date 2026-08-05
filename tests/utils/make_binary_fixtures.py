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
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')

from fastseq.binary import write_binary  # noqa: E402
from fastseq.sequence import read, write  # noqa: E402

#: One fixture per feature the binary format has its own encoding for.
SOURCES = [
    ('basic', '00_basic_rfstat.seq'),                  # rf, trap, shapes
    ('arbgrad', '05_ok_extended_with_delay.seq'),      # arbitrary gradients
    ('rotations', '09_fail_rot_identity.seq'),         # rotation quaternions
    ('rfshims', '07_rfstat_cp_8ch_180.seq'),           # pTx shim vectors
    ('labelset', 'gre_2d_1sl_3avg.seq'),               # LABELSET + chains
    ('labelinc', 'epi_2d_3sl_1avg.seq'),               # LABELINC + triggers
]

SOURCE_DIR = ROOT / 'tests' / 'utils' / 'expected'
OUT_DIR = SOURCE_DIR / 'binary'


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    for name, source in SOURCES:
        path = SOURCE_DIR / source
        if not path.exists():
            print(f'SKIP {name}: {source} not found')
            continue
        seq = read(path)
        write(seq, OUT_DIR / f'{name}.seq')
        write_binary(seq, OUT_DIR / f'{name}.bin')
        text_size = (OUT_DIR / f'{name}.seq').stat().st_size
        bin_size = (OUT_DIR / f'{name}.bin').stat().st_size
        print(f'{name:12s} from {source:44s} text {text_size:8d}  binary {bin_size:8d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
