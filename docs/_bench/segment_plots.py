#!/usr/bin/env python3
"""Render C-library-inferred PulSeg segments used by the representation page.

The figures intentionally use ``Sequence.segments`` instead of a guessed TR
slice.  Each real (non-pure-delay) segment is resolved to the instance
carrying the greatest gradient energy, exactly as the pulse-generation/safety
path does, and *every* real segment of a case is plotted -- not just the
first -- since the point of the figure set is to show which zoo sequences
the C-library segmenter actually splits (MPRAGE, fat-sat EPI) versus which
it leaves as one segment because it never finds a legal zero-gradient cut
(GRE, plain EPI, FSE).

Run from the repository root::

    python docs/_bench/segment_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pulserver.io as pio  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from pulserver import sequences  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "utils" / "expected"
OUT_DIR = REPO_ROOT / "docs" / "explanations" / "assets" / "segments"

# name -> Sequence factory
CASES = {
    "gre_2d": lambda: pio.read(FIXTURES / "gre_2d_1sl_1avg.seq", system=pp.Opts()),
    "epi_2d": lambda: pio.read(FIXTURES / "epi_2d_1sl_1avg.seq", system=pp.Opts()),
    "epi_2d_fatsat": lambda: sequences.design_epi_2d(nx=32, ny=16, etl=16, fat_sat=True),
    "fse_2d": lambda: pio.read(FIXTURES / "fse_2d_1sl_1avg.seq", system=pp.Opts()),
    "mprage_2d": lambda: pio.read(FIXTURES / "mprage_2d_1sl_1avg.seq", system=pp.Opts()),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, factory in CASES.items():
        sequence = factory()
        real_segments = [s for s in sequence.segments if not s.pure_delay]
        print(f"{name}: {len(real_segments)} real segment(s)")
        for segment in real_segments:
            sequence.plot(segment=segment.index, stacked=True)
            fig = plt.gcf()
            fig.set_size_inches(9.5, 7.0)
            fig.suptitle(
                f"{name}: C-inferred segment {segment.index} "
                f"({segment.num_blocks} blocks, {segment.duration * 1e3:.2f} ms)",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            path = OUT_DIR / f"{name}_segment_{segment.index}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(" ", path)


if __name__ == "__main__":
    main()
