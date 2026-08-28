#!/usr/bin/env python3
"""Representative-TR waveform and PNS figures for the safety explanations.

Documentation-only tooling: produces the per-sequence figures embedded in
``explanations/safety/pns.md`` and ``explanations/safety/mechanical_resonance.md``.
Not part of the shipped package.

Everything is drawn through the public ``pulserver.pypulseq.read`` /
``pulserver.pypulseq.Sequence`` API, so the figures are what a sequence author
sees from ``Sequence.plot()`` and ``Sequence.calculate_pns()`` while writing a
sequence.

Both nerve models are drawn the same way because both are asked the same way:
``calculate_pns(hardware, tr=...)`` picks Irnich or SAFE off the shape of
``hardware`` and draws either through upstream PyPulseq's ``safe_plot``, with
the 100 % threshold and the 80 % margin marked on top. Nothing here styles a
plot itself.

Usage:
    <venv>/bin/python docs/_bench/waveform_plots.py
    <venv>/bin/python docs/_bench/waveform_plots.py --only=pns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from pypulseq.utils.safe_pns_prediction import safe_example_hw  # noqa: E402

# House style: a figure has to be legible at the width a manual page gives it.
plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 13.0,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "python" / "fixtures"
OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets"

# Representative Irnich constants, the set examples/c/example_startup.c passes:
# a generic body-gradient point, not any particular scanner's configuration.
# Rheobase is in T/m/s, the unit of the slew waveform the model is handed.
IRNICH_HW = {"chronaxie_us": 360.0, "rheobase": 20.0, "alpha": 0.333}

# Upstream PyPulseq's own bundled coefficients, documented there as "EXAMPLE
# scanner hardware (not a real scanner)". Not fabricated here, and not any
# particular system's SAFE table.
SAFE_HW = safe_example_hw()

#: ``fixture -> which TR to draw``. The worst case is what the check judges;
#: the figures show it so the picture and the verdict are the same window.
CORPUS = ["gre_2d", "epi_2d", "fse_2d", "mprage_3d", "bssfp_2d"]

#: One pairing is enough to show what the other nerve model makes of the same
#: TR, and GRE's isolated events are the clearest baseline for it.
SAFE_CORPUS = ["gre_2d"]

TR = "worst_case"


def load(name: str) -> pp.Sequence:
    return pp.read(FIXTURES / f"{name}.seq")


def save(fig, subdir: str, stem: str) -> Path:
    out = OUT_DIR / subdir / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close("all")
    return out


def plot_tr_waveform(name: str, seq: pp.Sequence) -> Path:
    seq.plot(tr=TR, stacked=True)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 11)
    fig.suptitle(f"{name}  --  worst-case TR", fontsize=13.5)
    return save(fig, "representative_tr", f"{name}_tr")


def plot_pns(name: str, seq: pp.Sequence, hardware, subtitle: str, stem: str) -> Path:
    """One PNS figure, drawn by ``calculate_pns`` itself.

    It leaves two figures behind -- the TR's gradient trace, then the
    stimulation panel -- and the second is the one the docs embed.
    """
    seq.calculate_pns(hardware, tr=TR)
    fig = plt.gcf()
    fig.suptitle(f"{name}  --  {subtitle}, worst-case TR", fontsize=13.5)
    return save(fig, "pns_safety", stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", "pns", "tr"), default="all")
    args = parser.parse_args()

    for name in CORPUS:
        seq = load(name)
        written = []

        if args.only in ("all", "tr"):
            written.append(plot_tr_waveform(name, seq))
        if args.only in ("all", "pns"):
            written.append(plot_pns(name, seq, IRNICH_HW, "Irnich PNS", f"{name}_pns"))
            if name in SAFE_CORPUS:
                written.append(
                    plot_pns(
                        name,
                        seq,
                        SAFE_HW,
                        "SAFE PNS (PyPulseq example coefficients -- not a real scanner)",
                        f"{name}_pns_safe",
                    )
                )

        paths = "  ".join(str(path.relative_to(OUT_DIR.parent)) for path in written)
        print(f"{name:12} -> {paths}")


if __name__ == "__main__":
    main()
