#!/usr/bin/env python3
"""The figures for ``explanations/performance/mechanical_resonance``.

Documentation-only tooling. Everything on that page is drawn here, in the
same units the vendor tables and the safety pages use -- A_eq in mT/m.

Three pictures:

``epi_comb``
    The page's opening claim: a blipped echo train is a comb, and the verdict
    is those teeth read against the guarded bands -- flagged where a band
    lands on one, clear where it does not.

The spiral pictures come from one build of the sequence each way:

``spiral_bound``
    The arms, and the A_eq line spectrum of each of them under the canonical
    TR's. Every interleaf sits below the line the gate judges, at every
    harmonic, which is what makes one window's verdict a verdict about the
    scan.
``spiral_encodings``
    The same scan with its arms turned by a rotation extension and with them
    written out as their own waveforms. A scanner plays the identical field
    either way and the analysis reads it identically, arm by arm.

Everything is read through the public ``tr=`` axis of
:meth:`~pulserver.pypulseq.Sequence.calculate_gradient_spectrum`, so the
figures are what a sequence author sees, and the bold line is the object
``pulseg_check_safety`` decides on.

Usage:
    <venv>/bin/python docs/_bench/mechres_multishot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _figures import INK, MUTED, SERIES, _style

OUT_DIR = Path(__file__).resolve().parents[1] / "explanations" / "assets"

#: Four arms: enough that the interleaves are visibly different waveforms,
#: few enough that each one can be drawn and still be read.
ARMS = 4
MATRIX = 64
TR_S = 30e-3

#: Two guarded bands over the range the arms drive, in the shape a vendor
#: lockout table has: (low, high, tolerated amplitude in Hz/m). The tolerance
#: is at readout scale, where a real one is -- 0.3 G/cm, the unit an ESP table
#: states, and the same order as the hardware-anchored floor the engine falls
#: back on for a zero-tolerance row (0.08 x G_max, 3.2 mT/m at this
#: sequence's 40 mT/m). A tolerance well under that would flag every readout
#: ever written, which is a statement about the number chosen and not about
#: the sequence.
GAMMA_HZ_PER_MT_PER_M = 42.576e3
TOLERANCE_MT_PER_M = 3.0
BANDS = [
    (550.0, 700.0, TOLERANCE_MT_PER_M * GAMMA_HZ_PER_MT_PER_M),
    (1150.0, 1300.0, TOLERANCE_MT_PER_M * GAMMA_HZ_PER_MT_PER_M),
]

MAX_FREQ_HZ = 2000.0

#: The EPI hero: the checked-in fixture, read as a scanner would read it.
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "python" / "fixtures"

#: Zero-tolerance bands for the EPI figure, one on the echo-train tooth and
#: one in the quiet between teeth, so the same comb shows a flag and a pass.
#: Zero tolerance means the verdict falls back to the hardware-anchored floor,
#: 0.08 x G_max.
EPI_BANDS = [(1200.0, 1300.0, 0.0), (1550.0, 1650.0, 0.0)]
EPI_MAX_FREQ_HZ = 3000.0


def build(rotated: bool):
    from pulserver.app import gre_spiral2D_sequence

    return gre_spiral2D_sequence.main(
        plot=False,
        write_seq=False,
        n_x=MATRIX,
        n_arms=ARMS,
        n_dummy=0,
        tr=TR_S,
        # Golden angles, so no two arms are a sign flip or a swap of
        # another and each is genuinely its own waveform.
        angle_scheme="golden",
        readout_bandwidth_hz=125e3,
        use_rotation_ext=rotated,
    )


def lines(sequence, tr):
    """``(frequencies, A_eq per axis in mT/m)`` for one TR of a sequence."""
    resonances = sequence.calculate_gradient_spectrum(
        MAX_FREQ_HZ,
        plot=False,
        tr=tr,
        resonance_lines=True,
        bands=BANDS,
        compat=False,
    ).resonance_lines
    return resonances.line_freqs, resonances.line_a_eq / GAMMA_HZ_PER_MT_PER_M


def arm_waveforms(sequence):
    """Each interleaf's in-plane gradient, in mT/m against ms."""
    drawn = []
    for arm in range(sequence.num_trs):
        waveform = sequence._structure_for("figure").waveform(arm)
        channels = waveform.waveforms()
        drawn.append(
            (
                channels[0][0] * 1e3,
                channels[0][1] / GAMMA_HZ_PER_MT_PER_M,
                channels[1][1] / GAMMA_HZ_PER_MT_PER_M,
            )
        )
    return drawn


def _decade_ticks(axis) -> None:
    """Readable labels on a log axis whose data spans less than two decades."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    low, high = axis.get_ylim()
    ticks = [t for t in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) if low <= t <= high]
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _bands(axis) -> None:
    for low, high, tolerance in BANDS:
        axis.axvspan(low, high, color="#e34948", alpha=0.08, lw=0)
        axis.plot(
            [low, high],
            [tolerance / GAMMA_HZ_PER_MT_PER_M] * 2,
            color="#e34948",
            lw=1.2,
            ls=(0, (4, 2)),
            zorder=5,
        )


def _shade(index: int, total: int) -> str:
    """One hue per interleaf: an identity, not an order."""
    del total
    return SERIES[index % len(SERIES)]


def epi_comb() -> Path:
    """The echo-train comb, read against two guarded bands."""
    import pulserver.io as pio

    sequence = pio.read(FIXTURES / "epi_2d.seq")
    floor = 0.08 * float(sequence.system.max_grad) / GAMMA_HZ_PER_MT_PER_M
    resonances = sequence.calculate_gradient_spectrum(
        EPI_MAX_FREQ_HZ,
        plot=False,
        tr="worst_case",
        resonance_lines=True,
        bands=EPI_BANDS,
        compat=False,
    ).resonance_lines

    figure, axis = plt.subplots(figsize=(9.0, 3.8))
    amps = resonances.line_a_eq / GAMMA_HZ_PER_MT_PER_M
    for channel, label in enumerate(("$G_x$ (the echo train)", "$G_y$", "$G_z$")):
        axis.plot(
            resonances.line_freqs,
            amps[:, channel],
            color=SERIES[channel],
            lw=1.2 if channel == 0 else 0.9,
            alpha=1.0 if channel == 0 else 0.75,
            label=label,
        )

    for low, high, _ in EPI_BANDS:
        axis.axvspan(low, high, color="#e34948", alpha=0.08, lw=0)
        axis.plot([low, high], [floor] * 2, color="#e34948", lw=1.2, ls=(0, (4, 2)), zorder=5)
    flagged = resonances.violations
    candidates = resonances.candidate_a_eq.max(axis=1) / GAMMA_HZ_PER_MT_PER_M
    for real, color, label in ((~flagged, SERIES[2], "in-band line, under the floor"),
                               (flagged, "#e34948", "in-band line, flagged")):
        if np.any(real):
            axis.plot(resonances.candidate_freqs[real], candidates[real],
                      "o", color=color, ms=5, mew=0, lw=0, label=label, zorder=6)

    _style(axis, "the echo-train comb, against two zero-tolerance bands")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("$A_{eq}$ (mT/m)")
    axis.set_xlim(0, EPI_MAX_FREQ_HZ)
    axis.set_ylim(0, None)
    legend = axis.legend(fontsize=8, frameon=False, loc="upper right", ncols=2)
    for text in legend.get_texts():
        text.set_color(MUTED)

    return save(figure, "epi_comb")


def spiral_bound(sequence) -> Path:
    """Every interleaf's lines, under the one the gate judges."""
    figure = plt.figure(figsize=(9.0, 6.4))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.55), hspace=0.45, wspace=0.22)

    drawn = arm_waveforms(sequence)
    gradients = figure.add_subplot(grid[0, 0])
    for arm, (time_ms, gx, _) in enumerate(drawn):
        gradients.plot(time_ms, gx, color=_shade(arm, len(drawn)), lw=1.0)
    _style(gradients, "the arms, on the readout axis")
    gradients.set_xlabel("time (ms)")
    gradients.set_ylabel("$G_x$ (mT/m)")
    # A window into the readout rather than the whole TR: what matters is
    # that the arms are different waveforms, and at full width they are a
    # solid block of oscillation.
    onset = float(drawn[0][0][np.argmax(np.abs(drawn[0][1]) > 0.05)])
    gradients.set_xlim(onset, onset + 2.0)

    trajectory = figure.add_subplot(grid[0, 1])
    for arm, (_, gx, gy) in enumerate(drawn):
        trajectory.plot(
            np.cumsum(gx), np.cumsum(gy), color=_shade(arm, len(drawn)), lw=0.9
        )
    trajectory.set_aspect("equal", adjustable="datalim")
    _style(trajectory, "and where each of them goes")
    trajectory.set_xticks([])
    trajectory.set_yticks([])
    for side in ("left", "bottom"):
        trajectory.spines[side].set_visible(False)

    spectrum = figure.add_subplot(grid[1, :])
    freqs, bound = lines(sequence, "worst_case")
    for arm in range(sequence.num_trs):
        _, played = lines(sequence, arm)
        spectrum.plot(
            freqs,
            played.max(axis=1),
            color=_shade(arm, sequence.num_trs),
            lw=1.0,
            label=f"interleaf {arm}",
        )
    spectrum.plot(
        freqs,
        bound.max(axis=1),
        color=INK,
        lw=1.8,
        label="canonical TR (what the gate judges)",
    )
    _bands(spectrum)
    _style(spectrum, "$A_{eq}$ at the harmonics of the TR")
    spectrum.set_yscale("log")
    spectrum.set_xlabel("frequency (Hz)")
    spectrum.set_ylabel("$A_{eq}$ (mT/m)")
    spectrum.set_xlim(0, MAX_FREQ_HZ)
    _decade_ticks(spectrum)
    legend = spectrum.legend(
        fontsize=8,
        frameon=False,
        ncols=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
    )
    for text in legend.get_texts():
        text.set_color(MUTED)

    return save(figure, "spiral_bound")


def spiral_encodings(rotated_sequence, explicit_sequence) -> Path:
    """One scan, two encodings, arm by arm."""
    figure, axis = plt.subplots(figsize=(9.0, 3.6))

    for arm in range(rotated_sequence.num_trs):
        freqs, turned = lines(rotated_sequence, arm)
        _, written = lines(explicit_sequence, arm)
        axis.plot(
            freqs,
            turned.max(axis=1),
            color=_shade(arm, rotated_sequence.num_trs),
            lw=1.2,
        )
        axis.plot(
            freqs,
            written.max(axis=1),
            color=INK,
            lw=0,
            marker="o",
            ms=2.4,
            mfc="none",
            mew=0.7,
        )

    _bands(axis)
    _style(axis, "the same four arms, encoded both ways")
    axis.set_yscale("log")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("$A_{eq}$ (mT/m)")
    axis.set_xlim(0, MAX_FREQ_HZ)
    _decade_ticks(axis)
    handles = [
        Line2D([], [], color=MUTED, lw=1.2),
        Line2D([], [], color=INK, lw=0, marker="o", ms=2.4, mfc="none", mew=0.7),
    ]
    legend = axis.legend(
        handles,
        ["each arm, turned by a rotation", "the same arm, written out"],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    for text in legend.get_texts():
        text.set_color(MUTED)

    return save(figure, "spiral_encodings")


def save(figure, stem: str) -> Path:
    out = OUT_DIR / "mechanical_resonance" / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.patch.set_facecolor("white")
    figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    rotated = build(rotated=True)
    explicit = build(rotated=False)

    written = [epi_comb(), spiral_bound(explicit), spiral_encodings(rotated, explicit)]
    for path in written:
        print(path.relative_to(OUT_DIR.parent))


if __name__ == "__main__":
    main()
