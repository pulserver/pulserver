"""The two figures behind the performance page's calibration and cost sections.

``threshold_ladder.png``  the measured in-band drive of the shipped plugins
                          across realistic protocols, against the threshold
``basis_cost.png``        gate cost against each of its three candidate drivers,
                          and against a transform of the whole timeline

Vendor-side detail is deliberately absent: which product sequences carry a
frequency lockout, and on what parameter, is the vendor's business. What the
calibration needs from that inspection is one number, and that number is here.

    python docs/_bench/mechres_calibration.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pulserver.pypulseq as pp  # noqa: E402
from _figures import FAINT, INK, MUTED, SERIES, _style  # noqa: E402
from pulserver._ext.pulseg import _calc_mech_resonances, _check_safety  # noqa: E402

# House style: a figure has to be legible at the width a manual page gives it.
plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 13.0,
    }
)


ASSETS = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "mechanical_resonance"
GAMMA = 42.576e3
ACTUAL = 2

#: ``SA_AEQ_POLICY_MT_PER_M``. Drawn, not recomputed.
POLICY = 7.5

#: Every band in every vendor table inspected falls in here.
TERRITORY = (515.0, 1650.0)

#: The frame every spectrum is drawn against, so a quiet one looks quiet
#: instead of being autoscaled into structure. 1.05x the loudest readout comb
#: measured, rounded.
BASE_AMP = 16.0

SYSTEMS = {
    "a": pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s"),
    "b": pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s"),
}


def _spectrum(sequence, samples=8):
    """Max over sampled TR instances of the exact per-instance A_eq, worst axis."""
    structure = sequence._structure_for("bound")
    picks = sorted(
        set(np.linspace(0, structure.num_trs - 1, min(samples, structure.num_trs)).astype(int))
    )
    freqs = amps = None
    for index in picks:
        spectra = _calc_mech_resonances(
            structure.collection,
            0,
            int(index),
            ACTUAL,
            target_resolution_hz=1.0 / structure.tr_duration,
            max_freq_hz=3000.0,
            forbidden_bands=[(500.0, 600.0, 0.0)],
        )
        f = np.asarray(spectra["analytical_peak_freqs"], float)
        a = (
            np.max(
                np.stack(
                    [np.asarray(spectra[f"analytical_peak_amp_g{x}"], float) for x in "xyz"]
                ),
                axis=0,
            )
            / GAMMA
        )
        if amps is None:
            freqs, amps = f, a
        else:
            n = min(len(a), len(amps))
            freqs, amps = freqs[:n], np.maximum(amps[:n], a[:n])
    return freqs, amps


def _build(module, system, **kwargs):
    from importlib import import_module

    sequence = getattr(import_module("pulserver.app"), module).main(
        system=SYSTEMS[system], n_dummy=0, **kwargs
    )
    return sequence[0] if isinstance(sequence, tuple) else sequence


#: (label, module, system, kwargs). Protocols a console could prescribe.
LADDER = [
    ("bSSFP, shortest TR", "bssfp2D_sequence", "b", dict(n_x=128, tr=None)),
    ("bSSFP, shortest TR, weaker gradients", "bssfp2D_sequence", "a", dict(n_x=128, tr=None)),
    ("EPI, 64 matrix", "epi2D_sequence", "a", dict(n_x=64, n_y=64, readout_bandwidth_hz=250e3)),
    ("EPI, 96 matrix", "epi2D_sequence", "b", dict(n_x=96, n_y=96, readout_bandwidth_hz=250e3)),
    ("3D GRE, TR 6 ms", "gre3D_sequence", "b",
     dict(n_x=128, n_y=64, n_z=16, tr=None, te=None, readout_bandwidth_hz=125e3)),
    ("radial GRE, TR 10 ms", "gre_radial2D_sequence", "b", dict(n_x=128, tr=10e-3)),
    ("spiral, 8 arms", "gre_spiral2D_sequence", "a", dict(n_x=128, n_arms=8, tr=25e-3)),
    ("stack of stars, TR 10 ms", "gre_stack_of_stars3D_sequence", "b", dict(n_x=96, n_z=8, tr=10e-3)),
    ("2D GRE, TR 9 ms", "gre2D_sequence", "b",
     dict(n_x=128, n_y=96, tr=9e-3, te=4.5e-3, readout_bandwidth_hz=83e3)),
    ("MPRAGE", "mprage3D_sequence", "b", dict(n_x=96, n_y=32, n_z=16)),
    ("2D GRE, TR 100 ms", "gre2D_sequence", "b",
     dict(n_x=128, n_y=96, tr=100e-3, te=8e-3, readout_bandwidth_hz=83e3)),
    ("spin echo", "se2D_sequence", "b", dict(n_x=128, n_y=96)),
]


def threshold_ladder():
    rows = []
    for label, module, system, kwargs in LADDER:
        freqs, amps = _spectrum(_build(module, system, **kwargs))
        window = (freqs >= TERRITORY[0]) & (freqs <= TERRITORY[1])
        rows.append((label, freqs, amps, float(amps[window].max()) if window.any() else 0.0))
    rows.sort(key=lambda r: -r[3])

    figure, axes = plt.subplots(6, 2, figsize=(8.6, 11.6), dpi=170)
    figure.subplots_adjust(hspace=0.72, wspace=0.16, top=0.895, bottom=0.055,
                           left=0.08, right=0.985)
    for index, (label, freqs, amps, in_band) in enumerate(rows):
        axis = axes[index // 2, index % 2]
        drawn = freqs <= 2000.0
        loud = in_band > POLICY
        axis.axvspan(*TERRITORY, color=FAINT, alpha=0.28, lw=0, zorder=0)
        axis.axhline(POLICY, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        axis.vlines(freqs[drawn], 0.0, amps[drawn],
                    color=SERIES[1] if loud else SERIES[0], lw=0.9, zorder=2)
        axis.set_xlim(0, 2000)
        axis.set_ylim(0, max(BASE_AMP, 1.05 * float(amps[drawn].max())))
        _style(axis, label)
        axis.text(0.995, 0.94, f"{in_band:.1f} mT/m in band", transform=axis.transAxes,
                  ha="right", va="top", fontsize=10.0,
                  color=SERIES[1] if loud else MUTED)
        if index // 2 == 5:
            axis.set_xlabel("frequency (Hz)", fontsize=10.8)
        if index % 2 == 0:
            axis.set_ylabel("$A_{eq}$ (mT/m)", fontsize=10.8)
    figure.suptitle(
        "Equivalent sustained amplitude of the shipped plugins,\n"
        "across realistic protocols. Shaded: where vendor bands fall.\n"
        f"Dashed: the {POLICY} mT/m threshold. Orange: refused in band.",
        x=0.02, y=0.995, ha="left", va="top", fontsize=12.0, color=INK)
    figure.savefig(ASSETS / "threshold_ladder.png", facecolor="white")
    plt.close(figure)
    return [(label, in_band) for label, _, _, in_band in rows]


def _run_ms(collection, bands, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            _check_safety(collection, forbidden_bands=[(a, b, c * GAMMA) for a, b, c in bands])
        except Exception:
            pass
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def _gate_ms(collection, bands, repeats=3):
    """The acoustic analysis alone.

    ``pulseg_check_safety`` also runs the gradient and PNS checks, and those
    walk every block, so timing the whole entry point would attribute their
    scan-length growth to this one. Differencing against a bandless run leaves
    the acoustic part, which is the only part a band switches on.
    """
    return max(_run_ms(collection, bands, repeats) - _run_ms(collection, [], repeats), 0.0)


def basis_cost():
    band = [(600.0, 700.0, 0.0)]

    #: The arm sweep needs a band with many lines in it, or it measures the
    #: fixed cost of reading the waveforms in rather than the cost of
    #: transforming them, and the basis is not what is being varied.
    dense = [(500.0, 2500.0, 0.0)]

    scans, timeline, arms_free, arms_bound, harmonics = [], [], [], [], []
    for n_y in (32, 64, 128, 256, 512, 1024):
        sequence = _build("gre2D_sequence", "b", n_x=128, n_y=n_y, tr=15e-3)
        structure = sequence._structure_for("bound")
        scans.append((structure.num_trs, _gate_ms(structure.collection, band)))
        start = time.perf_counter()
        sequence.calculate_gradient_spectrum(3000.0, plot=False, tr=None, compat=False)
        timeline.append((structure.num_trs, 1e3 * (time.perf_counter() - start)))
    for n_arms in (4, 8, 16, 32, 64):
        for rotated, sink in ((False, arms_free), (True, arms_bound)):
            structure = _build("gre_spiral2D_sequence", "b", n_x=64, n_arms=n_arms,
                               tr=200e-3, use_rotation_ext=rotated)._structure_for("bound")
            sink.append((n_arms, _gate_ms(structure.collection, dense)))
    for tr in (20e-3, 40e-3, 80e-3, 160e-3, 320e-3):
        structure = _build("gre_spiral2D_sequence", "b", n_x=64, n_arms=16, tr=tr,
                           use_rotation_ext=False)._structure_for("bound")
        harmonics.append((100.0 * structure.tr_duration, _gate_ms(structure.collection, band)))

    figure, axes = plt.subplots(1, 3, figsize=(8.6, 4.1), dpi=170)
    figure.subplots_adjust(wspace=0.36, top=0.64, bottom=0.19, left=0.09, right=0.985)

    axis = axes[0]
    x, y = zip(*timeline)
    axis.plot(x, y, "o-", color=SERIES[1], lw=1.4, ms=4, label="a transform of the timeline")
    x, y = zip(*scans)
    axis.plot(x, y, "o-", color=SERIES[2], lw=1.4, ms=4, label="the gate")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_ylim(1e-2, 1e4)
    _style(axis, "scan length")
    axis.set_xlabel("repetitions", fontsize=10.8)
    axis.set_ylabel("ms", fontsize=10.8)
    axis.legend(frameon=False, fontsize=9.6, loc="lower left", ncol=1,
                bbox_to_anchor=(0.0, 1.14, 1.0, 0.16), mode="expand",
                borderaxespad=0.0, handlelength=1.2)

    axis = axes[1]
    for data, color, label in ((arms_free, SERIES[1], "written out"),
                               (arms_bound, SERIES[2], "one wave, turned")):
        x, y = zip(*data)
        axis.plot(x, y, "o-", color=color, lw=1.4, ms=4, label=label)
    axis.set_ylim(0, max(v for _, v in arms_free) * 1.6)
    _style(axis, "basis size")
    axis.set_xlabel("spiral arms", fontsize=10.8)
    axis.set_ylabel("gate (ms)", fontsize=10.8)
    axis.legend(frameon=False, fontsize=9.6, loc="lower left", ncol=1,
                bbox_to_anchor=(0.0, 1.14, 1.0, 0.16), mode="expand",
                borderaxespad=0.0, handlelength=1.2)

    axis = axes[2]
    x, y = zip(*harmonics)
    axis.plot(x, y, "o-", color=SERIES[3], lw=1.4, ms=4)
    axis.set_ylim(0, max(y) * 1.4)
    _style(axis, "harmonics inside the band")
    axis.set_xlabel("band width × $T_{TR}$", fontsize=10.8)
    axis.set_ylabel("gate (ms)", fontsize=10.8)

    figure.suptitle("What the gate's cost actually depends on", x=0.02, y=0.995,
                    ha="left", va="top", fontsize=12.0, color=INK)
    figure.savefig(ASSETS / "basis_cost.png", facecolor="white")
    plt.close(figure)
    return dict(scans=scans, timeline=timeline, written=arms_free, turned=arms_bound,
                harmonics=harmonics)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    ladder = threshold_ladder()
    cost = basis_cost()
    (Path(__file__).resolve().parent / "mechres_calibration.json").write_text(
        json.dumps({"ladder": ladder, "cost": cost}, indent=1)
    )
    for label, value in ladder:
        print(f"  {label:40} {value:6.2f} mT/m in band")
    print("\ncost:")
    for key, series in cost.items():
        print(f"  {key:10} " + "  ".join(f"{a:g}:{b:.2f}ms" for a, b in series))
