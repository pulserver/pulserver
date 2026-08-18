"""Zoo-wide evidence for the validation page: throughput, canonical TR, bounds.

Documentation-only tooling, like everything else in this directory. For every
sequence family in :mod:`pulserver.app` it measures

* design time at three problem sizes, and text serialisation plus C-side
  conversion (parse, structure detection, consistency) at the largest;
* the worst-case canonical TR against the actual TR instances: the PNS peak
  of the envelope next to the PNS peak of each real repetition, and the
  mechanical-resonance verdict of the envelope, on a synthetic zero-tolerance
  band table (performance and bounding evidence, not a safety claim);

and draws, for a subset of genuinely multishot families, the canonical-TR
waveform, the per-instance PNS bound chart, and the drive spectrum.

Run it as::

    python docs/_bench/zoo_report.py                # everything
    python docs/_bench/zoo_report.py --only=gre2D   # one family
    python docs/_bench/zoo_report.py --no-figures   # table only

Results land in ``docs/_bench/zoo_report.json`` and the figures in
``docs/explanations/assets/zoo/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = Path(__file__).resolve().parents[1]
ASSETS = DOCS / "explanations" / "assets" / "zoo"
RESULTS = Path(__file__).resolve().parent / "zoo_report.json"

#: Representative Irnich constants -- a generic body-gradient point, not any
#: particular scanner's configuration.
IRNICH = {"chronaxie_us": 360.0, "rheobase": 4.25e8, "alpha": 0.333}

#: Synthetic zero-tolerance forbidden bands (Hz). Chosen to sit where echo
#: trains and balanced readouts actually put energy, so verdicts are
#: exercised in both directions across the zoo.
BANDS = [(550.0, 650.0, 0.0), (1100.0, 1250.0, 0.0)]

#: How many actual TR instances to evaluate against the worst-case envelope.
MAX_INSTANCES = 12

#: Per-family problem sizes. Every plugin also gets plot=False,
#: write_seq=False. "L" is sized to keep the whole sweep in minutes while
#: still reaching protocol-scale block counts on the 3D families.
SIZES = {
    "gre2D": [
        {"n_x": 64, "n_y": 64, "n_slices": 1},
        {"n_x": 128, "n_y": 128, "n_slices": 8},
        {"n_x": 256, "n_y": 256, "n_slices": 16},
    ],
    "gre3D": [
        {"n_x": 64, "n_y": 64, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 32},
        {"n_x": 256, "n_y": 256, "n_z": 64},
    ],
    "se2D": [
        {"n_x": 64, "n_y": 64, "n_slices": 1},
        {"n_x": 128, "n_y": 128, "n_slices": 4},
        {"n_x": 256, "n_y": 256, "n_slices": 8},
    ],
    "se3D": [
        {"n_x": 64, "n_y": 64, "n_z": 8},
        {"n_x": 128, "n_y": 128, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 32},
    ],
    "fse2D": [
        {"n_x": 128, "n_y": 128, "n_slices": 1},
        {"n_x": 128, "n_y": 128, "n_slices": 4},
        {"n_x": 256, "n_y": 256, "n_slices": 8},
    ],
    "fse3D": [
        {"n_x": 128, "n_y": 128, "n_z": 8},
        {"n_x": 128, "n_y": 128, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 48},
    ],
    "epi2D": [
        {"n_x": 64, "n_y": 64, "n_slices": 1},
        {"n_x": 96, "n_y": 96, "n_slices": 8},
        {"n_x": 128, "n_y": 128, "n_slices": 16},
    ],
    "epi3D": [
        {"n_x": 64, "n_y": 64, "n_z": 8},
        {"n_x": 96, "n_y": 96, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 32},
    ],
    "bssfp2D": [
        {"n_x": 128, "n_y": 128, "n_slices": 1},
        {"n_x": 128, "n_y": 128, "n_slices": 4},
        {"n_x": 256, "n_y": 256, "n_slices": 8},
    ],
    "bssfp3D": [
        {"n_x": 64, "n_y": 64, "n_z": 8},
        {"n_x": 128, "n_y": 128, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 48},
    ],
    "gre_multiecho2D": [
        {"n_x": 64, "n_y": 64, "n_slices": 1},
        {"n_x": 128, "n_y": 128, "n_slices": 4},
        {"n_x": 256, "n_y": 256, "n_slices": 8},
    ],
    "gre_multiecho3D": [
        {"n_x": 64, "n_y": 64, "n_z": 8},
        {"n_x": 128, "n_y": 128, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 48},
    ],
    "mprage3D": [
        {"n_x": 64, "n_y": 64, "n_z": 16},
        {"n_x": 128, "n_y": 128, "n_z": 64},
        {"n_x": 192, "n_y": 192, "n_z": 128},
    ],
    "mprage_stack_of_spirals3D": [
        {"n_x": 64, "n_z": 16, "n_arms": 64, "etl": 64},
        {"n_x": 128, "n_z": 32, "n_arms": 128, "etl": 128},
        {"n_x": 128, "n_z": 64, "n_arms": 256, "etl": 256, "ti": 8.0, "tr_outer": 12.0},
    ],
    "gre_radial2D": [
        {"n_x": 64, "n_spokes": 101, "n_slices": 1},
        {"n_x": 128, "n_spokes": 201, "n_slices": 4},
        {"n_x": 256, "n_spokes": 403, "n_slices": 8},
    ],
    "gre_spiral2D": [
        {"n_x": 64, "n_arms": 8, "n_slices": 1},
        {"n_x": 128, "n_arms": 16, "n_slices": 4},
        {"n_x": 256, "n_arms": 32, "n_slices": 8},
    ],
    "gre_stack_of_spirals3D": [
        {"n_x": 64, "n_arms": 8, "n_z": 8},
        {"n_x": 128, "n_arms": 16, "n_z": 16},
        {"n_x": 128, "n_arms": 32, "n_z": 48},
    ],
    "gre_stack_of_stars3D": [
        {"n_x": 64, "n_spokes": 101, "n_z": 8},
        {"n_x": 128, "n_spokes": 201, "n_z": 16},
        {"n_x": 128, "n_spokes": 403, "n_z": 32},
    ],
    "se_propeller2D": [
        {"n_x": 64, "blade_width": 16, "n_blades": 8, "n_slices": 1},
        {"n_x": 128, "blade_width": 24, "n_blades": 12, "n_slices": 2},
        {"n_x": 256, "blade_width": 32, "n_blades": 16, "n_slices": 4, "te": 0.12},
    ],
    "zte3D": [
        {"n_x": 32},
        {"n_x": 48},
        {"n_x": 64},
    ],
}

#: Families whose per-shot waveforms genuinely differ or rotate -- the ones
#: the multishot-bound figures are drawn for.
MULTISHOT_FIGURES = [
    "gre_spiral2D",
    "gre_radial2D",
    "se_propeller2D",
    "fse3D",
    "mprage_stack_of_spirals3D",
]


def build(name: str, kwargs: dict):
    import pulserver.app as app

    fn = getattr(app, name + "_sequence")
    return fn(plot=False, write_seq=False, **kwargs)


def convert_with_c(seq) -> dict:
    """Serialise, then time the C-side parse + conversion of the result."""
    from pulserver._ext.pulseg import _PulseqCollection
    from pulserver.pypulseq import Opts

    system = Opts()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zoo.seq"
        t0 = time.perf_counter()
        seq.write(str(path))
        t_write = time.perf_counter() - t0
        payload = path.read_bytes()
        t0 = time.perf_counter()
        _PulseqCollection(
            [payload],
            float(system.gamma),
            float(system.B0),
            float(system.max_grad),
            float(system.max_slew),
            float(system.rf_raster_time),
            float(system.grad_raster_time),
            float(system.adc_raster_time),
            float(system.block_duration_raster),
            True,
            1,
        )
        t_convert = time.perf_counter() - t0
        return {
            "write_s": t_write,
            "convert_s": t_convert,
            "seq_bytes": len(payload),
        }


def pns_bound(seq) -> dict:
    """Worst-case-envelope PNS peak against every sampled real instance."""
    ok, norm, *_ = seq.calculate_pns(IRNICH, tr="worst_case", do_plots=False)
    envelope = float(np.max(norm))
    n = min(seq.num_trs, MAX_INSTANCES)
    instances = []
    for k in range(n):
        _, norm_k, *_ = seq.calculate_pns(IRNICH, tr=k, do_plots=False)
        instances.append(float(np.max(norm_k)))
    return {
        "envelope_peak": envelope,
        "instance_peaks": instances,
        "bound_holds": all(p <= envelope * (1 + 1e-9) for p in instances),
    }


def mechres_verdict(seq) -> dict:
    *_, lines = seq.calculate_gradient_spectrum(
        plot=False, tr="worst_case", resonance_lines=True, bands=BANDS
    )
    return {
        "ok": bool(lines.ok),
        "num_lines": int(np.size(lines.line_freqs)),
    }


def figure_tr(name: str, seq) -> None:
    plt.close("all")
    seq.plot(tr="worst_case")
    fig = plt.gcf()
    fig.suptitle(f"{name} — worst-case canonical TR (the window the checks run on)", fontsize=10)
    fig.savefig(ASSETS / f"{name}_tr.png", dpi=120, bbox_inches="tight")
    plt.close("all")


def figure_pns_bound(name: str, bound: dict) -> None:
    envelope = bound["envelope_peak"]
    peaks = [p / envelope for p in bound["instance_peaks"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(peaks)), peaks, color="#7aa6c2", label="actual TR instances")
    ax.axhline(1.0, color="#c23b22", lw=1.6, label="worst-case envelope")
    ax.set_xlabel("TR instance")
    ax.set_ylabel("peak PNS, relative to envelope")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{name} — the envelope bounds every instance", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / f"{name}_pns_bound.png", dpi=120)
    plt.close(fig)


def figure_mechres(name: str, seq) -> None:
    plt.close("all")
    seq.calculate_gradient_spectrum(plot=True, tr="worst_case", resonance_lines=True, bands=BANDS)
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        fig.suptitle(f"{name} — drive spectrum over the canonical TR (synthetic bands)", fontsize=10)
        fig.savefig(ASSETS / f"{name}_mechres{'' if i == 0 else f'_{i}'}.png", dpi=120)
    plt.close("all")


def figure_naive_comparison(name: str, seq, tr_index: int) -> None:
    """Our TR-window plot beside plain PyPulseq's plot of the same span.

    The sequence is written to a `.seq`, read back by *upstream* PyPulseq,
    and plotted over the time span of TR ``tr_index`` -- the qualitative
    counterpart of the bit-exactness tests: the window the checks run on is
    the sequence as any Pulseq reader sees it.
    """
    import pypulseq as upstream

    tr_size = seq.tr_size
    durations = np.asarray(seq.block_durations, dtype=float)
    ends = np.cumsum(durations)
    first = tr_index * tr_size
    t0 = float(ends[first - 1]) if first > 0 else 0.0
    t1 = float(ends[first + tr_size - 1])

    # Both panels are clipped to the active part of the TR: most of a GRE
    # repetition is recovery time, and a full-TR axis compresses the events
    # into unreadable slivers.
    active = min(0.03, t1 - t0)

    plt.close("all")
    seq.plot(tr=tr_index)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.set_xlim(0.0, active)
    fig.savefig(ASSETS / f"{name}_tr_ours.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "zoo.seq"
        seq.write(str(path))
        read = upstream.Sequence()
        read.read(str(path))
        read.plot(time_range=(t0, t0 + active))
    fig = plt.gcf()
    fig.savefig(ASSETS / f"{name}_tr_upstream.png", dpi=120, bbox_inches="tight")
    plt.close("all")


#: The family the side-by-side naive comparison is drawn for. Cartesian on
#: purpose: upstream PyPulseq does not compose ROTATIONS extensions, so a
#: rotated non-Cartesian file would not round-trip through its plotter. A
#: short, dense TR keeps the panels readable.
NAIVE_COMPARISON = "gre2D"


def run_family(name: str, figures: bool) -> dict:
    sizes = SIZES[name]
    entry: dict = {"sizes": []}
    seq = None
    for kwargs in sizes:
        t0 = time.perf_counter()
        seq = build(name, kwargs)
        t_design = time.perf_counter() - t0
        blocks = seq.num_blocks
        entry["sizes"].append(
            {
                "kwargs": kwargs,
                "blocks": blocks,
                "design_s": t_design,
                "us_per_block": t_design / blocks * 1e6,
            }
        )
        print(
            f"  {name:28s} blocks {blocks:>8d}  design {t_design:6.2f}s"
            f"  ({t_design / blocks * 1e6:5.2f} us/block)",
            flush=True,
        )
    entry["conversion"] = convert_with_c(seq)
    print(
        f"  {name:28s} write {entry['conversion']['write_s']:6.2f}s"
        f"  C convert {entry['conversion']['convert_s']:6.3f}s",
        flush=True,
    )

    small = build(name, sizes[0])
    entry["tr_size"] = small.declare_tr()
    entry["num_trs"] = int(small.num_trs)
    entry["pns"] = pns_bound(small)
    entry["mechres"] = mechres_verdict(small)
    print(
        f"  {name:28s} num_trs {entry['num_trs']:>6d}"
        f"  PNS envelope {entry['pns']['envelope_peak']:.3g}"
        f"  bound_holds {entry['pns']['bound_holds']}"
        f"  mechres_ok {entry['mechres']['ok']}",
        flush=True,
    )

    if figures and name in MULTISHOT_FIGURES:
        ASSETS.mkdir(parents=True, exist_ok=True)
        figure_tr(name, small)
        figure_pns_bound(name, entry["pns"])
        figure_mechres(name, small)
    if figures and name == NAIVE_COMPARISON:
        ASSETS.mkdir(parents=True, exist_ok=True)
        figure_naive_comparison(name, small, tr_index=min(2, entry["num_trs"] - 1))
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run one family")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    names = [args.only] if args.only else list(SIZES)
    build("gre2D", {"n_x": 64, "n_y": 64, "n_slices": 1})  # warm the imports
    results = {}
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text())
    for name in names:
        print(name, flush=True)
        try:
            results[name] = run_family(name, figures=not args.no_figures)
        except Exception as exc:  # noqa: BLE001 -- report and continue the sweep
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:28s} FAILED: {exc}", flush=True)
        RESULTS.write_text(json.dumps(results, indent=2))
    print(f"results -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
