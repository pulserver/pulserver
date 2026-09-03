"""Per-stage cost of the whole 128K-arm pipeline, against its budget.

Documentation-only tooling, like everything else in this directory. The
scenario is a scan whose every readout is a distinct precomputed (2, NPTS)
waveform -- the trajectory optimiser's output -- assembled through the plain
PyPulseq surface, written as binary, parsed, converted and safety-checked.
Each stage is timed at ``--arms`` synthetic arms and extrapolated linearly to
128K, next to its budget line:

======================  =====================================
assembly                <= 60 us/arm
dedup + declare_tr      <= 3 s at 128K
binary write            >= 1 GB/s
parse + convert         >= 1 GB/s
PNS gate                <= 7 s at 128K
mech-res check          <= 7 s at 128K
end to end              <= 30 s at 128K
======================  =====================================

Run it as::

    python docs/_bench/pipeline_budget.py [--arms=16384]

``--mech-only [ARMS ...]`` times the mechanical-resonance check alone on 2D
and 3D scans of distinct arms (8K, 32K and 128K by default) and records the
peak resident set size; results land in ``docs/_bench/mechres_scale.json``.

Results land in ``docs/_bench/pipeline_budget.json``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "pipeline_budget.json"

NPTS = 4096
# Forbidden bands the mechanical-resonance check is timed against. The
# amplitude is far above anything the synthetic arms sustain: the harness
# times the check, whose work does not depend on the threshold, and a verdict
# on random 50 mT/m waveforms would say nothing about a real design.
MECH_BANDS = [(550.0, 650.0, 1.0e12), (1100.0, 1250.0, 1.0e12)]
TARGET_ARMS = 131072

BUDGET = {
    "assembly_us_per_arm": 60.0,
    "declare_dedup_s_at_target": 3.0,
    "write_mb_per_s": 1000.0,
    "parse_mb_per_s": 1000.0,
    "gate_s_at_target": 7.0,
    "mech_s_at_target": 7.0,
    "end_to_end_s_at_target": 30.0,
}


def build_arms(n_arms: int):
    """Distinct (2, NPTS) waveforms standing in for an offline optimiser."""
    import numpy as np

    import pulserver.pypulseq as pp

    system = pp.Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        rf_raster_time=2e-6,
    )
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, NPTS)
    taper = 4.0 * t * (1.0 - t)
    base = np.stack([np.sin(40 * np.pi * t) * taper, np.cos(40 * np.pi * t) * taper])
    base *= 0.6 * system.max_grad / np.abs(base).max()
    arms = []
    for k in range(n_arms):
        c, s = np.cos(2 * np.pi * k / n_arms), np.sin(2 * np.pi * k / n_arms)
        gx = c * base[0] - s * base[1]
        gx = gx + 1e-3 * rng.standard_normal(NPTS) * np.abs(gx).max()
        gy = s * base[0] + c * base[1]
        arms.append((gx, gy))
    return system, arms


def run(n_arms: int) -> dict:
    import pulserver.pypulseq as pp
    from pulserver._ext.pulseg import _check_safety_profiled, _PulseqCollection

    system, arms = build_arms(n_arms)
    rf = pp.make_block_pulse(
        flip_angle=0.17453292519943295, duration=200e-6, system=system
    )
    adc = pp.make_adc(num_samples=NPTS, dwell=4e-6, system=system)
    spoil = pp.make_trapezoid(channel="z", area=1000.0, system=system)

    t0 = time.perf_counter()
    events = [
        (
            pp.make_arbitrary_grad(channel="x", waveform=gx, system=system),
            pp.make_arbitrary_grad(channel="y", waveform=gy, system=system),
        )
        for gx, gy in arms
    ]
    t_factory = time.perf_counter() - t0

    seq = pp.Sequence(system)
    t0 = time.perf_counter()
    for gx, gy in events:
        seq.add_block(rf)
        seq.add_block(gx, gy, adc)
        seq.add_block(spoil)
    t_add = time.perf_counter() - t0

    t0 = time.perf_counter()
    seq.declare_tr()
    t_declare = time.perf_counter() - t0
    t0 = time.perf_counter()
    # In place, which is the scale guidance: the copy the default makes is a
    # full clone of a library this size, and this pipeline owns its sequence.
    deduped = seq.remove_duplicates(in_place=True)
    t_dedup = time.perf_counter() - t0

    t0 = time.perf_counter()
    binary = deduped._to_binary()
    t_write = time.perf_counter() - t0

    t0 = time.perf_counter()
    collection = _PulseqCollection(
        [binary],
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
    )
    t_parse = time.perf_counter() - t0

    t0 = time.perf_counter()
    gate = _check_safety_profiled(
        collection, MECH_BANDS, 4.25e8 / 0.333, 360.0, 100.0, False
    )
    t_gate = time.perf_counter() - t0
    t_pns = gate["stages"]["pns"]["seconds"]
    t_mech = gate["stages"]["mech_resonance"]["seconds"]

    scale = TARGET_ARMS / n_arms
    mb = len(binary) / 1e6
    entry = {
        "arms": n_arms,
        "npts": NPTS,
        "blocks": int(seq.num_blocks),
        "binary_mb": mb,
        "factory_s": t_factory,
        "addblock_s": t_add,
        "assembly_us_per_arm": (t_factory + t_add) / n_arms * 1e6,
        "declare_s": t_declare,
        "dedup_s": t_dedup,
        "write_s": t_write,
        "write_mb_per_s": mb / t_write if t_write > 0 else float("inf"),
        "parse_s": t_parse,
        "parse_mb_per_s": mb / t_parse if t_parse > 0 else float("inf"),
        "gate_s": t_gate,
        "gate_code": gate["code"],
        "gate_stages": {
            k: gate["stages"][k]["seconds"]
            for k in ("pns", "pns_basis_build", "pns_score", "mech_resonance")
        },
        "at_target": {
            "assembly_s": (t_factory + t_add) * scale,
            "declare_dedup_s": (t_declare + t_dedup) * scale,
            "write_s": t_write * scale,
            "parse_s": t_parse * scale,
            "gate_s": t_gate * scale,
            "pns_s": t_pns * scale,
            "mech_s": t_mech * scale,
            "end_to_end_s": (
                t_factory + t_add + t_declare + t_dedup + t_write + t_parse + t_gate
            )
            * scale,
        },
        "budget": BUDGET,
    }
    return entry


MECH_ONLY_RESULTS = Path(__file__).resolve().parent / "mechres_scale.json"


def mech_only(n_arms: int, dims: int, repeats: int = 3) -> dict:
    """The mechanical-resonance check alone on a scan of distinct arms.

    Every arm is a distinct (dims, NPTS) waveform; a 3D arm plays a third
    waveform on z. The scan is assembled and parsed once, then the safety
    check runs ``repeats`` times against ``MECH_BANDS`` and the fastest
    mechanical-resonance stage is kept, with the peak resident set size of
    the process after the run.
    """
    import resource

    import numpy as np
    import pulserver.pypulseq as pp
    from pulserver._ext.pulseg import _check_safety_profiled, _PulseqCollection

    system, arms = build_arms(n_arms)
    rf = pp.make_block_pulse(
        flip_angle=0.17453292519943295, duration=200e-6, system=system
    )
    adc = pp.make_adc(num_samples=NPTS, dwell=4e-6, system=system)
    spoil = pp.make_trapezoid(channel="z", area=1000.0, system=system)
    seq = pp.Sequence(system)
    for k, (gx, gy) in enumerate(arms):
        grads = [
            pp.make_arbitrary_grad(channel="x", waveform=gx, system=system),
            pp.make_arbitrary_grad(channel="y", waveform=gy, system=system),
        ]
        if dims == 3:
            # a third distinct waveform, still starting and ending at zero
            gz = (0.8 - 0.1 * (k % 5) / 5.0) * gy[::-1]
            grads.append(
                pp.make_arbitrary_grad(channel="z", waveform=gz, system=system)
            )
        seq.add_block(rf)
        seq.add_block(*grads, adc)
        seq.add_block(spoil)
    seq.declare_tr()
    deduped = seq.remove_duplicates(in_place=True)
    binary = deduped._to_binary()
    collection = _PulseqCollection(
        [binary],
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
    )
    best = float("inf")
    code = None
    for _ in range(repeats):
        gate = _check_safety_profiled(
            collection, MECH_BANDS, 4.25e8 / 0.333, 360.0, 100.0, False
        )
        best = min(best, gate["stages"]["mech_resonance"]["seconds"])
        code = gate["code"]
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "arms": n_arms,
        "dims": dims,
        "npts": NPTS,
        "mech_s": best,
        "mech_s_at_target": best * TARGET_ARMS / n_arms,
        "gate_code": code,
        "peak_rss_mb": rss_mb,
    }


def report_mech_only(entries: list) -> None:
    print("mechanical-resonance check alone (fastest of the repeats):")
    for e in entries:
        print(
            f"  {e['dims']}D {e['arms']:>7d} arms: {e['mech_s'] * 1e3:8.1f} ms"
            f"  ({e['mech_s_at_target']:.2f} s at {TARGET_ARMS} arms)"
            f"  peak RSS {e['peak_rss_mb']:.0f} MB  code {e['gate_code']}"
        )


def report(entry: dict) -> None:
    at = entry["at_target"]

    def line(name, measured, target, ok):
        mark = "within" if ok else "OVER"
        print(f"  {name:<18s} {measured:>10s}  target {target:<12s} {mark}")

    print(
        f"pipeline budget at {entry['arms']} arms x {entry['npts']} pts "
        f"({entry['blocks']} blocks, {entry['binary_mb']:.0f} MB binary), "
        f"extrapolated to {TARGET_ARMS} arms:"
    )
    b = entry["budget"]
    line(
        "assembly",
        f"{entry['assembly_us_per_arm']:.0f} us/arm",
        f"{b['assembly_us_per_arm']:.0f} us/arm",
        entry["assembly_us_per_arm"] <= b["assembly_us_per_arm"],
    )
    line(
        "declare+dedup",
        f"{at['declare_dedup_s']:.1f} s",
        f"{b['declare_dedup_s_at_target']:.0f} s",
        at["declare_dedup_s"] <= b["declare_dedup_s_at_target"],
    )
    line(
        "binary write",
        f"{entry['write_mb_per_s']:.0f} MB/s",
        f"{b['write_mb_per_s']:.0f} MB/s",
        entry["write_mb_per_s"] >= b["write_mb_per_s"],
    )
    line(
        "parse+convert",
        f"{entry['parse_mb_per_s']:.0f} MB/s",
        f"{b['parse_mb_per_s']:.0f} MB/s",
        entry["parse_mb_per_s"] >= b["parse_mb_per_s"],
    )
    # Verdict codes are negative; any non-negative code is a gate that ran.
    gated = entry["gate_code"] >= 0
    line(
        "PNS gate",
        f"{at['pns_s']:.1f} s" if gated else f"declined ({entry['gate_code']})",
        f"{b['gate_s_at_target']:.0f} s",
        gated and at["pns_s"] <= b["gate_s_at_target"],
    )
    line(
        "mech-res check",
        f"{at['mech_s']:.1f} s" if gated else f"declined ({entry['gate_code']})",
        f"{b['mech_s_at_target']:.0f} s",
        gated and at["mech_s"] <= b["mech_s_at_target"],
    )
    line(
        "end to end",
        f"{at['end_to_end_s']:.1f} s",
        f"{b['end_to_end_s_at_target']:.0f} s",
        at["end_to_end_s"] <= b["end_to_end_s_at_target"],
    )
    print(f"  gate verdict code {entry['gate_code']}")


FIGURE = (
    Path(__file__).resolve().parents[1]
    / "explanations"
    / "assets"
    / "pipeline_budget"
    / "stages.png"
)


def plot(entry: dict, out: Path = FIGURE) -> Path:
    """Every stage at the target size against its budget line, as bars.

    Parameters
    ----------
    entry : dict
        A result of :func:`run`, as ``pipeline_budget.json`` holds it.
    out : Path
        Where the figure goes; the page that shows it reads it from there.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    at, b = entry["at_target"], entry["budget"]
    stages = [
        ("assembly", at["assembly_s"], b["assembly_us_per_arm"] * TARGET_ARMS * 1e-6),
        ("declare + dedup", at["declare_dedup_s"], b["declare_dedup_s_at_target"]),
        (
            "binary write",
            at["write_s"],
            entry["binary_mb"] * TARGET_ARMS / entry["arms"] / b["write_mb_per_s"],
        ),
        (
            "parse + convert",
            at["parse_s"],
            entry["binary_mb"] * TARGET_ARMS / entry["arms"] / b["parse_mb_per_s"],
        ),
        ("PNS gate", at["pns_s"], b["gate_s_at_target"]),
        ("mech-res check", at["mech_s"], b["mech_s_at_target"]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    names = [n for n, _, _ in stages]
    measured = [m for _, m, _ in stages]
    lines = [t for _, _, t in stages]
    y = range(len(stages))
    ax.barh(
        y,
        measured,
        color=[
            "#2a9d8f" if m <= t else "#e76f51"
            for m, t in zip(measured, lines, strict=True)
        ],
        height=0.55,
    )
    for i, t in enumerate(lines):
        ax.plot([t, t], [i - 0.35, i + 0.35], color="black", lw=1.4)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"seconds at {TARGET_ARMS:,} arms of {entry['npts']} points, extrapolated from {entry['arms']:,}"
    )
    total = sum(measured)
    ax.set_title(
        f"end to end {total:.0f} s against a {b['end_to_end_s_at_target']:.0f} s budget; ticks are each stage's line"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=int, default=16384)
    parser.add_argument(
        "--plot", action="store_true", help="also draw the stage figure the docs show"
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        help="draw from the saved result without running",
    )
    parser.add_argument(
        "--mech-only",
        nargs="*",
        type=int,
        metavar="ARMS",
        help="time the mechanical-resonance check alone, 2D and 3D, at these arm counts "
        "(default 8192 32768 131072); results land in mechres_scale.json",
    )
    args = parser.parse_args()

    if args.mech_only is not None:
        counts = args.mech_only or [8192, 32768, 131072]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entries = [mech_only(n, dims) for dims in (2, 3) for n in counts]
        report_mech_only(entries)
        MECH_ONLY_RESULTS.write_text(
            json.dumps(entries, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote {MECH_ONLY_RESULTS.relative_to(MECH_ONLY_RESULTS.parents[2])}")
        return 0
    if args.from_json:
        entry = json.loads(RESULTS.read_text())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entry = run(args.arms)
        report(entry)
        RESULTS.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        print(f"wrote {RESULTS.relative_to(RESULTS.parents[2])}")
    if args.plot or args.from_json:
        out = plot(entry)
        print(f"wrote {out.relative_to(out.parents[3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
