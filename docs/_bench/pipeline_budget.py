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
end to end              <= 30 s at 128K
======================  =====================================

Run it as::

    python docs/_bench/pipeline_budget.py [--arms=16384]

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
TARGET_ARMS = 131072

BUDGET = {
    "assembly_us_per_arm": 60.0,
    "declare_dedup_s_at_target": 3.0,
    "write_mb_per_s": 1000.0,
    "parse_mb_per_s": 1000.0,
    "gate_s_at_target": 7.0,
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
    base = np.stack(
        [np.sin(40 * np.pi * t) * taper, np.cos(40 * np.pi * t) * taper]
    )
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
        collection, [], 4.25e8 / 0.333, 360.0, 100.0, False
    )
    t_gate = time.perf_counter() - t0

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
            for k in ("pns", "pns_basis_build", "pns_score")
        },
        "at_target": {
            "assembly_s": (t_factory + t_add) * scale,
            "declare_dedup_s": (t_declare + t_dedup) * scale,
            "write_s": t_write * scale,
            "parse_s": t_parse * scale,
            "gate_s": t_gate * scale,
            "end_to_end_s": (
                t_factory + t_add + t_declare + t_dedup + t_write + t_parse + t_gate
            )
            * scale,
        },
        "budget": BUDGET,
    }
    return entry


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
    line(
        "PNS gate",
        f"{at['gate_s']:.1f} s",
        f"{b['gate_s_at_target']:.0f} s",
        at["gate_s"] <= b["gate_s_at_target"],
    )
    line(
        "end to end",
        f"{at['end_to_end_s']:.1f} s",
        f"{b['end_to_end_s_at_target']:.0f} s",
        at["end_to_end_s"] <= b["end_to_end_s_at_target"],
    )
    print(f"  gate verdict code {entry['gate_code']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=int, default=16384)
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entry = run(args.arms)
    report(entry)
    RESULTS.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    print(f"wrote {RESULTS.relative_to(RESULTS.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
