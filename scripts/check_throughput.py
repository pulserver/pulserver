"""How long the design-side checks take, and which of them is the cost.

A sequence designed while the operator waits is checked before the scan time
comes back, so the check has to answer in the gap between one interaction and
the next. This measures where that time goes.

Usage: python scripts/check_throughput.py [blocks-per-repetition-count ...]
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pypulseqpp as pp
from pypulseqpp import safety

from pulserver import ir

#: A gradient coil and a nerve model to measure against. The numbers are a
#: plausible 3 T magnet's; the cost is what is being measured, not the verdict.
SYSTEM = pp.Opts(B0=3.0, max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
NERVE = safety.ChronaxieModel(chronaxie=360e-6, rheobase=(20.0, 22.5, 18.0), alpha=1.0)
BANDS = (
    safety.ForbiddenBand("x", 550.0, 650.0, 5.0),
    safety.ForbiddenBand(None, 1100.0, 1250.0, 5.0),
)

CASES = {
    "amplitude, slew, timing": ir.CheckLimits(),
    "+ nerve stimulation": ir.CheckLimits(pns=NERVE, pns_limit=0.8),
    "+ mechanical resonance": ir.CheckLimits(bands=BANDS),
    "+ both": ir.CheckLimits(pns=NERVE, pns_limit=0.8, bands=BANDS),
}


def gre(repetitions: int) -> pp.Sequence:
    """Return a spoiled gradient echo of ``repetitions`` repetitions, four blocks each."""
    seq = pp.Sequence(SYSTEM)
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(15),
        duration=1e-3,
        slice_thickness=5e-3,
        system=SYSTEM,
        return_gz=True,
        use="excitation",
    )
    readout = pp.make_trapezoid(
        "x", flat_area=1 / 2.4e-3, flat_time=2.56e-3, system=SYSTEM
    )
    adc = pp.make_adc(
        num_samples=128, duration=2.56e-3, delay=readout.rise_time, system=SYSTEM
    )
    prewinder = pp.make_trapezoid("x", area=-readout.area / 2, system=SYSTEM)
    for line in range(repetitions):
        encode = pp.make_trapezoid(
            "y", area=(line / repetitions - 0.5) / 2.4e-3, system=SYSTEM
        )
        seq.add_block(rf, gz)
        seq.add_block(gzr, prewinder, encode)
        seq.add_block(readout, adc)
        seq.add_block(pp.make_delay(5e-3))
    return seq


def measure(path: Path, limits: ir.CheckLimits, runs: int = 5) -> float:
    """Median seconds one check of ``path`` takes."""
    taken = []
    for _ in range(runs):
        started = time.perf_counter()
        ir.check(path, SYSTEM, limits=limits)
        taken.append(time.perf_counter() - started)
    return statistics.median(taken)


def main(counts: list[int]) -> int:
    print(
        f"{'repetitions':>12s} {'blocks':>8s} {'design':>9s} "
        + "".join(f"{name:>26s}" for name in CASES)
    )
    for repetitions in counts:
        started = time.perf_counter()
        seq = gre(repetitions)
        design = time.perf_counter() - started
        blocks = len(seq.block_events)
        path = Path(tempfile.mkdtemp()) / "measured.seq"
        seq.write(path)
        row = "".join(
            f"{measure(path, limits) * 1e3:20.0f} ms   " for limits in CASES.values()
        )
        print(f"{repetitions:12d} {blocks:8d} {design * 1e3:7.0f}ms {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main([int(v) for v in sys.argv[1:]] or [64, 256, 1024]))
