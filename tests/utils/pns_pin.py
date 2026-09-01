"""The recorded PNS answer of the fixture corpus.

The safety suite asserts implementations against one another -- memoized
against exact, shipped model against an independent one -- and none of that
notices when every implementation moves together. This module records the
number itself: the worst-case Irnich peak the gate thresholds, one per corpus
sequence, under one fixed system, in ``tests/utils/expected/pns_peaks.json``.

An engine change that moves a verdict fails the pin test before it lands. A
fixture change moves the recording instead, as a reviewable diff at
regeneration time. Both directions are deliberate: the number may only change
on purpose, and visibly.

The system and nerve constants here are part of the pin's identity. Changing
them legitimately changes every recorded peak, so they are written into the
JSON beside the peaks they produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

EXPECTED_NAME = "pns_peaks.json"

#: Nerve constants, mirroring the shipped-model tests in
#: ``tests/ctests/test_safety_grad.c``.
IRNICH = {"chronaxie_us": 360.0, "rheobase": 4.25e8, "alpha": 0.333}

#: Relative tolerance the pin test asserts at. Wide enough for cross-compiler
#: float drift in a convolution pipeline, orders of magnitude tighter than any
#: change worth arguing about.
RTOL = 1e-4


def pin_system():
    """The one system every pinned peak is evaluated under.

    Returns:
        ``pulserver.pypulseq.Opts`` with the corpus-legal limits the safety
        gate tests use: 50 mT/m, 350 T/m/s, 3 T, 20 us gradient raster.
    """
    from pulserver.pypulseq import Opts

    return Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=20e-6,
        block_duration_raster=20e-6,
    )


def _system_record(system) -> dict:
    return {
        "gamma_hz_per_t": float(system.gamma),
        "b0_t": float(system.B0),
        "max_grad_hz_per_m": float(system.max_grad),
        "max_slew_hz_per_m_per_s": float(system.max_slew),
        "rf_raster_s": float(system.rf_raster_time),
        "grad_raster_s": float(system.grad_raster_time),
        "adc_raster_s": float(system.adc_raster_time),
        "block_raster_s": float(system.block_duration_raster),
    }


def compute_peak(seq_path: Path, system=None) -> float | None:
    """Worst-case Irnich peak of one ``.seq`` file, or None when the gate
    has no PNS answer for it (no gradients, not loadable on its own).

    Loads from raw bytes rather than by path, so a stale ``.pge`` cache
    beside the fixture can never stand in for the sequence itself.
    """
    from pulserver._ext.pulseg import _PulseqCollection, _calc_pns

    if system is None:
        system = pin_system()
    try:
        collection = _PulseqCollection(
            [Path(seq_path).read_bytes()],
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
        result = _calc_pns(
            collection,
            0,
            0,
            IRNICH["chronaxie_us"],
            IRNICH["rheobase"],
            IRNICH["alpha"],
        )
    except RuntimeError:
        return None

    peak = 0.0
    n = int(result["num_samples"])
    if n <= 0:
        return None
    sx, sy, sz = result["slew_x"], result["slew_y"], result["slew_z"]
    for i in range(n):
        v = math.sqrt(
            float(sx[i]) * float(sx[i])
            + float(sy[i]) * float(sy[i])
            + float(sz[i]) * float(sz[i])
        )
        if v > peak:
            peak = v
    return peak


def compute_all(corpus_dir: Path) -> dict[str, float | None]:
    """Peaks for every ``.seq`` in the corpus, keyed by stem, sorted."""
    system = pin_system()
    peaks: dict[str, float | None] = {}
    for path in sorted(Path(corpus_dir).glob("*.seq")):
        peak = compute_peak(path, system)
        peaks[path.stem] = None if peak is None else float(f"{peak:.9g}")
    return peaks


def write(expected_dir: Path, corpus_dir: Path) -> str:
    """Record the corpus's peaks; returns the file name written."""
    payload = {
        "irnich": IRNICH,
        "system": _system_record(pin_system()),
        "peaks": compute_all(corpus_dir),
    }
    out = Path(expected_dir) / EXPECTED_NAME
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return EXPECTED_NAME


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    name = write(here / "expected", here.parents[0] / "python" / "fixtures")
    print(f"wrote expected/{name}")
