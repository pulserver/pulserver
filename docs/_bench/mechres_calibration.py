"""The mechanical-resonance criterion measured against the vendor's own rules.

Reads every lockout table the vendor ships for the families it checks -- the
echo-planar echo-spacing tables (``lockout/epiesp*.dat``, one section per
physical axis, a tolerance column that is almost always zero), the FIESTA
repetition-time lockouts and the multi-echo echo-spacing lockout
(``greAcousticLimit*.dat``) -- and turns each into the frequency band it
guards. Then it designs the sequences the vendor refuses (an echo train whose
fundamental sits in a band, a FIESTA at a locked TR, a multi-echo train at a
locked spacing) and the families it runs without any check, reads every one
of them the way the predownload gate does, and reports:

* the **bracket**: the loudest in-band reading of anything the vendor runs
  against the quietest reading of anything it refuses, the accepted
  divergences set aside, which is where the zero-column floor
  ``SA_ZERO_BAND_SINUSOID_MT_PER_M`` has to sit;
* the stated tolerances, where a table states one, in the same sinusoid
  units the gate compares in;
* the **scenario table**: every family and the edge cases the criterion was
  argued over, with the gradient definition behind each loudest reading;
* the figures the explanation pages show.

Results land in ``docs/_bench/mechres_calibration.json`` and the figures under
``docs/explanations/assets/mechanical_resonance/``::

    <venv>/bin/python docs/_bench/mechres_calibration.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pulserver.pypulseq as pp
from _figures import FAINT, INK, MUTED, SERIES, _style

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[0] / "explanations" / "assets" / "mechanical_resonance"
RESULTS = HERE / "mechres_calibration.json"
SCALE = HERE / "mechres_scale.json"
LOCKOUT_DIRS = [
    Path.home() / "pulserver-project" / "lockout",
    Path.home() / "PulseStudio" / "efgre3d",
]

GAMMA_HZ_PER_T = 42.576e6
GAMMA_HZ_PER_MT_PER_M = GAMMA_HZ_PER_T * 1e-3
#: ``SA_ZERO_BAND_SINUSOID_MT_PER_M``: drawn, never recomputed here.
FLOOR_MT_PER_M = 10.0
#: The range every inspected band falls in, read densely on every axis.
TERRITORY = (300.0, 3000.0)
#: A tolerance nothing sustains under: every band is refused on the bound,
#: so what comes back is the scan's own reading and its contributors.
TINY_HZ_PER_M = 1.0
SYSTEM = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
WEAK = pp.Opts(max_grad=33, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
#: Designs a parameter sweep could not build, reported at the end.
INFEASIBLE: list[tuple] = []

plt.rcParams.update(
    {
        "font.size": 11.0,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.0,
    }
)


# ----------------------------------------------------------------------
# The vendor's tables


def _numbers(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        try:
            rows.append([float(x) for x in body.split()])
        except ValueError:
            continue
    return rows


def read_epi_table(path: Path) -> dict:
    """One echo-spacing lockout table as bands per axis.

    A row is an echo-spacing range in microseconds and a tolerance in G/cm;
    the band it guards is the train fundamental, f = 1 / (2 ESP).
    """
    text = path.read_text().splitlines()
    bands = {"x": [], "y": [], "z": []}
    axis = None
    pending = None
    for line in text:
        head = line.strip().lower()
        if head.startswith("#"):
            for name in ("x", "y", "z"):
                if head.startswith(f"# {name} axis"):
                    axis = name
                    pending = None
            continue
        parts = head.split()
        if axis is None or not parts:
            continue
        if pending is None and len(parts) == 1:
            pending = int(float(parts[0]))
            continue
        if pending and len(parts) == 3:
            lo_us, hi_us, tol = (float(p) for p in parts)
            bands[axis].append(
                {
                    "esp_us": (lo_us, hi_us),
                    "f_hz": (1e6 / (2.0 * hi_us), 1e6 / (2.0 * lo_us)),
                    "tolerance_mt_per_m": 10.0 * tol,  # G/cm -> mT/m
                }
            )
            pending -= 1
    return {"file": path.name, "bands": bands}


def read_tr_table(path: Path) -> dict:
    """A FIESTA repetition-time lockout: MinTR MaxTR deltaTR rows, in us."""
    rows = [r for r in _numbers(path) if len(r) == 3]
    return {"file": path.name, "tr_us": [(r[0], r[1]) for r in rows]}


def read_esp_table(path: Path) -> dict:
    """The multi-echo echo-spacing lockout: lower and upper ESP, in us."""
    rows = [r for r in _numbers(path) if len(r) == 2]
    return {"file": path.name, "esp_us": [(r[0], r[1]) for r in rows]}


def load_tables() -> dict:
    epi, tr, esp = [], [], []
    for folder in LOCKOUT_DIRS:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("epiesp*.dat")):
            epi.append(read_epi_table(path))
        for path in sorted(folder.glob("greAcousticLimit.*.dat")):
            tr.append(read_tr_table(path))
        for path in sorted(folder.glob("greAcousticLimitEsp*.dat")):
            esp.append(read_esp_table(path))
    return {"epi": epi, "tr": tr, "esp": esp}


def stated_tolerances(tables: dict) -> list[dict]:
    """Every nonzero tolerance a table states, and the sinusoid it allows
    through the triangle-train factor the gate applies when no fused train's
    fundamental lies in the band."""
    out = []
    for table in tables["epi"]:
        for axis, bands in table["bands"].items():
            for band in bands:
                if band["tolerance_mt_per_m"] > 0.0:
                    out.append(
                        {
                            "file": table["file"],
                            "axis": axis,
                            "f_hz": band["f_hz"],
                            "plateau_mt_per_m": band["tolerance_mt_per_m"],
                            "sinusoid_mt_per_m": band["tolerance_mt_per_m"]
                            * 8.0
                            / math.pi**2,
                        }
                    )
    return out


# ----------------------------------------------------------------------
# Reading a sequence the way the gate does


def _sequence(result):
    return result[0] if isinstance(result, tuple) else result


def dense_reading(sequence, memory: float | None = None) -> dict:
    """The exact window reading of a sequence over the territory, per axis.

    One wide band per axis at a tolerance nothing sustains under, so every
    band is refused and what comes back is the scan read exactly, not the
    bound over its repetitions.
    """
    bands = [(TERRITORY[0], TERRITORY[1], TINY_HZ_PER_M, axis) for axis in "xyz"]
    overlay = sequence.calculate_gradient_spectrum(
        plot=False,
        max_frequency=TERRITORY[1],
        tr="worst_case",
        resonance_lines=True,
        bands=bands,
        memory=memory,
    )[4]
    freqs = np.asarray(overlay.candidate_freqs, float)
    amps = np.asarray(overlay.candidate_a_eq, float) / GAMMA_HZ_PER_MT_PER_M
    return {"freqs": freqs, "amps": amps}


def in_band(reading: dict, band: tuple[float, float], axis: int) -> float:
    inside = (reading["freqs"] >= band[0]) & (reading["freqs"] <= band[1])
    return float(reading["amps"][inside, axis].max()) if inside.any() else 0.0


def contributors_at(
    sequence, band: tuple[float, float], axis: str
) -> list[tuple[int, float]]:
    """The definitions behind the loudest reading inside one band."""
    overlay = sequence.calculate_gradient_spectrum(
        plot=False,
        max_frequency=TERRITORY[1],
        tr="worst_case",
        resonance_lines=True,
        bands=[(band[0], band[1], TINY_HZ_PER_M, axis)],
    )[4]
    return list(overlay.contributors)


def describe_definitions(sequence) -> dict[int, str]:
    """What each gradient definition plays, from the interpreter's own block
    table: the axis it sits on and where in the segment it falls."""
    from pulserver._ext.pulseg import _get_segment_blocks

    collection = sequence._structure_for("bound").collection
    roles: dict[int, str] = {}
    for segment in _get_segment_blocks(collection):
        blocks = segment["blocks"]
        for index, block in enumerate(blocks):
            for axis, def_id in zip("xyz", block["grad_def_id"], strict=True):
                if def_id < 0 or def_id in roles:
                    continue
                if block["has_adc"]:
                    where = "the readout"
                elif block["has_rf"]:
                    where = "with the RF pulse"
                else:
                    later = blocks[index + 1 :]
                    next_adc = next(
                        (j for j, b in enumerate(later) if b["has_adc"]), None
                    )
                    next_rf = next(
                        (j for j, b in enumerate(later) if b["has_rf"]), None
                    )
                    if next_adc is not None and (next_rf is None or next_adc < next_rf):
                        where = "before the readout"
                    else:
                        where = "after the readout"
                roles[def_id] = f"{axis}: {where}"
    return roles


# ----------------------------------------------------------------------
# The scenarios


def _zoo(module, system=SYSTEM, **kwargs):
    from importlib import import_module

    return _sequence(
        getattr(import_module("pulserver.app"), module).main(
            system=system, n_dummy=0, **kwargs
        )
    )


def _acquisition_starts_and_durations_s(sequence) -> tuple[list[float], list[float]]:
    """Start time and duration of every block that acquires, in order."""
    starts, durations, t = [], [], 0.0
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if block.adc is not None:
            starts.append(t)
            durations.append(float(block.block_duration))
        t += float(block.block_duration)
    return starts, durations


def _acquisition_durations_s(sequence) -> list[float]:
    return _acquisition_starts_and_durations_s(sequence)[1]


def _echo_spacing_s(sequence) -> float:
    """The period of the acquisitions that repeat back to back: the most
    common spacing between consecutive acquisition starts, whatever blocks
    lie between them."""
    starts, _ = _acquisition_starts_and_durations_s(sequence)
    if len(starts) < 2:
        return float("nan")
    gaps = np.diff(np.asarray(starts))
    gaps = gaps[gaps <= 3.0 * np.median(gaps)]  # within a train, not across repetitions
    values, counts = np.unique(np.round(gaps, 7), return_counts=True)
    return float(values[np.argmax(counts)])


def _readout_duration_s(sequence) -> float:
    durations = _acquisition_durations_s(sequence)
    return max(durations) if durations else float("nan")


def _readout_plateau_mt_per_m(sequence) -> float:
    """The largest gradient amplitude played under an acquisition."""
    peak = 0.0
    for index in range(1, len(sequence.block_events) + 1):
        block = sequence.get_block(index)
        if block.adc is None:
            continue
        for grad in (block.gx, block.gy, block.gz):
            if grad is None:
                continue
            if grad.type == "trap":
                peak = max(peak, abs(float(grad.amplitude)))
            else:
                peak = max(peak, float(np.max(np.abs(np.asarray(grad.waveform)))))
    return peak / GAMMA_HZ_PER_MT_PER_M


def epi_with_fundamental_in(band_hz: tuple[float, float], n_x: int, n_y: int):
    """An echo-planar train whose fundamental 1/(2 ESP) lands inside a band,
    found by sweeping the receiver bandwidth."""
    target = 0.5 * (band_hz[0] + band_hz[1])
    best = None
    for bandwidth in np.geomspace(20e3, 500e3, 60):
        try:
            seq = _zoo(
                "epi2D_sequence",
                WEAK,
                n_x=n_x,
                n_y=n_y,
                readout_bandwidth_hz=float(bandwidth),
            )
        except Exception as exc:  # an infeasible design at this bandwidth
            INFEASIBLE.append(("epi2D_sequence", float(bandwidth), str(exc)[:60]))
            continue
        esp = _echo_spacing_s(seq)
        if not np.isfinite(esp):
            continue
        f0 = 0.5 / esp
        if best is None or abs(f0 - target) < abs(best[1] - target):
            best = (seq, f0, float(bandwidth))
        if band_hz[0] <= f0 <= band_hz[1]:
            return seq, f0, float(bandwidth)
    return best


def epi_in_band(band_hz: tuple[float, float], system=SYSTEM):
    """The loudest echo train a console could prescribe whose fundamental
    lands inside ``band_hz``: matrices from 128 down, bandwidths from the
    receiver's top down, the first feasible design wins."""
    for n in (128, 96, 64, 48, 32):
        for bandwidth in np.geomspace(500e3, 40e3, 40):
            try:
                seq = _zoo(
                    "epi2D_sequence",
                    system,
                    n_x=n,
                    n_y=n,
                    readout_bandwidth_hz=float(bandwidth),
                )
            except Exception as exc:
                INFEASIBLE.append(("epi2D_sequence", float(bandwidth), str(exc)[:60]))
                continue
            esp = _echo_spacing_s(seq)
            if not np.isfinite(esp):
                continue
            f0 = 0.5 / esp
            if band_hz[0] <= f0 <= band_hz[1]:
                return seq, f0, n, float(bandwidth)
    return None


def multiecho_with_spacing_in(
    esp_us: tuple[float, float], n_echoes: int, monopolar: bool
):
    target = 0.5e-6 * (esp_us[0] + esp_us[1])
    best = None
    for bandwidth in np.geomspace(30e3, 400e3, 50):
        try:
            seq = _zoo(
                "gre_multiecho2D_sequence",
                n_x=128,
                n_y=64,
                n_echoes=n_echoes,
                monopolar=monopolar,
                readout_bandwidth_hz=float(bandwidth),
            )
        except Exception as exc:  # an infeasible design at this bandwidth
            INFEASIBLE.append(
                ("gre_multiecho2D_sequence", float(bandwidth), str(exc)[:60])
            )
            continue
        esp = _echo_spacing_s(seq)
        if not np.isfinite(esp):
            continue
        if best is None or abs(esp - target) < abs(best[1] - target):
            best = (seq, esp, float(bandwidth))
        if esp_us[0] * 1e-6 <= esp <= esp_us[1] * 1e-6:
            return seq, esp, float(bandwidth)
    return best


def arms_scan(n_arms: int, peak_mt_per_m: float = 15.0, n_samples: int = 4096):
    """A scan of distinct arbitrary arms, the SPARKLING-like case: past the
    group cap, read from every event it plays. Each arm is a sum of chirps
    with its own start and end frequencies, so no arm sustains a tone and no
    two arms share a waveform."""
    system = pp.Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        rf_raster_time=2e-6,
    )
    rng = np.random.default_rng(3)
    t = np.arange(n_samples) * 4e-6
    envelope = np.sin(np.pi * np.arange(n_samples) / (n_samples - 1)) ** 2
    rf = pp.make_block_pulse(
        flip_angle=0.17453292519943295, duration=200e-6, system=system
    )
    adc = pp.make_adc(num_samples=n_samples, dwell=4e-6, system=system)
    spoil = pp.make_trapezoid(channel="z", area=1000.0, system=system)
    seq = pp.Sequence(system)
    for _ in range(n_arms):
        grads = []
        for channel in "xy":
            w = np.zeros(n_samples)
            for _chirp in range(3):
                f0, f1 = rng.uniform(200.0, 2000.0, size=2)
                w += np.sin(
                    2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t * t)
                    + rng.uniform(0, 2 * np.pi)
                )
            w *= envelope
            w *= peak_mt_per_m * GAMMA_HZ_PER_MT_PER_M / np.abs(w).max()
            grads.append(
                pp.make_arbitrary_grad(channel=channel, waveform=w, system=system)
            )
        seq.add_block(rf)
        seq.add_block(*grads, adc)
        seq.add_block(spoil)
    seq.declare_tr()
    return seq


def scenarios(tables: dict, quick: bool) -> list[dict]:
    """(label, the vendor's verdict, the coils whose tables it is judged on,
    sequence, note) for every case. A refusal is judged on the coil whose
    table refuses it; a family the vendor runs unchecked is judged on every
    coil's tables at once, since it runs on all of them."""
    rows = []

    def add(label, vendor, seq, note="", coils=None):
        """``seq`` may be a callable building the sequence, so a design this
        system cannot play is recorded rather than ending the run."""
        if callable(seq):
            try:
                seq = seq()
            except Exception as exc:
                rows.append(
                    {
                        "label": label,
                        "vendor": vendor,
                        "coils": coils,
                        "sequence": None,
                        "note": f"not built: {str(exc)[:70]}",
                    }
                )
                print(f"  {label}: not built ({str(exc)[:70]})", flush=True)
                return
        print(f"  {label}: built", flush=True)
        rows.append(
            {
                "label": label,
                "vendor": vendor,
                "coils": coils,
                "sequence": seq,
                "note": note,
            }
        )

    # Echo-planar trains at protocols a console prescribes: each one's
    # fundamental falls in some coils' bands, and those coils refuse it.
    for n, bandwidth, system, tag in (
        (64, 250e3, SYSTEM, ""),
        (96, 250e3, SYSTEM, ""),
        (128, 250e3, SYSTEM, ""),
        (64, 125e3, SYSTEM, ""),
        (96, 250e3, WEAK, ", weaker gradients"),
    ):
        try:
            seq = _zoo(
                "epi2D_sequence", system, n_x=n, n_y=n, readout_bandwidth_hz=bandwidth
            )
        except Exception as exc:
            rows.append(
                {
                    "label": f"EPI {n}, {bandwidth / 1e3:.0f} kHz{tag}",
                    "vendor": "edge",
                    "coils": None,
                    "sequence": None,
                    "note": str(exc)[:70],
                }
            )
            continue
        f0 = 0.5 / _echo_spacing_s(seq)
        locking = coils_locking(tables, f0)
        plateau = _readout_plateau_mt_per_m(seq)
        label = f"EPI {n}, {bandwidth / 1e3:.0f} kHz{tag} ({plateau:.0f} mT/m plateau, f0 {f0:.0f} Hz)"
        if locking:
            add(
                label,
                "refused",
                seq,
                f"locked on {', '.join(sorted(locking))}",
                coils=locking,
            )
        else:
            add(label, "allowed", seq, "fundamental clear of every table")

    # One echo train per coil band, at the loudest plateau a console prescribes
    seen = set()
    for table in tables["epi"]:
        coil = coil_of(table["file"])
        for band in table["bands"]["x"]:
            key = (coil, round(band["f_hz"][0]))
            if key in seen or (quick and len(seen) >= 2):
                continue
            seen.add(key)
            found = epi_in_band(band["f_hz"])
            if not found:
                continue
            seq, f0, n, bw = found
            plateau = _readout_plateau_mt_per_m(seq)
            add(
                f"EPI {n}, {bw / 1e3:.0f} kHz, f0 {f0:.0f} Hz in the {coil} band ({plateau:.0f} mT/m plateau)",
                "refused",
                seq,
                f"locked on {coil}",
                coils={coil},
            )
            if plateau < FLOOR_MT_PER_M:
                rows[-1]["divergence"] = (
                    "a plateau under the floor, refused on spacing alone"
                )

    # FIESTA at a locked repetition time, and at repetition times no table locks
    for table in tables["tr"]:
        if not table["tr_us"]:
            continue
        lo, hi = table["tr_us"][0]
        locked = 0.5e-6 * (lo + hi)
        coil = coil_of(table["file"])
        orders = sorted(
            {
                k
                for band in vendor_bands(tables, {coil})
                for k in range(1, 12)
                if band["f_hz"][0] <= k / locked <= band["f_hz"][1]
            }
        )
        add(
            f"bSSFP, TR {locked * 1e3:.2f} ms (locked on {coil})",
            "refused",
            lambda locked=locked: _zoo("bssfp2D_sequence", n_x=128, tr=locked),
            f"harmonic {', '.join(str(k) for k in orders) or 'none'} of 1/TR in the coil's band",
            coils={coil},
        )
    for tr in (4.5e-3, 5.15e-3, 6.0e-3):
        if coils_locking_tr(tables, tr):
            continue
        add(
            f"bSSFP, TR {tr * 1e3:.2f} ms",
            "allowed",
            lambda tr=tr: _zoo("bssfp2D_sequence", n_x=128, tr=tr),
            "no table locks this TR",
        )

    # multi-echo gradient echo at the locked spacing, monopolar and bipolar
    esp_table = tables["esp"][0] if tables["esp"] else None
    if esp_table and esp_table["esp_us"]:
        coil = {coil_of(esp_table["file"])}
        for n_echoes, monopolar in ((4, True), (8, True), (4, False), (8, False)):
            found = multiecho_with_spacing_in(
                esp_table["esp_us"][0], n_echoes, monopolar
            )
            if found:
                seq, esp, _ = found
                kind = "monopolar" if monopolar else "bipolar"
                add(
                    f"multi-echo GRE, {n_echoes} {kind} echoes, ESP {esp * 1e3:.2f} ms (locked spacing)",
                    "refused",
                    seq,
                    "1/ESP in band",
                    coils=coil,
                )
                if n_echoes <= 4 and rows[-1]["sequence"] is not None:
                    rows[-1]["divergence"] = "a short packet against the memory"

    # what the vendor runs without a check, on every coil
    add(
        "3D GRE, TR 6 ms",
        "allowed",
        lambda: _zoo(
            "gre3D_sequence",
            n_x=128,
            n_y=64,
            n_z=16,
            tr=None,
            te=None,
            readout_bandwidth_hz=125e3,
        ),
    )
    add(
        "2D GRE, TR 9 ms",
        "allowed",
        lambda: _zoo(
            "gre2D_sequence",
            n_x=128,
            n_y=96,
            tr=9e-3,
            te=4.5e-3,
            readout_bandwidth_hz=83e3,
        ),
    )
    add("MPRAGE", "allowed", lambda: _zoo("mprage3D_sequence", n_x=96, n_y=32, n_z=16))
    add(
        "radial GRE, TR 10 ms",
        "allowed",
        lambda: _zoo("gre_radial2D_sequence", n_x=128, tr=10e-3),
    )
    add(
        "spiral, 8 arms",
        "allowed",
        lambda: _zoo("gre_spiral2D_sequence", WEAK, n_x=128, n_arms=8, tr=25e-3),
    )
    add(
        "stack of stars",
        "allowed",
        lambda: _zoo("gre_stack_of_stars3D_sequence", n_x=96, n_z=8, tr=10e-3),
    )
    if not quick:
        add(
            "stack of spirals",
            "allowed",
            lambda: _zoo("gre_stack_of_spirals3D_sequence", n_x=96, n_z=8),
        )
        add(
            "2D GRE, TR 100 ms",
            "allowed",
            lambda: _zoo(
                "gre2D_sequence",
                n_x=128,
                n_y=96,
                tr=100e-3,
                te=8e-3,
                readout_bandwidth_hz=83e3,
            ),
        )
        add("spin echo", "allowed", lambda: _zoo("se2D_sequence", n_x=128, n_y=96))
        add("FSE", "allowed", lambda: _zoo("fse2D_sequence", n_x=64, n_y=64))
        add("ZTE", "allowed", lambda: _zoo("zte3D_sequence", n_x=48))
        add("PROPELLER", "allowed", lambda: _zoo("se_propeller2D_sequence", n_x=64))

    # the edges the criterion was argued over
    for bandwidth in (60e3, 40e3):
        try:
            found = _zoo(
                "epi2D_sequence", WEAK, n_x=64, n_y=64, readout_bandwidth_hz=bandwidth
            )
        except Exception as exc:
            rows.append(
                {
                    "label": f"EPI 64, {bandwidth / 1e3:.0f} kHz",
                    "vendor": "edge",
                    "coils": None,
                    "sequence": None,
                    "note": str(exc)[:70],
                }
            )
            continue
        f0 = 0.5 / _echo_spacing_s(found)
        locking = coils_locking(tables, f0)
        add(
            f"EPI 64, {bandwidth / 1e3:.0f} kHz ({_readout_plateau_mt_per_m(found):.0f} mT/m plateau, f0 {f0:.0f} Hz)",
            "edge",
            found,
            f"a low plateau; locked on {', '.join(sorted(locking)) or 'no coil'}",
            coils=locking or None,
        )
    try:
        long_spiral = _zoo("gre_spiral2D_sequence", WEAK, n_x=128, n_arms=1, tr=None)
        add(
            f"spiral, 1 arm ({_readout_duration_s(long_spiral) * 1e3:.0f} ms readout)",
            "edge",
            long_spiral,
            "a sweep longer than the memory",
        )
    except Exception as exc:  # the design may refuse a single arm at this matrix
        rows.append(
            {
                "label": "spiral, 1 arm",
                "vendor": "edge",
                "coils": None,
                "sequence": None,
                "note": str(exc)[:80],
            }
        )
    add(
        "SPARKLING-like, 256 distinct arms",
        "edge",
        lambda: arms_scan(256),
        "past the group cap",
    )
    return rows


def coil_of(name: str) -> str:
    """The gradient coil a table file belongs to, from its name."""
    stem = (
        name.replace("epiesp", "").replace("greAcousticLimit", "").replace(".dat", "")
    )
    stem = stem.lstrip(".").split(".")[0] or "default"
    if stem.lower() == "esp":
        return "HRMW"  # the multi-echo lockout is the Rio (HRMw) coil's
    return stem.upper()


def vendor_bands(tables: dict, coils: set | None = None) -> list[dict]:
    """Every band the echo-spacing tables guard, per axis; with ``coils`` only
    the tables of those coils. The repetition-time and multi-echo lockouts
    name the same resonances through their own family's parameter, so they
    pick the designs the vendor refuses and are read in these bands."""
    out = []

    def wanted(name: str) -> bool:
        return coils is None or coil_of(name) in coils

    for table in tables["epi"]:
        if not wanted(table["file"]):
            continue
        for axis, bands in table["bands"].items():
            for band in bands:
                out.append(
                    {
                        "source": table["file"],
                        "axis": axis,
                        "f_hz": band["f_hz"],
                        "tolerance_mt_per_m": band["tolerance_mt_per_m"],
                    }
                )
    return out


def coils_locking(tables: dict, f_hz: float, axis: str = "x") -> set:
    """The coils whose echo-spacing table guards ``f_hz`` on ``axis``."""
    out = set()
    for table in tables["epi"]:
        for band in table["bands"][axis]:
            if band["f_hz"][0] <= f_hz <= band["f_hz"][1]:
                out.add(coil_of(table["file"]))
    return out


def coils_locking_tr(tables: dict, tr_s: float) -> set:
    out = set()
    for table in tables["tr"]:
        for lo, hi in table["tr_us"]:
            if lo * 1e-6 <= tr_s <= hi * 1e-6:
                out.add(coil_of(table["file"]))
    return out


def loudest_peaks(reading: dict, count: int = 3) -> list[dict]:
    """The largest readings anywhere in the territory, worst axis each."""
    amps = reading["amps"]
    flat = np.argsort(amps.max(axis=1))[::-1]
    picked = []
    for index in flat:
        f = float(reading["freqs"][index])
        if any(abs(f - p["f_hz"]) < 25.0 for p in picked):
            continue
        ax = int(np.argmax(amps[index]))
        picked.append(
            {"f_hz": f, "mt_per_m": float(amps[index, ax]), "axis": "xyz"[ax]}
        )
        if len(picked) == count:
            break
    return picked


def measure(rows: list[dict], tables: dict) -> list[dict]:
    axes = "xyz"
    every_coil = vendor_bands(tables)
    for row in rows:
        seq = row["sequence"]
        if seq is None:
            continue
        bands = vendor_bands(tables, row["coils"]) if row.get("coils") else every_coil
        t0 = time.perf_counter()
        reading = dense_reading(seq)
        row["seconds"] = time.perf_counter() - t0
        row["reading"] = reading
        row["peaks"] = loudest_peaks(reading)
        loudest = (0.0, None, None)
        for band in bands:
            for ax in range(3):
                if band["axis"] is not None and axes[ax] != band["axis"]:
                    continue
                value = in_band(reading, band["f_hz"], ax)
                if value > loudest[0]:
                    loudest = (value, band, ax)
        row["loudest_in_band_mt_per_m"] = loudest[0]
        row["loudest_band"] = None if loudest[1] is None else loudest[1]["f_hz"]
        row["loudest_source"] = None if loudest[1] is None else loudest[1]["source"]
        row["loudest_axis"] = None if loudest[2] is None else axes[loudest[2]]
        row["companions_mt_per_m"] = {
            axes[ax]: float(reading["amps"][:, ax].max()) for ax in range(3)
        }
        row["contributors"] = []
        if loudest[1] is not None:
            roles = describe_definitions(seq)
            for def_id, share in contributors_at(
                seq, loudest[1]["f_hz"], axes[loudest[2]]
            ):
                row["contributors"].append(
                    {"def_id": def_id, "share": share, "plays": roles.get(def_id, "?")}
                )
    return rows


def bracket(rows: list[dict]) -> dict:
    """The loudest reading the vendor runs against the quietest it refuses,
    the accepted divergences set aside and listed."""
    allowed = [r for r in rows if r["vendor"] == "allowed" and "reading" in r]
    refused = [
        r
        for r in rows
        if r["vendor"] == "refused" and "reading" in r and not r.get("divergence")
    ]
    divergences = [r for r in rows if r.get("divergence") and "reading" in r]
    return {
        "loudest_allowed_mt_per_m": max(r["loudest_in_band_mt_per_m"] for r in allowed)
        if allowed
        else None,
        "loudest_allowed": max(allowed, key=lambda r: r["loudest_in_band_mt_per_m"])[
            "label"
        ]
        if allowed
        else None,
        "quietest_refused_mt_per_m": min(r["loudest_in_band_mt_per_m"] for r in refused)
        if refused
        else None,
        "quietest_refused": min(refused, key=lambda r: r["loudest_in_band_mt_per_m"])[
            "label"
        ]
        if refused
        else None,
        "floor_mt_per_m": FLOOR_MT_PER_M,
        "accepted_divergences": [
            {
                "label": r["label"],
                "mt_per_m": r["loudest_in_band_mt_per_m"],
                "why": r["divergence"],
            }
            for r in divergences
        ],
    }


def memory_sensitivity(rows: list[dict]) -> list[dict]:
    """How the loudest in-band reading moves with the memory: only a sweep
    or a burst shorter than the memory should."""
    out = []
    for row in rows:
        if row.get("sequence") is None or row.get("loudest_band") is None:
            continue
        ax = "xyz".index(row["loudest_axis"])
        values = {}
        for memory in (0.010, 0.020, 0.040):
            values[f"{memory * 1e3:.0f} ms"] = in_band(
                dense_reading(row["sequence"], memory=memory), row["loudest_band"], ax
            )
        out.append({"label": row["label"], **values})
    return out


def derate_example() -> dict:
    """The 3D GRE built against a system derated to half its gradient
    amplitude: the same areas, every trapezoid longer, the in-band line
    lower; the readout, set by bandwidth and field of view, is untouched."""
    band = (532.0, 600.0)
    kwargs = {
        "n_x": 128,
        "n_y": 64,
        "n_z": 16,
        "tr": None,
        "te": None,
        "readout_bandwidth_hz": 125e3,
    }
    plain = _zoo("gre3D_sequence", SYSTEM, **kwargs)
    derated = _zoo(
        "gre3D_sequence",
        pp.apply_system_derates(SYSTEM, grad_derate=0.5, slew_derate=1.0),
        **kwargs,
    )
    before, after = dense_reading(plain), dense_reading(derated)
    return {
        "band_hz": band,
        "before": before,
        "after": after,
        "before_mt_per_m": max(in_band(before, band, ax) for ax in range(3)),
        "after_mt_per_m": max(in_band(after, band, ax) for ax in range(3)),
    }


def _shade(axis, bands, colour=FAINT):
    for band in bands:
        axis.axvspan(*band, color=colour, alpha=0.35, lw=0, zorder=0)


def figure_scenarios(rows: list[dict], result: dict) -> Path:
    drawn = [r for r in rows if "reading" in r]
    drawn.sort(key=lambda r: r["loudest_in_band_mt_per_m"])
    colours = {"refused": SERIES[1], "allowed": SERIES[0], "edge": MUTED}
    figure, axis = plt.subplots(figsize=(8.6, 0.42 * len(drawn) + 1.6), dpi=170)
    figure.subplots_adjust(left=0.42, right=0.97, top=0.9, bottom=0.12)
    y = np.arange(len(drawn))
    axis.barh(
        y,
        [r["loudest_in_band_mt_per_m"] for r in drawn],
        color=[colours[r["vendor"]] for r in drawn],
        height=0.62,
        zorder=2,
    )
    axis.axvline(FLOOR_MT_PER_M, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=3)
    axis.set_yticks(y)
    axis.set_yticklabels([r["label"] for r in drawn], fontsize=9.5)
    axis.set_xlabel("loudest reading inside a vendor band (mT/m, sustained sinusoid)")
    _style(axis, "")
    b = result["bracket"]
    axis.set_title(
        "Every family, read as the gate reads it. Orange: the vendor refuses it; "
        "blue: it runs unchecked; grey: an edge case.\n"
        f"Dashed: the zero-column floor. Bracket: {b['loudest_allowed_mt_per_m'] or 0:.1f} allowed, "
        f"{b['quietest_refused_mt_per_m'] or 0:.1f} refused.",
        fontsize=10.5,
        loc="left",
        color=INK,
    )
    out = ASSETS / "scenario_table.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_spectra(
    rows: list[dict], labels: list[str], name: str, bands_hz: list, title: str
) -> Path | None:
    picked = [r for r in rows if r["label"] in labels and "reading" in r]
    if not picked:
        return None
    figure, axes = plt.subplots(
        len(picked), 1, figsize=(8.6, 2.3 * len(picked) + 0.8), dpi=170, sharex=True
    )
    axes = np.atleast_1d(axes)
    figure.subplots_adjust(hspace=0.5, top=0.9, bottom=0.1, left=0.1, right=0.98)
    for axis, row in zip(axes, picked, strict=True):
        f = row["reading"]["freqs"]
        a = row["reading"]["amps"]
        keep = f <= 2000.0
        for ax_index, colour in (
            (0, SERIES[0]),
            (1, SERIES[2 % len(SERIES)]),
            (2, MUTED),
        ):
            axis.plot(
                f[keep],
                a[keep, ax_index],
                lw=0.8,
                color=colour,
                label="xyz"[ax_index],
                zorder=2,
            )
        _shade(axis, bands_hz)
        axis.axhline(FLOOR_MT_PER_M, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
        axis.set_ylim(0, max(1.2 * FLOOR_MT_PER_M, 1.05 * float(a[keep].max())))
        _style(
            axis,
            f"{row['label']}  —  {row['loudest_in_band_mt_per_m']:.1f} mT/m in band",
        )
        axis.set_ylabel("mT/m")
    axes[-1].set_xlabel("frequency (Hz)")
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    figure.suptitle(title, x=0.02, ha="left", fontsize=11.5, color=INK)
    out = ASSETS / name
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_derate(example: dict) -> Path:
    figure, axis = plt.subplots(figsize=(8.6, 3.2), dpi=170)
    figure.subplots_adjust(top=0.85, bottom=0.16, left=0.1, right=0.98)
    for reading, colour, label in (
        (
            example["before"],
            SERIES[1],
            f"as designed: {example['before_mt_per_m']:.1f} mT/m in band",
        ),
        (
            example["after"],
            SERIES[0],
            f"system derated to half its gradient amplitude: {example['after_mt_per_m']:.1f} mT/m",
        ),
    ):
        f, a = reading["freqs"], reading["amps"].max(axis=1)
        keep = f <= 1500.0
        axis.plot(f[keep], a[keep], lw=0.9, color=colour, label=label)
    _shade(axis, [example["band_hz"]])
    axis.axhline(FLOOR_MT_PER_M, color=INK, lw=0.8, ls=(0, (4, 3)))
    axis.legend(frameon=False, loc="upper right")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("mT/m, worst axis")
    _style(
        axis,
        "3D GRE, TR 6 ms: the spoiler and prewinder carry the line; a derated system moves it",
    )
    out = ASSETS / "derate_example.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


def figure_scale() -> Path | None:
    if not SCALE.exists():
        return None
    entries = json.loads(SCALE.read_text())
    if not entries:
        return None
    figure, axis = plt.subplots(figsize=(8.6, 3.4), dpi=170)
    figure.subplots_adjust(top=0.88, bottom=0.16, left=0.1, right=0.98)
    for dims, colour in ((2, SERIES[0]), (3, SERIES[1])):
        rows = sorted(
            (e for e in entries if e["dims"] == dims), key=lambda e: e["arms"]
        )
        if not rows:
            continue
        arms = np.array([e["arms"] for e in rows], float)
        secs = np.array([e["mech_s"] for e in rows], float)
        axis.plot(
            arms, secs, "o-", color=colour, lw=1.0, label=f"{dims}D arms, measured"
        )
        axis.plot(
            [arms[-1], 131072],
            [secs[-1], secs[-1] * 131072 / arms[-1]],
            ls=(0, (3, 3)),
            color=colour,
            lw=0.8,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("distinct arms")
    axis.set_ylabel("check alone (s)")
    axis.legend(frameon=False)
    _style(
        axis,
        "The mechanical-resonance check on a scan of distinct 4096-sample arms; dotted: linear to 128K",
    )
    out = ASSETS / "scale.png"
    figure.savefig(out, facecolor="white")
    plt.close(figure)
    return out


# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="skip the slowest families"
    )
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    tables = load_tables()
    bands = vendor_bands(tables)
    print(
        f"{len(tables['epi'])} echo-spacing tables, {len(tables['tr'])} TR tables, {len(tables['esp'])} ESP tables: {len(bands)} bands"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = measure(scenarios(tables, args.quick), tables)
        result = {
            "floor_mt_per_m": FLOOR_MT_PER_M,
            "territory_hz": TERRITORY,
            "stated_tolerances": stated_tolerances(tables),
            "bracket": bracket(rows),
            "scenarios": [
                {
                    k: (sorted(v) if k == "coils" and v else v)
                    for k, v in r.items()
                    if k not in ("sequence", "reading")
                }
                for r in rows
            ],
            "memory": memory_sensitivity(
                [
                    r
                    for r in rows
                    if r["vendor"] == "edge"
                    or "bSSFP" in r["label"]
                    or "EPI 32" in r["label"]
                ]
            ),
        }
        example = derate_example()
        result["derate_example"] = {
            "band_hz": example["band_hz"],
            "before_mt_per_m": example["before_mt_per_m"],
            "after_mt_per_m": example["after_mt_per_m"],
        }
    b = result["bracket"]
    print(
        f"bracket: loudest allowed {b['loudest_allowed_mt_per_m']:.2f} mT/m ({b['loudest_allowed']}); "
        f"quietest refused {b['quietest_refused_mt_per_m']:.2f} mT/m ({b['quietest_refused']}); floor {b['floor_mt_per_m']}"
    )
    for d in b["accepted_divergences"]:
        print(
            f"  accepted divergence: {d['label']}: {d['mt_per_m']:.2f} mT/m -- {d['why']}"
        )
    for tol in result["stated_tolerances"]:
        print(
            f"  stated {tol['plateau_mt_per_m']:.0f} mT/m plateau on {tol['file']} {tol['axis']} -> {tol['sinusoid_mt_per_m']:.1f} mT/m sinusoid"
        )
    print(f"{'scenario':<58s} {'vendor':<8s} {'in band':>8s}  band / axis / who")
    for r in result["scenarios"]:
        if "loudest_in_band_mt_per_m" not in r:
            print(f"{r['label']:<58s} {r['vendor']:<8s}  (not built: {r['note']})")
            continue
        who = "; ".join(
            f"def {c['def_id']} {c['share'] * 100:.0f}% ({c['plays']})"
            for c in r["contributors"][:2]
        )
        band = r["loudest_band"]
        band_s = f"{band[0]:.0f}-{band[1]:.0f} Hz" if band else "-"
        peaks = ", ".join(
            f"{p['mt_per_m']:.1f}@{p['f_hz']:.0f}{p['axis']}"
            for p in r.get("peaks", [])
        )
        print(
            f"{r['label']:<70s} {r['vendor']:<8s} {r['loudest_in_band_mt_per_m']:8.2f}  "
            f"{band_s} / {r['loudest_axis']} / {who}  | peaks {peaks}"
        )
    print("memory sensitivity (loudest in-band reading at 10 / 20 / 40 ms):")
    for m in result["memory"]:
        print(
            f"  {m['label']:<58s} "
            + "  ".join(f"{k}: {v:.2f}" for k, v in m.items() if k != "label")
        )
    print(
        f"derate example: {example['before_mt_per_m']:.2f} -> {example['after_mt_per_m']:.2f} mT/m in {example['band_hz']}"
    )
    figures = [
        figure_scenarios(rows, result),
        figure_spectra(
            rows,
            [
                r["label"]
                for r in rows
                if r["vendor"] == "refused" and r["label"].startswith("EPI")
            ][:2]
            + [
                r["label"]
                for r in rows
                if "(locked on" in r["label"] and "bSSFP" in r["label"]
            ][:1]
            + [
                r["label"]
                for r in rows
                if r["vendor"] == "allowed" and r["label"].startswith("bSSFP")
            ][:1],
            "epi_fiesta.png",
            [b["f_hz"] for b in bands if b["axis"] in (None, "x")],
            "The vendor's refusals and a free TR, read per axis. Shaded: every band the tables guard on x or on the readout pair.",
        ),
        figure_spectra(
            rows,
            [r["label"] for r in rows if r["vendor"] == "edge"] + ["spiral, 8 arms"],
            "long_events.png",
            [b["f_hz"] for b in bands if b["axis"] in (None, "x")],
            "Sweeps and bursts: what a 20 ms memory prices on a short spiral, a long one and distinct arms.",
        ),
        figure_derate(example),
        figure_scale(),
    ]
    RESULTS.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=float) + "\n"
    )
    if INFEASIBLE:
        print(
            f"{len(INFEASIBLE)} sweep points were infeasible designs, e.g. {INFEASIBLE[0]}"
        )
    print("wrote", RESULTS.relative_to(RESULTS.parents[2]))
    for f in figures:
        if f is not None:
            print("wrote", f.relative_to(f.parents[4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
