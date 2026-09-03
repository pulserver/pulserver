"""The terms of a mechanical-resonance verdict.

What a refusal names, what a stated tolerance means for the train that drives
a band, how a bound is settled, what the memory prices, and that the verdict
does not depend on the grid it started from or on which axis a shared
waveform is read on.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest
from pulserver._ext.pulseg import (
    _PulseqCollection,
    _calc_mech_resonances,
    _check_safety,
    _mech_scan_window_probe,
)

import pulserver.pypulseq as pp

from .conftest import CORPUS, EXPECTED, build_collection

GAMMA_HZ_PER_T = 42.576e6
RASTER_S = 4e-6
SYSTEM = pp.Opts(
    max_grad=50,
    grad_unit="mT/m",
    max_slew=200,
    slew_unit="T/m/s",
    grad_raster_time=RASTER_S,
    rf_raster_time=2e-6,
    adc_raster_time=2e-6,
    block_duration_raster=RASTER_S,
)
STATED_UNIT_HZ_PER_M = 1.0


def _collection(seq: pp.Sequence, where: pathlib.Path, name: str) -> _PulseqCollection:
    seq.declare_tr()
    path = where / f"{name}.seq"
    path.write_bytes(seq._to_binary())
    s = seq.system
    return _PulseqCollection(
        str(path),
        float(s.gamma),
        float(s.B0),
        float(s.max_grad),
        float(s.max_slew),
        float(s.rf_raster_time),
        float(s.grad_raster_time),
        float(s.adc_raster_time),
        float(s.block_duration_raster),
    )


def _taper(n: int, samples: int = 250) -> np.ndarray:
    """Ones with a raised-cosine fall to zero over the first and last samples,
    so a synthetic waveform starts and ends where a gradient must."""
    edge = 0.5 * (1.0 - np.cos(np.pi * np.arange(samples) / samples))
    out = np.ones(n)
    out[:samples] = edge
    out[-samples:] = edge[::-1]
    return out


def _refusal(collection, bands) -> str | None:
    """The refusal message, or None when the bands pass."""
    try:
        _check_safety(collection, bands, 0.0, 0.0, 100.0, True)
    except RuntimeError as exc:
        return str(exc)
    return None


def _factor(collection, band) -> float:
    """The sinusoid amplitude the engine allows per unit of stated plateau."""
    spectra = _calc_mech_resonances(
        collection,
        0,
        0,
        0,
        target_resolution_hz=1.0,
        max_freq_hz=3000.0,
        forbidden_bands=[(band[0], band[1], STATED_UNIT_HZ_PER_M)],
    )
    return float(np.asarray(spectra["candidate_eps"], float)[0]) / STATED_UNIT_HZ_PER_M


def _train(
    where: pathlib.Path, name: str, alternating: bool, flat_s: float, rise_s: float
):
    """Two repetitions of a 32-lobe trapezoid train, one lobe every period."""
    seq = pp.Sequence(system=SYSTEM)
    amplitude = 0.020 * GAMMA_HZ_PER_T
    rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=SYSTEM)
    for _ in range(2):
        seq.add_block(rf)
        for k in range(32):
            sign = -1.0 if (alternating and k % 2) else 1.0
            seq.add_block(
                pp.make_trapezoid(
                    channel="x",
                    amplitude=sign * amplitude,
                    flat_time=flat_s,
                    rise_time=rise_s,
                    system=SYSTEM,
                )
            )
    period_s = flat_s + 2 * rise_s
    return _collection(seq, where, name), period_s


def _fundamental_ratio(
    alternating: bool, flat_s: float, rise_s: float
) -> tuple[float, float]:
    """The lobe pattern repeated without end: fundamental amplitude per unit
    plateau, and the frequency it sits at, from a rendered period."""
    dt = 1e-7
    period = flat_s + 2 * rise_s
    t = np.arange(0.0, period, dt)
    lobe = np.interp(t, [0.0, rise_s, rise_s + flat_s, period], [0.0, 1.0, 1.0, 0.0])
    pattern = np.concatenate([lobe, -lobe]) if alternating else lobe
    T = pattern.size * dt
    f0 = 1.0 / T
    tt = np.arange(pattern.size) * dt
    coefficient = 2.0 / T * abs(np.sum(pattern * np.exp(-2j * np.pi * f0 * tt)) * dt)
    return float(coefficient), f0


# ----------------------------------------------------------------------
# What a refusal names


def test_a_refusal_names_the_definitions_behind_it():
    system = pp.Opts(max_grad=80.0, grad_unit="mT/m", max_slew=400.0, slew_unit="T/m/s")
    bands = [(550.0, 650.0, 0.0)]
    message = _refusal(build_collection(CORPUS / "bssfp_2d.seq", system), bands)
    assert message is not None and "def=" in message and "share=" in message, message

    seq = pp.Sequence(system=system)
    seq.read(CORPUS / "bssfp_2d.seq")
    overlay = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=3000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=bands,
    )[4]
    assert not overlay.ok
    assert overlay.contributors, "a refused reading names its contributors"
    assert overlay.contributor_axis in (0, 1, 2)
    assert 550.0 <= overlay.contributor_freq <= 650.0
    shares = [share for _, share in overlay.contributors]
    assert shares == sorted(shares, reverse=True)
    # the reading is linear in the events: the named definitions carry it
    assert sum(shares) >= 0.9
    assert overlay.tolerance.shape == overlay.candidate_freqs.shape
    assert np.all(overlay.tolerance > 0.0)


def test_nothing_is_named_when_nothing_is_refused():
    system = pp.Opts(max_grad=80.0, grad_unit="mT/m", max_slew=400.0, slew_unit="T/m/s")
    seq = pp.Sequence(system=system)
    seq.read(CORPUS / "gre_2d.seq")
    overlay = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=3000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=[(550.0, 650.0, 0.0)],
    )[4]
    assert overlay.ok
    assert overlay.contributors == []
    assert overlay.contributor_axis == -1


# ----------------------------------------------------------------------
# A stated tolerance speaks the shape of the train that drives the band


@pytest.mark.parametrize("alternating", [True, False], ids=["alternating", "same_sign"])
def test_a_stated_plateau_is_converted_through_the_trains_own_fundamental(
    tmp_path, alternating
):
    """Square-ish lobes: 4/pi per unit plateau for an alternating train, 2/pi
    for a same-sign train at half the duty, both far from the 8/pi^2 of a
    triangle. The engine reads the ratio off the lobe transform."""
    flat_s, rise_s = 0.8e-3, 0.1e-3
    collection, _ = _train(
        tmp_path, f"train_{alternating}", alternating, flat_s, rise_s
    )
    expected, f0 = _fundamental_ratio(alternating, flat_s, rise_s)
    factor = _factor(collection, (f0 - 50.0, f0 + 50.0))
    assert abs(factor - expected) <= 0.02 * expected, (factor, expected, f0)
    assert abs(factor - 8.0 / math.pi**2) > 0.1


def test_a_band_no_train_fundamental_falls_in_keeps_the_triangle_factor(tmp_path):
    collection, period = _train(tmp_path, "train_off_band", True, 0.8e-3, 0.1e-3)
    f0 = 0.5 / period
    factor = _factor(collection, (2.6 * f0, 2.9 * f0))
    assert abs(factor - 8.0 / math.pi**2) <= 1e-3


# ----------------------------------------------------------------------
# A bound that refuses is settled on the scan itself


def test_a_refused_bound_is_settled_by_reading_the_scan():
    """The tiled TR prices every varying position at its magnitude: a bound.
    Where that bound refuses, the scan is read exactly, and a tolerance the
    bound alone would refuse passes when the scan itself sustains less."""
    from pulserver.app import gre2D_sequence

    # the default rasters: on this protocol the phase-encode bound sits well
    # above what the scan sustains, so there is something to settle
    system = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    made = gre2D_sequence.main(
        system=system, n_dummy=0, n_x=128, n_y=96, tr=9e-3, te=4.5e-3
    )
    sequence = made[0] if isinstance(made, tuple) else made
    collection = sequence._structure_for("bound").collection
    band = (500.0, 600.0)

    def peak(tolerance_hz_per_m):
        spectra = _calc_mech_resonances(
            collection,
            0,
            0,
            0,
            target_resolution_hz=1.0,
            max_freq_hz=3000.0,
            forbidden_bands=[(band[0], band[1], tolerance_hz_per_m)],
        )
        amps = np.asarray(spectra["candidate_grad_amps"], float)
        return float(amps.max()) if amps.size else 0.0

    bound = peak(0.0)  # a zero band the bound passes: the bound is reported
    exact = peak(STATED_UNIT_HZ_PER_M)  # refused on the bound: the scan is read
    assert 0.0 < exact <= bound * 1.001, (exact, bound)
    if exact > 0.985 * bound:
        pytest.skip("the bound is tight on this protocol; nothing to settle")
    factor = _factor(collection, band)

    def passes(sinusoid_hz_per_m):
        return (
            _refusal(collection, [(band[0], band[1], sinusoid_hz_per_m / factor)])
            is None
        )

    assert passes(0.99 * bound), "the bound refuses here; the scan does not"
    assert not passes(0.99 * exact)


# ----------------------------------------------------------------------
# What the memory prices


def test_the_memory_prices_a_sweep_and_leaves_a_comb_alone(tmp_path):
    """A chirp crosses a band once: what it sustains scales with how long the
    mode remembers. A comb sustains its line regardless."""
    seq = pp.Sequence(system=SYSTEM)
    rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=SYSTEM)
    n = 10000  # 40 ms
    t = np.arange(n) * RASTER_S
    chirp = 0.010 * GAMMA_HZ_PER_T * np.sin(2 * np.pi * (400.0 * t + 15000.0 * t * t))
    chirp *= _taper(n)
    for _ in range(2):
        seq.add_block(rf)
        seq.add_block(
            pp.make_arbitrary_grad(channel="x", waveform=chirp, system=SYSTEM)
        )
    seq.declare_tr()

    def in_band(sequence, memory):
        overlay = sequence.calculate_gradient_spectrum(
            plot=False,
            max_frequency=2000.0,
            tr="worst_case",
            resonance_lines=True,
            bands=[(550.0, 650.0, 0.0)],
            memory=memory,
        )[4]
        return float(overlay.candidate_a_eq.max())

    ratio = in_band(seq, 0.010) / in_band(seq, 0.020)
    assert 1.5 <= ratio <= 2.5, ratio

    system = pp.Opts(max_grad=80.0, grad_unit="mT/m", max_slew=400.0, slew_unit="T/m/s")
    comb = pp.Sequence(system=system)
    comb.read(CORPUS / "bssfp_2d.seq")
    # A window takes whole the events that start inside it, so a comb reads
    # its line times at most one element's share of the memory more; halving
    # the memory moves that by a few percent, not by the factor a sweep shows.
    ratio = in_band(comb, 0.010) / in_band(comb, 0.020)
    assert 0.9 <= ratio <= 1.2, ratio


# ----------------------------------------------------------------------
# The verdict does not depend on the grid it started from


@pytest.mark.parametrize("name", ["epi_2d_main.seq", "bssfp_2d.seq", "fse_2d.seq"])
@pytest.mark.parametrize(
    "bands",
    [
        [(550.0, 650.0, 0.0), (1100.0, 1250.0, 0.0)],
        [(500.0, 1500.0, 0.008 * GAMMA_HZ_PER_T)],
        [(100.0, 2000.0, 0.001 * GAMMA_HZ_PER_T, "x")],
    ],
    ids=["zero", "stated", "stated_x"],
)
def test_the_gate_and_the_drawn_verdict_agree(name, bands):
    """The gate starts on a coarse grid and refines only what its bound
    cannot settle; the drawn verdict reads every band on the fine grid."""
    system = pp.Opts(max_grad=80.0, grad_unit="mT/m", max_slew=400.0, slew_unit="T/m/s")
    gate_ok = _refusal(build_collection(CORPUS / name, system), bands) is None
    seq = pp.Sequence(system=system)
    seq.read(CORPUS / name)
    overlay = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=3000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=bands,
    )[4]
    assert overlay.ok is gate_ok


# ----------------------------------------------------------------------
# A shared waveform reads the same on every axis it is rotated onto


def test_shared_records_read_as_direct_transforms_under_a_prescription():
    system = pp.Opts(max_grad=40.0, grad_unit="mT/m", max_slew=150.0, slew_unit="T/m/s")
    collection = build_collection(EXPECTED / "arms_scan.seq", system)
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    collection.set_prescription_rotation([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
    grids = [(560.0, 5.0, 21), (1120.0, 10.0, 16)]
    records = _mech_scan_window_probe(collection, grids, 20000.0, 0, 0)
    direct = _mech_scan_window_probe(collection, grids, 20000.0, 1, 0)
    for axis in ("amp_gx", "amp_gy"):
        a = np.asarray(records[axis], float)
        b = np.asarray(direct[axis], float)
        loud = b > 0.01 * b.max()
        assert np.allclose(a[loud], b[loud], rtol=0.02), axis


# ----------------------------------------------------------------------
# A long event is read at its loudest stretch


def test_a_long_event_is_read_at_its_loudest_stretch(tmp_path):
    """A 60 ms tone whose amplitude ramps up: the memory holds a third of it,
    and the reading is what the loudest third sustains, not the average
    over the whole event. The reference slides the memory over the samples
    on the same grid of eighth-memory pieces."""
    n = 15000
    t = np.arange(n) * RASTER_S
    f_tone = 600.0
    ramp = t / t[-1]
    wave = 0.010 * GAMMA_HZ_PER_T * ramp * np.sin(2 * np.pi * f_tone * t) * _taper(n)
    seq = pp.Sequence(system=SYSTEM)
    rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=SYSTEM)
    for _ in range(2):
        seq.add_block(rf)
        seq.add_block(pp.make_arbitrary_grad(channel="x", waveform=wave, system=SYSTEM))
    collection = _collection(seq, tmp_path, "ramped_tone")
    memory_us = 20000.0
    reading = float(
        np.asarray(
            _mech_scan_window_probe(collection, [(f_tone, 1.0, 1)], memory_us, 0, 0)[
                "amp_gx"
            ]
        )[0]
    )

    # the reference: pieces of an eighth of the memory, windows opening at
    # piece starts and taking whole every piece that starts inside them
    duration_us = (n - 1) * RASTER_S * 1e6
    pieces = math.ceil(duration_us / (memory_us / 8.0))
    per_piece = math.ceil(n / pieces)
    starts = [p * per_piece for p in range(pieces)]
    phasor = np.exp(-2j * np.pi * f_tone * t) * wave * RASTER_S
    best = 0.0
    for p, s0 in enumerate(starts):
        q = p
        while q + 1 < pieces and (starts[q + 1] - s0) * RASTER_S * 1e6 < memory_us:
            q += 1
        s1 = min(n, starts[q] + per_piece)
        best = max(best, 2.0 / (memory_us * 1e-6) * abs(phasor[s0:s1].sum()))
    assert abs(reading - best) <= 0.03 * best, (reading, best)
    # the whole event averaged over the memory would say something else
    whole = 2.0 / (memory_us * 1e-6) * abs(phasor.sum())
    assert abs(reading - whole) > 0.2 * reading


def test_a_wide_band_on_one_axis_is_evaluated():
    system = pp.Opts(max_grad=80.0, grad_unit="mT/m", max_slew=400.0, slew_unit="T/m/s")
    seq = pp.Sequence(system=system)
    seq.read(CORPUS / "epi_2d_main.seq")
    overlay = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=4000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=[(0.0, 4000.0, 0.0, "z")],
    )[4]
    assert overlay.candidate_freqs.size >= 8192
    assert overlay.ok
