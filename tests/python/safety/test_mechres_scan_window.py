"""The mechanical-resonance scan window: a scan past the shape-group cap.

A scan with more distinct waveform sets than the grouping holds has no TR to
repeat, so the check prices what a mode sees: the amplitude sustained inside
its band over a window of its memory, slid over the whole scan. These tests
build such a scan from distinct arbitrary arms and verify the verdict both
ways: a modest scan passes with its spectrum reported on a fine grid over
every band, and a scan whose arms sustain a tone inside a band is refused.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver._ext.pulseg import _PulseqCollection, _check_safety_profiled

BANDS = [(550.0, 650.0, 0.0), (1100.0, 1250.0, 0.0)]  # 0: the policy amplitude
GAMMA_HZ_PER_T = 42.576e6
N_ARMS = 80  # past PULSEG__MAX_SHAPE_GROUPS (64)
N_SAMPLES = 1024
RASTER_S = 4e-6


def _system() -> pp.Opts:
    return pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=RASTER_S,
        rf_raster_time=2e-6,
        adc_raster_time=2e-6,
        block_duration_raster=RASTER_S,
    )


def _arm(k: int, rng: np.random.Generator, tone_hz_per_m: float) -> np.ndarray:
    """One arm: a swept spread of content plus a tone at 590 Hz of the given amplitude."""
    t = np.arange(N_SAMPLES) * RASTER_S
    envelope = np.sin(np.pi * np.arange(N_SAMPLES) / (N_SAMPLES - 1)) ** 2
    spread = np.sin(2 * np.pi * (200.0 * t + 1.0e5 * t * t) + rng.uniform(0, 2 * np.pi))
    spread += np.sin(
        2 * np.pi * (300.0 + 50.0 * (k % 7)) * t + rng.uniform(0, 2 * np.pi)
    )
    w = 0.05e6 * spread * envelope
    w += (
        tone_hz_per_m
        * np.sin(2 * np.pi * 590.0 * t + rng.uniform(0, 2 * np.pi))
        * envelope
    )
    return w


def _collection(tone_hz_per_m: float, tmp: pathlib.Path) -> _PulseqCollection:
    system = _system()
    seq = pp.Sequence(system=system)
    rng = np.random.default_rng(11)
    rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=system)
    adc = pp.make_adc(num_samples=N_SAMPLES, dwell=RASTER_S, system=system)
    for k in range(N_ARMS):
        gx = pp.make_arbitrary_grad(
            channel="x", waveform=_arm(k, rng, tone_hz_per_m), system=system
        )
        gy = pp.make_arbitrary_grad(
            channel="y", waveform=0.7 * _arm(k + 100, rng, tone_hz_per_m), system=system
        )
        seq.add_block(rf)
        seq.add_block(gx, gy, adc)
    seq.declare_tr()
    path = tmp / f"arms_{tone_hz_per_m:.0f}.seq"
    path.write_bytes(seq._to_binary())
    s = system
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


def _check(coll: _PulseqCollection) -> dict:
    return _check_safety_profiled(coll, BANDS, 4.25e8 / 0.333, 360.0, 1e6, False)


@pytest.fixture(scope="module")
def tmp_dir() -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp())


def test_a_modest_scan_past_the_cap_passes_the_mechanical_resonance_check(tmp_dir):
    coll = _collection(0.02e6, tmp_dir)
    result = _check(coll)
    assert result["code"] > 0, result  # PULSEG_SUCCESS
    stage = result["stages"]["mech_resonance"]
    assert stage["seconds"] < 20.0


def test_a_scan_past_the_cap_sustaining_a_tone_in_a_band_is_refused(tmp_dir):
    # 1.2e6 Hz/m ~ 28 mT/m at 590 Hz, sustained through every arm: well over
    # the band's policy amplitude, and inside 550-650 Hz.
    coll = _collection(1.2e6, tmp_dir)
    result = _check(coll)
    assert result["code"] == -404, result  # PULSEG_ERR_MECH_RESONANCES_VIOLATION
    assert "Hz/m" in result["message"] and ">" in result["message"], result["message"]


def test_the_tone_is_refused_only_when_it_lies_inside_a_band(tmp_dir):
    """The same tone amplitude at 900 Hz, between the bands, passes."""
    system = _system()
    seq = pp.Sequence(system=system)
    rng = np.random.default_rng(5)
    rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=system)
    adc = pp.make_adc(num_samples=N_SAMPLES, dwell=RASTER_S, system=system)
    t = np.arange(N_SAMPLES) * RASTER_S
    envelope = np.sin(np.pi * np.arange(N_SAMPLES) / (N_SAMPLES - 1)) ** 2
    for k in range(N_ARMS):
        w = 0.8e6 * np.sin(2 * np.pi * 900.0 * t + rng.uniform(0, 2 * np.pi)) * envelope
        w += (
            0.05e6
            * np.sin(
                2 * np.pi * (300.0 + 50.0 * (k % 7)) * t + rng.uniform(0, 2 * np.pi)
            )
            * envelope
        )
        gx = pp.make_arbitrary_grad(channel="x", waveform=w, system=system)
        seq.add_block(rf)
        seq.add_block(gx, adc)
    seq.declare_tr()
    path = tmp_dir / "arms_between.seq"
    path.write_bytes(seq._to_binary())
    s = system
    coll = _PulseqCollection(
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
    result = _check(coll)
    assert result["code"] > 0, result  # PULSEG_SUCCESS
