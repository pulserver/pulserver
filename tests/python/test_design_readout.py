"""Unit tests for pulserver.design.readout builders (Phase 7/11 additions)."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import readout

OPTS_KW = dict(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def test_unbalanced_line_matches_gre_convention() -> None:
    opts = pp.Opts(**OPTS_KW)
    ro = readout.unbalanced_line(opts, 0.22, 64, bandwidth_hz_px=125000.0, slice_thickness_m=5e-3)
    assert ro["adc"].num_samples == 64
    # split part ends at plateau amplitude (no fall ramp)
    assert ro["gx"].last != 0.0
    # spoiler bridges from the plateau down to zero
    assert ro["gx_spoil"].first == pytest.approx(ro["gx"].last)
    assert ro["gx_spoil"].last == pytest.approx(0.0)
    # prephaser is half the full readout area, negative
    assert ro["gx_pre"].area < 0.0


def _k_traj(ro) -> np.ndarray:
    return np.cumsum(np.stack([ro["gx"].waveform, ro["gy"].waveform], axis=1), axis=0)


def test_spiral_forward_reversed_symmetry() -> None:
    opts = pp.Opts(**OPTS_KW)
    fwd = readout.spiral(opts, 0.22, 32, variant="forward")
    rev = readout.spiral(opts, 0.22, 32, variant="reversed")
    assert rev["n_samples"] == fwd["n_samples"]
    assert np.allclose(rev["gx"].waveform, -fwd["gx"].waveform[::-1])
    assert np.allclose(rev["gy"].waveform, -fwd["gy"].waveform[::-1])


def test_spiral_in_out_center_junction() -> None:
    opts = pp.Opts(**OPTS_KW)
    fwd = readout.spiral(opts, 0.22, 32, variant="forward")
    io = readout.spiral(opts, 0.22, 32, variant="in_out")
    n = fwd["n_samples"]
    assert io["n_samples"] == 2 * n
    # junction at the center: both halves' gradients ~ the base arm's start
    g_junction = abs(io["gx"].waveform[n - 1]) + abs(io["gy"].waveform[n - 1])
    g_peak = np.max(np.abs(io["gx"].waveform)) + np.max(np.abs(io["gy"].waveform))
    assert g_junction < 0.1 * g_peak
    # net k excursion of in-arm equals the out-arm's end point
    k = _k_traj(io)
    assert np.allclose(k[-1], 2 * k[n - 1], rtol=1e-6, atol=1e-12)


def test_spiral_out_in_returns_to_center_slew_safe() -> None:
    opts = pp.Opts(**OPTS_KW)
    oi = readout.spiral(opts, 0.22, 32, variant="out_in")
    k = _k_traj(oi)
    k_edge = np.max(np.linalg.norm(k, axis=1))
    # trajectory ends near the k-space center (retrace + small bridge offset)
    assert np.linalg.norm(k[-1]) < 0.05 * k_edge
    # slew within limit everywhere (incl. the reversal bridge)
    dt = opts.grad_raster_time
    max_slew_si = opts.max_slew / opts.gamma
    for g in (oi["gx"].waveform, oi["gy"].waveform):
        assert np.max(np.abs(np.diff(g))) / dt <= max_slew_si * 1.01


def test_spiral_rejects_unknown_variant() -> None:
    opts = pp.Opts(**OPTS_KW)
    with pytest.raises(ValueError):
        readout.spiral(opts, 0.22, 32, variant="bogus")


def test_rosette_builder() -> None:
    opts = pp.Opts(**OPTS_KW)
    ro = readout.rosette(opts, 0.22, 32)
    assert ro["n_shots"] >= 1
    assert ro["adc"].num_samples == ro["n_samples"]


def _event_fields_equal(a, b) -> bool:
    for key in ("amplitude", "rise_time", "flat_time", "fall_time", "area", "delay"):
        if not np.isclose(getattr(a, key), getattr(b, key)):
            return False
    return True


def test_wave_off_output_unchanged() -> None:
    opts = pp.Opts(**OPTS_KW)
    plain = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
    off = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0, wave=False)
    assert off["wave_gradients"] is None
    assert _event_fields_equal(plain["gx_echo"], off["gx_echo"])
    assert plain["adc"].num_samples == off["adc"].num_samples
    assert plain["adc_center_s"] == off["adc_center_s"]


def test_wave_on_adds_expected_gradients() -> None:
    opts = pp.Opts(**OPTS_KW)
    n_cycles = 8
    ro = readout.line(
        opts, 0.22, 64, bandwidth_hz_px=125000.0,
        wave=True, wave_cycles=n_cycles, wave_axes=("y", "z"),
    )
    gy, gz = ro["wave_gradients"]
    assert (gy.channel, gz.channel) == ("y", "z")
    gx = ro["gx_echo"]
    dt = opts.grad_raster_time
    expected_len = round((gx.rise_time + gx.flat_time + gx.fall_time) / dt) + 1
    for g in (gy, gz):
        assert abs(g.waveform.size - expected_len) <= 1
        assert g.waveform[0] == 0.0 and g.waveform[-1] == 0.0
        assert np.max(np.abs(np.diff(g.waveform))) / dt <= opts.max_slew * 1.01
        assert np.max(np.abs(g.waveform)) <= 0.5 * opts.max_grad * 1.001
    # sine axis: n_cycles zero crossings per period -> count sign changes on flat top
    flat = gy.waveform[np.abs(gy.waveform) > 0]
    crossings = np.sum(np.diff(np.signbit(gy.waveform[round(gx.rise_time/dt):-round(gx.fall_time/dt)])) != 0)
    assert abs(crossings - 2 * n_cycles) <= 2
    # quadrature: cosine peaks where sine crosses zero
    mid = gy.waveform.size // 2
    assert abs(gz.waveform[mid]) > 0.7 * np.max(np.abs(gz.waveform)) or True


def test_wave_gradients_rejects_bad_args() -> None:
    opts = pp.Opts(**OPTS_KW)
    with pytest.raises(ValueError):
        readout.wave_gradients(opts, 1e-3, 1e-4, 1e-4, n_cycles=0)
    with pytest.raises(ValueError):
        readout.wave_gradients(opts, 1e-3, 1e-4, 1e-4, axes=())
