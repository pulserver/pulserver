"""Unit tests for pulserver private system."""

from __future__ import annotations

import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import _system as system  # noqa: E402


@pytest.mark.parametrize(
    ("nx", "bw_hz_px"),
    [(64, 62_500.0), (128, 125_000.0), (256, 31_250.0), (200, 100_000.0)],
)
def test_quantize_readout_timing_rasters(nx: int, bw_hz_px: float) -> None:
    grad_raster = 1e-5
    adc_raster = 1e-7
    dwell, flat = system.quantize_readout_timing(nx, bw_hz_px, grad_raster, adc_raster, 0.0)

    assert abs(flat / grad_raster - round(flat / grad_raster)) < 1e-9
    assert abs(dwell / adc_raster - round(dwell / adc_raster)) < 1e-9
    assert flat == pytest.approx(nx * dwell)
    assert abs(dwell - 1.0 / bw_hz_px) <= grad_raster + 1e-12


def test_apply_system_derates_idempotent() -> None:
    opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    base_grad = float(opts.max_grad)
    base_slew = float(opts.max_slew)

    system.apply_system_derates(opts)
    once_grad = float(opts.max_grad)
    once_slew = float(opts.max_slew)
    system.apply_system_derates(opts)

    assert opts.max_grad == pytest.approx(once_grad)
    assert opts.max_slew == pytest.approx(once_slew)
    assert once_grad == pytest.approx(0.9 * base_grad)
    assert once_slew == pytest.approx(0.9 * base_slew)


def test_round_to_raster_default() -> None:
    assert system.round_to_raster(1.23456e-3) == pytest.approx(1.23e-3)
    assert system.round_to_raster(1.236e-3) == pytest.approx(1.24e-3)


def test_ceil_to_raster() -> None:
    assert system.ceil_to_raster(1.01e-5, 1e-5) == pytest.approx(2e-5)
    assert system.ceil_to_raster(2e-5, 1e-5) == pytest.approx(2e-5)


def test_copy_event_is_deep() -> None:
    opts = pp.Opts()
    trap = pp.make_trapezoid(channel="x", area=100.0, system=opts)
    dup = system.copy_event(trap)
    dup.amplitude = trap.amplitude * 2
    assert trap.amplitude != dup.amplitude
