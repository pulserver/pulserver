"""Unit tests for pulserver.design.encoding."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import encoding


def test_phase_encode_gradient_defaults() -> None:
    opts = pp.Opts()
    template, areas = encoding.phase_encode_gradient(opts, "y", 0.22, 64)
    delta_k = 1.0 / 0.22
    assert len(areas) == 64
    assert areas[0] == pytest.approx(-32 * delta_k)
    assert areas[-1] == pytest.approx(31 * delta_k)
    assert abs(template.area) == pytest.approx(float(np.max(np.abs(areas))))


def test_phase_encode_gradient_partial_fourier_shrinks_template() -> None:
    opts = pp.Opts()
    full, _ = encoding.phase_encode_gradient(opts, "y", 0.22, 64)
    pf, areas = encoding.phase_encode_gradient(opts, "y", 0.22, 64, pf=0.75)
    assert abs(pf.area) < abs(full.area)
    assert len(areas) == 64


def test_phase_encode_gradient_rejects_bad_args() -> None:
    opts = pp.Opts()
    with pytest.raises(ValueError):
        encoding.phase_encode_gradient(opts, "y", 0.22, 64, pf=0.0)
    with pytest.raises(ValueError):
        encoding.phase_encode_gradient(opts, "y", 0.22, 64, accel=0.5)


def test_crusher_by_cycles_matches_explicit_area() -> None:
    opts = pp.Opts()
    by_cycles = encoding.crusher(opts, "z", cycles=4.0, voxel_size_m=5e-3)
    by_area = encoding.crusher(opts, "z", area=4.0 / 5e-3)
    assert by_cycles.area == pytest.approx(by_area.area)


def test_crusher_rejects_ambiguous_spec() -> None:
    opts = pp.Opts()
    with pytest.raises(ValueError):
        encoding.crusher(opts, "z")
    with pytest.raises(ValueError):
        encoding.crusher(opts, "z", cycles=4.0, area=100.0, voxel_size_m=5e-3)
    with pytest.raises(ValueError):
        encoding.crusher(opts, "z", cycles=4.0)


def test_spoiler_3axis_areas() -> None:
    opts = pp.Opts()
    gx, gy, gz = encoding.spoiler_3axis(opts, 3e-3)
    for g in (gx, gy, gz):
        assert g.area == pytest.approx(4.0 / 3e-3)
    assert (gx.channel, gy.channel, gz.channel) == ("x", "y", "z")


def test_partition_geometry() -> None:
    areas, max_area = encoding.partition_geometry(8, 1e-3)
    delta_k = 1.0 / (8 * 1e-3)
    assert areas[0] == pytest.approx(-4 * delta_k)
    assert max_area == pytest.approx(4 * delta_k)
    areas1, max1 = encoding.partition_geometry(1, 1e-3)
    assert max1 == 0.0
    assert np.all(areas1 == 0.0)


def test_combined_z_gradients_area_sums() -> None:
    opts = pp.Opts()
    gz_reph = pp.make_trapezoid(channel="z", area=-200.0, system=opts)
    gz_spoil = pp.make_trapezoid(channel="z", area=400.0, system=opts)
    gz_pe = pp.make_trapezoid(channel="z", area=100.0, system=opts)
    pre, post = encoding.combined_z_gradients(0.5, gz_pe, gz_reph, gz_spoil, opts)
    assert pre.area == pytest.approx(-200.0 + 50.0)
    assert post.area == pytest.approx(-50.0 + 400.0)
