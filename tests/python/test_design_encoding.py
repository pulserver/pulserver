"""Unit tests for pulserver gradient helpers."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design import _gradients as encoding  # noqa: E402
from pulserver.pypulseq import scale_grad  # noqa: E402
from pulserver.design import calc_encoding_scales


def test_phase_encode_gradient_is_fixed_by_resolution_alone() -> None:
    opts = pp.Opts()
    fov, ny = 0.22, 64
    template = encoding.make_phase_encoding(opts, "y", fov / ny)

    # The extreme step of an ny-view encode over `fov`, reached without ever
    # naming fov or ny: (ny / 2) * (1 / fov) == 1 / (2 * resolution).
    assert template.channel == "y"
    assert template.area == pytest.approx(0.5 * ny / fov)

    # Any (fov, matrix) pair with the same resolution gives the same gradient.
    same = encoding.make_phase_encoding(opts, "y", (2 * fov) / (2 * ny))
    assert same.area == pytest.approx(template.area)


def test_phase_encode_gradient_scales_to_every_view() -> None:
    opts = pp.Opts()
    fov, ny = 0.22, 64
    template = encoding.make_phase_encoding(opts, "y", fov / ny)

    areas = [scale_grad(template, s).area for s in calc_encoding_scales(np.arange(ny), ny)]
    assert areas[0] == pytest.approx(-32 / fov)
    assert areas[-1] == pytest.approx(31 / fov)


def test_phase_encode_gradient_rejects_bad_resolution() -> None:
    opts = pp.Opts()
    with pytest.raises(ValueError, match="resolution_m"):
        encoding.make_phase_encoding(opts, "y", 0.0)
    with pytest.raises(ValueError, match="resolution_m"):
        encoding.make_phase_encoding(opts, "y", -1e-3)


def test_crusher_by_cycles_matches_explicit_area() -> None:
    opts = pp.Opts()
    by_cycles = encoding.make_crusher(opts, "z", dephasing_cycles=4.0, voxel_size=5e-3)
    by_area = encoding.make_crusher(opts, "z", area=4.0 / 5e-3)
    assert by_cycles.area == pytest.approx(by_area.area)


def test_crusher_rejects_ambiguous_spec() -> None:
    opts = pp.Opts()
    with pytest.raises(ValueError):
        encoding.make_crusher(opts, "z")
    with pytest.raises(ValueError):
        encoding.make_crusher(opts, "z", dephasing_cycles=4.0, area=100.0, voxel_size=5e-3)
    with pytest.raises(ValueError):
        encoding.make_crusher(opts, "z", dephasing_cycles=4.0)


def test_spoiler_3axis_areas() -> None:
    opts = pp.Opts()
    gx, gy, gz = encoding.make_spoiler(opts, 3e-3)
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
