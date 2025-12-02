"""Test Cartesian sampling generation."""

import pytest

import numpy as np
import pypulseq as pp

from pulserver.plan import cartesian2D, cartesian3D


@pytest.fixture
def g_slice_select():
    return pp.make_trapezoid(channel="z", duration=1e-3, amplitude=1e-3)


@pytest.mark.parametrize(
    "ny, n_slices, slice_thickness, slice_gap, Ry, Rpf",
    [
        (128, 32, 5.0, 0.0, 1, 1.0),  # Standard case with no acceleration
        (128, 32, 5.0, 1.0, 2, 1.0),  # Parallel Imaging acceleration (Ry = 2)
        (128, 32, 5.0, 1.0, 1, 0.75),  # Partial Fourier acceleration (Rpf = 0.75)
        (128, 32, 5.0, 0.0, 2, 0.75),  # Combined Parallel Imaging and Partial Fourier
    ],
)
def test_cartesian2D_basic(
    ny, n_slices, slice_thickness, slice_gap, Ry, Rpf, g_slice_select
):
    result, sampling_pattern = cartesian2D(
        g_slice_select, slice_thickness, ny, n_slices, slice_gap, Ry=Ry, Rpf=Rpf
    )

    # Check that the sampling pattern is an ndarray of the correct length
    assert isinstance(sampling_pattern, np.ndarray)
    assert len(sampling_pattern) == ny


@pytest.mark.parametrize(
    "view_order, slice_order",
    [
        ("sequential", "sequential"),
        ("center-out", "sequential"),
        ("sequential", "interleaved"),
        ("center-out", "interleaved"),
    ],
)
def test_cartesian2D_view_slice_order(view_order, slice_order, g_slice_select):
    ny = 128
    n_slices = 32
    slice_thickness = 5.0
    slice_gap = 0.0

    result, sampling_pattern = cartesian2D(
        g_slice_select,
        slice_thickness,
        ny,
        n_slices,
        slice_gap,
        view_order=view_order,
        slice_order=slice_order,
    )

    # Check that the sampling pattern is an ndarray of the correct length
    assert isinstance(sampling_pattern, np.ndarray)
    assert len(sampling_pattern) == ny


@pytest.mark.parametrize("dummy_shots", [0, 5, 10])
def test_cartesian2D_dummy_shots(dummy_shots, g_slice_select):
    ny = 128
    n_slices = 32
    slice_thickness = 5.0
    slice_gap = 0.0

    result, sampling_pattern = cartesian2D(
        g_slice_select,
        slice_thickness,
        ny,
        n_slices,
        slice_gap,
        dummy_shots=dummy_shots,
    )

    # Check that the sampling pattern is an ndarray of the correct length
    assert isinstance(sampling_pattern, np.ndarray)
    assert len(sampling_pattern) == ny


def test_cartesian2D_default_params(g_slice_select):
    ny = 128
    n_slices = 32
    slice_thickness = 5.0
    slice_gap = 0.0

    result, sampling_pattern = cartesian2D(
        g_slice_select, slice_thickness, ny, n_slices, slice_gap
    )

    # Check that the sampling pattern is an ndarray of the correct length
    assert isinstance(sampling_pattern, np.ndarray)
    assert len(sampling_pattern) == ny


# Unit tests using pytest for cartesian3D
@pytest.mark.parametrize(
    "ny, nz, Ry, Rz, Rp, Rpf",
    [
        (128, 64, 1, 1, 1.0, 1.0),
        (128, 64, 2, 1, 1.0, 1.0),
        (128, 64, 1, 2, 1.0, 1.0),
        (128, 64, 2, 2, 1.0, 0.75),
        (128, 64, 1, 1, 8.0, 1.0),
        (128, 64, 2, 1, 8.0, 1.0),
        (128, 64, 1, 2, 8.0, 1.0),
        (128, 64, 2, 2, 8.0, 0.75),
    ],
)
def test_cartesian3D_basic(ny, nz, Ry, Rz, Rp, Rpf):
    result, tilt_angles = cartesian3D(ny, nz, Ry=Ry, Rz=Rz, Rp=Rp, Rpf=Rpf)

    # Check that the tilt angles is an ndarray of the correct length
    assert isinstance(tilt_angles, np.ndarray)
    assert len(tilt_angles) == ny


@pytest.mark.parametrize("view_order", ["sequential", "center-out"])
def test_cartesian3D_view_order(view_order):
    ny = 128
    nz = 64

    result, tilt_angles = cartesian3D(ny, nz, view_order=view_order)

    # Check that the tilt angles is an ndarray of the correct length
    assert isinstance(tilt_angles, np.ndarray)
    assert len(tilt_angles) == ny


@pytest.mark.parametrize("dummy_shots", [0, 5, 10])
def test_cartesian3D_dummy_shots(dummy_shots):
    ny = 128
    nz = 64

    result, tilt_angles = cartesian3D(ny, nz, dummy_shots=dummy_shots)

    # Check that the tilt angles is an ndarray of the correct length
    assert isinstance(tilt_angles, np.ndarray)
    assert len(tilt_angles) == ny


def test_cartesian3D_default_params():
    ny = 128
    nz = 64

    result, tilt_angles = cartesian3D(ny, nz)

    # Check that the tilt angles is an ndarray of the correct length
    assert isinstance(tilt_angles, np.ndarray)
    assert len(tilt_angles) == ny
