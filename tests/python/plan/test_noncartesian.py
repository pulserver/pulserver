"""Test Non Cartesian sampling generation."""

import pytest

import numpy as np
import pypulseq as pp

from pulserver.plan import noncartesian2D, noncartesian3D, stack3D


@pytest.fixture
def g_slice_select():
    return pp.make_trapezoid(channel="z", duration=1e-3, amplitude=1e-3)


@pytest.mark.parametrize(
    "n_views, n_slices, Rtheta, view_order",
    [
        (128, 32, 1.0, "mri-golden"),  # Standard case, no acceleration
        (128, 32, 2.0, "golden"),  # Angular undersampling with golden angle
        (128, 32, 1.0, "uniform"),  # Uniform tilt
    ],
)
def test_noncartesian2D_basic(n_views, n_slices, Rtheta, view_order, g_slice_select):
    slice_thickness = 5e-3
    slice_gap = 1e-3

    result, rotmat = noncartesian2D(
        g_slice_select,
        slice_thickness,
        n_views,
        n_slices,
        slice_gap,
        Rtheta=Rtheta,
        view_order=view_order,
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (n_views // int(Rtheta), 3, 3)


@pytest.mark.parametrize("slice_order", ["sequential", "interleaved"])
def test_noncartesian2D_slice_order(slice_order, g_slice_select):
    slice_thickness = 5e-3
    slice_gap = 1e-3
    n_views = 128
    n_slices = 32

    result, rotmat = noncartesian2D(
        g_slice_select,
        slice_thickness,
        n_views,
        n_slices,
        slice_gap,
        slice_order=slice_order,
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (n_views // 1, 3, 3)  # Rtheta defaults to 1.0


@pytest.mark.parametrize("view_loop_position", ["inner", "outer"])
def test_noncartesian2D_view_loop_position(view_loop_position, g_slice_select):
    slice_thickness = 5e-3
    slice_gap = 1e-3
    n_views = 128
    n_slices = 32

    result, rotmat = noncartesian2D(
        g_slice_select,
        slice_thickness,
        n_views,
        n_slices,
        slice_gap,
        view_loop_position=view_loop_position,
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (n_views // 1, 3, 3)


@pytest.mark.parametrize("dummy_shots", [0, 5, 10])
def test_noncartesian2D_dummy_shots(dummy_shots, g_slice_select):
    slice_thickness = 5e-3
    slice_gap = 1e-3
    n_views = 128
    n_slices = 32

    result, rotmat = noncartesian2D(
        g_slice_select,
        slice_thickness,
        n_views,
        n_slices,
        slice_gap,
        dummy_shots=dummy_shots,
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (n_views // 1, 3, 3)


def test_noncartesian2D_default_params(g_slice_select):
    slice_thickness = 5e-3
    slice_gap = 1e-3
    n_views = 128
    n_slices = 32

    result, rotmat = noncartesian2D(
        g_slice_select, slice_thickness, n_views, n_slices, slice_gap
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (n_views // 1, 3, 3)


@pytest.mark.parametrize(
    "n_views_plane, n_views_angular, Rplane, Rangular, angular_order",
    [
        (128, 32, 1.0, 1.0, "mri-golden"),  # Standard case, no acceleration
        (
            128,
            32,
            2.0,
            2.0,
            "golden",
        ),  # Acceleration along both planes with golden angle
        (128, 32, 1.0, 1.0, "uniform"),  # Uniform tilt
    ],
)
def test_noncartesian3D_basic(
    n_views_plane, n_views_angular, Rplane, Rangular, angular_order
):
    dummy_shots = 0

    result, rotmat = noncartesian3D(
        n_views_plane,
        n_views_angular,
        Rplane=Rplane,
        Rangular=Rangular,
        angular_order=angular_order,
        dummy_shots=dummy_shots,
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = n_views_plane * n_views_angular
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (int(nviews // (Rplane * Rangular)), 3, 3)


@pytest.mark.parametrize("dummy_shots", [0, 5, 10])
def test_noncartesian3D_dummy_shots(dummy_shots):
    n_views_plane = 128
    n_views_angular = 32
    Rplane = 1.0
    Rangular = 1.0

    result, rotmat = noncartesian3D(
        n_views_plane, n_views_angular, dummy_shots=dummy_shots
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = n_views_plane * n_views_angular
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (int(nviews // (Rplane * Rangular)), 3, 3)


@pytest.mark.parametrize(
    "angular_order",
    ["mri-golden", "golden", "none", "uniform", "intergaps", "inverted"],
)
def test_noncartesian3D_angular_order(angular_order):
    n_views_plane = 128
    n_views_angular = 32
    Rplane = 1.0
    Rangular = 1.0

    result, rotmat = noncartesian3D(
        n_views_plane, n_views_angular, angular_order=angular_order
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = n_views_plane * n_views_angular
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (int(nviews // (Rplane * Rangular)), 3, 3)


def test_noncartesian3D_default_params():
    n_views_plane = 128
    n_views_angular = 32

    result, rotmat = noncartesian3D(n_views_plane, n_views_angular)

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = n_views_plane * n_views_angular
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (
        int(nviews // 1),
        3,
        3,
    )  # Rplane and Rangular both default to 1.0


@pytest.mark.parametrize(
    "n_views, nz, Rtheta, Rz, Rpf, calib",
    [
        (128, 32, 1.0, 1, 1.0, None),  # Standard case, no acceleration or calibration
        (128, 32, 2.0, 2, 0.8, 16),  # Accelerated with calibration, partial Fourier
        (128, 32, 1.0, 1, 0.9, None),  # Partial Fourier, no calibration
        (128, 32, 1.0, 1, 1.0, 8),  # Calibration with no acceleration
    ],
)
def test_stack3D_basic(n_views, nz, Rtheta, Rz, Rpf, calib):
    dummy_shots = 0

    result, (rotmat, sampling_pattern) = stack3D(
        n_views, nz, Rtheta=Rtheta, Rz=Rz, Rpf=Rpf, calib=calib, dummy_shots=dummy_shots
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = int(n_views // Rtheta)
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (nviews, 3, 3)

    # Check that the sampling pattern is an ndarray of correct shape
    assert isinstance(sampling_pattern, np.ndarray)
    assert sampling_pattern.shape == (nz,)


@pytest.mark.parametrize("dummy_shots", [0, 5, 10])
def test_stack3D_dummy_shots(dummy_shots):
    n_views = 128
    nz = 32
    Rtheta = 1.0
    Rz = 1
    Rpf = 1.0
    calib = None

    result, (rotmat, sampling_pattern) = stack3D(
        n_views, nz, dummy_shots=dummy_shots, Rtheta=Rtheta, Rz=Rz, Rpf=Rpf, calib=calib
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = int(n_views // Rtheta)
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (nviews, 3, 3)

    # Check that the sampling pattern is an ndarray of correct shape
    assert isinstance(sampling_pattern, np.ndarray)
    assert sampling_pattern.shape == (nz,)


@pytest.mark.parametrize(
    "view_order", ["mri-golden", "golden", "none", "uniform", "intergaps", "inverted"]
)
def test_stack3D_view_order(view_order):
    n_views = 128
    nz = 32
    Rtheta = 1.0
    Rz = 1
    Rpf = 1.0
    calib = None

    result, (rotmat, sampling_pattern) = stack3D(
        n_views, nz, view_order=view_order, Rtheta=Rtheta, Rz=Rz, Rpf=Rpf, calib=calib
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = int(n_views // Rtheta)
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (nviews, 3, 3)


@pytest.mark.parametrize("slice_order", ["sequential", "interleaved", "center-out"])
def test_stack3D_slice_order(slice_order):
    n_views = 128
    nz = 32
    Rtheta = 1.0
    Rz = 1
    Rpf = 1.0
    calib = None

    result, (rotmat, sampling_pattern) = stack3D(
        n_views, nz, slice_order=slice_order, Rtheta=Rtheta, Rz=Rz, Rpf=Rpf, calib=calib
    )

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = int(n_views // Rtheta)
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (nviews, 3, 3)

    # Check that the sampling pattern is an ndarray of correct shape
    assert isinstance(sampling_pattern, np.ndarray)
    assert sampling_pattern.shape == (nz,)


def test_stack3D_default_params():
    n_views = 128
    nz = 32

    result, (rotmat, sampling_pattern) = stack3D(n_views, nz)

    # Check that the rotation matrix is an ndarray of the correct shape
    nviews = int(n_views // 1)  # Rtheta defaults to 1.0
    assert isinstance(rotmat, np.ndarray)
    assert rotmat.shape == (nviews, 3, 3)

    # Check that the sampling pattern is an ndarray of correct shape
    assert isinstance(sampling_pattern, np.ndarray)
    assert sampling_pattern.shape == (nz,)
