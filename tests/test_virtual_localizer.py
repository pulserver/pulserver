"""The three-plane localizer drawn from a phantom's ground truth: where each plane lies and what it shows."""

import math
import sys
import types

import numpy as np
import pytest

from pulserver import virtual
from pulserver.virtual import _brainweb
from pulserver.virtual._localizer import PLANES

FOV, MATRIX, THICKNESS = 0.128, 64, 5e-3
PIXEL = FOV / MATRIX


def _pixels(dataset):
    return dataset.pixel_array * float(dataset.RescaleSlope) + float(
        dataset.RescaleIntercept
    )


def _points(centre, read, phase):
    """The physical point at the centre of each pixel of an image, ``(rows, columns, 3)``."""
    offsets = (np.arange(MATRIX) - 0.5 * (MATRIX - 1)) * PIXEL
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    return (
        np.asarray(centre)
        + columns[..., None] * np.asarray(read)
        + rows[..., None] * np.asarray(phase)
    )


def _disc(centre=(0.03, -0.02, 0.0), radius=0.01, **placement):
    return virtual.Phantom(
        [virtual.Ellipse(centre, (radius, radius), intensity=2.0)], **placement
    )


def _scan(phantom, centre):
    return virtual.localizer(
        phantom,
        field_t=3.0,
        subject="disc",
        fov=FOV,
        matrix=MATRIX,
        thickness=THICKNESS,
        centre=centre,
    )


@pytest.mark.parametrize("plane", list(PLANES))
def test_each_plane_carries_its_orientation_and_the_centre_of_its_first_pixel(plane):
    centre = (0.01, -0.02, 0.03)
    read, phase = map(np.asarray, PLANES[plane])

    dataset = _scan(_disc(), centre)[list(PLANES).index(plane)]

    np.testing.assert_allclose(
        [float(v) for v in dataset.ImageOrientationPatient], [*read, *phase]
    )
    first = 1e3 * (np.asarray(centre) - 0.5 * (FOV - PIXEL) * (read + phase))
    np.testing.assert_allclose(
        [float(v) for v in dataset.ImagePositionPatient], first, atol=1e-6
    )
    np.testing.assert_allclose(
        [float(v) for v in dataset.PixelSpacing], [1e3 * PIXEL] * 2
    )
    assert (dataset.Rows, dataset.Columns) == (MATRIX, MATRIX)
    assert str(dataset.PatientName) == "disc"
    assert dataset.SeriesDescription == "Localizer"


def test_the_planes_are_the_radiological_views_of_a_head_first_supine_subject():
    normals = {
        name: np.cross(*map(np.asarray, directions))
        for name, directions in PLANES.items()
    }

    np.testing.assert_allclose(normals["axial"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(np.abs(normals["coronal"]), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(np.abs(normals["sagittal"]), [1.0, 0.0, 0.0])


def test_a_disc_shows_its_face_in_its_own_plane_and_its_section_across_it():
    centre = (0.03, -0.02, 0.0)

    images = dict(zip(PLANES, _scan(_disc(centre), centre), strict=True))

    for plane, dataset in images.items():
        points = _points(centre, *PLANES[plane])
        radial = np.hypot(points[..., 0] - centre[0], points[..., 1] - centre[1])
        expected = np.where(
            (radial <= 0.01) & (np.abs(points[..., 2]) <= 0.5 * THICKNESS), 2.0, 0.0
        )
        np.testing.assert_allclose(_pixels(dataset), expected, atol=2.0 / 1000)
    assert np.count_nonzero(_pixels(images["axial"]) > 1.0) > np.count_nonzero(
        _pixels(images["coronal"]) > 1.0
    )


def test_a_turned_and_moved_phantom_is_drawn_where_it_lies():
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    phantom = _disc((0.03, 0.0, 0.0), rotation=quarter, position=(0.0, 0.05, 0.0))
    lies_at = quarter @ np.array([0.03, 0.0, 0.0]) + np.array([0.0, 0.05, 0.0])

    axial = _pixels(_scan(phantom, lies_at)[0])

    rows, columns = np.nonzero(axial > 1.0)
    assert math.isclose(rows.mean(), 0.5 * (MATRIX - 1), abs_tol=0.5)
    assert math.isclose(columns.mean(), 0.5 * (MATRIX - 1), abs_tol=0.5)


@pytest.fixture
def brain(monkeypatch):
    """A small fuzzy model standing in for brainweb-dl's download."""
    fractions = np.zeros((6, 6, 8, len(_brainweb.TISSUES)), dtype=np.float32)
    monkeypatch.setitem(
        sys.modules,
        "brainweb_dl",
        types.SimpleNamespace(get_mri=lambda *args, **kwargs: fractions),
    )
    return types.SimpleNamespace(model=virtual.BrainWeb(), fractions=fractions)


def test_brainweb_density_is_each_tissues_density_times_its_fraction_in_the_voxel(
    brain,
):
    brain.fractions[2, 3, 1, 2] = 0.25  # grey matter
    brain.fractions[2, 3, 1, 3] = 0.75  # white matter
    voxel = 1e-3 * np.array([[-(1 - 90.0), -(3 - 126.0), 2 - 72.0]])
    along_z = np.array([0.0, 0.0, 1.0])

    thin = brain.model.proton_density(voxel, normal=along_z, thickness=1e-3)
    slab = brain.model.proton_density(voxel, normal=along_z, thickness=3e-3)
    outside = brain.model.proton_density(voxel + 1.0, normal=along_z, thickness=1e-3)

    np.testing.assert_allclose(thin, [0.86 * 0.25 + 0.77 * 0.75], rtol=1e-6)
    np.testing.assert_allclose(slab, thin / 3, rtol=1e-6)
    np.testing.assert_array_equal(outside, [0.0])
