"""A map one series measures, read by a later series on its own grid."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import calibration_header, coil_phantom, play, readouts, stream_plugin

from pulserver import recon

COILS = 2
LAST = ("ACQ_LAST_IN_SLICE",)
FOV = 220.0
SLOPE = np.array([0.01, -0.02, 0.0])


@pytest.fixture(autouse=True)
def reslice():
    tools = pytest.importorskip("bartorch.tools")
    pytest.importorskip("SimpleITK")
    if not hasattr(tools, "reslice"):
        pytest.skip("bartorch has no reslice")


def ramp(matrix, centre):
    """``SLOPE . r + 1`` at the voxel centres of an axial plane of ``matrix`` voxels centred on ``centre`` (mm)."""
    spacing = FOV / matrix
    offsets = (np.arange(matrix) - matrix // 2) * spacing
    x = centre[0] + offsets[None, :]
    y = centre[1] + offsets[:, None]
    return (SLOPE[0] * x + SLOPE[1] * y + 1.0)[None]


def unit(matrix, position):
    _, _, kspace = coil_phantom(COILS, matrix)
    plugin = stream_plugin(lambda context, data: (context, data))
    play(
        plugin,
        calibration_header(COILS, matrix),
        readouts(
            kspace,
            last=LAST,
            geometry={
                "position": position,
                "read_dir": (1.0, 0.0, 0.0),
                "phase_dir": (0.0, 1.0, 0.0),
                "slice_dir": (0.0, 0.0, 1.0),
            },
        ),
    )
    ((context, data),) = plugin.results
    return context, data


def test_a_map_is_read_on_the_grid_and_centre_of_the_reading_series():
    measured_at, read_at = (0.0, 0.0, 0.0), (12.0, -6.0, 0.0)
    stored = recon.ExamImage.measured(ramp(16, measured_at)[0], *unit(16, measured_at))

    resampled = stored.on(*unit(32, read_at))

    expected = ramp(32, read_at)
    assert resampled.shape == (1, 32, 32)
    inside = (slice(None), slice(4, -4), slice(4, -4))
    np.testing.assert_allclose(resampled[inside], expected[inside], rtol=1e-4)


def test_a_map_stores_the_geometry_of_the_image_its_unit_reconstructs():
    stored = recon.ExamImage.measured(np.ones((16, 16)), *unit(16, (1.0, 2.0, 3.0)))

    assert stored.data.shape == (1, 16, 16)
    assert stored.geometry["position"] == (1.0, 2.0, 3.0)
    assert stored.geometry["field_of_view"] == (FOV, FOV, 5.0)
    assert stored.geometry["matrix_size"] == (16, 16, 1)


def test_a_series_reads_the_receive_field_of_its_own_plane_with_mean_one():
    from pulserver.recon._correct import ReceiveFields

    fields = ReceiveFields()
    for z in (0.0, 5.0):
        position = (0.0, 0.0, z)
        fields.add(
            recon.ExamImage.measured(
                (1.0 + z) * (3.0 + ramp(16, position)[0]), *unit(16, position)
            )
        )

    field = fields.on(*unit(32, (12.0, -6.0, 5.0)))

    expected = 3.0 + ramp(32, (12.0, -6.0, 0.0))[0]
    inside = (slice(4, -4), slice(4, -4))
    assert field.shape == (32, 32)
    np.testing.assert_allclose(field.mean(), 1.0, rtol=1e-3)
    np.testing.assert_allclose(
        field[inside] / field[inside].mean(),
        expected[inside] / expected[inside].mean(),
        rtol=1e-3,
    )


def test_a_series_whose_slice_no_stored_plane_covers_reads_no_receive_field():
    from pulserver.recon._correct import ReceiveFields

    fields = ReceiveFields()
    fields.add(
        recon.ExamImage.measured(
            ramp(16, (0.0, 0.0, 0.0))[0], *unit(16, (0.0, 0.0, 0.0))
        )
    )

    assert fields.on(*unit(16, (0.0, 0.0, 10.0))) is None
