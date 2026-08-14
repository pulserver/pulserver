"""The non-Cartesian reconstruction of the recon zoo.

A radial stream from an analytic phantom, its samples computed by forward
NUFFT: fully sampled it must return the phantom through the direct branch,
undersampled it must go through the solve and beat the density-compensated
adjoint it would otherwise have been.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mrinufft")

from pulserver import ReconContext
from pulserver.reczoo import gre_radial_2d

N = 32
COILS = 4
NYQUIST = int(np.ceil(np.pi / 2 * N))


@pytest.fixture(scope="module")
def phantom():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + np.where((x / 0.9) ** 2 + (y / 0.9) ** 2 < 1.0, 0.7, 0.0)
    image[8:16, 6:20] += 0.5
    return image


@pytest.fixture(scope="module")
def coil_maps():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 3.0)
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return maps.astype(np.complex64)


def spoke_trajectory(n_spokes):
    """Full spokes over half a turn, in MRI-NUFFT's [-0.5, 0.5) units."""
    radius = np.linspace(-0.5, 0.5, 2 * N, endpoint=False)
    angles = np.arange(n_spokes) * np.pi / n_spokes
    return np.stack(
        [
            np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)
            for angle in angles
        ]
    ).astype(np.float32)


def sample(phantom, coil_maps, trajectory):
    """Forward-NUFFT the phantom onto the trajectory, per coil and view."""
    from mrinufft import get_operator

    operator = get_operator("finufft")(
        trajectory.reshape(-1, 2), (N, N), n_coils=COILS
    )
    data = operator.op(coil_maps * phantom)
    return data.reshape(COILS, trajectory.shape[0], trajectory.shape[1])


def stream(kspace, trajectory):
    """The acquisitions as array-backed objects the plugin accepts."""
    acquisitions = []
    for view in range(trajectory.shape[0]):
        acquisitions.append(
            SimpleNamespace(
                data=np.ascontiguousarray(kspace[:, view]),
                traj=trajectory[view],
                idx=SimpleNamespace(slice=0),
                flags=0,
            )
        )
    return acquisitions


def header():
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(kspace, trajectory, **kwargs):
    plugin = gre_radial_2d.GreRadial2DRecon(**kwargs).spawn()
    plugin.startup(ReconContext.offline(header()))
    for acquisition in stream(kspace, trajectory):
        plugin.receive(acquisition, None)
    nyq = int(np.ceil(np.pi / 2 * max(plugin.image_shape)))
    direct = plugin.mode == "direct" or (
        plugin.mode == "auto" and trajectory.shape[0] >= nyq
    )
    image = gre_radial_2d.reconstruct_plane(
        np.concatenate([kspace[:, view] for view in range(trajectory.shape[0])], -1),
        trajectory.reshape(-1, 2),
        plugin.image_shape,
        direct=direct,
        iterations=kwargs.get("iterations", 20),
    )
    return np.abs(np.asarray(image.cpu() if hasattr(image, "cpu") else image))


def test_a_fully_sampled_scan_is_the_phantom_through_the_direct_branch(
    phantom, coil_maps
):
    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)
    image = reconstruct(kspace, trajectory)
    # A 32x32 direct adjoint bottoms out near 0.2 relative error (DCF and
    # Gibbs at toy size); the claim under test is the branch, not convergence.
    assert relative_error(image, phantom) < 0.3


def test_an_undersampled_scan_goes_through_the_solve_and_beats_the_adjoint(
    phantom, coil_maps
):
    trajectory = spoke_trajectory(NYQUIST // 3)
    kspace = sample(phantom, coil_maps, trajectory)
    flat = np.concatenate(
        [kspace[:, view] for view in range(trajectory.shape[0])], axis=-1
    )
    points = trajectory.reshape(-1, 2)

    solved = gre_radial_2d.reconstruct_plane(
        flat, points, (N, N), direct=False, iterations=25
    )
    adjoint = gre_radial_2d.reconstruct_plane(flat, points, (N, N), direct=True)
    to_mag = lambda x: np.abs(np.asarray(x.cpu() if hasattr(x, "cpu") else x))
    assert relative_error(to_mag(solved), phantom) < relative_error(
        to_mag(adjoint), phantom
    )


def test_the_plugin_reconstructs_a_streamed_measurement(phantom, coil_maps):
    from pulserver.recon._mrd.application import _make_bucket

    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)

    import ismrmrd

    acquisitions = []
    for view in range(trajectory.shape[0]):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(trajectory.shape[1], COILS, trajectory_dimensions=2)
        acquisition.data[:] = kspace[:, view]
        acquisition.traj[:] = trajectory[view]
        acquisition.idx.slice = 0
        if view == trajectory.shape[0] - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        acquisitions.append(acquisition)

    results = gre_radial_2d.PLUGIN(
        _make_bucket(acquisitions, []), ReconContext.offline(header())
    )
    assert len(results) == 1
    assert relative_error(results[0].data.T, phantom) < 0.3
