"""The non-Cartesian stack of the recon zoo.

A stack factorises, and that is the whole claim: an inverse FFT along the
Cartesian partition axis turns the volume into independent radial planes, each
of which must come back as the phantom it was sampled from. The stream carries
the counters the sequence labels -- ``LIN`` the spoke, ``PAR`` the partition --
so the plugin sorts nothing for itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

pytest.importorskip("mrinufft")

from pulserver.recon import ReconContext
from pulserver.app import noncartesian_stack_recon

N = 32
N_Z = 4
COILS = 4
NYQUIST = int(np.ceil(np.pi / 2 * N))


@pytest.fixture(scope="module")
def phantom():
    """A slab whose planes differ, so a mis-stacked partition cannot pass."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    disc = 0.3 + np.where((x / 0.9) ** 2 + (y / 0.9) ** 2 < 1.0, 0.7, 0.0)
    volume = np.stack([disc.copy() for _ in range(N_Z)])
    for plane in range(N_Z):
        volume[plane, 8 + 2 * plane : 14 + 2 * plane, 6:20] += 0.5
    return volume


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


def sample(volume, coil_maps, trajectory):
    """Sample each plane on the trajectory, then encode along the partitions.

    The scanner acquires the Fourier transform of the slab along z, so the
    stream is the plane-wise samples transformed back along the partition axis
    -- which is exactly what the reconstruction undoes.
    """
    from mrinufft import get_operator

    operator = get_operator("finufft")(trajectory.reshape(-1, 2), (N, N), n_coils=COILS)
    planes = np.stack(
        [
            operator.op(coil_maps * plane).reshape(
                COILS, trajectory.shape[0], trajectory.shape[1]
            )
            for plane in volume
        ],
        axis=1,
    )
    # (coils, partitions, spokes, samples), Fourier along the partition axis.
    return np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(planes, axes=1), axis=1), axes=1
    ).astype(np.complex64)


def stream(kspace, trajectory):
    """The acquisitions as the server sends them, counters and all."""
    acquisitions = []
    n_spokes, n_samples = trajectory.shape[:2]
    for spoke in range(n_spokes):
        for partition in range(N_Z):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(n_samples, COILS, 2)
            acquisition.data[:] = kspace[:, partition, spoke]
            acquisition.traj[:] = trajectory[spoke]
            acquisition.idx.kspace_encode_step_1 = spoke
            acquisition.idx.kspace_encode_step_2 = partition
            acquisitions.append(acquisition)
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    return acquisitions


def header(n_spokes):
    def limit(extent):
        return SimpleNamespace(minimum=0, maximum=extent - 1, center=extent // 2)

    matrix = SimpleNamespace(x=2 * N, y=N, z=N_Z)
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(matrixSize=matrix),
                reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=N_Z)),
                encodingLimits=SimpleNamespace(
                    kspace_encoding_step_1=limit(n_spokes),
                    kspace_encoding_step_2=limit(N_Z),
                ),
                # Spokes, not a grid: the header has to say so, or the buffer
                # would be sized by the image matrix instead of the view count.
                trajectory="radial",
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(kspace, trajectory, **kwargs):
    from pulserver.recon._server.application import _make_bucket

    acquisitions = stream(kspace, trajectory)
    context = ReconContext.offline(header(trajectory.shape[0]))
    plugin = noncartesian_stack_recon.NonCartesianStackRecon(**kwargs).spawn()
    plugin.startup(context)
    for acquisition in acquisitions:
        plugin.receive(acquisition, context)
    result = plugin.recon(_make_bucket(acquisitions, []), context)
    # The plugin transposes into image order; undo it to compare with the
    # phantom the volume was sampled from.
    return np.asarray(result.data).transpose(0, 2, 1), plugin


def test_a_fully_sampled_stack_returns_the_slab_it_was_sampled_from(phantom, coil_maps):
    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)
    volume, _ = reconstruct(kspace, trajectory)
    assert volume.shape == (N_Z, N, N)
    # A 32x32 direct adjoint bottoms out near 0.2 relative error at toy size;
    # what is under test is the factorisation, not convergence.
    assert relative_error(volume, phantom) < 0.3


def test_each_partition_reconstructs_its_own_plane(phantom, coil_maps):
    """The planes differ, so a stack assembled in the wrong order would put
    one plane's bar on another and every plane would be wrong together."""
    trajectory = spoke_trajectory(NYQUIST)
    volume, _ = reconstruct(sample(phantom, coil_maps, trajectory), trajectory)
    for plane in range(N_Z):
        errors = [relative_error(volume[plane], phantom[other]) for other in range(N_Z)]
        assert int(np.argmin(errors)) == plane


def test_the_spoke_index_comes_from_the_label_not_from_arrival_order(
    phantom, coil_maps
):
    """The sequence labels ``LIN`` with the spoke, so a stream that arrives
    shuffled reconstructs the same volume."""
    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)
    ordered, _ = reconstruct(kspace, trajectory)

    from pulserver.recon._server.application import _make_bucket

    acquisitions = stream(kspace, trajectory)
    last = acquisitions[-1]
    shuffled = [*np.random.default_rng(0).permutation(acquisitions[:-1]), last]
    context = ReconContext.offline(header(trajectory.shape[0]))
    plugin = noncartesian_stack_recon.NonCartesianStackRecon().spawn()
    plugin.startup(context)
    for acquisition in shuffled:
        plugin.receive(acquisition, context)
    out = np.asarray(
        plugin.recon(_make_bucket(acquisitions, []), context).data
    ).transpose(0, 2, 1)
    np.testing.assert_allclose(out, ordered, rtol=1e-5, atol=1e-5)


def test_the_view_count_decides_the_branch_and_comes_from_the_buffer():
    """``auto`` sends an undersampled stack through the solve and a fully
    sampled one through the adjoint, and what it counts is the spoke axis the
    header laid out -- not how many acquisitions turned up."""
    for n_spokes, solves in ((NYQUIST // 3, True), (NYQUIST, False)):
        context = ReconContext.offline(header(n_spokes))
        plugin = noncartesian_stack_recon.NonCartesianStackRecon().spawn()
        plugin.startup(context)
        buffer = plugin.buffers.buffer(0, coils=COILS, readout=2 * N)
        assert buffer.kspace.shape == (COILS, N_Z, n_spokes, 2 * N)
        assert (buffer.kspace.shape[2] < NYQUIST) is solves
