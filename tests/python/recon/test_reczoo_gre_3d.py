"""The 3D Cartesian gradient-echo reconstruction of the recon zoo.

Driven over the acquisitions the runtime would stream, sampled from an
analytic phantom exactly the way :mod:`pulserver.seqzoo.gre_3d` samples it:
the calibration rectangle first, then the remaining ``(line, partition)``
pairs, with a readout that may be truncated before the echo.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import ReconContext
from pulserver.reczoo import gre_3d

N = 32
N_Z = 8
N_ACS = 8
N_ACS_Z = 4
ACCELERATION = 2
COILS = 4

#: Fewer CG iterations than the deployed default, because a CPU test suite
#: pays for every one of them and the assertions compare against zero-filling,
#: not against convergence.
PLUGIN = gre_3d.Gre3DRecon(iterations=20)


@pytest.fixture(scope="module")
def phantom():
    """An FOV-filling object whose shape changes along z."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    slabs = []
    for k in range(N_Z):
        radius = 0.95 - 0.06 * k
        plane = 0.3 + np.where((x / radius) ** 2 + (y / radius) ** 2 < 1.0, 0.7, 0.0)
        plane[6 + k : 14 + k, 4:18] += 0.5
        slabs.append(plane)
    return np.stack(slabs)


@pytest.fixture(scope="module")
def coil_maps():
    """Four smooth elements on a ring outside the field of view.

    Constant along z and normalised to unit root-sum-of-squares, so a coil
    combination of fully sampled data returns the phantom itself.
    """
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 3.0)
            * np.exp(1j * 1.5 * (np.cos(a) * x + np.sin(a) * y))
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return np.broadcast_to(maps[:, None], (COILS, N_Z, N, N)).astype(np.complex64)


@pytest.fixture(scope="module")
def kspace(phantom, coil_maps):
    """Fully sampled coil k-space, ``(coil, kz, ky, kx)``."""
    return _fft3c(coil_maps * phantom).astype(np.complex64)


def header():
    """The encoded and reconstructed spaces the plugin sizes its buffer from."""
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(
                    matrixSize=SimpleNamespace(x=N, y=N, z=N_Z)
                ),
                reconSpace=SimpleNamespace(
                    matrixSize=SimpleNamespace(x=N, y=N, z=N_Z)
                ),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


@pytest.fixture
def context():
    return ReconContext.offline(header())


def _all_pairs():
    return [(line, partition) for partition in range(N_Z) for line in range(N)]


def _sampling():
    """The pairs an accelerated scan plays, calibration rectangle first.

    Undersampled along y only: the ring of coils encodes in-plane, so that is
    the axis parallel imaging can actually resolve -- test coils constant
    along z would make a z-aliased volume unresolvable by construction.
    """
    lines = pp.calc_sampled_lines(N, ACCELERATION, N_ACS)
    partitions = list(range(N_Z))
    acs_y = set(range(N // 2 - N_ACS // 2, N // 2 + N_ACS // 2))
    acs_z = set(range(N_Z // 2 - N_ACS_Z // 2, N_Z // 2 + N_ACS_Z // 2))
    rectangle = [
        (line, partition)
        for partition in partitions
        for line in lines
        if line in acs_y and partition in acs_z
    ]
    body = [
        (line, partition)
        for partition in partitions
        for line in lines
        if not (line in acs_y and partition in acs_z)
    ]
    return rectangle + body, len(rectangle)


def acquisitions(
    kspace, pairs, *, n_samples=N, n_calibration=0, measurement_ends=True
):
    """The stream as the native objects the server sees.

    ``measurement_ends=False`` leaves the final flag off, for driving the
    calibration boundary alone -- a rectangle that closes its segment while
    the scan is still running.
    """
    stream = []
    for index, (line, partition) in enumerate(pairs):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(n_samples, kspace.shape[0])
        acquisition.data[:] = kspace[:, partition, line, N - n_samples :].astype(
            np.complex64
        )
        acquisition.idx.kspace_encode_step_1 = int(line)
        acquisition.idx.kspace_encode_step_2 = int(partition)
        acquisition.idx.segment = int(index >= n_calibration)
        acquisition.center_sample = n_samples - N // 2
        if n_calibration and index == n_calibration - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
        if measurement_ends and index == len(pairs) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        stream.append(acquisition)
    return stream


def bucket(kspace, pairs, n_samples=N, n_calibration=0):
    from pulserver.recon._mrd.application import _make_bucket

    return _make_bucket(
        acquisitions(kspace, pairs, n_samples=n_samples, n_calibration=n_calibration),
        [],
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(*args, **kwargs):
    """The reconstructed volume, transposed back onto the ``(z, y, x)`` grid."""
    return PLUGIN(*args, **kwargs).data.transpose(0, 2, 1)


# ----------------------------------------------------------------------
# Fully sampled
# ----------------------------------------------------------------------


def test_a_fully_sampled_full_echo_volume_is_the_phantom(kspace, phantom, context):
    image = reconstruct(bucket(kspace, _all_pairs()), context)
    assert image.shape == (N_Z, N, N)
    assert relative_error(image, phantom) < 1e-5


def test_a_truncated_echo_is_filled_rather_than_zero_padded(kspace, phantom, context):
    """POCS recovers what zero-filling would blur."""
    partial = reconstruct(bucket(kspace, _all_pairs(), n_samples=24), context)
    zero_filled = _zero_filled(kspace, _all_pairs(), n_samples=24)
    assert relative_error(partial, phantom) < 0.75 * relative_error(
        zero_filled, phantom
    )


# ----------------------------------------------------------------------
# Parallel imaging
# ----------------------------------------------------------------------


def test_the_accelerated_branch_suppresses_aliasing(kspace, phantom, context):
    pairs, n_calibration = _sampling()
    image = reconstruct(bucket(kspace, pairs, n_calibration=n_calibration), context)
    assert relative_error(image, phantom) < relative_error(
        _zero_filled(kspace, pairs), phantom
    )


def test_the_matrix_comes_from_the_header_not_from_the_last_pair(kspace, context):
    """Under acceleration the highest acquired line is not the matrix size."""
    pairs, n_calibration = _sampling()
    assert max(line for line, _ in pairs) < N - 1
    result = PLUGIN(bucket(kspace, pairs, n_calibration=n_calibration), context)
    assert result.data.shape == (N_Z, N, N)


# ----------------------------------------------------------------------
# Streaming: the lifecycle the runtime drives
# ----------------------------------------------------------------------


def test_the_calibration_boundary_produces_maps_and_no_image(kspace, context):
    """The rectangle's segment closing is a calibration, not a reconstruction."""
    pairs, n_calibration = _sampling()
    rectangle = acquisitions(
        kspace,
        pairs[:n_calibration],
        n_calibration=n_calibration,
        measurement_ends=False,
    )
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in rectangle:
        plugin.receive(acquisition, context)

    from pulserver.recon._mrd.application import _make_bucket

    assert plugin.recon(_make_bucket(rectangle, []), context) is None
    assert plugin.coil_maps is not None


# ----------------------------------------------------------------------
# Local subroutines
# ----------------------------------------------------------------------


def _fft3c(volume):
    axes = (-3, -2, -1)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(volume, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _ifft3c(kspace):
    axes = (-3, -2, -1)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _zero_filled(kspace, pairs, n_samples=N):
    grid = np.zeros_like(kspace)
    for line, partition in pairs:
        grid[:, partition, line, N - n_samples :] = kspace[
            :, partition, line, N - n_samples :
        ]
    return np.sqrt(np.sum(np.abs(_ifft3c(grid)) ** 2, axis=0))
