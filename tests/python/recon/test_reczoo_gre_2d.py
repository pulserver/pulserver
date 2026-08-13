"""The 2D Cartesian gradient-echo reconstruction of the recon zoo.

Driven offline, on k-space sampled from an analytic phantom exactly the way
:mod:`pulserver.seqzoo.gre_2d` samples it: the acquired phase encodes, the
calibration block it flags, and a readout truncated before the echo.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import AcquisitionBucket, ReconContext
from pulserver.reczoo import gre_2d

N = 64
N_ACS = 16
ACCELERATION = 2


@pytest.fixture(scope="module")
def phantom():
    """An FOV-filling object, so undersampling it actually aliases."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + np.where((x / 0.95) ** 2 + (y / 0.95) ** 2 < 1.0, 0.7, 0.0)
    image[10:22, 8:30] += 0.5
    return image


@pytest.fixture(scope="module")
def coil_maps():
    """Eight smooth elements on a ring outside the field of view.

    Normalised to unit root-sum-of-squares, so a coil combination of fully
    sampled data returns the phantom itself rather than the phantom under a
    sensitivity shading.
    """
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 3.0)
            * np.exp(1j * 1.5 * (np.cos(a) * x + np.sin(a) * y))
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return maps.astype(np.complex64)


@pytest.fixture(scope="module")
def kspace(phantom, coil_maps):
    """Fully sampled coil k-space, ``(coil, ky, kx)``."""
    return _fft2c(coil_maps * phantom).astype(np.complex64)


@pytest.fixture
def context():
    matrix = SimpleNamespace(x=N, y=N, z=1)
    space = SimpleNamespace(matrixSize=matrix)
    return ReconContext.offline(
        SimpleNamespace(encoding=[SimpleNamespace(encodedSpace=space)])
    )


def bucket(kspace, lines, n_samples=N, calibration=None):
    """One slice's acquisitions, as the runtime would hand them over."""
    center = n_samples - N // 2

    def acquisitions(selected):
        window = kspace[:, selected, N - n_samples :]
        labels = {
            "kspace_encode_step_1": np.asarray(selected),
            "center_sample": np.full(len(selected), center),
        }
        return np.moveaxis(window, 1, 0), labels

    data, labels = acquisitions(lines)
    if calibration is None:
        return AcquisitionBucket.from_arrays(data, labels=labels)
    reference, reference_labels = acquisitions(calibration)
    return AcquisitionBucket.from_arrays(
        data,
        labels=labels,
        reference=reference,
        reference_labels=reference_labels,
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(*args, **kwargs):
    """The reconstructed image, transposed back onto the ``(y, x)`` grid.

    The app transposes its output to the column/row order an image is read in;
    the phantom and zero-filled fixtures here are defined on the ``(y, x)``
    grid, so the test transposes back before comparing against them.
    """
    return gre_2d.PLUGIN(*args, **kwargs).data.T


# ----------------------------------------------------------------------
# Fully sampled
# ----------------------------------------------------------------------


def test_a_fully_sampled_full_echo_slice_is_the_phantom(kspace, phantom, context):
    image = reconstruct(bucket(kspace, list(range(N))), context)
    assert image.shape == (N, N)
    assert relative_error(image, phantom) < 1e-5


def test_a_truncated_echo_is_filled_rather_than_zero_padded(kspace, phantom, context):
    """POCS recovers what zero-filling would blur."""
    partial = reconstruct(bucket(kspace, list(range(N)), n_samples=48), context)
    zero_filled = _zero_filled(kspace, list(range(N)), n_samples=48)
    assert relative_error(partial, phantom) < 0.5 * relative_error(zero_filled, phantom)


# ----------------------------------------------------------------------
# Parallel imaging
# ----------------------------------------------------------------------


def test_the_calibration_block_selects_the_accelerated_branch(kspace, context):
    """A bucket with reference acquisitions is unaliased, not just gridded."""
    lines, calibration = _sampling()
    accelerated = gre_2d.PLUGIN(bucket(kspace, lines, calibration=calibration), context)
    gridded = gre_2d.PLUGIN(bucket(kspace, lines), context)
    assert accelerated.data.shape == gridded.data.shape == (N, N)
    assert not np.allclose(accelerated.data, gridded.data)


def test_the_accelerated_branch_suppresses_aliasing(kspace, phantom, context):
    lines, calibration = _sampling()
    image = reconstruct(bucket(kspace, lines, calibration=calibration), context)
    assert relative_error(image, phantom) < relative_error(
        _zero_filled(kspace, lines), phantom
    )


def test_an_accelerated_partial_echo_slice_reconstructs_on_the_full_grid(
    kspace, phantom, context
):
    lines, calibration = _sampling()
    image = reconstruct(
        bucket(kspace, lines, n_samples=48, calibration=calibration), context
    )
    assert image.shape == (N, N)
    assert relative_error(image, phantom) < relative_error(
        _zero_filled(kspace, lines, n_samples=48), phantom
    )


# ----------------------------------------------------------------------
# What it reads out of the acquisitions
# ----------------------------------------------------------------------


def test_the_matrix_comes_from_the_header_not_from_the_last_line(kspace, context):
    """Under acceleration the highest acquired line is not the matrix size."""
    lines, calibration = _sampling()
    assert max(lines) < N - 1
    image = gre_2d.PLUGIN(bucket(kspace, lines, calibration=calibration), context).data
    assert image.shape == (N, N)


def test_without_a_header_the_matrix_falls_back_to_what_arrived(kspace):
    image = gre_2d.PLUGIN(bucket(kspace, list(range(N))), ReconContext.offline()).data
    assert image.shape == (N, N)


def test_the_sampling_matches_what_the_sequence_would_have_played():
    lines, _ = _sampling()
    assert lines == pp.calc_sampled_lines(N, ACCELERATION, N_ACS)


# ----------------------------------------------------------------------
# Streaming: gridding as data arrives
# ----------------------------------------------------------------------


def test_gridding_on_arrival_matches_gridding_the_bucket(kspace, context):
    """receive() grids each line as it arrives; recon() from that staged grid
    equals recon() from the same bucket gridded in one go at the trigger."""
    from pulserver import ExamCache
    from pulserver.recon._mrd.application import _make_bucket

    lines, calibration = _sampling()
    acquisitions = _native_slice(kspace, lines, calibration)
    bucket = _make_bucket(acquisitions, [])

    # Streaming: startup, a line at a time, then recon from the staged grid.
    streaming = ReconContext(header=context.header, exam=ExamCache("streaming"))
    gre_2d.PLUGIN.startup(streaming)
    for acquisition in acquisitions:
        gre_2d.PLUGIN.receive(acquisition, streaming)
    streamed = gre_2d.PLUGIN.recon(bucket, streaming).data

    # Offline fallback: the same bucket, a fresh exam so nothing was staged.
    offline = ReconContext(header=context.header, exam=ExamCache("offline"))
    reconstructed = gre_2d.PLUGIN.recon(bucket, offline).data

    assert streamed.shape == (N, N)
    np.testing.assert_allclose(streamed, reconstructed, atol=1e-4)


# %% private module subroutines


def _sampling():
    calibration = list(range(N // 2 - N_ACS // 2, N // 2 + N_ACS // 2))
    lines = sorted(set(range(0, N, ACCELERATION)) | set(calibration))
    return lines, calibration


def _native_slice(kspace, lines, calibration):
    """One slice's acquisitions as the native ismrmrd objects the server sees.

    ACS lines carry the parallel-calibration flags the sequence's REF/IMA
    labels become: a line on the acceleration grid is calibration-and-imaging,
    one off it is calibration only.
    """
    import ismrmrd

    coils = kspace.shape[0]
    reference_lines = set(calibration)
    acquisitions = []
    for index, line in enumerate(lines):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(N, coils)
        acquisition.data[:] = kspace[:, line, :].astype(np.complex64)
        acquisition.idx.kspace_encode_step_1 = int(line)
        acquisition.idx.slice = 0
        acquisition.center_sample = N // 2
        if line in reference_lines:
            acquisition.setFlag(
                ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
                if line % ACCELERATION == 0
                else ismrmrd.ACQ_IS_PARALLEL_CALIBRATION
            )
        if index == len(lines) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        acquisitions.append(acquisition)
    return acquisitions


def _fft2c(image):
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _ifft2c(kspace):
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def _zero_filled(kspace, lines, n_samples=N):
    """What gridding alone would give, as the bar every branch has to clear."""
    grid = np.zeros_like(kspace)
    grid[:, lines, N - n_samples :] = kspace[:, lines, N - n_samples :]
    return np.sqrt(np.sum(np.abs(_ifft2c(grid)) ** 2, axis=0))
