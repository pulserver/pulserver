"""The 2D Cartesian gradient-echo reconstruction of the recon zoo.

Driven over the acquisitions the runtime would stream, sampled from an analytic
phantom exactly the way :mod:`pulserver.app.sequence.gre2D_sequence` samples it: the
calibration block first, then the remaining phase encodes, with a readout that
may be truncated before the echo.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import ReconContext
from pulserver.app import gre2D_recon

N = 64
N_ACS = 16
ACCELERATION = 2
COILS = 8


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
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
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


def header(n_slices=1):
    """The encoded and reconstructed spaces the plugin sizes its buffer from."""
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                encodingLimits=SimpleNamespace(
                    slice=SimpleNamespace(minimum=0, maximum=n_slices - 1, center=0)
                ),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


@pytest.fixture
def context():
    return ReconContext.offline(header())


def acquisitions(kspace, lines, *, n_samples=N, calibration=(), n_slices=1, slice_=0):
    """One slice's acquisitions as the native objects the server sees.

    The sequence acquires its calibration block first and flags the last line
    of it, the last line of the scan closes both the trailing segment and the
    slice, and a calibration line on the acceleration grid is imaging data too.
    """
    calibration = list(calibration)
    ordered = calibration + [line for line in lines if line not in calibration]
    stream = []
    for index, line in enumerate(ordered):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(n_samples, kspace.shape[0])
        acquisition.data[:] = kspace[:, line, N - n_samples :].astype(np.complex64)
        acquisition.idx.kspace_encode_step_1 = int(line)
        acquisition.idx.slice = int(slice_)
        acquisition.idx.segment = int(index >= len(calibration))
        acquisition.center_sample = n_samples - N // 2
        if line in calibration:
            acquisition.setFlag(
                ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
                if line % ACCELERATION == 0
                else ismrmrd.ACQ_IS_PARALLEL_CALIBRATION
            )
        if calibration and index == len(calibration) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
        if index == len(ordered) - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        stream.append(acquisition)
    del n_slices
    return stream


def bucket(kspace, lines, n_samples=N, calibration=()):
    """One slice's acquisitions, classified as the runtime classifies them."""
    from pulserver.recon._mrd.application import _make_bucket

    return _make_bucket(
        acquisitions(kspace, lines, n_samples=n_samples, calibration=calibration), []
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(*args, **kwargs):
    """The reconstructed image, transposed back onto the ``(y, x)`` grid.

    The plugin transposes its output to the column/row order an image is read
    in; the phantom and zero-filled fixtures here are defined on the ``(y, x)``
    grid, so the test transposes back before comparing against them.
    """
    return gre2D_recon.PLUGIN(*args, **kwargs).data.T


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


def test_the_missing_lines_select_the_accelerated_branch(kspace, context):
    """A slice with phase encodes missing is unaliased, not just gridded."""
    lines, calibration = _sampling()
    accelerated = gre2D_recon.PLUGIN(bucket(kspace, lines, calibration=calibration), context)
    gridded = gre2D_recon.PLUGIN(bucket(kspace, list(range(N))), context)
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
# What it reads out of the header and the acquisitions
# ----------------------------------------------------------------------


def test_the_matrix_comes_from_the_header_not_from_the_last_line(kspace, context):
    """Under acceleration the highest acquired line is not the matrix size."""
    lines, calibration = _sampling()
    assert max(lines) < N - 1
    image = gre2D_recon.PLUGIN(bucket(kspace, lines, calibration=calibration), context).data
    assert image.shape == (N, N)


def test_without_a_header_the_buffer_cannot_be_sized(kspace):
    with pytest.raises(ValueError, match="no encoding 0"):
        gre2D_recon.PLUGIN(bucket(kspace, list(range(N))), ReconContext.offline())


def test_the_image_is_cropped_to_the_reconstructed_matrix(kspace):
    """The encoded grid carries readout oversampling; the image does not."""
    oversampled = header()
    oversampled.encoding[0].reconSpace.matrixSize.x = N // 2
    result = gre2D_recon.PLUGIN(
        bucket(kspace, list(range(N))), ReconContext.offline(oversampled)
    )
    assert result.data.shape == (N // 2, N)


def test_the_sampling_matches_what_the_sequence_would_have_played():
    lines, calibration = _sampling()
    assert lines == pp.calc_sampled_lines(N, ACCELERATION, N_ACS)
    assert (
        calibration + [line for line in lines if line not in calibration]
        == pp.calc_sampled_lines(N, ACCELERATION, N_ACS, order="calibration_first")
    )


# ----------------------------------------------------------------------
# Streaming: the lifecycle the runtime drives
# ----------------------------------------------------------------------


def test_the_calibration_boundary_produces_maps_and_no_image(kspace, context):
    """The ACS block closes a segment without closing the slice."""
    lines, calibration = _sampling()
    stream = acquisitions(kspace, lines, calibration=calibration)
    boundary = len(calibration)

    plugin = gre2D_recon.PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in stream[:boundary]:
        plugin.receive(acquisition, context)
    from pulserver.recon._mrd.application import _make_bucket

    assert plugin.recon(_make_bucket(stream[:boundary], []), context) is None
    assert 0 in plugin.coil_maps
    assert boundary < len(stream)


def test_calibrating_early_and_late_give_the_same_maps(kspace, context):
    """Which boundary ran the calibration cannot change what it produced."""
    lines, calibration = _sampling()
    stream = acquisitions(kspace, lines, calibration=calibration)
    boundary = len(calibration)

    plugin = gre2D_recon.PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in stream[:boundary]:
        plugin.receive(acquisition, context)
    early = gre2D_recon.sensitivities(*plugin.buffer[0])
    for acquisition in stream[boundary:]:
        plugin.receive(acquisition, context)
    late = gre2D_recon.sensitivities(*plugin.buffer[0])

    np.testing.assert_allclose(early.numpy(), late.numpy(), atol=1e-5)


def test_driving_the_lifecycle_by_hand_matches_calling_the_plugin(kspace, context):
    """Calling the plugin replays the bucket through the very same hooks."""
    lines, calibration = _sampling()
    stream = acquisitions(kspace, lines, calibration=calibration)
    from pulserver.recon._mrd.application import _make_bucket

    plugin = gre2D_recon.PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in stream[: len(calibration)]:
        plugin.receive(acquisition, context)
    plugin.recon(_make_bucket(stream[: len(calibration)], []), context)
    for acquisition in stream[len(calibration) :]:
        plugin.receive(acquisition, context)
    streamed = plugin.recon(_make_bucket(stream, []), context).data

    called = gre2D_recon.PLUGIN(bucket(kspace, lines, calibration=calibration), context).data

    assert streamed.shape == (N, N)
    np.testing.assert_allclose(streamed, called, atol=1e-4)


def test_each_call_starts_from_a_clean_buffer(kspace, context):
    """The module-level PLUGIN is a template, so nothing leaks between scans."""
    first = reconstruct(bucket(kspace, list(range(N))), context)
    second = reconstruct(bucket(kspace, list(range(N))), context)
    np.testing.assert_allclose(first, second)
    assert not hasattr(gre2D_recon.PLUGIN, "buffer")


def test_a_second_slice_reconstructs_its_own_data(kspace, phantom):
    """The buffer spans the volume, and recon picks the slice that just closed."""
    lines = list(range(N))
    stream = acquisitions(kspace, lines, slice_=1)
    plugin = gre2D_recon.PLUGIN.spawn()
    plugin.startup(ReconContext.offline(header(n_slices=2)))
    for acquisition in stream:
        plugin.receive(acquisition, ReconContext.offline(header(n_slices=2)))
    from pulserver.recon._mrd.application import _make_bucket

    image = plugin.recon(
        _make_bucket(stream, []), ReconContext.offline(header(n_slices=2))
    ).data.T
    assert relative_error(image, phantom) < 1e-5
    assert not plugin.buffer.mask[0].any()


# %% private module subroutines


def _sampling():
    calibration = list(range(N // 2 - N_ACS // 2, N // 2 + N_ACS // 2))
    lines = sorted(set(range(0, N, ACCELERATION)) | set(calibration))
    return lines, calibration


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
    """What placing alone would give, as the bar every branch has to clear."""
    grid = np.zeros_like(kspace)
    grid[:, lines, N - n_samples :] = kspace[:, lines, N - n_samples :]
    return np.sqrt(np.sum(np.abs(_ifft2c(grid)) ** 2, axis=0))
