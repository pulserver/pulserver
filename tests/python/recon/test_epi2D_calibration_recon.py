"""The accelerated non-SMS branch of the 2D EPI reconstruction.

Driven over the acquisitions the runtime would stream from
:mod:`pulserver.app.sequence.epi2D_sequence` for a plain (non-multiband) accelerated scan: a
separate low-resolution gradient-echo calibration for each slice, then the
undersampled imaging lines. The reconstruction has to estimate each slice's coil
sensitivities from the calibration once and unalias the imaging against them --
the imaging file carries no autocalibration lines of its own.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

ismrmrd = pytest.importorskip("ismrmrd")
pytest.importorskip("torch")

from pulserver import ReconContext
from pulserver.app import epi2D_recon

N = 40
N_SLICES = 3
COILS = 8
N_ACS = 24
ACCELERATION = 2


@pytest.fixture(scope="module")
def phantom():
    """One distinct disk per slice, so a mix-up between slices is visible."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    slices = []
    for s in range(N_SLICES):
        radius = 0.85 - 0.05 * s
        plane = 0.2 + np.where(
            (x - 0.12 * s) ** 2 + (y + 0.1 * s) ** 2 < radius**2, 0.8, 0.0
        )
        plane[8 + 3 * s : 16 + 3 * s, 6:18] += 0.4
        slices.append(plane)
    return np.stack(slices).astype(np.complex64)


@pytest.fixture(scope="module")
def coil_maps():
    """Per-slice coil maps, the ring rotated per slice so slices differ."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = []
    for s in range(N_SLICES):
        offset = s * 2 * np.pi / (COILS * N_SLICES)
        element = np.stack(
            [
                np.exp(
                    -(
                        (x - 1.6 * np.cos(a + offset)) ** 2
                        + (y - 1.6 * np.sin(a + offset)) ** 2
                    )
                    / 1.6
                )
                * np.exp(1j * (1.3 * (np.cos(a) * x + np.sin(a) * y) + 0.6 * s))
                for a in angles
            ]
        )
        maps.append(
            element / np.sqrt(np.sum(np.abs(element) ** 2, axis=0, keepdims=True))
        )
    return np.stack(maps).astype(np.complex64)


def _fft2c(image):
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


@pytest.fixture(scope="module")
def kspace(phantom, coil_maps):
    """Per-slice fully sampled coil k-space, ``(slice, coil, ky, kx)``."""
    return _fft2c(coil_maps * phantom[:, None]).astype(np.complex64)


def _encoding():
    """One encoding space over the scan's slices."""
    matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1))
    return SimpleNamespace(
        encodedSpace=matrix,
        reconSpace=matrix,
        encodingLimits=SimpleNamespace(slice=SimpleNamespace(maximum=N_SLICES - 1)),
    )


def header():
    """Two encoding spaces: the EPI imaging, then the prescan.

    The calibration is its own subsequence, so it is its own encoding space --
    and it visits exactly the slices the imaging visits, which is what says the
    scan was not multiband.
    """
    return SimpleNamespace(
        encoding=[_encoding(), _encoding()],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


@pytest.fixture
def context():
    return ReconContext.offline(header())


def _line(data, *, slice_index, line, encoding=0, flags=(), last=False):
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(N, COILS)
    acquisition.data[:] = data.astype(np.complex64)
    acquisition.idx.kspace_encode_step_1 = int(line)
    acquisition.idx.slice = int(slice_index)
    acquisition.idx.repetition = 0
    acquisition.encoding_space_ref = int(encoding)
    for flag in flags:
        acquisition.setFlag(getattr(ismrmrd, flag))
    if last:
        acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    return acquisition


def _calibration_lines():
    """The fully sampled central ``N_ACS`` window, centre-anchored."""
    start = (N - N_ACS) // 2
    return list(range(start, start + N_ACS))


def stream(kspace):
    """Low-resolution GRE calibration (every slice) then the undersampled shots."""
    acquisitions = []
    # Calibration: a central block per slice, marked parallel-imaging calibration.
    for slice_index in range(N_SLICES):
        lines = _calibration_lines()
        for line in lines:
            flags = ["ACQ_IS_PARALLEL_CALIBRATION"]
            if line == lines[-1]:
                flags.append("ACQ_LAST_IN_SLICE")
            acquisitions.append(
                _line(
                    kspace[slice_index, :, line, :],
                    slice_index=slice_index,
                    line=line,
                    encoding=1,
                    flags=tuple(flags),
                )
            )
    # Imaging: every ``ACCELERATION``-th line per slice, no calibration flag.
    imaging_lines = list(range(0, N, ACCELERATION))
    for si, slice_index in enumerate(range(N_SLICES)):
        for li, line in enumerate(imaging_lines):
            last = si == N_SLICES - 1 and li == len(imaging_lines) - 1
            acquisitions.append(
                _line(
                    kspace[slice_index, :, line, :],
                    slice_index=slice_index,
                    line=line,
                    last=last,
                )
            )
    return acquisitions


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def test_the_calibration_maps_unalias_the_accelerated_slices(kspace, phantom, context):
    from pulserver.recon._mrd.application import _make_bucket

    plugin = epi2D_recon.Epi2DRecon(iterations=60)
    bucket = _make_bucket(stream(kspace), [])
    results = plugin(bucket, context)

    assert results is not None
    # The imaging series is 0; the calibration is never reconstructed as imaging.
    by_slice = {
        result.image_index: result.data
        for result in results
        if result.series_index == 0
    }
    assert sorted(by_slice) == list(range(N_SLICES))
    for slice_index in range(N_SLICES):
        # The recon transposes the image; compare against the transposed truth.
        # Low-resolution calibration maps leave a little more edge residual than
        # a full-resolution reference would, but a failed unalias is unmistakable
        # (its residual runs past 0.5), so this bound still pins the behaviour.
        truth = np.abs(phantom[slice_index]).T
        assert relative_error(by_slice[slice_index], truth) < 0.15


def test_the_calibration_is_not_mistaken_for_multiband(kspace, context):
    """The calibration visits every imaged slice, so the imaging does not
    collapse bands: one image per imaged slice, not one per band."""
    from pulserver.recon._mrd.application import _make_bucket

    plugin = epi2D_recon.Epi2DRecon()
    results = plugin(_make_bucket(stream(kspace), []), context)

    assert sorted(result.image_index for result in results) == list(range(N_SLICES))


def test_the_prescan_never_reaches_the_imaging_grid(kspace, context):
    """It is its own encoding space, so its low-resolution lines cannot be
    mistaken for the EPI lines they calibrate."""
    plugin = epi2D_recon.Epi2DRecon().spawn()
    plugin.startup(context)
    for acquisition in stream(kspace):
        plugin.receive(acquisition, context)

    imaging = plugin.buffers[0].mask.any(axis=-1)
    calibration = plugin.buffers[1].mask.any(axis=-1)
    # The imaging sampled every second line; the prescan a central block.
    assert imaging[0].sum() == N // ACCELERATION
    assert calibration[0].sum() == N_ACS
    assert sorted(plugin.coil_maps) == list(range(N_SLICES))


def test_the_imaging_buffer_holds_only_the_virtual_channels(kspace, context):
    """The prescan is a separate encoding space, so the basis exists before the
    first imaging line: the imaging grid is allocated compressed and never holds
    the full array."""
    virtual = 4
    plugin = epi2D_recon.Epi2DRecon(virtual_coils=virtual).spawn()
    plugin.startup(context)
    for acquisition in stream(kspace):
        plugin.receive(acquisition, context)

    assert plugin.buffers[0].kspace.shape[0] == virtual
    assert plugin.buffers[1].kspace.shape[0] == COILS


def test_the_basis_the_prescan_established_is_left_in_the_exam_cache(kspace, context):
    """The prescan is its own sequence, so the imaging may arrive as a stream of
    its own -- the exam cache is what carries the basis across."""
    plugin = epi2D_recon.Epi2DRecon().spawn()
    plugin.startup(context)
    for acquisition in stream(kspace):
        plugin.receive(acquisition, context)

    basis = context.exam.get("epi2D_coil_basis")
    assert basis is not None
    assert basis.shape == (min(8, COILS), COILS)
    # Orthonormal rows: the compression is a projection, not a rescaling.
    np.testing.assert_allclose(
        basis @ basis.conj().T, np.eye(basis.shape[0]), atol=1e-5
    )
