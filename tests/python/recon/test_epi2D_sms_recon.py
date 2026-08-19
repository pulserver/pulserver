"""The multiband (SMS) branch of the 2D EPI reconstruction.

Driven over the acquisitions the runtime would stream from
:mod:`pulserver.app.sequence.epi2D_sequence` under ``SMS_EXCITATION``: a low-resolution GRE
calibration for each slice, then the blipped-CAIPI multiband shots whose slices
collapse with the CAIPI phase the sequence played. The reconstruction has to
estimate each slice's coil maps from the calibration and unfold each group back
into its bands, placing them at the right slices.
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
N_SLICES = 6
N_BANDS = 3
N_GROUPS = N_SLICES // N_BANDS
COILS = 8
N_ACS = 24


@pytest.fixture(scope="module")
def phantom():
    """One distinct disk per slice, so a mix-up between slices is visible."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    slices = []
    for s in range(N_SLICES):
        radius = 0.85 - 0.05 * s
        plane = 0.2 + np.where(
            (x - 0.15 * s) ** 2 + (y + 0.1 * s) ** 2 < radius**2, 0.8, 0.0
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


def _encoding(n_slices):
    """One encoding space, over ``n_slices`` slices of the same matrix."""
    matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1))
    return SimpleNamespace(
        encodedSpace=matrix,
        reconSpace=matrix,
        encodingLimits=SimpleNamespace(slice=SimpleNamespace(maximum=n_slices - 1)),
    )


def header():
    """Two encoding spaces: the multiband imaging, then the prescan.

    The calibration is its own subsequence, so it is its own encoding space --
    and it visits every slice while the imaging excites ``N_GROUPS`` combs,
    which is what tells the reconstruction the scan was multiband.
    """
    return SimpleNamespace(
        encoding=[_encoding(N_GROUPS), _encoding(N_SLICES)],
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


def _caipi(n_bands):
    ky = np.arange(N)
    return np.exp(
        1j * 2 * np.pi * (np.arange(n_bands)[:, None] / n_bands) * ky[None, :]
    )


def _calibration_lines():
    """The fully sampled central ``N_ACS`` window, centre-anchored."""
    start = (N - N_ACS) // 2
    return list(range(start, start + N_ACS))


def stream(kspace):
    """GRE calibration (central lines per slice) then the collapsed shots."""
    acquisitions = []
    # Low-resolution GRE calibration: a central block per slice, marked
    # calibration -- the same block the plain accelerated scan uses.
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
    # Multiband imaging: one shot per group, its bands collapsed with the CAIPI
    # phase the sequence's gz blips play.
    caipi = _caipi(N_BANDS)
    for group in range(N_GROUPS):
        bands = [group + band * N_GROUPS for band in range(N_BANDS)]
        for line in range(N):
            collapsed = sum(
                caipi[band, line] * kspace[bands[band], :, line, :]
                for band in range(N_BANDS)
            )
            last = group == N_GROUPS - 1 and line == N - 1
            acquisitions.append(
                _line(collapsed, slice_index=group, line=line, last=last)
            )
    return acquisitions


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def test_the_multiband_branch_separates_the_slices(kspace, phantom, context):
    from pulserver.recon._mrd.application import _make_bucket

    plugin = epi2D_recon.Epi2DRecon(iterations=60)
    bucket = _make_bucket(stream(kspace), [])
    results = plugin(bucket, context)

    assert results is not None
    by_slice = {result.image_index: result.data for result in results}
    assert sorted(by_slice) == list(range(N_SLICES))
    for slice_index in range(N_SLICES):
        # The recon transposes the image; compare against the transposed truth.
        truth = np.abs(phantom[slice_index]).T
        assert relative_error(by_slice[slice_index], truth) < 0.1
