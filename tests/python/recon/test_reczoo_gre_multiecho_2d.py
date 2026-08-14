"""The multi-echo 2D reconstruction of the recon zoo.

A two-echo stream from an analytic phantom whose second echo is attenuated:
the plugin must keep the echoes apart, reconstruct each against the shared
maps, and emit one image per echo in echo order.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import ReconContext
from pulserver.reczoo import gre_multiecho_2d

N = 32
N_ECHOES = 2
COILS = 4
DECAY = 0.5


@pytest.fixture(scope="module")
def phantom():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + np.where((x / 0.95) ** 2 + (y / 0.95) ** 2 < 1.0, 0.7, 0.0)
    image[6:14, 4:18] += 0.5
    return image


@pytest.fixture(scope="module")
def kspace(phantom):
    """Coil k-space per echo, the second echo attenuated by ``DECAY``."""
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

    def fft2c(image):
        return np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), norm="ortho"),
            axes=(-2, -1),
        )

    full = fft2c(maps * phantom).astype(np.complex64)
    return np.stack([full, DECAY * full])


def header():
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                encodingLimits=SimpleNamespace(
                    slice=SimpleNamespace(minimum=0, maximum=0, center=0),
                    contrast=SimpleNamespace(minimum=0, maximum=N_ECHOES - 1, center=0),
                ),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def bucket(kspace):
    from pulserver.recon._mrd.application import _make_bucket

    stream = []
    total = N * N_ECHOES
    index = 0
    for line in range(N):
        for echo in range(N_ECHOES):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(N, COILS)
            acquisition.data[:] = kspace[echo, :, line, :]
            acquisition.idx.kspace_encode_step_1 = line
            acquisition.idx.contrast = echo
            acquisition.idx.slice = 0
            acquisition.center_sample = N // 2
            index += 1
            if index == total:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            stream.append(acquisition)
    return _make_bucket(stream, [])


def test_each_echo_reconstructs_to_its_own_series(kspace, phantom):
    results = gre_multiecho_2d.PLUGIN(
        bucket(kspace), ReconContext.offline(header())
    )
    assert [result.series_index for result in results] == list(range(N_ECHOES))

    first, second = (np.abs(result.data.T) for result in results)
    for image in (first, second):
        scaled = image / image.max()
        reference = phantom / phantom.max()
        assert float(
            np.linalg.norm(scaled - reference) / np.linalg.norm(reference)
        ) < 1e-5

    # The attenuation must survive: echoes are reconstructed apart, not mixed.
    assert second.max() == pytest.approx(DECAY * first.max(), rel=1e-3)


def test_the_echo_count_comes_from_the_header():
    assert gre_multiecho_2d.echo_count(header()) == N_ECHOES
    bare = header()
    bare.encoding[0].encodingLimits.contrast = None
    assert gre_multiecho_2d.echo_count(bare) == 1
