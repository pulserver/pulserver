"""The built-in Cartesian reconstruction, driven readout by readout as the proxy drives it."""

from types import SimpleNamespace

import ismrmrd
import numpy as np

from pulserver import mrd, recon
from pulserver.recon.handlers.cartesian import PLUGIN
from pulserver.recon.handlers.simplefft import SimpleFftRecon

LINES, COILS, SAMPLES = 4, 2, 8


def _header(averages):
    matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=SAMPLES, y=LINES, z=1))
    limits = SimpleNamespace(
        kspace_encoding_step_1=SimpleNamespace(maximum=LINES - 1),
        average=SimpleNamespace(maximum=averages - 1),
    )
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=matrix, reconSpace=matrix, encodingLimits=limits
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def _average_closes(kspace):
    """The image each average's last line returns, ``None`` where it returns none."""
    averages = kspace.shape[0]
    bucket = mrd.AcquisitionBucket.from_arrays(
        kspace.reshape(averages * LINES, COILS, SAMPLES),
        labels={
            "kspace_encode_step_1": np.tile(np.arange(LINES), averages),
            "average": np.repeat(np.arange(averages), LINES),
        },
    )
    for index, acquisition in enumerate(bucket.acquisitions):
        if index % LINES == LINES - 1:
            acquisition.flags |= mrd.AcquisitionFlag.LAST_IN_SLICE.value
    context = recon.ReconContext.offline(_header(averages))
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    closes = []
    for index, acquisition in enumerate(bucket.acquisitions):
        emitted = plugin.receive(acquisition, context)
        if index % LINES == LINES - 1:
            closes.append(emitted[0][1] if emitted else None)
    return closes


def test_an_image_is_made_once_its_last_average_is_in_and_sums_them_all():
    generator = np.random.default_rng(0)
    kspace = generator.standard_normal((2, LINES, COILS, SAMPLES)) + 0j

    first, last = _average_closes(kspace)
    (alone,) = _average_closes(kspace.sum(axis=0, keepdims=True))

    assert first is None
    np.testing.assert_allclose(last.data, alone.data, rtol=1e-5)


def _transformed(kspace):
    """The root-sum-of-squares image of ``(coils, y, x)`` k-space, from numpy's own transform."""
    axes = (1, 2)
    image = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes
    )
    return np.sqrt((np.abs(image) ** 2).sum(axis=0))


def test_the_cartesian_image_is_the_transform_of_its_kspace_and_nothing_else():
    generator = np.random.default_rng(1)
    kspace = 1e-3 * (generator.standard_normal((1, LINES, COILS, SAMPLES)) + 0j) + 0j

    (image,) = _average_closes(kspace)[-1:]

    expected = _transformed(kspace[0].transpose(1, 0, 2))
    assert image.data.dtype.kind == "f"
    np.testing.assert_allclose(image.data, expected, rtol=1e-4)
    assert float(image.data.max()) < 1.0


def test_the_simple_fft_crops_to_the_matrix_of_the_space_its_lines_are_in_and_scales_nothing():
    lines = 8

    def space(x, y):
        matrix = SimpleNamespace(
            matrixSize=SimpleNamespace(x=x, y=y, z=1), fieldOfView_mm=None
        )
        return SimpleNamespace(encodedSpace=matrix, reconSpace=matrix)

    context = recon.ReconContext.offline(
        SimpleNamespace(encoding=[space(SAMPLES, lines), space(4, 6)])
    )
    generator = np.random.default_rng(2)
    kspace = 1e-3 * (
        generator.standard_normal((COILS, lines, SAMPLES))
        + 1j * generator.standard_normal((COILS, lines, SAMPLES))
    ).astype(np.complex64)
    plugin = SimpleFftRecon().spawn()
    plugin.startup(context)
    emitted = []
    for line in range(lines):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(SAMPLES, COILS)
        acquisition.data[:] = kspace[:, line]
        acquisition.encoding_space_ref = 1
        acquisition.idx.kspace_encode_step_1 = line
        if line == lines - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        emitted += plugin.receive(acquisition, context)

    ((_, image),) = emitted

    assert image.data.shape == (6, 4)
    assert image.data.dtype.kind == "f"
    # The image about its centre, cropped to 4 samples by 6 lines.
    np.testing.assert_allclose(
        image.data, _transformed(kspace)[1:7, 2:6], rtol=1e-4, atol=1e-9
    )
