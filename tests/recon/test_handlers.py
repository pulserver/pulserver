"""The built-in Cartesian reconstruction, driven readout by readout as the proxy drives it."""

from types import SimpleNamespace

import numpy as np

from pulserver import mrd, recon
from pulserver.recon.handlers.cartesian import PLUGIN

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
    context = recon.ReconContext.offline(_header(averages))
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    closes = []
    for index, acquisition in enumerate(bucket.acquisitions):
        plugin.receive(acquisition, context)
        if index % LINES == LINES - 1:
            closes.append(plugin.recon("imaging", context))
    return closes


def test_an_image_is_made_once_its_last_average_is_in_and_sums_them_all():
    generator = np.random.default_rng(0)
    kspace = generator.standard_normal((2, LINES, COILS, SAMPLES)) + 0j

    first, last = _average_closes(kspace)
    (alone,) = _average_closes(kspace.sum(axis=0, keepdims=True))

    assert first is None
    np.testing.assert_array_equal(last.data, alone.data)
