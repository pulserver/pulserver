"""The shipped reconstructions: their gadgets and rejected flags, and a unit of averages closed readout by readout as the proxy drives it."""

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import mrd, recon
from pulserver.recon.handlers.epi import EpiRecon
from pulserver.recon.handlers.nufft import NufftRecon
from pulserver.recon.handlers.nufft_train import NufftTrainRecon
from pulserver.recon.handlers.pics import PicsRecon, averaged
from pulserver.recon.handlers.pmc import PmcRecon

LINES, COILS, SAMPLES = 4, 2, 8


class _Summed(recon.ReconPlugin):
    """The k-space of each closed unit, its averages summed, as PicsRecon buffers it."""

    def __init__(self):
        super().__init__(
            triggers={"imaging": mrd.AcquisitionFlag.LAST_IN_SLICE},
            axes=("average",),
        )

    def recon(self, context, branch, data):
        return recon.ReconResult(np.abs(averaged(data.data)).astype(np.float32))


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
    plugin = _Summed().spawn()
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


@pytest.mark.parametrize(
    "handler",
    [
        PicsRecon,
        EpiRecon,
        NufftRecon,
        NufftTrainRecon,
        PmcRecon,
    ],
)
def test_a_shipped_reconstruction_leaves_receive_to_the_framework(handler):
    assert handler.receive is recon.ReconPlugin.receive


@pytest.mark.parametrize(
    "handler", [PicsRecon, EpiRecon, NufftRecon, NufftTrainRecon, PmcRecon]
)
def test_a_reconstruction_through_bartorch_whitens_first_and_takes_noise_readouts(
    handler,
):
    plugin = handler()

    assert isinstance(plugin.gadgets[0], recon.Prewhiten)
    assert mrd.AcquisitionFlag.IS_NOISE_MEASUREMENT not in plugin.reject_flags
    assert mrd.AcquisitionFlag.IS_PHASECORR_DATA in plugin.reject_flags


SHOTS = 6


def _train_header(places):
    matrix = SimpleNamespace(
        matrixSize=SimpleNamespace(x=SAMPLES, y=SHOTS, z=1), fieldOfView_mm=None
    )
    limits = SimpleNamespace(
        kspace_encoding_step_1=SimpleNamespace(maximum=SHOTS - 1),
        contrast=SimpleNamespace(maximum=places - 1),
    )
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=matrix,
                reconSpace=matrix,
                encodingLimits=limits,
                trajectory="radial",
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def _train(places):
    """The ``SHOTS`` readouts of an image, shot ``n`` at place ``n % places`` of a train, each place closing with its last shot."""
    closing = set({shot % places: shot for shot in range(SHOTS)}.values())
    readouts = []
    for shot in range(SHOTS):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(SAMPLES, COILS)
        acquisition.data[:] = 1.0
        acquisition.idx.kspace_encode_step_1 = shot
        acquisition.idx.contrast = shot % places
        if shot in closing:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        readouts.append(acquisition)
    return readouts


def _solved(handler, places):
    """The k-space of each unit ``handler`` is asked to solve from a train of ``places`` places."""

    class Probe(handler):
        def __init__(self):
            super().__init__()
            self.solved = []

        def recon(self, context, branch, data):
            self.solved.append(data.data)

    plugin = Probe()
    context = recon.ReconContext.offline(_train_header(places))
    plugin.startup(context)
    for acquisition in _train(places):
        plugin.receive(acquisition, context)
    plugin.flush(context)
    return plugin.solved


@pytest.mark.parametrize("places", [2, 3])
def test_a_train_reconstruction_solves_the_readouts_of_every_place_as_one_unit(places):
    each = _solved(NufftRecon, places)
    (together,) = _solved(NufftTrainRecon, places)

    assert len(each) == places
    assert not any(unit.mask.all() for unit in each)
    assert together.axes == ("coil", "phase_encode", "readout")
    assert together.mask.all()
