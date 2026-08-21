"""Tests for built-in reconstruction apps through the private MRD adapter."""

from __future__ import annotations

import ismrmrd
import numpy as np

from pulserver import ExamCache, ReconContext
from pulserver.recon import AcquisitionFlag
from pulserver.recon._mrd.application import run_application


class FakeConnection:
    def __init__(self, items) -> None:
        self._items = list(items)
        self.sent: list = []

    def __iter__(self):
        yield from self._items

    def send(self, item) -> None:
        self.sent.append(item)


def _make_acquisitions(
    n_pe: int = 16,
    n_ro: int = 32,
    n_channels: int = 2,
    n_slices: int = 1,
) -> list[ismrmrd.Acquisition]:
    acquisitions = []
    generator = np.random.default_rng(42)
    for slice_index in range(n_slices):
        for phase_index in range(n_pe):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(n_ro, n_channels)
            acquisition.data[:] = generator.standard_normal((n_channels, n_ro)).astype(
                np.complex64
            )
            acquisition.idx.kspace_encode_step_1 = phase_index
            acquisition.idx.slice = slice_index
            if phase_index == n_pe - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
            acquisitions.append(acquisition)
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    return acquisitions


def _make_header(
    n_ro: int = 32,
    n_pe: int = 16,
    fov_x: float = 256.0,
    fov_y: float = 256.0,
    fov_z: float = 5.0,
) -> ismrmrd.xsd.ismrmrdHeader:
    encoded = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(x=n_ro, y=n_pe, z=1),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z),
    )
    reconstructed = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(x=n_ro, y=n_pe, z=1),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z),
    )
    encoding = ismrmrd.xsd.encodingType(
        encodedSpace=encoded,
        reconSpace=reconstructed,
        encodingLimits=ismrmrd.xsd.encodingLimitsType(
            kspace_encoding_step_1=ismrmrd.xsd.limitType(
                minimum=0,
                maximum=n_pe - 1,
                center=n_pe // 2,
            )
        ),
        trajectory=ismrmrd.xsd.trajectoryType("cartesian"),
    )
    return ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        ),
        encoding=[encoding],
    )


def _context(header) -> ReconContext:
    return ReconContext(header=header, exam=ExamCache(("exam", "test")), config="test")


def test_simplefft_emits_one_image_per_slice():
    from pulserver.recon._mrd.handlers.simplefft import PLUGIN

    acquisitions = _make_acquisitions(n_pe=8, n_ro=16, n_channels=2, n_slices=3)
    connection = FakeConnection(acquisitions)
    run_application(PLUGIN, connection, _context(_make_header(16, 8)))

    assert len(connection.sent) == 3
    assert all(isinstance(image, ismrmrd.Image) for image in connection.sent)
    assert all(image.data.dtype == np.int16 for image in connection.sent)


def test_simplefft_passes_non_acquisition_items_through():
    from pulserver.recon._mrd.handlers.simplefft import PLUGIN

    acquisitions = _make_acquisitions(n_pe=4, n_ro=8, n_channels=1)
    incoming_image = ismrmrd.Image.from_array(np.ones((2, 2), dtype=np.float32))
    connection = FakeConnection([incoming_image, *acquisitions])
    run_application(PLUGIN, connection, _context(_make_header(8, 4)))

    assert connection.sent[0] is incoming_image
    assert isinstance(connection.sent[1], ismrmrd.Image)


def test_fftrecon_emits_one_dicom_per_slice(monkeypatch):
    import pulserver.recon._mrd.application as application
    from pulserver.recon._mrd.handlers.fftrecon import PLUGIN

    monkeypatch.setattr(
        application,
        "MrdDicomBuilder",
        lambda _header: lambda image: ("dicom", image),
    )
    acquisitions = _make_acquisitions(n_pe=8, n_ro=16, n_channels=2, n_slices=3)
    connection = FakeConnection(acquisitions)
    run_application(PLUGIN, connection, _context(_make_header(16, 8)))

    assert len(connection.sent) == 3
    assert all(item[0] == "dicom" for item in connection.sent)
    assert [item[1].slice for item in connection.sent] == [0, 1, 2]


def test_the_waveforms_of_a_measurement_reach_every_bucket():
    """A scanner sends its waveforms once, ahead of the data."""
    from pulserver.recon._mrd.application import _make_bucket

    acquisitions = _make_acquisitions(n_pe=4, n_ro=8, n_channels=1)
    waveform = ismrmrd.Waveform()
    waveform.resize(16, 1)

    bucket = _make_bucket(acquisitions, [waveform])

    assert bucket.waveforms == (waveform,)
    assert bucket.kspace().shape == (4, 1, 8)


def test_bucket_matches_gadgetron_data_and_reference_classification():
    from pulserver.recon._mrd.application import _make_bucket

    imaging, calibration, combined, phase = _make_acquisitions(
        n_pe=4,
        n_ro=8,
        n_channels=1,
    )
    calibration.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
    combined.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING)
    phase.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)

    bucket = _make_bucket([imaging, calibration, combined, phase], [])

    assert bucket.data == (imaging, combined)
    assert bucket.ref == (calibration, combined)
    assert bucket.datastats[0].kspace_encode_step_1 == frozenset({0, 2})
    assert bucket.refstats[0].kspace_encode_step_1 == frozenset({1, 2})


def test_savedataonly_consumes_without_output():
    from pulserver.recon._mrd.handlers.savedataonly import PLUGIN

    connection = FakeConnection(_make_acquisitions(n_pe=4, n_ro=8, n_channels=1))
    run_application(PLUGIN, connection, _context(_make_header(8, 4)))
    assert connection.sent == []


def test_the_runtime_drives_the_lifecycle_hooks_in_order():
    """startup once, then receive per acquisition on arrival -- and receive is
    what calls recon, at the boundary it routes."""
    from pulserver import ReconPlugin, ReconResult

    events = []

    class Lifecycle(ReconPlugin):
        def startup(self, context):
            del context
            events.append("startup")

        def receive(self, acquisition, context):
            events.append(("receive", int(acquisition.idx.slice)))
            return super().receive(acquisition, context)

        def recon(self, branch, context):
            del context
            events.append(("recon", branch))
            return ReconResult(np.ones((2, 2), dtype=np.float32))

    acquisitions = _make_acquisitions(n_pe=3, n_ro=8, n_channels=1, n_slices=2)
    connection = FakeConnection(acquisitions)
    run_application(
        Lifecycle(branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"}, buffered=False),
        connection,
        _context(_make_header(8, 3)),
    )

    assert events == [
        "startup",
        ("receive", 0),
        ("receive", 0),
        ("receive", 0),
        ("recon", "imaging"),
        ("receive", 1),
        ("receive", 1),
        ("receive", 1),
        ("recon", "imaging"),
    ]
    assert len(connection.sent) == 2
    assert all(isinstance(item, ismrmrd.Image) for item in connection.sent)


def test_receive_routes_the_branch_the_acquisition_closed():
    """The routing reads the arriving acquisition, not an assembled bucket."""
    from pulserver import ReconPlugin
    from pulserver.recon import has_acquisition_flag

    seen = []

    class Splitting(ReconPlugin):
        def receive(self, acquisition, context):
            line = int(acquisition.idx.kspace_encode_step_1)
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
                return self.recon(f"imaging:{line}", context)
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
                return self.recon(f"calibration:{line}", context)
            return None

        def recon(self, branch, context):
            del context
            seen.append(branch)
            return None

    acquisitions = _make_acquisitions(n_pe=6, n_ro=8, n_channels=1)
    acquisitions[1].setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
    run_application(
        Splitting(buffered=False),
        FakeConnection(acquisitions),
        _context(_make_header(8, 6)),
    )

    # The segment closes at line 1 and the slice at line 5.
    assert seen == ["calibration:1", "imaging:5"]


def test_one_flag_may_be_given_without_wrapping_it():
    from pulserver import ReconPlugin

    class Nothing(ReconPlugin):
        def recon(self, branch, context):
            del branch, context
            return None

    routed = Nothing(branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"})
    assert routed.branches == {AcquisitionFlag.LAST_IN_SLICE: "imaging"}
    # No boundary at all: the stream reconstructs once, at its end.
    assert Nothing(branches={}).branches == {}
    # And the default is the end of the measurement.
    assert Nothing().branches == {AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"}


def test_each_stream_reconstructs_through_its_own_instance():
    """The configured plugin is a template; two streams cannot see each other."""
    from pulserver import ReconPlugin

    seen = []

    class Stateful(ReconPlugin):
        def startup(self, context):
            del context
            self.lines = []

        def receive(self, acquisition, context):
            self.lines.append(int(acquisition.idx.kspace_encode_step_1))
            return super().receive(acquisition, context)

        def recon(self, branch, context):
            del branch, context
            seen.append(tuple(self.lines))
            return None

    template = Stateful(branches={})
    for _ in range(2):
        run_application(
            template,
            FakeConnection(_make_acquisitions(n_pe=3, n_ro=8, n_channels=1)),
            _context(_make_header(8, 3)),
        )

    assert seen == [(0, 1, 2), (0, 1, 2)]
    assert not hasattr(template, "lines")
