"""What a reconstruction unit's images carry from ``recon`` to the MRD image and to DICOM.

The geometry of an image is the unit's: the field of view and matrix of its
own encoding space, the position and physiology of the acquisition nearest its
k-space centre, the time stamp of its earliest acquisition. The values are the
reconstruction's: the MRD image holds them as they were returned, and integers
are made only where DICOM is written.
"""

from __future__ import annotations

import warnings

import ismrmrd
import numpy as np
import pytest

from pulserver.mrd import AcquisitionFlag
from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon._runtime.application import run_application
from pulserver.recon._runtime.mrd2dicom import DicomWithName

N_X = 8
LINES = 8
COILS = 2


def encoding(fov_mm, *, encoded=(N_X, LINES), recon=None, centre=None, slab=1):
    """An encoding space of ``encoded`` samples and lines reconstructed to ``recon``, ``fov_mm`` wide along x, y and z."""
    recon = recon or encoded
    extent = ismrmrd.xsd.fieldOfViewMm(x=fov_mm[0], y=fov_mm[1], z=fov_mm[2])
    return ismrmrd.xsd.encodingType(
        encodedSpace=ismrmrd.xsd.encodingSpaceType(
            matrixSize=ismrmrd.xsd.matrixSizeType(x=encoded[0], y=encoded[1], z=slab),
            fieldOfView_mm=extent,
        ),
        reconSpace=ismrmrd.xsd.encodingSpaceType(
            matrixSize=ismrmrd.xsd.matrixSizeType(x=recon[0], y=recon[1], z=slab),
            fieldOfView_mm=extent,
        ),
        encodingLimits=ismrmrd.xsd.encodingLimitsType(
            kspace_encoding_step_1=ismrmrd.xsd.limitType(
                minimum=0,
                maximum=encoded[1] - 1,
                center=encoded[1] // 2 if centre is None else centre,
            )
        ),
        trajectory=ismrmrd.xsd.trajectoryType("cartesian"),
    )


def header(*encodings):
    head = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        ),
        acquisitionSystemInformation=ismrmrd.xsd.acquisitionSystemInformationType(
            receiverChannels=COILS
        ),
    )
    head.encoding.extend(encodings)
    return head


def unit(lines=LINES, *, space=0, contrast=0, **fields):
    """The readouts of one image, the last carrying ``LAST_IN_SLICE``; each of ``fields`` is a function of the line."""
    acquisitions = []
    for line in range(lines):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(N_X, COILS)
        acquisition.data[:] = 1.0
        acquisition.center_sample = N_X // 2
        acquisition.encoding_space_ref = space
        acquisition.idx.kspace_encode_step_1 = line
        acquisition.idx.contrast = contrast
        for name, value in fields.items():
            setattr(acquisition, name, value(line))
        if line == lines - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        acquisitions.append(acquisition)
    return acquisitions


class Stream:
    """The connection ``run_application`` reads items from and sends outputs on."""

    def __init__(self, *items):
        self.items = items
        self.sent = []

    def __iter__(self):
        return iter(self.items)

    def send(self, item):
        self.sent.append(item)


class Emit(ReconPlugin):
    """Closes a unit at ``LAST_IN_SLICE`` and returns the result ``make`` makes of it."""

    def __init__(self, make):
        super().__init__(triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE})
        self.make = make

    def recon(self, context, branch, data):
        return self.make(data)


def run(make, head, *units):
    """Run a plugin returning ``make(data)`` for each unit over the readouts of ``units``; return what it sent."""
    stream = Stream(*(acquisition for readouts in units for acquisition in readouts))
    run_application(Emit(make), stream, ReconContext.offline(head))
    return stream.sent


def images(sent):
    return [item for item in sent if isinstance(item, ismrmrd.Image)]


def datasets(sent):
    return [item.dset for item in sent if isinstance(item, DicomWithName)]


def decoded(dataset):
    """The values a DICOM dataset's pixels stand for."""
    return dataset.pixel_array.astype(np.float64) * float(dataset.RescaleSlope) + float(
        dataset.RescaleIntercept
    )


# %% the geometry of an image is the unit's


def test_images_of_a_unit_in_the_second_encoding_space_carry_that_spaces_field_of_view_and_matrix():
    head = header(
        encoding((220.0, 220.0, 5.0)),
        encoding((300.0, 150.0, 10.0), encoded=(N_X, 4), recon=(6, 4)),
    )

    sent = images(
        run(
            lambda data: ReconResult(np.ones(data.data.image_shape, dtype=np.float32)),
            head,
            unit(),
            unit(4, space=1),
        )
    )

    assert [tuple(image.field_of_view) for image in sent] == [
        pytest.approx((220.0, 220.0, 5.0)),
        pytest.approx((300.0, 150.0, 10.0)),
    ]
    assert [tuple(image.matrix_size) for image in sent] == [(8, 8, 1), (6, 4, 1)]


def test_the_slices_of_a_volume_are_as_thick_as_the_unit_spaces_slab_over_its_partitions():
    head = header(
        encoding((220.0, 220.0, 5.0)),
        encoding((300.0, 150.0, 40.0), encoded=(N_X, 4), recon=(N_X, 4)),
    )

    sent = images(
        run(
            lambda data: ReconResult(np.ones((4, 4, N_X), dtype=np.float32)),
            head,
            unit(4, space=1),
        )
    )

    assert [float(image.field_of_view[2]) for image in sent] == [10.0] * 4
    assert [float(image.field_of_view[0]) for image in sent] == [300.0] * 4


def test_the_position_and_physiology_are_those_of_the_acquisition_nearest_the_k_space_centre():
    head = header(encoding((200.0, 200.0, 5.0), encoded=(N_X, 2 * LINES), centre=2))

    (image,) = images(
        run(
            lambda data: ReconResult(np.zeros(data.data.image_shape)),
            head,
            unit(
                position=lambda line: (float(line), 10.0 * line, 0.0),
                physiology_time_stamp=lambda line: (100 + line, 0, 0),
                patient_table_position=lambda line: (0.0, 0.0, 7.0 * line),
            ),
        )
    )

    assert tuple(image.position) == (2.0, 20.0, 0.0)
    assert image.physiology_time_stamp[0] == 102
    assert tuple(image.patient_table_position) == (0.0, 0.0, 14.0)


def test_the_time_stamp_is_the_earliest_of_the_units_acquisitions():
    stamps = [40, 30, 20, 10, 50, 60, 70, 80]
    head = header(encoding((200.0, 200.0, 5.0)))

    (image,) = images(
        run(
            lambda data: ReconResult(np.zeros(data.data.image_shape)),
            head,
            unit(acquisition_time_stamp=lambda line: stamps[line]),
        )
    )

    assert image.acquisition_time_stamp == 10


@pytest.mark.parametrize("index", [1, -1])
def test_an_integer_reference_warns_and_still_indexes_the_units_acquisitions(index):
    head = header(encoding((200.0, 200.0, 5.0)))
    readouts = unit(position=lambda line: (float(line), 0.0, 0.0))

    with pytest.warns(DeprecationWarning, match="reference") as caught:
        (image,) = images(
            run(
                lambda data: ReconResult(
                    np.zeros(data.data.image_shape), reference=index
                ),
                head,
                readouts,
            )
        )

    assert tuple(image.position) == tuple(readouts[index].position)
    assert caught[0].filename == __file__


def test_an_acquisition_is_a_reference_without_a_warning():
    head = header(encoding((200.0, 200.0, 5.0)))
    readouts = unit(position=lambda line: (float(line), 0.0, 0.0))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (image,) = images(
            run(
                lambda data: ReconResult(
                    np.zeros(data.data.image_shape), reference=data.acquisitions[3]
                ),
                head,
                readouts,
            )
        )

    assert tuple(image.position) == (3.0, 0.0, 0.0)


def test_a_reference_that_is_not_in_the_unit_is_refused():
    head = header(encoding((200.0, 200.0, 5.0)))

    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(IndexError, match="reference 8"),
    ):
        run(
            lambda data: ReconResult(np.zeros(data.data.image_shape), reference=LINES),
            head,
            unit(),
        )


# %% the values of an image are the reconstruction's


@pytest.mark.parametrize(
    ("given", "held"),
    [
        (np.float64, np.float32),
        (np.float32, np.float32),
        (np.complex128, np.complex64),
        (np.complex64, np.complex64),
        (np.int16, np.int16),
        (np.uint16, np.uint16),
    ],
)
def test_an_mrd_image_is_float32_or_complex64_and_an_integer_image_stays_an_integer(
    given, held
):
    generator = np.random.default_rng(0)
    values = 1e-4 * generator.standard_normal((LINES, N_X))
    if np.issubdtype(given, np.complexfloating):
        values = values + 1e-4j * generator.standard_normal((LINES, N_X))
    if np.issubdtype(given, np.integer):
        values = np.abs(values) * 1e5
    values = values.astype(given)

    (image,) = images(
        run(
            lambda data: ReconResult(values),
            header(encoding((200.0, 200.0, 5.0))),
            unit(),
        )
    )

    assert image.data.dtype == held
    np.testing.assert_array_equal(np.squeeze(image.data), values.astype(held))


AMPLITUDES = {
    "echo decay": (1.0, 0.6, 0.36, 0.216),
    "frame ratios": (1.0, 1.9, 0.25, 1.2),
}


@pytest.mark.parametrize("amplitudes", AMPLITUDES.values(), ids=AMPLITUDES.keys())
def test_the_ratios_between_the_images_of_a_series_survive_to_the_mrd_images_and_the_dicom_pixels(
    amplitudes,
):
    pattern = np.linspace(1e-4, 3e-4, LINES * N_X).reshape(LINES, N_X)
    head = header(encoding((200.0, 200.0, 5.0)))
    units = [unit(contrast=echo) for echo in range(len(amplitudes))]

    def make(dicom):
        return lambda data: ReconResult(
            amplitudes[data.counters["contrast"]] * pattern, dicom=dicom
        )

    mrd = images(run(make(False), head, *units))
    dicom = datasets(run(make(True), head, *units))

    expected = np.asarray(amplitudes)
    peaks = np.array([image.data.max() for image in mrd])
    np.testing.assert_allclose(peaks / peaks[0], expected, rtol=1e-6)
    stored = np.array([dataset.pixel_array.max() for dataset in dicom], dtype=float)
    np.testing.assert_allclose(stored / stored[0], expected, rtol=1e-4)
    recovered = np.array([decoded(dataset).max() for dataset in dicom])
    np.testing.assert_allclose(recovered / recovered[0], expected, rtol=1e-4)
    np.testing.assert_allclose(recovered, expected * pattern.max(), rtol=1e-4)


def test_an_explicit_rescale_in_the_attributes_is_the_mapping_of_the_dicom_pixels():
    values = np.linspace(-5.0, 95.0, LINES * N_X).reshape(LINES, N_X)
    head = header(encoding((200.0, 200.0, 5.0)))
    attributes = {"RescaleSlope": 0.01, "RescaleIntercept": -5.0}

    (dataset,) = datasets(
        run(
            lambda data: ReconResult(values, attributes=attributes, dicom=True),
            head,
            unit(),
        )
    )

    assert float(dataset.RescaleSlope) == 0.01
    assert float(dataset.RescaleIntercept) == -5.0
    np.testing.assert_array_equal(
        dataset.pixel_array, np.rint((values + 5.0) / 0.01).astype(np.uint16)
    )


# %% the partitions of a volume are mapped as one


def meta(image):
    return ismrmrd.Meta.deserialize(image.attribute_string)


@pytest.mark.parametrize(
    ("volume", "expected"),
    [
        (
            np.array([[[-2.0, 1.0]], [[0.5, 3.0]]]),
            {"ArrayMinimum": -2.0, "ArrayMaximum": 3.0},
        ),
        (
            np.array([[[3 + 4j, 0.0]], [[1j, 6 + 8j]]]),
            {"ArrayMinimum": 0.0, "ArrayMaximum": 10.0},
        ),
        (
            np.array([[[-2.0, np.inf]], [[np.nan, 3.0]]]),
            {"ArrayMinimum": -2.0, "ArrayMaximum": 3.0},
        ),
        (np.arange(4, dtype=np.int16).reshape(2, 1, 2), {}),
    ],
    ids=["real", "complex magnitude", "finite values", "integer"],
)
def test_the_images_of_a_volume_state_its_range_as_meta_attributes(volume, expected):
    sent = images(
        run(
            lambda data: ReconResult(volume),
            header(encoding((200.0, 200.0, 5.0))),
            unit(),
        )
    )

    assert len(sent) == 2
    for image in sent:
        stated = meta(image)
        assert {
            name: float(stated[name])
            for name in ("ArrayMinimum", "ArrayMaximum")
            if name in stated
        } == expected


def test_an_image_that_is_not_a_partition_states_no_array_range():
    (image,) = images(
        run(
            lambda data: ReconResult(np.ones((4, N_X))),
            header(encoding((200.0, 200.0, 5.0))),
            unit(),
        )
    )

    assert "ArrayMinimum" not in meta(image)
    assert "ArrayMaximum" not in meta(image)


def test_a_volume_sent_as_dicom_keeps_the_ratios_between_its_partitions():
    pattern = np.linspace(0.1, 1.0, 4 * N_X).reshape(4, N_X)
    volume = np.array([peak * pattern for peak in (0.01, 0.4, 1.5, 0.3)])

    sent = datasets(
        run(
            lambda data: ReconResult(volume, dicom=True),
            header(encoding((200.0, 200.0, 5.0))),
            unit(),
        )
    )

    slope = float(sent[0].RescaleSlope)
    assert len(sent) == 4
    assert sent[2].pixel_array.max() == 32768
    for plane, dataset in zip(volume, sent, strict=True):
        np.testing.assert_allclose(decoded(dataset), plane, atol=slope)


def test_an_integer_reference_of_a_volume_is_warned_of_once():
    with pytest.warns(DeprecationWarning, match="reference") as caught:
        run(
            lambda data: ReconResult(np.ones((3, 4, N_X)), reference=1),
            header(encoding((200.0, 200.0, 5.0))),
            unit(),
        )

    assert len(caught) == 1
