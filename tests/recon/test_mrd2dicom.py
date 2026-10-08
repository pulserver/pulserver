"""Tests for private ISMRMRD-to-DICOM conversion."""

import io
import logging
from datetime import date, time

import ismrmrd
import numpy as np
import pydicom
import pytest

from pulserver.recon._runtime.mrd2dicom import (
    DicomWithName,
    MrdDicomBuilder,
    convert_string_vrs,
    to_dicom_date,
    to_dicom_time,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal ISMRMRD headers for testing
# ---------------------------------------------------------------------------


@pytest.fixture
def parsed_mrd_header(minimal_mrd_header) -> ismrmrd.xsd.ismrmrdHeader:
    """The same header as the runtime sees it: read back from its XML.

    A parsed header carries every optional element the schema declares, where
    one built by hand leaves them ``None``.
    """
    return ismrmrd.xsd.CreateFromDocument(minimal_mrd_header.toXML())


@pytest.fixture
def minimal_mrd_header() -> ismrmrd.xsd.ismrmrdHeader:
    """Create a minimal valid ISMRMRD header."""
    # experimentalConditions is a required keyword-only field as of ismrmrd
    # >= 1.15's generated xsd dataclasses; construct it eagerly rather than
    # assigning it after the fact.
    header = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        )
    )

    header.acquisitionSystemInformation = ismrmrd.xsd.acquisitionSystemInformationType()
    header.acquisitionSystemInformation.systemVendor = "GE"
    header.acquisitionSystemInformation.systemModel = "Model 1"
    header.acquisitionSystemInformation.systemFieldStrength_T = 1.5
    header.acquisitionSystemInformation.institutionName = "Test Hospital"

    space = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(x=256, y=192, z=1),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(x=300.0, y=300.0, z=5.0),
    )
    enc = ismrmrd.xsd.encodingType(
        encodedSpace=space,
        reconSpace=space,
        encodingLimits=ismrmrd.xsd.encodingLimitsType(),
        trajectory=ismrmrd.xsd.trajectoryType("cartesian"),
    )

    header.encoding.append(enc)
    header.sequenceParameters = ismrmrd.xsd.sequenceParametersType()

    return header


@pytest.fixture
def minimal_mrd_image() -> ismrmrd.Image:
    """Create a minimal valid ISMRMRD image."""
    img = ismrmrd.Image()
    img.resize(256, 256, 1, 1)
    img.data[:] = np.random.default_rng().integers(
        0, 4096, size=img.data.shape, dtype=np.int16
    )
    img.position = (0.0, 0.0, 0.0)
    img.read_dir = (1.0, 0.0, 0.0)
    img.phase_dir = (0.0, 1.0, 0.0)
    img.slice_dir = (0.0, 0.0, 1.0)
    img.field_of_view = (300.0, 300.0, 5.0)
    img.image_index = 0
    img.image_series_index = 0
    img.slice = 0

    return img


def _bare_header(**sections) -> ismrmrd.xsd.ismrmrdHeader:
    """A header with only the required fields, plus the given sections."""
    header = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        )
    )
    header.acquisitionSystemInformation = ismrmrd.xsd.acquisitionSystemInformationType()
    header.sequenceParameters = ismrmrd.xsd.sequenceParametersType()
    for name, value in sections.items():
        setattr(header, name, value)
    return header


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_to_dicom_date_passes_a_preformatted_string_through():
    assert to_dicom_date("20260505") == "20260505"


def test_to_dicom_date_returns_an_isoformat_string_as_is():
    assert to_dicom_date("2026-05-05") == "2026-05-05"


def test_to_dicom_date_stringifies_a_python_date():
    assert to_dicom_date(date(2026, 5, 5)) == "2026-05-05"


def test_to_dicom_time_passes_a_preformatted_string_through():
    assert to_dicom_time("120000") == "120000"


def test_to_dicom_time_stringifies_a_python_time():
    assert to_dicom_time(time(12, 30, 45)) == "12:30:45"


def test_convert_string_vrs_turns_numeric_values_into_strings():
    ds = pydicom.Dataset()
    ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", 20260505))  # StudyDate
    ds.add(pydicom.DataElement((0x0008, 0x0030), "TM", 120000))  # StudyTime

    result = convert_string_vrs(ds)

    assert result.StudyDate == "20260505"
    assert result.StudyTime == "120000"
    assert isinstance(result.StudyDate, str)
    assert isinstance(result.StudyTime, str)


def test_convert_string_vrs_leaves_strings_unchanged():
    ds = pydicom.Dataset()
    ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", "20260505"))

    assert convert_string_vrs(ds).StudyDate == "20260505"


def test_convert_string_vrs_preserves_multi_valued_elements():
    ds = pydicom.Dataset()
    ds.add(pydicom.DataElement((0x0008, 0x0008), "CS", ["1", "2", "3"]))  # ImageType

    assert convert_string_vrs(ds).ImageType == ["1", "2", "3"]


def test_convert_string_vrs_recurses_into_sequences():
    ds = pydicom.Dataset()
    seq_item = pydicom.Dataset()
    seq_item.add(pydicom.DataElement((0x0008, 0x0020), "DA", "20260505"))
    ds.add(pydicom.DataElement((0x0040, 0x0100), "SQ", pydicom.Sequence([seq_item])))

    result = convert_string_vrs(ds)

    assert result[(0x0040, 0x0100)][0].StudyDate == "20260505"


def test_convert_string_vrs_modifies_in_place_and_returns_the_same_object():
    ds = pydicom.Dataset()
    ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", 20260505))

    assert convert_string_vrs(ds) is ds


# ---------------------------------------------------------------------------
# DicomWithName
# ---------------------------------------------------------------------------


def test_dicom_with_name_carries_dataset_and_filename():
    ds = pydicom.Dataset()
    result = DicomWithName(dset=ds, filename="test.dcm")
    assert result.dset is ds
    assert result.filename == "test.dcm"


def test_dicom_with_name_allows_a_none_dataset_for_failed_conversions():
    result = DicomWithName(dset=None, filename="")
    assert result.dset is None
    assert result.filename == ""


# ---------------------------------------------------------------------------
# MrdDicomBuilder
# ---------------------------------------------------------------------------


def test_the_builder_initializes_the_dicom_template(minimal_mrd_header):
    builder = MrdDicomBuilder(minimal_mrd_header)

    assert builder.dicomDset is not None
    assert builder.mrdHead is minimal_mrd_header
    # DICOM InstanceNumber is 1-based (see MrdDicomBuilder.__init__).
    assert builder.instanceNumber == 1
    assert builder.dicomDset.SamplesPerPixel == 1
    assert builder.dicomDset.PhotometricInterpretation == "MONOCHROME2"
    assert builder.dicomDset.PixelRepresentation == 0


def test_the_builder_generates_uids_when_none_are_present(minimal_mrd_header):
    builder = MrdDicomBuilder(minimal_mrd_header)

    assert builder.dicomDset.StudyInstanceUID
    assert builder.dicomDset.SeriesInstanceUID
    assert builder.dicomDset.FrameOfReferenceUID


def test_the_builder_maps_acquisition_system_information(minimal_mrd_header):
    builder = MrdDicomBuilder(minimal_mrd_header)

    assert builder.dicomDset.Manufacturer == "GE"
    assert builder.dicomDset.ManufacturerModelName == "Model 1"
    assert builder.dicomDset.MagneticFieldStrength == 1.5
    assert builder.dicomDset.InstitutionName == "Test Hospital"


def test_the_builder_maps_subject_information():
    subject = ismrmrd.xsd.subjectInformationType()
    subject.patientName = "Doe^John"
    subject.patientID = "12345"
    subject.patientWeight_kg = 75.0
    subject.patientGender = "M"

    builder = MrdDicomBuilder(_bare_header(subjectInformation=subject))

    assert builder.dicomDset.PatientName == "Doe^John"
    assert builder.dicomDset.PatientID == "12345"
    assert builder.dicomDset.PatientWeight == 75.0
    assert builder.dicomDset.PatientSex == "M"


def test_the_builder_maps_study_information():
    study = ismrmrd.xsd.studyInformationType()
    study.studyID = "STUDY001"
    study.studyDescription = "Test Study"
    study.accessionNumber = "ACC12345"

    builder = MrdDicomBuilder(_bare_header(studyInformation=study))

    assert builder.dicomDset.StudyID == "STUDY001"
    assert builder.dicomDset.StudyDescription == "Test Study"
    assert builder.dicomDset.AccessionNumber == "ACC12345"


def test_the_builder_maps_subject_study_and_system_together():
    subject = ismrmrd.xsd.subjectInformationType()
    subject.patientName = "Doe^John"
    subject.patientID = "12345"
    subject.patientWeight_kg = 75.0
    subject.patientGender = "M"

    study = ismrmrd.xsd.studyInformationType()
    study.studyID = "STUDY001"
    study.accessionNumber = "ACC12345"

    header = _bare_header(subjectInformation=subject, studyInformation=study)
    header.acquisitionSystemInformation.systemVendor = "Siemens"

    builder = MrdDicomBuilder(header)

    assert builder.dicomDset.PatientName == "Doe^John"
    assert builder.dicomDset.PatientID == "12345"
    assert builder.dicomDset.StudyID == "STUDY001"
    assert builder.dicomDset.Manufacturer == "Siemens"


# %% storing an image's real values in integer pixels

FULL = np.iinfo(np.uint16).max


def _convert(builder, values, series=0, image_type=ismrmrd.IMTYPE_MAGNITUDE, **meta):
    """The dataset ``builder`` makes of ``values``, an image of ``series`` whose meta attributes are ``meta``."""
    image = ismrmrd.Image.from_array(
        values, transpose=False, image_series_index=series, image_type=image_type
    )
    if meta:
        image.attribute_string = ismrmrd.Meta(meta).serialize()
    return builder(image).dset


def _decoded(dicom):
    """The values a dataset's pixels stand for, ``stored * RescaleSlope + RescaleIntercept``."""
    return dicom.pixel_array.astype(np.float64) * float(dicom.RescaleSlope) + float(
        dicom.RescaleIntercept
    )


def test_an_integer_image_is_stored_as_it_is(parsed_mrd_header):
    """Already storable: nothing to map, so no rescale is asked for."""
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)

    dicom = _convert(MrdDicomBuilder(parsed_mrd_header), image)

    np.testing.assert_array_equal(dicom.pixel_array, image)
    assert dicom.pixel_array.dtype == np.uint16
    assert dicom.PixelRepresentation == 0
    assert "RescaleSlope" not in dicom
    assert "RescaleIntercept" not in dicom


def test_a_signed_integer_image_is_stored_as_it_is_and_declared_signed(
    parsed_mrd_header,
):
    image = np.arange(-8, 8, dtype=np.int16).reshape(4, 4)
    builder = MrdDicomBuilder(parsed_mrd_header)

    dicom = _convert(builder, image)

    np.testing.assert_array_equal(dicom.pixel_array, image)
    assert dicom.PixelRepresentation == 1
    assert "RescaleSlope" not in dicom
    assert not builder.clipped


def test_a_real_image_is_stored_with_the_mapping_back_to_its_values(parsed_mrd_header):
    """DICOM pixels are integers, so a reconstruction's real values are stored
    through ``RescaleSlope``/``RescaleIntercept`` rather than having their
    bytes reinterpreted."""
    values = np.linspace(-3.5, 12.25, 256, dtype=np.float32).reshape(16, 16)

    dicom = _convert(MrdDicomBuilder(parsed_mrd_header), values)

    assert float(dicom.RescaleIntercept) == 0.0
    assert np.abs(_decoded(dicom) - values).max() <= float(dicom.RescaleSlope)


def test_the_first_image_of_a_series_is_stored_with_its_peak_at_half_the_stored_range(
    parsed_mrd_header,
):
    values = np.linspace(0.0, 7.0, 64, dtype=np.float32).reshape(8, 8)

    dicom = _convert(MrdDicomBuilder(parsed_mrd_header), values)

    assert dicom.pixel_array.dtype == np.uint16
    assert float(dicom.RescaleSlope) == pytest.approx(2 * 7.0 / FULL)
    # round(7 / slope) = round(65535 / 2), the nearest even integer to a half.
    assert dicom.pixel_array.max() == 32768
    assert dicom.RescaleType == "US"


def test_zero_is_stored_as_zero(parsed_mrd_header):
    values = np.zeros((4, 4), dtype=np.float32)
    values[1, 1] = 5.0

    dicom = _convert(MrdDicomBuilder(parsed_mrd_header), values)

    assert float(dicom.RescaleIntercept) == 0.0
    assert dicom.pixel_array[0, 0] == 0


def test_a_constant_image_survives_having_no_range(parsed_mrd_header):
    """Nothing to spread over the integers, and the one value it does have
    still has to come back."""
    dicom = _convert(
        MrdDicomBuilder(parsed_mrd_header), np.full((4, 4), 0.7, dtype=np.float32)
    )

    assert dicom.pixel_array.dtype == np.uint16
    assert _decoded(dicom) == pytest.approx(0.7, abs=float(dicom.RescaleSlope))


@pytest.mark.parametrize(
    "amplitudes",
    [(1.0, 0.6, 0.36, 0.216), (1.0, 1.9, 0.25, 1.2)],
    ids=["echo decay", "frame ratios"],
)
def test_the_ratios_between_the_images_of_a_series_survive_in_the_stored_integers_and_after_decoding(
    parsed_mrd_header, amplitudes
):
    pattern = np.linspace(0.1, 3.0, 64).reshape(8, 8)
    builder = MrdDicomBuilder(parsed_mrd_header)

    series = [
        _convert(builder, (amplitude * pattern).astype(np.float32))
        for amplitude in amplitudes
    ]

    expected = np.asarray(amplitudes)
    stored = np.array([dicom.pixel_array.max() for dicom in series], dtype=float)
    np.testing.assert_allclose(stored / stored[0], expected, rtol=1e-4)
    recovered = np.array([_decoded(dicom).max() for dicom in series])
    np.testing.assert_allclose(recovered / recovered[0], expected, rtol=1e-4)
    assert len({float(dicom.RescaleSlope) for dicom in series}) == 1
    assert not builder.clipped


def test_a_series_that_has_not_fixed_its_mapping_is_not_fixed_by_an_image_without_signal(
    parsed_mrd_header,
):
    builder = MrdDicomBuilder(parsed_mrd_header)
    values = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)

    empty = _convert(builder, np.zeros((4, 4), dtype=np.float32))
    first = _convert(builder, values)

    assert not _decoded(empty).any()
    assert float(first.RescaleSlope) == pytest.approx(2 * 4.0 / FULL)
    assert first.pixel_array.max() == 32768


def test_each_series_has_its_own_mapping(parsed_mrd_header):
    builder = MrdDicomBuilder(parsed_mrd_header)
    pattern = np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(4, 4)

    small = _convert(builder, pattern, series=0)
    large = _convert(builder, 1000 * pattern, series=1)
    later = _convert(builder, 0.5 * pattern, series=0)

    assert float(large.RescaleSlope) == pytest.approx(1000 * float(small.RescaleSlope))
    assert float(later.RescaleSlope) == float(small.RescaleSlope)
    assert not builder.clipped


def test_the_stored_type_is_signed_when_the_first_image_of_the_series_has_a_negative_value(
    parsed_mrd_header,
):
    builder = MrdDicomBuilder(parsed_mrd_header)
    signed = np.array([[-2.0, 1.0], [0.5, 4.0]], dtype=np.float32)
    unsigned = np.array([[0.0, 1.0], [0.5, 4.0]], dtype=np.float32)

    first = _convert(builder, signed, series=0)
    second = _convert(builder, unsigned, series=0)
    other = _convert(builder, unsigned, series=1)

    for dicom in (first, second):
        assert dicom.pixel_array.dtype == np.int16
        assert dicom.PixelRepresentation == 1
    assert other.pixel_array.dtype == np.uint16
    assert other.PixelRepresentation == 0
    # The signed range is half the unsigned one, so the same peak takes a slope
    # twice as large.
    assert float(first.RescaleSlope) == pytest.approx(2 * 4.0 / np.iinfo(np.int16).max)
    np.testing.assert_allclose(_decoded(first), signed, atol=float(first.RescaleSlope))


def test_a_complex_image_is_written_as_its_magnitude(parsed_mrd_header):
    generator = np.random.default_rng(0)
    values = (
        generator.standard_normal((6, 6)) + 1j * generator.standard_normal((6, 6))
    ).astype(np.complex64)

    dicom = _convert(
        MrdDicomBuilder(parsed_mrd_header),
        values,
        image_type=ismrmrd.IMTYPE_COMPLEX,
    )

    np.testing.assert_allclose(
        _decoded(dicom), np.abs(values), atol=float(dicom.RescaleSlope)
    )
    assert dicom.ImageType[2] == "M"


# %% the partitions of a volume are mapped as one


def _partitions(builder, peaks, series=0, **meta):
    """A volume whose partition ``k`` peaks at ``peaks[k]``, and the datasets of its partitions, each stating the range of the volume as its array."""
    pattern = np.linspace(0.1, 1.0, 16, dtype=np.float32).reshape(4, 4)
    volume = np.array([peak * pattern for peak in peaks])
    stated = {"ArrayMinimum": float(volume.min()), "ArrayMaximum": float(volume.max())}
    return volume, [
        _convert(builder, plane, series=series, **stated, **meta) for plane in volume
    ]


def test_the_partitions_of_a_volume_keep_their_ratios_when_the_first_is_the_faintest(
    parsed_mrd_header,
):
    builder = MrdDicomBuilder(parsed_mrd_header)

    volume, datasets = _partitions(builder, (0.01, 0.4, 1.5, 0.3))

    slope = float(datasets[0].RescaleSlope)
    assert slope == pytest.approx(2 * volume.max() / FULL)
    assert datasets[2].pixel_array.max() == 32768
    for plane, dicom in zip(volume, datasets, strict=True):
        np.testing.assert_allclose(_decoded(dicom), plane, atol=slope)
    assert not builder.clipped


def test_the_stored_type_follows_the_array_range_of_the_first_image(parsed_mrd_header):
    first = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)

    dicom = _convert(
        MrdDicomBuilder(parsed_mrd_header),
        first,
        ArrayMinimum=-4.0,
        ArrayMaximum=3.0,
    )

    assert dicom.pixel_array.dtype == np.int16
    assert float(dicom.RescaleSlope) == pytest.approx(2 * 4.0 / np.iinfo(np.int16).max)


def test_a_stated_mapping_takes_its_stored_type_from_the_array_range_of_the_first_image(
    parsed_mrd_header,
):
    first = np.array([[0.5, 1.0], [2.0, 3.0]], dtype=np.float32)

    dicom = _convert(
        MrdDicomBuilder(parsed_mrd_header),
        first,
        RescaleSlope="0.01",
        ArrayMinimum=-4.0,
        ArrayMaximum=3.0,
    )

    assert dicom.pixel_array.dtype == np.int16
    assert float(dicom.RescaleSlope) == 0.01


@pytest.mark.parametrize(
    "meta",
    [
        {"ArrayMaximum": 100.0},
        {"ArrayMinimum": 0.0},
        {"ArrayMinimum": 0.0, "ArrayMaximum": "inf"},
    ],
    ids=["no minimum", "no maximum", "not finite"],
)
def test_an_array_range_that_is_incomplete_or_not_finite_leaves_the_image_its_own_range(
    parsed_mrd_header, meta
):
    values = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)

    dicom = _convert(MrdDicomBuilder(parsed_mrd_header), values, **meta)

    assert float(dicom.RescaleSlope) == pytest.approx(2 * 4.0 / FULL)


def test_a_later_image_keeps_the_mapping_of_its_series_whatever_array_range_it_states(
    parsed_mrd_header,
):
    builder = MrdDicomBuilder(parsed_mrd_header)
    values = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)

    first = _convert(builder, values)
    later = _convert(builder, values, ArrayMinimum=0.0, ArrayMaximum=1000.0)

    assert float(later.RescaleSlope) == float(first.RescaleSlope)


# %% an explicit mapping is the mapping


def test_an_explicit_rescale_is_the_mapping_and_the_tags_state_it(parsed_mrd_header):
    values = np.linspace(-5.0, 95.0, 64, dtype=np.float32).reshape(8, 8)
    builder = MrdDicomBuilder(parsed_mrd_header)

    dicom = _convert(builder, values, RescaleSlope="0.01", RescaleIntercept="-5")

    assert float(dicom.RescaleSlope) == 0.01
    assert float(dicom.RescaleIntercept) == -5.0
    np.testing.assert_array_equal(
        dicom.pixel_array,
        np.rint((values.astype(float) + 5.0) / 0.01).astype(np.uint16),
    )
    np.testing.assert_allclose(_decoded(dicom), values, atol=0.01)
    assert not builder.clipped


def test_a_slope_stated_alone_leaves_the_intercept_at_zero_and_an_intercept_alone_the_slope_at_one(
    parsed_mrd_header,
):
    values = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    builder = MrdDicomBuilder(parsed_mrd_header)

    slope = _convert(builder, values, series=0, RescaleSlope="0.5")
    intercept = _convert(builder, values, series=1, RescaleIntercept="-4")

    np.testing.assert_array_equal(slope.pixel_array, [[0, 20], [40, 60]])
    assert float(slope.RescaleIntercept) == 0.0
    np.testing.assert_array_equal(intercept.pixel_array, [[4, 14], [24, 34]])
    assert float(intercept.RescaleSlope) == 1.0


def test_an_explicit_rescale_of_the_first_image_is_the_series_mapping(
    parsed_mrd_header,
):
    values = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)
    builder = MrdDicomBuilder(parsed_mrd_header)

    first = _convert(builder, values, RescaleSlope="0.001")
    later = _convert(builder, 0.5 * values)

    assert float(later.RescaleSlope) == 0.001
    np.testing.assert_array_equal(
        later.pixel_array, np.rint(0.5 * values.astype(float) / 0.001).astype(np.uint16)
    )
    assert first.pixel_array.max() == 4000


def test_an_image_stating_its_own_mapping_does_not_change_the_series_mapping(
    parsed_mrd_header,
):
    values = np.linspace(0.0, 4.0, 16, dtype=np.float32).reshape(4, 4)
    builder = MrdDicomBuilder(parsed_mrd_header)

    first = _convert(builder, values)
    own = _convert(builder, values, RescaleSlope="0.001")
    after = _convert(builder, values)

    assert float(own.RescaleSlope) == 0.001
    assert float(after.RescaleSlope) == float(first.RescaleSlope)
    np.testing.assert_array_equal(after.pixel_array, first.pixel_array)


@pytest.mark.parametrize(
    "meta",
    [
        {"RescaleSlope": "0"},
        {"RescaleSlope": "inf"},
        {"RescaleSlope": "nan"},
        {"RescaleIntercept": "inf"},
    ],
)
def test_a_stated_mapping_that_cannot_be_inverted_is_refused(parsed_mrd_header, meta):
    with pytest.raises(ValueError, match="RescaleSlope"):
        _convert(
            MrdDicomBuilder(parsed_mrd_header),
            np.ones((4, 4), dtype=np.float32),
            **meta,
        )


# %% clipping is counted and reported, never wrapped


def _clipped_series(parsed_mrd_header):
    """A builder after a series whose second image has 3 pixels beyond the range its first fixed."""
    builder = MrdDicomBuilder(parsed_mrd_header)
    first = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    second = first.copy()
    second[0, :3] = [2.5, 10.0, 1e6]
    return builder, _convert(builder, first), second


def test_a_value_beyond_the_range_of_its_series_is_clipped_and_not_wrapped(
    parsed_mrd_header,
):
    builder, _, second = _clipped_series(parsed_mrd_header)

    dicom = _convert(builder, second)

    np.testing.assert_array_equal(dicom.pixel_array[0, :3], [FULL] * 3)
    assert dicom.pixel_array[0, 3] == pytest.approx(
        second[0, 3] / float(dicom.RescaleSlope), abs=1
    )


def test_clipped_pixels_are_counted_on_the_builder_by_series(parsed_mrd_header):
    builder, _, second = _clipped_series(parsed_mrd_header)

    _convert(builder, second)
    _convert(builder, second)
    _convert(builder, second, series=1)

    assert builder.clipped[0] == 6
    assert builder.clipped[1] == 0
    assert builder.clipped[2] == 0


def test_the_first_clipping_of_a_series_is_logged_once_with_its_count(
    parsed_mrd_header, caplog
):
    builder, _, second = _clipped_series(parsed_mrd_header)

    with caplog.at_level(logging.WARNING):
        _convert(builder, second)
        _convert(builder, second)

    clipping = [record for record in caplog.records if "clipped" in record.getMessage()]
    assert len(clipping) == 1
    assert clipping[0].levelno == logging.WARNING
    assert "series 0" in clipping[0].getMessage()
    assert " 3 pixels" in clipping[0].getMessage()


def test_a_series_that_clips_does_not_silence_the_report_of_another(
    parsed_mrd_header, caplog
):
    builder, _, second = _clipped_series(parsed_mrd_header)
    pattern = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    _convert(builder, pattern, series=1)

    with caplog.at_level(logging.WARNING):
        _convert(builder, second, series=0)
        _convert(builder, second, series=1)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "clipped" in record.getMessage()
    ]
    assert len(messages) == 2
    assert "series 0" in messages[0]
    assert "series 1" in messages[1]


def test_a_value_beyond_a_stated_mapping_is_clipped_and_counted(parsed_mrd_header):
    builder = MrdDicomBuilder(parsed_mrd_header)
    values = np.array([[-1.0, 0.0], [100.0, 1e9]], dtype=np.float32)

    dicom = _convert(builder, values, RescaleSlope="0.001")

    # The first image has a negative value, so the series is signed, and 100
    # and 1e9 lie beyond ``int16`` at a slope of 0.001.
    assert dicom.pixel_array.dtype == np.int16
    np.testing.assert_array_equal(dicom.pixel_array, [[-1000, 0], [32767, 32767]])
    assert builder.clipped[0] == 2


def test_a_value_below_the_range_of_an_unsigned_series_is_clipped_at_zero(
    parsed_mrd_header,
):
    builder = MrdDicomBuilder(parsed_mrd_header)
    _convert(builder, np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4))

    dicom = _convert(builder, np.array([[-1.0, 0.5], [0.25, 1.0]], dtype=np.float32))

    assert dicom.pixel_array[0, 0] == 0
    assert builder.clipped[0] == 1


def test_nan_is_stored_as_zero_and_infinity_is_clipped(parsed_mrd_header):
    builder = MrdDicomBuilder(parsed_mrd_header)
    _convert(builder, np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4))
    values = np.array([[np.nan, np.inf], [0.5, 1.0]], dtype=np.float32)

    dicom = _convert(builder, values)

    assert dicom.pixel_array[0, 0] == 0
    assert dicom.pixel_array[0, 1] == FULL
    assert builder.clipped[0] == 1


def test_integer_images_are_never_clipped_or_counted(parsed_mrd_header):
    builder = MrdDicomBuilder(parsed_mrd_header)
    image = np.array([[0, 1], [40000, 65535]], dtype=np.uint16)

    dicom = _convert(builder, image, RescaleSlope="0.5")

    np.testing.assert_array_equal(dicom.pixel_array, image)
    assert float(dicom.RescaleSlope) == 0.5
    assert not builder.clipped


def test_a_reconstructed_image_round_trips_through_the_builder(parsed_mrd_header):
    """End to end: what a plugin computed is what a reader recovers."""
    values = np.linspace(0.3, 1.0, 64, dtype=np.float32).reshape(8, 8)
    mrd_image = ismrmrd.Image.from_array(values, transpose=False)

    dicom = MrdDicomBuilder(parsed_mrd_header)(mrd_image).dset

    assert dicom.BitsAllocated == 16
    assert dicom.PixelRepresentation == 0
    recovered = dicom.pixel_array.astype(float) * float(dicom.RescaleSlope) + float(
        dicom.RescaleIntercept
    )
    np.testing.assert_allclose(recovered, values, atol=1e-4)


def _placed_image(position, read_dir, phase_dir, fov):
    """A 6-row, 8-column image with the given geometry."""
    image = ismrmrd.Image.from_array(np.ones((6, 8), dtype=np.float32), transpose=False)
    image.position = position
    image.read_dir = read_dir
    image.phase_dir = phase_dir
    image.slice_dir = (0.0, 0.0, 1.0)
    image.field_of_view = fov
    return image


def test_pixel_spacing_is_the_row_spacing_then_the_column_spacing(parsed_mrd_header):
    """A rectangular field of view over a rectangular matrix: rows run along the
    phase direction, so their spacing is the phase field of view over the rows."""
    image = _placed_image(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (240.0, 120.0, 5.0)
    )
    dicom = MrdDicomBuilder(parsed_mrd_header)(image).dset
    assert [float(v) for v in dicom.PixelSpacing] == [120.0 / 6, 240.0 / 8]


def test_the_image_position_is_the_centre_of_the_first_pixel(parsed_mrd_header):
    """MRD places pixel n // 2, where an FFT puts the centre; DICOM places the first transmitted pixel."""
    image = _placed_image(
        (10.0, 20.0, 30.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (240.0, 120.0, 5.0)
    )
    dicom = MrdDicomBuilder(parsed_mrd_header)(image).dset
    column, row = 240.0 / 8, 120.0 / 6
    expected = (
        np.array([10.0, 20.0, 30.0])
        - 4 * column * np.array([0.0, 1.0, 0.0])
        - 3 * row * np.array([-1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose([float(v) for v in dicom.ImagePositionPatient], expected)


def test_the_default_window_is_centred_on_the_data_not_on_half_its_width(
    parsed_mrd_header,
):
    """A window is a centre and a width in real units; half the width is only
    the centre when the data starts at zero."""
    values = np.linspace(10.0, 20.0, 64, dtype=np.float32).reshape(8, 8)
    dicom = MrdDicomBuilder(parsed_mrd_header)(
        ismrmrd.Image.from_array(values, transpose=False)
    ).dset
    assert float(dicom.WindowCenter) == pytest.approx(15.0, abs=0.5)


def _ge_header(measurement):
    """A GE header whose ``measurementInformation`` holds ``measurement``, read from XML."""
    return ismrmrd.xsd.CreateFromDocument(
        '<?xml version="1.0"?>'
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
        "<acquisitionSystemInformation><systemVendor>GE MEDICAL SYSTEMS"
        "</systemVendor></acquisitionSystemInformation>"
        "<experimentalConditions><H1resonanceFrequency_Hz>63500000"
        "</H1resonanceFrequency_Hz></experimentalConditions>"
        f"<measurementInformation>{measurement}</measurementInformation>"
        "</ismrmrdHeader>"
    )


def _saved(dicom):
    buffer = io.BytesIO()
    dicom.save_as(buffer, enforce_file_format=True)
    return pydicom.dcmread(io.BytesIO(buffer.getvalue()))


def test_a_header_carrying_the_table_position_converts_and_saves():
    header = _ge_header(
        "<patientPosition>HFS</patientPosition>"
        "<relativeTablePosition><x>0</x><y>0</y><z>-120</z></relativeTablePosition>"
    )
    image = ismrmrd.Image.from_array(np.ones((4, 4), dtype=np.float32), transpose=False)
    saved = _saved(MrdDicomBuilder(header)(image).dset)
    assert saved.PatientPosition == "HFS"
    assert "TablePosition" not in saved


def test_the_series_number_on_a_ge_system_is_the_measurement_id():
    header = _ge_header(
        "<measurementID>12</measurementID><patientPosition>HFS</patientPosition>"
    )
    image = ismrmrd.Image.from_array(np.ones((4, 4), dtype=np.float32), transpose=False)
    converted = MrdDicomBuilder(header)(image)
    assert _saved(converted.dset).SeriesNumber == 12
    assert converted.filename == "EX0_12_image_001.dcm"


def test_a_ge_measurement_id_that_is_not_a_series_number_is_refused():
    header = _ge_header(
        "<measurementID>M1</measurementID><patientPosition>HFS</patientPosition>"
    )
    with pytest.raises(ValueError, match="measurementID"):
        MrdDicomBuilder(header)
