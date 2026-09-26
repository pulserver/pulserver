"""Tests for private ISMRMRD-to-DICOM conversion."""

import io
from datetime import date, time

import ismrmrd
import numpy as np
import pydicom
import pytest

from pulserver.recon._runtime.mrd2dicom import (
    DicomWithName,
    MrdDicomBuilder,
    _quantize,
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


def test_an_integer_image_is_stored_as_it_is():
    """Already storable: nothing to map, so no rescale is asked for."""
    image = np.arange(16, dtype=np.int16).reshape(4, 4)
    stored, rescale = _quantize(image)
    assert rescale is None
    assert stored is image


def test_a_real_image_is_quantized_with_the_mapping_back_to_it():
    """DICOM pixels are integers, so a reconstruction's real values are stored
    through ``RescaleSlope``/``RescaleIntercept`` rather than having their
    bytes reinterpreted."""
    image = np.linspace(-3.5, 12.25, 256, dtype=np.float32).reshape(16, 16)
    stored, (intercept, slope) = _quantize(image)

    assert stored.dtype == np.uint16
    assert stored.min() == 0
    assert stored.max() == np.iinfo(np.uint16).max
    recovered = stored.astype(float) * slope + intercept
    assert np.abs(recovered - image).max() < (image.max() - image.min()) / 65534


def test_a_constant_image_survives_having_no_range():
    """Nothing to spread over the integers, and the one value it does have
    still has to come back."""
    stored, (intercept, slope) = _quantize(np.full((4, 4), 0.7, dtype=np.float32))
    assert stored.dtype == np.uint16
    assert not stored.any()
    assert stored.astype(float) * slope + intercept == pytest.approx(0.7)


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
    """MRD places the image centre; DICOM places the first transmitted pixel."""
    image = _placed_image(
        (10.0, 20.0, 30.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (240.0, 120.0, 5.0)
    )
    dicom = MrdDicomBuilder(parsed_mrd_header)(image).dset
    column, row = 240.0 / 8, 120.0 / 6
    expected = (
        np.array([10.0, 20.0, 30.0])
        - 3.5 * column * np.array([0.0, 1.0, 0.0])
        - 2.5 * row * np.array([-1.0, 0.0, 0.0])
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
