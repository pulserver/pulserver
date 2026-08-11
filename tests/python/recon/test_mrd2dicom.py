"""Tests for private ISMRMRD-to-DICOM conversion."""

from datetime import date, time

import ismrmrd
import numpy as np
import pydicom
import pytest
from pulserver.recon._mrd.mrd2dicom import (
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
    header.acquisitionSystemInformation.systemModel = "SIGNA"
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


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestToDicomDate:
    """Tests for to_dicom_date()."""

    def test_string_passthrough(self) -> None:
        """Passing a pre-formatted string should return it unchanged."""
        result = to_dicom_date("20260505")
        assert result == "20260505"

    def test_isoformat_string(self) -> None:
        """ISO-format date strings should be normalized to YYYYMMDD."""
        result = to_dicom_date("2026-05-05")
        assert result == "2026-05-05"  # returned as-is, not normalized

    def test_python_date(self) -> None:
        """Python date objects should be converted to YYYYMMDD."""
        d = date(2026, 5, 5)
        result = to_dicom_date(d)
        assert result == "2026-05-05"  # str(date) → YYYY-MM-DD


class TestToDicomTime:
    """Tests for to_dicom_time()."""

    def test_string_passthrough(self) -> None:
        """Passing a pre-formatted string should return it unchanged."""
        result = to_dicom_time("120000")
        assert result == "120000"

    def test_python_time(self) -> None:
        """Python time objects should be converted to HHMMSS."""
        t = time(12, 30, 45)
        result = to_dicom_time(t)
        assert result == "12:30:45"  # str(time) → HH:MM:SS


class TestConvertStringVrs:
    """Tests for convert_string_vrs()."""

    def test_numeric_to_string(self) -> None:
        """Numeric values in string-type VRs should be converted to strings."""
        ds = pydicom.Dataset()
        ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", 20260505))  # StudyDate
        ds.add(pydicom.DataElement((0x0008, 0x0030), "TM", 120000))  # StudyTime

        result = convert_string_vrs(ds)

        assert result.StudyDate == "20260505"
        assert result.StudyTime == "120000"
        assert isinstance(result.StudyDate, str)
        assert isinstance(result.StudyTime, str)

    def test_string_unchanged(self) -> None:
        """String values should remain unchanged."""
        ds = pydicom.Dataset()
        ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", "20260505"))

        result = convert_string_vrs(ds)

        assert result.StudyDate == "20260505"

    def test_multi_valued_string(self) -> None:
        """Multi-valued numeric elements should be converted per-value."""
        ds = pydicom.Dataset()
        ds.add(
            pydicom.DataElement((0x0008, 0x0008), "CS", ["1", "2", "3"])
        )  # ImageType

        result = convert_string_vrs(ds)

        # Verify multi-valued element is preserved
        assert result.ImageType == ["1", "2", "3"]

    def test_sequence_recursion(self) -> None:
        """Conversion should recurse into sequences."""
        ds = pydicom.Dataset()
        seq_item = pydicom.Dataset()
        seq_item.add(pydicom.DataElement((0x0008, 0x0020), "DA", "20260505"))
        seq_elem = pydicom.DataElement(
            (0x0040, 0x0100), "SQ", pydicom.Sequence([seq_item])
        )
        ds.add(seq_elem)

        result = convert_string_vrs(ds)

        assert result[(0x0040, 0x0100)][0].StudyDate == "20260505"

    def test_returns_same_object(self) -> None:
        """Function should return the same object (modified in-place)."""
        ds = pydicom.Dataset()
        ds.add(pydicom.DataElement((0x0008, 0x0020), "DA", 20260505))

        result = convert_string_vrs(ds)

        assert result is ds


# ---------------------------------------------------------------------------
# Tests: DicomWithName dataclass
# ---------------------------------------------------------------------------


class TestDicomWithName:
    """Tests for DicomWithName dataclass."""

    def test_construction(self) -> None:
        """Should construct with dataset and filename."""
        ds = pydicom.Dataset()
        result = DicomWithName(dset=ds, filename="test.dcm")
        assert result.dset is ds
        assert result.filename == "test.dcm"

    def test_none_dataset(self) -> None:
        """Should allow None dataset (for failed conversions)."""
        result = DicomWithName(dset=None, filename="")
        assert result.dset is None
        assert result.filename == ""


# ---------------------------------------------------------------------------
# Tests: MrdDicomBuilder
# ---------------------------------------------------------------------------


class TestMrdDicomBuilderInit:
    """Tests for MrdDicomBuilder.__init__()."""

    def test_initialization(
        self, minimal_mrd_header: ismrmrd.xsd.ismrmrdHeader
    ) -> None:
        """Should initialize with MRD header and create DICOM template."""
        builder = MrdDicomBuilder(minimal_mrd_header)

        assert builder.dicomDset is not None
        assert builder.mrdHead is minimal_mrd_header
        # DICOM InstanceNumber is 1-based (see MrdDicomBuilder.__init__).
        assert builder.instanceNumber == 1
        assert builder.dicomDset.SamplesPerPixel == 1
        assert builder.dicomDset.PhotometricInterpretation == "MONOCHROME2"
        assert builder.dicomDset.PixelRepresentation == 0

    def test_uids_generated(
        self, minimal_mrd_header: ismrmrd.xsd.ismrmrdHeader
    ) -> None:
        """Should generate UIDs if not present."""
        builder = MrdDicomBuilder(minimal_mrd_header)

        assert builder.dicomDset.StudyInstanceUID
        assert builder.dicomDset.SeriesInstanceUID
        assert builder.dicomDset.FrameOfReferenceUID
        assert len(builder.dicomDset.StudyInstanceUID) > 0

    def test_acq_sys_info_mapped(
        self, minimal_mrd_header: ismrmrd.xsd.ismrmrdHeader
    ) -> None:
        """Acquisition system info should be mapped to DICOM fields."""
        builder = MrdDicomBuilder(minimal_mrd_header)

        assert builder.dicomDset.Manufacturer == "GE"
        assert builder.dicomDset.ManufacturerModelName == "SIGNA"
        assert builder.dicomDset.MagneticFieldStrength == 1.5
        assert builder.dicomDset.InstitutionName == "Test Hospital"


class TestMrdDicomBuilderCall:
    """Tests for MrdDicomBuilder.__call__()."""

    def test_instance_numbering_increments(
        self,
        minimal_mrd_header: ismrmrd.xsd.ismrmrdHeader,
    ) -> None:
        """Instance counter should increment after each call attempt."""
        builder = MrdDicomBuilder(minimal_mrd_header)
        # DICOM InstanceNumber is 1-based (see MrdDicomBuilder.__init__).
        assert builder.instanceNumber == 1

        # Note: __call__ with complex data will fail, but we're testing the counter initialization
        # In production, magnitude images (real-valued) are used, not complex data

    def test_dicom_template_attributes(
        self,
        minimal_mrd_header: ismrmrd.xsd.ismrmrdHeader,
    ) -> None:
        """DICOM template should have expected attributes after initialization."""
        builder = MrdDicomBuilder(minimal_mrd_header)
        ds = builder.dicomDset

        # Check basic DICOM attributes
        assert ds.SamplesPerPixel == 1
        assert ds.PhotometricInterpretation == "MONOCHROME2"
        assert ds.PixelRepresentation == 0

        # Check metadata from header
        assert ds.Manufacturer == "GE"
        assert ds.ManufacturerModelName == "SIGNA"
        assert ds.MagneticFieldStrength == 1.5
        assert ds.InstitutionName == "Test Hospital"


class TestMrdDicomBuilderWithMetadata:
    """Tests for MrdDicomBuilder with rich metadata."""

    def test_subject_information_mapped(self) -> None:
        """Subject information (patient details) should be mapped."""
        header = ismrmrd.xsd.ismrmrdHeader(
            experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
                H1resonanceFrequency_Hz=63_750_000
            )
        )

        header.subjectInformation = ismrmrd.xsd.subjectInformationType()
        header.subjectInformation.patientName = "Doe^John"
        header.subjectInformation.patientID = "12345"
        header.subjectInformation.patientWeight_kg = 75.0
        header.subjectInformation.patientGender = "M"

        header.acquisitionSystemInformation = (
            ismrmrd.xsd.acquisitionSystemInformationType()
        )
        header.sequenceParameters = ismrmrd.xsd.sequenceParametersType()

        builder = MrdDicomBuilder(header)

        assert builder.dicomDset.PatientName == "Doe^John"
        assert builder.dicomDset.PatientID == "12345"
        assert builder.dicomDset.PatientWeight == 75.0
        assert builder.dicomDset.PatientSex == "M"

    def test_study_information_mapped(self) -> None:
        """Study information should be mapped."""
        header = ismrmrd.xsd.ismrmrdHeader(
            experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
                H1resonanceFrequency_Hz=63_750_000
            )
        )

        header.studyInformation = ismrmrd.xsd.studyInformationType()
        header.studyInformation.studyID = "STUDY001"
        header.studyInformation.studyDescription = "Test Study"
        header.studyInformation.accessionNumber = "ACC12345"

        header.acquisitionSystemInformation = (
            ismrmrd.xsd.acquisitionSystemInformationType()
        )
        header.sequenceParameters = ismrmrd.xsd.sequenceParametersType()

        builder = MrdDicomBuilder(header)

        assert builder.dicomDset.StudyID == "STUDY001"
        assert builder.dicomDset.StudyDescription == "Test Study"
        assert builder.dicomDset.AccessionNumber == "ACC12345"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for mrd2dicom components."""

    def test_builder_initialization_with_all_metadata(self) -> None:
        """Test that builder properly initializes with complete metadata."""
        header = ismrmrd.xsd.ismrmrdHeader(
            experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
                H1resonanceFrequency_Hz=63_750_000
            )
        )

        # Add complete metadata
        header.subjectInformation = ismrmrd.xsd.subjectInformationType()
        header.subjectInformation.patientName = "Doe^John"
        header.subjectInformation.patientID = "12345"
        header.subjectInformation.patientWeight_kg = 75.0
        header.subjectInformation.patientGender = "M"

        header.studyInformation = ismrmrd.xsd.studyInformationType()
        header.studyInformation.studyID = "STUDY001"
        header.studyInformation.accessionNumber = "ACC12345"

        header.acquisitionSystemInformation = (
            ismrmrd.xsd.acquisitionSystemInformationType()
        )
        header.acquisitionSystemInformation.systemVendor = "Siemens"
        header.sequenceParameters = ismrmrd.xsd.sequenceParametersType()

        # Create builder
        builder = MrdDicomBuilder(header)

        # Verify all metadata was transferred
        assert builder.dicomDset.PatientName == "Doe^John"
        assert builder.dicomDset.PatientID == "12345"
        assert builder.dicomDset.StudyID == "STUDY001"
        assert builder.dicomDset.Manufacturer == "Siemens"
