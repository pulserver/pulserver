"""ISMRMRD images to DICOM datasets, with series fields from the MRD XML header.

Adapted from the converter of
python-ismrmrd-server (Copyright (c) 2024 Kelvin Chow; MIT, see ``LICENSES/python-ismrmrd-server-MIT.txt``).
"""

__all__ = ["DicomWithName", "MrdDicomBuilder"]

import collections
import copy
import dataclasses
import logging
import math
from typing import Any

import ismrmrd
import numpy as np
import pydicom


@dataclasses.dataclass
class DicomWithName:
    """DICOM dataset with the file name it is sent under.

    Attributes
    ----------
    dset : pydicom.Dataset or None
        ``None`` for an image that was not converted.
    filename : str
        ``EX<StudyID>_<SeriesNumber:02>_<SeriesDescription>_<InstanceNumber:03>.dcm``,
        or empty with ``dset`` ``None``.
    """

    dset: pydicom.Dataset | None
    filename: str


#: MRD image type to the third ``ImageType`` value, or to the value of the GE
#: private image-type tag (0043,102F).
IMTYPE_MAPS = {
    ismrmrd.IMTYPE_MAGNITUDE: {"default": "M", "GE": 0},
    ismrmrd.IMTYPE_PHASE: {"default": "P", "GE": 1},
    ismrmrd.IMTYPE_REAL: {"default": "R", "GE": 2},
    ismrmrd.IMTYPE_IMAG: {"default": "I", "GE": 3},
    0: {"default": 0, "GE": 0},
}

#: DICOM value representations whose values must be strings.
STRING_VRS = {
    "AE",
    "AS",
    "CS",
    "DA",
    "DT",
    "LO",
    "LT",
    "PN",
    "SH",
    "ST",
    "TM",
    "UI",
    "UT",
}

#: The first floating-point image of a DICOM series is stored with its peak at
#: ``1 / _HEADROOM`` of the stored range, so a later image of the series may be
#: this many times larger before it clips.
_HEADROOM = 2.0


def to_dicom_date(value: Any) -> str:
    """Return an ISMRMRD ``XmlDate`` as DICOM DA (``YYYYMMDD``), any other value as ``str``."""
    if value.__class__.__name__ == "XmlDate":
        return value.to_date().isoformat().replace("-", "")  # YYYYMMDD
    return str(value)  # assume it's already in correct format


def to_dicom_time(value: Any) -> str:
    """Return an ISMRMRD ``XmlTime`` as DICOM TM (``HHMMSS``, fraction dropped), any other value as ``str``."""
    if value.__class__.__name__ == "XmlTime":
        return value.to_time().isoformat().replace(":", "").split(".")[0]  # HHMMSS
    return str(value)  # assume it's already in correct format


def convert_string_vrs(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Convert numeric and person-name values of string-VR elements to ``str``.

    Recurses into sequences. Modifies ``ds`` and returns it.
    """
    for elem in ds:
        # If this element is a sequence, recurse
        if elem.VR == "SQ":
            for item in elem.value:
                convert_string_vrs(item)

        # If the element is a string-type VR
        elif elem.VR in STRING_VRS:
            val = elem.value

            # Single numeric value → convert to str
            if isinstance(val, (int, float, pydicom.valuerep.PersonName)):
                elem.value = str(val)

            # Multi-valued → convert each numeric item to str
            elif isinstance(val, (list, tuple)):
                elem.value = [str(v) if isinstance(v, (int, float)) else v for v in val]

            # If it is already a str, leave it
    return ds


class MrdDicomBuilder:
    """Converts MRD images to DICOM, numbering instances from 1.

    Patient, study, series, system and imaging-frequency fields are read from the
    MRD header once. Header sections that fail to convert are logged and skipped.
    ``relativeTablePosition`` is not converted: the MR image has no attribute
    for it.

    One builder serves a whole connection. Its series, told apart by
    ``image_series_index``, each have their own rescale mapping and their own
    clipping count; the instance numbers run across them.

    Parameters
    ----------
    mrdHead
        Parsed MRD XML header of the series.

    Raises
    ------
    ValueError
        If the header names a GE system and its ``measurementID``, the series
        number there, is not a non-negative integer.

    Attributes
    ----------
    dicomDset : pydicom.Dataset
        Template every image starts from.
    instanceNumber : int
        ``InstanceNumber`` of the next image.
    clipped : collections.Counter
        Pixels clipped to the stored range so far, by ``image_series_index``;
        0 for a series that has clipped none.
    """

    def __init__(self, mrdHead: ismrmrd.xsd.ismrmrdHeader) -> None:
        dicomDset = pydicom.dataset.Dataset()

        # Enforce explicit little endian for written DICOM files
        dicomDset.file_meta = pydicom.dataset.FileMetaDataset()
        dicomDset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        dicomDset.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        dicomDset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        pydicom.dataset.validate_file_meta(dicomDset.file_meta)

        # ----- Set some mandatory default values -----
        if "Modality" not in dicomDset:
            dicomDset.Modality = "MR"
        if "SamplesPerPixel" not in dicomDset:
            dicomDset.SamplesPerPixel = 1
        if "PhotometricInterpretation" not in dicomDset:
            dicomDset.PhotometricInterpretation = "MONOCHROME2"
        if "PixelRepresentation" not in dicomDset:
            dicomDset.PixelRepresentation = 0  # Unsigned integer
        if "ImageType" not in dicomDset:
            dicomDset.ImageType = ["ORIGINAL", "PRIMARY", "M"]
        if "SeriesNumber" not in dicomDset:
            dicomDset.SeriesNumber = 1
        if "SeriesDescription" not in dicomDset:
            dicomDset.SeriesDescription = ""
        if "InstanceNumber" not in dicomDset:
            dicomDset.InstanceNumber = 1

        # ----- Update DICOM header from MRD header -----
        try:
            if mrdHead.subjectInformation is None:
                pass
            else:
                if mrdHead.subjectInformation.patientName is not None:
                    dicomDset.PatientName = mrdHead.subjectInformation.patientName
                if mrdHead.subjectInformation.patientWeight_kg is not None:
                    dicomDset.PatientWeight = (
                        mrdHead.subjectInformation.patientWeight_kg
                    )
                if mrdHead.subjectInformation.patientHeight_m is not None:
                    dicomDset.PatientHeight = mrdHead.subjectInformation.patientHeight_m
                if mrdHead.subjectInformation.patientID is not None:
                    dicomDset.PatientID = mrdHead.subjectInformation.patientID
                if mrdHead.subjectInformation.patientBirthdate is not None:
                    dicomDset.PatientBirthDate = to_dicom_date(
                        mrdHead.subjectInformation.patientBirthdate
                    )
                if mrdHead.subjectInformation.patientGender is not None:
                    dicomDset.PatientSex = mrdHead.subjectInformation.patientGender
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's subjectInformation section"
            )

        try:
            if mrdHead.studyInformation is None:
                pass
            else:
                if mrdHead.studyInformation.studyDate is not None:
                    dicomDset.StudyDate = to_dicom_date(
                        mrdHead.studyInformation.studyDate
                    )
                if mrdHead.studyInformation.studyTime is not None:
                    dicomDset.StudyTime = to_dicom_time(
                        mrdHead.studyInformation.studyTime
                    )
                if mrdHead.studyInformation.studyID is not None:
                    dicomDset.StudyID = mrdHead.studyInformation.studyID
                if mrdHead.studyInformation.accessionNumber is not None:
                    dicomDset.AccessionNumber = str(
                        mrdHead.studyInformation.accessionNumber
                    )
                if mrdHead.studyInformation.referringPhysicianName is not None:
                    dicomDset.ReferringPhysicianName = (
                        mrdHead.studyInformation.referringPhysicianName
                    )
                if mrdHead.studyInformation.studyDescription is not None:
                    dicomDset.StudyDescription = (
                        mrdHead.studyInformation.studyDescription
                    )
                if mrdHead.studyInformation.studyInstanceUID is not None:
                    dicomDset.StudyInstanceUID = (
                        mrdHead.studyInformation.studyInstanceUID
                    )
                if mrdHead.studyInformation.bodyPartExamined is not None:
                    dicomDset.BodyPartExamined = (
                        mrdHead.studyInformation.bodyPartExamined
                    )
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's studyInformation section"
            )

        try:
            if mrdHead.measurementInformation is None:
                pass
            else:
                # if mrdHead.measurementInformation.measurementID           is not None: dicomDset.SeriesInstanceUID   = mrdHead.measurementInformation.measurementID
                if mrdHead.measurementInformation.seriesDate is not None:
                    dicomDset.SeriesDate = to_dicom_date(
                        mrdHead.measurementInformation.seriesDate
                    )
                if mrdHead.measurementInformation.seriesTime is not None:
                    dicomDset.SeriesTime = to_dicom_time(
                        mrdHead.measurementInformation.seriesTime
                    )
                if mrdHead.measurementInformation.patientPosition is not None:
                    dicomDset.PatientPosition = (
                        mrdHead.measurementInformation.patientPosition.name
                    )
                if mrdHead.measurementInformation.protocolName is not None:
                    dicomDset.ProtocolName = mrdHead.measurementInformation.protocolName
                if mrdHead.measurementInformation.sequenceName is not None:
                    dicomDset.SequenceName = mrdHead.measurementInformation.sequenceName
                if mrdHead.measurementInformation.seriesDescription is not None:
                    dicomDset.SeriesDescription = (
                        mrdHead.measurementInformation.seriesDescription
                    )
                if mrdHead.measurementInformation.seriesInstanceUIDRoot is not None:
                    dicomDset.SeriesInstanceUID = (
                        mrdHead.measurementInformation.seriesInstanceUIDRoot
                    )
                if mrdHead.measurementInformation.frameOfReferenceUID is not None:
                    dicomDset.FrameOfReferenceUID = (
                        mrdHead.measurementInformation.frameOfReferenceUID
                    )
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's measurementInformation section"
            )

        try:
            if mrdHead.acquisitionSystemInformation is None:
                pass
            else:
                if mrdHead.acquisitionSystemInformation.systemVendor is not None:
                    dicomDset.Manufacturer = (
                        mrdHead.acquisitionSystemInformation.systemVendor
                    )
                if mrdHead.acquisitionSystemInformation.systemModel is not None:
                    dicomDset.ManufacturerModelName = (
                        mrdHead.acquisitionSystemInformation.systemModel
                    )
                if (
                    mrdHead.acquisitionSystemInformation.systemFieldStrength_T
                    is not None
                ):
                    dicomDset.MagneticFieldStrength = (
                        mrdHead.acquisitionSystemInformation.systemFieldStrength_T
                    )
                if mrdHead.acquisitionSystemInformation.institutionName is not None:
                    dicomDset.InstitutionName = (
                        mrdHead.acquisitionSystemInformation.institutionName
                    )
                if mrdHead.acquisitionSystemInformation.stationName is not None:
                    dicomDset.StationName = (
                        mrdHead.acquisitionSystemInformation.stationName
                    )
                if mrdHead.acquisitionSystemInformation.deviceID is not None:
                    dicomDset.DeviceSerialNumber = (
                        mrdHead.acquisitionSystemInformation.deviceID
                    )
                if mrdHead.acquisitionSystemInformation.deviceSerialNumber is not None:
                    dicomDset.DeviceSerialNumber = (
                        mrdHead.acquisitionSystemInformation.deviceSerialNumber
                    )
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's acquisitionSystemInformation section"
            )

        try:
            if mrdHead.experimentalConditions is None:
                pass
            else:
                if mrdHead.experimentalConditions.H1resonanceFrequency_Hz is not None:
                    dicomDset.ImagingFrequency = (
                        mrdHead.experimentalConditions.H1resonanceFrequency_Hz / 1000000
                    )
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's experimentalConditions section"
            )

        # UIDs
        if "StudyInstanceUID" not in dicomDset:
            dicomDset.StudyInstanceUID = pydicom.uid.generate_uid()
        if "SeriesInstanceUID" not in dicomDset:
            dicomDset.SeriesInstanceUID = pydicom.uid.generate_uid()
        # Use the header-provided Frame of Reference UID only if it is a valid
        # UID; otherwise fall back to a generated one (covers absent, empty, or
        # malformed values from the MRD header).
        if not pydicom.uid.UID(dicomDset.get("FrameOfReferenceUID", "")).is_valid:
            dicomDset.FrameOfReferenceUID = pydicom.uid.generate_uid()

        measurement_id = getattr(mrdHead.measurementInformation, "measurementID", None)
        if (
            "GE" in str(dicomDset.get("Manufacturer", "")).upper()
            and measurement_id is not None
            and not str(measurement_id).strip().isdigit()
        ):
            raise ValueError(
                "the header's measurementID is the DICOM series number on a GE "
                f"system, and {measurement_id!r} is not a non-negative integer"
            )

        self.dicomDset = dicomDset
        self.mrdHead = mrdHead
        # Image numbers start at 1: a scanner's image database can reject a
        # C-STORE with image number 0 (A700 OutOfResources).
        self.instanceNumber = 1
        self.clipped: collections.Counter[int] = collections.Counter()
        self._scales: dict[int, _Scale] = {}
        self._reported: set[int] = set()

    def __call__(self, mrdImg: ismrmrd.Image) -> DicomWithName:
        """Convert one image and increment :attr:`instanceNumber`.

        Returns ``DicomWithName(None, "")`` for RGB, multi-slice or multi-channel
        images. Integer pixels are stored as they are. Floating-point pixels, and
        the magnitude of complex ones, are stored as 16-bit integers under the
        mapping ``value = stored * RescaleSlope + RescaleIntercept``, so that the
        values of one series keep their ratios across its images:

        - Meta attributes that state ``RescaleSlope`` or ``RescaleIntercept``
          are the mapping, a member left out reading as 1 or 0, and
          ``stored = round((value - intercept) / slope)``. On the first
          floating-point image of a series it is the mapping of the series.
        - Otherwise the mapping is the series', fixed by its first
          floating-point image: intercept 0, and a slope that stores the peak
          magnitude of that image at half the stored range. The images that
          follow use the same mapping. A first image with no nonzero finite
          value fixes none.
        - An image whose meta attributes state ``ArrayMinimum`` and
          ``ArrayMaximum``, as the runtime states them for each partition of
          a ``(z, y, x)`` result, is the first image of its series as that
          array: the peak and the sign are the array's.
        - The stored type is ``int16`` when the first floating-point image of
          the series has a negative value under its mapping, else ``uint16``;
          later images of the series use it too.
        - A value outside the stored range is clipped to its end, never wrapped,
          and counted in :attr:`clipped`; the first clipping of a series is
          logged. NaN is stored as 0.

        The default window spans the 5th to 95th percentile of the values. Image
        meta attributes override the series description, image comment, image
        type, orientation, window, TE and TI. When the manufacturer names GE, the
        series number is the header's ``measurementID`` and the image type is
        written to a private tag.

        ``PixelSpacing`` is the row spacing, along ``phase_dir``, then the column
        spacing, along ``read_dir``. MRD's ``position`` is the centre of pixel
        ``n // 2`` along each in-plane axis, where an FFT places the
        field-of-view centre; ``ImagePositionPatient`` is the centre of the
        first pixel.

        Raises
        ------
        ValueError
            If the meta attributes state a slope that is zero or not finite, or
            an intercept that is not finite.
        """
        dicomDset = copy.deepcopy(self.dicomDset)
        mrdHead = self.mrdHead

        if (mrdImg.data.shape[0] == 3) and (mrdImg.getHead().image_type == 6):
            # RGB images
            logging.info("RGB data not yet supported")
            return DicomWithName(dset=None, filename="")
        else:
            if mrdImg.data.shape[1] != 1:
                logging.info("Multi-slice data not yet supported - skipping")
                return DicomWithName(dset=None, filename="")

            if mrdImg.data.shape[0] != 1:
                logging.info("Multi-channel data not yet supported - skipping")
                return DicomWithName(dset=None, filename="")

        # Fill Sequence Parameters from DICOM
        try:
            if mrdHead.sequenceParameters is None:
                pass
            else:
                contrastIdx = mrdImg.contrast

                # Get number of unique values per parameter
                numFA = len(mrdHead.sequenceParameters.flipAngle_deg)
                numTI = len(mrdHead.sequenceParameters.TI)
                numTE = len(mrdHead.sequenceParameters.TE)
                numTR = len(mrdHead.sequenceParameters.TR)

                # Get parameter for current image
                FA = (
                    mrdHead.sequenceParameters.flipAngle_deg[contrastIdx % numFA]
                    if mrdHead.sequenceParameters.flipAngle_deg
                    else np.nan
                )
                TI = (
                    mrdHead.sequenceParameters.TI[contrastIdx % numTI]
                    if mrdHead.sequenceParameters.TI
                    else np.nan
                )
                TE = (
                    mrdHead.sequenceParameters.TE[contrastIdx % numTE]
                    if mrdHead.sequenceParameters.TE
                    else np.nan
                )
                TR = (
                    mrdHead.sequenceParameters.TR[contrastIdx % numTR]
                    if mrdHead.sequenceParameters.TR
                    else np.nan
                )

                # Assign to dicom Header
                if mrdHead.sequenceParameters.sequence_type is not None:
                    dicomDset.SequenceVariant = mrdHead.sequenceParameters.sequence_type
                if not (np.isnan(FA)):
                    dicomDset.FlipAngle = FA
                if not (np.isnan(TI)):
                    dicomDset.InversionTime = TI
                if not (np.isnan(TE)):
                    dicomDset.EchoTime = TE
                if not (np.isnan(TR)):
                    dicomDset.RepetitionTime = TR

                # Get diffusion
                if mrdHead.sequenceParameters.diffusionDimension is not None:
                    diffusionAxis = (
                        mrdHead.sequenceParameters.diffusionDimension.name.lower()
                    )
                    diffusionIdx = getattr(mrdImg, diffusionAxis)
                    diffusionBValue = mrdHead.sequenceParameters[diffusionIdx].bvalue
                    diffusionGradientOrientation = mrdHead.sequenceParameters[
                        diffusionIdx
                    ].gradientDirection

                    # bValue = 0 -> direcionality None
                    # bValue != 0, direction None -> directionality ISOTROPIC
                    # bvalue != 0, direction not None -> directionality DIRECTIONAL

                    if diffusionBValue is None or diffusionBValue == 0.0:
                        diffusionDirectionality = "NONE"
                    else:
                        diffusionDirectionality = (
                            "ISOTROPIC"
                            if diffusionGradientOrientation is None
                            else "DIRECTIONAL"
                        )

                    # Assign
                    dicomDset.DiffusionDirectionality = diffusionDirectionality
                    if diffusionBValue is not None:
                        dicomDset.DiffusionBValue = diffusionBValue
                    if diffusionGradientOrientation is not None:
                        dicomDset.DiffusionGradientOrientation = [
                            diffusionGradientOrientation.rl,
                            diffusionGradientOrientation.ap,
                            diffusionGradientOrientation.fh,
                        ]
        except Exception:
            logging.warning(
                "Error setting header information from MRD header's sequenceParameters section"
            )

        # ----- Update DICOM header from MRD Image Data -----
        dicomDset.Rows = mrdImg.data.shape[2]
        dicomDset.Columns = mrdImg.data.shape[3]

        meta = ismrmrd.Meta.deserialize(mrdImg.attribute_string)
        values = np.squeeze(mrdImg.data)
        image_type = mrdImg.image_type
        if np.iscomplexobj(values):
            values, image_type = np.abs(values), ismrmrd.IMTYPE_MAGNITUDE
        pixels, rescale, real = self._store(mrdImg.image_series_index, values, meta)
        if rescale is not None:
            dicomDset.RescaleSlope, dicomDset.RescaleIntercept = rescale
            dicomDset.RescaleType = "US"
        dicomDset.PixelRepresentation = int(pixels.dtype.kind == "i")

        if (pixels.dtype == "uint16") or (pixels.dtype == "int16"):
            dicomDset.BitsAllocated = 16
            dicomDset.BitsStored = 16
            dicomDset.HighBit = 15
        elif (pixels.dtype == "uint32") or (pixels.dtype == "int32"):
            dicomDset.BitsAllocated = 32
            dicomDset.BitsStored = 32
            dicomDset.HighBit = 31
        else:
            logging.warning("Unsupported data type: %s", pixels.dtype)

        # Default window, in the real units a viewer sees after rescaling.
        finite = real[np.isfinite(real)]
        windowMin, windowMax = np.percentile(finite, (5, 95)) if finite.size else (0, 0)
        dicomDset.WindowWidth = f"{windowMax - windowMin:.6g}"
        dicomDset.WindowCenter = f"{0.5 * (windowMin + windowMax):.6g}"

        # ----- Update DICOM header from MRD ImageHeader -----
        vendor = dicomDset.get("Manufacturer", "default")
        vendor = "GE" if "GE" in vendor.upper() else "default"
        if "GE" in vendor.upper():
            dicomDset.add(
                pydicom.DataElement(
                    (0x0043, 0x102F), "SS", IMTYPE_MAPS[image_type][vendor]
                )
            )
        else:
            dicomDset.ImageType[2] = str(IMTYPE_MAPS[image_type][vendor])

        measurement = getattr(mrdHead, "measurementInformation", None)
        measurement_id = getattr(measurement, "measurementID", None)
        if "GE" in vendor.upper() and measurement_id is not None:
            dicomDset.SeriesNumber = measurement_id
        else:
            dicomDset.SeriesNumber = mrdImg.image_series_index
        dicomDset.InstanceNumber = self.instanceNumber
        rows, columns = mrdImg.data.shape[2], mrdImg.data.shape[3]
        row_spacing = float(mrdImg.field_of_view[1]) / rows
        column_spacing = float(mrdImg.field_of_view[0]) / columns
        dicomDset.PixelSpacing = [round(row_spacing, 6), round(column_spacing, 6)]
        dicomDset.SliceThickness = round(mrdImg.field_of_view[2], 6)
        first_pixel = (
            np.asarray(mrdImg.position, dtype=float)
            - (columns // 2) * column_spacing * np.asarray(mrdImg.read_dir, float)
            - (rows // 2) * row_spacing * np.asarray(mrdImg.phase_dir, float)
        )
        dicomDset.ImagePositionPatient = [round(float(v), 6) for v in first_pixel]
        dicomDset.ImageOrientationPatient = [
            round(mrdImg.read_dir[0], 6),
            round(mrdImg.read_dir[1], 6),
            round(mrdImg.read_dir[2], 6),
            round(mrdImg.phase_dir[0], 6),
            round(mrdImg.phase_dir[1], 6),
            round(mrdImg.phase_dir[2], 6),
        ]

        if vendor != "GE":
            time_sec = mrdImg.acquisition_time_stamp / 1000 / 2.5
            hour = int(np.floor(time_sec / 3600))
            minutes = int(np.floor((time_sec - hour * 3600) / 60))
            sec = time_sec - hour * 3600 - minutes * 60
            logging.debug(
                "acquisition time stamp %s: %.3f s",
                mrdImg.acquisition_time_stamp,
                time_sec,
            )
            dicomDset.AcquisitionTime = f"{hour:02.0f}{minutes:02.0f}{sec:09.6f}"
        dicomDset.TriggerTime = mrdImg.physiology_time_stamp[0] / 2.5

        # ----- Update DICOM header from MRD Image MetaAttributes -----
        if meta.get("SeriesDescription") is not None:
            dicomDset.SeriesDescription = meta["SeriesDescription"]

        if meta.get("SeriesDescriptionAdditional") is not None:
            dicomDset.SeriesDescription = (
                dicomDset.SeriesDescription + meta["SeriesDescriptionAdditional"]
            )

        if meta.get("ImageComment") is not None:
            dicomDset.ImageComment = "_".join(meta["ImageComment"])

        if meta.get("ImageType") is not None:
            dicomDset.ImageType = meta["ImageType"]

        if (meta.get("ImageRowDir") is not None) and (
            meta.get("ImageColumnDir") is not None
        ):
            dicomDset.ImageOrientationPatient = [
                float(meta["ImageRowDir"][0]),
                float(meta["ImageRowDir"][1]),
                float(meta["ImageRowDir"][2]),
                float(meta["ImageColumnDir"][0]),
                float(meta["ImageColumnDir"][1]),
                float(meta["ImageColumnDir"][2]),
            ]

        if meta.get("WindowCenter") is not None:
            dicomDset.WindowCenter = meta["WindowCenter"]

        if meta.get("WindowWidth") is not None:
            dicomDset.WindowWidth = meta["WindowWidth"]

        if meta.get("EchoTime") is not None:
            dicomDset.EchoTime = meta["EchoTime"]

        if meta.get("InversionTime") is not None:
            dicomDset.InversionTime = meta["InversionTime"]

        # ----- Set DICOM image from MRD Image Data -----
        # mrdImg.data is [cha z y x]; squeezed to [y x] for [row col].
        dicomDset.PixelData = pixels.tobytes()

        # UID
        dicomDset.SOPClassUID = pydicom.uid.MRImageStorage
        dicomDset.SOPInstanceUID = pydicom.uid.generate_uid()
        dicomDset.file_meta.MediaStorageSOPClassUID = dicomDset.SOPClassUID
        dicomDset.file_meta.MediaStorageSOPInstanceUID = dicomDset.SOPInstanceUID

        # Enforce correct value representation
        dicomDset = convert_string_vrs(dicomDset)

        # Generate FileName. A file acquired outside a clinical exam carries no
        # study or series identification, and a name is still needed for it.
        study = dicomDset.get("StudyID", None) or "0"
        series = dicomDset.get("SeriesNumber", None) or 0
        description = dicomDset.get("SeriesDescription", None) or "image"
        instance = dicomDset.get("InstanceNumber", None) or 0
        fileName = (
            f"EX{study}_{float(series):02.0f}_{description}_{float(instance):03.0f}.dcm"
        )

        # Update instance Number
        self.instanceNumber += 1

        return DicomWithName(dset=dicomDset, filename=fileName)

    def _store(
        self, series: int, values: np.ndarray, meta: Any
    ) -> tuple[np.ndarray, tuple[float, float] | None, np.ndarray]:
        """Return the pixels of one image of ``series``, their ``(slope, intercept)``, and the real values they stand for.

        The mapping is ``None`` for an integer image whose meta attributes state
        none, which is stored as it is. A floating-point image fixes the
        series' mapping when it is the first to; see :meth:`__call__`.
        """
        stated = _stated_mapping(meta)
        if np.issubdtype(values.dtype, np.integer):
            if stated is None:
                return values, None, values
            return values, stated, values * stated[0] + stated[1]

        known = self._scales.get(series)
        scale = known
        if stated is not None or known is None:
            scale = _first_scale(values, stated, _array_range(meta))
            if scale is not None and known is not None:
                scale = dataclasses.replace(scale, signed=known.signed)
        if scale is None:
            return np.zeros(values.shape, dtype=np.uint16), (1.0, 0.0), values
        if known is None:
            self._scales[series] = scale

        pixels, clipped = _quantize(values, scale)
        if clipped:
            self.clipped[series] += clipped
            if series not in self._reported:
                self._reported.add(series)
                logging.warning(
                    "DICOM series %d: %d pixels lie outside the stored range of "
                    "RescaleSlope %g and RescaleIntercept %g and were clipped; "
                    "later clipping is counted in MrdDicomBuilder.clipped",
                    series,
                    clipped,
                    scale.slope,
                    scale.intercept,
                )
        return pixels, (scale.slope, scale.intercept), values


@dataclasses.dataclass(frozen=True)
class _Scale:
    """Mapping of a DICOM series between its 16-bit stored integers and its real values, ``value = stored * slope + intercept``."""

    slope: float
    intercept: float
    signed: bool

    @property
    def dtype(self) -> type[np.integer]:
        return np.int16 if self.signed else np.uint16


def _stated_mapping(meta: Any) -> tuple[float, float] | None:
    """Return the ``(slope, intercept)`` the meta attributes state, a member left out reading as 1 or 0; ``None`` when they state neither.

    Raises
    ------
    ValueError
        If the slope is zero or not finite, or the intercept is not finite.
    """
    slope, intercept = meta.get("RescaleSlope"), meta.get("RescaleIntercept")
    if slope is None and intercept is None:
        return None
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)
    if slope == 0.0 or not (math.isfinite(slope) and math.isfinite(intercept)):
        raise ValueError(
            f"RescaleSlope {slope} and RescaleIntercept {intercept} are not a "
            "mapping: the slope must be nonzero and both finite"
        )
    return slope, intercept


def _array_range(meta: Any) -> tuple[float, float] | None:
    """Return the ``(ArrayMinimum, ArrayMaximum)`` the meta attributes state, ``None`` when they state either not, or one is not finite."""
    low, high = meta.get("ArrayMinimum"), meta.get("ArrayMaximum")
    if low is None or high is None:
        return None
    low, high = float(low), float(high)
    return (low, high) if math.isfinite(low) and math.isfinite(high) else None


def _first_scale(
    values: np.ndarray,
    stated: tuple[float, float] | None,
    array: tuple[float, float] | None = None,
) -> _Scale | None:
    """Return the mapping a series takes from its first floating-point image ``values``.

    That is the stated mapping, if any. Otherwise intercept 0 and the slope
    that stores the peak magnitude at ``1 / _HEADROOM`` of the stored range,
    ``int16`` when the smallest value is negative and ``uint16`` when it is
    not; ``None`` when no finite value is nonzero. The range is that of
    ``values``, or ``array``, the ``(minimum, maximum)`` of the array of images
    ``values`` is one of, when there is one.
    """
    if array is None:
        finite = values[np.isfinite(values)]
        if finite.size:
            array = (float(finite.min()), float(finite.max()))
    if array is None:
        return None if stated is None else _Scale(*stated, signed=False)
    low, high = array
    if stated is not None:
        slope, intercept = stated
        ends = np.rint((np.array(array) - intercept) / slope)
        return _Scale(slope, intercept, bool((ends < 0).any()))
    peak = max(abs(low), abs(high))
    if peak == 0.0:
        return None
    full = np.iinfo(np.int16 if low < 0 else np.uint16).max
    return _Scale(_HEADROOM * peak / full, 0.0, low < 0)


def _quantize(values: np.ndarray, scale: _Scale) -> tuple[np.ndarray, int]:
    """Return ``values`` as the integers of ``scale`` and the number of pixels clipped.

    A value is stored as ``round((value - intercept) / slope)``. One outside
    the range of the stored type, an infinite one included, is clipped to the
    nearest end of it. NaN is stored as 0 and is not counted.
    """
    limits = np.iinfo(scale.dtype)
    scaled = (values.astype(np.float64) - scale.intercept) / scale.slope
    stored = np.rint(np.where(np.isnan(scaled), 0.0, scaled))
    outside = (stored < limits.min) | (stored > limits.max)
    clipped = np.clip(stored, limits.min, limits.max).astype(scale.dtype)
    return clipped, int(outside.sum())
