"""Private conversion from ISMRMRD/MRD images to DICOM format.

This module provides utilities to convert ISMRMRD (``ismrmrd.Image``) objects
to DICOM datasets (``pydicom.Dataset``), populating DICOM headers from ISMRMRD
metadata while preserving image pixel data and geometry information.

Classes
-------
MrdDicomBuilder
    Stateful converter that maps ISMRMRD XML headers → DICOM datasets,
    accumulating per-scan instance numbering.
DicomWithName
    Lightweight struct coupling a DICOM dataset with a generated filename.

Functions
---------
to_dicom_date
    Convert ISMRMRD date values to DICOM DA (YYYYMMDD) format.
to_dicom_time
    Convert ISMRMRD time values to DICOM TM (HHMMSS) format.
convert_string_vrs
    Recursively enforce string-type value representations in DICOM elements.
"""

__all__ = ["DicomWithName", "MrdDicomBuilder"]

import copy
import dataclasses
import logging
from typing import Any

import ismrmrd
import numpy as np
import pydicom


@dataclasses.dataclass
class DicomWithName:
    """DICOM dataset coupled with a generated filename.

    Attributes
    ----------
    dset : pydicom.Dataset or None
        The DICOM dataset, or ``None`` if conversion failed.
    filename : str
        Generated filename for the DICOM instance, typically in format
        ``EX<StudyID>_<SeriesNumber>_<SeriesDescription>_<InstanceNumber>.dcm``.
    """

    dset: pydicom.Dataset | None
    filename: str


IMTYPE_MAPS = {
    ismrmrd.IMTYPE_MAGNITUDE: {"default": "M", "GE": 0},
    ismrmrd.IMTYPE_PHASE: {"default": "P", "GE": 1},
    ismrmrd.IMTYPE_REAL: {"default": "R", "GE": 2},
    ismrmrd.IMTYPE_IMAG: {"default": "I", "GE": 3},
    0: {"default": 0, "GE": 0},
}

# Lookup table between DICOM and Siemens flow directions
venc_dir_map = {
    "FLOW_DIR_R_TO_L": "rl",
    "FLOW_DIR_L_TO_R": "lr",
    "FLOW_DIR_A_TO_P": "ap",
    "FLOW_DIR_P_TO_A": "pa",
    "FLOW_DIR_F_TO_H": "fh",
    "FLOW_DIR_H_TO_F": "hf",
    "FLOW_DIR_TP_IN": "in",
    "FLOW_DIR_TP_OUT": "out",
}

# All DICOM VRs that require string values
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


def to_dicom_date(value: Any) -> str:
    """Convert to DICOM DA (YYYYMMDD) format.

    Parameters
    ----------
    value : Any
        Input value, either an ISMRMRD XML date object (XmlDate) or a
        pre-formatted DICOM date string.

    Returns
    -------
    str
        Date in DICOM DA format (``YYYYMMDD``).
    """
    if value.__class__.__name__ == "XmlDate":
        return value.to_date().isoformat().replace("-", "")  # YYYYMMDD
    return str(value)  # assume it's already in correct format


def to_dicom_time(value: Any) -> str:
    """Convert to DICOM TM (HHMMSS) format.

    Parameters
    ----------
    value : Any
        Input value, either an ISMRMRD XML time object (XmlTime) or a
        pre-formatted DICOM time string.

    Returns
    -------
    str
        Time in DICOM TM format (``HHMMSS.ffffff``) or ``HHMMSS`` if
        no fractional seconds.
    """
    if value.__class__.__name__ == "XmlTime":
        return value.to_time().isoformat().replace(":", "").split(".")[0]  # HHMMSS
    return str(value)  # assume it's already in correct format


def convert_string_vrs(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Recursively enforce string-type DICOM value representations.

    Converts numeric values to strings in all DICOM elements marked as
    string-type value representations (VR).  Handles sequences,
    multi-valued elements, and PersonName objects.

    Parameters
    ----------
    ds : pydicom.Dataset
        Input DICOM dataset to be modified in-place.

    Returns
    -------
    pydicom.Dataset
        The modified input dataset (same object).

    Notes
    -----
    This function recurses into DICOM sequences (VR == "SQ") and processes
    all elements with string-type VRs (defined in the module-level
    ``STRING_VRS`` set).
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
    """Stateful builder for converting ISMRMRD images to DICOM datasets.

    This class maintains a DICOM template and per-scan instance counter,
    allowing incremental conversion of MRD images while ensuring consistent
    numbering across a series.

    Parameters
    ----------
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        Parsed ISMRMRD XML header for the measurement.  Used to populate
        DICOM fields from patient, study, acquisition, and sequence metadata.

    Attributes
    ----------
    dicomDset : pydicom.Dataset
        The DICOM template, updated with metadata from ``mrdHead``.
    mrdHead : ismrmrd.xsd.ismrmrdHeader
        Stored reference to the input header.
    instanceNumber : int
        Counter that increments after each ``__call__``.
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
                if mrdHead.measurementInformation.relativeTablePosition is not None:
                    dicomDset.TablePosition = (
                        mrdHead.measurementInformation.relativeTablePosition
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

        self.dicomDset = dicomDset
        self.mrdHead = mrdHead
        # GE image numbers are 1-based; the local image-database broker rejects
        # a C-STORE with image number 0 (A700 OutOfResources), so start at 1.
        self.instanceNumber = 1

    def __call__(self, mrdImg: ismrmrd.Image) -> DicomWithName:
        """Convert a single ISMRMRD image to DICOM.

        Parameters
        ----------
        mrdImg : ismrmrd.Image
            ISMRMRD image to be converted.  Pixel data and geometry (position,
            orientation) are extracted; metadata from the image header and
            attribute string are used to populate DICOM fields.

        Returns
        -------
        DicomWithName
            Struct containing the output DICOM dataset and generated filename.
            If conversion fails (e.g. unsupported geometry), ``dset`` will be
            ``None`` and the filename will be empty.

        Raises
        ------
        Exception
            May raise if ISMRMRD metadata is malformed or DICOM value
            assignment fails.  Exceptions are logged and processing continues.

        Notes
        -----
        - The ``instanceNumber`` is incremented after each successful call.
        - Image orientation is extracted from the ISMRMRD ImageHeader
          (ImageRowDir, ImageColumnDir) if present in metadata, otherwise
          computed from the read and phase direction vectors.
        - Window width and center are auto-computed from the 5th and 95th
          percentiles of pixel data if not overridden by metadata.
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

        if (mrdImg.data.dtype == "uint16") or (mrdImg.data.dtype == "int16"):
            dicomDset.BitsAllocated = 16
            dicomDset.BitsStored = 16
            dicomDset.HighBit = 15
        elif (
            (mrdImg.data.dtype == "uint32")
            or (mrdImg.data.dtype == "int")
            or (mrdImg.data.dtype == "float32")
        ):
            dicomDset.BitsAllocated = 32
            dicomDset.BitsStored = 32
            dicomDset.HighBit = 31
        elif mrdImg.data.dtype == "float64":
            dicomDset.BitsAllocated = 64
            dicomDset.BitsStored = 64
            dicomDset.HighBit = 63
        else:
            logging.warning("Unsupported data type: ", mrdImg.data.dtype)

        # Default window
        windowMin = np.percentile(mrdImg.data, 5)
        windowMax = np.percentile(mrdImg.data, 95)
        windowWidth = windowMax - windowMin
        dicomDset.WindowWidth = f"{windowWidth:.6g}"
        dicomDset.WindowCenter = f"{0.5 * windowWidth:.6g}"

        # ----- Update DICOM header from MRD ImageHeader -----
        vendor = dicomDset.get("Manufacturer", "default")
        vendor = "GE" if "GE" in vendor.upper() else "default"
        if "GE" in vendor.upper():
            dicomDset.add(
                pydicom.DataElement(
                    (0x0043, 0x102F), "SS", IMTYPE_MAPS[mrdImg.image_type][vendor]
                )
            )
        else:
            dicomDset.ImageType[2] = str(IMTYPE_MAPS[mrdImg.image_type][vendor])

        if (
            "GE" in vendor.upper()
            and mrdHead.measurementInformation.measurementID is not None
        ):
            dicomDset.SeriesNumber = mrdHead.measurementInformation.measurementID
        else:
            dicomDset.SeriesNumber = mrdImg.image_series_index
        dicomDset.InstanceNumber = self.instanceNumber
        dicomDset.PixelSpacing = [
            round(float(mrdImg.field_of_view[0]) / mrdImg.data.shape[2], 6),
            round(float(mrdImg.field_of_view[1]) / mrdImg.data.shape[3], 6),
        ]
        dicomDset.SliceThickness = round(mrdImg.field_of_view[2], 6)
        dicomDset.ImagePositionPatient = [
            round(mrdImg.position[0], 6),
            round(mrdImg.position[1], 6),
            round(mrdImg.position[2], 6),
        ]
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
            logging.info(mrdImg.acquisition_time_stamp)
            logging.info(time_sec, hour, minutes, sec)
            dicomDset.AcquisitionTime = f"{hour:02.0f}{minutes:02.0f}{sec:09.6f}"
        dicomDset.TriggerTime = mrdImg.physiology_time_stamp[0] / 2.5

        # ----- Update DICOM header from MRD Image MetaAttributes -----
        meta = ismrmrd.Meta.deserialize(mrdImg.attribute_string)
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

        if meta.get("RescaleIntercept") is not None:
            dicomDset.RescaleIntercept = meta["RescaleIntercept"]

        if meta.get("RescaleSlope") is not None:
            dicomDset.RescaleSlope = meta["RescaleSlope"]

        if meta.get("WindowCenter") is not None:
            dicomDset.WindowCenter = meta["WindowCenter"]

        if meta.get("WindowWidth") is not None:
            dicomDset.WindowWidth = meta["WindowWidth"]

        if meta.get("EchoTime") is not None:
            dicomDset.EchoTime = meta["EchoTime"]

        if meta.get("InversionTime") is not None:
            dicomDset.InversionTime = meta["InversionTime"]

        # ----- Set DICOM image from MRD Image Data -----
        dicomDset.PixelData = np.squeeze(
            mrdImg.data
        ).tobytes()  # mrdImg.data is [cha z y x] -- squeeze to [y x] for [row col]

        # UID
        dicomDset.SOPClassUID = pydicom.uid.MRImageStorage
        dicomDset.SOPInstanceUID = pydicom.uid.generate_uid()
        dicomDset.file_meta.MediaStorageSOPClassUID = dicomDset.SOPClassUID
        dicomDset.file_meta.MediaStorageSOPInstanceUID = dicomDset.SOPInstanceUID

        # Enforce correct value representation
        dicomDset = convert_string_vrs(dicomDset)

        # Generate FileName
        fileName = (
            f"EX{dicomDset.StudyID}_{dicomDset.SeriesNumber:02.0f}_"
            f"{dicomDset.SeriesDescription}_{dicomDset.InstanceNumber:03.0f}.dcm"
        )

        # Update instance Number
        self.instanceNumber += 1

        return DicomWithName(dset=dicomDset, filename=fileName)
