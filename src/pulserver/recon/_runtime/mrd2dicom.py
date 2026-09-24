"""ISMRMRD images to DICOM datasets, with series fields from the MRD XML header.

Adapted from the converter of
python-ismrmrd-server (Copyright (c) 2024 Kelvin Chow; MIT, see ``LICENSES/python-ismrmrd-server-MIT.txt``).
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
    """Converts the images of one series to DICOM, numbering instances from 1.

    Patient, study, series, system and imaging-frequency fields are read from the
    MRD header once. Header sections that fail to convert are logged and skipped.

    Parameters
    ----------
    mrdHead
        Parsed MRD XML header of the series.

    Attributes
    ----------
    dicomDset : pydicom.Dataset
        Template every image starts from.
    instanceNumber : int
        ``InstanceNumber`` of the next image.
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
        """Convert one image and increment :attr:`instanceNumber`.

        Returns ``DicomWithName(None, "")`` for RGB, multi-slice or multi-channel
        images. Floating-point pixels are quantised to ``uint16`` with
        ``RescaleIntercept`` and ``RescaleSlope``; integer pixels are stored as they
        are. The default window spans the 5th to 95th pixel percentile. Image meta
        attributes override the series description, image comment, image type,
        orientation, rescale, window, TE and TI. When the manufacturer names GE, the
        series number is the header's ``measurementID`` and the image type is
        written to a private tag.

        ``PixelSpacing`` is the row spacing, along ``phase_dir``, then the column
        spacing, along ``read_dir``. MRD's ``position`` is the image centre;
        ``ImagePositionPatient`` is the centre of the first pixel.
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

        # DICOM stores integers. A reconstruction produces real numbers, and
        # the mapping back to them is what RescaleSlope and RescaleIntercept
        # are for, so a floating-point image is quantised here rather than
        # having its bytes reinterpreted as integers.
        pixels, rescale = _quantize(np.squeeze(mrdImg.data))
        if rescale is not None:
            dicomDset.RescaleIntercept, dicomDset.RescaleSlope = rescale
            dicomDset.RescaleType = "normalized"

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
        windowMin = float(np.percentile(mrdImg.data, 5))
        windowMax = float(np.percentile(mrdImg.data, 95))
        dicomDset.WindowWidth = f"{windowMax - windowMin:.6g}"
        dicomDset.WindowCenter = f"{0.5 * (windowMin + windowMax):.6g}"

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
            - (columns - 1) / 2 * column_spacing * np.asarray(mrdImg.read_dir, float)
            - (rows - 1) / 2 * row_spacing * np.asarray(mrdImg.phase_dir, float)
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


def _quantize(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Return an image as DICOM stores it, with the mapping back to its real values.

    DICOM pixels are integers. An integer image is already storable and passes
    through untouched; a floating-point one is mapped onto the full unsigned
    16-bit range, and the ``(intercept, slope)`` returned is what recovers the
    values it came from -- ``stored * slope + intercept``.

    Parameters
    ----------
    image
        The reconstructed image, any real dtype.

    Returns
    -------
    tuple
        The pixels to store, and their ``(RescaleIntercept, RescaleSlope)``, or
        ``None`` when the image was already integral and needs no rescaling.
    """
    if not np.issubdtype(image.dtype, np.floating):
        return image, None

    finite = image[np.isfinite(image)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 0.0
    span = high - low
    if span <= 0.0:
        # A constant image: one stored value, and the intercept carries it.
        return np.zeros(image.shape, dtype=np.uint16), (low, 1.0)

    full = float(np.iinfo(np.uint16).max)
    stored = np.rint((np.nan_to_num(image, nan=low) - low) * (full / span))
    return stored.astype(np.uint16), (low, span / full)
