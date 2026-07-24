"""Convert a folder of DICOM files to an ISMRMRD / MRD ``.h5`` file.

This module provides both a programmatic API and a ``dicom2mrd`` CLI entry
point.  It mirrors the existing ``mrd2dicom`` module in the opposite direction.

CLI usage::

    dicom2mrd /path/to/dicom_folder -o output.h5

Or equivalently::

    python -m pulserver.recon.dicom2mrd /path/to/dicom_folder -o output.h5
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import logging
import os
import re
from collections.abc import Generator
from pathlib import Path

import ismrmrd
import numpy as np
import pydicom

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DICOM → MRD type maps
# ---------------------------------------------------------------------------

_IMAGE_TYPE_MAP: dict[str, int] = {
    "M": ismrmrd.IMTYPE_MAGNITUDE,
    "P": ismrmrd.IMTYPE_PHASE,
    "R": ismrmrd.IMTYPE_REAL,
    "I": ismrmrd.IMTYPE_IMAG,
}

_VENC_DIR_MAP: dict[str, str] = {
    "rl": "FLOW_DIR_R_TO_L",
    "lr": "FLOW_DIR_L_TO_R",
    "ap": "FLOW_DIR_A_TO_P",
    "pa": "FLOW_DIR_P_TO_A",
    "fh": "FLOW_DIR_F_TO_H",
    "hf": "FLOW_DIR_H_TO_F",
    "in": "FLOW_DIR_TP_IN",
    "out": "FLOW_DIR_TP_OUT",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dicom_folder_to_mrd(
    folder: str | Path,
    output_file: str | Path,
    group: str = "dataset",
) -> None:
    """Convert all DICOM files in *folder* to an ISMRMRD / MRD ``.h5`` file.

    Parameters
    ----------
    folder : str or Path
        Directory containing ``.dcm`` or ``.ima`` files (searched
        recursively).
    output_file : str or Path
        Destination ``.h5`` file path.  Created or overwritten.
    group : str, optional
        HDF5 group name inside the output file.  Defaults to ``"dataset"``.

    Raises
    ------
    FileNotFoundError
        If *folder* contains no DICOM files.
    """
    folder = Path(folder)
    output_file = Path(output_file)

    dcm_files = list(_iter_dicom_files(folder))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {folder}")

    datasets = [pydicom.dcmread(p) for p in dcm_files]
    logger.info("Read %d DICOM files from %s", len(datasets), folder)

    # Group by series, normalising split-series numbering from multi→single frame export
    series_nums = np.array([d.SeriesNumber for d in datasets])
    if all(series_nums > 1000):
        series_nums = (series_nums // 1000).astype(int)
        for i, d in enumerate(datasets):
            d.SeriesNumber = int(series_nums[i])

    unique_series = np.unique(series_nums)
    logger.info(
        "Found %d unique series from %d files", len(unique_series), len(datasets)
    )

    header = _build_mrd_header(datasets[0])

    mrd_dset = ismrmrd.Dataset(str(output_file), group)
    mrd_dset._file.require_group(group)
    mrd_dset.write_xml_header(bytes(header.toXML(), "utf-8"))

    for series_num in unique_series:
        series = sorted(
            [d for d in datasets if d.SeriesNumber == series_num],
            key=lambda d: d.InstanceNumber,
        )
        images = _convert_series(series)
        for img in images:
            mrd_dset.append_image(f"image_{img.image_series_index}", img)

    mrd_dset.close()
    logger.info("Wrote MRD file: %s", output_file)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_dicom_files(directory: Path) -> Generator[Path, None, None]:
    """Recursively yield paths to ``.dcm`` / ``.ima`` files.

    Parameters
    ----------
    directory : Path
        Root directory to search.

    Yields
    ------
    Path
        Path to each discovered DICOM file.
    """
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith((".dcm", ".ima")):
            yield Path(entry.path)
        elif entry.is_dir():
            yield from _iter_dicom_files(Path(entry.path))


def _build_mrd_header(dcm: pydicom.Dataset) -> ismrmrd.xsd.ismrmrdHeader:
    """Construct an ISMRMRD XML header from a single DICOM dataset.

    Parameters
    ----------
    dcm : pydicom.Dataset
        Representative DICOM file for the series (first instance is used).

    Returns
    -------
    ismrmrd.xsd.ismrmrdHeader
        Populated ISMRMRD header object.
    """
    head = ismrmrd.xsd.ismrmrdHeader()

    head.measurementInformation = ismrmrd.xsd.measurementInformationType()
    head.measurementInformation.measurementID = dcm.SeriesInstanceUID
    head.measurementInformation.patientPosition = dcm.PatientPosition
    head.measurementInformation.protocolName = dcm.SeriesDescription
    head.measurementInformation.frameOfReferenceUID = dcm.FrameOfReferenceUID

    acq_sys = ismrmrd.xsd.acquisitionSystemInformationType()
    acq_sys.systemVendor = dcm.Manufacturer
    acq_sys.systemModel = dcm.ManufacturerModelName
    acq_sys.systemFieldStrength_T = float(dcm.MagneticFieldStrength)
    acq_sys.institutionName = getattr(dcm, "InstitutionName", "Virtual")
    acq_sys.stationName = getattr(dcm, "StationName", None)
    head.acquisitionSystemInformation = acq_sys

    head.experimentalConditions = ismrmrd.xsd.experimentalConditionsType()
    head.experimentalConditions.H1resonanceFrequency_Hz = int(
        dcm.MagneticFieldStrength * 42.58e6
    )

    enc = ismrmrd.xsd.encodingType()
    enc.trajectory = ismrmrd.xsd.trajectoryType("cartesian")

    enhanced = dcm.SOPClassUID.name == "Enhanced MR Image Storage"
    if enhanced:
        pfgs = dcm.PerFrameFunctionalGroupsSequence[0]
        px = pfgs.PixelMeasuresSequence[0]
        fov_x = float(px.PixelSpacing[0]) * dcm.Rows
        fov_y = float(px.PixelSpacing[1]) * dcm.Columns
        fov_z = float(px.SliceThickness)
        nx, ny = dcm.Columns, dcm.Rows
    else:
        fov_x = float(dcm.PixelSpacing[0]) * dcm.Rows
        fov_y = float(dcm.PixelSpacing[1]) * dcm.Columns
        fov_z = float(dcm.SliceThickness)
        nx, ny = dcm.Columns, dcm.Rows

    space = ismrmrd.xsd.encodingSpaceType()
    space.matrixSize = ismrmrd.xsd.matrixSizeType(x=nx, y=ny, z=1)
    space.fieldOfView_mm = ismrmrd.xsd.fieldOfViewMm(x=fov_x, y=fov_y, z=fov_z)
    enc.encodedSpace = space
    enc.reconSpace = space
    enc.encodingLimits = ismrmrd.xsd.encodingLimitsType()

    pi = ismrmrd.xsd.parallelImagingType()
    pi.accelerationFactor = ismrmrd.xsd.accelerationFactorType()
    if enhanced:
        sfgs = dcm.SharedFunctionalGroupsSequence[0]
        mod = sfgs.MRModifierSequence[0]
        pi.accelerationFactor.kspace_encoding_step_1 = (
            mod.ParallelReductionFactorInPlane
        )
        pi.accelerationFactor.kspace_encoding_step_2 = (
            mod.ParallelReductionFactorOutOfPlane
        )
    else:
        pi.accelerationFactor.kspace_encoding_step_1 = 1
        pi.accelerationFactor.kspace_encoding_step_2 = 1
    enc.parallelImaging = pi

    head.encoding.append(enc)
    head.sequenceParameters = ismrmrd.xsd.sequenceParametersType()
    return head


def _convert_series(
    series: list[pydicom.Dataset],
) -> list[ismrmrd.Image]:
    """Convert one sorted DICOM series to a list of ``ismrmrd.Image`` objects.

    Parameters
    ----------
    series : list[pydicom.Dataset]
        DICOM instances for a single series, sorted by ``InstanceNumber``.

    Returns
    -------
    list[ismrmrd.Image]
        One ``ismrmrd.Image`` per DICOM instance, with meta-attributes and
        orientation fields populated.
    """
    series_idx = series[0].SeriesNumber

    # Unique slice locations and trigger times for index mapping
    slice_locs = np.unique([d.SliceLocation for d in series])
    if series[0].SliceLocation != slice_locs[0]:
        slice_locs = slice_locs[::-1]

    try:
        trig_times = np.unique([d.TriggerTime for d in series])
        if series[0].TriggerTime != trig_times[0]:
            trig_times = trig_times[::-1]
    except AttributeError:
        trig_times = np.zeros_like(slice_locs)

    images: list[ismrmrd.Image] = []

    for dcm in series:
        # pixel_array shape is [row, col] = [y, x]; from_array with
        # transpose=False expects [..., y, x]
        img = ismrmrd.Image.from_array(dcm.pixel_array, transpose=False)

        try:
            img.image_type = _IMAGE_TYPE_MAP.get(
                dcm.ImageType[2], ismrmrd.IMTYPE_MAGNITUDE
            )
        except (AttributeError, IndexError):
            img.image_type = ismrmrd.IMTYPE_MAGNITUDE

        img.field_of_view = (
            float(dcm.PixelSpacing[0]) * dcm.Rows,
            float(dcm.PixelSpacing[1]) * dcm.Columns,
            float(dcm.SliceThickness),
        )
        img.position = tuple(float(v) for v in dcm.ImagePositionPatient)
        orient = dcm.ImageOrientationPatient
        img.read_dir = tuple(float(v) for v in orient[:3])
        img.phase_dir = tuple(float(v) for v in orient[3:6])
        img.slice_dir = tuple(
            float(v)
            for v in np.cross(
                [float(v) for v in orient[:3]],
                [float(v) for v in orient[3:6]],
            )
        )

        # Acquisition time stamp (units: 2.5 ms ticks)
        t_str = dcm.AcquisitionTime
        t_s = (
            int(t_str[0:2]) * 3600
            + int(t_str[2:4]) * 60
            + int(t_str[4:6])
            + float(t_str[6:])
        )
        img.acquisition_time_stamp = round(t_s * 1000 / 2.5)

        with contextlib.suppress(AttributeError):
            img.physiology_time_stamp[0] = round(int(dcm.TriggerTime) / 2.5)

        try:
            table_pos = dcm.get_private_item(0x0019, 0x13, "SIEMENS MR HEADER").value
            img.patient_table_position = tuple(ctypes.c_float(v) for v in table_pos)
        except Exception:
            pass

        img.image_series_index = series_idx
        img.image_index = getattr(dcm, "InstanceNumber", 0)
        img.slice = slice_locs.tolist().index(dcm.SliceLocation)

        with contextlib.suppress(AttributeError):
            img.phase = trig_times.tolist().index(dcm.TriggerTime)

        meta = ismrmrd.Meta()
        meta["SequenceDescription"] = dcm.SeriesDescription

        try:
            res = re.search(r"(?<=_v).*$", dcm.SequenceName)
            venc_val = re.search(r"^\d+", res.group(0))
            venc_dir = re.search(r"(?<=\d)[^\d]*$", res.group(0))
            meta["FlowVelocity"] = float(venc_val.group(0))
            meta["FlowDirDisplay"] = _VENC_DIR_MAP.get(venc_dir.group(0), "")
        except (AttributeError, TypeError):
            pass

        with contextlib.suppress(AttributeError):
            meta["ImageComments"] = dcm.ImageComments

        # Embed full DICOM header as base64-encoded JSON so downstream tools
        # can recover non-MRD metadata if needed.
        dcm_copy = dcm.copy()
        del dcm_copy["PixelData"]
        meta["DicomJson"] = base64.b64encode(dcm_copy.to_json().encode("utf-8")).decode(
            "utf-8"
        )

        img.attribute_string = meta.serialize()
        images.append(img)

    return images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dicom2mrd",
        description="Convert a folder of DICOM files to an ISMRMRD/MRD .h5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", help="Input folder containing DICOM files")
    parser.add_argument(
        "-o", "--output", metavar="FILE", help="Output .h5 file (default: <folder>.h5)"
    )
    parser.add_argument(
        "-g",
        "--group",
        default="dataset",
        metavar="GROUP",
        help="HDF5 group name inside the output file",
    )
    return parser


def main() -> None:
    """CLI entry point for the ``dicom2mrd`` command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = _build_parser().parse_args()

    output = args.output or (Path(args.folder).name + ".h5")
    dicom_folder_to_mrd(args.folder, output, group=args.group)


if __name__ == "__main__":
    main()
