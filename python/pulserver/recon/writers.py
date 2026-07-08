"""
Writers for ISMRMRD data types.

This module provides functions to serialize Python objects (e.g., acquisitions,
images, waveforms) into binary ISMRMRD data for writing to a destination
(e.g., a socket or file-like object).
"""

__all__ = [
    "write_optional",
    "write_vector",
    "write_array",
    "write_object_array",
    "write_acquisition_header",
    "write_image_header",
    "write_byte_string",
    "write_acquisition",
    "write_waveform",
    "write_image",
    "write_dicom",
    "write_text",
]

import io
import logging
import struct

from typing import Any, Callable

import numpy as np
import ismrmrd

from . import constants
from .mrd2dicom import DicomWithName


def write_optional(
    destination: Any,
    optional: Any,
    continuation: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Write an optional value to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    optional : Any
        The value to write (None if absent).
    continuation : Callable
        Function to call if the value is present.
    *args : Any
        Additional arguments for continuation.
    **kwargs : Any
        Additional keyword arguments for continuation.
    """
    if optional is None:
        destination.write(constants.bool.pack(False))
    else:
        destination.write(constants.bool.pack(True))
        continuation(destination, optional, *args, **kwargs)


def write_vector(
    destination: Any, values: list[int] | np.ndarray, type: type = constants.uint64
) -> None:
    """
    Write a vector of values to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    values : list or np.ndarray
        The values to write.
    type : type, optional
        The type for each value (default: constants.uint64).
    """
    destination.write(constants.uint64.pack(len(values)))
    for val in values:
        destination.write(type.pack(val))


def write_array(destination: Any, array: np.ndarray, dtype: type) -> None:
    """
    Write a multi-dimensional array to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    array : np.ndarray
        The array to write.
    dtype : type
        The numpy data type.
    """
    write_vector(destination, array.shape)
    array_view = np.array(array, dtype=dtype, copy=False)
    destination.write(array_view.tobytes(order="F"))


def write_object_array(
    destination: Any,
    array: np.ndarray,
    writer: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Write an array of objects to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    array : np.ndarray
        The object array to write.
    writer : Callable
        Function to write each object.
    *args : Any
        Additional arguments for writer.
    **kwargs : Any
        Additional keyword arguments for writer.
    """
    write_vector(destination, array.shape)
    for item in np.nditer(array, ("refs_ok", "zerosize_ok"), order="F"):
        item = item.item()  # Get rid of the numpy 0-dimensional array.
        writer(destination, item, *args, **kwargs)


def write_acquisition_header(
    destination: Any, header: ismrmrd.AcquisitionHeader
) -> None:
    """
    Write an acquisition header to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    header : ismrmrd.AcquisitionHeader
        The header to write.
    """
    destination.write(header)


def write_image_header(destination: Any, header: ismrmrd.ImageHeader) -> None:
    """
    Write an image header to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    header : ismrmrd.ImageHeader
        The header to write.
    """
    destination.write(header)


def write_byte_string(
    destination: Any, byte_string: bytes, type: type = constants.uint32
) -> None:
    """
    Write a byte string to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    byte_string : bytes
        The byte string to write.
    type : type, optional
        The type for the length (default: constants.uint32).
    """
    destination.write(type.pack(len(byte_string)))
    destination.write(byte_string)


def write_acquisition(destination: Any, acquisition: ismrmrd.Acquisition) -> None:
    """
    Write an acquisition to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    acquisition : ismrmrd.Acquisition
        The acquisition to write.
    """
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
    )
    destination.write(message_id_bytes)
    acquisition.serialize_into(destination.write)


def write_waveform(destination: Any, waveform: ismrmrd.Waveform) -> None:
    """
    Write a waveform to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    waveform : ismrmrd.Waveform
        The waveform to write.
    """
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM
    )
    destination.write(message_id_bytes)
    waveform.serialize_into(destination.write)


def write_image(destination: Any, image: ismrmrd.Image) -> None:
    """
    Write an image to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    image : ismrmrd.Image
        The image to write.
    """
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_ISMRMRD_IMAGE
    )
    destination.write(message_id_bytes)
    image.serialize_into(destination.write)


def write_dicom(destination: Any, dset_with_filename: DicomWithName) -> None:
    """
    Write DICOM data with filename to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    dset_with_filename : DicomWithName
        A DicomWithName object with two attributes:
            dset : pydicom.Dataset
                The DICOM dataset to write.
            filename : str
                The filename to use for the DICOM dataset.
    """
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_DICOM_WITHNAME
    )

    # Unpack input
    dset = dset_with_filename.dset
    filename = dset_with_filename.filename

    # Send each DICOM in the list
    if dset is None:
        logging.info("No DICOM dataset to send - skipping")
        return

    logging.info("-------------------------------------------------")
    logging.info(f"        sending DICOM {filename}          ")
    logging.info("-------------------------------------------------")

    # Encode file name
    filename_encoded = filename.encode("utf-8")

    # Serialize dataset to raw DICOM bytes
    buf = io.BytesIO()
    dset.save_as(buf, enforce_file_format=True)
    dicom_bytes = buf.getvalue()

    filename_length = len(filename_encoded)
    bytes_to_read = (
        filename_length + len(dicom_bytes) + 4
    )  # +4 for filename_length field

    # Pack header
    header = struct.pack("<I", bytes_to_read)
    header += struct.pack("<I", filename_length)

    # Assemble message
    message = header + filename_encoded + dicom_bytes

    # Send ID first
    destination.write(message_id_bytes)

    # Send payload
    destination.write(message)


def write_text(destination: Any, contents: str) -> None:
    """
    Write arbitrary text to the destination.

    Parameters
    ----------
    destination : Any
        The destination to write to.
    contents : str
        The text to write.
    """
    logging.info("--> Sending GADGET_MESSAGE_TEXT")
    logging.info("    %s", contents)
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_TEXT
    )
    destination.write(message_id_bytes)
    contents_with_nul = "%s\0" % contents  # Add null terminator
    destination.write(constants.uint32.pack(len(contents_with_nul.encode())))
    destination.write(contents_with_nul.encode())
