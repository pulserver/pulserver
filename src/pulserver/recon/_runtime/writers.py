"""Writers for MRD streaming messages; each takes a destination with ``write(bytes)``.

Adapted from the writers of
gadgetron-python (Copyright (c) 2019 Gadgetron; MIT, see ``LICENSES/gadgetron-python-MIT.txt``).
"""

__all__ = [
    "header_document",
    "write_acquisition",
    "write_acquisition_header",
    "write_array",
    "write_byte_string",
    "write_config_text",
    "write_dicom",
    "write_header",
    "write_image",
    "write_image_header",
    "write_object_array",
    "write_optional",
    "write_text",
    "write_vector",
    "write_waveform",
]

import io
import logging
import struct
from collections.abc import Callable
from typing import Any

import ismrmrd
import numpy as np

from . import constants
from .mrd2dicom import DicomWithName


def write_optional(
    destination: Any,
    optional: Any,
    continuation: Callable[..., None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Write a presence flag, then the value with ``continuation`` when it is not ``None``."""
    if optional is None:
        destination.write(constants.bool.pack(False))
    else:
        destination.write(constants.bool.pack(True))
        continuation(destination, optional, *args, **kwargs)


def write_vector(
    destination: Any,
    values: list[int] | np.ndarray,
    struct_type: struct.Struct = constants.uint64,
) -> None:
    """Write a ``uint64`` count, then each value packed with ``struct_type``."""
    destination.write(constants.uint64.pack(len(values)))
    for val in values:
        destination.write(struct_type.pack(val))


def write_array(destination: Any, array: np.ndarray, dtype: type) -> None:
    """Write the shape as a vector, then the values as ``dtype`` in Fortran order."""
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
    """Write the shape as a vector, then each element with ``writer`` in Fortran order."""
    write_vector(destination, array.shape)
    for item in np.nditer(array, ("refs_ok", "zerosize_ok"), order="F"):
        writer(destination, item.item(), *args, **kwargs)


def write_acquisition_header(
    destination: Any, header: ismrmrd.AcquisitionHeader
) -> None:
    destination.write(header)


def write_image_header(destination: Any, header: ismrmrd.ImageHeader) -> None:
    destination.write(header)


def write_byte_string(
    destination: Any,
    byte_string: bytes,
    struct_type: struct.Struct = constants.uint32,
) -> None:
    """Write a length-prefixed byte string."""
    destination.write(struct_type.pack(len(byte_string)))
    destination.write(byte_string)


def write_acquisition(destination: Any, acquisition: ismrmrd.Acquisition) -> None:
    destination.write(
        constants.GadgetMessageIdentifier.pack(
            constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION
        )
    )
    acquisition.serialize_into(destination.write)


def write_waveform(destination: Any, waveform: ismrmrd.Waveform) -> None:
    destination.write(
        constants.GadgetMessageIdentifier.pack(
            constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM
        )
    )
    waveform.serialize_into(destination.write)


def write_image(destination: Any, image: ismrmrd.Image) -> None:
    destination.write(
        constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_ISMRMRD_IMAGE)
    )
    image.serialize_into(destination.write)


def write_dicom(destination: Any, dset_with_filename: DicomWithName) -> None:
    """Write a named DICOM message, laid out as :func:`.readers.read_dicom` reads it.

    Writes nothing when the dataset is ``None``.
    """
    message_id_bytes = constants.GadgetMessageIdentifier.pack(
        constants.GADGET_MESSAGE_DICOM_WITHNAME
    )
    dset = dset_with_filename.dset
    filename = dset_with_filename.filename
    if dset is None:
        logging.info("No DICOM dataset to send - skipping")
        return

    logging.info("-------------------------------------------------")
    logging.info(f"        sending DICOM {filename}          ")
    logging.info("-------------------------------------------------")

    filename_encoded = filename.encode("utf-8")
    buf = io.BytesIO()
    dset.save_as(buf, enforce_file_format=True)
    dicom_bytes = buf.getvalue()

    filename_length = len(filename_encoded)
    bytes_to_read = filename_length + len(dicom_bytes) + 4
    header = struct.pack("<I", bytes_to_read) + struct.pack("<I", filename_length)

    destination.write(message_id_bytes)
    destination.write(header + filename_encoded + dicom_bytes)


def write_text(destination: Any, contents: str) -> None:
    """Write a text message; a NUL is appended and counted in the length."""
    logging.info("--> Sending GADGET_MESSAGE_TEXT")
    logging.info("    %s", contents)
    destination.write(
        constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_TEXT)
    )
    contents_with_nul = f"{contents}\0"
    destination.write(constants.uint32.pack(len(contents_with_nul.encode())))
    destination.write(contents_with_nul.encode())


def write_config_text(destination: Any, contents: str) -> None:
    """Write a config text message; a NUL is appended and counted in the length."""
    destination.write(
        constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CONFIG)
    )
    contents_with_nul = f"{contents}\0".encode()
    destination.write(constants.uint32.pack(len(contents_with_nul)))
    destination.write(contents_with_nul)


def header_document(header: Any) -> str | bytes:
    """Return an MRD header as its XML document; text is returned as it stands."""
    return header if isinstance(header, (str, bytes)) else header.toXML("utf-8")


def write_header(destination: Any, header: Any) -> None:
    """Write an MRD XML header message, from an ``ismrmrd.xsd`` header or its document."""
    destination.write(
        constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_HEADER)
    )
    document = header_document(header)
    write_byte_string(
        destination, document.encode() if isinstance(document, str) else document
    )
