"""
Readers for ISMRMRD data types.

This module provides functions to deserialize binary ISMRMRD data from a source
(e.g., a socket or file-like object) into usable Python objects like acquisitions,
images, and waveforms.
"""

__all__ = [
    "read",
    "read_acquisition",
    "read_acquisition_header",
    "read_array",
    "read_byte_string",
    "read_config_file",
    "read_config_text",
    "read_dicom",
    "read_gadget_message_length",
    "read_header",
    "read_image",
    "read_image_header",
    "read_object_array",
    "read_optional",
    "read_text",
    "read_vector",
    "read_waveform",
    "read_waveform_header",
]

import ast
import contextlib
import ctypes
import functools
import io
import json
import logging
import xml.etree.ElementTree as xml
from collections.abc import Callable
from typing import Any

import ismrmrd
import numpy as np
import pydicom

from . import constants
from .mrd2dicom import DicomWithName

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    import xmltodict

    HAS_XMLTODICT = True
except ImportError:
    HAS_XMLTODICT = False
    xmltodict = None


def read(source: Any, type: type) -> Any:
    """
    Read and unpack a single value from the source.

    Parameters
    ----------
    source : Any
        The source to read from (e.g., a socket wrapper).
    type : type
        The struct type to unpack (e.g., constants.uint32).

    Returns
    -------
    Any
        The unpacked value.
    """
    return type.unpack(source.read(type.size))[0]


def read_optional(
    source: Any, continuation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any | None:
    """
    Read an optional value from the source.

    Parameters
    ----------
    source : Any
        The source to read from.
    continuation : Callable
        Function to call if the value is present.
    *args : Any
        Additional arguments for continuation.
    **kwargs : Any
        Additional keyword arguments for continuation.

    Returns
    -------
    Any or None
        The result of continuation if present, else None.
    """
    is_present = read(source, constants.bool)
    return continuation(source, *args, **kwargs) if is_present else None


def read_vector(source: Any, numpy_type: type = np.uint64) -> np.ndarray:
    """
    Read a vector of values from the source.

    Parameters
    ----------
    source : Any
        The source to read from.
    numpy_type : type, optional
        The numpy data type (default: np.uint64).

    Returns
    -------
    np.ndarray
        The read vector.
    """
    size = read(source, constants.uint64)
    dtype = np.dtype(numpy_type)
    return np.frombuffer(source.read(size * dtype.itemsize), dtype=dtype)


def read_array(source: Any, numpy_type: type = np.uint64) -> np.ndarray:
    """
    Read a multi-dimensional array from the source.

    Parameters
    ----------
    source : Any
        The source to read from.
    numpy_type : type, optional
        The numpy data type (default: np.uint64).

    Returns
    -------
    np.ndarray
        The read array, reshaped in Fortran order.
    """
    dtype = np.dtype(numpy_type)
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(
        np.frombuffer(source.read(elements * dtype.itemsize), dtype=dtype),
        dimensions,
        order="F",
    )


def read_object_array(source: Any, read_object: Callable[[Any], Any]) -> np.ndarray:
    """
    Read an array of objects from the source.

    Parameters
    ----------
    source : Any
        The source to read from.
    read_object : Callable
        Function to read each object.

    Returns
    -------
    np.ndarray
        The read object array, reshaped in Fortran order.
    """
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(
        np.asarray([read_object(source) for _ in range(elements)], dtype=object),
        dimensions,
        order="F",
    )


def read_image_header(source: Any) -> ismrmrd.ImageHeader:
    """
    Read an image header from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.ImageHeader
        The deserialized image header.
    """
    header_bytes = source.read(ctypes.sizeof(ismrmrd.ImageHeader))
    return ismrmrd.ImageHeader.from_buffer_copy(header_bytes)


def read_acquisition_header(source: Any) -> ismrmrd.AcquisitionHeader:
    """
    Read an acquisition header from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.AcquisitionHeader
        The deserialized acquisition header.
    """
    header_bytes = source.read(ctypes.sizeof(ismrmrd.AcquisitionHeader))
    return ismrmrd.AcquisitionHeader.from_buffer_copy(header_bytes)


def read_waveform_header(source: Any) -> ismrmrd.Waveform:
    """
    Read a waveform header from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.Waveform
        The deserialized waveform header.
    """
    header_bytes = source.read(ctypes.sizeof(ismrmrd.WaveformHeader))
    return ismrmrd.Waveform.from_buffer_copy(header_bytes)


def read_gadget_message_length(source: Any, type: type = constants.uint32) -> int:
    """
    Read the length of a Gadget message.

    Parameters
    ----------
    source : Any
        The source to read from.
    type : type, optional
        The type for the length (default: constants.uint32).

    Returns
    -------
    int
        The message length.
    """
    return read(source, type)


def read_byte_string(source: Any, type: type = constants.uint32) -> bytes:
    """
    Read a byte string from the source.

    Parameters
    ----------
    source : Any
        The source to read from.
    type : type, optional
        The type for the length (default: constants.uint32).

    Returns
    -------
    bytes
        The read byte string.
    """
    length = read_gadget_message_length(source, type)
    byte_string = source.read(length)
    return byte_string


def read_acquisition(source: Any) -> ismrmrd.Acquisition:
    """
    Read an acquisition from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.Acquisition
        The deserialized acquisition.
    """
    return ismrmrd.Acquisition.deserialize_from(source.read)


def read_waveform(source: Any) -> ismrmrd.Waveform:
    """
    Read a waveform from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.Waveform
        The deserialized waveform.
    """
    return ismrmrd.Waveform.deserialize_from(source.read)


def read_image(source: Any) -> ismrmrd.Image:
    """
    Read an image from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    ismrmrd.Image
        The deserialized image.
    """
    return ismrmrd.Image.deserialize_from(source.read)


def read_dicom(source: Any) -> DicomWithName:
    """
    Read DICOM data with filename from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    DicomWithName
        A DicomWithName object with two attributes:
            dset : pydicom.Dataset
                The DICOM dataset read.
            filenames : str
                The filename.
    """
    # Read header
    bytes_to_read = read(source, constants.uint32)
    filename_length = read(source, constants.uint32)

    # Read filename
    filename_encoded = source.read(filename_length)
    filename = filename_encoded.decode("utf-8")

    # Read DICOM bytes
    dicom_bytes = source.read(bytes_to_read - filename_length - 4)
    dset = pydicom.dcmread(io.BytesIO(dicom_bytes))

    logging.info("-------------------------------------------------")
    logging.info(f"        received DICOM {filename}        ")
    logging.info("-------------------------------------------------")

    return DicomWithName(dset=dset, filename=filename)


def read_text(source: Any) -> str:
    """
    Read arbitrary text from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    str
        The decoded text.
    """
    length = read(source, constants.uint32)
    text_bytes = source.read(length)
    text = text_bytes.split(b"\x00", 1)[0].decode("utf-8")  # Strip null terminator
    logging.info("    %s", text)
    return text


def read_config_text(source: Any) -> dict:
    """
    Read and deserialize config text from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    dict
        The deserialized config dict.
    """
    length = read(source, constants.uint32)
    config_bytes = source.read(length)
    content = config_bytes.decode("utf-8").rstrip("\x00")
    config_dict = _deserialize_config(content, "default")
    # Note: raw_bytes.config would be handled in Connection if needed
    return config_dict


def read_config_file(source: Any) -> str:
    """
    Read the Gadgetron configuration filename from the source.

    The wire format is a fixed 1024-byte ``GadgetMessageConfigurationFile``
    struct (a null-padded char array), NOT a length-prefixed string.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    str
        The configuration filename / handler module name.
    """
    config_file_bytes = read(source, constants.GadgetMessageConfigurationFile)
    filename = config_file_bytes.decode("utf-8").rstrip("\x00")
    return filename


def read_header(source: Any) -> Any:
    """
    Read and deserialize ISMRMRD header from the source.

    Parameters
    ----------
    source : Any
        The source to read from.

    Returns
    -------
    Any
        The deserialized header (e.g., ismrmrd.xsd object).
    """
    header_bytes = read_byte_string(source)
    item = ismrmrd.xsd.CreateFromDocument(header_bytes)
    # Note: saver.save would be handled in Connection if needed
    return item


def _auto_cast_str(val):
    # Try fails if cannot eval, therefore is string
    with contextlib.suppress(Exception):
        val = ast.literal_eval(val)
    return val


def _xml_postprocessor(path, key, value):
    # XML standard requires lower case bools
    if value == "true":
        value = "True"
    if value == "false":
        value = "False"
    return key, _auto_cast_str(value)


def _gadgetron2mrd(input: dict) -> dict:
    """Transform config dict to MRD server format (copied from server.py)."""
    if "RECON" in input:
        cmd = input["RECON"].pop("cmd")
        return {"parameters": {"config": cmd, **input["RECON"]}}
    return input


def _deserialize_config(content: str, default_config: str = "default") -> dict:
    """
    Deserialize config content into a dict, trying JSON, TOML, YAML, XML in order.

    Parameters
    ----------
    content : str
        The config content as a string.
    default_config : str
        Default config name if deserialization fails.

    Returns
    -------
    dict
        The deserialized config dict.
    """
    # Try JSON
    try:
        config_dict = json.loads(content)
        logging.debug("Parsed config as JSON")
        return _gadgetron2mrd(config_dict)
    except json.JSONDecodeError:
        pass

    # Try YAML (if available)
    if HAS_YAML:
        try:
            config_dict = yaml.safe_load(content)
            logging.debug("Parsed config as YAML")
            return _gadgetron2mrd(config_dict)
        except yaml.YAMLError:
            pass
    else:
        logging.debug("YAML not available, skipping")

    # Try XML (convert to dict)
    if HAS_XMLTODICT:
        try:
            config_dict = xmltodict.parse(content, postprocessor=_xml_postprocessor)
            logging.debug("Parsed config as XML")
            return _gadgetron2mrd(config_dict)
        except xml.ParseError:
            pass

    # Fallback
    logging.warning(
        "Failed to parse config as JSON, TOML, YAML, or XML. Using default config."
    )
    return {"parameters": {"config": default_config}}


def _load_config_from_file(filename: str, default_config: str = "default") -> dict:
    """Load and deserialize config from file."""
    try:
        with open(filename) as f:
            content = f.read()
        return _deserialize_config(content, default_config)
    except (OSError, FileNotFoundError) as e:
        logging.error("Failed to load config file '%s': %s", filename, e)
        return {"parameters": {"config": default_config}}
