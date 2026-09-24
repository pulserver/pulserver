"""Readers for MRD streaming messages; each takes a source with ``read(nbytes)``.

Adapted from the readers of
gadgetron-python (Copyright (c) 2019 Gadgetron; MIT, see ``LICENSES/gadgetron-python-MIT.txt``).
"""

__all__ = [
    "deserialize_config",
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
import struct
import xml.etree.ElementTree as xml
from collections.abc import Callable
from typing import Any
from xml.parsers import expat

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


def read(source: Any, struct_type: struct.Struct) -> Any:
    """Read one value of a ``struct.Struct`` format."""
    return struct_type.unpack(source.read(struct_type.size))[0]


def read_optional(
    source: Any, continuation: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any | None:
    """Read a presence flag, then the value with ``continuation``; ``None`` when absent."""
    is_present = read(source, constants.bool)
    return continuation(source, *args, **kwargs) if is_present else None


def read_vector(source: Any, numpy_type: type = np.uint64) -> np.ndarray:
    """Read a ``uint64`` count, then that many values of ``numpy_type``."""
    size = read(source, constants.uint64)
    dtype = np.dtype(numpy_type)
    return np.frombuffer(source.read(size * dtype.itemsize), dtype=dtype)


def read_array(source: Any, numpy_type: type = np.uint64) -> np.ndarray:
    """Read a dimension vector, then the values in Fortran order."""
    dtype = np.dtype(numpy_type)
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(
        np.frombuffer(source.read(elements * dtype.itemsize), dtype=dtype),
        dimensions,
        order="F",
    )


def read_object_array(source: Any, read_object: Callable[[Any], Any]) -> np.ndarray:
    """Read a dimension vector, then one object per element in Fortran order."""
    dimensions = read_vector(source)
    elements = int(functools.reduce(lambda a, b: a * b, dimensions))
    return np.reshape(
        np.asarray([read_object(source) for _ in range(elements)], dtype=object),
        dimensions,
        order="F",
    )


def read_image_header(source: Any) -> ismrmrd.ImageHeader:
    header_bytes = source.read(ctypes.sizeof(ismrmrd.ImageHeader))
    return ismrmrd.ImageHeader.from_buffer_copy(header_bytes)


def read_acquisition_header(source: Any) -> ismrmrd.AcquisitionHeader:
    header_bytes = source.read(ctypes.sizeof(ismrmrd.AcquisitionHeader))
    return ismrmrd.AcquisitionHeader.from_buffer_copy(header_bytes)


def read_waveform_header(source: Any) -> ismrmrd.Waveform:
    header_bytes = source.read(ctypes.sizeof(ismrmrd.WaveformHeader))
    return ismrmrd.Waveform.from_buffer_copy(header_bytes)


def read_gadget_message_length(
    source: Any, struct_type: struct.Struct = constants.uint32
) -> int:
    return read(source, struct_type)


def read_byte_string(
    source: Any, struct_type: struct.Struct = constants.uint32
) -> bytes:
    """Read a length-prefixed byte string."""
    length = read_gadget_message_length(source, struct_type)
    return source.read(length)


def read_acquisition(source: Any) -> ismrmrd.Acquisition:
    return ismrmrd.Acquisition.deserialize_from(source.read)


def read_waveform(source: Any) -> ismrmrd.Waveform:
    return ismrmrd.Waveform.deserialize_from(source.read)


def read_image(source: Any) -> ismrmrd.Image:
    return ismrmrd.Image.deserialize_from(source.read)


def read_dicom(source: Any) -> DicomWithName:
    """Read a named DICOM message.

    Layout: ``uint32`` payload length (name length + 4 + DICOM bytes),
    ``uint32`` name length, the UTF-8 file name, then the DICOM file bytes.
    """
    bytes_to_read = read(source, constants.uint32)
    filename_length = read(source, constants.uint32)
    filename = source.read(filename_length).decode("utf-8")
    dicom_bytes = source.read(bytes_to_read - filename_length - 4)
    dset = pydicom.dcmread(io.BytesIO(dicom_bytes))

    logging.info("-------------------------------------------------")
    logging.info(f"        received DICOM {filename}        ")
    logging.info("-------------------------------------------------")

    return DicomWithName(dset=dset, filename=filename)


def read_text(source: Any) -> str:
    """Read a length-prefixed text message, truncated at its first NUL."""
    length = read(source, constants.uint32)
    text_bytes = source.read(length)
    text = text_bytes.split(b"\x00", 1)[0].decode("utf-8")
    logging.info("    %s", text)
    return text


def read_config_text(source: Any) -> Any:
    """Read a length-prefixed config text and parse it.

    Parsers are tried in order JSON, YAML, XML, each only when installed; the
    first that reads the text as a mapping wins, and a Gadgetron ``RECON``
    mapping is translated by :func:`_gadgetron2mrd`. Text no parser reads as a
    mapping, empty text included, yields ``{"parameters": {"config":
    "default"}}``.
    """
    length = read(source, constants.uint32)
    content = source.read(length).decode("utf-8").rstrip("\x00")
    return deserialize_config(content, "default")


def read_config_file(source: Any) -> str:
    """Read a config file name: a fixed 1024-byte NUL-padded field, not length-prefixed."""
    config_file_bytes = read(source, constants.GadgetMessageConfigurationFile)
    return config_file_bytes.decode("utf-8").rstrip("\x00")


def read_header(source: Any) -> Any:
    """Read a length-prefixed MRD XML header and parse it with ``ismrmrd.xsd``."""
    return ismrmrd.xsd.CreateFromDocument(read_byte_string(source))


def _auto_cast_str(val):
    with contextlib.suppress(Exception):
        val = ast.literal_eval(val)
    return val


def _xml_postprocessor(_path, key, value):
    # XML booleans are lower case; literal_eval needs Python's spelling.
    if value == "true":
        value = "True"
    if value == "false":
        value = "False"
    return key, _auto_cast_str(value)


def _gadgetron2mrd(config: Any) -> Any:
    """Map ``{"RECON": {"cmd": c, ...}}`` to ``{"parameters": {"config": c, ...}}``; other input unchanged."""
    if isinstance(config, dict) and "RECON" in config:
        cmd = config["RECON"].pop("cmd")
        return {"parameters": {"config": cmd, **config["RECON"]}}
    return config


def deserialize_config(content: str, default_config: str = "default") -> Any:
    """Parse a config text as :func:`read_config_text` documents it."""
    try:
        config_dict = json.loads(content)
        if isinstance(config_dict, dict):
            logging.debug("Parsed config as JSON")
            return _gadgetron2mrd(config_dict)
    except json.JSONDecodeError:
        pass

    if HAS_YAML:
        try:
            config_dict = yaml.safe_load(content)
            if isinstance(config_dict, dict):
                logging.debug("Parsed config as YAML")
                return _gadgetron2mrd(config_dict)
        except yaml.YAMLError:
            pass
    else:
        logging.debug("YAML not available, skipping")

    if HAS_XMLTODICT:
        try:
            config_dict = xmltodict.parse(content, postprocessor=_xml_postprocessor)
            if isinstance(config_dict, dict):
                logging.debug("Parsed config as XML")
                return _gadgetron2mrd(config_dict)
        except (xml.ParseError, expat.ExpatError):
            pass

    logging.warning(
        "Failed to parse config as JSON, YAML, or XML. Using default config."
    )
    return {"parameters": {"config": default_config}}
