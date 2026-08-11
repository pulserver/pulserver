"""Private accessors for Gadgetron-style MRD metadata.

The helpers deliberately use duck typing so handlers can use ISMRMRD objects,
test doubles, and objects supplied by upstream Gadgetron bindings alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "MrdMetadata",
    "acquisition_label",
    "acquisition_labels",
    "has_acquisition_flag",
    "user_parameter",
]


def user_parameter(metadata: Any, name: str, default: Any = None) -> Any:
    """Return a typed MRD user parameter by name, or ``default`` if absent.

    All standard MRD parameter collections (long, double, string, and base64)
    are searched in order.  The value is returned unmodified by the XML
    binding, except that a missing parameter yields ``default``.
    """
    parameters = getattr(metadata, "userParameters", None)
    if parameters is None:
        return default
    for collection_name in (
        "userParameterLong",
        "userParameterDouble",
        "userParameterString",
        "userParameterBase64",
    ):
        for parameter in getattr(parameters, collection_name, ()) or ():
            if getattr(parameter, "name", None) == name:
                return getattr(parameter, "value", default)
    return default


def acquisition_label(acquisition: Any, name: str, default: Any = None) -> Any:
    """Return one acquisition index/header label by its MRD field name.

    Index labels (for example ``slice``, ``repetition``,
    ``kspace_encode_step_1``) are read from ``acquisition.idx``.  Header
    labels such as ``encoding_space_ref`` are read directly from the
    acquisition object.
    """
    index = getattr(acquisition, "idx", None)
    if index is not None and hasattr(index, name):
        return getattr(index, name)
    return getattr(acquisition, name, default)


def acquisition_labels(acquisition: Any) -> dict[str, Any]:
    """Return the standard MRD index and encoding labels as a dictionary."""
    names = (
        "encoding_space_ref",
        "kspace_encode_step_1",
        "kspace_encode_step_2",
        "average",
        "slice",
        "contrast",
        "phase",
        "repetition",
        "set",
        "segment",
    )
    return {name: acquisition_label(acquisition, name) for name in names}


def has_acquisition_flag(acquisition: Any, flag: int | str) -> bool:
    """Return whether an MRD acquisition has a numeric or named flag set.

    Named flags use the ``ismrmrd.ACQ_*`` constant names, for example
    ``"ACQ_LAST_IN_MEASUREMENT"``.  This helper does not import ISMRMRD unless
    a named flag is requested.
    """
    if isinstance(flag, str):
        try:
            import ismrmrd
        except ImportError as error:
            raise ImportError("Named acquisition flags require ismrmrd.") from error
        try:
            flag = getattr(ismrmrd, flag)
        except AttributeError as error:
            raise ValueError(f"Unknown ISMRMRD acquisition flag {flag!r}") from error
    is_set = getattr(acquisition, "is_flag_set", None)
    if callable(is_set):
        return bool(is_set(flag))
    return bool(getattr(acquisition, "flags", 0) & flag)


@dataclass(frozen=True)
class MrdMetadata:
    """Gadgetron-style convenience view over one parsed MRD XML header.

    Parameters
    ----------
    header : object
        Parsed ISMRMRD ``ismrmrdHeader`` or an API-compatible object.
    """

    header: Any

    def encoding(self, index: int = 0) -> Any:
        """Return encoding ``index`` or raise ``IndexError`` if it is absent."""
        return self.header.encoding[index]

    def encoded_matrix(self, index: int = 0) -> tuple[int, int, int]:
        """Return the encoded matrix as ``(x, y, z)``."""
        matrix = self.encoding(index).encodedSpace.matrixSize
        return int(matrix.x), int(matrix.y), int(matrix.z)

    def recon_matrix(self, index: int = 0) -> tuple[int, int, int]:
        """Return the reconstruction matrix as ``(x, y, z)``."""
        matrix = self.encoding(index).reconSpace.matrixSize
        return int(matrix.x), int(matrix.y), int(matrix.z)

    def field_of_view_mm(self, index: int = 0) -> tuple[float, float, float]:
        """Return reconstruction field of view in millimetres as ``(x, y, z)``."""
        fov = self.encoding(index).reconSpace.fieldOfView_mm
        return float(fov.x), float(fov.y), float(fov.z)

    def user_parameter(self, name: str, default: Any = None) -> Any:
        """Return an MRD user parameter by name."""
        return user_parameter(self.header, name, default)
