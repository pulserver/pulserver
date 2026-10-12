"""Duck-typed accessors for MRD headers and acquisitions.

They accept ``ismrmrd`` objects and any object with the same attribute names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._labels import MRD_FLAGS, canonical_label

__all__ = [
    "DIFFUSION_TABLE",
    "MrdMetadata",
    "acquisition_label",
    "acquisition_labels",
    "diffusion_table",
    "has_acquisition_flag",
    "max_stored_value",
    "user_parameter",
]


def user_parameter(metadata: Any, name: str, default: Any = None) -> Any:
    """Return an MRD header user parameter by name, or ``default`` when absent.

    The long, double, string and base64 collections are searched in that order;
    the value is returned as the XML binding typed it.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import pulserver.mrd as mrd
    >>> metadata = SimpleNamespace(
    ...     userParameters=SimpleNamespace(
    ...         userParameterLong=[SimpleNamespace(name="EchoTrainLength", value=8)],
    ...         userParameterDouble=[],
    ...         userParameterString=[],
    ...     )
    ... )
    >>> mrd.user_parameter(metadata, "EchoTrainLength")
    8
    >>> mrd.user_parameter(metadata, "NotThere", 0)
    0
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


#: User parameter string an MRD header states a diffusion table in: one
#: ``x y z b`` row per line.
DIFFUSION_TABLE = "diffusion_table"


def diffusion_table(metadata: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the diffusion table an MRD header states, or None when it states none.

    Row ``i`` is the volume whose ``repetition`` counter is ``i``.

    Returns
    -------
    b : numpy.ndarray
        ``(volumes,)``, s/mm².
    g : numpy.ndarray
        ``(volumes, 3)``, unit directions along the logical readout, phase and
        slice axes, which an acquisition's ``read_dir``, ``phase_dir`` and
        ``slice_dir`` state in the patient frame; zero where ``b`` is zero.
    """
    text = user_parameter(metadata, DIFFUSION_TABLE)
    if not text:
        return None
    rows = np.loadtxt(str(text).splitlines(), dtype=float, ndmin=2)
    return rows[:, 3], rows[:, :3]


def max_stored_value(metadata: Any) -> int:
    """Return the largest pixel value of ``BitsStored`` bits, 12 bits when unstated."""
    return 2 ** int(user_parameter(metadata, "BitsStored") or 12) - 1


def acquisition_label(acquisition: Any, name: str, default: Any = None) -> Any:
    """Return one acquisition label by MRD field name.

    Index counters (``slice``, ``kspace_encode_step_1``) are read from
    ``acquisition.idx``; any other name from the acquisition itself.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import pulserver.mrd as mrd
    >>> acquisition = SimpleNamespace(
    ...     idx=SimpleNamespace(slice=2), encoding_space_ref=0
    ... )
    >>> mrd.acquisition_label(acquisition, "slice")
    2
    >>> mrd.acquisition_label(acquisition, "encoding_space_ref")
    0
    >>> mrd.acquisition_label(acquisition, "repetition", 0)
    0
    """
    index = getattr(acquisition, "idx", None)
    if index is not None and hasattr(index, name):
        return getattr(index, name)
    return getattr(acquisition, name, default)


def acquisition_labels(acquisition: Any) -> dict[str, Any]:
    """Return ``encoding_space_ref`` and the MRD index counters; absent ones are ``None``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import pulserver.mrd as mrd
    >>> acquisition = SimpleNamespace(idx=SimpleNamespace(slice=2, repetition=1))
    >>> labels = mrd.acquisition_labels(acquisition)
    >>> labels["slice"], labels["repetition"]
    (2, 1)
    """
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
    """Return whether an acquisition carries one flag.

    Parameters
    ----------
    acquisition
        ``ismrmrd.Acquisition``, or any object with ``is_flag_set`` or ``flags``.
    flag
        A single :class:`~pulserver.mrd.AcquisitionFlag`, a 1-based bit position
        as the ``ismrmrd.ACQ_*`` constants are, a constant name
        (``"ACQ_LAST_IN_MEASUREMENT"``), or a flag label (``"LASTSCAN"``).

    Raises
    ------
    ValueError
        If a name matches no ISMRMRD flag.

    Examples
    --------
    >>> import ismrmrd
    >>> import pulserver.mrd as mrd
    >>> acquisition = ismrmrd.Acquisition()
    >>> acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    >>> mrd.has_acquisition_flag(acquisition, mrd.AcquisitionFlag.LAST_IN_SLICE)
    True
    >>> mrd.has_acquisition_flag(acquisition, "LASTSLC")
    True
    >>> mrd.has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT")
    False
    """
    name = getattr(flag, "flag", None)
    if name is not None and not isinstance(flag, (str, int)):
        flag = name
    if isinstance(flag, str):
        try:
            import ismrmrd
        except ImportError as error:
            raise ImportError("Named acquisition flags require ismrmrd.") from error
        try:
            flag = getattr(ismrmrd, MRD_FLAGS.get(canonical_label(flag), flag))
        except AttributeError as error:
            raise ValueError(f"Unknown ISMRMRD acquisition flag {flag!r}") from error
    is_set = getattr(acquisition, "is_flag_set", None)
    if callable(is_set):
        return bool(is_set(flag))
    return bool(getattr(acquisition, "flags", 0) & (1 << (flag - 1)))


@dataclass(frozen=True)
class MrdMetadata:
    """Accessors over one parsed MRD XML header.

    Parameters
    ----------
    header
        Parsed ``ismrmrd.xsd`` header, or an object with the same attributes.
    """

    header: Any

    def encoding(self, index: int = 0) -> Any:
        """Return encoding space ``index``."""
        return self.header.encoding[index]

    def encoded_matrix(self, index: int = 0) -> tuple[int, int, int]:
        """Return the encoded matrix size as ``(x, y, z)``."""
        matrix = self.encoding(index).encodedSpace.matrixSize
        return int(matrix.x), int(matrix.y), int(matrix.z)

    def recon_matrix(self, index: int = 0) -> tuple[int, int, int]:
        """Return the reconstruction matrix size as ``(x, y, z)``."""
        matrix = self.encoding(index).reconSpace.matrixSize
        return int(matrix.x), int(matrix.y), int(matrix.z)

    def field_of_view_mm(self, index: int = 0) -> tuple[float, float, float]:
        """Return the reconstruction field of view as ``(x, y, z)``, in mm."""
        fov = self.encoding(index).reconSpace.fieldOfView_mm
        return float(fov.x), float(fov.y), float(fov.z)

    def user_parameter(self, name: str, default: Any = None) -> Any:
        """Return a user parameter of the header; see :func:`user_parameter`."""
        return user_parameter(self.header, name, default)
