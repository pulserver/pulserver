"""Private accessors for Gadgetron-style MRD metadata.

The helpers deliberately use duck typing so handlers can use ISMRMRD objects,
test doubles, and objects supplied by upstream Gadgetron bindings alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._labels import MRD_FLAGS, canonical_label

__all__ = [
    "MrdMetadata",
    "acquisition_label",
    "acquisition_labels",
    "diffusion_table",
    "has_acquisition_flag",
    "max_stored_value",
    "user_parameter",
]

#: The header entries a Pulserver sequence carries its diffusion encoding in.
#: Named on this side too, and only here, so the reader and
#: ``Sequence.DIFFUSION_DEFINITIONS`` cannot drift apart silently.
DIFFUSION_PARAMETERS = (
    "bTensorFixed",
    "bTensorRotatable",
    "bTensorCross",
    "bTensorAxis",
)


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


def max_stored_value(metadata: Any) -> int:
    """Largest pixel value the scan's stored bit depth can hold."""
    return 2 ** int(user_parameter(metadata, "BitsStored") or 12) - 1


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


def diffusion_table(metadata: Any, *, rotation: Any = None) -> Any:
    """The scan's diffusion gradient table, or ``None`` if it carries none.

    Reads the ``bTensor*`` user parameters the scanner copied out of the
    sequence's ``[DEFINITIONS]`` (``mrdserver::add_diffusion_parameters``) and
    returns a :class:`~pulserver.pypulseq.DiffusionTable` -- the same type the
    design side produces, so a table recovered from a scan and one computed
    from the script that made it are directly comparable.

    Parameters
    ----------
    metadata : object
        A parsed ISMRMRD header, or anything with a ``userParameters``.
    rotation : array_like, optional
        The prescription as a ``(3, 3)`` matrix taking the sequence's logical
        axes to physical ones -- in MRD terms
        ``numpy.stack((read_dir, phase_dir, slice_dir))`` from any acquisition
        of the scan. The b-tensor is carried in three parts because the
        console's FOV rotation is not in the ``.seq``, and this is what puts
        them together. Leaving it out is right only when the diffusion
        preparation was played entirely under ``NOROT``, which is the usual
        case and the one the design should aim for.

    Returns
    -------
    DiffusionTable or None

    Examples
    --------
    Feeding DIPY, which wants the unnormalised tensor whose trace is the
    b-value::

        table = diffusion_table(header)
        gtab = gradient_table(table.b_values, table.b_vectors,
                              btens=table.b_tensors)

    and MRtrix3, whose ``-grad`` table is ``[x y z b]``::

        numpy.savetxt("grad.b", table.mrtrix_table())

    ``table.axis`` names the MRD counter whose value indexes a row, so an
    acquisition's encoding is ``table.b_tensors[acq.idx.set]`` when it is
    ``"SET"``.
    """
    from pulserver.pypulseq import DiffusionTable

    found = {}
    for name in DIFFUSION_PARAMETERS:
        value = user_parameter(metadata, name)
        if value is not None:
            found[name] = value
    if "bTensorFixed" not in found:
        return None
    return DiffusionTable.from_definitions(found, rotation=rotation)


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

    The flag may be an :class:`~pulserver.AcquisitionFlag`, or a name written
    either way round: the ``ismrmrd.ACQ_*`` constant, as in
    ``"ACQ_LAST_IN_MEASUREMENT"``, or the name the sequence set the label
    under, as in ``"LASTSCAN"``.  All reach the same bit, so a plugin can say
    what it is waiting for in the same words the sequence used.  This helper
    does not import ISMRMRD unless a named flag is requested.
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
