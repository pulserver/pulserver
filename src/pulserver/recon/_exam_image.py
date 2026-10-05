"""A map one series of an exam measures, with the geometry a later series resamples it from."""

from __future__ import annotations

__all__ = ["ExamImage"]

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._buffers import ReconData
from .plugin import ReconContext

#: The fields of an MRD ``ImageHeader`` that place an image in the scanner.
_GEOMETRY = (
    "position",
    "read_dir",
    "phase_dir",
    "slice_dir",
    "field_of_view",
    "matrix_size",
)


@dataclass(frozen=True, eq=False)
class ExamImage:
    """An image measured by one series and read by later series of the exam.

    A calibration series stores a map with the geometry it was measured on,
    as ``context.b1_map = ExamImage.measured(b1, context, data)``, and a
    later series reads it resampled onto its own grid with
    ``context.b1_map.on(context, data)``. The resampling uses the two
    geometries alone, no registration: :func:`bartorch.tools.reslice`.

    Attributes
    ----------
    data : ndarray
        ``(..., z, y, x)``: ``x`` the readout, ``y`` the phase encode, ``z``
        the partition or slice.
    header : str
        The MRD XML header of the series that measured it; empty offline.
    geometry : dict or tuple of dict
        Where ``data`` lies, as the fields of an MRD ``ImageHeader``:
        ``position`` (mm, the centre of voxel ``n // 2`` along each axis),
        ``read_dir``, ``phase_dir``, ``slice_dir``, ``field_of_view`` (mm,
        read, phase, slice) and ``matrix_size`` (read, phase, slice). One for
        the volume, or a tuple of one per plane along ``z`` for a stack of
        slices.
    """

    data: np.ndarray
    header: str
    geometry: dict[str, tuple] | tuple[dict[str, tuple], ...]

    @classmethod
    def measured(cls, data: Any, context: ReconContext, unit: ReconData) -> ExamImage:
        """Return ``data`` with the geometry of the image ``unit`` reconstructs.

        ``data`` lies on the unit's reconstruction grid, ``(..., [z,] y, x)``,
        as a :class:`~pulserver.recon.ReconResult` of the unit would; its
        geometry is the one the result's image would carry.
        """
        values = np.asarray(data.detach().cpu() if hasattr(data, "detach") else data)
        if values.ndim == 2:
            values = values[None]
        return cls(values, _xml(context.header), _unit_geometry(context, unit))

    def on(self, context: ReconContext, unit: ReconData, **kwargs: Any) -> np.ndarray:
        """Return the image resampled onto the reconstruction grid of ``unit``.

        ``(..., z, y, x)`` with the unit's matrix, ``z`` one for a plane.
        Keyword arguments go to :func:`bartorch.tools.reslice`: ``interpolation``
        and ``fill``. bartorch is imported here, and SimpleITK by it.
        """
        from bartorch.tools import reslice

        return reslice(
            self.data, self.geometry, _unit_geometry(context, unit), **kwargs
        )


def _xml(header: Any) -> str:
    """Return an MRD header serialized; empty for anything else, such as an offline header."""
    import ismrmrd

    if isinstance(header, ismrmrd.xsd.ismrmrdHeader):
        return ismrmrd.xsd.ToXML(header)
    return header if isinstance(header, str) else ""


def _unit_geometry(context: ReconContext, unit: ReconData) -> dict[str, tuple]:
    """Return the geometry of the image a unit reconstructs, as its images carry it."""
    buffer = unit.data if unit.data is not None else unit.ref
    if buffer is None:
        raise ValueError("the unit holds no k-space, so its geometry is not known")
    reference = buffer.reference
    encoding = context.header.encoding[buffer.space.index]
    region = getattr(encoding, "reconSpace", None) or encoding.encodedSpace
    fov, matrix = region.fieldOfView_mm, region.matrixSize
    geometry = {
        name: tuple(float(v) for v in getattr(reference, name))
        for name in _GEOMETRY[:4]
    }
    geometry["field_of_view"] = (float(fov.x), float(fov.y), float(fov.z))
    geometry["matrix_size"] = (int(matrix.x), int(matrix.y), int(matrix.z))
    return geometry
