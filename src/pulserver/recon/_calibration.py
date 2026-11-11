"""Coil sensitivities of a reconstruction unit, and the compression of its receive channels."""

from __future__ import annotations

__all__ = ["CoilCompression", "CoilSensitivities", "MissingCalibration", "coil_maps"]

import copy
import dataclasses
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._buffers import ReconBuffer, ReconData
from .plugin import COIL_SENSITIVITIES, ReconContext

_log = logging.getLogger("pulserver.recon")

#: The placement axes the calibration k-space of a unit may vary along.
_ENCODED = ("partition", "phase_encode")


def _field_matches(name: str, stored: Any, requested: Any) -> bool:
    continuous = {
        "geometry.table_position",
        "geometry.position",
        "geometry.read_dir",
        "geometry.phase_dir",
        "geometry.slice_dir",
        "geometry.fov",
        "geometry.extent",
    }
    if name in continuous and stored is not None and requested is not None:
        return np.shape(stored) == np.shape(requested) and bool(
            np.allclose(stored, requested, rtol=0, atol=1e-6)
        )
    return stored == requested


class MissingCalibration(RuntimeError):
    """No source holds a calibration the stream can use.

    The message names each source consulted and why it was rejected.
    """


@dataclass(frozen=True, eq=False)
class Whitening:
    """The matrix a :class:`~pulserver.recon.Prewhiten` multiplies the channels of a readout by.

    ``matrix`` ``W``, ``(coils, coils)``, satisfies ``W Ψ W^H = I`` for the
    covariance ``Ψ`` of noise sampled with dwell time ``dwell_us``, in µs;
    ``None`` where the noise readouts stated none. ``coils`` are the receive
    coil labels of the header in channel order, and ``id`` identifies the
    matrix.
    """

    matrix: np.ndarray
    coils: tuple[str, ...]
    dwell_us: float | None
    id: str


@dataclass(frozen=True, eq=False)
class Basis:
    """The basis a :class:`CoilCompression` projects the channels of k-space onto.

    ``matrix`` is the unitary ``(coils, coils)`` matrix ``bartorch.tools.cc``
    returns, of which ``n_virtual`` virtual channels are kept; ``id``
    identifies the basis.
    """

    matrix: np.ndarray
    n_virtual: int
    id: str


@dataclass(frozen=True, eq=False)
class CoilSensitivities:
    """Receive coil sensitivity maps, with what they were estimated from.

    Maps stored in the stream serve only data with the coils, prewhitening,
    compression and geometry of the calibration data they were estimated
    from; :meth:`incompatibility` states the comparison. Maps stored in the
    exam also serve data at another position, orientation or grid, resliced
    onto it, and data compressed by a basis of its own when they were
    estimated from uncompressed channels; :func:`coil_maps` states when.

    Attributes
    ----------
    maps : ndarray
        ``(coils, [z,] y, x)`` complex64, one set of sensitivities on the grid
        of the calibration k-space, as the estimate returned them. The channels
        are those of the data after compression, when it was compressed.
    coils : tuple of str
        Receive coil labels of the header in channel order,
        ``acquisitionSystemInformation.coilLabel`` by ``coilNumber``; the
        channel indices where the header states none.
    noise : str or None
        Identifier of the prewhitening the calibration data went through;
        ``None`` when it was not whitened.
    basis : str or None
        Identifier of the compression basis the calibration data was
        projected onto; ``None`` when it was not compressed.
    geometry : dict
        Where the maps are defined, by component: ``frame_of_reference``
        (header ``measurementInformation.frameOfReferenceUID``);
        ``table_position``, ``position`` in mm, and ``read_dir``,
        ``phase_dir`` and ``slice_dir`` of the calibration k-space's reference
        acquisition; ``fov`` in metres and ``matrix`` of the reconstruction
        space, ordered as ``(z,) y, x``; ``grid``, the shape of the voxel
        axes of ``maps``; and ``extent``, the size of that grid in mm as
        ``(z, y, x)``, ``z`` the slice thickness for a plane. A component the
        header or acquisition does not state is ``None``.
    method : str
        The estimate that produced the maps.
    source : str or None
        ``measurementID`` of the series that measured them.
    """

    maps: np.ndarray
    coils: tuple[str, ...]
    noise: str | None
    basis: str | None
    geometry: dict[str, Any]
    method: str
    source: str | None

    def incompatibility(self, data: ReconData, context: ReconContext) -> str | None:
        """Return why the maps cannot serve a unit, or ``None`` when they can.

        Each field is compared for equality, in this order: ``coils`` with the
        labels of the context's header, ``noise`` with the prewhitening in
        ``context.noise``, ``basis`` with the compression in
        ``context.coil_compression``, then each component of ``geometry`` with
        that of the unit's imaging k-space, or of its calibration k-space where
        it holds no imaging data. Continuous geometry fields admit an absolute
        tolerance of 1e-6 (mm for positions and extents, dimensionless for
        direction cosines); identifiers and matrix sizes must match exactly.

        Returns
        -------
        str or None
            The first field that differs, with the stored value and the
            unit's.
        """
        buffer = data.data if data.data is not None else data.ref
        if buffer is None:
            return "the unit holds no k-space, so its geometry is not known"
        wanted = _geometry(context.header, buffer)
        fields = (
            ("coils", self.coils, coil_labels(context.header)),
            ("noise", self.noise, _id(context.noise)),
            ("basis", self.basis, _id(context.coil_compression)),
            *(
                (f"geometry.{name}", stored, wanted.get(name))
                for name, stored in self.geometry.items()
            ),
        )
        for name, stored, requested in fields:
            if not _field_matches(name, stored, requested):
                return f"{name} differs: stored {stored!r}, this unit {requested!r}"
        return None


def coil_maps(
    context: ReconContext,
    data: ReconData,
    *,
    estimate: Any,
    required: bool = True,
) -> Any:
    """Return the coil sensitivities of a unit from the first source that has them.

    The sources, in order:

    1. The unit's calibration k-space, ``data.ref``: the maps are estimated
       from it and stored in ``context.coil_maps`` under the unit's slice
       counter, replacing the maps stored there.
    2. ``context.coil_maps`` for the unit's slice.
    3. The exam's ``COIL_SENSITIVITIES``, as a calibration series stored them.

    Maps of the stream serve only where
    :meth:`CoilSensitivities.incompatibility` finds nothing. Maps of the exam
    serve a unit with the same coils, prewhitening, frame of reference and
    table position; on another grid, position or orientation they are
    resliced onto the unit's grid from the two geometries,
    :func:`bartorch.tools.reslice`, and refused where the unit's grid reaches
    outside theirs. Maps estimated from uncompressed channels are projected
    onto the unit's compression basis, as its k-space was. The location is the
    slice counter, not the encoding space, so maps estimated from a
    calibration unit in an encoding space of its own serve the imaging units
    of its slice.

    Maps estimated from a unit's calibration readouts are logged at ``INFO``,
    with the lines and the time they took; maps stored in the stream or the
    exam that serve a unit are logged at ``DEBUG``.

    Parameters
    ----------
    context : ReconContext
        Scan context.
    data : ReconData
        The unit that needs maps; ``data.counters["slice"]`` is its slice,
        0 when the unit has no such counter.
    estimate : callable
        ``estimate(kspace) -> maps``, such as :func:`bartorch.apps.nlinv_maps`.
        ``kspace`` is the calibration k-space of the unit as a torch tensor
        ``(coils, [z,] y, x)`` on the host, zero outside the lines it holds, on
        the grid of its encoding space; ``maps`` are the sensitivities on the
        same grid.
    required : bool, default=True
        Return ``None``, rather than raise, when no source has usable maps.

    Returns
    -------
    torch.Tensor or None
        A copy of the maps ``(coils, [z,] y, x)``, complex64, on the host.

    Raises
    ------
    MissingCalibration
        If no source has usable maps and ``required`` is true. The message
        names each source and why it was rejected.
    ValueError
        If the unit's calibration k-space is of a non-Cartesian space, varies
        along an axis other than the partition and phase-encode axes, or the
        estimate returns maps on another grid than it was given.
    """
    import torch

    location = int(data.counters.get("slice", 0))
    found: CoilSensitivities | None = None
    rejected: list[str] = []

    if data.ref is None:
        rejected.append("this unit: it holds no calibration readouts")
        found = _usable(
            context.coil_maps.get(location),
            f"this series, slice {location}",
            data,
            context,
            rejected,
        )
        if found is not None:
            _log.debug(
                "slice %d takes the coil sensitivities its calibration stored (%s)",
                location,
                found.method,
            )
    else:
        start = time.perf_counter()
        found = _estimated(context, data.ref, estimate)
        context.coil_maps[location] = found
        _log.info(
            "coil sensitivities of slice %d estimated by %s from %d calibration "
            "readouts of encoding space %d, maps %s, in %.1f s",
            location,
            found.method,
            int(data.ref.mask.any(axis=-1).sum()),
            data.ref.space.index,
            tuple(found.maps.shape),
            time.perf_counter() - start,
        )
    if found is not None:
        return torch.tensor(found.maps)
    maps = _fitted(context.exam.get(COIL_SENSITIVITIES), data, context, rejected)
    if maps is not None:
        _log.debug("slice %d takes the coil sensitivities the exam stored", location)
        return torch.tensor(maps)

    if not required:
        return None
    raise MissingCalibration(
        f"no coil sensitivities for slice {location}; sources consulted:\n  "
        + "\n  ".join(rejected)
    )


class CoilCompression:
    """Reduce the receive channels of a unit's k-space to virtual channels.

    The basis is estimated once per stream, from the first unit it is given,
    and applied to every unit. It consists of the eigenvectors of the channel
    covariance over every placed sample of the unit's calibration k-space, or
    of its imaging k-space where it holds no calibration data
    (:func:`bartorch.tools.cc`), of which the ``n_virtual`` with the largest
    eigenvalues are kept: the virtual channels are uncorrelated and ordered by
    decreasing power. :func:`bartorch.tools.ccapply` projects the k-space of
    ``data.data`` and ``data.ref`` onto it. The basis is stored in
    ``context.coil_compression``, and :func:`coil_maps` records it with the
    maps it estimates, which only units compressed by the same basis reuse:
    compress a unit before asking for its maps.

    The returned unit holds new buffers with ``n_virtual`` channels, and shares
    its ``counters`` and ``acquisitions`` with the unit it was made from. The
    basis and the projection are computed on the host. bartorch is imported
    when a unit is compressed.

    Corresponds to Gadgetron's ``GenericReconEigenChannelGadget``.

    Parameters
    ----------
    n_virtual : int
        Virtual channels kept, at most the number of channels of the first
        unit.
    """

    def __init__(self, n_virtual: int) -> None:
        if n_virtual < 1:
            raise ValueError(f"n_virtual is {n_virtual}; it has to be at least 1")
        self.n_virtual = int(n_virtual)

    def __call__(self, context: ReconContext, data: ReconData) -> ReconData:
        """Return ``data`` with its k-space projected onto the stream's basis.

        Raises
        ------
        ValueError
            If the stream's basis was made for another ``n_virtual`` or
            another number of channels, or the first unit holds no k-space.
        """
        basis = context.coil_compression
        if basis is None:
            source = data.ref if data.ref is not None else data.data
            if source is None:
                raise ValueError("a unit with no k-space cannot give the basis")
            basis = _basis(source, self.n_virtual)
            context.coil_compression = basis
        elif basis.n_virtual != self.n_virtual:
            raise ValueError(
                f"the stream's basis keeps {basis.n_virtual} virtual channels, "
                f"not the {self.n_virtual} this step keeps"
            )
        return dataclasses.replace(
            data,
            data=_compressed(data.data, basis),
            ref=_compressed(data.ref, basis),
        )


def coil_labels(header: Any) -> tuple[str, ...]:
    """Return the receive coil labels of a header in channel order.

    ``acquisitionSystemInformation.coilLabel`` ordered by ``coilNumber``; the
    channel indices up to ``receiverChannels`` where it has none.
    """
    system = getattr(header, "acquisitionSystemInformation", None)
    labels = getattr(system, "coilLabel", None) or ()
    if labels:
        ordered = sorted(
            labels, key=lambda label: int(getattr(label, "coilNumber", 0) or 0)
        )
        return tuple(str(getattr(label, "coilName", "")) for label in ordered)
    return tuple(str(index) for index in range(_channels(system)))


def digest(*parts: Any) -> str:
    """Return a 12-digit hexadecimal identifier of arrays and values."""
    state = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            state.update(np.ascontiguousarray(part).tobytes())
        else:
            state.update(repr(part).encode())
    return state.hexdigest()[:12]


def _channels(system: Any) -> int:
    return int(getattr(system, "receiverChannels", 0) or 0)


def _id(state: Any) -> str | None:
    return None if state is None else state.id


def _grid(buffer: ReconBuffer) -> tuple[int, ...]:
    """Return the shape of the voxel axes of the image the whole k-space of ``buffer``'s space makes."""
    extents = dict(buffer.space.extents)
    return (
        *(extents[name] for name in _ENCODED if extents[name] > 1),
        buffer.kspace.shape[-1],
    )


def _geometry(header: Any, buffer: ReconBuffer) -> dict[str, Any]:
    reference = buffer.reference

    def stated(name: str) -> tuple[float, ...] | None:
        value = getattr(reference, name, None)
        return None if value is None else tuple(float(item) for item in value)

    measurement = getattr(header, "measurementInformation", None)
    return {
        "frame_of_reference": getattr(measurement, "frameOfReferenceUID", None),
        "table_position": stated("patient_table_position"),
        "position": stated("position"),
        "read_dir": stated("read_dir"),
        "phase_dir": stated("phase_dir"),
        "slice_dir": stated("slice_dir"),
        "fov": buffer.space.recon_fov,
        "matrix": buffer.space.recon_matrix,
        "grid": _grid(buffer),
        "extent": _extent(header, buffer),
    }


def _extent(header: Any, buffer: ReconBuffer) -> tuple[float, float, float] | None:
    """Return the size in mm of the grid of :func:`_grid` as ``(z, y, x)``, from the encoded space.

    ``z`` is the slice thickness for a grid of one plane. ``None`` when the
    header states no encoded field of view.
    """
    encodings = getattr(header, "encoding", None) or ()
    if buffer.space.index >= len(encodings):
        return None
    encoded = encodings[buffer.space.index].encodedSpace
    fov = getattr(encoded, "fieldOfView_mm", None)
    if fov is None:
        return None
    matrix = encoded.matrixSize
    shape = (1, 1, *_grid(buffer))[-3:]
    counts = (int(matrix.z), int(matrix.y), int(matrix.x))
    sizes = (float(fov.z), float(fov.y), float(fov.x))
    return tuple(
        size if n == 1 else size * n / max(count, 1)
        for size, n, count in zip(sizes, shape, counts, strict=True)
    )


def _method(estimate: Any) -> str:
    function = getattr(estimate, "func", estimate)
    name = getattr(function, "__name__", type(function).__name__)
    keywords = getattr(estimate, "keywords", None) or {}
    options = ", ".join(f"{key}={value!r}" for key, value in keywords.items())
    return f"{name}({options})" if options else name


def _estimated(
    context: ReconContext, ref: ReconBuffer, estimate: Any
) -> CoilSensitivities:
    """Estimate maps from a unit's calibration k-space, zero-filled onto the grid of its space."""
    import torch

    space = ref.space
    if not space.cartesian:
        raise ValueError(
            f"encoding space {space.index} is not Cartesian, and coil_maps estimates "
            "maps from the calibration k-space of a Cartesian space only"
        )
    calibration = ref.kspace
    placement = ref.axes[1:-1]
    if "average" in placement:
        calibration = calibration.mean(axis=ref.axes.index("average"))
        placement = tuple(name for name in placement if name != "average")
    other = [name for name in placement if name not in _ENCODED]
    if other:
        raise ValueError(
            f"the calibration k-space of encoding space {space.index} varies along "
            f"{other}, and coil_maps estimates one set of maps from a k-space "
            "that varies along the partition and phase-encode axes only"
        )
    grid = _grid(ref)
    kspace = np.zeros((ref.coils, *grid), dtype=np.complex64)
    lines = tuple(
        slice(ref.origin[name], ref.origin[name] + size)
        for name, size in zip(placement, calibration.shape[1:-1], strict=True)
    )
    kspace[(slice(None), *lines, slice(None))] = calibration

    maps = estimate(torch.from_numpy(kspace))
    maps = np.asarray(maps.detach().cpu().numpy(), dtype=np.complex64)
    if maps.shape != kspace.shape:
        raise ValueError(
            f"the estimate returned maps of shape {maps.shape} for k-space of "
            f"shape {kspace.shape}"
        )
    measurement = getattr(context.header, "measurementInformation", None)
    source = getattr(measurement, "measurementID", None)
    return CoilSensitivities(
        maps=maps,
        coils=coil_labels(context.header),
        noise=_id(context.noise),
        basis=_id(context.coil_compression),
        geometry=_geometry(context.header, ref),
        method=_method(estimate),
        source=None if source is None else str(source),
    )


def _usable(
    stored: Any,
    where: str,
    data: ReconData,
    context: ReconContext,
    rejected: list[str],
) -> CoilSensitivities | None:
    """Return ``stored`` if it serves ``data``, else record why it does not."""
    if stored is None:
        rejected.append(f"{where}: no maps stored")
        return None
    if not isinstance(stored, CoilSensitivities):
        rejected.append(
            f"{where}: holds a {type(stored).__name__}, not CoilSensitivities"
        )
        return None
    reason = stored.incompatibility(data, context)
    if reason is None:
        return stored
    origin = (
        f"measurement {stored.source}" if stored.source else "an unnamed measurement"
    )
    rejected.append(f"{where}: {reason} (estimated by {stored.method} from {origin})")
    return None


#: Components of :attr:`CoilSensitivities.geometry` that place the maps on a
#: grid; maps of the exam that differ only in these are resliced.
_PLACEMENT = ("position", "read_dir", "phase_dir", "slice_dir", "grid", "extent")


def _fitted(
    stored: Any,
    data: ReconData,
    context: ReconContext,
    rejected: list[str],
) -> np.ndarray | None:
    """Return the exam's maps on the grid and channels of ``data``, else record why they do not serve it."""
    where = "this exam"
    if stored is None:
        rejected.append(f"{where}: no maps stored")
        return None
    if not isinstance(stored, CoilSensitivities):
        rejected.append(
            f"{where}: holds a {type(stored).__name__}, not CoilSensitivities"
        )
        return None
    origin = (
        f"measurement {stored.source}" if stored.source else "an unnamed measurement"
    )
    buffer = data.data if data.data is not None else data.ref
    if buffer is None:
        rejected.append(
            f"{where}: the unit holds no k-space, so its geometry is not known"
        )
        return None
    wanted = _geometry(context.header, buffer)
    basis = context.coil_compression
    project = stored.basis is None and basis is not None
    fields = (
        ("coils", stored.coils, coil_labels(context.header)),
        ("noise", stored.noise, _id(context.noise)),
        *(() if project else (("basis", stored.basis, _id(basis)),)),
        *(
            (f"geometry.{name}", stored.geometry.get(name), wanted.get(name))
            for name in ("frame_of_reference", "table_position")
        ),
    )
    for name, value, requested in fields:
        if not _field_matches(name, value, requested):
            rejected.append(
                f"{where}: {name} differs: stored {value!r}, this unit "
                f"{requested!r} (estimated by {stored.method} from {origin})"
            )
            return None

    maps = stored.maps
    if any(
        not _field_matches(
            f"geometry.{name}", stored.geometry.get(name), wanted.get(name)
        )
        for name in _PLACEMENT
    ):
        source, target = _placement(stored.geometry), _placement(wanted)
        if source is None or target is None:
            rejected.append(
                f"{where}: the geometry of the maps or of this unit is not fully "
                f"stated, so the maps cannot be resliced (estimated by "
                f"{stored.method} from {origin})"
            )
            return None
        from bartorch.tools import reslice

        planes = (1, 1, *stored.geometry["grid"])[-3:]
        maps = reslice(
            maps.reshape(maps.shape[0], *planes), source, target, fill=float("nan")
        )
        maps = maps.reshape(maps.shape[0], *wanted["grid"])
        if np.isnan(maps).any():
            rejected.append(
                f"{where}: this unit's grid reaches outside the calibration "
                f"volume (estimated by {stored.method} from {origin})"
            )
            return None
    if project:
        maps = _projected(np.asarray(maps, dtype=np.complex64), basis)
    return np.asarray(maps, dtype=np.complex64)


def _placement(geometry: dict[str, Any]) -> dict[str, tuple] | None:
    """Return the grid of a :attr:`CoilSensitivities.geometry` as :func:`bartorch.tools.reslice` reads it.

    ``None`` when a component it needs is not stated.
    """
    if any(geometry.get(name) is None for name in _PLACEMENT):
        return None
    if any(not np.any(geometry[name]) for name in _PLACEMENT[1:4]):
        return None
    shape = (1, 1, *geometry["grid"])[-3:]
    extent = geometry["extent"]
    return {
        **{name: geometry[name] for name in _PLACEMENT[:4]},
        "field_of_view": tuple(float(v) for v in extent[::-1]),
        "matrix_size": tuple(int(n) for n in shape[::-1]),
    }


def _projected(maps: np.ndarray, basis: Basis) -> np.ndarray:
    """Return maps of the physical channels on the virtual channels of ``basis``, as k-space is projected."""
    import torch
    from bartorch import tools

    coils = maps.shape[0]
    if coils != basis.matrix.shape[0]:
        raise ValueError(
            f"{coils} channels cannot be projected onto a basis of "
            f"{basis.matrix.shape[0]}"
        )
    virtual = tools.ccapply(
        torch.from_numpy(np.ascontiguousarray(maps)).reshape(coils, -1, 1, 1),
        torch.from_numpy(basis.matrix).reshape(coils, coils, 1, 1, 1),
        p=basis.n_virtual,
        S=True,
    )
    return virtual.cpu().numpy().reshape(basis.n_virtual, *maps.shape[1:])


def _basis(buffer: ReconBuffer, n_virtual: int) -> Basis:
    return basis_of(buffer.kspace[:, buffer.mask], n_virtual)


def basis_of(samples: np.ndarray, n_virtual: int) -> Basis:
    """Return the basis of the ``n_virtual`` strongest virtual channels of ``(coils, n)`` samples (:func:`bartorch.tools.cc`)."""
    import torch
    from bartorch import tools

    coils = samples.shape[0]
    if n_virtual > coils:
        raise ValueError(f"cannot keep {n_virtual} virtual channels of {coils}")
    samples = np.ascontiguousarray(samples, dtype=np.complex64)
    matrix = tools.cc(
        torch.from_numpy(samples).reshape(coils, -1, 1, 1),
        p=n_virtual,
        M=True,
        A=True,
        S=True,
    )
    matrix = matrix.cpu().numpy()[:, :, 0, 0, 0]
    return Basis(matrix, n_virtual, digest(matrix[:, :n_virtual], n_virtual))


def projected_readout(data: np.ndarray, basis: Basis) -> np.ndarray:
    """Return a ``(coils, samples)`` readout on the virtual channels of ``basis``, as :func:`bartorch.tools.ccapply` projects k-space."""
    virtual = basis.matrix[: basis.n_virtual].conj()
    return (virtual.astype(data.dtype, copy=False) @ data).astype(
        data.dtype, copy=False
    )


def _compressed(buffer: ReconBuffer | None, basis: Basis) -> ReconBuffer | None:
    if buffer is None:
        return None
    compressed = copy.copy(buffer)
    compressed.kspace = _projected(
        np.ascontiguousarray(buffer.kspace, dtype=np.complex64), basis
    )
    compressed.coils = basis.n_virtual
    return compressed
