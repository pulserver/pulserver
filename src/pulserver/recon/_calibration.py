"""Coil sensitivities of a reconstruction unit, and the compression of its receive channels."""

from __future__ import annotations

__all__ = ["CoilCompression", "CoilSensitivities", "MissingCalibration", "coil_maps"]

import copy
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._buffers import ReconBuffer, ReconData
from .plugin import COIL_SENSITIVITIES, ReconContext

#: The placement axes the calibration k-space of a unit may vary along.
_ENCODED = ("partition", "phase_encode")


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

    Maps are reused only for data with the coils, prewhitening, compression
    and geometry of the calibration data they were estimated from;
    :meth:`incompatibility` states the comparison. Nothing is resampled,
    regridded or resliced to make maps fit.

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
        space, ordered as ``(z,) y, x``; and ``grid``, the shape of the voxel
        axes of ``maps``. A component the header or acquisition does not state
        is ``None``.
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
        it holds no imaging data.

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
            if stored != requested:
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
    3. The exam's ``COIL_SENSITIVITIES``.

    Stored maps serve only where :meth:`CoilSensitivities.incompatibility`
    finds nothing. The location is the slice counter, not the encoding space,
    so maps estimated from a calibration unit in an encoding space of its own
    serve the imaging units of its slice.

    Parameters
    ----------
    context
        Scan context; its ``device`` is where the estimate runs and the maps
        are returned.
    data
        The unit that needs maps; ``data.counters["slice"]`` is its slice,
        0 when the unit has no such counter.
    estimate
        ``estimate(kspace) -> maps``, such as :func:`bartorch.apps.nlinv_maps`.
        ``kspace`` is the calibration k-space of the unit as a torch tensor
        ``(coils, [z,] y, x)``, zero outside the lines it holds, on the grid of
        its encoding space; ``maps`` are the sensitivities on the same grid.
    required
        Return ``None``, rather than raise, when no source has usable maps.

    Returns
    -------
    torch.Tensor or None
        A copy of the maps ``(coils, [z,] y, x)``, complex64, on
        ``context.device``.

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
    else:
        found = _estimated(context, data.ref, estimate)
        context.coil_maps[location] = found

    if found is None:
        stored = context.coil_maps.get(location)
        found = _usable(
            stored, f"this series, slice {location}", data, context, rejected
        )
    if found is None:
        stored = context.exam.get(COIL_SENSITIVITIES)
        found = _usable(stored, "this exam", data, context, rejected)

    if found is None:
        if not required:
            return None
        raise MissingCalibration(
            f"no coil sensitivities for slice {location}; sources consulted:\n  "
            + "\n  ".join(rejected)
        )
    return torch.tensor(found.maps, device=context.device)


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
    its ``counters`` and ``acquisitions`` with the unit it was made from.
    bartorch is imported when a unit is compressed; the ``coils`` extra
    installs it.

    Corresponds to Gadgetron's ``GenericReconEigenChannelGadget``.

    Parameters
    ----------
    n_virtual
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
            basis = _basis(source, self.n_virtual, context.device)
            context.coil_compression = basis
        elif basis.n_virtual != self.n_virtual:
            raise ValueError(
                f"the stream's basis keeps {basis.n_virtual} virtual channels, "
                f"not the {self.n_virtual} this step keeps"
            )
        return dataclasses.replace(
            data,
            data=_compressed(data.data, basis, context.device),
            ref=_compressed(data.ref, basis, context.device),
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
    }


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
    placement = ref.axes[1:-1]
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
        for name, size in zip(placement, ref.kspace.shape[1:-1], strict=True)
    )
    kspace[(slice(None), *lines, slice(None))] = ref.kspace

    maps = estimate(torch.from_numpy(kspace).to(context.device))
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


def _basis(buffer: ReconBuffer, n_virtual: int, device: str | None) -> Basis:
    import torch
    from bartorch import tools

    coils = buffer.coils
    if n_virtual > coils:
        raise ValueError(f"cannot keep {n_virtual} virtual channels of {coils}")
    samples = np.ascontiguousarray(buffer.kspace[:, buffer.mask], dtype=np.complex64)
    matrix = tools.cc(
        torch.from_numpy(samples).reshape(coils, -1, 1, 1).to(device),
        p=n_virtual,
        M=True,
        A=True,
        S=True,
    )
    matrix = matrix.cpu().numpy()[:, :, 0, 0, 0]
    return Basis(matrix, n_virtual, digest(matrix[:, :n_virtual], n_virtual))


def _compressed(
    buffer: ReconBuffer | None, basis: Basis, device: str | None
) -> ReconBuffer | None:
    if buffer is None:
        return None
    import torch
    from bartorch import tools

    coils = buffer.kspace.shape[0]
    if coils != basis.matrix.shape[0]:
        raise ValueError(
            f"a unit of {coils} channels cannot be projected onto a basis of "
            f"{basis.matrix.shape[0]}"
        )
    kspace = np.ascontiguousarray(buffer.kspace, dtype=np.complex64)
    virtual = tools.ccapply(
        torch.from_numpy(kspace).reshape(coils, -1, 1, 1).to(device),
        torch.from_numpy(basis.matrix).reshape(coils, coils, 1, 1, 1).to(device),
        p=basis.n_virtual,
        S=True,
    )
    compressed = copy.copy(buffer)
    compressed.kspace = (
        virtual.cpu().numpy().reshape(basis.n_virtual, *kspace.shape[1:])
    )
    compressed.coils = basis.n_virtual
    return compressed
