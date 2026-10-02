"""Encoding spaces as the MRD header describes them."""

from __future__ import annotations

__all__ = ["LOOP_COUNTERS", "EncodingSpace"]

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: Counters that become buffer axes when they vary, outermost first. ``segment``
#: is not among them: it indexes part of one readout train, not an image.
LOOP_COUNTERS = (
    "repetition",
    "phase",
    "slice",
    "contrast",
    "set",
    "average",
)

#: The counters a buffer can be laid out along: ``segment`` is one only when asked for.
_LAYOUT_COUNTERS = (*LOOP_COUNTERS, "segment")


def _is_cartesian(encoding: Any) -> bool:
    """Whether an encoding samples a grid; a header stating no trajectory is Cartesian."""
    trajectory = getattr(encoding, "trajectory", None)
    if trajectory is None:
        return True
    name = getattr(trajectory, "name", None) or str(trajectory)
    return name.rsplit(".", 1)[-1].upper() == "CARTESIAN"


def _fov(encoding: Any, dimensions: int) -> tuple[float, ...] | None:
    """Return the reconstructed field of view in metres, ``(z, y, x)`` cut to ``dimensions``.

    ``None`` unless the space states a positive extent along each of them.
    """
    space = getattr(encoding, "reconSpace", None) or encoding.encodedSpace
    fov = getattr(space, "fieldOfView_mm", None)
    if fov is None:
        return None
    extents = tuple(1e-3 * float(getattr(fov, axis, 0) or 0) for axis in "zyx")
    extents = extents[3 - dimensions :]
    return extents if all(extent > 0 for extent in extents) else None


def _limit(limits: Any, name: str) -> int:
    """Extent of one encoding limit, or 0 when the header does not state it."""
    entry = getattr(limits, name, None) if limits is not None else None
    maximum = getattr(entry, "maximum", None) if entry is not None else None
    return 0 if maximum is None else int(maximum) + 1


def _center(limits: Any, name: str) -> int | None:
    """Counter of the k-space centre an encoding limit states, or ``None`` when it states none."""
    entry = getattr(limits, name, None) if limits is not None else None
    center = getattr(entry, "center", None) if entry is not None else None
    return None if center is None else int(center)


@dataclass(frozen=True)
class EncodingSpace:
    """Buffer layout of one encoding space of an MRD header.

    Parameters
    ----------
    index
        Position in the header's encoding list; what ``encoding_space_ref`` names.
    coils
        Receive channels.
    readout
        Samples per readout.
    phase_encodes, partitions
        Extent along ``kspace_encode_step_1`` and ``kspace_encode_step_2``.
    loops
        Counters that vary in this space and are laid out as axes, outermost
        first: those of :data:`LOOP_COUNTERS`, and ``segment`` when
        :meth:`from_header` was asked for it.
    loop_sizes
        Extent of each counter in ``loops``.
    recon_matrix
        Image matrix, ``(n_y, n_x)`` for a plane or ``(n_z, n_y, n_x)`` for a
        volume. Independent of the buffer axes: a stack of spokes has no
        partition axis and a three-dimensional matrix.
    recon_fov
        Field of view of the image in metres, ordered as ``recon_matrix``;
        ``None`` when the header states none.
    cartesian
        Whether the space samples a Cartesian grid. A readout of a Cartesian
        space is placed by its echo along the readout, and its lines by their
        offset from the k-space centre; a non-Cartesian space places a readout
        as it arrives and a view by its counter.
    phase_center, partition_center
        Counter value of the k-space centre along ``kspace_encode_step_1`` and
        ``kspace_encode_step_2``, from the ``center`` of the header's encoding
        limits; ``None`` where the header states none, so the counter is the
        position on the grid.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> space.shape
    (4, 2, 32, 64)
    """

    index: int
    coils: int
    readout: int
    phase_encodes: int
    partitions: int
    loops: tuple[str, ...]
    loop_sizes: tuple[int, ...]
    recon_matrix: tuple[int, ...]
    recon_fov: tuple[float, ...] | None = None
    cartesian: bool = True
    phase_center: int | None = None
    partition_center: int | None = None

    @classmethod
    def from_header(
        cls, header: Any, index: int = 0, loops: Iterable[str] | None = None
    ) -> EncodingSpace:
        """Read encoding space ``index`` of a parsed MRD header.

        For a Cartesian space ``phase_encodes`` and ``partitions`` are the encoded
        matrix, the whole grid whatever number of lines an undersampled scan took,
        and ``phase_center`` and ``partition_center`` place the counters on it. For
        a non-Cartesian space ``phase_encodes`` is the ``kspace_encoding_step_1``
        limit, which counts views, or the encoded matrix when no limit is stated;
        its ``partitions`` is the larger of the encoded matrix and the
        ``kspace_encoding_step_2`` limit, a stack being Cartesian along z, and no
        counter is shifted. ``recon_matrix`` and ``recon_fov`` fall back to the
        encoded space when the header has no ``reconSpace``.

        Parameters
        ----------
        header
            Parsed MRD header.
        index
            Position in the header's encoding list.
        loops
            Counters to lay out as axes where the header gives them more than one
            value, in the order of :data:`LOOP_COUNTERS` followed by ``segment``;
            ``None`` takes every counter of :data:`LOOP_COUNTERS`.

        Raises
        ------
        IndexError
            If the header describes no such encoding space.
        """
        encodings = getattr(header, "encoding", None) or ()
        encoding = encodings[index]
        encoded = encoding.encodedSpace.matrixSize
        limits = getattr(encoding, "encodingLimits", None)

        wanted = LOOP_COUNTERS if loops is None else set(loops)
        names: list[str] = []
        sizes: list[int] = []
        for name in _LAYOUT_COUNTERS:
            extent = _limit(limits, name)
            if name in wanted and extent > 1:
                names.append(name)
                sizes.append(extent)

        system = getattr(header, "acquisitionSystemInformation", None)
        coils = int(getattr(system, "receiverChannels", 1) or 1)

        matrix = encoded if getattr(encoding, "reconSpace", None) is None else None
        matrix = matrix or getattr(encoding.reconSpace, "matrixSize", None) or encoded
        recon_matrix = (int(matrix.z), int(matrix.y), int(matrix.x))
        if recon_matrix[0] == 1:
            recon_matrix = recon_matrix[1:]

        gridded = _is_cartesian(encoding)
        if gridded:
            phase_encodes, partitions = int(encoded.y), int(encoded.z)
        else:
            phase_encodes = _limit(limits, "kspace_encoding_step_1") or int(encoded.y)
            partitions = max(_limit(limits, "kspace_encoding_step_2"), int(encoded.z))

        return cls(
            index=index,
            coils=coils,
            readout=int(encoded.x),
            phase_encodes=phase_encodes,
            partitions=partitions,
            loops=tuple(names),
            loop_sizes=tuple(sizes),
            recon_matrix=recon_matrix,
            recon_fov=_fov(encoding, len(recon_matrix)),
            cartesian=gridded,
            phase_center=_center(limits, "kspace_encoding_step_1") if gridded else None,
            partition_center=_center(limits, "kspace_encoding_step_2")
            if gridded
            else None,
        )

    @classmethod
    def all_from_header(
        cls, header: Any, loops: Iterable[str] | None = None
    ) -> tuple[EncodingSpace, ...]:
        """Read every encoding space of a parsed MRD header, in index order.

        ``loops`` is as for :meth:`from_header`.
        """
        encodings = getattr(header, "encoding", None) or ()
        return tuple(
            cls.from_header(header, index, loops) for index in range(len(encodings))
        )

    def offset(self, axis: str) -> int:
        """Shift that places a counter of ``axis`` on this space's grid.

        ``extent // 2 - center`` along ``phase_encode`` and, in a volume,
        ``partition`` of a Cartesian space whose header states the centre, so the
        centre line lands on ``extent // 2``; 0 for every other axis and for a
        space that states no centre.
        """
        if not self.cartesian:
            return 0
        if axis == "phase_encode" and self.phase_center is not None:
            return self.phase_encodes // 2 - self.phase_center
        if (
            axis == "partition"
            and self.partitions > 1
            and self.partition_center is not None
        ):
            return self.partitions // 2 - self.partition_center
        return 0

    @property
    def extents(self) -> tuple[tuple[str, int], ...]:
        """Placement axes and their extents, outermost first, including extents of 1.

        Coil and readout are not placement axes.
        """
        return (
            *zip(self.loops, self.loop_sizes, strict=True),
            ("partition", self.partitions),
            ("phase_encode", self.phase_encodes),
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """Axis names of a buffer of this space.

        ``coil``, the placement axes whose extent exceeds 1, then ``readout``.
        """
        varying = tuple(name for name, extent in self.extents if extent > 1)
        return ("coil", *varying, "readout")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a buffer of this space, named by :attr:`axes`."""
        varying = tuple(extent for _, extent in self.extents if extent > 1)
        return (self.coils, *varying, self.readout)
