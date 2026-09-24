"""Encoding spaces as the MRD header describes them."""

from __future__ import annotations

__all__ = ["LOOP_COUNTERS", "EncodingSpace"]

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
        Counters of :data:`LOOP_COUNTERS` that vary in this space, outermost first.
    loop_sizes
        Extent of each counter in ``loops``.
    recon_matrix
        Image matrix, ``(n_y, n_x)`` for a plane or ``(n_z, n_y, n_x)`` for a
        volume. Independent of the buffer axes: a stack of spokes has no
        partition axis and a three-dimensional matrix.
    recon_fov
        Field of view of the image in metres, ordered as ``recon_matrix``;
        ``None`` when the header states none.

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

    @classmethod
    def from_header(cls, header: Any, index: int = 0) -> EncodingSpace:
        """Read encoding space ``index`` of a parsed MRD header.

        For a Cartesian space ``phase_encodes`` is the larger of the encoded matrix
        and the ``kspace_encoding_step_1`` limit, since an undersampled grid still
        needs every line. For a non-Cartesian space it is the limit, which counts
        views, or the encoded matrix when no limit is stated. ``partitions`` is
        always the larger of the two. ``recon_matrix`` and ``recon_fov`` fall
        back to the encoded space when the header has no ``reconSpace``.

        Raises
        ------
        IndexError
            If the header describes no such encoding space.
        """
        encodings = getattr(header, "encoding", None) or ()
        encoding = encodings[index]
        encoded = encoding.encodedSpace.matrixSize
        limits = getattr(encoding, "encodingLimits", None)

        loops: list[str] = []
        sizes: list[int] = []
        for name in LOOP_COUNTERS:
            extent = _limit(limits, name)
            if extent > 1:
                loops.append(name)
                sizes.append(extent)

        system = getattr(header, "acquisitionSystemInformation", None)
        coils = int(getattr(system, "receiverChannels", 1) or 1)

        matrix = encoded if getattr(encoding, "reconSpace", None) is None else None
        matrix = matrix or getattr(encoding.reconSpace, "matrixSize", None) or encoded
        recon_matrix = (int(matrix.z), int(matrix.y), int(matrix.x))
        if recon_matrix[0] == 1:
            recon_matrix = recon_matrix[1:]

        views = _limit(limits, "kspace_encoding_step_1")
        partitions = _limit(limits, "kspace_encoding_step_2")
        gridded = _is_cartesian(encoding)

        return cls(
            index=index,
            coils=coils,
            readout=int(encoded.x),
            phase_encodes=max(views, int(encoded.y))
            if gridded
            else views or int(encoded.y),
            # A stack is Cartesian along z whatever it does in plane.
            partitions=max(partitions, int(encoded.z)),
            loops=tuple(loops),
            loop_sizes=tuple(sizes),
            recon_matrix=recon_matrix,
            recon_fov=_fov(encoding, len(recon_matrix)),
        )

    @classmethod
    def all_from_header(cls, header: Any) -> tuple[EncodingSpace, ...]:
        """Read every encoding space of a parsed MRD header, in index order."""
        encodings = getattr(header, "encoding", None) or ()
        return tuple(cls.from_header(header, index) for index in range(len(encodings)))

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
