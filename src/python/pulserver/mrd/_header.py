"""The header a scan is described by, and the ways into it.

The encoding space says how large the acquired matrix is and how much of it
was sampled; the user parameters and the diffusion table are where a sequence
leaves what no standard field has a place for.
"""

from __future__ import annotations

__all__ = ["diffusion_table", "user_parameter"]

from dataclasses import dataclass
from typing import Any

from ._metadata import diffusion_table, user_parameter


LOOP_COUNTERS = (
    "repetition",
    "phase",
    "slice",
    "contrast",
    "set",
    "average",
)


#: Every axis an acquisition can be placed along, which is the vocabulary
#: :meth:`ReconBuffer.select` accepts whether or not this scan varies it.
_AXIS_NAMES = frozenset((*LOOP_COUNTERS, "partition", "phase_encode"))


def _is_cartesian(encoding: Any) -> bool:
    """Whether this encoding samples a grid rather than a trajectory.

    A header that does not say is read as Cartesian, which is what an MRD
    header omitting the field has always meant.
    """
    trajectory = getattr(encoding, "trajectory", None)
    if trajectory is None:
        return True
    name = getattr(trajectory, "name", None) or str(trajectory)
    return name.rsplit(".", 1)[-1].upper() == "CARTESIAN"


def _limit(limits: Any, name: str) -> int:
    """Extent of one encoding limit, or 0 when the header does not state it."""
    entry = getattr(limits, name, None) if limits is not None else None
    maximum = getattr(entry, "maximum", None) if entry is not None else None
    return 0 if maximum is None else int(maximum) + 1


@dataclass(frozen=True)
class EncodingSpace:
    """What one encoding space of the header says its buffer must hold.

    Parameters
    ----------
    index
        Position in the header's encoding list, which is what an acquisition's
        ``encoding_space_ref`` names.
    coils
        Receive channels.
    readout
        Samples per readout.
    phase_encodes, partitions
        Extent along ``kspace_encode_step_1`` and ``kspace_encode_step_2``.
        Which of the header's two statements answers this is what the encoding's
        ``trajectory`` decides: a Cartesian space is a grid, and an undersampled
        one still needs every column of it, so the encoded matrix wins over a
        limit that counts only the lines acquired. A non-Cartesian space has no
        grid -- its ``kspace_encode_step_1`` counts views, which bear no
        relation to the image matrix -- so the limit wins.
    loops
        Names of the counters that vary in this space, outermost first.
    loop_sizes
        Extent of each of those counters.
    recon_matrix
        The image matrix the header asks for, ``(n_y, n_x)`` for a plane or
        ``(n_z, n_y, n_x)`` for a volume -- which of the two is the header's
        answer, not the buffer's: a stack of spokes has no partition axis and
        still reconstructs a volume.

    Examples
    --------
    >>> import pulserver.mrd as mrd
    >>> space = mrd.EncodingSpace(
    ...     index=0, coils=4, readout=64, phase_encodes=32, partitions=1,
    ...     loops=("slice",), loop_sizes=(2,), recon_matrix=(32, 32),
    ... )
    >>> space.recon_matrix
    (32, 32)
    """

    index: int
    coils: int
    readout: int
    phase_encodes: int
    partitions: int
    loops: tuple[str, ...]
    loop_sizes: tuple[int, ...]
    recon_matrix: tuple[int, ...]

    @classmethod
    def from_header(cls, header: Any, index: int = 0) -> EncodingSpace:
        """Read encoding space ``index`` out of a parsed MRD header.

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
        )

    @classmethod
    def all_from_header(cls, header: Any) -> tuple[EncodingSpace, ...]:
        """Read every encoding space the header describes.

        One per subsequence, so this is the whole scan's layout, known before a
        single acquisition arrives.
        """
        encodings = getattr(header, "encoding", None) or ()
        return tuple(cls.from_header(header, index) for index in range(len(encodings)))

    @property
    def extents(self) -> tuple[tuple[str, int], ...]:
        """Every axis an acquisition is placed along, named, outermost first.

        Coils and readout are not among them: an acquisition spans those
        rather than being placed along them.
        """
        return (
            *zip(self.loops, self.loop_sizes, strict=True),
            ("partition", self.partitions),
            ("phase_encode", self.phase_encodes),
        )

    @property
    def axes(self) -> tuple[str, ...]:
        """Name of every axis of :attr:`ReconBuffer.kspace`, in order.

        The ones that vary. A scan with one partition has no partition axis,
        which is what makes a two-dimensional buffer two-dimensional.
        """
        varying = tuple(name for name, extent in self.extents if extent > 1)
        return ("coil", *varying, "readout")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape a buffer of this space has, before any widening."""
        varying = tuple(extent for _, extent in self.extents if extent > 1)
        return (self.coils, *varying, self.readout)
