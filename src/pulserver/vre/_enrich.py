"""Enrichment of an MRD stream from the sequence that produced it.

The chain's :class:`~pulserver.mrd.ReadoutTable` rows are mapped onto MRD
counters, flags and encoding spaces, and acquisitions are matched to rows in
stream order.
"""

from __future__ import annotations

__all__ = [
    "SequenceTable",
    "TableSpace",
    "enrich_acquisition",
    "enrich_header",
]

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ismrmrd.xsd as xsd
import numpy as np

from .._labels import MRD_COUNTERS, MRD_FLAGS
from ..mrd._acquisitions import AcquisitionFlag
from ..mrd._sequence import ReadoutTable, SequenceDefinitions, read_chain

_F = AcquisitionFlag

#: Counter label, its first and last boundary flags, and whether it selects a
#: whole image rather than a position within one.
_BOUNDARY_COUNTERS = (
    ("SLC", _F.FIRST_IN_SLICE, _F.LAST_IN_SLICE, True),
    ("ECO", _F.FIRST_IN_CONTRAST, _F.LAST_IN_CONTRAST, True),
    ("PHS", _F.FIRST_IN_PHASE, _F.LAST_IN_PHASE, True),
    ("REP", _F.FIRST_IN_REPETITION, _F.LAST_IN_REPETITION, True),
    ("AVG", _F.FIRST_IN_AVERAGE, _F.LAST_IN_AVERAGE, True),
    ("SET", _F.FIRST_IN_SET, _F.LAST_IN_SET, True),
    ("LIN", _F.FIRST_IN_ENCODE_STEP1, _F.LAST_IN_ENCODE_STEP1, False),
    ("PAR", _F.FIRST_IN_ENCODE_STEP2, _F.LAST_IN_ENCODE_STEP2, False),
    ("SEG", _F.FIRST_IN_SEGMENT, _F.LAST_IN_SEGMENT, False),
)

#: Flag labels copied from the sequence, as bit masks. Boundary flags are
#: derived from the counters instead.
_SEQUENCE_FLAGS = {
    name: _F[constant.removeprefix("ACQ_")].value
    for name, constant in MRD_FLAGS.items()
    if not name.startswith(("FIRST", "LAST"))
}

#: Counter label to its ``encodingLimits`` field.
_LIMIT_FIELDS = {
    name: field.replace("encode_step", "encoding_step")
    for name, field in MRD_COUNTERS.items()
}

#: Counters every encoding space states limits for, written or not.
_STANDARD_LIMITS = ("LIN", "PAR", "AVG", "SLC", "ECO", "PHS", "REP", "SET", "SEG")


@dataclass(frozen=True)
class TableSpace:
    """One encoding space of a :class:`SequenceTable`.

    Attributes
    ----------
    subsequence
        Index of the chain file the space belongs to.
    navigator
        Whether the space holds the subsequence's ``NAV`` readouts.
    matrix
        ``(x, y, z)`` from ``Matrix``, or ``NavMatrix`` for a navigator;
        ``None`` when undefined.
    fov_mm
        ``(x, y, z)`` from ``FOV``, or ``NavFOV`` for a navigator, in mm;
        ``None`` when undefined.
    trajectory
        Whether some readout of the space keeps more than one k axis, which
        makes the space non-Cartesian.
    """

    subsequence: int
    navigator: bool
    matrix: tuple[int, int, int] | None
    fov_mm: tuple[float, float, float] | None
    trajectory: bool


@dataclass(frozen=True, eq=False)
class SequenceTable:
    """The readouts of a sequence chain as MRD describes them, in play order.

    k-space is not tabulated: :meth:`readout_k` integrates a readout's when
    asked, through the :class:`~pulserver.mrd.ReadoutTable` of its chain file.

    Attributes
    ----------
    counters : dict of str to ndarray
        Every label of ``MRD_COUNTERS``, one ``int32`` value per readout; 0
        where the sequence never writes it.
    flags : ndarray
        ``uint64`` ISMRMRD flag masks: the non-boundary flags the sequence
        sets, first/last flags of every counter the sequence writes, and
        ``LAST_IN_MEASUREMENT`` on the final readout of the chain. A boundary
        is read within its encoding space and the other image-selecting
        counters, so a slice closes once per echo.
    center_sample, trajectory_dimensions, num_samples : ndarray
        As :class:`~pulserver.mrd.ReadoutTable` states them, over the chain.
    sample_time_us : ndarray
        ``float32`` dwell, in µs.
    encoding_space : ndarray
        ``int32`` index into :attr:`spaces`.
    spaces : tuple of TableSpace
        Numbered in chain order: each subsequence's primary space, then its
        navigator space when it has ``NAV`` readouts.
    sequence_parameters : dict of str to list of float
        Over the chain's definitions: the ``TR`` and ``TI`` minima and every
        distinct ``TE`` in ascending order, in ms, and the ``FlipAngle``
        maximum, in degrees. Keys no file defines are absent.
    """

    counters: dict[str, np.ndarray]
    flags: np.ndarray
    center_sample: np.ndarray
    trajectory_dimensions: np.ndarray
    sample_time_us: np.ndarray
    encoding_space: np.ndarray
    num_samples: np.ndarray
    spaces: tuple[TableSpace, ...]
    sequence_parameters: dict[str, list[float]]
    _files: tuple[ReadoutTable, ...] = dataclasses.field(repr=False)
    _first_rows: np.ndarray = dataclasses.field(repr=False)

    def __len__(self) -> int:
        return int(self.num_samples.size)

    def readout_k(self, index: int) -> np.ndarray:
        """Return the k-space position of each sample of one readout, in 1/m.

        ``(3, num_samples)``, absolute, with block rotations applied.
        """
        file = int(np.searchsorted(self._first_rows, index, side="right")) - 1
        return self._files[file].readout_k(index - int(self._first_rows[file]))

    @classmethod
    def read(cls, path: Path | str) -> SequenceTable:
        """Tabulate the ``NextSequence`` chain starting at a sequence file.

        Raises
        ------
        FileNotFoundError
            If a file of the chain does not exist.
        ValueError
            If the chain names a file it has already played.
        """
        parts: list[dict[str, Any]] = []
        files: list[ReadoutTable] = []
        spaces: list[TableSpace] = []
        tr: list[float] = []
        te: list[float] = []
        ti: list[float] = []
        flip: list[float] = []
        for subsequence, (_, seq) in enumerate(read_chain(path)):
            readouts = ReadoutTable.from_sequence(seq)
            definitions = SequenceDefinitions.from_sequence(seq)
            part, part_spaces = _map_readouts(
                readouts, definitions, subsequence, len(spaces)
            )
            files.append(readouts)
            parts.append(part)
            spaces.extend(part_spaces)
            tr.extend(definitions.tr)
            te.extend(definitions.te)
            ti.extend(definitions.ti)
            flip.extend(definitions.flip_angle)

        def joined(name: str, dtype: Any) -> np.ndarray:
            return np.concatenate([part[name] for part in parts]).astype(dtype)

        flags = joined("flags", np.uint64)
        if flags.size:
            flags[-1] |= np.uint64(_F.LAST_IN_MEASUREMENT.value)

        parameters: dict[str, list[float]] = {}
        if tr:
            parameters["TR"] = [1e3 * min(tr)]
        if te:
            parameters["TE"] = [1e3 * value for value in sorted(set(te))]
        if ti:
            parameters["TI"] = [1e3 * min(ti)]
        if flip:
            parameters["FlipAngle"] = [max(flip)]

        return cls(
            counters={
                name: np.concatenate([part["counters"][name] for part in parts]).astype(
                    np.int32
                )
                for name in MRD_COUNTERS
            },
            flags=flags,
            center_sample=joined("center_sample", np.int32),
            trajectory_dimensions=joined("trajectory_dimensions", np.int8),
            sample_time_us=joined("sample_time_us", np.float32),
            encoding_space=joined("encoding_space", np.int32),
            num_samples=joined("num_samples", np.int32),
            spaces=tuple(spaces),
            sequence_parameters=parameters,
            _files=tuple(files),
            _first_rows=np.cumsum([0] + [len(file) for file in files[:-1]]),
        )


def enrich_header(header: Any, table: SequenceTable) -> None:
    """Describe the table's encoding spaces and sequence parameters in an MRD header.

    For every space, sets ``encodedSpace`` and ``reconSpace`` from the
    sequence's matrix and field of view when defined, ``encodingLimits`` from
    the counters of its readouts (minimum 0, centre half the maximum), and
    ``trajectory`` to ``OTHER`` or ``CARTESIAN``. Other fields of an existing
    encoding are kept; missing encodings are appended. Sequence parameters the
    table holds replace the header's.
    """
    parameters = header.sequenceParameters
    if table.sequence_parameters and parameters is None:
        parameters = xsd.sequenceParametersType()
        header.sequenceParameters = parameters
    for key, field in (
        ("TR", "TR"),
        ("TE", "TE"),
        ("TI", "TI"),
        ("FlipAngle", "flipAngle_deg"),
    ):
        if key in table.sequence_parameters:
            setattr(parameters, field, list(table.sequence_parameters[key]))

    encodings = list(header.encoding or ())
    for index, space in enumerate(table.spaces):
        members = table.encoding_space == index
        written = [
            name
            for name in MRD_COUNTERS
            if name in _STANDARD_LIMITS or table.counters[name][members].any()
        ]
        limits = xsd.encodingLimitsType(
            **{
                _LIMIT_FIELDS[name]: _limit(table.counters[name][members])
                for name in written
            }
        )
        trajectory = (
            xsd.trajectoryType.OTHER
            if space.trajectory
            else xsd.trajectoryType.CARTESIAN
        )
        if index < len(encodings):
            encoding = encodings[index]
            if space.matrix is not None or space.fov_mm is not None:
                encoding.encodedSpace = _encoding_space(space, encoding.encodedSpace)
                encoding.reconSpace = _encoding_space(space, encoding.reconSpace)
            encoding.encodingLimits = limits
            encoding.trajectory = trajectory
        else:
            encodings.append(
                xsd.encodingType(
                    encodedSpace=_encoding_space(space, None),
                    reconSpace=_encoding_space(space, None),
                    encodingLimits=limits,
                    trajectory=trajectory,
                )
            )
    header.encoding = encodings


def enrich_acquisition(acquisition: Any, table: SequenceTable, index: int) -> None:
    """Stamp row ``index`` of the table on one acquisition, in place.

    Sets the encoding counters, flags, ``sample_time_us`` and
    ``encoding_space_ref``, and ``center_sample`` unless k does not move
    across the readout, in which case the received value stays. A readout
    whose k moves gets it as ``traj``, trailing constant axes dropped. The
    samples are left as received.

    Raises
    ------
    ValueError
        If the acquisition's sample count differs from the row's.
    """
    count = int(table.num_samples[index])
    if int(acquisition.number_of_samples) != count:
        raise ValueError(
            f"acquisition {index} has {acquisition.number_of_samples} samples; "
            f"the sequence plays {count}"
        )

    counters = acquisition.idx
    for name, field in MRD_COUNTERS.items():
        value = int(table.counters[name][index])
        if name.startswith("USER"):
            counters.user[int(name.removeprefix("USER"))] = value
        else:
            setattr(counters, field, value)
    acquisition.flags = int(table.flags[index])
    if table.center_sample[index] >= 0:
        acquisition.center_sample = int(table.center_sample[index])
    acquisition.sample_time_us = float(table.sample_time_us[index])
    acquisition.encoding_space_ref = int(table.encoding_space[index])

    dimensions = int(table.trajectory_dimensions[index])
    if dimensions:
        k = table.readout_k(index)
        data = np.array(acquisition.data)
        acquisition.resize(count, int(acquisition.active_channels), dimensions)
        acquisition.traj[:] = k[:dimensions].T
        acquisition.data[:] = data


# %% private module subroutines


def _map_readouts(
    readouts: ReadoutTable,
    definitions: SequenceDefinitions,
    subsequence: int,
    first_space: int,
) -> tuple[dict[str, Any], list[TableSpace]]:
    """Rows and encoding spaces of one subsequence, spaces numbered from ``first_space``."""
    count = len(readouts)
    labels = readouts.labels
    counters = {
        name: labels.get(name, np.zeros(count, dtype=np.int64)) for name in MRD_COUNTERS
    }

    flags = np.zeros(count, dtype=np.uint64)
    for name, bit in _SEQUENCE_FLAGS.items():
        if name in labels:
            flags[labels[name] != 0] |= np.uint64(bit)
    navigator = (flags & np.uint64(_F.IS_NAVIGATION_DATA.value)) != 0
    local_space = navigator.astype(np.int64)
    flags |= _boundary_flags(counters, local_space, set(labels))

    spaces = []
    for local, is_navigator in ((0, False), (1, True)):
        if is_navigator and not navigator.any():
            break
        fov = definitions.navigator_fov if is_navigator else definitions.fov
        spaces.append(
            TableSpace(
                subsequence=subsequence,
                navigator=is_navigator,
                matrix=definitions.navigator_matrix
                if is_navigator
                else definitions.matrix,
                fov_mm=None if fov is None else tuple(round(1e3 * v, 9) for v in fov),
                trajectory=bool(
                    (readouts.trajectory_dimensions[local_space == local] > 1).any()
                ),
            )
        )

    part = {
        "counters": counters,
        "flags": flags,
        "center_sample": readouts.center_sample,
        "trajectory_dimensions": readouts.trajectory_dimensions,
        "sample_time_us": 1e6 * readouts.dwell,
        "encoding_space": first_space + local_space,
        "num_samples": readouts.num_samples,
    }
    return part, spaces


def _boundary_flags(
    counters: dict[str, np.ndarray], space: np.ndarray, written: set[str]
) -> np.ndarray:
    """First and last flags of every written counter, keyed by space and the other written image counters."""
    count = space.size
    flags = np.zeros(count, dtype=np.uint64)
    if not count:
        return flags
    for name, first, last, _ in _BOUNDARY_COUNTERS:
        if name not in written:
            continue
        enclosing = [
            other
            for other, _, _, selects_image in _BOUNDARY_COUNTERS
            if selects_image and other != name and other in written
        ]
        keys = np.stack(
            [space, *(counters[other] for other in enclosing), counters[name]], axis=1
        )
        _, first_at = np.unique(keys, axis=0, return_index=True)
        _, last_from_end = np.unique(keys[::-1], axis=0, return_index=True)
        flags[first_at] |= np.uint64(first.value)
        flags[count - 1 - last_from_end] |= np.uint64(last.value)
    return flags


def _limit(values: np.ndarray) -> Any:
    maximum = int(values.max()) if values.size else 0
    return xsd.limitType(minimum=0, maximum=maximum, center=maximum // 2)


def _encoding_space(space: TableSpace, current: Any) -> Any:
    """Return an ``encodingSpaceType`` from the space's definitions, falling back to ``current`` per field."""
    if space.matrix is not None:
        matrix = xsd.matrixSizeType(
            x=space.matrix[0], y=space.matrix[1], z=space.matrix[2]
        )
    elif current is not None:
        matrix = current.matrixSize
    else:
        matrix = xsd.matrixSizeType()
    if space.fov_mm is not None:
        fov = xsd.fieldOfViewMm(x=space.fov_mm[0], y=space.fov_mm[1], z=space.fov_mm[2])
    elif current is not None:
        fov = current.fieldOfView_mm
    else:
        fov = xsd.fieldOfViewMm(x=0.0, y=0.0, z=0.0)
    return xsd.encodingSpaceType(matrixSize=matrix, fieldOfView_mm=fov)
