"""Within-shot echo orderings for EPI trains.

An EPI shot walks a fixed pattern of phase-encode steps and repeats it every
repetition, shifted bodily to wherever that shot samples. These build the
pattern: offsets from the shot's own origin, one row per echo.

References
----------
Segmented blipped-CAIPI (``'caipi'``) follows Stirnberg and Stocker, "Segmented
k-space blipped-controlled aliasing in parallel imaging for high spatiotemporal
resolution EPI", Magn Reson Med 2021;85:1540-1551 (DOI 10.1002/mrm.28486).
The ``'zigzag'`` traversal follows Dong et al., "Single-shot echo planar
time-resolved imaging for multi-echo functional MRI", Magn Reson Med
2025;93:993-1013 (DOI 10.1002/mrm.30327).
"""

from __future__ import annotations

__all__ = ["calc_epi_order"]

import numpy as np

from ._masks import make_caipirinha_mask

_SCHEMES = ("linear", "caipi", "zigzag")


def calc_epi_order(
    etl: int,
    *,
    scheme: str = "linear",
    acceleration: int = 1,
    segments: int = 1,
    partition_acceleration: int = 1,
    caipi_shift: int = 1,
    extent: int | None = None,
) -> np.ndarray:
    """Phase-encode offsets an EPI shot visits, relative to where it starts.

    The first row is always ``(0, 0)``: the pattern says where the shot goes
    *from* its origin, and placing that origin is the scan loop's job. The
    second column is the partition offset, zero throughout for the schemes
    that do not encode one.

    Parameters
    ----------
    etl : int
        Echoes in the train.
    scheme : {'linear', 'caipi', 'zigzag'}, optional
        ``'linear'`` steps by ``segments * acceleration`` every echo and never
        leaves its partition -- plain segmented EPI, and plain single-shot EPI
        at the defaults. ``'caipi'`` adds the partition sawtooth that turns
        that into segmented blipped-CAIPI. ``'zigzag'`` walks up and down a
        phase-encode segment instead of across the whole matrix, which is what
        lets a shot sample the same lines at many echo times.
    acceleration : int, optional
        Phase-encode undersampling, ``Ry``: lines the blip skips.
    segments : int, optional
        Shots the train is interleaved across, ``S``. The blip becomes
        ``S * Ry``, which shortens the train and widens the phase-encode
        bandwidth without changing the lattice sampled.
    partition_acceleration : int, optional
        Partition undersampling ``Rz``, the height of the CAIPI cycle.
        ``'caipi'`` only.
    caipi_shift : int, optional
        Partitions the pattern climbs per acquired line, ``delta_z``.
        ``'caipi'`` only.
    extent : int, optional
        Phase-encode lines one pass spans, ``R_seg``. Required by
        ``'zigzag'``, refused by the others.

    Returns
    -------
    numpy.ndarray
        ``(etl, 2)`` integer ``(ky, kz)`` offsets from the shot's origin.

    Raises
    ------
    ValueError
        If a count is out of range, ``scheme`` is unknown, or ``extent`` is
        given for a scheme that has no use for it and vice versa.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> pp.calc_epi_order(4, acceleration=2)[:, 0]
    array([0, 2, 4, 6])

    Segmenting widens the blip without moving the lattice, so two shots of
    three cover what one shot of six did:

    >>> pp.calc_epi_order(3, acceleration=2, segments=2)[:, 0]
    array([0, 4, 8])

    A CAIPI shell climbs the partitions and wraps, which is what makes the
    partition blips take only two values:

    >>> order = pp.calc_epi_order(6, scheme="caipi", partition_acceleration=3)
    >>> order[:, 1]
    array([0, 1, 2, 0, 1, 2])

    A zigzag turns inside its segment, and the return pass is offset by half a
    blip so it samples between the outward one:

    >>> pp.calc_epi_order(9, scheme="zigzag", acceleration=4, extent=12)[:, 0]
    array([ 0,  4,  8, 12, 10,  6,  2,  0,  4])

    See Also
    --------
    calc_traversal_order : orderings over a single axis, for an outer loop.
    make_caipirinha_mask : the lattice a ``'caipi'`` train tiles.
    """
    etl = int(etl)
    if etl < 1:
        raise ValueError("etl must be positive")
    if scheme not in _SCHEMES:
        raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
    acceleration, segments = int(acceleration), int(segments)
    if acceleration < 1 or segments < 1:
        raise ValueError("acceleration and segments must be positive")
    if (extent is None) != (scheme != "zigzag"):
        raise ValueError(
            "extent is what a 'zigzag' turns around inside and means nothing to the "
            "other schemes; give it for 'zigzag' and leave it unset otherwise"
        )

    if scheme == "zigzag":
        line = _zigzag(etl, acceleration * segments, int(extent))
        partition = np.zeros(etl, dtype=np.intp)
    else:
        line = np.arange(etl, dtype=np.intp) * (acceleration * segments)
        partition = (
            _caipi_partitions(line, acceleration, int(partition_acceleration), int(caipi_shift))
            if scheme == "caipi"
            else np.zeros(etl, dtype=np.intp)
        )
    return np.column_stack((line, partition))


def _caipi_partitions(line: np.ndarray, ry: int, rz: int, shift: int) -> np.ndarray:
    """The partition each phase-encode line sits on, read off the CAIPI lattice.

    Taken from :func:`make_caipirinha_mask` rather than restated, so a train
    and the mask it is meant to tile cannot drift apart. One period of the
    lattice is enough: every sampled row carries exactly one partition.
    """
    if rz < 1:
        raise ValueError("partition_acceleration must be positive")
    mask = make_caipirinha_mask((int(line[-1]) + 1, rz), ry, rz, delta=shift)
    return np.array([np.flatnonzero(mask[index])[0] for index in line], dtype=np.intp)


def _zigzag(etl: int, step: int, extent: int) -> np.ndarray:
    """Up and down one phase-encode segment, alternating between two lattices.

    The outward pass walks ``0, step, 2 * step, ...`` as far as ``extent``
    allows; the return pass walks the same ladder offset by ``step // 2``, so
    consecutive passes sample between each other and every blip stays within
    one ``step``. The two passes together are the cycle, repeated until the
    train runs out.
    """
    if extent < 0:
        raise ValueError("extent must be nonnegative")
    outward = np.arange(0, extent + 1, step, dtype=np.intp)
    inward = outward[:-1] + step // 2 if step > 1 else outward[:-1]
    cycle = np.concatenate((outward, inward[::-1]))
    repeats = -(-etl // len(cycle))
    return np.tile(cycle, repeats)[:etl]
