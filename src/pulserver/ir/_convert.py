"""Conversion of a Pulseq sequence into the scanner's segmented binary IR."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pypulseqpp as pp

from .._accelerators import require
from ..mrd._sequence import read_chain
from ._source import conversion_payload


def _scanner(system: pp.Opts) -> tuple[float, ...]:
    """Gyromagnetic ratio, field strength and the four rasters, in Hz/T, T and us."""
    return (
        float(system.gamma),
        float(system.B0),
        system.rf_raster_time * 1e6,
        system.grad_raster_time * 1e6,
        system.adc_raster_time * 1e6,
        system.block_duration_raster * 1e6,
    )


def cache_path(seq_path: Path | str, cache_ext: str = ".pseg") -> Path:
    """Return the cache file of a sequence: its last suffix replaced by ``cache_ext``."""
    return Path(seq_path).with_suffix(cache_ext)


def chain(seq_path: Path | str) -> list[Path]:
    """Return the files of the ``NextSequence`` chain starting at a sequence file, in play order.

    Raises
    ------
    FileNotFoundError
        If a file of the chain does not exist.
    ValueError
        If the chain names a file it has already played.
    """
    return [path for path, _ in read_chain(seq_path)]


def convert(
    seq_path: Path | str,
    system: pp.Opts,
    *,
    fov_offset: Sequence[float] | None = None,
    vendor: int = 0,
    label_column_map: Sequence[int] = (0, 1, 2),
    cache_ext: str = ".pseg",
    verify_signature: bool = True,
) -> Path:
    """Segment a sequence file and write its IR cache beside it.

    The ``NextSequence`` chain starting at the file is read as the
    subsequences of one scan. An existing cache at the destination is
    replaced. RF vendor statistics are left at zero: ``vendor`` only tags the
    cache for the reader that loads it, which must be built for that vendor.

    The files are expected in the logical frame, and each is moved to
    ``fov_offset`` with :func:`prescribe` before it is segmented. The
    prescription's rotation is not applied here: the scanner plays the cache
    through its rotation matrix, composed after each block's own rotation.

    Parameters
    ----------
    seq_path
        Text or binary Pulseq file.
    system
        Limits and rasters the scan is segmented under.
    fov_offset
        Translation of the field-of-view centre along the logical readout,
        phase and slice axes, in metres. None or zero leaves the files as
        designed.
    vendor
        ``PULSEG_VENDOR_*`` code; 0 is vendor-neutral.
    label_column_map
        Pulseq label state indices filling the three ADC label columns:
        0 SLC, 1 PHS, 2 REP, 3 AVG, 4 SEG, 5 SET, 6 ECO, 7 PAR, 8 LIN, 9 ACQ.
    cache_ext
        Extension of the cache file, dot included.
    verify_signature
        Refuse a file whose contents do not match the signature it carries. A
        file carrying none is read either way.

    Returns
    -------
    Path
        The cache file.

    Raises
    ------
    ValueError
        If a file of the chain cannot be read, verified or segmented.
    OSError
        If no cache was written.
    """
    seq_path = Path(seq_path)
    target = cache_path(seq_path, cache_ext)
    target.unlink(missing_ok=True)
    require("convert_libraries")(
        _payload(seq_path, verify_signature, fov_offset),
        str(seq_path),
        *_scanner(system),
        int(vendor),
        list(label_column_map),
        cache_ext,
    )
    if not target.is_file():
        raise OSError(f"no cache was written for {seq_path}")
    return target


def summary(
    seq_path: Path | str,
    system: pp.Opts,
    *,
    cache_ext: str | None = None,
    label_column_map: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    """Return the segmentation of a sequence: subsequences, segments and readouts.

    Each subsequence lists its unique RF definitions under ``rf``: the
    bandwidth at half the spectral peak, the number of bands, each band's
    offset from the carrier and the widest band's bandwidth, all in Hz, as
    ``pypulseqpp.calc_rf_bandwidth`` measures them.

    With ``cache_ext``, the cache beside the file is loaded instead of the
    chain being read and segmented again; this build loads only vendor-neutral
    caches, and only when the size recorded in the cache matches the file.

    Raises
    ------
    ValueError
        If the file cannot be read or the cache cannot be loaded.
    """
    seq_path = Path(seq_path)
    if cache_ext is None:
        return require("summary_from_libraries")(
            _payload(seq_path, verify_signature=False),
            *_scanner(system),
            list(label_column_map),
        )
    return require("summary_from_cache")(
        str(cache_path(seq_path, cache_ext)), seq_path.stat().st_size
    )


def prescribe(sequence: pp.Sequence, fov_offset: Sequence[float]) -> pp.Sequence:
    """Move a logical-frame sequence to a prescribed field-of-view centre, in place.

    An offset along the slice axis becomes an RF frequency, an in-plane offset
    an RF and ADC phase, each referenced to the excitation it follows; blocks
    labelled ``NOPOS`` are exempt. The gradient area is counted from the
    sequence's first block.

    Parameters
    ----------
    sequence
        One file of a chain, as designed.
    fov_offset
        Translation along the logical readout, phase and slice axes, in metres.

    Returns
    -------
    pypulseqpp.Sequence
        ``sequence`` itself.

    Raises
    ------
    ValueError
        If ``fov_offset`` is not three values.
    """
    shift = tuple(float(v) for v in fov_offset)
    if len(shift) != 3:
        raise ValueError(f"fov_offset takes three values, got {len(shift)}")
    if any(shift):
        pp.TransformFOV(translation=shift).apply_to_sequence(sequence, in_place=True)
    return sequence


def _payload(
    seq_path: Path, verify_signature: bool, fov_offset: Sequence[float] | None = None
) -> list[dict[str, Any]]:
    """Read the chain and return each file's libraries, in play order, prescribed to ``fov_offset``.

    A file the reader refuses raises ``ValueError``, whatever the reader
    itself raised.
    """
    try:
        chain_read = read_chain(seq_path, verify=verify_signature)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    payload = []
    for _, sequence in chain_read:
        if fov_offset is not None:
            prescribe(sequence, fov_offset)
        payload.append(conversion_payload(sequence))
    return payload
