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

    Parameters
    ----------
    seq_path
        Text or binary Pulseq file.
    system
        Limits and rasters the scan is segmented under.
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
        _payload(seq_path, verify_signature),
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


def play(seq_path: Path | str, cache_ext: str = ".pseg") -> dict[str, Any]:
    """Walk the cache beside a sequence file as the scanner's playout does.

    The cache is loaded by the C library a scanner links, and its execution
    stream is walked with that library's cursor: one entry per played block,
    in play order, across the subsequences of the chain. This build loads only
    vendor-neutral caches, and only when the size recorded in the cache
    matches the file.

    Returns
    -------
    dict of str to ndarray
        One entry per played block in each:

        - ``subsequence``, ``segment``: chain file and cache segment indices;
        - ``duration_us``: block duration, in µs;
        - ``rf_amp_hz``, ``rf_freq_hz``, ``rf_phase_rad``: RF amplitude
          (gamma B1) and frequency and phase offsets, with ppm offsets
          resolved at the field strength the cache was converted under; 0
          without RF;
        - ``gradient_hz_per_m``: ``(blocks, 3)``, the amplitude of each
          gradient event along x, y and z; an arbitrary gradient's shape
          carries its own sign;
        - ``rotation``: ``(blocks, 3, 3)``, the block's rotation, identity
          without one;
        - ``norot``, ``nopos``: the block's NOROT and NOPOS flags;
        - ``adc``, ``adc_freq_hz``, ``adc_phase_rad``: whether the block
          acquires, and its frequency and phase offsets;
        - ``trid``: the TRID group in force, 0 when ungrouped.

    Raises
    ------
    ValueError
        If the cache cannot be loaded.
    """
    seq_path = Path(seq_path)
    return require("play_cache")(
        str(cache_path(seq_path, cache_ext)), seq_path.stat().st_size
    )


def _payload(seq_path: Path, verify_signature: bool) -> list[dict[str, Any]]:
    """Read the chain and return each file's libraries, in play order.

    A file the reader refuses raises ``ValueError``, whatever the reader
    itself raised.
    """
    try:
        chain_read = read_chain(seq_path, verify=verify_signature)
    except RuntimeError as failure:
        raise ValueError(f"cannot read {seq_path}: {failure}") from failure
    return [conversion_payload(sequence) for _, sequence in chain_read]
