"""Conversion of a Pulseq sequence into the scanner's segmented binary IR."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pypulseqpp as pp

from .._accelerators import require
from ..mrd._sequence import read_chain
from ._checks import SarRatio
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
    sar_ratios: Sequence[SarRatio] | None = None,
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
    sar_ratios
        One per file of the chain, as :func:`sar_ratios` returns them, written
        into each subsequence of the cache; zero when None.

    Returns
    -------
    Path
        The cache file.

    Raises
    ------
    ValueError
        If a file of the chain cannot be read, verified or segmented, or
        ``sar_ratios`` does not give one per file.
    OSError
        If no cache was written.
    """
    seq_path = Path(seq_path)
    target = cache_path(seq_path, cache_ext)
    target.unlink(missing_ok=True)
    payload = _payload(seq_path, system, verify_signature, fov_offset)
    if sar_ratios is not None:
        if len(sar_ratios) != len(payload):
            raise ValueError(
                f"expected one SAR ratio per file of the chain, {len(payload)}; "
                f"got {len(sar_ratios)}"
            )
        for libraries, ratio in zip(payload, sar_ratios, strict=True):
            libraries["reserved"]["vop_sar_ratio"] = float(ratio.local_sar)
            libraries["reserved"]["vop_global_sar_ratio"] = float(ratio.global_sar)
    require("convert_libraries")(
        payload,
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

    Each subsequence lists its unique RF definitions under ``rf``: the flip
    angle in degrees at the largest amplitude the definition plays, as
    ``pypulseqpp.Sequence.rf_flip_angles`` gives it; and the bandwidth at half
    the spectral peak, the number of bands, each band's offset from the
    carrier and the widest band's bandwidth, all in Hz, as
    ``pypulseqpp.calc_rf_bandwidth`` measures them; and ``b1sq_integral_s``,
    the integral of the squared envelope scaled to unit peak, in s:
    ``pypulseqpp.calc_rf_power``'s energy over its peak power, both summed
    over a dynamic pTx pulse's channels.
    ``vop_sar_ratio`` and ``vop_global_sar_ratio`` are those the cache was
    written with, zero when the chain is read and segmented again.

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
            _payload(seq_path, system, verify_signature=False),
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
    sequence's first block. A block that carries a rotation extension is moved
    by the gradients it plays: those it draws, turned by that rotation.

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
        pp.TransformFOV(translation=shift, through_rotation=True).apply_to_sequence(
            sequence, in_place=True
        )
    return sequence


def play(
    seq_path: Path | str, cache_ext: str = ".pseg", *, waveforms: bool = False
) -> dict[str, Any]:
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
        - ``rf_use``: the ``PULSEG_RF_USE_*`` code of the RF event, 1 for an
          excitation and 2 for a refocusing pulse; a pulse the file leaves
          unlabelled takes the use pypulseqpp detects when the chain is read,
          and an event a cache carries without one is a refocusing pulse at a
          flip angle of 162 to 198 degrees and an excitation otherwise; 0
          without RF;
        - ``rf_delay_us``: RF delay from the block's start, in µs;
        - ``rf_channels``: the transmit channels the RF waveform holds, one
          after another over one time base for a dynamic pTx pulse; 0 without
          RF;
        - ``gradient_hz_per_m``: ``(blocks, 3)``, the amplitude of each
          gradient event along x, y and z; an arbitrary gradient's shape
          carries its own sign;
        - ``rotation``: ``(blocks, 3, 3)``, the block's rotation, identity
          without one;
        - ``norot``, ``nopos``: the block's NOROT and NOPOS flags;
        - ``adc``, ``adc_freq_hz``, ``adc_phase_rad``: whether the block
          acquires, and its frequency and phase offsets;
        - ``adc_delay_us``, ``adc_dwell_ns``, ``adc_samples``: the ADC delay
          from the block's start, dwell time and sample count; 0 without ADC;
        - ``trid``: the TRID group in force, 0 when ungrouped.

        With ``waveforms``, also:

        - ``rf_center_us``: the time of the RF centre the design records, from
          the block's start, in µs; NaN without RF;
        - ``gradient_time_us``, ``gradient_waveform_hz_per_m``: the corners of
          every played gradient, concatenated in play order: their times from
          the block's start, in µs, and the gradient there, the instance's
          amplitude times the shape the instance plays. A waveform on the
          gradient raster holds its end values over the half raster
          intervals before its first sample and after its last;
        - ``gradient_span``: ``(blocks, 3, 2)``, the start and stop of each
          block's gradient along x, y and z in those arrays; empty without
          one;
        - ``adc_phase_modulation_rad``: the phase modulation of every played
          readout, one phase per sample in radians, concatenated in play
          order; the receiver phase of a sample is the ADC phase offset, plus
          its frequency offset times the time since the ADC's start, plus
          this;
        - ``adc_modulation_span``: ``(blocks, 2)``, the start and stop of each
          block's modulation in that array; empty without one.

    Raises
    ------
    ValueError
        If the cache cannot be loaded.
    """
    seq_path = Path(seq_path)
    return require("play_cache")(
        str(cache_path(seq_path, cache_ext)), seq_path.stat().st_size, waveforms
    )


def _payload(
    seq_path: Path,
    system: pp.Opts,
    verify_signature: bool,
    fov_offset: Sequence[float] | None = None,
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
        payload.append(conversion_payload(sequence, system))
    return payload
