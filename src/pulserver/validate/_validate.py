"""Checking a sequence against what a machine plays, or against its own cache."""

from __future__ import annotations

__all__ = ["VENDORS", "validate"]

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pypulseqpp as pp

from ._compare import Comparison, compare_gradients, gradients_of_sequence
from ._xml import read_waveform_xml

#: The machines a recording can be read from. A sequence checked without naming
#: one is checked against its own cache, which establishes that the conversion
#: kept the sequence, not that a machine plays it.
VENDORS = ("ge",)


def validate(
    seq_path: Path | str,
    *,
    vendor: str | None = None,
    played: Path | str | None = None,
    shift_us: float = 0.0,
    tolerance_mt_per_m: float = 0.05,
    cache_ext: str = ".pseg",
    system: Any = None,
) -> Comparison:
    """Check that a sequence plays as it was written.

    Reads the gradients the sequence asks for, and compares them against a
    second rendering: a recording of a machine playing it, where one is given
    or can be made, and otherwise the waveforms its own cache holds.

    The two answer different questions, and :attr:`Comparison.reference` says
    which was answered. A recording establishes that the machine plays the
    sequence. The cache establishes only that the conversion kept it -- a
    conversion and a playout that are wrong in the same way agree.

    Parameters
    ----------
    seq_path
        The sequence file.
    vendor
        Which machine ``played`` came from, one of :data:`VENDORS`. ``None``
        compares against the cache and ignores ``played``.
    played
        A recording to compare against. Absent, and with a vendor named, one is
        asked for from that vendor's tooling if it is installed here.
    shift_us
        Moves the recording in time before comparing. A machine drives its
        transmit and gradient channels on separate timelines, so a recording
        holds them offset by that machine's own constant.
    tolerance_mt_per_m
        The largest difference that still counts as agreement.
    cache_ext
        Extension of the cache read when comparing against it.
    system
        ``pypulseqpp.Opts`` the cache is converted under, when one has to be.

    Returns
    -------
    Comparison

    Raises
    ------
    ValueError
        If ``vendor`` is not one this knows.
    NotImplementedError
        If ``vendor`` is known but nothing here reads its recordings.
    FileNotFoundError
        If no recording is given and none can be made.
    """
    seq_path = Path(seq_path)
    sequence = pp.io.read(seq_path)
    asked = gradients_of_sequence(sequence)

    if vendor is None:
        return compare_gradients(
            asked,
            _gradients_of_cache(seq_path, cache_ext, system),
            reference="ir",
            tolerance_mt_per_m=tolerance_mt_per_m,
        )

    if vendor not in VENDORS:
        raise ValueError(
            f"{vendor!r} is not a machine this reads recordings from; "
            f"known: {', '.join(VENDORS)}"
        )
    if vendor != "ge":
        raise NotImplementedError(f"nothing here reads what a {vendor} machine records")

    if played is None:
        played = _record(seq_path)
    recording = read_waveform_xml(played)
    return compare_gradients(
        asked,
        {axis: recording.gradient_mt_per_m(axis) for axis in ("gx", "gy", "gz")},
        reference="played",
        shift_us=shift_us,
        tolerance_mt_per_m=tolerance_mt_per_m,
    )


def _record(seq_path: Path) -> Path:
    """Ask the vendor's tooling to play the sequence and record it.

    Raises
    ------
    FileNotFoundError
        If that tooling is not installed here.
    """
    try:
        from pulserver_gehc import record_playout
    except ImportError as error:
        raise FileNotFoundError(
            "no recording was given and the tooling that makes one is not "
            "installed here; pass one, or check against the cache by naming "
            "no vendor"
        ) from error
    return Path(record_playout(seq_path))


def _gradients_of_cache(
    seq_path: Path, cache_ext: str, system: Any
) -> dict[str, tuple[Any, Any]]:
    """Return the gradients a sequence's cache holds, in microseconds and mT/m.

    The cache holds each block's corners from that block's own start, so they
    are laid end to end over the durations the blocks are played for.
    """
    from .. import ir
    from ._compare import _HZ_PER_M_IN_MT_PER_M

    cache = ir.cache_path(seq_path, cache_ext)
    if cache.is_file():
        played = ir.play(seq_path, cache_ext, waveforms=True)
    else:
        # Converted beside a copy, not beside the sequence: checking a file is
        # not a reason to leave a cache in the directory it was found in.
        with tempfile.TemporaryDirectory(prefix="pulserver-validate-") as elsewhere:
            copy = Path(elsewhere) / seq_path.name
            shutil.copy(seq_path, copy)
            ir.convert(
                copy, pp.Opts() if system is None else system, cache_ext=cache_ext
            )
            played = ir.play(copy, cache_ext, waveforms=True)

    duration_us = np.asarray(played["duration_us"], dtype=np.float64)
    starts = np.concatenate(([0.0], np.cumsum(duration_us)))[:-1]
    corner_time = np.asarray(played["gradient_time_us"], dtype=np.float64)
    corner_amplitude = np.asarray(
        played["gradient_waveform_hz_per_m"], dtype=np.float64
    )
    span = np.asarray(played["gradient_span"], dtype=np.int64)

    out: dict[str, tuple[Any, Any]] = {}
    for at, axis in enumerate(("gx", "gy", "gz")):
        times: list[Any] = []
        amplitudes: list[Any] = []
        for block in range(span.shape[0]) if span.size else ():
            first, last = int(span[block, at, 0]), int(span[block, at, 1])
            if last <= first:
                continue
            times.append(corner_time[first:last] + starts[block])
            amplitudes.append(corner_amplitude[first:last])
        if times:
            out[axis] = (
                np.concatenate(times),
                np.concatenate(amplitudes) * _HZ_PER_M_IN_MT_PER_M,
            )
        else:
            out[axis] = (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
    return out
