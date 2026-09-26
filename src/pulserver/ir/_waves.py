"""Where a playout holds a cache's rotated waves, and what it loads into them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._accelerators import require
from ._convert import cache_path


@dataclass(frozen=True)
class WaveBudget:
    """What a playout's waveform memory affords the rotated waves.

    A property of the playout, not of the sequence: the C library's
    ``pulseg_wave_budget``.

    Attributes
    ----------
    max_samples
        Samples each gradient axis holds for rotated waves.
    raster_us
        The playout's gradient raster, in µs per sample.
    load_us_per_sample
        Time to sample and load one sample on one axis, in µs; 0 leaves the
        loading unchecked.
    headroom
        Share of a segment instance's duration that loading the next one may
        take.

    Raises
    ------
    ValueError
        If the raster or the headroom is not positive, or the memory or the
        load rate is negative.
    """

    max_samples: int
    raster_us: float
    load_us_per_sample: float = 0.0
    headroom: float = 0.5

    def __post_init__(self) -> None:
        if self.raster_us <= 0.0 or self.headroom <= 0.0:
            raise ValueError("the raster and the headroom must be positive")
        if self.max_samples < 0 or self.load_us_per_sample < 0.0:
            raise ValueError("the memory and the load rate cannot be negative")


def plan_waves(
    seq_path: Path | str, budget: WaveBudget, cache_ext: str = ".pseg"
) -> dict[str, Any]:
    """Lay out the rotated waves of the cache beside a sequence file in a playout's waveform memory.

    The layout is the C library's, computed from the definitions alone, as
    both stages of a playout compute it. Every wave is held at once where
    that fits the budget's memory on each gradient axis; otherwise each
    segment position that plays waves holds two slots, and the n-th instance
    of a segment in the scan plays the slots of half ``n % 2``, loaded while
    the instance before it plays. With a load rate, that loading is checked
    against the headroom times the duration of the instance before, for
    every instance but the first.

    Returns
    -------
    dict of str to Any
        - ``mode``: ``"none"``, ``"resident"`` or ``"streamed"``;
        - ``samples``, ``resident_samples``, ``streamed_samples``: per axis,
          what the mode holds, every wave at once, and two slots per
          position;
        - ``waves``: per subsequence, the region of each wave, for a resident
          layout;
        - ``slots``: per segment, the regions of each position's two halves,
          for a streamed layout; ``samples`` is 0 where the position plays no
          wave;
        - ``loading_checked``, ``least_spare_us``, ``tightest``: whether the
          loading was checked, the least spare time over segment instances in
          µs, and the subsequence and execution-stream position of the
          instance it is found at.

        A region holds ``samples`` samples on each axis it drives, from
        ``offset`` in that axis's memory, -1 on another, sampled at the
        centres of the raster intervals from ``start_us``, in µs from the
        block's start, as :func:`sample_wave` samples them.

    Raises
    ------
    ValueError
        If the cache cannot be loaded, if neither layout fits, or if a
        segment instance cannot be loaded in time.
    """
    seq_path = Path(seq_path)
    return require("plan_waves_from_cache")(
        str(cache_path(seq_path, cache_ext)),
        seq_path.stat().st_size,
        budget.max_samples,
        budget.raster_us,
        budget.load_us_per_sample,
        budget.headroom,
    )


def sample_wave(
    seq_path: Path | str,
    wave: tuple[int, int],
    region: dict[str, Any],
    budget: WaveBudget,
    cache_ext: str = ".pseg",
) -> np.ndarray:
    """Return a rotated wave as a playout loads it into a region, normalised to unit peak.

    ``wave`` is the subsequence and the wave's index there, as
    :func:`~pulserver.ir.play` reports it, and ``region`` one that
    :func:`plan_waves` lays out. The wave is sampled at the centres of the
    region's ``samples`` raster intervals from its ``start_us``, linear
    between the points :func:`~pulserver.ir.play` plays it through and zero
    outside them.

    Returns
    -------
    ndarray
        ``(3, samples)``, along x, y and z; zero on an axis the wave does not
        drive.

    Raises
    ------
    ValueError
        If the cache cannot be loaded or holds no such wave.
    """
    seq_path = Path(seq_path)
    sample = require("sample_wave_from_cache")
    subsequence, index = wave
    return np.stack(
        [
            sample(
                str(cache_path(seq_path, cache_ext)),
                seq_path.stat().st_size,
                subsequence,
                index,
                axis,
                region["start_us"],
                budget.raster_us,
                region["samples"],
            )
            for axis in range(3)
        ]
    )
