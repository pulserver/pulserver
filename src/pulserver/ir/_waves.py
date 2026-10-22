"""Waves and heaviest stretches of a cache's gradients, as a playout plays them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .._accelerators import require
from ._convert import WaveBudget, _held, cache_path


def plan_waves(
    seq_path: Path | str, budget: WaveBudget | None = None, cache_ext: str = ".pseg"
) -> dict[str, Any]:
    """Lay out the waves of the cache beside a sequence file in a playout's waveform memory.

    With ``budget``, the layout is computed for it here, as :func:`convert`
    computes the one a cache carries; without one, it is the layout the cache
    carries. Every wave is held at once where that fits the budget's memory
    on each gradient axis; otherwise each segment position that plays waves
    holds a ring of ``slots`` slots, and the n-th instance of a segment in the
    scan plays slot ``n % slots``. With a load rate, the loading is checked
    on the playout's timeline scaled by the headroom: the scan starts with
    its first instance's waves loaded, and the waves of each later instance
    are loaded one instance after another, from once the instance
    ``slots - 1`` before it has started, and have to be loaded before it
    starts.

    Parameters
    ----------
    seq_path : Path | str
        The first file of the chain; the cache is the file beside it.
    budget : WaveBudget, default=None
        The waveform memory to lay the waves out for; None reads the layout
        the cache carries.
    cache_ext : str, default='.pseg'
        Extension of the cache, dot included.

    Returns
    -------
    dict of str to Any
        - ``budget``: the budget it is laid out for, as :class:`WaveBudget`
          fields;
        - ``mode``: ``"none"``, ``"resident"`` or ``"streamed"``;
        - ``samples``, ``resident_samples``, ``streamed_samples``: per axis,
          what the mode holds, every wave at once, and the slots of every
          position;
        - ``waves``: per subsequence, the region of each wave, for a resident
          layout;
        - ``slots``: per segment, the ring of regions of each position, for a
          streamed layout; ``samples`` is 0 where the position plays no wave;
        - ``loading_checked``, ``least_spare_us``, ``tightest``: whether the
          loading was checked; the least, over the instances after the first
          up to the first whose loading ends after it starts, of how long
          before its start its loading ends, on the scaled timeline, in µs;
          and the subsequence and execution-stream position of the instance
          it is found at.

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
        str(cache_path(seq_path, cache_ext)), seq_path.stat().st_size, _held(budget)
    )


def sample_wave(
    seq_path: Path | str,
    wave: tuple[int, int],
    region: dict[str, Any],
    budget: WaveBudget,
    cache_ext: str = ".pseg",
) -> np.ndarray:
    """Return a wave as a playout loads it into a region, normalised to unit peak.

    ``wave`` is the subsequence and the wave's index there, as
    :func:`~pulserver.ir.play` reports it, and ``region`` one that
    :func:`plan_waves` lays out. The wave is sampled at the centres of the
    region's ``samples`` raster intervals from its ``start_us``, linear
    between the points :func:`~pulserver.ir.play` plays it through and zero
    outside them.

    Parameters
    ----------
    seq_path : Path | str
        The first file of the chain; the cache is the file beside it.
    wave : tuple[int, int]
        ``(subsequence, index)``.
    region : dict[str, Any]
        One region of the layout :func:`plan_waves` returns.
    budget : WaveBudget
        The budget the layout was planned for; its ``raster_us`` is the sampling
        interval.
    cache_ext : str, default='.pseg'
        Extension of the cache, dot included.

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


_STRETCHES = {"heaviest": 0, "heaviest_repetition": 1}


def gradient_stretch(
    seq_path: Path | str,
    subsequence: int = 0,
    stretch: Literal["heaviest", "heaviest_repetition"] = "heaviest",
    *,
    raster_us: float | None = None,
    cache_ext: str = ".pseg",
) -> dict[str, Any]:
    """Return the gradients of one of a subsequence's stretches.

    The C library's choice (``pulseg_get_gradient_stretch``). A stretch is the
    run of whole segment instances from one instance's start until they last
    the subsequence's hyper-repetition (``hyper_us`` of the summary's
    ``tr_timing``), or 10 s (``PULSEG_STRETCH_US``) where that is shorter, or
    to the subsequence's end where it comes first, each block at its played
    duration. ``"heaviest"`` is, of those lasting that long, the one over
    which the squared gradient summed over the axes integrates to the most,
    the earliest on a tie, and the whole subsequence where none does.
    ``"heaviest_repetition"`` starts at
    the instance holding the start of the repetition of most gradient energy,
    of the repetitions the execution stream holds from its first block. A
    scanner evaluates its gradient-heating model on both, each as if played
    without end.

    Parameters
    ----------
    seq_path : Path | str
        The first file of the chain; the cache is the file beside it.
    subsequence : int, default=0
        Index of the subsequence, in chain order.
    stretch : {'heaviest', 'heaviest_repetition'}, default='heaviest'
        Which stretch.
    raster_us : float, default=None
        Gradient raster in µs for ``samples_hz_per_m``; None omits it.
    cache_ext : str, default='.pseg'
        Extension of the cache, dot included.

    Returns
    -------
    dict of str to Any
        - ``first_position``, ``num_positions``: execution-stream position of
          its first block, and the positions it spans;
        - ``duration_us``: its played duration, in µs;
        - ``energy``: that integral, in (Hz/m)² s;
        - ``time_us``, ``gradient_hz_per_m``: ``(points,)`` and
          ``(points, 3)``, its corner points from its start, along the logical
          axes, linear in between; from zero at 0 where no gradient plays
          there, to zero at its end where none plays up to it;
        - with ``raster_us``, ``samples_hz_per_m``: ``(3, samples)``, the
          gradient at the centres of the raster intervals that cover it.

    Raises
    ------
    ValueError
        If the cache cannot be loaded, holds no such subsequence, or
        ``stretch`` is neither name.
    """
    if stretch not in _STRETCHES:
        raise ValueError(
            f"stretch is 'heaviest' or 'heaviest_repetition', not {stretch!r}"
        )
    seq_path = Path(seq_path)
    return require("gradient_stretch_from_cache")(
        str(cache_path(seq_path, cache_ext)),
        seq_path.stat().st_size,
        subsequence,
        _STRETCHES[stretch],
        0.0 if raster_us is None else raster_us,
    )
