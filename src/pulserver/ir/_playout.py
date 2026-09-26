"""Both stages of a segmented playout, over a backend that records what each sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._accelerators import require
from ._convert import cache_path
from ._waves import WaveBudget


def playout(
    seq_path: Path | str,
    budget: WaveBudget,
    *,
    prescan: int | None = None,
    prescan_readouts: int = 0,
    cache_ext: str = ".pseg",
) -> dict[str, Any]:
    """Play the cache beside a sequence file as a segmented playout plays it, recording both stages.

    The stages are the C library's, ``pulseg_playout_prepare`` and
    ``pulseg_playout_scan``, over a backend that plays nothing: it keeps what
    the first stage prepares at each segment position, a model of the
    waveform memory the waves are loaded into, and the registers the scan
    loop sets on each block, with the samples its wave reads from that
    memory when it is set.

    With ``prescan``, the scan loop plays the receive-gain calibration of
    that subsequence instead, from its start to the instance that completes
    ``prescan_readouts`` readouts, or as many as its sequence declares. It
    waits for no trigger, drives no digital output, and plays at zero every
    gradient whose amplitude varies across repetitions.

    Returns
    -------
    dict of str to Any
        - ``mode``, ``memory_samples``: how :func:`plan_waves` holds the
          waves, and the samples per axis it reserves;
        - ``positions``: per prepared segment position, in segment order,
          ``segment`` and ``position``; where the position plays no wave, its
          gradient events as corners at unit amplitude, ``event_time_us`` from
          the block's start and ``event_shape``, with ``event_span``
          ``(positions, 3, 2)`` into them; where it does, ``slot_offset``
          ``(positions, 2, 3)``, ``slot_samples`` and ``slot_start_us``: its
          two slots, or, with offsets -1, the span every resident wave it
          plays covers;
        - ``blocks``: per played block, in play order, ``subsequence``,
          ``segment``, ``position``, ``instance`` (instances of its segment
          played before its own), ``half`` (of the streamed slots, -1
          otherwise), ``rotate`` (0 where NOROT keeps the prescription
          rotation out), ``await_trigger``, ``first_position`` (the
          execution-stream position of its instance's first block) and
          ``duration_us``; the registers ``rf_amp_hz``, ``rf_phase_rad``,
          ``rf_freq_hz``, ``rf_shim``, ``gradient_hz_per_m`` ``(blocks, 3)``,
          ``adc``, ``adc_freq_hz``, ``adc_phase_rad`` and ``digitalout``, and
          ``gradient_variable``, where the amplitude of that gradient varies
          across repetitions; and its wave, ``wave`` (-1
          without one) at ``wave_amp_hz_per_m``, from the region
          ``wave_offset``, ``wave_samples`` and ``wave_start_us``, with the
          samples it reads there in ``wave_read``, ``wave_read_span``
          ``(blocks, 3, 2)`` into them;
        - ``instances`` and ``loads``: the segment instances played and the
          loads into waveform memory;
        - ``overwrites``: loads that wrote over memory the instance in play
          was reading, and ``unloaded``: samples a block read where nothing
          was loaded. Both are 0 where every block plays what was loaded for
          it.

    Raises
    ------
    ValueError
        If the cache cannot be loaded, if the waves do not fit the budget or
        cannot be loaded in time, or if the prescan's subsequence does not
        exist.
    """
    seq_path = Path(seq_path)
    return require("playout_from_cache")(
        str(cache_path(seq_path, cache_ext)),
        seq_path.stat().st_size,
        (
            budget.max_samples,
            budget.raster_us,
            budget.load_us_per_sample,
            budget.headroom,
        ),
        (-1 if prescan is None else prescan, prescan_readouts),
    )
