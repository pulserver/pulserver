"""Both stages of a segmented playout, over a backend that records what each sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._accelerators import require
from ._convert import cache_path
from ._waves import WaveBudget


@dataclass(frozen=True)
class Prescan:
    """The receive-gain calibration a playout plays instead of the scan.

    Attributes
    ----------
    subsequence
        The subsequence it plays, from its start.
    readouts
        The readouts after which it ends, with the segment instance that
        completes them; 0 takes the number its sequence declares
        (``NumGainCalibrationReadouts``), at least one.
    """

    subsequence: int = 0
    readouts: int = 0


def playout(
    seq_path: Path | str,
    budget: WaveBudget | None = None,
    *,
    waveforms: bool = False,
    prescan: Prescan | None = None,
    cache_ext: str = ".pseg",
) -> dict[str, Any]:
    """Play the cache beside a sequence file as a segmented playout plays it, recording both stages.

    The stages are the C library's, ``pulseg_playout_prepare`` and
    ``pulseg_playout_scan``, over a backend that plays nothing: it keeps what
    the first stage prepares at each segment position, a model of the
    waveform memory the waves are loaded into, and the registers the scan
    loop sets on each block, with the samples its wave reads from that
    memory when it is set. Without a ``budget``, the playout holds every
    wave at once on the gradient raster of the chain's first file.

    With ``prescan``, the scan loop plays that receive-gain calibration
    instead. It waits for no trigger, drives no digital output, and plays at
    zero every gradient whose amplitude varies across repetitions.

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
          ``rf_freq_hz``, ``rf_shim``, ``rf_use``, ``gradient_hz_per_m``
          ``(blocks, 3)``, ``adc``, ``adc_freq_hz``, ``adc_phase_rad`` and
          ``digitalout``, and ``gradient_variable``, where the amplitude of
          that gradient varies across repetitions; and its wave, ``wave`` (-1
          without one) at ``wave_amp_hz_per_m``, from the region
          ``wave_offset``, ``wave_samples`` and ``wave_start_us``, with the
          samples it reads there in ``wave_read``, ``wave_read_span``
          ``(blocks, 3, 2)`` into them. With ``waveforms``, also what each
          block plays, keyed as :func:`play` keys it: the RF pulse and
          readout its position prepares, the pulse at the block's amplitude,
          the readout's phase modulation, and the gradients: the events its
          position prepares at the block's amplitudes, or the samples its wave
          reads at the centres of the raster intervals, the first and last
          held over the half intervals at its two ends;
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
    held = (
        None
        if budget is None
        else (
            budget.max_samples,
            budget.raster_us,
            budget.load_us_per_sample,
            budget.headroom,
        )
    )
    calibration = (
        (-1, 0) if prescan is None else (prescan.subsequence, prescan.readouts)
    )
    return require("playout_from_cache")(
        str(cache_path(seq_path, cache_ext)),
        seq_path.stat().st_size,
        held,
        calibration,
        waveforms,
    )
