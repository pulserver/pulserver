"""Segmented playout of a cache over a backend that records what each stage sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .._accelerators import require
from ._convert import WaveBudget, _held, cache_path


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
        (``NumGainCalibrationReadouts``), whole repetitions of it.
    tour
        Whether it first plays the pTx tour of :func:`prescan_tour`, each
        visit once: ``"played"`` acquires none of the tour's readouts,
        ``"acquired"`` acquires them and adds them to ``readouts``.
    """

    subsequence: int = 0
    readouts: int = 0
    tour: Literal["none", "played", "acquired"] = "none"


_TOURS = {"none": 0, "played": 1, "acquired": 2}


def playout(
    seq_path: Path | str,
    budget: WaveBudget | None = None,
    *,
    waveforms: bool = False,
    pulses: bool = True,
    prescan: Prescan | None = None,
    cache_ext: str = ".pseg",
) -> dict[str, Any]:
    """Play the cache beside a sequence file as a segmented playout plays it, recording both stages.

    The stages are the C library's, ``pulseg_playout_prepare`` and
    ``pulseg_playout_scan``, over a backend that plays nothing: it keeps what
    the first stage prepares at each segment position, a model of the
    waveform memory the waves are loaded into, and the registers the scan
    loop sets on each block, with the samples its wave reads from that
    memory when it is set. The waves are held as :func:`plan_waves` lays
    them out for ``budget``, or, without one, as the cache lays them out.

    With ``prescan``, the scan loop plays that receive-gain calibration
    instead. It waits for no trigger, drives no digital output, plays at
    zero every gradient whose amplitude varies across repetitions, and plays
    an RF pulse whose frequency offset varies across repetitions at their
    mean, ``rf_freq_mean_hz`` of :func:`play`.

    Parameters
    ----------
    seq_path : Path | str
        The first file of the chain; the cache is the file beside it.
    budget : WaveBudget, default=None
        The waveform memory the waves are held in; None takes the cache's own
        layout.
    waveforms : bool, default=False
        Also return what each block plays, under the keys of :func:`play`.
    pulses : bool, default=True
        With ``waveforms``, include the RF pulses.
    prescan : Prescan, default=None
        Play this receive-gain calibration instead of the scan.
    cache_ext : str, default='.pseg'
        Extension of the cache, dot included.

    Returns
    -------
    dict of str to Any
        - ``mode``, ``memory_samples``, ``slots``: how :func:`plan_waves`
          holds the waves, the samples per axis it reserves, and the slots
          of a position's ring;
        - ``positions``: per prepared segment position, in segment order,
          ``segment`` and ``position``; where the position plays no wave, its
          gradient events as corners at unit amplitude, ``event_time_us`` from
          the block's start and ``event_shape``, with ``event_span``
          ``(positions, 3, 2)`` into them; where it does, ``slot_offset``
          ``(positions, slots, 3)``, ``slot_samples`` and ``slot_start_us``:
          its ring of slots, or, with offsets -1, the span every resident wave
          it plays covers; and the RF pulse it prepares, at unit amplitude,
          ``rf_time_us`` from the block's start and ``rf_waveform_hz``, with
          ``rf_span`` ``(positions, 2)`` into them;
        - ``blocks``: per played block, in play order, ``subsequence``,
          ``segment``, ``position``, ``instance`` (instances of its segment
          played before its own), ``slot`` (of its position's ring, -1 unless
          the waves are streamed), ``visit`` (of the prescan's tour, -1
          outside it), ``rotate`` (0 where NOROT keeps the prescription
          rotation out), ``await_trigger``, ``first_position`` (the
          execution-stream position of its instance's first block) and
          ``duration_us``; the registers ``rf_amp_hz``, ``rf_phase_rad``,
          ``rf_freq_hz``, ``rf_shim``, ``rf_use``, ``gradient_hz_per_m``
          ``(blocks, 3)``, ``adc``, ``adc_freq_hz``, ``adc_phase_rad`` and
          ``digitalout``, ``gradient_variable``, where the amplitude of
          that gradient varies across repetitions, and ``rf_freq_variable``,
          where the frequency offset of its RF pulse does; and its wave, ``wave`` (-1
          without one) at ``wave_amp_hz_per_m``, from the region
          ``wave_offset``, ``wave_samples`` and ``wave_start_us``, with the
          samples it reads there in ``wave_read``, ``wave_read_span``
          ``(blocks, 3, 2)`` into them. With ``waveforms``, also what each
          block plays, keyed as :func:`play` keys it: the RF pulse and
          readout its position prepares, the pulse at the block's amplitude,
          the readout's phase modulation, and the gradients: the events its
          position prepares at the block's amplitudes, or the samples its wave
          reads at the centres of the raster intervals, the first and last
          held over the half intervals at its two ends; without ``pulses``,
          all but the RF pulse, which its position holds;
        - ``instances`` and ``loads``: the segment instances played and the
          loads into waveform memory;
        - ``overwrites``: loads that wrote over memory the instance in play
          was reading, and ``unloaded``: samples a block read where nothing
          was loaded. Both are 0 where every block plays what was loaded for
          it;
        - ``rf_shims``: per subsequence, each RF shim's complex channel
          weights, which a block's ``rf_shim`` indexes.

    Raises
    ------
    ValueError
        If the cache cannot be loaded, if the waves do not fit the budget or
        cannot be loaded in time, or if the prescan's subsequence does not
        exist.
    """
    seq_path = Path(seq_path)
    calibration = (
        (-1, 0, 0)
        if prescan is None
        else (prescan.subsequence, prescan.readouts, _TOURS[prescan.tour])
    )
    return require("playout_from_cache")(
        str(cache_path(seq_path, cache_ext)),
        seq_path.stat().st_size,
        _held(budget),
        calibration,
        waveforms,
        pulses,
    )


def prescan_tour(seq_path: Path | str, cache_ext: str = ".pseg") -> dict[str, Any]:
    """List the pTx prescan tour of the cache beside a sequence file.

    One visit per segment instance holding the largest drive one channel of a
    parallel-transmit RF definition receives in its subsequence: the RF
    amplitude times the channel's shim magnitude, or the amplitude alone for
    a pulse with a waveform per channel. An instance that holds several is
    listed once. A sequence without parallel-transmit RF has no visit.

    Returns
    -------
    dict of str to numpy.ndarray
        Per visit, by subsequence and then stream position: ``subsequence``,
        ``segment`` (global), ``first_position`` and ``num_blocks`` of its
        instance, ``rf_definition`` (local to the subsequence), ``channels``
        whose largest drive it plays, the largest of them ``drive_hz``,
        ``duration_us``, ``rf_pulses`` and ``readouts``: those the scan
        acquires that NOPRESCAN does not exempt.
    """
    seq_path = Path(seq_path)
    return require("prescan_tour_from_cache")(
        str(cache_path(seq_path, cache_ext)), seq_path.stat().st_size
    )
