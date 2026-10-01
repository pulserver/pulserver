"""The cache beside a sequence file played on isochromats."""

from __future__ import annotations

__all__ = ["simulate"]

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import ir
from ._isochromats import MEMORY, Isochromats, Repetitions
from ._repeats import Run, runs

#: The ``tolerance`` the console and the ``pulserver scan`` command play runs
#: of repetitions and read ADC windows by a transform to, relative to the sum
#: of the magnitudes of the terms each sample sums.
TOLERANCE = 1e-4


def simulate(
    seq_path: Path | str,
    isochromats: Isochromats,
    cache_ext: str = ".pseg",
    *,
    rotation: np.ndarray | None = None,
    default_shim: np.ndarray | None = None,
    tolerance: float = 0.0,
) -> list[np.ndarray]:
    """Return what each coil receives at every ADC sample the cache beside a sequence file plays on isochromats.

    One ``(coils, samples)`` complex64 array per readout, in play order,
    demodulated. The blocks the cache plays (:func:`pulserver.ir.playout`)
    advance the magnetization of ``isochromats``, whose positions are along
    the physical axes, from where it stands, across the files of a chain as
    one scan. Each block's gradients, its own rotation in them, are turned by
    the prescription's ``rotation`` from logical to physical axes, a
    reflection included, except in blocks labelled ``NOROT``. Its RF pulse
    and ADC play as :meth:`Isochromats.play` plays RF and ADC events, with
    the frequency and phase offsets the playout sets. With
    ``default_shim``, the channel weights of a coil of several transmit
    channels, a single-channel pulse plays on every channel, weighted by its
    block's RF shim or, without one, by ``default_shim``.

    Runs of repetitions of blocks that differ only in their phase offsets,
    phase encodings and turned readouts play from each isochromat's map over
    one repetition, as :meth:`Isochromats.repetitions` plays them to within
    ``tolerance``; at zero, their samples are those of the blocks played one
    by one, to rounding. Isochromats that move or diffuse play every block
    one by one. The other blocks read their ADC windows as
    :meth:`Isochromats.play` reads them to within ``tolerance``.

    :doc:`/explanations/virtual-scanner` states the signal model.
    """
    player = Player(
        seq_path,
        isochromats,
        cache_ext,
        rotation=rotation,
        default_shim=default_shim,
        tolerance=tolerance,
    )
    return list(player.readouts(0, player.blocks))


class Player:
    """The blocks the cache beside a sequence file plays, played on isochromats in turn, as :func:`simulate` plays them.

    Attributes
    ----------
    played
        The playout's blocks, as :func:`pulserver.ir.playout` records them.
    runs
        The runs of repetitions played from each isochromat's map, by first
        block.
    """

    def __init__(
        self,
        seq_path: Path | str,
        isochromats: Isochromats,
        cache_ext: str = ".pseg",
        *,
        rotation: np.ndarray | None = None,
        default_shim: np.ndarray | None = None,
        tolerance: float = 0.0,
    ) -> None:
        seq_path = Path(seq_path)
        playout = ir.playout(seq_path, waveforms=True, cache_ext=cache_ext)
        self.played = playout["blocks"]
        self._turn = None if rotation is None else np.asarray(rotation, dtype=float)
        self._drive = _drive(playout, default_shim)
        self._isochromats = isochromats
        self._tolerance = tolerance
        self.runs = (
            []
            if isochromats.moving
            else runs(
                self.played,
                self._turn,
                windows=_windows(isochromats),
                rounded=tolerance > 0.0,
            )
        )
        self._firsts = np.array([run.first for run in self.runs], dtype=int)
        # The run being played, its repetitions and the next to be played.
        self._playing: tuple[Run, Repetitions, int] | None = None

    @property
    def blocks(self) -> int:
        """Blocks in the scan."""
        return int(self.played["duration_us"].size)

    def boundary(self, block: int) -> int:
        """Return the first block from ``block`` on that starts a repetition of a run or lies outside every run."""
        run = self._run(block)
        return block if run is None else block + (run.first - block) % run.size

    def readouts(self, first: int, last: int) -> Iterator[np.ndarray]:
        """Play the blocks from ``first`` to before ``last`` and yield each readout, as :func:`simulate` returns them.

        A run's repetitions play whole from each repetition that starts
        within the blocks and ends by ``last``; the rest play block by block.
        """
        block = first
        while block < last:
            run = self._run(block)
            aligned = run is not None and (block - run.first) % run.size == 0
            count = (min(last, run.stop) - block) // run.size if aligned else 0
            if count > 0:
                yield from self._repeated(run, (block - run.first) // run.size, count)
                block += count * run.size
                continue
            self._playing = None
            readout = _played(
                self.played,
                block,
                self._isochromats,
                self._turn,
                self._drive,
                self._tolerance,
            )
            if readout is not None:
                yield readout
            block += 1

    def _run(self, block: int) -> Run | None:
        at = int(np.searchsorted(self._firsts, block, side="right")) - 1
        return self.runs[at] if at >= 0 and block < self.runs[at].stop else None

    def _repeated(self, run: Run, index: int, count: int) -> Iterator[np.ndarray]:
        """Play ``count`` repetitions of ``run`` from repetition ``index``; yield their readouts."""
        playing = self._playing
        if playing is None or playing[0] is not run or playing[2] != index:
            # The maps of the run played before are freed before these are made.
            self._playing = playing = None
            playing = (run, self._repetitions(run, index), index)
        done = index + count == run.count
        self._playing = None if done else (run, playing[1], index + count)
        windows = [
            int(self.played["adc_samples"][block])
            for block in range(run.first, run.first + run.size)
            if self.played["adc"][block]
        ]
        signal = playing[1].play(count).astype(np.complex64)
        if not windows:
            return
        for samples in signal:
            for readout in np.split(samples, np.cumsum(windows)[:-1], axis=1):
                yield np.ascontiguousarray(readout)

    def _repetitions(self, run: Run, index: int) -> Repetitions:
        """Return the repetitions of ``run`` from repetition ``index`` on, from the magnetization where it stands."""
        blocks = []
        for position in range(run.size):
            block = run.first + position
            blocks.append(
                {
                    "duration": 1e-6 * float(self.played["duration_us"][block]),
                    "gradients": _gradients(self.played, block),
                    "rotation": self._turn if self.played["rotate"][block] else None,
                    "rf": _rf(self.played, block, self._drive),
                    "adc": _adc(self.played, block),
                }
            )
        return self._isochromats.repetitions(
            blocks,
            run.phases[index:],
            run.areas[index:],
            adc_phases=run.adc_phases[index:],
            readouts=run.readouts[index:],
            nets=run.nets[index:],
            tolerance=self._tolerance,
        )


def _windows(isochromats: Isochromats) -> int:
    """Return the most ADC windows a repetition may hold for its maps on the isochromats to take at most :data:`MEMORY` bytes; negative where none fits."""
    # Per isochromat, its map and the slot it is played from, in double
    # precision, with its coordinates; per window, the map of the transverse
    # magnetisation at its first sample and the slot's kernel weights,
    # receive factors and phase step.
    per_window = 244 + 16 * isochromats.coils
    return (MEMORY // max(len(isochromats), 1) - 288) // per_window


def _drive(
    playout: dict, default_shim: np.ndarray | None
) -> tuple[list, np.ndarray] | None:
    """Return the playout's RF shims beside a coil's default shim; None without one."""
    if default_shim is None:
        return None
    return playout["rf_shims"], np.asarray(default_shim, dtype=complex).ravel()


def _played(
    played: dict,
    block: int,
    isochromats: Isochromats,
    turn: np.ndarray | None,
    drive: tuple[list, np.ndarray] | None = None,
    tolerance: float = 0.0,
) -> np.ndarray | None:
    """Play one block on the isochromats; return its readout, or None where it has no ADC."""
    adc = _adc(played, block)
    signal = isochromats.play(
        1e-6 * float(played["duration_us"][block]),
        gradients=_gradients(played, block),
        rotation=turn if played["rotate"][block] else None,
        rf=_rf(played, block, drive),
        adc=adc,
        tolerance=tolerance,
    )
    return None if adc is None else signal.astype(np.complex64)


def _gradients(played: dict, block: int) -> list[np.ndarray | None]:
    """Return the block's gradient corners per logical axis: times in s from its start over Hz/m."""
    span = played["gradient_span"][block]
    corners = []
    for axis in range(3):
        where = slice(*span[axis])
        times = 1e-6 * played["gradient_time_us"][where].astype(float)
        values = played["gradient_waveform_hz_per_m"][where].astype(float)
        corners.append(np.array([times, values]) if times.size else None)
    return corners


def _rf(
    played: dict, block: int, drive: tuple[list, np.ndarray] | None = None
) -> SimpleNamespace | None:
    """Return the block's RF pulse as an RF event, timed from the pulse's start; None where it plays none.

    With ``drive``, a single-channel pulse becomes a pTx pulse of the
    default shim's channels, each the pulse times its weight.

    Raises
    ------
    ValueError
        If the block's RF shim weighs another number of channels.
    """
    start, stop = played["rf_span"][block]
    if stop <= start or played["rf_amp_hz"][block] == 0.0:
        return None
    delay_us = float(played["rf_delay_us"][block])
    signal = played["rf_waveform_hz"][start:stop].astype(complex)
    times = 1e-6 * (played["rf_time_us"][start:stop].astype(float) - delay_us)
    if drive is not None and played["rf_channels"][block] <= 1:
        rf_shims, weights = drive
        shim = int(played["rf_shim"][block])
        if shim >= 0:
            shimmed = rf_shims[int(played["subsequence"][block])][shim]
            if shimmed.size != weights.size:
                raise ValueError(
                    f"the RF shim of block {block} weighs {shimmed.size} channels, "
                    f"not the coil's {weights.size}"
                )
            weights = shimmed
        signal = np.outer(weights, signal).ravel()
        times = np.tile(times, weights.size)
    return SimpleNamespace(
        signal=signal,
        t=times,
        delay=1e-6 * delay_us,
        phase_offset=float(played["rf_phase_rad"][block]),
        freq_offset=float(played["rf_freq_hz"][block]),
    )


def _adc(played: dict, block: int) -> SimpleNamespace | None:
    """Return the block's ADC as an ADC event; None where it has none."""
    if not played["adc"][block]:
        return None
    start, stop = played["adc_modulation_span"][block]
    return SimpleNamespace(
        num_samples=int(played["adc_samples"][block]),
        dwell=1e-9 * float(played["adc_dwell_ns"][block]),
        delay=1e-6 * float(played["adc_delay_us"][block]),
        phase_offset=float(played["adc_phase_rad"][block]),
        freq_offset=float(played["adc_freq_hz"][block]),
        phase_modulation=played["adc_phase_modulation_rad"][start:stop].astype(float),
    )
