"""A virtual scan played against a scan clock, with the sound of its gradients."""

from __future__ import annotations

__all__ = ["SAMPLE_RATE", "Chunk", "Scan"]

import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseqpp as pp

from .. import ir
from ._bloch import _gradients, _played

#: MATLAB Pulseq's audio sample rate, the default of ``pypulseqpp.gradient_sound``, in Hz.
SAMPLE_RATE = 44100.0

# Audio samples the loudest sample of a scan is looked for in at once.
_PIECE = 1 << 16


@dataclass(frozen=True, eq=False)
class Chunk:
    """A span of a virtual scan between two block boundaries.

    Attributes
    ----------
    start, stop
        Its bounds, in s of scan time.
    readouts
        What each coil receives in the readouts of its blocks, in play order,
        as :func:`~pulserver.virtual.simulate` returns them.
    sound
        ``(2, n)`` stereo audio of the gradients it plays, sample ``k`` of the
        scan at ``k / sample_rate`` s; ``(2, 0)`` without sound.
    """

    start: float
    stop: float
    readouts: tuple[np.ndarray, ...]
    sound: np.ndarray


class Scan:
    """The cache beside a sequence file played on isochromats block by block, against a scan clock.

    The blocks are those :func:`~pulserver.virtual.simulate` plays, and they
    advance the magnetization of the isochromats from where it stands.

    Parameters
    ----------
    seq_path
        The first file of the sequence's chain, beside its cache.
    isochromats
        The isochromats scanned, positioned along the physical axes.
    cache_ext
        The extension of the cache beside each file.
    rotation
        ``(3, 3)`` rotation of the prescription from logical to physical axes,
        a reflection included; the identity by default.
    """

    def __init__(
        self,
        seq_path: Path | str,
        isochromats: pp.Isochromats,
        cache_ext: str = ".pseg",
        *,
        rotation: np.ndarray | None = None,
    ) -> None:
        self._played = ir.playout(Path(seq_path), waveforms=True, cache_ext=cache_ext)[
            "blocks"
        ]
        self._isochromats = isochromats
        self._turn = None if rotation is None else np.asarray(rotation, dtype=float)
        durations = 1e-6 * self._played["duration_us"].astype(float)
        self._starts = np.concatenate([[0.0], np.cumsum(durations)])

    @property
    def duration(self) -> float:
        """The scan's duration, in s."""
        return float(self._starts[-1])

    def chunks(
        self,
        length: float = 0.1,
        *,
        speed: float | None = None,
        sound: bool = True,
        sample_rate: float = SAMPLE_RATE,
        channel_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> Iterator[Chunk]:
        """Play the scan in spans of whole blocks; yield each once played.

        The sound is :func:`pypulseqpp.gradient_sound` of the gradients along
        the physical axes, as ``Sequence.sound`` makes it of a design: the
        spans' sounds, joined, are the sound of the whole scan,
        ``floor(duration * sample_rate) + 1`` samples scaled so that the
        loudest is 0.95.

        Parameters
        ----------
        length
            The scan time a span lasts at least, in s, unless it ends the scan.
        speed
            How many times as fast as a scanner the scan is played: each span
            is yielded no sooner than ``stop / speed`` s of wall-clock time
            after the first was asked for. As fast as it is computed when
            None.
        sound
            Whether the spans carry their sound.
        sample_rate
            Audio sample rate, in Hz.
        channel_weights
            Weights of the x, y and z axes in the sound.

        Raises
        ------
        ValueError
            If ``length`` or ``speed`` is not positive.
        """
        if not length > 0.0:
            raise ValueError(f"a span lasts a positive time, not {length} s")
        if speed is not None and not speed > 0.0:
            raise ValueError(f"a scan is played at a positive speed, not {speed}")
        blocks = self._starts.size - 1
        peak = self._loudest(sample_rate, channel_weights) if sound else 0.0
        started = time.monotonic()
        first = 0
        while first < blocks:
            last = self._span_end(first, length)
            start, stop = float(self._starts[first]), float(self._starts[last])
            audio = (
                self._sound(
                    start, stop, last == blocks, peak, sample_rate, channel_weights
                )
                if sound
                else np.zeros((2, 0))
            )
            chunk = Chunk(start, stop, self._readouts(first, last), audio)
            if speed is not None:
                time.sleep(max(0.0, started + stop / speed - time.monotonic()))
            yield chunk
            first = last

    def _span_end(self, first: int, length: float) -> int:
        """Return the block after the last of the span that starts at block ``first``."""
        blocks = self._starts.size - 1
        last = first + 1
        while last < blocks and self._starts[last] - self._starts[first] < length:
            last += 1
        return last

    def _readouts(self, first: int, last: int) -> tuple[np.ndarray, ...]:
        played = (
            _played(self._played, block, self._isochromats, self._turn)
            for block in range(first, last)
        )
        return tuple(readout for readout in played if readout is not None)

    def _samples(
        self, start: float, stop: float, final: bool, sample_rate: float
    ) -> tuple[int, int]:
        """Return the first of the audio samples at ``k / sample_rate`` within the span, and their count."""
        dwell = 1.0 / sample_rate
        first = math.ceil(start / dwell)
        after = int(np.floor(stop / dwell)) + 1 if final else math.ceil(stop / dwell)
        return first, max(after - first, 0)

    def _sound(
        self,
        start: float,
        stop: float,
        final: bool,
        peak: float,
        sample_rate: float,
        channel_weights: Sequence[float],
    ) -> np.ndarray:
        first, count = self._samples(start, stop, final, sample_rate)
        if peak <= 0.0:
            return np.zeros((2, count))
        return self._filtered(first, count, peak, sample_rate, channel_weights)

    def _loudest(self, sample_rate: float, channel_weights: Sequence[float]) -> float:
        """Return the largest magnitude of the scan's filtered sound, before scaling."""
        _, total = self._samples(0.0, self.duration, True, sample_rate)
        loudest = 0.0
        for first in range(0, total, _PIECE):
            unscaled = self._filtered(
                first, min(_PIECE, total - first), 0.95, sample_rate, channel_weights
            )
            loudest = max(loudest, float(np.abs(unscaled).max(initial=0.0)))
        return loudest

    def _filtered(
        self,
        first: int,
        count: int,
        peak: float,
        sample_rate: float,
        channel_weights: Sequence[float],
    ) -> np.ndarray:
        # gradient_sound reads the waveforms up to round(sample_rate / 6000)
        # samples beyond each end of the samples it returns.
        margin = math.ceil(sample_rate / 6000.0) + 1
        times = np.arange(first - margin, first + count + margin) * (1.0 / sample_rate)
        return pp.gradient_sound(
            [np.array([times, values]) for values in self._physical(times)],
            count,
            first_sample=first,
            channel_weights=channel_weights,
            sample_rate=sample_rate,
            peak=peak,
        )

    def _physical(self, times: np.ndarray) -> np.ndarray:
        """Return the gradients along the physical axes at ``times`` s of scan time, ``(3, n)``, zero outside the scan."""
        values = np.zeros((3, times.size))
        blocks = np.searchsorted(self._starts, times, side="right") - 1
        inside = (blocks >= 0) & (blocks < self._starts.size - 1)
        for block in np.unique(blocks[inside]):
            at = np.flatnonzero(blocks == block)
            since = times[at] - self._starts[block]
            logical = np.stack(
                [
                    np.zeros(at.size)
                    if corners is None
                    else np.interp(since, corners[0], corners[1], left=0.0, right=0.0)
                    for corners in _gradients(self._played, block)
                ]
            )
            turned = self._turn is not None and self._played["rotate"][block]
            values[:, at] = self._turn @ logical if turned else logical
        return values
