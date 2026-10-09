"""A virtual scan played against a scan clock, with the sound of its gradients."""

from __future__ import annotations

__all__ = ["SAMPLE_RATE", "Chunk", "Scan"]

import math
import queue
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseqpp as pp

from ._girf import Girf
from ._motion import RigidMotion
from ._tissue import Tissue

#: MATLAB Pulseq's audio sample rate, the default of ``pypulseqpp.gradient_sound``, in Hz.
SAMPLE_RATE = 44100.0

# Audio samples the loudest sample of a scan is looked for in at once.
_PIECE = 1 << 16

# How much longer than the spans simulated so far a span is taken to last
# when the clock's start is planned.
_MARGIN = 1.1

# Spans of each kind whose simulation times estimate the next ones'.
_RECENT = 8

# Wall-clock time, in s, between reports of a scan's preparation.
_REPORT = 0.5


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
    """The cache beside a sequence file acquired of a phantom's tissue by the Fourier engine, against a scan clock.

    The blocks are acquired as :class:`~pulserver.virtual.FourierPlayer`
    acquires them.

    Parameters
    ----------
    seq_path
        The first file of the sequence's chain, beside its cache.
    tissue
        The phantom's tissue.
    cache_ext
        The extension of the cache beside each file.
    rotation
        ``(3, 3)`` rotation of the prescription from logical to physical axes,
        a reflection included; the identity by default.
    centre
        ``(3,)`` the prescription's centre along the physical axes, in m; the
        isocentre by default.
    device
        Where the Fourier engine runs, as
        :class:`~pulserver.virtual.FourierPlayer` takes it.
    motion, girf
        How the subject moves and the gradients' impulse response, as
        :class:`~pulserver.virtual.FourierPlayer` takes them.
    """

    def __init__(
        self,
        seq_path: Path | str,
        tissue: Tissue,
        cache_ext: str = ".pseg",
        *,
        rotation: np.ndarray | None = None,
        centre: np.ndarray | None = None,
        device: str | None = None,
        motion: RigidMotion | None = None,
        girf: Girf | None = None,
    ) -> None:
        from ._fourier import FourierPlayer

        self._player = FourierPlayer(
            seq_path,
            tissue,
            cache_ext,
            rotation=rotation,
            centre=centre,
            device=device,
            motion=motion,
            girf=girf,
        )
        self._played = self._player.played
        self._turn = None if rotation is None else np.asarray(rotation, dtype=float)
        durations = 1e-6 * self._played["duration_us"].astype(float)
        self._starts = np.concatenate([[0.0], np.cumsum(durations)])
        # Readouts before each block: one in every block that acquires.
        reads = (self._played["adc"] != 0).astype(np.int64)
        self._read_before = np.concatenate([[0], np.cumsum(reads)])

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
        preparing: Callable[[float | None], None] | None = None,
        heard: Callable[[Chunk], None] | None = None,
    ) -> Iterator[Chunk]:
        """Play the scan in spans of whole blocks; yield each once released.

        The spans are simulated in a thread of their own, ahead of their
        release, a stretch of them at a time: a stretch ends at the first
        block, ``length`` or more after its start, after which the samples
        acquired since the start of the scan pass a multiple of ``2**18``, so
        that each costs about one transform of the images; a stretch is
        released in spans. The sound is :func:`pypulseqpp.gradient_sound` of the
        gradients along the physical axes, as ``Sequence.sound`` makes it of a
        design: the spans' sounds, joined, are the sound of the whole scan,
        ``floor(duration * sample_rate) + 1`` samples scaled so that the
        loudest is 0.95.

        Parameters
        ----------
        length
            The scan time a span lasts at least, in s, unless it ends the
            scan's last stretch.
        speed
            How many times as fast as a scanner the scan is played. Its clock
            starts once the simulation, at the rate it has run, stays ahead of
            the clock to the end of the scan, and each span is yielded once the
            clock has passed its end. A stretch not yet simulated by its start
            holds the clock until it is. As fast as the stretches are
            simulated when None.
        sound
            Whether the spans carry their sound.
        sample_rate
            Audio sample rate, in Hz.
        channel_weights
            Weights of the x, y and z axes in the sound.
        preparing
            Called about twice a second until the clock starts, at a speed,
            with the wall-clock time left before it does, in s, or None before
            a stretch that acquires has been simulated. An exception
            it raises ends the scan, and is raised where the spans are taken.
        heard
            Called with each span as its sound starts to play: at a speed,
            once the clock reaches the span's start, a span before the span is
            yielded; without one, just before it is yielded.

        Raises
        ------
        ValueError
            If ``length`` or ``speed`` is not positive.
        """
        if not length > 0.0:
            raise ValueError(f"a span lasts a positive time, not {length} s")
        if speed is not None and not speed > 0.0:
            raise ValueError(f"a scan is played at a positive speed, not {speed}")
        stretches = self._stretches(length)
        ready: queue.SimpleQueue = queue.SimpleQueue()
        halt = threading.Event()
        worker = threading.Thread(
            target=self._simulate,
            args=(stretches, length, ready, halt, sound, sample_rate, channel_weights),
            name="pulserver-scan",
            daemon=True,
        )
        worker.start()
        try:
            if speed is None:
                for _ in stretches:
                    for span in _simulated(ready)[0]:
                        if heard is not None:
                            heard(span)
                        yield span
            else:
                samples = [self._adc_samples(*stretch) for stretch in stretches]
                bounds = [
                    (float(self._starts[first]), float(self._starts[last]))
                    for first, last in stretches
                ]
                clock = _Clock(bounds, samples, speed)
                yield from clock.release(ready, preparing, heard)
        finally:
            halt.set()
            worker.join()

    def _stretches(self, length: float) -> list[tuple[int, int]]:
        """Return each stretch's first block and the block after its last."""
        blocks = self._starts.size - 1
        stretches = []
        first = 0
        while first < blocks:
            last = self._stretch_end(first, length)
            stretches.append((first, last))
            first = last
        return stretches

    def _pieces(self, first: int, last: int, length: float) -> list[tuple[int, int]]:
        """Return the spans of a stretch: each at least ``length`` long, the stretch's tail joined to the span before it."""
        pieces = []
        at = first
        while at < last:
            end = int(np.searchsorted(self._starts, self._starts[at] + length))
            end = min(max(end, at + 1), last)
            while end < last and self._starts[end] - self._starts[at] < length:
                end += 1
            if self._starts[last] - self._starts[end] < length:
                end = last
            pieces.append((at, end))
            at = end
        return pieces

    def _simulate(
        self,
        stretches: list[tuple[int, int]],
        length: float,
        ready: queue.SimpleQueue,
        halt: threading.Event,
        sound: bool,
        sample_rate: float,
        channel_weights: Sequence[float],
    ) -> None:
        """Put the spans of each stretch on ``ready`` once simulated, with the time it was; an error in their place."""
        blocks = self._starts.size - 1
        try:
            peak = self._loudest(sample_rate, channel_weights) if sound else 0.0
            for first, last in stretches:
                if halt.is_set():
                    return
                start, stop = float(self._starts[first]), float(self._starts[last])
                audio = (
                    self._sound(
                        start, stop, last == blocks, peak, sample_rate, channel_weights
                    )
                    if sound
                    else np.zeros((2, 0))
                )
                readouts = self._readouts(first, last)
                read = self._read_before - self._read_before[first]
                heard = self._samples(start, stop, False, sample_rate)[0]
                spans = []
                for a, b in self._pieces(first, last, length):
                    begin, end = float(self._starts[a]), float(self._starts[b])
                    at, count = self._samples(begin, end, b == blocks, sample_rate)
                    at -= heard
                    spans.append(
                        Chunk(
                            begin,
                            end,
                            readouts[read[a] : read[b]],
                            audio[:, at : at + count] if sound else audio,
                        )
                    )
                ready.put((tuple(spans), time.monotonic()))
        except Exception as error:  # raised again where the spans are taken
            ready.put((error, time.monotonic()))

    def _adc_samples(self, first: int, last: int) -> int:
        """Return how many ADC samples the blocks from ``first`` to before ``last`` acquire."""
        acquired = self._played["adc"][first:last].astype(bool)
        return int(self._played["adc_samples"][first:last][acquired].sum())

    def _stretch_end(self, first: int, length: float) -> int:
        """Return the block after the last of the stretch that starts at block ``first``."""
        blocks = self._starts.size - 1
        last = first + 1
        while last < blocks and self._starts[last] - self._starts[first] < length:
            last += 1
        return self._player.boundary(last)

    def _readouts(self, first: int, last: int) -> tuple[np.ndarray, ...]:
        return tuple(self._player.readouts(first, last))

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


def _simulated(
    ready: queue.SimpleQueue, timeout: float | None = None
) -> tuple[Chunk, float]:
    """Return the next span simulated and the time it was; raise the simulation's error in its place.

    Raises
    ------
    queue.Empty
        If no span is simulated within ``timeout`` s.
    """
    item, done = ready.get(timeout=timeout)
    if isinstance(item, Exception):
        raise item
    return item, done


class _Clock:
    """A scan clock that starts once the simulation will stay ahead of it.

    A stretch's simulation time is estimated from the stretches simulated
    before it: per ADC sample where it acquires, per second of scan time where
    it does not, each the median of the latest stretches of its kind. A
    stretch is due on the clock at its start, where its first span begins.
    """

    def __init__(
        self, bounds: list[tuple[float, float]], samples: list[int], speed: float
    ) -> None:
        self._starts = [start for start, _ in bounds]
        self._samples = samples
        self._speed = speed
        self._durations = [stop - start for start, stop in bounds]
        self._spent: list[float] = []

    def release(
        self,
        ready: queue.SimpleQueue,
        preparing: Callable[[float | None], None] | None,
        heard: Callable[[Chunk], None] | None = None,
    ) -> Iterator[Chunk]:
        """Yield each span once the clock has passed its end, having handed it to ``heard`` once the clock reached its start.

        The clock starts once the simulation is estimated to stay ahead of it
        to the end of the scan; a stretch simulated after its start on the
        clock holds the clock there until it is.
        """
        simulated: deque[tuple[Chunk, ...]] = deque()
        last = reported = time.monotonic()
        while True:
            now = time.monotonic()
            lead = self._lead(now - last)
            if lead is not None and lead <= 0.0:
                break
            if preparing is not None and now >= reported:
                preparing(lead)
                reported = now + _REPORT
            try:
                spans, done = _simulated(ready, timeout=_REPORT)
            except queue.Empty:
                continue
            self._spent.append(done - last)
            last = done
            simulated.append(spans)
        started = time.monotonic()
        for start in self._starts:
            if simulated:
                spans = simulated.popleft()
            else:
                spans, _ = _simulated(ready)
                started += max(0.0, time.monotonic() - started - start / self._speed)
            for span in spans:
                if heard is not None:
                    self._until(started + span.start / self._speed)
                    heard(span)
                self._until(started + span.stop / self._speed)
                yield span

    @staticmethod
    def _until(moment: float) -> None:
        time.sleep(max(0.0, moment - time.monotonic()))

    def _lead(self, running: float) -> float | None:
        """Return how long the clock must wait to stay behind the simulation, in s.

        ``running`` is the time the stretch being simulated has taken so far,
        in s. None where a stretch still to come acquires and none that
        acquires has been simulated, or where none has been.
        """
        per_sample = self._rate(acquiring=True)
        per_second = self._rate(acquiring=False)
        lead = 0.0
        remaining = -running
        for k in range(len(self._spent), len(self._starts)):
            if self._samples[k] > 0:
                rate, size = per_sample, self._samples[k]
            else:
                rate, size = per_second, self._durations[k]
            if rate is None:
                return None
            remaining = max(remaining + _MARGIN * rate * size, 0.0)
            lead = max(lead, remaining - self._starts[k] / self._speed)
        return lead

    def _rate(self, *, acquiring: bool) -> float | None:
        """Return the median simulation time of the latest stretches of a kind; None without one.

        Per ADC sample for stretches that acquire, and per second of scan time
        for those that do not: until one that does not has been simulated, per
        second of the stretches that acquire, which bounds it.
        """
        for kind in (acquiring, True):
            rates = [
                spent / (self._samples[k] if acquiring else self._durations[k])
                for k, spent in enumerate(self._spent)
                if (self._samples[k] > 0) == kind and self._durations[k] > 0.0
            ]
            if rates:
                return statistics.median(rates[-_RECENT:])
        return None


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
