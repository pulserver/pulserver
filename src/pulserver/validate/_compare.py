"""Comparing the waveforms a sequence asks for against the ones a machine played."""

from __future__ import annotations

__all__ = ["ChannelAgreement", "Comparison", "compare_gradients"]

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: Gyromagnetic ratio of the proton, in hertz per tesla.
GAMMA_HZ_PER_T = 42.576e6

#: A gradient in hertz per metre, in millitesla per metre.
_HZ_PER_M_IN_MT_PER_M = 1.0e3 / GAMMA_HZ_PER_T


@dataclass(frozen=True)
class ChannelAgreement:
    """How far apart one channel's two renderings are.

    Attributes
    ----------
    channel
        Which channel, named as :data:`~pulserver.validate.CHANNELS` names it.
    peak
        The larger of the two renderings' peak amplitudes, in the channel's
        unit. The scale the difference is read against.
    largest_difference
        The largest absolute difference between them, in the same unit.
    agrees
        Whether the difference stayed within the tolerance asked for.
    """

    channel: str
    peak: float
    largest_difference: float
    agrees: bool

    @property
    def relative_difference(self) -> float:
        """The largest difference over the peak; 0 where nothing is played."""
        return 0.0 if self.peak == 0.0 else self.largest_difference / self.peak


@dataclass(frozen=True)
class Comparison:
    """What a comparison established, and what it compared against.

    Attributes
    ----------
    reference
        What the sequence was checked against: ``"played"`` for a recording of
        a machine playing it, ``"ir"`` for the waveforms its cache holds.
    channels
        One per channel compared.
    shift_us
        The time the reference was moved by before comparing, in microseconds.
    agrees
        Whether every channel agreed.
    """

    reference: str
    channels: tuple[ChannelAgreement, ...]
    shift_us: float = 0.0

    @property
    def agrees(self) -> bool:
        """Whether every channel agreed."""
        return all(channel.agrees for channel in self.channels)

    def __str__(self) -> str:
        verdict = "agrees with" if self.agrees else "DIFFERS from"
        lines = [f"the sequence {verdict} the {self.reference} waveforms"]
        if self.shift_us:
            lines.append(f"  reference shifted by {self.shift_us:.0f} us")
        for channel in self.channels:
            mark = " " if channel.agrees else "*"
            lines.append(
                f" {mark}{channel.channel:6s} largest difference "
                f"{channel.largest_difference:.4g} of {channel.peak:.4g} "
                f"({100 * channel.relative_difference:.2f}%)"
            )
        return "\n".join(lines)


def compare_gradients(
    asked: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    played: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    reference: str,
    shift_us: float = 0.0,
    tolerance_mt_per_m: float = 0.05,
) -> Comparison:
    """Compare two renderings of the gradients, sampled where the second has samples.

    Both are given as ``(time_us, amplitude_mt_per_m)`` per axis. The first is
    read at the second's sample times, so a difference is read where the
    machine actually stepped rather than on a grid of this function's choosing.

    ``shift_us`` moves the second in time before comparing. The transmit and
    gradient channels of a machine are driven on their own timelines and a
    recording holds them as each was driven, so a comparison that does not
    account for the offset between them reads one against the other at times
    that never coincided.

    Parameters
    ----------
    asked
        What the sequence asks for.
    played
        What is being checked against.
    reference
        Names what ``played`` is, for :attr:`Comparison.reference`.
    shift_us
        Moves ``played`` forward in time by this much before comparing.
    tolerance_mt_per_m
        The largest difference that still counts as agreement.

    Returns
    -------
    Comparison
    """
    channels = []
    for axis in ("gx", "gy", "gz"):
        asked_time, asked_amplitude = asked.get(axis, (_none(), _none()))
        played_time, played_amplitude = played.get(axis, (_none(), _none()))
        at = played_time - shift_us
        resampled = (
            np.interp(at, asked_time, asked_amplitude, left=0.0, right=0.0)
            if asked_time.size
            else np.zeros_like(at)
        )
        difference = (
            float(np.max(np.abs(resampled - played_amplitude)))
            if at.size
            else float(np.max(np.abs(asked_amplitude)))
            if asked_amplitude.size
            else 0.0
        )
        peak = max(
            float(np.max(np.abs(asked_amplitude))) if asked_amplitude.size else 0.0,
            float(np.max(np.abs(played_amplitude))) if played_amplitude.size else 0.0,
        )
        channels.append(
            ChannelAgreement(
                channel=axis,
                peak=peak,
                largest_difference=difference,
                agrees=difference <= tolerance_mt_per_m,
            )
        )
    return Comparison(reference=reference, channels=tuple(channels), shift_us=shift_us)


def gradients_of_sequence(
    sequence: object,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Return a sequence's gradient corners per axis, in microseconds and mT/m.

    The corners are what the sequence asks for, before any machine has rounded
    them to a raster.
    """
    played = sequence.waveforms()
    out: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for axis, corners in zip(("gx", "gy", "gz"), played, strict=False):
        corners = np.asarray(corners, dtype=np.float64)
        if corners.size == 0:
            out[axis] = (_none(), _none())
            continue
        out[axis] = (
            corners[0] * 1e6,
            corners[1] * _HZ_PER_M_IN_MT_PER_M,
        )
    return out


def _none() -> NDArray[np.float64]:
    return np.array([], dtype=np.float64)
