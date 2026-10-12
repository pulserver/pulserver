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

#: How many raster steps of slew two renderings of a gradient may differ by and
#: still be the same gradient. A recording holds a value per raster point, and
#: the two renderings need not step on the same one.
_STEPS_OF_SLEW = 3.0


def gradient_tolerance_mt_per_m(
    max_slew_t_per_m_per_s: float, raster_us: float
) -> float:
    """Return the gradient tolerance in mT/m: the amplitude slewed through in three raster steps.

    Parameters
    ----------
    max_slew_t_per_m_per_s
        Maximum slew rate, in T/m/s.
    raster_us
        Gradient raster time, in µs.
    """
    # A slew in T/m/s over a raster in us is mT/m once per thousand.
    return _STEPS_OF_SLEW * max_slew_t_per_m_per_s * raster_us * 1e-3


@dataclass(frozen=True)
class ChannelAgreement:
    """How far apart one channel's two renderings are.

    Attributes
    ----------
    channel
        Which channel: a gradient axis ``gx``, ``gy`` or ``gz``, or a transmit
        channel ``rho`` or ``theta``.
    peak
        The scale the difference is read against: the larger of the two
        renderings' peak amplitudes in mT/m for a gradient, the sequence's
        peak magnitude for ``rho``, and π for ``theta``.
    largest_difference
        The largest absolute difference in mT/m for a gradient, the root mean
        square difference of ``rho`` as a percentage of its peak, and the
        largest phase difference of ``theta`` in rad.
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
        Subtracted from the reference's gradient times before comparing, in µs.
    rf_shift_us
        The same for the transmit channels, in µs.
    magnitude_scale
        Hz per recorded count of the transmit magnitude, which is compared by
        shape; 1 without a transmit comparison.
    agrees
        Whether every channel agreed.
    """

    reference: str
    channels: tuple[ChannelAgreement, ...]
    shift_us: float = 0.0
    rf_shift_us: float = 0.0
    magnitude_scale: float = 1.0

    @property
    def agrees(self) -> bool:
        """Whether every channel agreed."""
        return all(channel.agrees for channel in self.channels)

    def __str__(self) -> str:
        verdict = "agrees with" if self.agrees else "DIFFERS from"
        lines = [f"the sequence {verdict} the {self.reference} waveforms"]
        if self.shift_us or self.rf_shift_us:
            lines.append(
                f"  reference shifted by {self.shift_us:.0f} us, "
                f"its transmit by {self.rf_shift_us:.0f} us"
            )
        if self.magnitude_scale != 1.0:
            lines.append(
                f"  transmit magnitude compared by shape, at "
                f"{self.magnitude_scale:.6g} Hz per recorded count"
            )
        for channel in self.channels:
            mark = " " if channel.agrees else "*"
            lines.append(
                f" {mark}{channel.channel:6s} largest difference "
                f"{channel.largest_difference:.4g} of {channel.peak:.4g} "
                f"({100 * channel.relative_difference:.2f}%)"
            )
        return "\n".join(lines)


def compare_shape(
    asked_time_us: NDArray[np.float64],
    asked: NDArray[np.float64],
    played_time_us: NDArray[np.float64],
    played: NDArray[np.float64],
    *,
    channel: str,
    tolerance_percent: float,
) -> tuple[ChannelAgreement, float]:
    """Compare two renderings of a channel whose scale is unknown, by shape.

    The recorded transmit magnitude is in the converter's own counts, and the
    recording does not carry the peak transmit field that converts them to Hz.
    The recording is scaled to the peak the sequence asks for, and the scale is
    returned.

    Parameters
    ----------
    asked_time_us, asked
        Sample times in µs and amplitudes the sequence asks for.
    played_time_us, played
        Sample times in µs and amplitudes of the recording, in counts. The
        sequence is interpolated at these times, zero outside its extent.
    channel
        Name recorded in the result.
    tolerance_percent
        Largest root mean square difference, as a percentage of the peak.

    Returns
    -------
    tuple of (ChannelAgreement, float)
        The agreement, whose ``largest_difference`` is the root mean square
        difference as a percentage of the peak, and the scale applied to the
        recording. Without samples on either side, an agreement and a scale
        of 1.
    """
    if not played_time_us.size or not asked_time_us.size:
        return (
            ChannelAgreement(
                channel=channel, peak=0.0, largest_difference=0.0, agrees=True
            ),
            1.0,
        )
    resampled = np.interp(played_time_us, asked_time_us, asked, left=0.0, right=0.0)
    peak_asked = float(np.max(np.abs(resampled)))
    peak_played = float(np.max(np.abs(played)))
    scale = peak_asked / peak_played if peak_played else 1.0
    difference = resampled - played * scale
    error = (
        float(np.sqrt(np.mean(difference**2))) / peak_asked * 100.0
        if peak_asked
        else 0.0
    )
    return (
        ChannelAgreement(
            channel=channel,
            peak=peak_asked,
            largest_difference=error,
            agrees=error <= tolerance_percent,
        ),
        scale,
    )


def compare_gradients(
    asked: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    played: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    reference: str,
    shift_us: float = 0.0,
    tolerance_mt_per_m: float = 0.05,
) -> Comparison:
    """Compare two renderings of the gradients, at the sample times of the second.

    Both are given as ``{axis: (time_us, amplitude_mt_per_m)}`` for ``gx``,
    ``gy`` and ``gz``. The first is interpolated linearly at the second's
    sample times, zero outside its extent.

    Parameters
    ----------
    asked
        What the sequence asks for.
    played
        What is being checked against.
    reference
        Names what ``played`` is, for :attr:`Comparison.reference`.
    shift_us
        Subtracted from the times of ``played`` before comparing, in µs.
    tolerance_mt_per_m
        The largest difference that still counts as agreement, in mT/m.

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
    """Return a sequence's gradient corners per axis, in µs and mT/m, unrounded to any raster."""
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
