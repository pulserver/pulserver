"""The blocks a cache plays, written as a Pulseq file an external simulator reads."""

from __future__ import annotations

__all__ = ["export"]

import math
import types
from pathlib import Path

import numpy as np
import pypulseqpp as pp

from .. import ir

#: Width of the ramp, in µs, that stands in for a gradient's step from or to
#: zero at its first or last corner: a time shape holds one value per time.
_STEP_US = 1e-2
#: Corner times are compared at this resolution, in µs.
_RESOLUTION_US = 1e-3


def export(
    seq_path: Path | str,
    target: Path | str,
    system: pp.Opts,
    cache_ext: str = ".pseg",
    *,
    rotation: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Write the blocks the cache beside a sequence file plays as one Pulseq 1.4.1 file.

    Every block :func:`pulserver.ir.play` walks, across the files of a chain,
    is one block of ``target`` of the duration it plays, holding what the
    cache plays in it:

    - its RF pulse on the RF raster of ``system``, with the pulse's phase
      offset and its frequency offset from the pulse's start applied to the
      samples, and the channels of a pTx pulse summed, as at unit, in-phase
      sensitivity;
    - its gradients along the physical axes: turned by the block's rotation
      and then, except in a block labelled ``NOROT``, by the prescription's
      ``rotation`` from logical to physical axes, as
      :func:`~pulserver.virtual.trajectory` turns them, each as a time-shaped
      gradient through the corners the cache plays, a step from or to zero
      as a ramp 10 ns wide;
    - its ADC window, without frequency or phase offsets.

    The file holds the standard sections alone: no rotation, label, trigger or
    soft-delay extension, and no RF use, which Pulseq 1.4.1 does not carry.

    Parameters
    ----------
    seq_path
        The first file of a chain, beside its cache.
    target
        Where to write the file.
    system
        The scanner the cache was converted for; its rasters time the file.
    cache_ext
        The extension of the cache.
    rotation
        The prescription's rotation from logical to physical axes; the
        identity by default.

    Returns
    -------
    list of ndarray
        The receiver phase of every readout, in play order, in radians, one
        per sample: the ADC phase offset at the ADC's start, advancing at its
        frequency offset, plus its phase modulation. The playout demodulates by
        it, as :func:`~pulserver.virtual.acquire` does, and ``target`` leaves
        it out.
    """
    played = ir.play(seq_path, cache_ext, waveforms=True)
    turn = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    span = played["gradient_span"]
    corners = played["gradient_time_us"].astype(float)
    values = played["gradient_waveform_hz_per_m"].astype(float)
    modulation = played["adc_phase_modulation_rad"].astype(float)
    modulated = played["adc_modulation_span"]
    raster_us = 1e6 * system.grad_raster_time
    seq = pp.Sequence(system)
    receiver = []
    for block in range(played["duration_us"].size):
        duration_us = float(played["duration_us"][block])
        turned = played["rotation"][block].astype(float)
        if not played["norot"][block]:
            turned = turn @ turned
        waves = [
            (corners[slice(*span[block, axis])], values[slice(*span[block, axis])])
            for axis in range(3)
        ]
        events = _gradients(waves, turned, duration_us, raster_us)
        start, stop = played["rf_span"][block]
        if stop > start:
            events.append(_pulse(played, block, system))
        if played["adc"][block]:
            samples = int(played["adc_samples"][block])
            dwell_us = 1e-3 * float(played["adc_dwell_ns"][block])
            events.append(
                pp.make_adc(
                    samples,
                    dwell=1e-6 * dwell_us,
                    delay=1e-6 * float(played["adc_delay_us"][block]),
                    system=system,
                )
            )
            since = dwell_us * (np.arange(samples) + 0.5)
            phase = (
                float(played["adc_phase_rad"][block])
                + 2.0 * math.pi * float(played["adc_freq_hz"][block]) * 1e-6 * since
            )
            if modulated[block, 1] > modulated[block, 0]:
                phase = phase + modulation[slice(*modulated[block])]
            receiver.append(phase)
        events.append(pp.make_delay(1e-6 * duration_us))
        seq.add_block(*events)
    seq.write_v141(str(target))
    return receiver


def _gradients(
    waves: list[tuple[np.ndarray, np.ndarray]],
    turned: np.ndarray,
    duration_us: float,
    raster_us: float,
) -> list[types.SimpleNamespace]:
    """Return one block's gradients along the physical axes, as time-shaped gradient events.

    ``waves`` are the corners of the block's gradients along its channel
    axes, in µs from the block's start, zero outside them; ``turned`` takes
    those axes to the physical ones. The events share the corners of all
    three, and start and end on the gradient raster.
    """
    played = [(t, g) for t, g in waves if t.size]
    if not played:
        return []
    points = [t for t, _ in played]
    for t, g in played:
        if g[0] != 0.0 and t[0] - _STEP_US >= 0.0:
            points.append(np.array([t[0] - _STEP_US]))
        if g[-1] != 0.0 and t[-1] + _STEP_US <= duration_us:
            points.append(np.array([t[-1] + _STEP_US]))
    grid = _resolved(np.concatenate(points))
    first = math.floor(grid[0] / raster_us + 1e-9) * raster_us
    last = min(math.ceil(grid[-1] / raster_us - 1e-9) * raster_us, duration_us)
    grid = _resolved(np.concatenate([[first], grid, [last]]))
    logical = np.stack(
        [
            np.interp(grid, t, g, left=0.0, right=0.0)
            if t.size
            else np.zeros(grid.size)
            for t, g in waves
        ]
    )
    physical = turned @ logical
    events = []
    for axis, waveform in zip("xyz", physical, strict=True):
        if not np.any(waveform):
            continue
        events.append(
            types.SimpleNamespace(
                type="grad",
                channel=axis,
                waveform=waveform,
                tt=1e-6 * (grid - grid[0]),
                first=float(waveform[0]),
                last=float(waveform[-1]),
                delay=1e-6 * grid[0],
                shape_dur=1e-6 * (grid[-1] - grid[0]),
                area=float(np.trapezoid(waveform, 1e-6 * grid)),
            )
        )
    return events


def _resolved(times_us: np.ndarray) -> np.ndarray:
    return np.unique(np.round(times_us / _RESOLUTION_US)) * _RESOLUTION_US


def _pulse(
    played: dict[str, np.ndarray], block: int, system: pp.Opts
) -> types.SimpleNamespace:
    """Return the RF pulse ``block`` plays as an RF event on the RF raster, its offsets in its samples."""
    start, stop = played["rf_span"][block]
    channels = max(int(played["rf_channels"][block]), 1)
    delay_us = float(played["rf_delay_us"][block])
    times = played["rf_time_us"][start:stop].astype(float).reshape(channels, -1)[0]
    b1 = played["rf_waveform_hz"][start:stop].astype(complex)
    b1 = b1.reshape(channels, -1).sum(axis=0)
    raster_us = 1e6 * system.rf_raster_time
    b1 = _on_raster(times - delay_us, b1, raster_us)
    since = raster_us * (np.arange(b1.size) + 0.5)
    signal = b1 * np.exp(
        1j
        * (
            float(played["rf_phase_rad"][block])
            + 2.0 * math.pi * float(played["rf_freq_hz"][block]) * 1e-6 * since
        )
    )
    return types.SimpleNamespace(
        type="rf",
        signal=signal,
        t=1e-6 * since,
        shape_dur=1e-6 * raster_us * b1.size,
        delay=1e-6 * delay_us,
        center=1e-6 * (float(played["rf_center_us"][block]) - delay_us),
        freq_offset=0.0,
        phase_offset=0.0,
        freq_ppm=0.0,
        phase_ppm=0.0,
        dead_time=system.rf_dead_time,
        ringdown_time=system.rf_ringdown_time,
        use="undefined",
    )


def _on_raster(times_us: np.ndarray, b1: np.ndarray, raster_us: float) -> np.ndarray:
    """Return an RF pulse's samples at the middles of the raster intervals from its start.

    ``times_us`` are from the pulse's start. Samples at the middles of equal
    intervals a whole number of rasters wide are held over each interval, as
    the pulse plays them; the points of a time shape are joined linearly, up
    to the last of them.
    """
    steps = np.diff(times_us)
    if (
        steps.size
        and np.allclose(steps, steps[0])
        and np.isclose(times_us[0], 0.5 * steps[0])
    ):
        held = steps[0] / raster_us
        if np.isclose(held, round(held)) and round(held) >= 1:
            return np.repeat(b1, round(held))
    elif not steps.size and np.isclose(times_us[0], 0.5 * raster_us):
        return b1
    count = max(1, round(float(times_us[-1]) / raster_us))
    grid = raster_us * (np.arange(count) + 0.5)
    return np.interp(grid, times_us, b1.real) + 1j * np.interp(grid, times_us, b1.imag)
