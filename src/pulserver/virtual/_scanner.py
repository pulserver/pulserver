"""The virtual interpreter: an IR cache played block by block, and a phantom acquired with it."""

from __future__ import annotations

__all__ = ["acquire", "trajectory"]

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseqpp as pp

from .. import ir
from ._phantom import Phantom

_EXCITATION = 1
_REFOCUSING = 2
_SATURATION = 4


@dataclass(frozen=True)
class _Readout:
    kspace: np.ndarray
    phase: np.ndarray
    excited: bool
    #: Seconds of free precession each sample has accrued, as refocused.
    precession: np.ndarray
    #: Each species' longitudinal magnetization when the readout's excitation
    #: tipped it, as a fraction of equilibrium.
    longitudinal: np.ndarray


def trajectory(
    seq_path: Path | str,
    cache_ext: str = ".pseg",
    *,
    rotation: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return the k-space location of every ADC sample the cache beside a sequence file plays.

    One ``(3, samples)`` array per readout, in play order, in 1/m along the
    physical axes. The locations are integrated from the gradients the cache
    plays (:func:`pulserver.ir.play`), along the logical axes, each block's
    own rotation included, turned, except in blocks labelled ``NOROT``, by
    the prescription's ``rotation`` from logical to physical axes, a
    reflection included, as :func:`pulserver.ir.check` turns them. Under the
    default identity the physical axes are the logical ones. An excitation
    returns k to zero, and a refocusing pulse negates it, at the RF centre
    the cache records (``rf_center_us``); each file of a chain starts from
    zero.
    """
    return [readout.kspace for readout in _play(Path(seq_path), cache_ext, rotation)]


def acquire(
    seq_path: Path | str,
    phantom: Phantom,
    cache_ext: str = ".pseg",
    *,
    rotation: np.ndarray | None = None,
    field_t: float | None = None,
    off_resonance_hz: float = 0.0,
) -> list[np.ndarray]:
    """Return the samples the cache beside a sequence file acquires of a phantom.

    One ``(coils, samples)`` complex64 array per readout, in play order, of
    the phantom as it lies in the physical frame, sampled along the
    :func:`trajectory` the cache plays under the prescription's ``rotation``
    and multiplied by ``exp(i theta)``, where the receiver phase ``theta`` is
    the ADC phase offset at the ADC's start, advancing at its frequency
    offset, plus its phase modulation, as the playout demodulates.

    Each chemical shift of the phantom precesses at its frequency from the
    scanner's centre frequency: the shift resolved at the field ``field_t``,
    in T, with pypulseqpp's default gamma, plus ``off_resonance_hz``. It
    accrues its phase from each excitation and a refocusing pulse negates
    what it has accrued. Excitation and refocusing pulses act ideally and on
    every shift alike; a saturation pulse scales the longitudinal
    magnetization of each shift by the z component the pulse the cache plays
    leaves at that frequency, in pypulseqpp's Bloch simulation, and the next
    excitation tips what remains. No other pulse acts, and a readout before
    the first excitation of its file acquires zeros. The signal model is that
    of :doc:`/explanations/virtual-scanner`.

    Raises
    ------
    ValueError
        If the phantom has a chemical shift and ``field_t`` is not given, or
        a saturation pulse plays under a gradient.
    """
    shifts = phantom.shifts_ppm
    if field_t is None and any(shifts):
        raise ValueError("a phantom with a chemical shift is scanned at a field_t")
    per_ppm = 0.0 if field_t is None else 1e-6 * pp.Opts().gamma * field_t
    frequencies = np.array([per_ppm * s + off_resonance_hz for s in shifts])
    samples = []
    for readout in _play(Path(seq_path), cache_ext, rotation, frequencies):
        if not readout.excited:
            samples.append(
                np.zeros((phantom.coils, readout.kspace.shape[1]), dtype=np.complex64)
            )
            continue
        signal = sum(
            m
            * phantom.kspace(readout.kspace, s)
            * np.exp(-2j * math.pi * f * readout.precession)
            for s, f, m in zip(shifts, frequencies, readout.longitudinal, strict=True)
        )
        samples.append((signal * np.exp(1j * readout.phase)).astype(np.complex64))
    return samples


def _play(
    seq_path: Path,
    cache_ext: str,
    prescription: np.ndarray | None,
    frequencies_hz: np.ndarray | None = None,
) -> Iterator[_Readout]:
    played = ir.play(seq_path, cache_ext, waveforms=True)
    identity = np.eye(3)
    turn = identity if prescription is None else np.asarray(prescription, float)
    frequencies = np.zeros(0) if frequencies_hz is None else frequencies_hz
    span = played["gradient_span"]
    corners = played["gradient_time_us"].astype(float)
    values = played["gradient_waveform_hz_per_m"].astype(float)
    modulation = played["adc_phase_modulation_rad"].astype(float)
    modulated = played["adc_modulation_span"]
    subsequence = -1
    for block in range(played["duration_us"].size):
        if played["subsequence"][block] != subsequence:
            subsequence = played["subsequence"][block]
            unbroken = np.zeros(3)
            origin = None
            reference = 0.0
            start_us = 0.0
            precession_origin_us = 0.0
            longitudinal = np.ones(frequencies.size)
            tipped = longitudinal
        rotation = identity if played["norot"][block] else turn
        waves = [
            (corners[slice(*span[block, axis])], values[slice(*span[block, axis])])
            for axis in range(3)
        ]
        use = played["rf_use"][block]
        if played["rf_amp_hz"][block] != 0.0 and use in (_EXCITATION, _REFOCUSING):
            centre = float(played["rf_center_us"][block])
            at_centre = _k(unbroken, rotation, waves, np.array([centre]))[:, 0]
            phase = float(played["rf_phase_rad"][block]) + 2.0 * math.pi * float(
                played["rf_freq_hz"][block]
            ) * 1e-6 * (centre - float(played["rf_delay_us"][block]))
            if use == _EXCITATION:
                origin = at_centre
                reference = -phase - 0.5 * math.pi
                precession_origin_us = start_us + centre
                tipped = longitudinal
                longitudinal = np.ones(frequencies.size)
            elif origin is not None:
                origin = 2.0 * at_centre - origin
                reference = -2.0 * phase - reference
                precession_origin_us = 2.0 * (start_us + centre) - precession_origin_us
        elif played["rf_amp_hz"][block] != 0.0 and use == _SATURATION:
            longitudinal = longitudinal * _saturation(played, block, waves, frequencies)
        if played["adc"][block]:
            opened = 1e-3 * float(played["adc_dwell_ns"][block])
            since = opened * (np.arange(played["adc_samples"][block]) + 0.5)
            sampled_us = float(played["adc_delay_us"][block]) + since
            k = _k(unbroken, rotation, waves, sampled_us)
            receiver = (
                float(played["adc_phase_rad"][block])
                + 2.0 * math.pi * float(played["adc_freq_hz"][block]) * 1e-6 * since
            )
            if modulated[block, 1] > modulated[block, 0]:
                receiver = receiver + modulation[slice(*modulated[block])]
            yield _Readout(
                kspace=k - (origin[:, None] if origin is not None else 0.0),
                phase=reference + receiver,
                excited=origin is not None,
                precession=1e-6 * (start_us + sampled_us - precession_origin_us),
                longitudinal=tipped,
            )
        unbroken = _k(
            unbroken, rotation, waves, np.array([float(played["duration_us"][block])])
        )[:, 0]
        start_us += float(played["duration_us"][block])


def _saturation(
    played: dict[str, np.ndarray],
    block: int,
    waves: list[tuple[np.ndarray, np.ndarray]],
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Return the z component the saturation pulse of ``block`` leaves at each frequency, from +z.

    The pulse is simulated in the frame of its own frequency, with the
    channels of a pTx pulse at unit, in-phase sensitivity.

    Raises
    ------
    ValueError
        If a gradient plays during the pulse, which then saturates a band in
        space rather than a species.
    """
    start, stop = played["rf_span"][block]
    channels = max(int(played["rf_channels"][block]), 1)
    times = played["rf_time_us"][start:stop].astype(float).reshape(channels, -1)[0]
    if any(
        np.any(np.interp(times, corners, gradient, left=0.0, right=0.0))
        for corners, gradient in waves
        if corners.size
    ):
        raise ValueError(
            f"the saturation pulse of block {block} plays under a gradient, "
            "which saturates a band in space the phantom's species cannot hold"
        )
    b1 = played["rf_waveform_hz"][start:stop].astype(complex)
    b1 = b1.reshape(channels, -1).sum(axis=0)
    b1, step_us = _on_a_raster(times - float(played["rf_delay_us"][block]), b1)
    detuning = frequencies_hz - float(played["rf_freq_hz"][block])
    return pp.sim_bloch(b1, detuning[:, None], 1e-6 * step_us)[:, 2]


def _on_a_raster(times_us: np.ndarray, b1: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an RF pulse's samples on a uniform raster, and the raster in µs.

    ``times_us`` are from the pulse's start. Samples at the middles of equal
    intervals from the start, as a pulse on the RF raster holds them, are
    returned as they are. The points of a time shape, a block pulse's two
    corners among them, are joined linearly and sampled at the middles of
    1 µs intervals, or of the shape's shortest step where that is shorter.
    """
    steps = np.diff(times_us)
    if not steps.size:
        return b1, 2.0 * float(times_us[0])
    if np.allclose(steps, steps[0]) and np.isclose(times_us[0], 0.5 * steps[0]):
        return b1, float(steps[0])
    step = min(1.0, float(steps[steps > 0].min()))
    count = max(1, round((times_us[-1] - times_us[0]) / step))
    grid = times_us[0] + step * (np.arange(count) + 0.5)
    return np.interp(grid, times_us, b1.real) + 1j * np.interp(
        grid, times_us, b1.imag
    ), step


def _k(
    start: np.ndarray,
    rotation: np.ndarray,
    waves: list[tuple[np.ndarray, np.ndarray]],
    times_us: np.ndarray,
) -> np.ndarray:
    """Return k at times from a block's start: ``start`` plus the rotated gradient areas."""
    areas = np.stack([_area(t, g, times_us) for t, g in waves])
    return start[:, None] + 1e-6 * (rotation @ areas)


def _area(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Return the integral of the piecewise-linear waveform through ``(times, values)`` up to each of ``at``.

    Zero before the first corner and constant after the last.
    """
    at = np.asarray(at, dtype=float)
    if times.size == 0:
        return np.zeros(at.shape)
    steps = np.diff(times)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * steps)]
    )
    inside = (at > times[0]) & (at < times[-1])
    area = np.where(at >= times[-1], cumulative[-1], 0.0)
    j = np.searchsorted(times, at[inside], side="right") - 1
    elapsed = at[inside] - times[j]
    width = np.where(steps[j] > 0.0, steps[j], 1.0)
    slope = (values[j + 1] - values[j]) / width
    area[inside] = cumulative[j] + values[j] * elapsed + 0.5 * slope * elapsed**2
    return area
