"""The virtual interpreter: an IR cache played block by block, and a phantom acquired with it."""

from __future__ import annotations

__all__ = ["acquire", "trajectory"]

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import ir
from ._phantom import Phantom

_EXCITATION = 1
_REFOCUSING = 2


@dataclass(frozen=True)
class _Readout:
    kspace: np.ndarray
    phase: np.ndarray
    excited: bool


def trajectory(seq_path: Path | str, cache_ext: str = ".pseg") -> list[np.ndarray]:
    """Return the k-space location of every ADC sample the cache beside a sequence file plays.

    One ``(3, samples)`` array per readout, in play order, in 1/m along the
    logical axes. The locations are integrated from the gradients the cache
    plays (:func:`pulserver.ir.play`), rotated by each block's rotation. An
    excitation returns k to zero, and a refocusing pulse negates it, at the
    RF centre the cache records (``rf_center_us``); each file of a chain
    starts from zero.
    """
    return [readout.kspace for readout in _play(Path(seq_path), cache_ext)]


def acquire(
    seq_path: Path | str, phantom: Phantom, cache_ext: str = ".pseg"
) -> list[np.ndarray]:
    """Return the samples the cache beside a sequence file acquires of a phantom.

    One ``(coils, samples)`` complex64 array per readout, in play order,
    multiplied by ``exp(i theta)``, where the receiver phase ``theta`` is the
    ADC phase offset at the ADC's start, advancing at its frequency offset,
    plus its phase modulation, as the playout demodulates. Only excitation and refocusing pulses act on
    the magnetization, and ideally; a readout before the first excitation of
    its file acquires zeros. The signal model is that of
    :doc:`/explanations/virtual-scanner`.
    """
    samples = []
    for readout in _play(Path(seq_path), cache_ext):
        if not readout.excited:
            samples.append(
                np.zeros((phantom.coils, readout.kspace.shape[1]), dtype=np.complex64)
            )
            continue
        signal = phantom.kspace(readout.kspace) * np.exp(1j * readout.phase)
        samples.append(signal.astype(np.complex64))
    return samples


def _play(seq_path: Path, cache_ext: str) -> Iterator[_Readout]:
    played = ir.play(seq_path, cache_ext, waveforms=True)
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
        rotation = played["rotation"][block].astype(float)
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
            elif origin is not None:
                origin = 2.0 * at_centre - origin
                reference = -2.0 * phase - reference
        if played["adc"][block]:
            opened = 1e-3 * float(played["adc_dwell_ns"][block])
            since = opened * (np.arange(played["adc_samples"][block]) + 0.5)
            k = _k(
                unbroken, rotation, waves, float(played["adc_delay_us"][block]) + since
            )
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
            )
        unbroken = _k(
            unbroken, rotation, waves, np.array([float(played["duration_us"][block])])
        )[:, 0]


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
