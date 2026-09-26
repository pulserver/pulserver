"""The cache beside a sequence file played on isochromats, with pypulseqpp's Bloch simulation."""

from __future__ import annotations

__all__ = ["simulate"]

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp

from .. import ir


def simulate(
    seq_path: Path | str,
    isochromats: pp.Isochromats,
    cache_ext: str = ".pseg",
    *,
    rotation: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return what each coil receives at every ADC sample the cache beside a sequence file plays on isochromats.

    One ``(coils, samples)`` complex64 array per readout, in play order,
    demodulated. The blocks the cache plays (:func:`pulserver.ir.playout`)
    advance the magnetization of ``isochromats``, whose positions are along
    the physical axes, from where it stands, across the files of a chain as
    one scan. Each block's gradients, its own rotation in them, are turned by
    the prescription's ``rotation`` from logical to physical axes, a
    reflection included, except in blocks labelled ``NOROT``. Its RF pulse
    and ADC play as :meth:`pypulseqpp.Isochromats.play` plays RF and ADC
    events, with the frequency and phase offsets the playout sets.

    :doc:`/explanations/virtual-scanner` states the signal model.
    """
    played = ir.playout(Path(seq_path), waveforms=True, cache_ext=cache_ext)["blocks"]
    turn = None if rotation is None else np.asarray(rotation, dtype=float)
    readouts = []
    for block in range(played["duration_us"].size):
        adc = _adc(played, block)
        signal = isochromats.play(
            1e-6 * float(played["duration_us"][block]),
            gradients=_gradients(played, block),
            rotation=turn if played["rotate"][block] else None,
            rf=_rf(played, block),
            adc=adc,
        )
        if adc is not None:
            readouts.append(signal.astype(np.complex64))
    return readouts


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


def _rf(played: dict, block: int) -> SimpleNamespace | None:
    """Return the block's RF pulse as an RF event, timed from the pulse's start; None where it plays none."""
    start, stop = played["rf_span"][block]
    if stop <= start or played["rf_amp_hz"][block] == 0.0:
        return None
    delay_us = float(played["rf_delay_us"][block])
    return SimpleNamespace(
        signal=played["rf_waveform_hz"][start:stop].astype(complex),
        t=1e-6 * (played["rf_time_us"][start:stop].astype(float) - delay_us),
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
