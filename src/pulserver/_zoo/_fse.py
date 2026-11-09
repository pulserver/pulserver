"""Fast spin echo on the scanner: SAR burst limits, and the RF layout of trains that differ in length and TR."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pypulseqpp as pp

from pulserver.design import RfLayout

from ._evaluation import rf_layout


def sar_burst(app: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``app``, its main sequence setting ``EnableSarBurstMode``.

    The definition asks the scanner to cost the scan under its SAR burst
    limits, which a long refocusing train at a short TR reaches first; the
    plugin declares the same to the console with
    ``ConfigKey.ENABLE_SAR_BURST_MODE``.
    """

    @functools.wraps(app)
    def design(*args: Any, **kwargs: Any) -> Any:
        designed = app(*args, **kwargs)
        main = designed if isinstance(designed, pp.Sequence) else designed[-1]
        main.set_definition(key="EnableSarBurstMode", value=1)
        return designed

    return design


def echo_train_layout(
    train: pp.Sequence,
    lengths: Sequence[int],
    times: Sequence[float],
    peak_deg: float | None = None,
) -> RfLayout:
    """Return the RF layout of the train of an FSE scan that plays the most RF power.

    ``train`` is one train as long as the longest, which an evaluation
    designs: its excitation, its first refocusing pulse and the rest of the
    refocusing pulses, its startup echoes among them. Shot ``i`` plays
    ``lengths[i]`` acquired echoes in ``times[i]`` s; dummy and reference
    trains play shot 0's. Each shot plays the train's excitation and first
    pulse as designed, each later pulse of its own length at the largest
    amplitude of the train's later pulses, or of ``peak_deg`` where that is
    larger, and those past its length at zero, as the scan does; no train's
    energy is underestimated wherever its angles lie below that peak. The
    layout is the shot whose energy over its TR is the largest, which bounds
    the mean power over any run of whole trains, and its period is that TR.
    """
    base = rf_layout(train, scaled=False, period=float(times[0]))
    instances = base.instances
    definitions = instances.definitions
    definition = np.asarray(instances.definition)
    amplitude = np.asarray(instances.amplitude, dtype=float)
    refocusing = np.flatnonzero(
        [definitions[d].use == "refocusing" for d in definition]
    )
    startup = len(refocusing) - max(lengths)
    later = refocusing[1:]
    top = float(amplitude[later].max()) if later.size else 0.0
    if peak_deg is not None and later.size:
        top = max(top, peak_deg / definitions[definition[later[0]]].flip_deg)
    energy = np.array([_energy(d) for d in definitions])[definition]

    best, played, period = -1.0, amplitude, float(times[0])
    for length, time in zip(lengths, times, strict=True):
        shot = amplitude.copy()
        shot[later] = top
        shot[refocusing[startup + length :]] = 0.0
        power = float(energy @ shot**2) / time
        if power > best:
            best, played, period = power, shot, float(time)
    kept = np.flatnonzero(played != 0.0)
    return RfLayout(
        pp.RfInstances(definitions, definition[kept], played[kept]),
        tuple(None for _ in kept),
        period,
    )


def _energy(definition: pp.RfDefinition) -> float:
    """Return the RF energy of one instance at amplitude one, in Hz^2 s."""
    if definition.time.size < 2:
        return 0.0
    power = (np.abs(definition.waveform) ** 2).sum(0)
    return float(definition.peak_hz**2 * np.trapezoid(power, definition.time))
