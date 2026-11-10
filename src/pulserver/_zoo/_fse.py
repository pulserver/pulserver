"""Fast spin echo on the scanner: SAR burst limits, and the RF layout of trains that differ in length and TR."""

import functools
import math
from collections.abc import Callable, Mapping, Sequence
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


def bounding_train(a: Mapping[str, Any], peak_deg: float) -> dict[str, Any]:
    """Return the arguments that design the constant train bounding a train whose angles a prescribed signal fixes.

    A Busse or PD train plays its startup echoes and then its acquired ones
    at angles an inverse extended-phase-graph solve finds, none above
    ``peak_deg``. The train of as many pulses at ``peak_deg`` lasts as long,
    its pulses and gradients being the same but for the angles, and plays at
    least the RF power: an evaluation designs it, and the solve is left to
    the design of the scan. ``a`` holds the arguments of the full design.
    """
    startup = a["busse_startup_echoes"]
    periphery = a["etl_periphery"]
    return {
        "flip_modulation": "constant",
        "refocusing_angle_deg": peak_deg,
        "etl": a["etl"] + startup,
        "etl_periphery": None if periphery is None else periphery + startup,
        "busse_centre_echo": None,
        "busse_effective_te": None,
        "busse_optimize": False,
        "pd_effective_te": None,
        "pd_optimize": False,
    }


def check_busse(a: Mapping[str, Any]) -> None:
    """Refuse a Busse prescription no train realizes, as its design would: the checks that take no solve.

    Raises
    ------
    ValueError
        If the angles are not ``0 < minimum < centre < maximum < 180``
        degrees, the effective TE is not positive, or no acquired echo, or
        not the one asked for, lies after the sixth pulse and before the last
        of every train.
    """
    angles = [a[f"busse_{name}_angle_deg"] for name in ("minimum", "centre", "maximum")]
    if not all(map(math.isfinite, angles)) or not (
        0 < angles[0] < angles[1] < angles[2] < 180
    ):
        raise ValueError(
            "the Busse angles must satisfy 0 < minimum < centre < maximum < 180 degrees"
        )
    startup = a["busse_startup_echoes"]
    shortest = (
        a["etl"] if a["etl_periphery"] is None else min(a["etl"], a["etl_periphery"])
    )
    first, last = max(0, 6 - startup), shortest - 2
    centre = a["busse_centre_echo"]
    target = a["busse_effective_te"]
    if centre is None and target is not None:
        if not math.isfinite(target) or target <= 0:
            raise ValueError("the effective TE must be positive and finite")
        if first > last:
            raise ValueError(
                "no acquired echo lies after the sixth pulse and before the last "
                "of every train"
            )
        return
    if centre is None:
        centre = (a["etl"] - 1) // 2
    if not first <= centre <= last:
        raise ValueError(
            f"the centre echo {centre} must lie after the sixth pulse and before "
            f"the last of every train: from {first} to {last}"
        )


def check_pd(a: Mapping[str, Any]) -> None:
    """Refuse a PD prescription no train realizes, as its design would: the checks that take no solve.

    Raises
    ------
    ValueError
        If the angles are not ``0 < first, final <= maximum < 180`` degrees,
        the signal-tail onset is not a fraction between 0 and 1, or a train
        acquires fewer than two echoes.
    """
    first, maximum, final = (
        a[f"pd_{name}_angle_deg"] for name in ("first", "maximum", "final")
    )
    periphery = a.get("pd_final_angle_periphery_deg")
    finals = [final] if periphery is None else [final, periphery]
    if not all(map(math.isfinite, (first, maximum, *finals))) or not (
        0 < first <= maximum < 180 and all(0 < angle <= maximum for angle in finals)
    ):
        raise ValueError(
            "the PD angles must satisfy 0 < first, final <= maximum < 180 degrees"
        )
    if not 0 < a["pd_plateau_fraction"] < 1:
        raise ValueError("the signal-tail onset must be a fraction between 0 and 1")
    shortest = (
        a["etl"] if a["etl_periphery"] is None else min(a["etl"], a["etl_periphery"])
    )
    if shortest < 2:
        raise ValueError("a PD train acquires at least two echoes after its startup")


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
