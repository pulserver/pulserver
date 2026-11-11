"""Fast spin echo on the scanner: SAR burst limits, and the RF layout of trains that differ in length and TR."""

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pypulseqpp as pp

from pulserver.design import RfLayout
from pulserver.host import _memo

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
    least the RF power: an evaluation designs it for its scan time, its RF
    and its gradients, and asks the solve only for the effective TE
    (:func:`effective_te`). ``a`` holds the arguments of the full design.
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


def effective_te(
    app: Callable[..., Any], system: pp.Opts, designed: Mapping[str, Any]
) -> float:
    """Return the effective TE the train of a Busse or PD prescription reaches, in seconds, as its design finds it.

    ``designed`` are the arguments of one train of the prescription, its
    angles as the inverse solve of pypulseqpp gives them without fitting:
    the centre echo nearest the TE asked for, or the midpoint, and the
    contrast-equivalent TE there, the ``EffectiveEchoTime`` of the design. A
    prescription no train realizes is refused as the design refuses it.

    Under a warm design server the answer is kept for the calls after this
    one (:mod:`pulserver.host._memo`), under the arguments and the limits, so
    a train is solved for once however often its protocol is evaluated.

    Raises
    ------
    ValueError
        With the message of the design, where it refuses the prescription.
    """
    key = json.dumps(
        [
            sorted((name, repr(value)) for name, value in designed.items()),
            sorted((name, repr(value)) for name, value in vars(system).items()),
        ]
    )
    found, kept = _memo.read(_TRAINS, key)
    if not found:
        try:
            train = app(system, **designed)
        except (ValueError, AssertionError) as error:
            kept = {"refused": str(error) or type(error).__name__}
        else:
            kept = {"te": float(train.definitions["EffectiveEchoTime"][0])}
        _memo.write(_TRAINS, key, kept)
    if "refused" in kept:
        raise ValueError(kept["refused"])
    return kept["te"]


# The entries of :mod:`pulserver.host._memo` the solved trains are.
_TRAINS = "fse-train"


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
