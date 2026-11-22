"""The triggered, segmented 3D balanced SSFP of the MRA plugins: the app each binds and its evaluation."""

from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pypulseqpp as pp

from ..design import Evaluation, RfLayout
from ..protocol import UIParam
from ._evaluation import achieved, arguments, extremes


def specialised(
    app: Callable[..., Any],
    fixed: Mapping[str, Any],
    defaults: Mapping[str, Any],
    period: tuple[str, float, Callable[[float], float]],
) -> Callable[..., Any]:
    """Return ``app`` with ``fixed`` arguments bound, ``defaults`` changed and its trigger period taken from a rate.

    ``period`` is ``(name, default, to_period)``: the argument the returned
    function takes in place of ``trigger_period``, its default, and the
    conversion from it to the period in seconds.
    """
    name, default, to_period = period
    signature = inspect.signature(app)
    parameters = []
    for p in signature.parameters.values():
        if p.name in fixed:
            continue
        if p.name == "trigger_period":
            parameters.append(p.replace(name=name, default=default))
        elif p.name in defaults:
            parameters.append(p.replace(default=defaults[p.name]))
        else:
            parameters.append(p)
    stated = signature.replace(parameters=parameters)

    @functools.wraps(app)
    def bound(*args: Any, **kwargs: Any) -> Any:
        values = stated.bind(*args, **kwargs)
        values.apply_defaults()
        given = dict(values.arguments)
        given["trigger_period"] = to_period(given.pop(name))
        return app(**given, **fixed)

    bound.__signature__ = stated  # type: ignore[attr-defined]
    return bound


def segmented_evaluation(
    plugin: Any, system: Any, protocol: Any, **fixed: Any
) -> Evaluation:
    """Evaluate a segmented scan from the design of one segment of its centre and edge views.

    The scan deals its views into segments of equal size, one per trigger,
    the largest the length of the designed segment and the views it leaves
    out, one repetition each; its duration is the number of segments times
    the nominal trigger period. The RF of one period is that of the largest
    segment: the designed segment's preparations, ramp and views, and one
    excitation more per view left out.

    ``fixed`` are arguments the plugin passes its app beside the protocol's.

    Raises
    ------
    ValueError
        If a segment is longer than the trigger period.
    """
    a = arguments(plugin, protocol)
    one = extremes(a, "y", "z") | {"views_per_segment": 1 << 20}
    [designed] = plugin.app(system, **(protocol.arguments | one | fixed))
    tr = designed.definitions["TR"][0]
    period = designed.definitions["TriggerPeriod"][0]
    adc = designed.evaluate_labels(evolution="adc")
    navigators = np.atleast_1d(adc.get("NAV", 0))
    lines = np.atleast_1d(adc["LIN"])
    views = int(lines.size - np.count_nonzero(np.broadcast_to(navigators, lines.shape)))
    calibrating, imaging = pp.count_cartesian_plane_sampling(
        (a["n_y"], a["n_z"]),
        (a["ry"], a["rz"]),
        (a["n_acs_y"], a["n_acs_z"]),
        caipi_shift=a["caipi_shift"],
        partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
        elliptical=True,
        elliptical_acs=a["elliptical_acs"],
    )
    total = calibrating + imaging
    segments = math.ceil(total / int(a["views_per_segment"]))
    # The views are dealt into segments of equal size, the first the largest.
    per_segment = math.ceil(total / segments)
    left_out = max(per_segment - views, 0)
    segment = designed.definitions["SegmentDuration"][0] + left_out * tr
    if segment > period + 1e-9:
        raise ValueError(
            f"a segment of {per_segment} views takes {segment * 1e3:.0f} ms, longer "
            f"than the {period * 1e3:.0f} ms trigger period"
        )
    return Evaluation(
        protocol.replace(achieved(plugin, designed)),
        segments * period,
        rf_layout=_segment_layout(designed, left_out, period),
        sequence=designed,
    )


def _segment_layout(designed: pp.Sequence, left_out: int, period: float) -> RfLayout:
    """Return the RF of one trigger period: the designed segment's and ``left_out`` excitations at the nominal flip.

    The definitions are numbered by first play with the saturation pulses
    last, so a fat saturation removed leaves the numbers of the others as the
    listing, made with it placed, states them.
    """
    instances = designed.rf_instances()
    definitions = instances.definitions
    order = sorted(
        range(len(definitions)), key=lambda d: (definitions[d].use == "saturation", d)
    )
    number = np.argsort(order)
    excitation = int(number[instances.definition[-1]])
    nominal = float(instances.amplitude[-1])
    instances = pp.RfInstances(
        tuple(definitions[d] for d in order),
        np.concatenate([number[instances.definition], np.full(left_out, excitation)]),
        np.concatenate([instances.amplitude, np.full(left_out, nominal)]),
    )
    control = tuple(
        UIParam.FLIP if instances.definitions[number].use == "excitation" else None
        for number in instances.definition
    )
    return RfLayout(instances, control, period)
