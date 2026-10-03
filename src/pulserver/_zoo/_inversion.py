"""The evaluation of a scan of inversion-prepared shots, each a train of excitations, from the design of a shot of one excitation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pypulseqpp as pp

from ..design import Evaluation, Protocol, RfLayout, SequencePlugin
from ..protocol import UIParam
from ._evaluation import achieved, arguments, rf_layout

#: From the arguments of the full design: the excitations a shot plays, the
#: excitations of the fully sampled set they are taken from, and the shots of
#: the scan.
Shots = Callable[[dict[str, Any]], tuple[int, int, int]]


def _partitions(a: dict[str, Any]) -> int:
    """Return the partitions the scan acquires, the calibration ones included."""
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        a["n_z"], a["rz"], a["n_acs_z"], partial_fourier=a["partial_fourier_z"]
    )
    return len(calibrating) + len(imaging)


def shots_of_stars(a: dict[str, Any]) -> tuple[int, int, int]:
    """Return the spokes a shot of a stack of stars plays, the spokes of its Nyquist set and the shots."""
    nyquist = math.ceil(math.pi / 2 * a["n"])
    return len(range(0, nyquist, a["ry"])), nyquist, a["n_dummy"] + _partitions(a)


def shots_of_spirals(a: dict[str, Any]) -> tuple[int, int, int]:
    """Return the interleaves a shot of a stack of spirals plays, the interleaves of its set and the shots."""
    return (
        len(range(0, a["n_shots"], a["ry"])),
        a["n_shots"],
        a["n_dummy"] + _partitions(a),
    )


def inversion_train(
    plugin: SequencePlugin, system: pp.Opts, protocol: Protocol, shots: Shots
) -> Evaluation:
    """Return the evaluation of a protocol from the design of a shot of one excitation.

    A shot is an inversion, the wait that puts the first excitation at the
    inversion time, the excitations of one partition and the recovery that
    makes the shots ``TR`` apart. The shot is designed at the shortest TR for
    one excitation, and the TR, the scan time and the layout follow from the
    spacing of the excitations it states (``EchoSpacing``) and the number of
    them ``shots`` gives. The layout is the inversion followed by the
    excitations of a shot.

    Raises
    ------
    ValueError
        If the requested TR is shorter than one shot.
    """
    a = arguments(plugin, protocol)
    excitations, nyquist, count = shots(a)
    one = {
        "n_dummy": 0,
        "ry": nyquist,
        "rz": a["n_z"],
        "n_acs_z": 0,
        "partial_fourier_z": 1.0,
        "tr": None,
    }
    main = plugin.app(system, **(protocol.arguments | one))
    raster = system.block_duration_raster
    # The shot lasts until its last excitation has been played out, and then
    # the recovery of at least one raster.
    body = main.definitions["TR"][0] - raster
    body += (excitations - 1) * main.definitions["EchoSpacing"][0]
    tr = body + raster if a["tr"] is None else a["tr"]
    if tr - body < raster - 1e-9:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
            f"{(body + raster) * 1e3:.3f} ms one shot takes"
        )
    repetition = body + pp.round_to_raster(tr - body, raster)
    values = achieved(plugin, main) | {UIParam.TR: repetition}
    layout = rf_layout(main, UIParam.FLIP in plugin.protocol, period=repetition)
    return Evaluation(
        protocol.replace(values),
        count * repetition,
        rf_layout=_train(layout, excitations),
    )


def _train(layout: RfLayout, excitations: int) -> RfLayout:
    """Return ``layout``, an inversion and one excitation, with the excitation played ``excitations`` times."""
    instances = layout.instances

    def played(values: Any) -> Any:
        return np.concatenate([values[:1], np.tile(values[1:], excitations)])

    return RfLayout(
        pp.RfInstances(
            instances.definitions,
            played(instances.definition),
            played(instances.amplitude),
        ),
        (*layout.control[:1], *layout.control[1:] * excitations),
        layout.period,
    )
