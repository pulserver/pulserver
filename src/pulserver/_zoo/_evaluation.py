"""The evaluation of a shipped scanner sequence from the design of one repetition, extrapolated to the scan."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pypulseqpp as pp

from ..design import Evaluation, Protocol, RfLayout, SequencePlugin, StatedParam
from ..protocol import ImagingMode, ProtocolKey, UIParam

#: Arguments that design one repetition of the scan, and the number of
#: repetitions the scan plays, from the arguments of the full design.
Repetition = Callable[[dict[str, Any]], tuple[dict[str, Any], int]]


def _readout_bandwidth(main: pp.Sequence) -> float:
    """Return the receiver bandwidth in Hz of the first ADC event outside a navigator: the inverse of its dwell time."""
    tables = main.libraries()
    played = tables.blocks[:, 4]
    adcs = played[played > 0]
    navigator = main.evaluate_labels(evolution="adc").get("NAV")
    if navigator is not None:
        adcs = adcs[np.asarray(navigator) == 0]
    return 1.0 / float(tables.adc[adcs[0] - 1, 1])


#: The entries that hold the value the design achieved, and where the main
#: sequence records it. A multi-echo sequence lists every echo time; the entry
#: holds the first.
_ACHIEVED: dict[ProtocolKey, Callable[[pp.Sequence], float]] = {
    UIParam.TE: lambda main: main.definitions["TE"][0],
    UIParam.TR: lambda main: main.definitions["TR"][0],
    UIParam.BANDWIDTH: _readout_bandwidth,
}


def arguments(plugin: SequencePlugin, protocol: Protocol) -> dict[str, Any]:
    """Return the arguments the app designs ``protocol`` with, its defaults included.

    An entry binding a name the app does not take is left out.
    """
    signature = inspect.signature(plugin.app)
    taken = {
        name: value
        for name, value in protocol.arguments.items()
        if name in signature.parameters
    }
    bound = signature.bind_partial(**taken)
    bound.apply_defaults()
    return bound.arguments


def achieved(plugin: SequencePlugin, main: pp.Sequence) -> dict[ProtocolKey, float]:
    """Return the values the design achieved of the entries the plugin declares.

    The echo time and the repetition time are the ``TE`` and ``TR``
    definitions of ``main``, the receiver bandwidth the inverse of the dwell
    time of its first ADC event outside a navigator. A 2D sequence's slice
    thickness is its ``SliceThickness`` definition, and the requested one where
    it states none; a 3D sequence's is its slab, the ``z`` of its ``FOV``
    definition, over its locations, the ``z`` of its ``Matrix``.
    """
    values = {
        key: read(main) for key, read in _ACHIEVED.items() if key in plugin.protocol
    }
    if UIParam.SLICE_THICKNESS in plugin.protocol:
        definitions = main.definitions
        if three_d(plugin):
            values[UIParam.SLICE_THICKNESS] = (
                definitions["FOV"][2] / definitions["Matrix"][2]
            )
        elif "SliceThickness" in definitions:
            values[UIParam.SLICE_THICKNESS] = definitions["SliceThickness"][0]
    return values


def three_d(plugin: SequencePlugin) -> bool:
    """Whether a plugin states a 3D acquisition."""
    mode = plugin.protocol.get(UIParam.IMAGING_MODE)
    return isinstance(mode, StatedParam) and mode.value == ImagingMode.THREE_D


def rf_layout(
    main: pp.Sequence,
    scaled: bool,
    *,
    start: float = 0.0,
    copies: int = 1,
    period: float | None = None,
) -> RfLayout:
    """Return the RF of the TR of ``main`` that begins ``start`` seconds into it.

    The TR holds the pulses whose centre lies within the ``TR`` definition of
    ``main`` from ``start``, ``copies`` times over, in play order, and its
    period is that definition unless ``period`` is given. The definitions are
    those the TR plays, numbered by first play in it. The flip angle scales
    each excitation when ``scaled``, and no instance has a control otherwise.
    """
    tr = main.definitions["TR"][0]
    tables = main.libraries()
    rf = tables.blocks[:, 0]
    pulses = np.flatnonzero(rf)
    starts = np.cumsum(tables.block_durations) - tables.block_durations
    centres = (
        starts[pulses] + tables.rf[rf[pulses] - 1, 5] + tables.rf[rf[pulses] - 1, 4]
    )
    in_tr = pulses[(centres >= start) & (centres < start + tr)]
    if in_tr.size == pulses.size:
        instances = main.rf_instances()
    else:
        one_tr = pp.Sequence(main.system)
        for index in in_tr:
            one_tr.add_block(main.get_block(int(index) + 1))
        instances = one_tr.rf_instances()
    instances = pp.RfInstances(
        instances.definitions,
        np.tile(instances.definition, copies),
        np.tile(instances.amplitude, copies),
    )
    control = tuple(
        UIParam.FLIP
        if scaled and instances.definitions[number].use == "excitation"
        else None
        for number in instances.definition
    )
    return RfLayout(instances, control, tr if period is None else period)


def packets(
    n_slices: int, cycle: float | None, shot: float, raster: float
) -> tuple[list[int], list[float]]:
    """Return the slices of each packet, and the duration of one cycle of each.

    As the shipped 2D sequences play them, the slices one ``cycle`` holds,
    one shot of ``shot`` seconds each, form a packet, the slices are dealt
    round-robin into packets, and the last shot of a packet is padded to the
    cycle on the block raster. Without a cycle one packet holds every slice.
    """
    per_packet = n_slices if cycle is None else max(1, int(cycle / shot + 1e-9))
    count = -(-n_slices // per_packet)
    sizes = [len(range(start, n_slices, count)) for start in range(count)]
    if cycle is None:
        cycle = max(sizes) * shot
    return sizes, [
        size * shot + pp.round_to_raster(cycle - size * shot, raster) for size in sizes
    ]


def evaluation(
    plugin: SequencePlugin, system: pp.Opts, protocol: Protocol, repetition: Repetition
) -> Evaluation:
    """Return the evaluation of a protocol from the design of one repetition.

    ``repetition`` gives the arguments that design one repetition, its view
    the first of the scan and no dummy before it, and the number of
    repetitions the scan plays. The entries the design achieves are read from
    it, and the RF layout is its first TR.

    A sequence with slices is designed for one slice at the shortest TR,
    which is one shot, and the TR and the scan time follow from the packets
    the shot deals the slices into (:func:`packets`). The layout repeats the
    shot once per slice of the largest packet.

    Raises
    ------
    ValueError
        If the requested TR is shorter than one shot.
    """
    a = arguments(plugin, protocol)
    one, repetitions = repetition(a)
    scaled = UIParam.FLIP in plugin.protocol
    if "n_slices" not in a:
        main = plugin.app(system, **(protocol.arguments | one))
        return Evaluation(
            protocol.replace(achieved(plugin, main)),
            repetitions * main.duration()[0],
            rf_layout=rf_layout(main, scaled),
        )
    one |= {"n_slices": 1, "tr": None}
    main = plugin.app(system, **(protocol.arguments | one))
    shot, tr = main.definitions["TR"][0], a["tr"]
    raster = system.block_duration_raster
    if tr is not None and pp.round_to_raster(tr - shot, raster) < 0:
        raise ValueError(
            f"the requested TR of {tr * 1e3:.3f} ms is shorter than the "
            f"{shot * 1e3:.3f} ms one slice takes"
        )
    sizes, cycles = packets(a["n_slices"], tr, shot, raster)
    values = achieved(plugin, main)
    if UIParam.TR in values:
        values[UIParam.TR] = max(cycles)
    return Evaluation(
        protocol.replace(values),
        repetitions * sum(cycles),
        rf_layout=rf_layout(main, scaled, copies=max(sizes), period=max(cycles)),
    )


def waved(a: dict[str, Any]) -> bool:
    """Whether the arguments play wave-encoding gradients, under which the calibration region is acquired again without them first."""
    return a.get("wave_amplitude", 0.0) > 0 and a.get("wave_cycles", 0) > 0


def _views(n: int, acceleration: int, n_acs: int, partial_fourier: float) -> int:
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        n, acceleration, n_acs, partial_fourier=partial_fourier
    )
    return len(calibrating) + len(imaging)


def cartesian_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One line of a 2D Cartesian scan, and its dummies and lines."""
    lines = _views(a["n_y"], a["ry"], a["n_acs_y"], a["partial_fourier_y"])
    return {"n_dummy": 0, "ry": a["n_y"], "n_acs_y": 0}, a["n_dummy"] + lines


def cartesian_3d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One view of a 3D Cartesian scan, and its dummies and views.

    Under the wave the views include the wave-free calibration region.
    """
    calibrating, imaging = pp.make_cartesian_plane_sampling(
        (a["n_y"], a["n_z"]),
        (a["ry"], a["rz"]),
        (a["n_acs_y"], a["n_acs_z"]),
        caipi_shift=a["caipi_shift"],
        partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
        elliptical=a["elliptical_sampling"],
        elliptical_acs=a["elliptical_acs"],
    )
    one = {"n_dummy": 0, "ry": a["n_y"], "rz": a["n_z"], "n_acs_y": 0, "n_acs_z": 0}
    references = len(calibrating) if waved(a) else 0
    return one, a["n_dummy"] + references + len(calibrating) + len(imaging)


def _nyquist_spokes(n: int) -> int:
    """Return the full spokes that sample a matrix of ``n`` at Nyquist."""
    return math.ceil(math.pi / 2 * n)


def propeller_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One line of one blade of a 2D PROPELLER scan, and its dummies and lines.

    A blade is ``blade_width`` lines and the Nyquist set is
    ``ceil(pi * n / (2 * blade_width))`` blades, so a blade of one line has
    the set of ``_nyquist_spokes(n)``, of which one in that many is played.
    """
    blades = math.ceil(math.pi * a["n"] / (2 * a["blade_width"]))
    lines = len(range(0, blades, a["ry"])) * a["blade_width"]
    one = {"n_dummy": 0, "blade_width": 1, "ry": _nyquist_spokes(a["n"])}
    return one, a["n_dummy"] + lines


def radial_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One spoke of a 2D radial scan, and its dummies and spokes."""
    nyquist = _nyquist_spokes(a["n"])
    spokes = len(range(0, nyquist, a["ry"]))
    return {"n_dummy": 0, "ry": nyquist}, a["n_dummy"] + spokes


def spiral_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One interleaf of a 2D spiral scan, and its dummies and interleaves."""
    interleaves = len(range(0, a["n_shots"], a["ry"]))
    return {"n_dummy": 0, "ry": a["n_shots"]}, a["n_dummy"] + interleaves


def stack_of_stars(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One spoke at one partition of a stack of stars, and its dummies and views."""
    nyquist = _nyquist_spokes(a["n"])
    spokes = len(range(0, nyquist, a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = {"n_dummy": 0, "ry": nyquist, "rz": a["n_z"], "n_acs_z": 0}
    return one, a["n_dummy"] + spokes * partitions


def stack_of_blades(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One line of one blade at one partition of a stack of blades, and its dummies and views.

    The scan plays every ``ry``-th blade of the Nyquist set of ``ceil(pi n /
    (2 blade_width))``, each of its ``blade_width`` lines at every acquired
    partition. A blade of one line, with ``ry`` the size of its Nyquist set,
    leaves the first blade and its one line.
    """
    width = a["blade_width"]
    blades = len(range(0, math.ceil(math.pi * a["n"] / (2 * width)), a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = {
        "n_dummy": 0,
        "ry": _nyquist_spokes(a["n"]),
        "rz": a["n_z"],
        "n_acs_z": 0,
        "blade_width": 1,
    }
    return one, a["n_dummy"] + blades * width * partitions


def stack_of_spirals(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One interleaf at one partition of a stack of spirals, and its dummies and views."""
    interleaves = len(range(0, a["n_shots"], a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = {"n_dummy": 0, "ry": a["n_shots"], "rz": a["n_z"], "n_acs_z": 0}
    return one, a["n_dummy"] + interleaves * partitions
