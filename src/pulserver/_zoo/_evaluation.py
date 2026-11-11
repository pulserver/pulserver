"""The evaluation of a shipped scanner sequence from the design of a few repetitions, extrapolated to the scan."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pypulseqpp as pp
from pypulseqpp import sequences

from ..design import Evaluation, Protocol, RfLayout, SequencePlugin, StatedParam
from ..protocol import ImagingMode, ProtocolKey, UIParam

#: Arguments that design the repetitions an evaluation reads, and the number
#: of repetitions the scan plays, from the arguments of the full design.
Repetition = Callable[[dict[str, Any]], tuple[dict[str, Any], int]]


def edge(n: int) -> int:
    """Return the undersampling at which a lattice of ``n`` lines plays its centre line and its first.

    The first line is the edge of k-space, where the phase encode is
    largest. A lattice is laid from the centre line, so at an undersampling
    of ``n // 2`` it holds lines ``0`` and ``n // 2``, and the last where
    ``n`` is odd.
    """
    return max(1, n // 2)


def extremes(a: dict[str, Any], *axes: str) -> dict[str, Any]:
    """Return the arguments that play, on each phase-encoding axis of ``axes``, the centre line and the edge of k-space.

    The views a sequence function then designs are few, and hold the
    largest phase encode of each axis, at one edge of k-space: the first
    line. The nerve response read on them is the scan's largest where the
    nerve model is the same on every axis, since no sign of an encode
    changes it. The other edge plays the same encodes reversed: on an oblique
    prescription, where a physical axis sums the logical ones, the scan can
    read more slew rate there, up to 11 % over random rotations, and at a
    given rotation up to about 19 % more amplitude on one axis (never more
    than the largest amplitude the representative reads at any rotation);
    the check of the generated design finds either; under
    elliptical sampling the views between the two axes are likewise not
    designed. The
    lattice is whole: no calibration region, partial Fourier or CAIPI shift
    drops or moves a line of it.
    """
    one: dict[str, Any] = {}
    for axis in axes:
        one |= {
            f"r{axis}": edge(a[f"n_{axis}"]),
            f"n_acs_{axis}": 0,
            f"partial_fourier_{axis}": 1.0,
        }
    if "caipi_shift" in a:
        one["caipi_shift"] = 0
    if "sampling" in a:
        one["sampling"] = "lattice"
    return one


#: The directions a readout turned about the slice axis is designed in: the
#: physical axis an oblique prescription puts most of a readout on reads, of
#: the nearest of them, the cosine of half their spacing, within 2 %.
DIRECTIONS = 8


def turned(count: int) -> int:
    """Return the undersampling at which ``count`` readouts spread evenly in angle leave :data:`DIRECTIONS` of them, or all where they are fewer."""
    return max(1, count // DIRECTIONS)


def repetitions_of(main: pp.Sequence) -> tuple[int, float]:
    """Return how many repetitions ``main`` plays and how long each lasts, in seconds.

    ``main`` plays whole repetitions of its ``TR`` definition and nothing
    else; the delay closing each is rounded to the block raster, so one lasts
    the definition or a raster more.
    """
    total = main.duration()[0]
    count = max(1, round(total / main.definitions["TR"][0]))
    return count, total / count


def first_readout(main: pp.Sequence) -> tuple[int, float]:
    """Return the samples and the dwell time in seconds of the first ADC event of ``main`` outside a navigator."""
    tables = main.libraries()
    played = tables.blocks[:, 4]
    adcs = played[played > 0]
    navigator = main.evaluate_labels(evolution="adc").get("NAV")
    if navigator is not None:
        adcs = adcs[np.asarray(navigator) == 0]
    row = tables.adc[adcs[0] - 1]
    return int(row[0]), float(row[1])


def _readout_bandwidth(main: pp.Sequence) -> float:
    """Return the receiver bandwidth in Hz of the first ADC event outside a navigator: the inverse of its dwell time."""
    return 1.0 / first_readout(main)[1]


def bandwidth_note(
    system: pp.Opts, requested_hz: float, samples: int, dwell: float
) -> str:
    """Return why an EPI line samples at ``1 / dwell`` rather than the ``requested_hz`` asked; empty where it samples at it.

    The dwell is a multiple of ``system.adc_raster_time``: the one nearest
    the request, unless the read gradient cannot sweep a ramp-sampled line in
    the time ``samples`` take at that dwell under the design's slew rate,
    which then sets it.
    """
    achieved = 1.0 / dwell
    if abs(achieved - requested_hz) <= 1e-6 * requested_hz:
        return ""
    raster = system.adc_raster_time
    nearest = max(1, round(1.0 / (requested_hz * raster))) * raster
    if dwell > nearest * (1 + 1e-9):
        why = (
            f"at the design's slew rate of {system.max_slew / system.gamma:.1f} "
            "T/m/s a ramp-sampled line needs longer than the "
            f"{1e3 * samples * nearest:.3f} ms its {samples} "
            f"samples take at {1e-3 / nearest:.0f} kHz, and the sampling interval "
            f"steps by the {1e6 * raster:g} us ADC raster"
        )
    else:
        why = (
            f"the sampling interval is a multiple of the {1e6 * raster:g} us ADC raster"
        )
    return (
        f"bandwidth {1e-3 * achieved:.0f} kHz, not {1e-3 * requested_hz:.0f} kHz: {why}"
    )


#: The entries that hold the value the design achieved, and where the main
#: sequence records it. A multi-echo sequence lists every echo time; the entry
#: holds the first.
_ACHIEVED: dict[ProtocolKey, Callable[[pp.Sequence], float]] = {
    UIParam.TE: lambda main: main.definitions.get(
        "EffectiveEchoTime", main.definitions["TE"]
    )[0],
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
    definitions of ``main`` (``EffectiveEchoTime`` takes precedence for
    contrast-equivalent FSE annotation), the receiver bandwidth the inverse of the dwell
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
    banded: bool = False,
) -> RfLayout:
    """Return the RF of the TR of ``main`` that begins ``start`` seconds into it.

    The TR holds the pulses whose centre lies within the ``TR`` definition of
    ``main`` from ``start``, ``copies`` times over, in play order, and its
    period is that definition unless ``period`` is given. The definitions are
    those the TR plays, numbered by first play in it; when ``banded``, those
    of the saturation pulses after the others, so a saturation band or a fat
    saturation placed or removed leaves the numbers of the others as the
    listing, made with every one placed, states them. The flip angle scales
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
    definitions = instances.definitions
    order = sorted(
        range(len(definitions)),
        key=lambda d: (banded and definitions[d].use == "saturation", d),
    )
    number = np.argsort(order)
    instances = pp.RfInstances(
        tuple(definitions[d] for d in order),
        np.tile(number[instances.definition], copies),
        np.tile(instances.amplitude, copies),
    )
    control = tuple(
        UIParam.FLIP
        if scaled and instances.definitions[number].use == "excitation"
        else None
        for number in instances.definition
    )
    return RfLayout(instances, control, tr if period is None else period)


def rf_power(layout: RfLayout) -> float:
    """Return the mean RF power of a layout over its period, in Hz².

    Each instance plays its peak, ``amplitude * peak_hz``, squared, times the
    integral of the squared magnitude of its definition's unit-peak waveform
    over its channels, by the trapezoidal rule over the sample times.
    """
    instances = layout.instances
    energy = np.array(
        [
            definition.peak_hz**2
            * sum(
                np.trapezoid(np.abs(channel) ** 2, definition.time)
                for channel in definition.waveform
            )
            for definition in instances.definitions
        ]
    )
    played = instances.amplitude**2 * energy[instances.definition]
    return float(np.sum(played) / layout.period)


def _same_pulse(a: pp.RfDefinition, b: pp.RfDefinition) -> bool:
    """Whether two definitions play one pulse: one use, sample times and unit-peak waveform."""
    return (
        a.use == b.use
        and a.time.shape == b.time.shape
        and a.waveform.shape == b.waveform.shape
        and np.allclose(a.time, b.time, rtol=0.0, atol=1e-9)
        and np.allclose(a.waveform, b.waveform, rtol=0.0, atol=1e-6)
    )


def densest(layout: RfLayout, *prescans: RfLayout) -> RfLayout:
    """Return ``layout``, or the RF of a prescan of the chain where it plays more RF power, over the layout's period.

    A scanner costs each sequence of a chain over its own repetition, and a
    layout is one TR of the main sequence. A prescan that does not repeat at
    the TR, such as a calibration at its own shortest TR, can play more RF
    power than the TR does, and a protocol whose TR meets the limits is then
    refused before the scan. Each of ``prescans`` states the prescan's
    instances over its duration (:meth:`RfLayout.of`). The one of the most RF
    power (:func:`rf_power`), where it is more than the layout's, is returned
    as its instances in play order, repeated, as many as it plays in
    ``layout.period``, rounded up, over that period: the period stays the TR
    the console sets, and the power is at least the prescan's at every TR, so
    a scanner that lengthens the TR to meet its limits cannot meet them for a
    prescan that exceeds them.

    The prescan's definitions take the numbers of the layout's, which the
    listing names: the definition that plays the same pulse, else the first
    of its use, and the amplitudes are over that definition's peak, so the
    peak each instance plays is its own. A definition of a use the layout
    plays no pulse of is numbered after the layout's. The instances follow no
    control: a prescan's flip angle is its own.
    """
    strongest = max(prescans, key=rf_power, default=None)
    if strongest is None or rf_power(strongest) <= rf_power(layout):
        return layout
    own = layout.instances.definitions
    definitions = list(own)
    numbers = []
    for pulse in strongest.instances.definitions:
        number = next((i for i, d in enumerate(own) if _same_pulse(d, pulse)), None)
        if number is None:
            number = next((i for i, d in enumerate(own) if d.use == pulse.use), None)
        if number is None:
            number = len(definitions)
            definitions.append(pulse)
        numbers.append(number)
    peak = np.array([d.peak_hz for d in strongest.instances.definitions])
    over = np.array([definitions[n].peak_hz for n in numbers])
    scale = np.divide(peak, over, out=np.ones_like(peak), where=over > 0)
    played = len(strongest.instances.definition)
    count = math.ceil(layout.period * played / strongest.period - 1e-9)
    picks = np.arange(count) % played
    which = strongest.instances.definition[picks]
    instances = pp.RfInstances(
        tuple(definitions),
        np.asarray(numbers)[which],
        strongest.instances.amplitude[picks] * scale[which],
    )
    return RfLayout(instances, (None,) * count, layout.period)


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
    """Return the evaluation of a protocol from the design of a few repetitions.

    ``repetition`` gives the arguments that design them, with no dummy
    before them, and the number of repetitions the scan acquires; the dummies
    are played before them (:func:`dummies`). The entries the design achieves
    are read from it, the RF layout is its first TR, and it is the
    evaluation's representative sequence: a Cartesian sequence plays the
    views of :func:`extremes`, and one that turns a readout about the slice
    axis plays it in :data:`DIRECTIONS` directions (:func:`turned`).

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
    banded = UIParam.EXSAT_MASK in plugin.protocol
    if "n_slices" not in a:
        main = plugin.app(system, **(protocol.arguments | one))
        _, duration = repetitions_of(main)
        return Evaluation(
            protocol.replace(achieved(plugin, main)),
            (dummies(a, duration) + repetitions) * duration,
            rf_layout=rf_layout(main, scaled, banded=banded),
            sequence=main,
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
        (dummies(a, max(cycles)) + repetitions) * sum(cycles),
        rf_layout=rf_layout(
            main, scaled, copies=max(sizes), period=max(cycles), banded=banded
        ),
        sequence=main,
    )


def dummies(a: dict[str, Any], repetition_s: float) -> int:
    """Return the dummy repetitions the sequence function plays before acquiring.

    Its ``n_dummy``, or where that is ``None`` as many as bring its steady
    state within 1 % at a repetition every ``repetition_s``
    (:func:`pypulseqpp.sequences.steady_state_dummies`), at its flip angle or,
    without one, a spin echo's 90 degrees.
    """
    if a.get("n_dummy") is not None:
        return int(a["n_dummy"])
    return sequences.steady_state_dummies(
        repetition_s, float(a.get("flip_angle_deg", 90.0))
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
    """Centre and edge lines of a 2D Cartesian scan, and its lines."""
    lines = _views(a["n_y"], a["ry"], a["n_acs_y"], a["partial_fourier_y"])
    return {"n_dummy": 0} | extremes(a, "y"), lines


def cartesian_3d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Centre and edge views of a 3D Cartesian scan, and its views.

    Under the wave the views include the wave-free calibration region. The
    views are counted with the app's ``sampling``, a lattice where it takes
    none; the views designed are a lattice's (:func:`extremes`).
    """
    calibrating, imaging = pp.count_cartesian_plane_sampling(
        (a["n_y"], a["n_z"]),
        (a["ry"], a["rz"]),
        (a["n_acs_y"], a["n_acs_z"]),
        caipi_shift=a["caipi_shift"],
        partial_fourier=(a["partial_fourier_y"], a["partial_fourier_z"]),
        elliptical=a["elliptical_sampling"],
        elliptical_acs=a["elliptical_acs"],
        sampling=a.get("sampling", "lattice"),
    )
    references = calibrating if waved(a) else 0
    one = {"n_dummy": 0} | extremes(a, "y", "z")
    return one, references + calibrating + imaging


def _nyquist_spokes(n: int) -> int:
    """Return the full spokes that sample a matrix of ``n`` at Nyquist."""
    return math.ceil(math.pi / 2 * n)


def propeller_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One line of one blade of a 2D PROPELLER scan, and its lines.

    A blade is ``blade_width`` lines and the Nyquist set is
    ``ceil(pi * n / (2 * blade_width))`` blades, so a blade of one line has
    the set of ``_nyquist_spokes(n)``, of which one in that many is played.
    """
    blades = math.ceil(math.pi * a["n"] / (2 * a["blade_width"]))
    lines = len(range(0, blades, a["ry"])) * a["blade_width"]
    one = {"n_dummy": 0, "blade_width": 1, "ry": turned(_nyquist_spokes(a["n"]))}
    return one, lines


def radial_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One spoke of a 2D radial scan, and its spokes."""
    nyquist = _nyquist_spokes(a["n"])
    spokes = len(range(0, nyquist, a["ry"]))
    return {"n_dummy": 0, "ry": turned(nyquist)}, spokes


def spiral_2d(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One interleaf of a 2D spiral scan, and its interleaves."""
    interleaves = len(range(0, a["n_shots"], a["ry"]))
    return {"n_dummy": 0, "ry": turned(a["n_shots"])}, interleaves


def stack_of_stars(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One spoke at the centre and edge partitions of a stack of stars, and its views."""
    nyquist = _nyquist_spokes(a["n"])
    spokes = len(range(0, nyquist, a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = {"n_dummy": 0} | extremes(a, "z") | {"ry": turned(nyquist)}
    return one, spokes * partitions


def stack_of_blades(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One line of one blade at the centre and edge partitions of a stack of blades, and its views.

    The scan plays every ``ry``-th blade of the Nyquist set of ``ceil(pi n /
    (2 blade_width))``, each of its ``blade_width`` lines at every acquired
    partition. A blade of one line, with ``ry`` the size of its Nyquist set,
    leaves the first blade and its one line.
    """
    width = a["blade_width"]
    blades = len(range(0, math.ceil(math.pi * a["n"] / (2 * width)), a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = (
        {"n_dummy": 0}
        | extremes(a, "z")
        | {"ry": turned(_nyquist_spokes(a["n"])), "blade_width": 1}
    )
    return one, blades * width * partitions


def stack_of_spirals(a: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """One interleaf at the centre and edge partitions of a stack of spirals, and its views."""
    interleaves = len(range(0, a["n_shots"], a["ry"]))
    partitions = _views(a["n_z"], a["rz"], a["n_acs_z"], a["partial_fourier_z"])
    one = {"n_dummy": 0} | extremes(a, "z") | {"ry": turned(a["n_shots"])}
    return one, interleaves * partitions
