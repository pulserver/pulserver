"""Repetitions of a sequence of blocks against the same blocks played one by one."""

from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.virtual import Isochromats

RNG = np.random.default_rng(29)
SYSTEM = pp.Opts()
SAMPLES, DWELL, FOV = 32, 20e-6, 0.24
#: The readout gradient, in Hz/m, whose area over a dwell is one cycle per FOV.
READOUT = 1.0 / (FOV * DWELL)
#: When the readout starts in its block, in s.
PRE = 0.4e-3
#: A phase encoding's corners, times and unit amplitude, and its area, in s.
ENCODE = np.array([[0.0, 0.05e-3, 0.3e-3, 0.35e-3], [0.0, 1.0, 1.0, 0.0]])
ENCODE_AREA = 0.3e-3


def _area(times, values):
    """The area of a piecewise-linear waveform, in its units times s."""
    return float(np.sum(np.diff(times) * (values[1:] + values[:-1]) / 2))


_SPAN = SAMPLES * DWELL
_HEAD = np.array(
    [
        [0.0, 0.05e-3, 0.35e-3, PRE, PRE + _SPAN],
        [0.0, -READOUT * _SPAN / 0.6e-3, -READOUT * _SPAN / 0.6e-3, READOUT, READOUT],
    ]
)
_REWINDER = -(_area(*_HEAD) + 0.025e-3 * READOUT) / 0.35e-3
#: A radial spoke along x, in s over Hz/m: its prephaser, the readout and a
#: rewinder, which leave no area.
SPOKE = np.column_stack(
    [
        _HEAD,
        [
            [PRE + _SPAN + 0.05e-3, PRE + _SPAN + 0.35e-3, PRE + _SPAN + 0.4e-3],
            [_REWINDER, _REWINDER, 0.0],
        ],
    ]
)
#: The spoke without its rewinder, which leaves the area of its prephaser and
#: readout.
OPEN_SPOKE = np.column_stack([_HEAD, [[PRE + _SPAN + 0.05e-3], [0.0]]])
#: The spoke's area at the first sample, in 1/m.
SPOKE_FIRST = _area(*_HEAD[:, :4]) + READOUT * DWELL / 2

#: A ZTE spoke's pulse block, the readout held after it, and the turn onto
#: the next spoke, in s; and its first sample, from its window's block.
ZTE_HOLD, ZTE_READ, ZTE_TURN = 60e-6, 0.7e-3, 0.2e-3
ZTE_FIRST = 20e-6 + DWELL / 2


def _properties(positions, coils=2, transmit=None):
    count = len(positions)
    return {
        "positions": positions,
        "proton_density": RNG.uniform(0.5, 1.0, count),
        "t1": RNG.choice([0.8, 1.2, 2.5], count),
        "t2": RNG.choice([0.05, 0.08, 0.3], count),
        "off_resonance": RNG.normal(0.0, 30.0, count),
        "transmit": transmit,
        "receive": RNG.normal(size=(count, coils))
        + 1j * RNG.normal(size=(count, coils)),
    }


def _engines(properties):
    """Two isochromats alike: one to play block by block, one to play repetitions."""
    given = dict(properties)
    positions = given.pop("positions")
    return Isochromats(positions, **given), Isochromats(positions, **given)


def _repetition(encoding=(0.0, 0.0), rf_phase=0.0, adc_phase=0.0, rewound=True):
    """A slice-selective pulse; a readout prephaser and phase encodings along y and z; a readout; the rewinders.

    ``encoding`` is each phase encoding's area, in 1/m; without ``rewound``,
    the encodings are left as they are.
    """
    rf, gz, _ = pp.make_sinc_pulse(
        np.pi / 6,
        duration=1e-3,
        slice_thickness=5e-3,
        phase_offset=rf_phase,
        return_gz=True,
        system=SYSTEM,
    )
    adc = pp.make_adc(
        SAMPLES, dwell=DWELL, delay=PRE, phase_offset=adc_phase, system=SYSTEM
    )
    span = SAMPLES * DWELL
    prephaser = -READOUT * span / 2 / 0.3e-3
    gx = np.array(
        [
            [0.0, 0.05e-3, 0.35e-3, PRE, PRE + span, PRE + span + 0.05e-3],
            [0.0, prephaser, prephaser, READOUT, READOUT, 0.0],
        ]
    )
    select = np.array(
        [[0.0, 0.1e-3, 1.1e-3, 1.2e-3], [0.0, gz.amplitude, gz.amplitude, 0.0]]
    )
    phase = [ENCODE * [[1.0], [a / ENCODE_AREA]] if a else None for a in encoding]
    rewind = [
        ENCODE * [[1.0], [-a / ENCODE_AREA]] if a and rewound else None
        for a in encoding
    ]
    return [
        {"duration": 1.2e-3, "rf": rf, "gradients": [None, None, select]},
        {"duration": PRE + span + 0.05e-3, "adc": adc, "gradients": [gx, *phase]},
        {"duration": 0.5e-3, "gradients": [None, *rewind]},
    ]


def _played(spins, encodings, rf_phases, adc_phases, rewound=True):
    """Every repetition's samples, played block by block."""
    out = []
    for encoding, rf_phase, adc_phase in zip(
        encodings, rf_phases, adc_phases, strict=True
    ):
        for block in _repetition(tuple(encoding), rf_phase, adc_phase, rewound):
            signal = spins.play(**block)
            if signal.shape[1]:
                out.append(signal)
    return np.stack(out)


def _repeated(spins, encodings, rf_phases, adc_phases, tolerance=0.0, first=3):
    """Every repetition's samples, played as repetitions, the first few alone."""
    areas = np.zeros((len(encodings), 1, 3))
    areas[:, 0, 1:] = encodings
    scan = spins.repetitions(
        _repetition(), rf_phases, areas, adc_phases=adc_phases, tolerance=tolerance
    )
    return np.concatenate([scan.play(first), scan.play()]), scan


def _spoke(angle, rf_phase=0.0, adc_phase=0.0, spoke=SPOKE):
    """:func:`_repetition` unencoded, its readout ``spoke`` turned by ``angle`` about z."""
    pulse, readout, pause = _repetition(rf_phase=rf_phase, adc_phase=adc_phase)
    along = [spoke * [[1.0], [np.cos(angle)]], spoke * [[1.0], [np.sin(angle)]], None]
    return [pulse, {**readout, "duration": SPOKE[0, -1], "gradients": along}, pause]


def _directions(count):
    """``count`` spokes' directions and the next one's, on a spiral over a hemisphere."""
    n = np.arange(count + 1)
    z = 1.0 - (n + 0.5) / (count + 1)
    angle = n * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack(
        [np.sqrt(1 - z * z) * np.cos(angle), np.sqrt(1 - z * z) * np.sin(angle), z]
    )


def _zte_spoke(along, after, rf_phase=0.0, weights=None):
    """A ZTE spoke along ``along``: a hard pulse on the readout gradient, held through its block, then the window under it and the turn onto ``after``; the pulse on channels of ``weights`` where given."""
    rf = pp.make_block_pulse(
        np.deg2rad(10.0),
        duration=10e-6,
        delay=10e-6,
        phase_offset=rf_phase,
        system=SYSTEM,
    )
    if weights is not None:
        rf = SimpleNamespace(
            signal=np.outer(weights, rf.signal).ravel(),
            t=np.tile(rf.t, len(weights)),
            delay=rf.delay,
            freq_offset=rf.freq_offset,
            phase_offset=rf.phase_offset,
        )
    hold = [np.array([[0.0, ZTE_HOLD], [READOUT * a] * 2]) for a in along]
    read = [
        np.array(
            [[0.0, ZTE_READ, ZTE_READ + ZTE_TURN], [READOUT * a] * 2 + [READOUT * b]]
        )
        for a, b in zip(along, after, strict=True)
    ]
    adc = pp.make_adc(
        SAMPLES, dwell=DWELL, delay=20e-6, phase_offset=rf_phase, system=SYSTEM
    )
    return [
        {"duration": ZTE_HOLD, "rf": rf, "gradients": hold},
        {"duration": ZTE_READ + ZTE_TURN, "adc": adc, "gradients": read},
    ]


def _phases(kind, count):
    n = np.arange(count)
    if kind == "alternating":
        return np.pi * n
    if kind == "quadratic":
        return np.deg2rad(117.0) * n * (n + 1) / 2
    return RNG.uniform(-np.pi, np.pi, count)


def _slab(count=48):
    return np.column_stack(
        [
            RNG.uniform(-0.1, 0.1, count),
            RNG.uniform(-0.1, 0.1, count),
            RNG.uniform(-0.002, 0.002, count),
        ]
    )


def _grid(side=8):
    """Isochromats on a lattice in the slice, a column of them per coordinate along y."""
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, side), np.linspace(-0.09, 0.09, side))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(side * side)])


def _lines(count, lines=8):
    """Phase encodings along y, line after line."""
    along_y = (np.arange(count) % lines - lines // 2) * 2.0 / FOV * 10.0
    return np.column_stack([along_y, np.zeros(count)])


@pytest.mark.parametrize("phases", ["alternating", "quadratic", "arbitrary"])
@pytest.mark.parametrize("transmit", [None, "map"])
def test_repetitions_answer_as_their_blocks_played_one_by_one(phases, transmit):
    count = 12
    positions = _slab()
    sensitivities = None if transmit is None else RNG.uniform(0.7, 1.2, len(positions))
    reference, repeated = _engines(_properties(positions, transmit=sensitivities))
    rf_phases = _phases(phases, count)
    adc_phases = rf_phases + RNG.uniform(-0.3, 0.3, count)
    encodings = _lines(count)

    expected = _played(reference, encodings, rf_phases, adc_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, adc_phases)

    assert got.shape == expected.shape == (count, 2, SAMPLES)
    np.testing.assert_allclose(
        got, expected, rtol=0, atol=1e-10 * np.abs(expected).max()
    )
    # The isochromats stand where the blocks played one by one leave them.
    np.testing.assert_allclose(
        repeated.magnetization, reference.magnetization, rtol=0, atol=1e-11
    )
    assert scan.played == len(scan) == count
    assert repeated.elapsed == pytest.approx(reference.elapsed)


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
@pytest.mark.parametrize("count", [1, 50, 83])
def test_any_number_of_isochromats_answers_as_its_blocks_played_one_by_one(
    count, tolerance
):
    """Isochromats are carried a vector's width at a time, the last vector part filled."""
    repetitions = 12
    reference, repeated = _engines(_properties(_slab(count)))
    rf_phases = _phases("quadratic", repetitions)
    encodings = _lines(repetitions)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, _ = _repeated(repeated, encodings, rf_phases, rf_phases, tolerance)

    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < (1e-10 if tolerance == 0.0 else 1e-3)


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
def test_isochromats_shared_between_threads_answer_as_their_blocks_played_one_by_one(
    tolerance,
):
    """Each thread carries isochromats of a few T2s; at 1e-4 the transients below it are dropped and the rest gathered."""
    count = 300
    properties = _properties(_grid(142))
    reference, repeated = _engines({**properties, "threads": 4})
    rf_phases = _phases("alternating", count)
    encodings = _lines(count, 32)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, tolerance)

    assert (scan.carried < len(repeated)) == (tolerance > 0.0)
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < (1e-10 if tolerance == 0.0 else 10 * tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
@pytest.mark.parametrize(
    ("positions", "threads", "spoke"),
    [
        ("lattice", 1, SPOKE),
        ("slab", 1, SPOKE),
        ("lattice", 4, SPOKE),
        ("lattice", 1, OPEN_SPOKE),
        ("slab", 1, OPEN_SPOKE),
    ],
    ids=["lattice", "slab", "threads", "unrewound", "unrewound-slab"],
)
def test_spokes_turned_by_each_repetition_answer_as_their_blocks_played_one_by_one(
    positions, threads, spoke, tolerance
):
    """Each repetition reads its window along its own direction, from the maps of the first's; an unrewound spoke turns each isochromat by the area it leaves."""
    count = 20
    where = {"lattice": _grid(40), "slab": _slab()}[positions]
    reference, repeated = _engines({**_properties(where), "threads": threads})
    angles = np.pi * np.arange(count) / count + 0.3
    rf_phases = _phases("quadratic", count)
    expected = []
    for angle, phase in zip(angles, rf_phases, strict=True):
        for block in _spoke(angle, phase, phase, spoke):
            signal = reference.play(**block)
            if signal.shape[1]:
                expected.append(signal)
    turned = np.zeros((count, 1, 3))
    turned[:, 0, 0] = np.cos(angles) - np.cos(angles[0])
    turned[:, 0, 1] = np.sin(angles) - np.sin(angles[0])

    scan = repeated.repetitions(
        _spoke(angles[0], spoke=spoke),
        rf_phases,
        SPOKE_FIRST * turned,
        readouts=READOUT * turned,
        nets=_area(*spoke) * turned[:, 0],
        tolerance=tolerance,
    )
    got = np.concatenate([scan.play(3), scan.play()])

    error = np.abs(got - np.stack(expected)).max() / np.abs(expected).max()
    assert error < (1e-10 if tolerance == 0.0 else 1e-3)
    np.testing.assert_allclose(
        repeated.magnetization,
        reference.magnetization,
        rtol=0,
        atol=1e-11 if tolerance == 0.0 else tolerance,
    )


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
@pytest.mark.parametrize("transmit", [None, "map", "channels"])
def test_spokes_whose_pulses_play_under_their_own_gradients_answer_as_their_blocks_played_one_by_one(
    transmit, tolerance
):
    """Each spoke's pulse, under its own held gradient, read off the pulse's tables at each isochromat's field; without transmit sensitivities, with a map, and on two channels."""
    count = 20
    positions = RNG.uniform(-0.1, 0.1, size=(200, 3))
    weights = None
    sensitivities = {
        None: None,
        "map": RNG.uniform(0.7, 1.2, len(positions)),
        "channels": RNG.uniform(0.4, 0.7, (len(positions), 2))
        * np.exp(1j * RNG.uniform(-0.5, 0.5, (len(positions), 2))),
    }[transmit]
    if transmit == "channels":
        weights = np.array([1.0, np.exp(0.7j)])
    reference, repeated = _engines(_properties(positions, transmit=sensitivities))
    along = _directions(count)
    rf_phases = _phases("quadratic", count)
    expected = []
    for k, phase in enumerate(rf_phases):
        for block in _zte_spoke(along[k], along[k + 1], phase, weights):
            signal = reference.play(**block)
            if signal.shape[1]:
                expected.append(signal)
    change = READOUT * (along[:-1] - along[0])
    turned = READOUT * (along[1:] - along[1])

    scan = repeated.repetitions(
        _zte_spoke(along[0], along[1], weights=weights),
        rf_phases,
        ZTE_FIRST * change[:, None],
        readouts=change[:, None],
        nets=ZTE_READ * change + ZTE_TURN * (change + turned) / 2,
        pulse_gradients=change,
        tolerance=tolerance,
    )
    got = np.concatenate([scan.play(3), scan.play()])

    # The tables against the pulses stepped for each group of isochromats.
    error = np.abs(got - np.stack(expected)).max() / np.abs(expected).max()
    assert error < (1e-6 if tolerance == 0.0 else 1e-3)
    np.testing.assert_allclose(
        repeated.magnetization,
        reference.magnetization,
        rtol=0,
        atol=1e-6 if tolerance == 0.0 else tolerance,
    )
    assert scan._native.divided is False
    assert repeated.ungrouped_pulses == 0


def _saturation():
    """A pulse of 60 degrees, then a spoiler along z and a pause, played between the repetitions of a run."""
    pulse = pp.make_block_pulse(np.pi / 3, duration=0.2e-3, system=SYSTEM)
    spoiler = np.array([[0.0, 0.1e-3, 0.9e-3, 1.0e-3], [0.0, 4e3, 4e3, 0.0]])
    return [
        {"duration": 0.3e-3, "rf": pulse},
        {"duration": 1.0e-3, "gradients": [None, None, spoiler]},
        {"duration": 2.5e-3},
    ]


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
@pytest.mark.parametrize("phases", ["quadratic", "alternating"])
def test_repetitions_resumed_after_blocks_played_between_them_answer_as_their_blocks_played_one_by_one(
    phases, tolerance
):
    """A saturation played between two halves of a run, which resumes from the magnetisation it leaves; alternating phases on a lattice would split it, which ``split`` declines."""
    count = 16
    half = count // 2
    reference, repeated = _engines(_properties(_grid()))
    rf_phases = _phases(phases, count)
    adc_phases = rf_phases + RNG.uniform(-0.3, 0.3, count)
    encodings = _lines(count)
    expected = [
        _played(reference, encodings[:half], rf_phases[:half], adc_phases[:half])
    ]
    for block in _saturation():
        reference.play(**block)
    expected.append(
        _played(reference, encodings[half:], rf_phases[half:], adc_phases[half:])
    )
    areas = np.zeros((count, 1, 3))
    areas[:, 0, 1:] = encodings

    scan = repeated.repetitions(
        _repetition(),
        rf_phases,
        areas,
        adc_phases=adc_phases,
        tolerance=tolerance,
        split=False,
    )
    got = [scan.play(half)]
    for block in _saturation():
        repeated.play(**block)
    scan.resume()
    got.append(scan.play())

    expected, got = np.concatenate(expected), np.concatenate(got)
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < (1e-10 if tolerance == 0.0 else 1e-3)
    assert scan._steady is None
    np.testing.assert_allclose(
        repeated.magnetization,
        reference.magnetization,
        rtol=0,
        atol=1e-11 if tolerance == 0.0 else tolerance,
    )
    assert repeated.elapsed == pytest.approx(reference.elapsed)


def _turned_about(axis, angle):
    """The rotation by ``angle`` about the unit vector ``axis``."""
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
def test_the_spokes_of_two_shells_resumed_after_the_blocks_between_them_answer_as_their_blocks_played_one_by_one(
    tolerance,
):
    """The spokes of two ZTE shells as one run of repetitions; between them the closing spoke, its gradient ramped to zero, and the ramp onto the next shell's first spoke, played alone."""
    count = 10
    positions = RNG.uniform(-0.1, 0.1, size=(200, 3))
    transmit = RNG.uniform(0.7, 1.2, len(positions))
    reference, repeated = _engines(_properties(positions, transmit=transmit))
    first = _directions(count)
    second = first @ _turned_about(np.array([0.6, 0.0, 0.8]), 0.9).T
    rf_phases = _phases("quadratic", 2 * count + 1)
    closing = _zte_spoke(first[count], np.zeros(3), rf_phases[count])
    ramp = {
        "duration": ZTE_TURN,
        "gradients": [
            np.array([[0.0, ZTE_TURN], [0.0, READOUT * b]]) for b in second[0]
        ],
    }
    between = [*closing, ramp]
    spokes = [(first[k], first[k + 1]) for k in range(count)]
    spokes += [(second[k], second[k + 1]) for k in range(count)]
    phases = np.delete(rf_phases, count)
    expected = []
    for k, (along, after) in enumerate(spokes):
        if k == count:
            for block in between:
                signal = reference.play(**block)
                if signal.shape[1]:
                    expected.append(signal)
        for block in _zte_spoke(along, after, phases[k]):
            signal = reference.play(**block)
            if signal.shape[1]:
                expected.append(signal)
    along = np.array([a for a, _ in spokes])
    after = np.array([b for _, b in spokes])
    change = READOUT * (along - along[0])
    turned = READOUT * (after - after[0])

    scan = repeated.repetitions(
        _zte_spoke(along[0], after[0]),
        phases,
        ZTE_FIRST * change[:, None],
        readouts=change[:, None],
        nets=ZTE_READ * change + ZTE_TURN * (change + turned) / 2,
        pulse_gradients=change,
        tolerance=tolerance,
        split=False,
    )
    got = [scan.play(count)]
    for block in between:
        signal = repeated.play(**block)
        if signal.shape[1]:
            got.append(signal[None])
    scan.resume()
    got.append(scan.play())

    expected, got = np.stack(expected), np.concatenate(got)
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < (1e-6 if tolerance == 0.0 else 1e-3)
    np.testing.assert_allclose(
        repeated.magnetization,
        reference.magnetization,
        rtol=0,
        atol=1e-6 if tolerance == 0.0 else tolerance,
    )


def test_repetitions_split_into_fixed_points_cannot_resume():
    count = 300
    _, repeated = _engines(_properties(_grid()))
    rf_phases = _phases("alternating", count)
    _, scan = _repeated(repeated, _lines(count, 32), rf_phases, rf_phases, 1e-4)

    assert scan._steady is not None
    with pytest.raises(RuntimeError, match="cannot resume"):
        scan.resume()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("no pulse", "plays no pulse"),
        ("window", "read no window"),
        ("ramp", "held through it"),
        ("tables", "cannot be read off tables"),
    ],
)
def test_pulse_gradients_need_a_first_block_whose_pulse_tables_can_serve(
    change, message
):
    """The first block must play the pulse and no window, hold its gradient through it, and need fewer table points than stepping the pulse would cost."""
    count = 4
    along = _directions(count)
    tr = _zte_spoke(along[0], along[1])
    spins = Isochromats(_slab(4))
    if change == "no pulse":
        tr[0].pop("rf")
    elif change == "window":
        tr[0]["adc"] = pp.make_adc(4, dwell=DWELL, delay=35e-6, system=SYSTEM)
        tr[0]["duration"] = 0.2e-3
        for corners in tr[0]["gradients"]:
            corners[0, -1] = 0.2e-3
    elif change == "ramp":
        tr[0]["gradients"][2] = np.array([[0.0, 30e-6, ZTE_HOLD], [0.0, 2e5, 2e5]])
    else:
        # Four isochromats spread over a metre, under a gradient a hundred
        # times as strong: more points than stepping the pulses would take.
        spins = Isochromats(
            np.column_stack([np.linspace(-0.5, 0.5, 4), np.zeros(4), np.zeros(4)])
        )
        along = along * 100.0
    change_of = READOUT * (along[:-1] - along[0])
    with pytest.raises(ValueError, match=message):
        spins.repetitions(
            tr,
            np.zeros(count),
            np.zeros((count, 1 if change != "window" else 2, 3)),
            pulse_gradients=change_of,
        )


@pytest.mark.parametrize("tolerance", [0.0, 1e-4])
def test_phase_encodings_left_unrewound_turn_each_isochromat_by_the_area_they_leave(
    tolerance,
):
    """No fixed point holds under a turn that varies from one repetition to the next: the magnetisation is not split."""
    count = 40
    reference, repeated = _engines(_properties(_grid(40)))
    rf_phases = _phases("alternating", count)
    encodings = _lines(count)
    areas = np.zeros((count, 1, 3))
    areas[:, 0, 1:] = encodings

    expected = _played(reference, encodings, rf_phases, rf_phases, rewound=False)
    scan = repeated.repetitions(
        _repetition(rewound=False),
        rf_phases,
        areas,
        nets=areas[:, 0],
        tolerance=tolerance,
    )
    got = np.concatenate([scan.play(3), scan.play()])

    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < (1e-10 if tolerance == 0.0 else 1e-3)
    assert scan._steady is None
    np.testing.assert_allclose(
        repeated.magnetization,
        reference.magnetization,
        rtol=0,
        atol=1e-11 if tolerance == 0.0 else tolerance,
    )


@pytest.mark.parametrize("tolerance", [1e-8, 1e-4])
def test_a_split_holds_every_repetition_to_the_tolerance(tolerance):
    """Fixed points summed by columns, and transients carried until they fall below the tolerance."""
    count = 300
    reference, repeated = _engines(_properties(_grid()))
    rf_phases = _phases("alternating", count)
    encodings = _lines(count, 32)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, tolerance)

    assert scan._steady is not None
    # Transients fall below 1e-4 within the scan, and below 1e-8 after it.
    assert (scan.carried < len(repeated)) == (tolerance > 1e-6)
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < 10 * tolerance
    np.testing.assert_allclose(
        repeated.magnetization, reference.magnetization, rtol=0, atol=tolerance
    )


def test_phases_that_do_not_step_evenly_are_carried_whole():
    count = 40
    reference, repeated = _engines(_properties(_grid()))
    rf_phases = _phases("quadratic", count)
    encodings = _lines(count)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, 1e-4)

    assert scan._steady is None
    assert scan.carried == len(repeated)
    assert np.abs(got - expected).max() < 1e-3 * np.abs(expected).max()


def test_phases_rounded_as_a_sequence_file_holds_them_split_on_their_mean_step():
    count = 200
    reference, repeated = _engines(_properties(_grid()))
    rf_phases = np.round(_phases("alternating", count), 5)
    encodings = _lines(count, 16)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, 1e-4)

    assert scan._steady is not None
    assert np.abs(got - expected).max() < 1e-3 * np.abs(expected).max()


@pytest.mark.parametrize("along", ["y", "y and z"])
def test_the_fixed_points_of_isochromats_on_a_lattice_are_read_by_columns(along):
    """Each column's readout summed per isochromat, then the phase encodings over the columns."""
    x, y, z = np.meshgrid(
        np.linspace(-0.1, 0.1, 6), np.linspace(-0.09, 0.09, 4), [-0.002, 0.0, 0.002]
    )
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    reference, repeated = _engines(_properties(positions))
    count = 240
    lines = (np.arange(count) % 4 - 2) / FOV
    partitions = (np.arange(count) // 4 % 3 - 1) * 60.0 if along == "y and z" else 0.0
    encodings = np.column_stack([lines, np.zeros(count) + partitions])
    rf_phases = _phases("alternating", count)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, 1e-8)

    assert scan._steady is not None
    error = np.abs(got - expected).max() / np.abs(expected).max()
    assert error < 1e-6


def test_isochromats_off_a_lattice_are_carried_whole():
    """Without columns along the phase encoding, no split: every repetition is carried exactly."""
    count = 40
    reference, repeated = _engines(_properties(_slab()))
    rf_phases = _phases("alternating", count)
    encodings = _lines(count)

    expected = _played(reference, encodings, rf_phases, rf_phases)
    got, scan = _repeated(repeated, encodings, rf_phases, rf_phases, 1e-8)

    assert scan._steady is None
    assert np.abs(got - expected).max() < 1e-6 * np.abs(expected).max()


def test_repetitions_without_a_window_leave_the_isochromats_as_their_blocks_do():
    """A split of repetitions that read nothing still carries the transients."""
    count = 300
    reference, repeated = _engines(_properties(_grid()))
    rf_phases = _phases("alternating", count)
    silent = [{k: v for k, v in block.items() if k != "adc"} for block in _repetition()]
    for phase in rf_phases:
        for block in _repetition(rf_phase=phase):
            reference.play(**{k: v for k, v in block.items() if k != "adc"})
    scan = repeated.repetitions(silent, rf_phases, tolerance=1e-4)

    assert scan.play().shape == (count, 2, 0)
    assert scan._steady == []
    np.testing.assert_allclose(
        repeated.magnetization, reference.magnetization, rtol=0, atol=1e-4
    )


def test_a_window_of_one_sample_is_read_as_played():
    count = 20
    reference, repeated = _engines(_properties(_slab(), coils=1))
    rf_phases = _phases("arbitrary", count)
    tr = [
        {"duration": 1e-3, "rf": pp.make_block_pulse(np.pi / 5, duration=0.2e-3)},
        {"duration": 3e-3, "adc": pp.make_adc(1, dwell=10e-6, delay=1e-3)},
    ]
    expected = []
    for phase in rf_phases:
        reference.play(
            **tr[0]
            | {
                "rf": pp.make_block_pulse(
                    np.pi / 5, duration=0.2e-3, phase_offset=phase
                )
            }
        )
        expected.append(
            reference.play(
                **tr[1]
                | {"adc": pp.make_adc(1, dwell=10e-6, delay=1e-3, phase_offset=phase)}
            )
        )
    got = repeated.repetitions(tr, rf_phases).play()
    np.testing.assert_allclose(got, np.stack(expected), rtol=0, atol=1e-11)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"areas": np.zeros((4, 2, 3))}, "areas"),
        ({"adc_phases": np.zeros(3)}, "adc_phases"),
    ],
)
def test_repetitions_that_do_not_fit_are_refused(change, message):
    spins = Isochromats(_slab(4))
    arguments = {"areas": np.zeros((4, 1, 3)), "adc_phases": None} | change
    with pytest.raises(ValueError, match=message):
        spins.repetitions(_repetition(), np.zeros(4), **arguments)


def test_a_window_not_read_under_a_held_gradient_is_refused():
    spins = Isochromats(_slab(4))
    tr = _repetition()
    tr[1]["gradients"][0] = np.array([[0.0, 1e-3], [0.0, 2e4]])
    with pytest.raises(ValueError, match="held"):
        spins.repetitions(tr, np.zeros(4))
