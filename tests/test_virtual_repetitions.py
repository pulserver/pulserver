"""Repetitions of a sequence of blocks against the same blocks played one by one."""

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


def _repetition(encoding=(0.0, 0.0), rf_phase=0.0, adc_phase=0.0):
    """A slice-selective pulse; a readout prephaser and phase encodings along y and z; a readout; the rewinders.

    ``encoding`` is each phase encoding's area, in 1/m.
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
    rewind = [ENCODE * [[1.0], [-a / ENCODE_AREA]] if a else None for a in encoding]
    return [
        {"duration": 1.2e-3, "rf": rf, "gradients": [None, None, select]},
        {"duration": PRE + span + 0.05e-3, "adc": adc, "gradients": [gx, *phase]},
        {"duration": 0.5e-3, "gradients": [None, *rewind]},
    ]


def _played(spins, encodings, rf_phases, adc_phases):
    """Every repetition's samples, played block by block."""
    out = []
    for encoding, rf_phase, adc_phase in zip(
        encodings, rf_phases, adc_phases, strict=True
    ):
        for block in _repetition(tuple(encoding), rf_phase, adc_phase):
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
