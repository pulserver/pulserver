"""RF pulses played from tables of the field and drive isochromats see, against the same pulses stepped for each isochromat."""

from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver.virtual import Isochromats

RNG = np.random.default_rng(11)


GRID_PULSES = pytest.mark.parametrize(
    "pulses",
    [
        [(np.pi / 2, 0.0, 0.0), (np.pi / 2, 1.3, 0.0)],
        [(np.pi, 0.0, 0.0)],
        [(np.pi / 6, 0.0, 0.0), (np.pi / 6, 0.0, 4e3)],
    ],
    ids=["phase", "refocusing", "offset"],
)


def _on_channels(rf, weights, cancelled=False):
    """``rf`` played on channels of ``weights`` times its waveform.

    With ``cancelled``, a chirp is added to the first channel and taken away
    on one more, so that no one waveform times a weight per channel describes
    the channels, while a sensitivity of the first channel's on the last
    leaves the field unchanged.
    """
    signal = np.asarray(rf.signal, dtype=complex)
    channels = np.outer(weights, signal)
    if cancelled:
        chirp = (
            0.3
            * np.abs(signal).max()
            * np.exp(1j * np.linspace(0.0, 6.0, signal.size) ** 2)
        )
        channels[0] += chirp
        channels = np.vstack([channels, -chirp])
    return SimpleNamespace(
        signal=channels.ravel(),
        t=np.tile(np.asarray(rf.t, dtype=float), channels.shape[0]),
        delay=rf.delay,
        freq_offset=rf.freq_offset,
        phase_offset=rf.phase_offset,
    )


def _along(gz, direction):
    """The corner points, per axis, of slice gradient ``gz`` played along ``direction``."""
    rise, flat, fall = gz.rise_time, gz.flat_time, gz.fall_time
    times = gz.delay + np.array([0.0, rise, rise + flat, rise + flat + fall])
    return [
        np.array([times, [0.0, a, a, 0.0]]) if a else None
        for a in gz.amplitude * np.asarray(direction, dtype=float)
    ]


def _slice_pulses(
    spins, pulses, *channels, cancelled=False, directions=((0, 0, 1),), thickness=5e-3
):
    """Play sinc pulses of ``(flip, phase, offset)`` under their slice gradients along each of ``directions`` in turn, each on ``channels`` when given."""
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    for direction in directions:
        for flip, phase, offset in pulses:
            rf, gz, _ = pp.make_sinc_pulse(
                flip,
                duration=2e-3,
                slice_thickness=thickness,
                phase_offset=phase,
                freq_offset=offset,
                return_gz=True,
                system=system,
            )
            spins.play(
                pp.calc_duration(rf, gz),
                gradients=_along(gz, direction),
                rf=_on_channels(rf, *channels, cancelled=cancelled) if channels else rf,
                system=system,
            )
            spins.play(2e-3)
    return spins.magnetization


def _grid_isochromats(n=20000):
    """Positions and properties that give each isochromat its own field during a slice's pulse."""
    positions = np.column_stack(
        [RNG.uniform(-0.1, 0.1, (n, 2)), RNG.uniform(-0.02, 0.02, n)]
    )
    properties = {
        "t1": RNG.choice([0.8, 1.2], n),
        "t2": 0.08,
        "off_resonance": RNG.uniform(-150.0, 150.0, n),
    }
    return positions, properties


@GRID_PULSES
def test_a_pulse_computed_on_a_grid_of_fields_answers_as_one_stepped_for_each_isochromat(
    pulses,
):
    positions, properties = _grid_isochromats()
    n = len(positions)
    # Every isochromat sees its own field, so the grid costs fewer maps than
    # stepping each; channels no one waveform describes step each instead.
    on_grid = _slice_pulses(Isochromats(positions, **properties), pulses)
    stepped = _slice_pulses(
        Isochromats(positions, transmit=np.ones((n, 2)), **properties),
        pulses,
        [1.0],
        cancelled=True,
    )

    np.testing.assert_allclose(on_grid, stepped, rtol=0, atol=1e-6)


def _transmit(n):
    """Two channels' sensitivities of a phase common to both, uniform over the circle, and magnitudes and a phase between them within 0.1 of one and of zero.

    A shim's drive then spans a few of a table's rows, so that the table holds
    fewer new points than there are isochromats.
    """
    common = np.exp(1j * RNG.uniform(-np.pi, np.pi, (n, 1)))
    return (
        common
        * RNG.uniform(0.9, 1.1, (n, 2))
        * np.exp(1j * RNG.uniform(-0.1, 0.1, (n, 2)))
    )


SHIM = np.array([0.8 * np.exp(0.4j), 0.5 * np.exp(-1.1j)])


@GRID_PULSES
def test_a_pulse_on_transmit_sensitivities_computed_on_a_grid_of_fields_and_drives_answers_as_one_stepped(
    pulses,
):
    """Each isochromat's drive of the channels' one waveform: its magnitude a second axis of the grid, its phase a turn of the map."""
    positions, properties = _grid_isochromats()
    transmit = _transmit(len(positions))
    on_grid = _slice_pulses(
        Isochromats(positions, transmit=transmit, **properties),
        pulses,
        SHIM,
        thickness=0.03,
    )
    stepped = _slice_pulses(
        Isochromats(
            positions,
            transmit=np.column_stack([transmit, transmit[:, 0]]),
            **properties,
        ),
        pulses,
        SHIM,
        cancelled=True,
        thickness=0.03,
    )

    np.testing.assert_allclose(on_grid, stepped, rtol=0, atol=1e-6)


def _spokes(count):
    """``count`` unit directions spread over the sphere, none along an axis."""
    k = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * k / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * k
    return np.column_stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ]
    )


@GRID_PULSES
@pytest.mark.parametrize("driven", [False, True], ids=["summed", "driven"])
def test_pulses_under_more_slab_directions_than_groupings_kept_answer_as_stepped_ones(
    pulses, driven
):
    """Six slab directions: the four the engine keeps groupings for, then two whose pulses are played isochromat by isochromat from their tables."""
    positions, properties = _grid_isochromats()
    n = len(positions)
    if driven:
        transmit = _transmit(n)
        shim = SHIM
        stepped = np.column_stack([transmit, transmit[:, 0]])
    else:
        transmit, shim, stepped = None, None, np.ones((n, 2))
    spins = Isochromats(positions, transmit=transmit, **properties)
    on_grid = _slice_pulses(
        spins,
        pulses,
        *([] if shim is None else [shim]),
        directions=_spokes(6),
        thickness=0.15,
    )
    each_stepped = _slice_pulses(
        Isochromats(positions, transmit=stepped, **properties),
        pulses,
        [1.0] if shim is None else shim,
        cancelled=True,
        directions=_spokes(6),
        thickness=0.15,
    )

    assert spins.ungrouped_pulses == 2 * len(pulses)
    np.testing.assert_allclose(on_grid, each_stepped, rtol=0, atol=1e-6)
