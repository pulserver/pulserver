"""The excitations a cache lets the scanner move by a carrier offset, and the gradient each plays under."""

import math
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir
from pulserver.mrd import read_chain

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(B0=3.0)
# ZTE turns every spoke, its excitation's too, so no position plays one gradient.
UNFLAGGED = {"zte_3d.seq", "zte3D_sequence"}
Rotation = pytest.importorskip("scipy.spatial.transform").Rotation


def _pypulseqpp(path):
    """Per block of the chain, in play order: whether its pulse is steady on every axis, and the gradient there."""
    steady, gradient = [], []
    for _, sequence in read_chain(path):
        count = len(sequence.block_events)
        under = sequence.rf_gradients()
        s = np.zeros(count, dtype=bool)
        g = np.zeros((count, 3))
        s[under.block - 1] = under.steady.all(axis=1)
        g[under.block - 1] = under.gradient
        steady.append(s)
        gradient.append(g)
    return np.concatenate(steady), np.concatenate(gradient)


def _assert_flags_follow_pypulseqpp(path, name):
    ir.convert(path, SYSTEM)
    played = ir.play(path)
    steady, gradient = _pypulseqpp(path)
    flagged = played["rf_grad_constant"] == 1
    assert bool(flagged.any()) is (name not in UNFLAGGED)
    assert steady[flagged].all()
    np.testing.assert_allclose(
        (played["rf_grad_level"] * played["gradient_hz_per_m"])[flagged],
        gradient[flagged],
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "name", sorted(p.name for p in FIXTURES.glob("*.seq") if not p.stem.endswith("_b"))
)
def test_a_flagged_excitation_plays_under_the_steady_gradient_pypulseqpp_finds(
    name, tmp_path
):
    for fixture in FIXTURES.glob("*.seq"):
        (tmp_path / fixture.name).write_bytes(fixture.read_bytes())
    _assert_flags_follow_pypulseqpp(tmp_path / name, name)


@pytest.mark.parametrize("name", sorted(SMALL))
def test_every_shipped_sequence_flags_only_the_excitations_pypulseqpp_finds_steady(
    name, tmp_path
):
    path = tmp_path / "scan.seq"
    getattr(sequences, name)(**SMALL[name]).write(str(path))
    _assert_flags_follow_pypulseqpp(path, name)


def _selective():
    pulse, gz, _ = pp.make_sinc_pulse(
        math.pi / 2,
        duration=2e-3,
        slice_thickness=5e-3,
        return_gz=True,
        use="excitation",
        system=SYSTEM,
    )
    return pulse, gz


def _played(tmp_path, *blocks):
    seq = pp.Sequence(SYSTEM)
    for events in blocks:
        seq.add_block(*events)
    path = tmp_path / "scan.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return ir.play(path)


def test_a_slice_selective_excitation_is_flagged_at_its_plateau(tmp_path):
    pulse, gz = _selective()

    played = _played(tmp_path, (pulse, gz), (pp.make_delay(1e-3),))

    assert played["rf_grad_constant"].tolist() == [1, 0]
    np.testing.assert_allclose(played["rf_grad_level"][0], [0.0, 0.0, 1.0])


def test_an_excitation_under_a_gradient_that_ends_under_it_is_not_flagged(tmp_path):
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=2e-3, delay=1e-4, use="excitation", system=SYSTEM
    )
    gz = pp.make_trapezoid(
        "z", amplitude=1000, flat_time=1e-3, rise_time=1e-4, system=SYSTEM
    )

    played = _played(tmp_path, (pulse, gz))

    assert played["rf_grad_constant"].tolist() == [0]


def test_an_excitation_in_a_rotated_block_is_not_flagged(tmp_path):
    """The gradient it plays is the rotation of the recorded one: steady, but not that vector."""
    pulse, gz = _selective()
    turn = pp.make_rotation(Rotation.from_euler("x", 30, degrees=True))

    played = _played(tmp_path, (pulse, gz, turn))

    assert played["rf_grad_constant"].tolist() == [0]


def test_a_position_steady_in_one_instance_and_not_in_another_is_not_flagged(tmp_path):
    """One definition plays both waveforms, so the position has no single gradient."""
    raster = SYSTEM.grad_raster_time
    ramp = np.linspace(0.0, 1.0, 10)
    flat = np.concatenate((ramp, np.ones(80), ramp[::-1])) * 2e4
    sloped = np.concatenate((ramp, np.linspace(1.0, 0.5, 80), 0.5 * ramp[::-1])) * 2e4
    pulse = pp.make_block_pulse(
        math.pi / 2,
        duration=60 * raster,
        delay=20 * raster,
        use="excitation",
        system=SYSTEM,
    )
    steady, moving = (
        pp.make_arbitrary_grad("z", waveform, first=0, last=0, system=SYSTEM)
        for waveform in (flat, sloped)
    )

    played = _played(
        tmp_path,
        (pulse, steady),
        (pp.make_delay(1e-3),),
        (pulse, moving),
        (pp.make_delay(1e-3),),
    )

    assert played["segment"][0] == played["segment"][2]
    assert played["rf_grad_constant"].tolist() == [0, 0, 0, 0]
