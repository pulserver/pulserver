"""Isochromats that move with their subject, diffuse, and spread over their voxel and their T2' line."""

import math

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual
from pulserver.virtual import _voxels
from pulserver.virtual._bloch import Player

#: 40 mT/m, in Hz/m.
STRONG = 40e-3 * pp.Opts().gamma


def _transverse(spins) -> np.ndarray:
    m = spins.magnetization
    return m[:, 0] + 1j * m[:, 1]


def _lobe(sign, duration, ramp=0.0):
    """A gradient lobe along x of ``STRONG`` Hz/m, flat or with ramps."""
    if ramp == 0.0:
        return [np.array([[0.0, duration], [sign * STRONG] * 2]), None, None]
    times = [0.0, ramp, duration - ramp, duration]
    return [np.array([times, [0.0, sign * STRONG, sign * STRONG, 0.0]]), None, None]


def _b_value(waveform, duration):
    """b, in s/m², of a gradient ``waveform(t)`` in Hz/m starting from no dephasing."""
    t = np.linspace(0.0, duration, 400001)
    k = np.concatenate(
        [[0.0], np.cumsum(0.5 * (waveform(t[1:]) + waveform(t[:-1])) * np.diff(t))]
    )
    return np.trapezoid((2.0 * math.pi * k) ** 2, t)


def test_isochromats_moved_precess_where_they_stood_and_then_where_they_are():
    stood = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, -0.01]])
    spins = virtual.Isochromats(stood)
    spins.magnetization = [1.0, 0.0, 0.0]
    gradients = [np.array([[0.0, 1e-3], [1e3, 1e3]]), None, None]

    spins.play(1e-3, gradients=gradients)
    spins.positions = stood + 0.005
    spins.play(1e-3, gradients=gradients)

    moved = stood + 0.005
    expected = np.exp(-2j * math.pi * (stood[:, 0] + moved[:, 0]))
    np.testing.assert_allclose(_transverse(spins), expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(spins.positions, moved)


def test_isochromats_given_a_motion_or_a_diffusion_are_not_placed_by_hand():
    for given in ({"motion": virtual.RigidMotion()}, {"diffusion": 1e-9}):
        spins = virtual.Isochromats(np.zeros((2, 3)), **given)
        assert spins.moving
        with pytest.raises(ValueError, match="placed by them"):
            spins.positions = np.ones((2, 3))
        with pytest.raises(ValueError, match="block by block"):
            spins.repetitions([{"duration": 1e-3}], [0.0])
    assert not virtual.Isochromats(np.zeros((2, 3))).moving


@pytest.mark.parametrize(
    "motion",
    [
        virtual.RigidMotion(offset=lambda t: [0.05 * t, 0.0, 0.0]),
        lambda t, positions: positions + np.array([0.05 * t, 0.0, 0.0]),
    ],
    ids=["rigid", "callable"],
)
def test_a_spin_moving_through_a_bipolar_gradient_accrues_the_phase_of_its_first_moment(
    motion,
):
    tau, speed = 1e-3, 0.05
    bipolar = [np.array([[0.0, tau, tau, 2 * tau], [2e3, 2e3, -2e3, -2e3]]), None, None]
    spins = virtual.Isochromats([[0.0, 0.0, 0.0]], motion=motion)
    spins.magnetization = [1.0, 0.0, 0.0]

    spins.play(2 * tau, gradients=bipolar)

    # The integral of the gradient times the position is -2e3 tau**2 speed.
    expected = np.exp(-2j * math.pi * (-2e3 * tau**2 * speed))
    np.testing.assert_allclose(_transverse(spins), [expected], rtol=0, atol=1e-12)


def test_a_moving_subject_is_placed_where_it_is_at_each_block_s_start():
    def turned(t):
        angle = 0.3 * t
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    rest = np.array([[0.02, 0.0, 0.0], [0.0, -0.01, 0.03]])
    motion = virtual.RigidMotion(
        turned, lambda t: [0.0, 0.0, 1e-3 * t], centre=(0.0, 0.01, 0.0)
    )
    spins = virtual.Isochromats(rest, motion=motion)

    spins.play(0.5)
    spins.play(0.25)

    centre = np.array([0.0, 0.01, 0.0])
    expected = (rest - centre) @ turned(0.5).T + centre + [0.0, 0.0, 0.5e-3]
    np.testing.assert_allclose(spins.positions, expected, rtol=0, atol=1e-15)


def _spiral_lattice(seed=3):
    """Isochromats two to a point of a 1 mm lattice in x and y, anywhere in z, and a spiral window over them."""
    rng = np.random.default_rng(seed)
    lattice = 1e-3 * np.indices((32, 32)).reshape(2, -1).T - 16e-3
    rest = np.repeat(np.column_stack([lattice, np.zeros(len(lattice))]), 2, axis=0)
    rest[:, 2] = rng.uniform(-1e-3, 1e-3, len(rest))
    properties = {"t2": 0.05, "off_resonance": rng.uniform(-30.0, 30.0, len(rest))}
    t = np.linspace(0.0, 4e-3, 80)
    spiral = 2e4 * np.exp(1j * 3e3 * t) * (1.0 + 1j * 3e3 * t)
    window = {
        "gradients": [np.array([t, spiral.real]), np.array([t, spiral.imag]), None],
        "adc": np.linspace(0.1e-3, 3.9e-3, 256),
    }
    return rest, properties, window


def test_a_subject_moved_between_windows_is_read_on_the_lattice_it_was_moved_onto():
    rest, properties, window = _spiral_lattice()
    shift = np.array([1.5e-3, -2.25e-3, 0.0])
    moving = virtual.Isochromats(
        rest,
        motion=virtual.RigidMotion(offset=lambda t: shift * (t >= 4e-3)),
        **properties,
    )
    exact = virtual.Isochromats(rest, **properties)
    read = []
    for spins, tolerance in ((moving, 1e-6), (exact, 0.0)):
        spins.magnetization = [0.0, 1.0, 0.0]
        spins.play(4e-3, **window, tolerance=tolerance)
        if spins is exact:
            spins.positions = rest + shift
        read.append(spins.play(4e-3, **window, tolerance=tolerance))

    assert moving.lattice_windows == 2
    terms = np.exp(-8e-3 / 0.05) * len(rest)
    np.testing.assert_allclose(read[0], read[1], rtol=0, atol=1e-6 * terms)


def _pgse(spins, ramp, refocused):
    """Diffusion-encode ``spins`` along x: two lobes 40 ms apart, of a pulse each side of a 180 or of each sign."""
    delta, gap = 20e-3, 20e-3
    spins.play(delta, gradients=_lobe(1, delta, ramp))
    if refocused:
        spins.play(gap, rf=(0.0, 0.1e-3, [5e3]))
        spins.play(delta, gradients=_lobe(1, delta, ramp))
    else:
        spins.play(gap)
        spins.play(delta, gradients=_lobe(-1, delta, ramp))


@pytest.mark.parametrize("ramp", [0.0, 2e-3], ids=["flat", "ramped"])
@pytest.mark.parametrize("refocused", [False, True], ids=["bipolar", "spin echo"])
def test_isochromats_diffusing_through_a_diffusion_encoding_lose_signal_as_exp_minus_b_d(
    ramp, refocused
):
    count, diffusion = 40000, 0.7e-9
    spins = virtual.Isochromats(np.zeros((count, 3)), diffusion=diffusion, seed=4)
    spins.magnetization = [1.0, 0.0, 0.0]

    _pgse(spins, ramp, refocused)

    def waveform(t):
        lobe = np.interp(t, [0.0, ramp, 20e-3 - ramp, 20e-3], [0.0, 1.0, 1.0, 0.0])
        later = np.interp(
            t - 40e-3, [0.0, ramp, 20e-3 - ramp, 20e-3], [0.0, 1.0, 1.0, 0.0]
        )
        return STRONG * (
            np.where(t <= 20e-3, lobe, 0.0) - np.where(t >= 40e-3, later, 0.0)
        )

    expected = math.exp(-_b_value(waveform, 60e-3) * diffusion)
    signal = abs(_transverse(spins).mean())
    # Four standard errors of the mean of count unit phasors.
    assert abs(signal - expected) < 4.0 / math.sqrt(2.0 * count)


def test_isochromats_that_diffuse_keep_their_positions_and_their_lattice():
    rest, properties, window = _spiral_lattice()
    spins = virtual.Isochromats(rest, diffusion=2e-9, seed=0, **properties)
    spins.magnetization = [0.0, 1.0, 0.0]

    spins.play(4e-3, **window, tolerance=1e-6)
    spins.play(4e-3, **window, tolerance=1e-6)

    np.testing.assert_array_equal(spins.positions, rest)
    assert spins.lattice_windows == 2


def test_a_reset_starts_each_diffusing_isochromat_on_a_new_walk_and_a_seed_repeats_it():
    played = []
    for _ in range(2):
        spins = virtual.Isochromats(np.zeros((50, 3)), diffusion=1e-9, seed=11)
        for _ in range(2):
            spins.reset()
            spins.magnetization = [1.0, 0.0, 0.0]
            _pgse(spins, 0.0, refocused=False)
            played.append(_transverse(spins))
    np.testing.assert_array_equal(played[0], played[2])
    assert not np.array_equal(played[0], played[1])


def test_a_voxel_of_spins_decays_with_its_t2_prime_and_refocuses_in_a_spin_echo():
    disk = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.05, 0.05), t2_prime=0.05)]
    )
    spins = disk.isochromats(2e-3, spins=8, seed=1)
    t = np.array([0.0, 0.01, 0.05, 0.1])

    spins.magnetization = [0.0, 1.0, 0.0]
    free = np.abs(spins.play(0.1, adc=t)[0]) / len(spins)
    spins.magnetization = [0.0, 1.0, 0.0]
    spins.play(0.025)
    spins.play(1e-4, rf=(0.0, 1e-4, [5e3]))
    echo = abs(spins.play(0.0249, adc=[0.0249])[0, 0]) / len(spins)

    # The cut at 32 half widths takes 2% of the line.
    np.testing.assert_allclose(free, np.exp(-t / 0.05), rtol=0, atol=0.03)
    assert echo == pytest.approx(1.0, abs=1e-3)


def test_one_spin_per_voxel_or_no_t2_prime_leaves_each_voxel_at_its_frequency():
    ellipse = virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01), t2_prime=0.05)
    alone = virtual.Phantom([ellipse]).isochromats(2e-3)
    still = virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01))]
    ).isochromats(2e-3, spins=4)
    assert len(still) == 4 * len(alone)
    for spins in (alone, still):
        spins.magnetization = [0.0, 1.0, 0.0]
        signal = spins.play(0.1, adc=[0.1])
        assert abs(signal[0, 0]) == pytest.approx(len(spins), rel=1e-12)


def test_the_line_s_strata_shifted_per_voxel_decay_as_exp_minus_t_over_t2_prime_on_average():
    t2_prime = np.full(20000, 0.04)
    rng = np.random.default_rng(5)
    order = np.arange(8)
    frequencies = _voxels.frequencies(t2_prime, order, rng).reshape(-1, 8)
    t = np.array([0.005, 0.02, 0.04, 0.08, 0.16])

    mean = np.cos(2.0 * math.pi * frequencies[..., None] * t).mean(axis=(0, 1))

    # The cut at 32 half widths takes 2% of the line.
    np.testing.assert_allclose(mean, np.exp(-t / 0.04), rtol=0, atol=0.03)
    assert np.abs(frequencies).max() <= _voxels.CUT / (2.0 * math.pi * 0.04)


def test_a_box_voxel_dephases_under_a_cycle_across_it_where_a_point_voxel_does_not():
    ellipse = virtual.Ellipse((0.0, 0.0, 0.0), (0.004, 0.004))
    phantom = virtual.Phantom([ellipse])
    spacing = 2e-3
    read = {}
    for voxel, spins in (("point", 1), ("box", 9)):
        iso = phantom.isochromats(spacing, spins=spins, voxel=voxel, seed=0)
        iso.magnetization = [0.0, 1.0, 0.0]
        # One cycle across a voxel along x.
        crusher = [np.array([[0.0, 1e-3], [1.0 / spacing / 1e-3] * 2]), None, None]
        read[voxel] = iso.play(1e-3, gradients=crusher, adc=[1e-3])[0, 0]

    assert abs(read["point"]) == pytest.approx(len(phantom.isochromats(spacing)))
    assert abs(read["box"]) < 1e-12


@pytest.mark.parametrize(
    "given, message",
    [
        ({"spins": 0}, "one isochromat or more"),
        ({"spins": 8, "voxel": "box"}, "whole number of cells"),
        ({"voxel": "sinc"}, "a voxel is one of"),
    ],
)
def test_spins_that_do_not_fill_a_voxel_are_refused(given, message):
    phantom = virtual.Phantom([virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01))])
    with pytest.raises(ValueError, match=message):
        phantom.isochromats(2e-3, **given)


def test_brainweb_refuses_t2_prime_or_diffusion_of_a_tissue_it_does_not_have():
    with pytest.raises(ValueError, match="no tissue class"):
        virtual.BrainWeb(t2_prime={"cortex": 0.05})
    with pytest.raises(ValueError, match="no tissue class"):
        virtual.BrainWeb(diffusion={"grey": 1e-9})


@pytest.fixture(scope="module")
def balanced(tmp_path_factory):
    path = tmp_path_factory.mktemp("bssfp") / "scan.seq"
    sequences.bssfp3D_sequence(**SMALL["bssfp3D_sequence"]).write(str(path))
    ir.convert(path, pp.Opts())
    return path


def test_the_virtual_scanner_plays_isochromats_that_move_block_by_block(balanced):
    rest = np.random.default_rng(1).uniform(-0.05, 0.05, (20, 3))
    still = Player(balanced, virtual.Isochromats(rest, t2=0.1))
    moving = Player(
        balanced, virtual.Isochromats(rest, t2=0.1, motion=virtual.RigidMotion())
    )

    assert still.runs
    assert moving.runs == []
    for a, b in zip(
        still.readouts(0, still.blocks), moving.readouts(0, moving.blocks), strict=True
    ):
        # A run scales one repetition's gradient shape by each repetition's
        # amplitude, which the cache holds in single precision.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6 * len(rest))
