"""The isochromat Bloch engine against closed forms and the relaxation-free kernel."""

from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import design_samples

from pulserver.virtual import Isochromats

RNG = np.random.default_rng(11)
ORIGIN = [[0.0, 0.0, 0.0]]


def _hard(flip_deg, phase=0.0, step=1e-7):
    """A pulse of one short step turning by ``flip_deg`` about ``phase`` from +x."""
    amplitude = flip_deg / 360.0 / step
    return (0.0, step, np.array([amplitude * np.exp(1j * phase)]))


def test_a_pulse_turns_as_the_mirror_of_the_right_handed_kernel():
    """Clockwise about (Re b1, Im b1, bz) is counter-clockwise about its mirror image."""
    b1 = 150.0 * (RNG.normal(size=300) + 1j * RNG.normal(size=300))
    off_resonance = RNG.normal(scale=400.0, size=6)
    spins = Isochromats(np.zeros((6, 3)), off_resonance=off_resonance)
    spins.play(300e-6, rf=(0.0, 1e-6, b1))
    mirrored = pp.sim_bloch(np.conj(b1), off_resonance[:, None], 1e-6)
    np.testing.assert_allclose(
        spins.magnetization, mirrored * [1.0, -1.0, 1.0], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("kind", ["none", "one direction", "crossing"])
def test_a_pulse_under_a_gradient_turns_each_isochromat_about_its_own_field(kind):
    """Each step's field is the mean of the isochromat's over the step."""
    steps, step = 200, 2e-6
    b1 = 200.0 * (RNG.normal(size=steps) + 1j * RNG.normal(size=steps))
    times = np.linspace(0.0, steps * step, 7)
    ramp = np.array([0.0, 4e4, 1e4, 3e4, -2e4, 1e4, 0.0])
    other = {"none": 0.0 * ramp, "one direction": -0.5 * ramp, "crossing": ramp[::-1]}[
        kind
    ]
    gradients = {"none": [None, None, None]}.get(
        kind, [np.array([times, ramp]), np.array([times, other]), None]
    )
    count = 12
    positions = np.column_stack([RNG.uniform(-0.05, 0.05, (count, 2)), np.zeros(count)])
    off_resonance = RNG.normal(scale=100.0, size=count)
    spins = Isochromats(positions, off_resonance=off_resonance)
    spins.play(steps * step, gradients=gradients, rf=(0.0, step, b1))

    edges = np.arange(steps + 1) * step
    areas = np.zeros((steps + 1, 3))
    if kind != "none":
        areas[:, 0] = [_area(times, ramp, t) for t in edges]
        areas[:, 1] = [_area(times, other, t) for t in edges]
    bz = (np.diff(areas, axis=0) @ positions.T).T / step + off_resonance[:, None]
    mirrored = pp.sim_bloch(np.conj(b1), bz, step)
    np.testing.assert_allclose(
        spins.magnetization, mirrored * [1.0, -1.0, 1.0], rtol=0, atol=1e-11
    )


def test_ninety_degrees_about_x_takes_z_to_y():
    spins = Isochromats(ORIGIN)
    spins.play(1e-3, rf=_hard(90.0))
    np.testing.assert_allclose(spins.magnetization, [[0.0, 1.0, 0.0]], atol=1e-12)


def test_free_precession_is_exact_under_a_piecewise_linear_gradient():
    x, t2, t1, off_resonance = 0.03, 0.05, 0.4, 37.0
    spins = Isochromats([[x, 0.0, 0.0]], t1=t1, t2=t2, off_resonance=off_resonance)
    spins.magnetization = [0.0, 1.0, 0.0]
    ramp = np.array([[0.2e-3, 0.7e-3, 1.9e-3, 2.3e-3], [0.0, 3e4, -1e4, 0.0]])
    duration = 3e-3
    spins.play(duration, gradients=[ramp, None, None])
    times, values = ramp
    area = np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(times))
    phase = -2 * np.pi * (x * area + off_resonance * duration)
    expected = 1j * np.exp(1j * phase - duration / t2)
    m = spins.magnetization[0]
    assert m[0] + 1j * m[1] == pytest.approx(expected, abs=1e-12)
    assert m[2] == pytest.approx(1.0 - np.exp(-duration / t1), abs=1e-12)


def _area(times, values, at):
    """The integral of the piecewise-linear waveform through the corners, up to ``at``."""
    fine = np.union1d(times, [at])
    fine = fine[fine <= at]
    return np.sum(
        0.5
        * np.diff(fine)
        * (np.interp(fine[1:], times, values) + np.interp(fine[:-1], times, values))
    )


def test_samples_within_a_window_follow_the_gradient_between_them():
    x = np.array([-0.02, 0.01, 0.045])
    spins = Isochromats(np.column_stack([x, np.zeros(3), np.zeros(3)]))
    spins.magnetization = [0.0, 1.0, 0.0]
    # A trapezoid read out on its ramps and plateau, which no single
    # increment per sample describes.
    trapezoid = np.array([[0.0, 0.4e-3, 1.6e-3, 2e-3], [0.0, 2e4, 2e4, 0.0]])
    adc = np.linspace(0.05e-3, 1.95e-3, 40)
    signal = spins.play(2e-3, gradients=[trapezoid, None, None], adc=adc)
    areas = np.array([_area(*trapezoid, t) for t in adc])
    expected = 1j * np.exp(-2j * np.pi * np.outer(areas, x)).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-11)


def test_a_spin_echo_refocuses_off_resonance_and_decays_with_t2():
    count, t2, echo_time = 64, 0.05, 20e-3
    off_resonance = np.linspace(-500.0, 500.0, count)
    spins = Isochromats(np.zeros((count, 3)), t2=t2, off_resonance=off_resonance)
    dephased = spins.play(echo_time / 2, rf=_hard(90.0), adc=[echo_time / 2])
    assert abs(dephased[0, 0]) < 0.05 * count
    echo = spins.play(
        echo_time / 2, rf=_hard(180.0, phase=np.pi / 2), adc=[echo_time / 2]
    )
    assert echo[0, 0] == pytest.approx(1j * count * np.exp(-echo_time / t2), rel=1e-4)


def test_a_short_t2_steady_state_is_the_ernst_signal():
    t1, t2, repetition, echo, flip = 0.1, 2e-3, 30e-3, 1e-3, 30.0
    spins = Isochromats(ORIGIN, t1=t1, t2=t2)
    for _ in range(60):
        signal = spins.play(repetition, rf=_hard(flip), adc=[echo])
    e1 = np.exp(-repetition / t1)
    alpha = np.deg2rad(flip)
    ernst = np.sin(alpha) * (1 - e1) / (1 - e1 * np.cos(alpha)) * np.exp(-echo / t2)
    assert signal[0, 0] == pytest.approx(1j * ernst, rel=1e-4)


def test_balanced_ssfp_reaches_its_steady_state():
    t1, t2, repetition, flip = 0.3, 0.1, 5e-3, 60.0
    spins = Isochromats(ORIGIN, t1=t1, t2=t2)
    for n in range(4000):
        # Alternating the pulse's phase puts the isochromat at the centre of
        # the passband.
        signal = spins.play(
            repetition, rf=_hard(flip, phase=np.pi * (n % 2)), adc=[repetition / 2]
        )
    e1, e2 = np.exp(-repetition / t1), np.exp(-repetition / t2)
    alpha = np.deg2rad(flip)
    steady = (1 - e1) * np.sin(alpha) / (1 - (e1 - e2) * np.cos(alpha) - e1 * e2)
    assert abs(signal[0, 0]) == pytest.approx(steady * np.sqrt(e2), rel=1e-4)


def _gradient_echo(fov=0.128, samples=64, rotation=None):
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    seq = pp.Sequence(system)
    gx = pp.make_trapezoid(
        "x", flat_area=samples / fov, flat_time=3.2e-3, system=system
    )
    adc = pp.make_adc(samples, duration=gx.flat_time, delay=gx.rise_time, system=system)
    pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    extra = [] if rotation is None else [rotation]
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.1e-3, system=system))
    seq.add_block(pre, *extra)
    seq.add_block(gx, adc, *extra)
    return seq


def test_a_gradient_echo_samples_the_transform_of_the_isochromats_at_their_k():
    positions = np.array([[-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
    density = np.array([1.0, 0.5, 2.0])
    seq = _gradient_echo()
    signal = design_samples(seq, Isochromats(positions, proton_density=density))
    k = seq.calculate_kspace()[0]
    expected = 1j * (density * np.exp(-2j * np.pi * (k.T @ positions.T))).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-8)
    # The inverse transform puts each isochromat at its own pixel of 2 mm.
    image = np.abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(signal[0]))))
    assert sorted(np.argsort(image)[-3:] - 32) == [-10, 0, 15]


def test_a_rotated_block_plays_its_gradients_along_the_rotated_axes():
    positions = np.array([[0.0, -0.02, 0.0], [0.0, 0.031, 0.0]])
    seq = _gradient_echo(rotation=pp.make_rotation(np.pi / 2))
    signal = design_samples(seq, Isochromats(positions))
    k = seq.calculate_kspace()[0]
    assert np.abs(k[1]).max() > 0.0 and np.abs(k[0]).max() < 1e-9
    expected = 1j * np.exp(-2j * np.pi * (k.T @ positions.T)).sum(axis=1)
    np.testing.assert_allclose(signal[0], expected, rtol=0, atol=1e-8)


def test_a_rotated_gradient_steps_where_it_starts_and_ends_away_from_zero():
    """A step on one axis stays a step in the sum a rotation plays."""
    step = pp.make_extended_trapezoid(
        "x", times=np.array([0.0, 1e-3]), amplitudes=np.array([2e4, 2e4])
    )
    step.delay = 0.5e-3
    ramp = pp.make_trapezoid("y", amplitude=1e4, rise_time=0.2e-3, flat_time=1.6e-3)
    turn = pp.make_rotation(np.pi / 3)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(step, ramp, turn, pp.make_delay(2.5e-3))
    positions = RNG.uniform(-0.05, 0.05, size=(6, 3))
    spins = Isochromats(positions)
    spins.magnetization = [0.0, 1.0, 0.0]
    design_samples(seq, spins)
    area = np.array([2e4 * 1e-3, 1e4 * (0.2e-3 + 1.6e-3), 0.0])
    matrix = np.array(
        [
            [np.cos(np.pi / 3), -np.sin(np.pi / 3), 0.0],
            [np.sin(np.pi / 3), np.cos(np.pi / 3), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = 1j * np.exp(-2j * np.pi * positions @ (matrix @ area))
    m = spins.magnetization
    np.testing.assert_allclose(m[:, 0] + 1j * m[:, 1], expected, rtol=0, atol=1e-9)


def test_a_reflection_plays_the_gradients_mirrored():
    gradient = np.array([[0.0, 0.2e-3, 0.8e-3, 1e-3], [0.0, 3e4, 3e4, 0.0]])
    mirrored, plain = (
        Isochromats([[0.0, 0.0, 0.01]]),
        Isochromats([[0.0, 0.0, -0.01]]),
    )
    for spins in (mirrored, plain):
        spins.magnetization = [1.0, 0.0, 0.0]
    mirrored.play(
        1e-3, gradients=[None, None, gradient], rotation=np.diag([1.0, 1.0, -1.0])
    )
    plain.play(1e-3, gradients=[None, None, gradient])
    np.testing.assert_allclose(mirrored.magnetization, plain.magnetization, atol=1e-12)


def test_consecutive_block_ranges_play_one_scan():
    seq = pp.Sequence(pp.Opts())
    for n in range(6):
        seq.add_block(
            pp.make_block_pulse(np.deg2rad(20), duration=0.1e-3, phase_offset=0.3 * n)
        )
        seq.add_block(pp.make_trapezoid("z", area=200.0, duration=1e-3))
        seq.add_block(pp.make_adc(8, dwell=20e-6, phase_offset=0.3 * n))
    positions = RNG.uniform(-0.01, 0.01, size=(50, 3))
    whole = design_samples(seq, Isochromats(positions, t1=0.2, t2=0.05))
    spins = Isochromats(positions, t1=0.2, t2=0.05)
    parts = [
        design_samples(seq, spins, block_range=r) for r in ((1, 5), (6, 6), (7, 18))
    ]
    np.testing.assert_allclose(np.concatenate(parts, axis=1), whole, rtol=0, atol=1e-12)
    assert spins.elapsed == pytest.approx(seq.duration()[0])


def test_the_adc_phase_offset_cancels_the_same_rf_phase_offset():
    for phase in (0.0, 0.7, 2.9):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(
            pp.make_block_pulse(np.pi / 2, duration=0.1e-3, phase_offset=phase)
        )
        seq.add_block(pp.make_adc(1, dwell=10e-6, phase_offset=phase))
        signal = design_samples(seq, Isochromats(ORIGIN))
        assert signal[0, 0] == pytest.approx(1j, abs=1e-12)


def test_the_adc_follows_its_frequency_offset_and_phase_modulation():
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    gx = pp.make_trapezoid("x", flat_area=64 / 0.128, flat_time=3.2e-3, system=system)
    x = 0.023
    modulation = np.linspace(0.0, 1.0, 64) ** 2
    adc = pp.make_adc(
        64,
        duration=gx.flat_time,
        delay=gx.rise_time,
        freq_offset=gx.amplitude * x,
        phase_modulation=modulation,
        system=system,
    )
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(np.pi / 2, duration=0.1e-3, system=system))
    seq.add_block(gx, adc)
    signal = design_samples(seq, Isochromats([[x, 0.0, 0.0]]))[0]
    # Demodulated at the isochromat's own frequency, it keeps the phase it
    # had at the window's start, plus the modulation.
    np.testing.assert_allclose(
        signal * np.exp(-1j * modulation),
        signal[0] * np.exp(-1j * modulation[0]),
        atol=1e-9,
    )


def test_a_shim_weights_a_pulse_onto_the_transmit_channels():
    rf = pp.make_block_pulse(np.pi / 6, duration=0.2e-3)
    flips = []
    for shim in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(rf, pp.make_rf_shim(shim))
        spins = Isochromats(ORIGIN, transmit=[[1.0, 2.0]])
        design_samples(seq, spins, transmit_channels=2)
        flips.append(np.degrees(np.arccos(spins.magnetization[0, 2])))
    np.testing.assert_allclose(flips, [30.0, 60.0, 45.0], atol=1e-9)
    # Without a shim, the pulse plays on every channel.
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf)
    spins = Isochromats(ORIGIN, transmit=[[1.0, 2.0]])
    design_samples(seq, spins, transmit_channels=2)
    assert np.degrees(np.arccos(spins.magnetization[0, 2])) == pytest.approx(
        90.0, abs=1e-9
    )
    # Without transmit sensitivities the shim has nothing to weight.
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf, pp.make_rf_shim([0.0, 1.0]))
    spins = Isochromats(ORIGIN)
    design_samples(seq, spins)
    assert np.degrees(np.arccos(spins.magnetization[0, 2])) == pytest.approx(
        30.0, abs=1e-9
    )


def test_a_positive_frequency_offset_excites_isochromats_at_a_positive_frequency():
    rf = pp.make_sinc_pulse(
        np.pi / 2, duration=4e-3, time_bw_product=4, freq_offset=2000.0
    )
    seq = pp.Sequence(pp.Opts())
    seq.add_block(rf)
    spins = Isochromats(np.zeros((2, 3)), off_resonance=[2000.0, -2000.0])
    design_samples(seq, spins)
    transverse = np.hypot(*spins.magnetization[:, :2].T)
    assert transverse[0] > 0.99 and transverse[1] < 0.02


def test_a_frequency_offset_and_the_same_phase_ramp_play_one_pulse():
    envelope = np.sinc(np.linspace(-2, 2, 1000)) * 300.0
    offset = 1500.0
    shifted = pp.make_arbitrary_rf(
        envelope, 0.0, freq_offset=offset, no_signal_scaling=True
    )
    # The ramp over the samples' own times, from the pulse's start.
    ramp = np.exp(2j * np.pi * offset * np.asarray(shifted.t))
    ramped = pp.make_arbitrary_rf(envelope * ramp, 0.0, no_signal_scaling=True)
    off_resonance = np.linspace(-3000.0, 3000.0, 13)
    results = []
    for rf in (shifted, ramped):
        seq = pp.Sequence(pp.Opts())
        seq.add_block(rf)
        spins = Isochromats(np.zeros((13, 3)), off_resonance=off_resonance)
        design_samples(seq, spins)
        results.append(spins.magnetization)
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-9)


def test_a_time_shaped_block_pulse_flips_as_one_on_the_raster():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_block_pulse(np.pi / 3, duration=0.25e-3))
    spins = Isochromats(ORIGIN)
    design_samples(seq, spins)
    np.testing.assert_allclose(
        spins.magnetization, [[0.0, np.sin(np.pi / 3), np.cos(np.pi / 3)]], atol=1e-9
    )


def test_isochromats_sharing_a_pulse_computation_answer_as_separate_ones():
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    rf, gz, _ = pp.make_sinc_pulse(
        np.pi / 2, duration=2e-3, slice_thickness=5e-3, return_gz=True, system=system
    )
    seq = pp.Sequence(system)
    seq.add_block(rf, gz)
    z = np.repeat(np.linspace(-6e-3, 6e-3, 25), 4)
    positions = np.column_stack([RNG.uniform(-0.1, 0.1, (z.size, 2)), z])
    shared = Isochromats(positions, t2=0.08)
    # A T2 of its own for each isochromat, too close to change the answer,
    # gives each its own computation of the pulse.
    alone = Isochromats(positions, t2=0.08 * (1 + 1e-13 * np.arange(z.size)))
    design_samples(seq, shared)
    design_samples(seq, alone)
    np.testing.assert_allclose(
        shared.magnetization, alone.magnetization, rtol=0, atol=1e-10
    )
    inside = np.abs(z) < 2e-3
    assert np.all(np.hypot(*shared.magnetization[inside, :2].T) > 0.9)
    assert np.all(np.hypot(*shared.magnetization[np.abs(z) > 4.5e-3, :2].T) < 0.05)


@pytest.mark.parametrize(
    ("flip", "phase"), [(np.pi / 4, 2.1), (np.pi / 3, 0.0)], ids=["phase", "flip"]
)
def test_a_pulse_played_after_another_answers_as_one_played_afresh(flip, phase):
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    z = np.linspace(-8e-3, 8e-3, 400)
    positions = np.column_stack([RNG.uniform(-0.1, 0.1, (z.size, 2)), z])
    # An off-resonance of its own gives each isochromat its own map.
    properties = {
        "t1": 0.8,
        "t2": 0.06,
        "off_resonance": RNG.uniform(-40.0, 40.0, z.size),
    }

    def pulse(flip, phase):
        rf, gz, _ = pp.make_sinc_pulse(
            flip,
            duration=2e-3,
            slice_thickness=5e-3,
            phase_offset=phase,
            return_gz=True,
            system=system,
        )
        rise, flat, fall = gz.rise_time, gz.flat_time, gz.fall_time
        times = gz.delay + np.array([0.0, rise, rise + flat, rise + flat + fall])
        corners = np.array([times, [0.0, gz.amplitude, gz.amplitude, 0.0]])
        return pp.calc_duration(rf, gz), [None, None, corners], rf

    def played(spins, *pulses):
        for duration, gradients, rf in pulses:
            spins.play(duration, gradients=gradients, rf=rf, system=system)
            spins.play(3e-3)
        return spins.magnetization

    first, second = pulse(np.pi / 4, 0.0), pulse(flip, phase)
    after = played(Isochromats(positions, **properties), first, second)
    afresh = Isochromats(positions, **properties)
    afresh.magnetization = played(Isochromats(positions, **properties), first)

    np.testing.assert_allclose(played(afresh, second), after, rtol=0, atol=1e-12)


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


def _slice_pulses(spins, pulses, *channels, cancelled=False):
    """Play sinc pulses of ``(flip, phase, offset)`` under their slice gradients, each on ``channels`` when given."""
    system = pp.Opts(max_grad=30, grad_unit="mT/m", max_slew=120, slew_unit="T/m/s")
    for flip, phase, offset in pulses:
        rf, gz, _ = pp.make_sinc_pulse(
            flip,
            duration=2e-3,
            slice_thickness=5e-3,
            phase_offset=phase,
            freq_offset=offset,
            return_gz=True,
            system=system,
        )
        rise, flat, fall = gz.rise_time, gz.flat_time, gz.fall_time
        times = gz.delay + np.array([0.0, rise, rise + flat, rise + flat + fall])
        corners = np.array([times, [0.0, gz.amplitude, gz.amplitude, 0.0]])
        spins.play(
            pp.calc_duration(rf, gz),
            gradients=[None, None, corners],
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


@GRID_PULSES
def test_a_pulse_on_transmit_sensitivities_computed_on_a_grid_of_fields_and_drives_answers_as_one_stepped(
    pulses,
):
    """Each isochromat's drive of the channels' one waveform: its magnitude a second axis of the grid, its phase a turn of the map."""
    positions, properties = _grid_isochromats()
    n = len(positions)
    transmit = RNG.uniform(0.3, 1.5, (n, 2)) * np.exp(
        1j * RNG.uniform(-np.pi, np.pi, (n, 2))
    )
    shim = np.array([0.8 * np.exp(0.4j), 0.5 * np.exp(-1.1j)])
    on_grid = _slice_pulses(
        Isochromats(positions, transmit=transmit, **properties), pulses, shim
    )
    stepped = _slice_pulses(
        Isochromats(
            positions,
            transmit=np.column_stack([transmit, transmit[:, 0]]),
            **properties,
        ),
        pulses,
        shim,
        cancelled=True,
    )

    np.testing.assert_allclose(on_grid, stepped, rtol=0, atol=1e-6)


def test_threads_do_not_change_the_answer():
    positions = RNG.uniform(-0.05, 0.05, size=(20000, 3))
    receive = np.exp(1j * RNG.uniform(0, 2 * np.pi, size=(20000, 3)))
    seq = _gradient_echo()
    answers = [
        design_samples(
            seq,
            Isochromats(
                positions, t2=0.07, off_resonance=12.0, receive=receive, threads=n
            ),
        )
        for n in (1, 4)
    ]
    np.testing.assert_allclose(answers[0], answers[1], rtol=1e-11, atol=1e-9)


def test_each_coil_receives_its_sensitivity_times_the_magnetisation():
    positions = RNG.uniform(-0.05, 0.05, size=(30, 3))
    receive = RNG.normal(size=(30, 2)) + 1j * RNG.normal(size=(30, 2))
    spins = Isochromats(positions, receive=receive)
    spins.play(1e-3, rf=_hard(40.0))
    gradient = np.array([[0.0, 1e-3], [5e3, 5e3]])
    signal = spins.play(1e-3, gradients=[gradient, gradient, None], adc=[1e-3])
    m = spins.magnetization
    np.testing.assert_allclose(
        signal[:, 0], receive.T @ (m[:, 0] + 1j * m[:, 1]), atol=1e-12
    )


@pytest.mark.parametrize(
    "gradient",
    [
        np.array([[0.0, 2e-3], [2e4, 2e4]]),
        np.array([[0.0, 0.4e-3, 1.6e-3, 2e-3], [0.0, 2e4, 2e4, 0.0]]),
    ],
    ids=["uniform steps", "ramps"],
)
def test_many_coils_receive_their_sensitivities_times_the_magnetisation_at_each_sample(
    gradient,
):
    # Coils beyond one block of the coil sum, and isochromats that fill no
    # whole number of tiles.
    n, coils = 1003, 9
    positions = RNG.uniform(-0.05, 0.05, size=(n, 3))
    receive = RNG.normal(size=(n, coils)) + 1j * RNG.normal(size=(n, coils))
    t2 = RNG.uniform(0.02, 0.2, size=n)
    off_resonance = RNG.uniform(-50.0, 50.0, size=n)
    start = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    spins = Isochromats(positions, t2=t2, off_resonance=off_resonance, receive=receive)
    spins.magnetization = np.column_stack([start.real, start.imag, np.zeros(n)])
    adc = np.linspace(0.1e-3, 1.9e-3, 12)
    signal = spins.play(2e-3, gradients=[gradient, None, None], adc=adc)

    areas = np.array([_area(*gradient, t) for t in adc])
    phase = np.outer(areas, positions[:, 0]) + np.outer(adc, off_resonance)
    transverse = start * np.exp(-adc[:, None] / t2 - 2j * np.pi * phase)
    np.testing.assert_allclose(signal, (transverse @ receive).T, rtol=0, atol=1e-10)


@pytest.mark.parametrize("t2", ["three", "each its own"])
def test_a_long_window_under_a_held_gradient_samples_each_isochromat_s_geometric_series(
    t2,
):
    """Read by a non-uniform FFT where the isochromats share few T2s, sample by sample where they do not."""
    n, coils, samples = 5000, 4, 256
    positions = RNG.uniform(-0.05, 0.05, size=(n, 3))
    receive = RNG.normal(size=(n, coils)) + 1j * RNG.normal(size=(n, coils))
    t2s = (
        RNG.choice([0.03, 0.08, np.inf], n)
        if t2 == "three"
        else RNG.uniform(0.02, 0.2, n)
    )
    off_resonance = RNG.uniform(-200.0, 200.0, n)
    start = RNG.normal(size=n) + 1j * RNG.normal(size=n)
    spins = Isochromats(
        positions, t1=0.9, t2=t2s, off_resonance=off_resonance, receive=receive
    )
    spins.magnetization = np.column_stack([start.real, start.imag, np.full(n, 0.2)])
    held = [np.array([[0.0, 4e-3], [value, value]]) for value in (2e4, 1e4)]
    adc = np.linspace(0.1e-3, 3.9e-3, samples)
    signal = spins.play(4e-3, gradients=[*held, None], adc=adc)

    def transverse(times):
        phase = np.outer(times, positions @ [2e4, 1e4, 0.0]) + np.outer(
            times, off_resonance
        )
        return start * np.exp(-times[:, None] / t2s - 2j * np.pi * phase)

    terms = np.abs(start) @ np.abs(receive)
    np.testing.assert_allclose(
        signal, (transverse(adc) @ receive).T, rtol=0, atol=1e-12 * terms.max()
    )
    # The window leaves each isochromat as it stands at the last sample.
    m = spins.magnetization
    np.testing.assert_allclose(
        m[:, 0] + 1j * m[:, 1], transverse(np.array([4e-3]))[0], rtol=0, atol=1e-12
    )
    e1 = np.exp(-4e-3 / 0.9)
    np.testing.assert_allclose(m[:, 2], 0.2 * e1 + 1 - e1, rtol=0, atol=1e-12)


def test_receive_sensitivities_in_a_read_only_memory_mapped_file_are_received_as_in_memory(
    tmp_path,
):
    positions = RNG.uniform(-0.05, 0.05, size=(30, 3))
    receive = RNG.normal(size=(30, 3)) + 1j * RNG.normal(size=(30, 3))
    path = tmp_path / "receive.dat"
    np.memmap(path, dtype=complex, mode="w+", shape=receive.shape)[:] = receive
    mapped = np.memmap(path, dtype=complex, mode="r", shape=receive.shape)
    gradient = np.array([[0.0, 1e-3], [5e3, 5e3]])
    signals = []
    for sensitivities in (receive, mapped):
        spins = Isochromats(positions, receive=sensitivities)
        spins.play(1e-3, rf=_hard(40.0))
        signals.append(
            spins.play(1e-3, gradients=[gradient, gradient, None], adc=[5e-4, 1e-3])
        )

    np.testing.assert_array_equal(signals[1], signals[0])


def test_a_transmit_sensitivity_scales_the_flip():
    spins = Isochromats(np.zeros((2, 3)), transmit=[1.0, 0.5])
    spins.play(1e-3, rf=_hard(90.0))
    np.testing.assert_allclose(
        spins.magnetization,
        [[0.0, 1.0, 0.0], [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]],
        atol=1e-12,
    )


def test_the_channels_of_a_ptx_pulse_are_summed_without_transmit_sensitivities():
    samples = np.full((2, 100), 250.0 + 0j)
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_ptx_pulse(samples * [[1.0], [0.0]]))
    one = Isochromats(ORIGIN)
    design_samples(seq, one)
    ptx = pp.Sequence(pp.Opts())
    ptx.add_block(pp.make_ptx_pulse(samples * 0.5))
    summed = Isochromats(ORIGIN)
    design_samples(ptx, summed)
    np.testing.assert_allclose(summed.magnetization, one.magnetization, atol=1e-12)
    mapped = Isochromats(ORIGIN, transmit=[[2.0, 0.0]])
    design_samples(ptx, mapped)
    np.testing.assert_allclose(mapped.magnetization, one.magnetization, atol=1e-12)


def test_reset_and_the_magnetisation_setter():
    spins = Isochromats(np.zeros((3, 3)), proton_density=[1.0, 2.0, 0.5])
    spins.play(1e-3, rf=_hard(90.0))
    spins.magnetization = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(spins.magnetization, np.tile([0.1, 0.2, 0.3], (3, 1)))
    spins.reset()
    np.testing.assert_allclose(spins.magnetization[:, 2], [1.0, 2.0, 0.5])
    assert spins.elapsed == 0.0 and len(spins) == 3 and spins.coils == 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"positions": np.zeros((2, 2))}, "positions"),
        ({"t1": -1.0}, "positive"),
        ({"t2": [0.1, 0.1, 0.1]}, "t2"),
        ({"off_resonance": np.nan}, "finite"),
        ({"receive": np.ones((3, 2))}, "receive"),
    ],
)
def test_malformed_isochromats_are_refused(arguments, message):
    given = {"positions": np.zeros((2, 3)), **arguments}
    with pytest.raises(ValueError, match=message):
        Isochromats(given.pop("positions"), **given)


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ({"rf": (0.0, 1e-3, [100.0]), "adc": [0.5e-3]}, "inside the RF pulse"),
        ({"adc": [2e-3]}, "within their block"),
        ({"adc": [0.5e-3, 0.2e-3]}, "increasing order"),
        ({"rf": (0.5e-3, 1e-3, [100.0])}, "within its block"),
        ({"rf": (0.0, 1e-4, np.ones((2, 3)))}, "channels"),
        (
            {"gradients": [np.array([[1e-4, 0.0], [1.0, 1.0]]), None, None]},
            "increasing",
        ),
        (
            {
                "gradients": [np.array([[1e-4, 0.0], [1.0, 1.0]]), None, None],
                "rotation": np.eye(3),
            },
            "increasing",
        ),
        ({"gradients": [None, None]}, "three axes"),
        ({"rotation": np.eye(2)}, "rotation"),
        (
            {
                "rf": SimpleNamespace(
                    signal=np.ones(3), t=[0.0, 2e-6, 1e-6], delay=0.0, freq_offset=0.0
                )
            },
            "sample times",
        ),
        (
            {
                "adc": SimpleNamespace(
                    num_samples=4, dwell=1e-5, delay=0.0, phase_modulation=np.zeros(3)
                )
            },
            "one value per sample",
        ),
        ({"adc": SimpleNamespace(num_samples=4, dwell=0.0, delay=0.0)}, "dwell"),
    ],
)
def test_events_that_do_not_fit_are_refused(events, message):
    spins = Isochromats(ORIGIN, transmit=[1.0])
    with pytest.raises(ValueError, match=message):
        spins.play(1e-3, **events)
