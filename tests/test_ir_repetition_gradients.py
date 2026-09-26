"""The gradients of a subsequence's heaviest repetition, for heating and acoustic models."""

import math
import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest

from pulserver import ir

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(
    max_grad=40.0,
    grad_unit="mT/m",
    max_slew=170.0,
    slew_unit="T/m/s",
    B0=3.0,
    rf_raster_time=1e-6,
    grad_raster_time=1e-5,
    adc_raster_time=1e-7,
    block_duration_raster=1e-5,
)
ONE_FILE = [
    "gre_2d_3sl.seq",
    "epi_2d_main.seq",
    "zte_3d.seq",
    "mprage_stack_of_spirals_3d.seq",
]


def _converted(name, directory):
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    ir.convert(directory / name, SYSTEM)
    return directory / name


def _squared_integral(times_us, values):
    """The integral of the square of the piecewise-linear curve, in s."""
    dt = 1e-6 * np.diff(times_us)
    v0, v1 = values[:-1], values[1:]
    return float(np.sum(dt * (v0 * v0 + v0 * v1 + v1 * v1) / 3.0))


def _block_energy(played, block):
    total = 0.0
    for axis in range(3):
        start, stop = played["gradient_span"][block, axis]
        total += _squared_integral(
            played["gradient_time_us"][start:stop].astype(float),
            played["gradient_waveform_hz_per_m"][start:stop].astype(float),
        )
    return total


def _repetition_energies(played, size):
    blocks = np.flatnonzero(played["subsequence"] == 0)
    energies = [_block_energy(played, block) for block in blocks]
    return [sum(energies[i : i + size]) for i in range(0, len(energies), size)]


@pytest.mark.parametrize("name", ONE_FILE)
def test_the_repetition_of_most_gradient_energy_is_the_one_returned(name, tmp_path):
    seq = _converted(name, tmp_path)
    size = ir.summary(seq, SYSTEM, cache_ext=".pseg")["subsequences"][0]["tr_size"]
    energies = _repetition_energies(ir.play(seq, waveforms=True), size)
    heaviest = ir.repetition_gradients(seq)
    chosen = energies[heaviest["first_position"] // size]
    assert heaviest["first_position"] % size == 0
    assert chosen >= max(energies) * (1.0 - 1e-3)
    assert heaviest["energy"] == pytest.approx(chosen, rel=1e-3)


def _block_inside(played, block, axis):
    """A block's played corners on one axis away from its edges, where it meets another."""
    start, stop = played["gradient_span"][block, axis]
    times = played["gradient_time_us"][start:stop].astype(float)
    inside = (times > 0.0) & (times < played["duration_us"][block])
    return times[inside], played["gradient_waveform_hz_per_m"][start:stop][inside]


@pytest.mark.parametrize("name", ONE_FILE)
def test_the_corner_points_are_the_gradients_the_repetition_plays(name, tmp_path):
    seq = _converted(name, tmp_path)
    size = ir.summary(seq, SYSTEM, cache_ext=".pseg")["subsequences"][0]["tr_size"]
    heaviest = ir.repetition_gradients(seq)
    played = ir.play(seq, waveforms=True)
    blocks = range(heaviest["first_position"], heaviest["first_position"] + size)
    starts = np.concatenate([[0.0], np.cumsum(played["duration_us"][blocks])])
    scale = np.abs(heaviest["gradient_hz_per_m"]).max()
    for offset, block in enumerate(blocks):
        for axis in range(3):
            times, values = _block_inside(played, block, axis)
            joined = np.interp(
                starts[offset] + times,
                heaviest["time_us"],
                heaviest["gradient_hz_per_m"][:, axis],
            )
            np.testing.assert_allclose(joined, values, atol=1e-4 * scale)
    assert heaviest["duration_us"] == pytest.approx(starts[-1])


def test_the_samples_are_the_corner_points_at_the_raster_centres(tmp_path):
    seq = _converted("zte_3d.seq", tmp_path)
    raster = 4.0
    heaviest = ir.repetition_gradients(seq, raster_us=raster)
    samples = heaviest["samples_hz_per_m"]
    assert samples.shape[1] == math.ceil(heaviest["duration_us"] / raster)
    centres = (np.arange(samples.shape[1]) + 0.5) * raster
    for axis in range(3):
        np.testing.assert_allclose(
            samples[axis],
            np.interp(
                centres, heaviest["time_us"], heaviest["gradient_hz_per_m"][:, axis]
            ),
            atol=1e-2,
        )


def _turned_repetitions(path):
    """Three repetitions of one readout: 1.0 turned by 30°, 1.2 by 60°, 1.2 unturned."""
    gx = pp.make_trapezoid(
        "x", amplitude=2e5, flat_time=1e-3, rise_time=2e-4, system=SYSTEM
    )
    seq = pp.Sequence(SYSTEM)
    for factor, turn in ((1.0, 30.0), (1.2, 60.0), (1.2, None)):
        events = [pp.scale_grad(gx, factor)]
        if turn is not None:
            events.append(pp.make_rotation(math.radians(turn)))
        seq.add_block(*events)
        seq.add_block(pp.make_delay(1e-3))
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return gx


def test_a_rotation_changes_neither_the_energy_nor_the_repetition_returned(tmp_path):
    path = tmp_path / "turned.seq"
    gx = _turned_repetitions(path)
    heaviest = ir.repetition_gradients(path)
    # The second and the third tie; the earlier stands.
    assert heaviest["first_position"] == 2
    rise, flat = 1e-6 * round(gx.rise_time * 1e6), 1e-6 * round(gx.flat_time * 1e6)
    expected = (1.2 * gx.amplitude) ** 2 * (2.0 * rise / 3.0 + flat)
    assert heaviest["energy"] == pytest.approx(expected, rel=1e-5)
    turned = heaviest["gradient_hz_per_m"]
    peak = np.linalg.norm(turned, axis=1).max()
    assert peak == pytest.approx(1.2 * gx.amplitude, rel=1e-5)
    assert np.abs(turned[:, 1]).max() > 0.5 * peak
