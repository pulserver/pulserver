"""The gradients of a subsequence's heaviest stretches, for heating models."""

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
STRETCHES = ("heaviest", "heaviest_repetition")
WINDOW_US = 10.0e6
GX = pp.make_trapezoid(
    "x", amplitude=1e5, flat_time=1e-3, rise_time=2e-4, system=SYSTEM
)


def _converted(name, directory):
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    ir.convert(directory / name, SYSTEM)
    return directory / name


def _written(path, repetitions):
    """One readout and one pure delay per ``(amplitude factor, delay s)``."""
    seq = pp.Sequence(SYSTEM)
    for factor, wait in repetitions:
        seq.add_block(pp.scale_grad(GX, factor))
        seq.add_block(pp.make_delay(wait))
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    return path


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


def _played(seq):
    """Per position of subsequence 0: energy, played duration and instance starts."""
    played = ir.play(seq, waveforms=True)
    blocks = np.flatnonzero(played["subsequence"] == 0)
    energies = np.array([_block_energy(played, b) for b in blocks])
    durations = played["duration_us"][blocks].astype(float)
    segments = played["segment"][blocks]
    sizes = [s["num_blocks"] for s in ir.summary(seq, SYSTEM)["segments"]]
    starts, within = [], 0
    for n, segment in enumerate(segments):
        if n == 0 or segment != segments[n - 1] or within == sizes[segment]:
            starts.append(n)
            within = 0
        within += 1
    return energies, durations, starts


def _reference(energies, durations, starts):
    """The heaviest stretch, every start tried in turn."""
    ends = [*starts[1:], len(energies)]
    best, first = -1.0, 0
    for i, start in enumerate(starts):
        j = i
        while j < len(starts) and durations[start : ends[j]].sum() < WINDOW_US:
            j += 1
        if j == len(starts):
            break
        energy = energies[start : ends[j]].sum()
        if energy > best * (1.0 + 1e-6):
            best, first = energy, start
    return first


def _heaviest_repetition(energies, size):
    totals = [energies[i : i + size].sum() for i in range(0, len(energies), size)]
    return size * int(np.argmax(totals)), max(totals)


@pytest.mark.parametrize("name", ONE_FILE)
def test_a_stretch_from_the_heaviest_repetition_starts_where_it_does(name, tmp_path):
    seq = _converted(name, tmp_path)
    size = ir.summary(seq, SYSTEM, cache_ext=".pseg")["subsequences"][0]["tr_size"]
    energies, _, _ = _played(seq)
    _, heaviest = _heaviest_repetition(energies, size)
    stretch = ir.gradient_stretch(seq, 0, "heaviest_repetition")
    first = stretch["first_position"]
    assert first % size == 0
    assert energies[first : first + size].sum() >= heaviest * (1.0 - 1e-3)
    span = energies[first : first + stretch["num_positions"]].sum()
    assert stretch["energy"] == pytest.approx(span, rel=1e-3)


def _block_inside(played, block, axis):
    """A block's played corners on one axis away from its edges, where it meets another."""
    start, stop = played["gradient_span"][block, axis]
    times = played["gradient_time_us"][start:stop].astype(float)
    inside = (times > 0.0) & (times < played["duration_us"][block])
    return times[inside], played["gradient_waveform_hz_per_m"][start:stop][inside]


@pytest.mark.parametrize("stretch", STRETCHES)
@pytest.mark.parametrize("name", ONE_FILE)
def test_the_corner_points_are_the_gradients_the_stretch_plays(name, stretch, tmp_path):
    seq = _converted(name, tmp_path)
    joined = ir.gradient_stretch(seq, 0, stretch)
    played = ir.play(seq, waveforms=True)
    first = joined["first_position"]
    blocks = range(first, first + joined["num_positions"])
    starts = np.concatenate([[0.0], np.cumsum(played["duration_us"][blocks])])
    scale = np.abs(joined["gradient_hz_per_m"]).max()
    for offset, block in enumerate(blocks):
        for axis in range(3):
            times, values = _block_inside(played, block, axis)
            np.testing.assert_allclose(
                np.interp(
                    starts[offset] + times,
                    joined["time_us"],
                    joined["gradient_hz_per_m"][:, axis],
                ),
                values,
                atol=1e-4 * scale,
            )
    assert joined["duration_us"] == pytest.approx(starts[-1])


def test_the_samples_are_the_corner_points_at_the_raster_centres(tmp_path):
    seq = _converted("zte_3d.seq", tmp_path)
    raster = 4.0
    stretch = ir.gradient_stretch(seq, raster_us=raster)
    samples = stretch["samples_hz_per_m"]
    assert samples.shape[1] == math.ceil(stretch["duration_us"] / raster)
    centres = (np.arange(samples.shape[1]) + 0.5) * raster
    for axis in range(3):
        np.testing.assert_allclose(
            samples[axis],
            np.interp(
                centres, stretch["time_us"], stretch["gradient_hz_per_m"][:, axis]
            ),
            atol=1e-2,
        )


def test_a_rotation_changes_neither_the_energy_nor_the_repetition_returned(tmp_path):
    """Three repetitions of one readout: 1.0 turned by 30°, 1.2 by 60°, 1.2 unturned."""
    seq = pp.Sequence(SYSTEM)
    for factor, turn in ((1.0, 30.0), (1.2, 60.0), (1.2, None)):
        events = [pp.scale_grad(GX, factor)]
        if turn is not None:
            events.append(pp.make_rotation(math.radians(turn)))
        seq.add_block(*events)
        seq.add_block(pp.make_delay(1e-3))
    path = tmp_path / "turned.seq"
    seq.write(str(path))
    ir.convert(path, SYSTEM)
    stretch = ir.gradient_stretch(path, 0, "heaviest_repetition")
    # The second and the third tie; the earlier stands, and the stretch runs
    # to the end.
    assert (stretch["first_position"], stretch["num_positions"]) == (2, 4)
    rise, flat = 1e-6 * round(GX.rise_time * 1e6), 1e-6 * round(GX.flat_time * 1e6)
    expected = 2.0 * (1.2 * GX.amplitude) ** 2 * (2.0 * rise / 3.0 + flat)
    assert stretch["energy"] == pytest.approx(expected, rel=1e-5)
    turned = stretch["gradient_hz_per_m"]
    peak = np.linalg.norm(turned, axis=1).max()
    assert peak == pytest.approx(1.2 * GX.amplitude, rel=1e-5)
    assert np.abs(turned[:, 1]).max() > 0.5 * peak


def test_a_periodic_subsequence_is_costed_per_repetition_as_its_heaviest(tmp_path):
    """Identical repetitions: both stretches hold whole ones, each as heavy as the heaviest."""
    path = _written(tmp_path / "periodic.seq", [(1.0, 0.1)] * 200)
    energies, durations, _ = _played(path)
    _, heaviest = _heaviest_repetition(energies, 2)
    repetition_us = durations[:2].sum()
    for name in STRETCHES:
        stretch = ir.gradient_stretch(path, 0, name)
        repetitions = stretch["num_positions"] / 2
        assert repetitions == math.ceil(WINDOW_US / repetition_us)
        assert stretch["duration_us"] == pytest.approx(repetitions * repetition_us)
        assert stretch["energy"] / repetitions == pytest.approx(heaviest, rel=1e-5)


def test_a_pad_on_the_last_slice_is_played_at_its_own_length(tmp_path):
    """Five slices 40 µs apart and a 0.3 s pad: a stretch spans packets as they play."""
    packet = [(1.0, 4e-5)] * 4 + [(1.0, 0.3)]
    path = _written(tmp_path / "packets.seq", packet * 40)
    energies, durations, starts = _played(path)
    assert ir.summary(path, SYSTEM)["subsequences"][0]["tr_size"] == 2
    for name in STRETCHES:
        stretch = ir.gradient_stretch(path, 0, name)
        first, count = stretch["first_position"], stretch["num_positions"]
        assert first in starts
        played = durations[first : first + count]
        assert stretch["duration_us"] == pytest.approx(played.sum())
        assert played.sum() >= WINDOW_US > played[:-2].sum()
    heaviest = ir.gradient_stretch(path, 0, "heaviest")
    assert heaviest["first_position"] == _reference(energies, durations, starts)


def test_a_burst_in_a_sparse_stretch_is_costed_from_where_it_plays(tmp_path):
    """A full-scale readout among sparse ones, before ten seconds of denser ones.

    The densest ten seconds hold most energy and miss the burst; the stretch
    from the heaviest repetition holds it.
    """
    sparse = [(0.05, 0.999)] * 12
    sparse[3] = (4.0, 0.999)
    path = _written(tmp_path / "burst.seq", sparse + [(0.8, 0.019)] * 600)
    energies, durations, starts = _played(path)
    heaviest = ir.gradient_stretch(path, 0, "heaviest")
    burst = ir.gradient_stretch(path, 0, "heaviest_repetition")
    assert heaviest["first_position"] == _reference(energies, durations, starts)
    assert heaviest["first_position"] >= 2 * len(sparse)
    assert burst["first_position"] == 2 * 3
    assert burst["duration_us"] >= WINDOW_US
    assert burst["energy"] < heaviest["energy"]


def test_a_subsequence_shorter_than_the_window_is_its_own_stretch(tmp_path):
    repetitions = [(1.0, 0.1)] * 20
    repetitions[7] = (1.5, 0.1)
    path = _written(tmp_path / "short.seq", repetitions)
    played_us = ir.summary(path, SYSTEM)["subsequences"][0]["tr_timing"]["played_us"]
    whole = ir.gradient_stretch(path, 0, "heaviest")
    assert (whole["first_position"], whole["num_positions"]) == (0, 40)
    assert whole["duration_us"] == pytest.approx(played_us)
    tail = ir.gradient_stretch(path, 0, "heaviest_repetition")
    assert (tail["first_position"], tail["num_positions"]) == (14, 26)


def test_an_unknown_stretch_is_refused(tmp_path):
    path = _written(tmp_path / "one.seq", [(1.0, 0.1)])
    with pytest.raises(ValueError, match="stretch"):
        ir.gradient_stretch(path, 0, "densest")
