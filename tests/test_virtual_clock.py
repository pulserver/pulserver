"""The virtual scanner's scan clock: a scan simulated in stretches and released in spans, with the sound of its gradients."""

import itertools
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import ORIENTATIONS

from pulserver import ir, virtual
from pulserver.mrd import read_chain
from pulserver.virtual import _fourier

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = ["gre_2d_3sl.seq", "epi_2d_main.seq"]
SYSTEM = pp.Opts(B0=3.0)
#: Samples a stretch ends after, so that each fixture is simulated in several.
STRETCH = 512


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Each fixture beside the cache it converts to, at the magnet's field."""
    directory = tmp_path_factory.mktemp("fixtures")
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    for name in SEQUENCES:
        ir.convert(directory / name, SYSTEM)
    return directory


@pytest.fixture(scope="module")
def tissue():
    return virtual.Phantom(
        [virtual.Ellipse((0.0, 0.0, 0.0), (0.04, 0.03), t1=0.8, t2=0.08)], coils=2
    ).tissue(4e-3, field_t=3.0)


@pytest.mark.parametrize("name", SEQUENCES)
def test_a_scan_played_in_spans_plays_its_blocks_once_each(name, converted, tissue):
    scan = virtual.Scan(converted / name, tissue)
    chunks = list(scan.chunks(0.01, sound=False))
    whole = virtual.simulate(converted / name, tissue)
    streamed = [readout for chunk in chunks for readout in chunk.readouts]

    assert chunks[0].start == 0.0 and chunks[-1].stop == scan.duration
    assert all(a.stop == b.start for a, b in itertools.pairwise(chunks))
    assert all(chunk.stop - chunk.start >= 0.01 for chunk in chunks[:-1])
    assert len(streamed) == len(whole)
    assert all(np.array_equal(a, b) for a, b in zip(streamed, whole, strict=True))


def test_a_scan_simulated_in_one_stretch_is_released_in_spans_of_the_length_asked_for(
    converted, tissue
):
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    chunks = list(scan.chunks(0.05, sound=False))

    assert len(scan._stretches(0.05)) == 1
    assert len(chunks) > 1
    assert all(0.05 <= chunk.stop - chunk.start < 0.1 for chunk in chunks[:-1])


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize("name", SEQUENCES)
def test_the_spans_of_a_scan_sound_as_its_design_sounds(
    name, rotation, converted, tissue, monkeypatch
):
    """The sound of a single-file scan, span by span across several stretches, is ``Sequence.sound`` of the design as the checks turn it."""
    monkeypatch.setattr(_fourier, "_SPAN_SAMPLES", STRETCH)
    scan = virtual.Scan(converted / name, tissue, rotation=rotation)
    sound = np.concatenate([chunk.sound for chunk in scan.chunks(0.01)], axis=1)
    ((_, design),) = read_chain(converted / name)
    designed = pp.TransformFOV(rotation=rotation).apply_to_sequence(design).sound()

    assert len(scan._stretches(0.01)) > 1
    assert sound.shape == designed.shape
    np.testing.assert_allclose(sound, designed, rtol=0, atol=1e-5)


def test_a_scan_played_at_a_speed_yields_each_span_once_its_clock_passes_it(
    converted, tissue
):
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    started = time.monotonic()
    for chunk in scan.chunks(0.05, speed=4.0, sound=False):
        assert time.monotonic() - started >= chunk.stop / 4.0
        assert chunk.sound.shape == (2, 0)


#: Played at a third of a scanner's speed, the gradient echo lasts more than
#: a second on the wall clock, and a held stretch holds the clock 2 s: both far
#: more than a shared runner's sleeps overshoot, ``JITTER``.
SPEED = 1.0 / 3.0
JITTER = 0.5


def _slowed(scan, seconds_per_sample, seconds_per_second=0.01, held=None):
    """Make ``scan`` simulate each stretch in the time given per ADC sample, or per second of scan time without one.

    The stretch that starts at block ``held``, if any, takes 2 s more.
    """
    simulate = scan._readouts

    def slow(first, last):
        samples = scan._adc_samples(first, last)
        duration = scan._starts[last] - scan._starts[first]
        time.sleep(
            seconds_per_sample * samples if samples else seconds_per_second * duration
        )
        if first == held:
            time.sleep(2.0)
        return simulate(first, last)

    scan._readouts = slow


def _offsets(scan, preparing=None):
    """Play ``scan`` at ``SPEED``; return how far after its end on the clock each span was released, in s of wall-clock time."""
    return [
        time.monotonic() - chunk.stop / SPEED
        for chunk in scan.chunks(0.05, speed=SPEED, sound=False, preparing=preparing)
    ]


def test_a_scan_simulated_slower_than_it_plays_starts_its_clock_once_it_will_not_be_held(
    converted, tissue, monkeypatch
):
    monkeypatch.setattr(_fourier, "_SPAN_SAMPLES", STRETCH)
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    stretches = scan._stretches(0.05)
    total = sum(scan._adc_samples(*stretch) for stretch in stretches)
    playing = scan.duration / SPEED
    # Twice as long to simulate as to play, all of it in the readouts: a clock
    # started at once would lag by the whole playing time.
    _slowed(scan, 2.0 * playing / total)
    reports = []
    offsets = _offsets(scan, reports.append)

    # No estimate before a stretch that acquires has been simulated.
    assert len(stretches) > 2
    assert reports[0] is None and all(left > 0.0 for left in reports if left)
    assert max(offsets) - min(offsets) < JITTER < playing


def test_a_stretch_simulated_after_its_time_holds_the_clock_and_the_rest_keep_its_pace(
    converted, tissue, monkeypatch
):
    monkeypatch.setattr(_fourier, "_SPAN_SAMPLES", STRETCH)
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    stretches = scan._stretches(0.05)
    total = sum(scan._adc_samples(*stretch) for stretch in stretches)
    held = stretches[len(stretches) // 2]
    _slowed(scan, 0.5 * scan.duration / total, held=held[0])
    chunks, offsets = [], []
    for chunk in scan.chunks(0.05, speed=SPEED, sound=False):
        offsets.append(time.monotonic() - chunk.stop / SPEED)
        chunks.append(chunk)
    split = next(
        k for k, chunk in enumerate(chunks) if chunk.start >= scan._starts[held[0]]
    )

    before, after = offsets[:split], offsets[split:]
    assert max(before) - min(before) < JITTER
    assert min(after) > max(before) + 2 * JITTER
    assert max(after) - min(after) < JITTER


def test_a_scan_closed_while_playing_stops_simulating(converted, tissue):
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    _slowed(scan, 0.0, seconds_per_second=1.0)
    chunks = scan.chunks(0.05, sound=False)
    next(chunks)
    chunks.close()

    assert not any(t.name == "pulserver-scan" for t in threading.enumerate())


def test_an_error_in_the_simulation_is_raised_where_the_spans_are_taken(
    converted, tissue
):
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)

    def failing(first, last):
        raise RuntimeError("the engine refused a block")

    scan._readouts = failing
    for speed in (None, 1.0):
        with pytest.raises(RuntimeError, match="refused a block"):
            list(scan.chunks(0.05, speed=speed, sound=False))


@pytest.mark.parametrize(
    ("given", "message"),
    [({"length": 0.0}, "positive time"), ({"speed": 0.0}, "speed")],
)
def test_a_span_of_no_time_or_a_scan_at_no_speed_is_refused(
    given, message, converted, tissue
):
    scan = virtual.Scan(converted / "gre_2d_3sl.seq", tissue)
    with pytest.raises(ValueError, match=message):
        next(scan.chunks(**given))
