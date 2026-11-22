"""The cine reconstruction: cardiac phases from the trigger stamps, the binning of the lines, and the frames of a beating heart."""

import _cine
import numpy as np
import pytest
from conftest import play

from pulserver.recon.handlers.cine import (
    CineRecon,
    binned,
    cardiac_phases,
    realtime,
)

pytest.importorskip("bartorch")

MATRIX, COILS, TR = 48, 8, 3e-3
HEART = (slice(14, 34), slice(14, 36))


def test_a_readout_is_placed_in_its_beat_by_the_last_trigger_and_the_next():
    times = np.array([600.0, 900.0, 1250.0])
    beats = np.array([500.0, 500.0, 1100.0])
    later = np.array([1700.0])
    phases, rr = cardiac_phases(
        np.concatenate([times, later]), np.concatenate([beats, [1700.0]])
    )

    assert rr == pytest.approx(600.0)
    assert phases[:3] == pytest.approx([100 / 600, 400 / 600, 150 / 600])


def test_the_readouts_of_a_premature_beat_and_its_compensatory_pause_are_rejected():
    waves = np.cumsum([100.0, 800, 800, 480, 1120, 800, 800])
    times = np.arange(waves[0], waves[-1], 20.0)
    beats = waves[np.searchsorted(waves, times, side="right") - 1]

    phases, rr = cardiac_phases(times, beats)

    assert rr == pytest.approx(800.0)
    rejected = (beats == waves[2]) | (beats == waves[3])
    assert np.isnan(phases[rejected]).all()
    assert np.isfinite(phases[~rejected & (beats < waves[-1])]).all()


def test_readouts_before_the_first_trigger_are_rejected():
    phases, _ = cardiac_phases(
        np.array([10.0, 600.0, 1400.0]), np.array([0.0, 500.0, 1300.0])
    )

    assert np.isnan(phases[0])


def test_binning_needs_two_triggers():
    with pytest.raises(ValueError, match="two"):
        cardiac_phases(np.array([1.0, 2.0]), np.array([0.0, 0.0]))


def test_an_interpolated_line_weighs_its_two_nearest_readouts_around_the_beat():
    phases = np.array([0.1, 0.3, 0.95])
    lines = np.zeros(3, int)

    weights = binned(phases, lines, 1, 4, interpolate=True)

    # Phase 0.25 lies three quarters of the way from 0.1 to 0.3.
    assert weights[1, 0] == pytest.approx([0.25, 0.75, 0.0])
    # Phase 0 lies between 0.95 and 0.1 across the end of the beat.
    assert weights[0, 0] == pytest.approx([1 / 3, 0.0, 2 / 3])
    assert weights.sum(axis=2) == pytest.approx(1.0)


def test_a_binned_line_takes_only_the_readouts_within_one_phase():
    phases = np.array([0.0, 0.05, 0.6, np.nan])
    lines = np.array([0, 0, 0, 0])

    weights = binned(phases, lines, 2, 8, interpolate=False)

    assert weights[0, 0, :2].sum() == pytest.approx(1.0)
    assert weights[0, 0, 2:] == pytest.approx(0.0)
    assert weights[6, 0].sum() == pytest.approx(0.0)
    assert weights[:, 1].sum() == pytest.approx(0.0)


def test_a_real_time_frame_is_a_run_of_consecutive_readouts():
    assert realtime(np.arange(12), 4).tolist() == [0] * 4 + [1] * 4 + [2] * 4


class Frames(CineRecon):
    def __init__(self, **kwargs):
        super().__init__(
            kspace_filter=False,
            gradient_unwarp=False,
            intensity_correction=False,
            iterations=40,
            **kwargs,
        )
        self.results = []

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.append(result)
        return result


def _segmented(views, cycles, n_segments):
    """A retrospective train: each segment of ``views`` lines cycled ``cycles`` times, one TR apart."""
    lines, phases, segments = [], [], []
    for s in range(n_segments):
        for c in range(cycles):
            lines += list(range(s * views, (s + 1) * views))
            phases += [c] * views
            segments += [s] * views
    count = len(lines)
    return _cine.Timeline(
        np.arange(count) * TR + 0.05,
        np.asarray(lines),
        np.asarray(phases),
        np.asarray(segments),
        np.zeros(count, int),
    )


def _stream(played, waves, parameters, frames):
    maps = _cine.body_array(COILS, MATRIX)

    def kspace(state):
        return _cine.centred_fft(maps * _cine.short_axis(MATRIX, state))

    acquisitions = _cine.acquisitions(played, waves, kspace)
    header = _cine.header(COILS, MATRIX, frames, parameters)
    return acquisitions, header


def _images(plugin):
    return np.stack([np.asarray(r.data) for rs in plugin.results if rs for r in rs])


def _cavity(image):
    """Blood pool area of the left ventricle: pixels of the heart above half the brightest."""
    heart = image[HEART]
    return int((heart > 0.5 * heart.max()).sum())


def test_a_retrospective_cine_follows_the_heart_through_systole(device):
    """End systole, a third of the beat in, shows a smaller cavity than end diastole."""
    waves = _cine.r_waves(20.0, rr=0.8, variability=0.0)
    played = _segmented(views=8, cycles=round(0.8 / (8 * TR)), n_segments=6)
    acquisitions, header = _stream(
        played,
        waves,
        {"Gating": "retrospective", "CardiacPhases": 12},
        int(played.phases.max()) + 1,
    )
    plugin = Frames()

    play(plugin, header, acquisitions, device=device)
    images = _images(plugin)

    assert images.shape == (12, MATRIX, MATRIX)
    assert _cavity(images[4]) < 0.8 * _cavity(images[0])


def test_a_prospective_cine_makes_one_image_per_cardiac_phase(device):
    waves = _cine.r_waves(20.0, rr=0.8, variability=0.0)
    lines = np.tile(np.arange(MATRIX), 3)
    phases = np.repeat(np.arange(3), MATRIX)
    count = lines.size
    played = _cine.Timeline(
        np.arange(count) * TR + 0.05,
        lines,
        phases,
        np.zeros(count, int),
        np.zeros(count, int),
    )
    acquisitions, header = _stream(played, waves, {"Gating": "prospective"}, 3)
    plugin = Frames()

    play(plugin, header, acquisitions, device=device)

    assert _images(plugin).shape == (3, MATRIX, MATRIX)


def test_a_golden_free_running_series_makes_real_time_frames_of_consecutive_readouts(
    device,
):
    waves = _cine.r_waves(20.0, rr=0.8, variability=0.0)
    order = np.argsort(np.argsort((np.arange(MATRIX) / 1.618034) % 1.0))
    lines = np.tile(order, 4)
    count = lines.size
    played = _cine.Timeline(
        np.arange(count) * TR + 0.05,
        lines,
        np.repeat(np.arange(4), MATRIX),
        np.zeros(count, int),
        np.zeros(count, int),
    )
    acquisitions, header = _stream(
        played, waves, {"Gating": "none", "Ordering": "golden"}, 4
    )
    plugin = Frames(realtime_acceleration=4)

    play(plugin, header, acquisitions, device=device)

    assert _images(plugin).shape == (count // (MATRIX // 4), MATRIX, MATRIX)


def test_a_golden_series_binned_by_the_ecg_makes_the_cardiac_phases_asked_for(device):
    waves = _cine.r_waves(30.0, rr=0.8, variability=0.0)
    order = np.argsort(np.argsort((np.arange(MATRIX) / 1.618034) % 1.0))
    lines = np.tile(order, 60)
    count = lines.size
    played = _cine.Timeline(
        np.arange(count) * TR + 0.05,
        lines,
        np.repeat(np.arange(60), MATRIX),
        np.zeros(count, int),
        np.zeros(count, int),
    )
    acquisitions, header = _stream(
        played,
        waves,
        {"Gating": "none", "Ordering": "golden", "CardiacPhases": 10},
        60,
    )
    plugin = Frames(binning="ecg")

    play(plugin, header, acquisitions, device=device)
    images = _images(plugin)

    assert images.shape == (10, MATRIX, MATRIX)
    assert _cavity(images[3]) < 0.8 * _cavity(images[0])


def test_binning_by_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="binning"):
        CineRecon(binning="eeg")
