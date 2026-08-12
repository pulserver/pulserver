"""Looking at a sequence: the window handed to PyPulseq, and the plot it draws."""

from __future__ import annotations

import matplotlib
import numpy as np
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from matplotlib import pyplot as plt
from pulserver.pypulseq import _safety
from pulserver.pypulseq import _sequence as sequence_module
from pulserver.pypulseq._pulseqpp import to_upstream
from pulserver.pypulseq._rotate3d import rotate3D
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _no_open_figures():
    """Every test here draws; none of them should leave a window behind."""
    yield
    plt.close("all")


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


def _events(make, system):
    """One repetition's worth, from ``make``'s factories -- ours or upstream's.

    The two sets are built from the same numbers, so a sequence built either way
    holds the same rows; what differs is only which libraries they travelled
    through, which is exactly what the conversion is being asked about.
    """
    rf, gz, gzr = make.make_sinc_pulse(
        flip_angle=np.pi / 6,
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        use="excitation",
    )
    return {
        "make": make,
        "rf": rf,
        "gz": gz,
        "gzr": gzr,
        "gx": make.make_trapezoid(channel="x", flat_area=581.8, flat_time=3.2e-3, system=system),
        "gy": make.make_arbitrary_grad(
            channel="y", waveform=np.sin(np.linspace(0, np.pi, 200)) * 1e5, system=system
        ),
        "adc": make.make_adc(num_samples=128, dwell=2.4e-5, delay=1e-4, system=system),
        "trigger": make.make_trigger(channel="physio1", duration=1e-3, system=system),
    }


@pytest.fixture
def events(system):
    return _events(pp, system)


def _repetition(built, events, line, extra=()):
    """One repetition, appended to whichever kind of sequence is handed in."""
    make = events["make"]
    events["rf"].phase_offset = 0.5 * line
    built.add_block(events["rf"], events["gz"])
    built.add_block(events["gzr"])
    built.add_block(
        events["gx"],
        events["gy"],
        events["adc"],
        *extra,
        *([events["trigger"]] if line == 0 else []),
        make.make_label(type="SET", label="LIN", value=line),
    )
    # numID is stated rather than left to be assigned: a slotted soft delay
    # carries one from the start, where upstream hands one out as the block is
    # added, and the two would not otherwise agree.
    built.add_block(
        make.make_delay(5e-3), make.make_soft_delay(hint="TR", numID=0, offset=1e-3)
    )


def _built(sequence, events):
    for line in range(4):
        _repetition(sequence, events, line)
    return sequence


@pytest.fixture
def plain(system, events):
    """Four repetitions of everything upstream understands, and nothing else."""
    return _built(pp.Sequence(system=system), events)


@pytest.fixture
def mirror(system):
    """The same sequence again, built straight into upstream PyPulseq's own."""
    return _built(upstream.Sequence(system=system), _events(upstream, system))


@pytest.fixture
def turned(system, events):
    """Four repetitions, each rotated further, and a pTx-shimmed pulse after."""
    built = pp.Sequence(system=system)
    for line in range(4):
        _repetition(
            built,
            events,
            line,
            extra=(pp.make_rotation(Rotation.from_euler("z", 30.0 * line, degrees=True)),),
        )
    built.add_block(events["rf"], pp.make_rf_shim([1.0, 0.5j, -0.25]))
    return built


def _labels(block):
    """A decoded block's labels, in the order it reports them."""
    return [(one.type, one.label, one.value) for one in (block.label or {}).values()]


def _triggers(block):
    """A decoded block's triggers, likewise. Upstream leaves the field off."""
    return [
        (one.type, one.channel, one.delay, one.duration)
        for one in getattr(block, "trigger", {}).values()
    ]


def _same_event(ours, theirs, where):
    assert (ours is None) == (theirs is None), where
    if theirs is None:
        return
    for field, expected in vars(theirs).items():
        found = getattr(ours, field)
        if isinstance(expected, np.ndarray):
            assert np.allclose(found, expected), f"{where}.{field}"
        elif isinstance(expected, (int, float, np.floating)):
            assert np.isclose(found, expected), f"{where}.{field}"
        else:
            assert found == expected, f"{where}.{field}"


# %% the window


def test_a_window_decodes_exactly_as_upstream_would(plain, mirror):
    """The rows are PyPulseq's rows; carrying them across changes nothing."""
    window = plain._upstream_window(1, len(plain))

    assert list(window.block_events) == list(mirror.block_events)
    for index in mirror.block_events:
        ours, theirs = window.get_block(index), mirror.get_block(index)
        assert np.isclose(ours.block_duration, theirs.block_duration)
        for name in ("rf", "gx", "gy", "gz", "adc"):
            _same_event(getattr(ours, name), getattr(theirs, name), f"block {index} {name}")
        assert _labels(ours) == _labels(theirs), f"block {index} labels"
        assert _triggers(ours) == _triggers(theirs), f"block {index} triggers"
        _same_event(ours.soft_delay, theirs.soft_delay, f"block {index} soft delay")


def test_only_the_blocks_the_time_range_touches_are_decoded(plain):
    """The point of windowing: a slice of a scan costs a slice of the blocks."""
    edges = np.concatenate(([0.0], np.cumsum(plain.block_durations)))
    first, last = plain._blocks_over(edges[4] + 1e-9, edges[6] - 1e-9)

    assert (first, last) == (5, 6)
    window = plain._upstream_window(first, last)
    assert len(window.block_events) == 3  # blocks 5 and 6, behind a lead-in


def test_a_window_keeps_the_block_numbers_and_the_time_it_is_played_at(plain):
    """A window is drawn where it belongs on the axis, not from zero."""
    edges = np.concatenate(([0.0], np.cumsum(plain.block_durations)))
    window = plain._upstream_window(5, 6)

    assert list(window.block_events) == [0, 5, 6]
    assert np.isclose(window.block_durations[0], edges[4])
    assert np.isclose(window.block_durations[5], plain.block_durations[4])
    assert window.get_block(0).rf is None  # the lead-in is silence


def test_a_window_of_a_whole_sequence_has_no_lead_in(plain):
    window = plain._upstream_window(1, len(plain))
    assert 0 not in window.block_events


# %% what upstream has no vocabulary for


def test_a_rotation_arrives_resolved_into_the_gradients(turned, system):
    """What is drawn is what the scanner plays, not the base waveform."""
    assert turned._rotated_or_shimmed(1, len(turned))
    window = turned._upstream_window(1, len(turned))

    readout = 4 * 2 + 3  # the third repetition's readout, turned 60 degrees
    block = turned.get_block(readout)
    wanted = rotate3D(
        block.gx,
        block.gy,
        rotation_matrix=Rotation.from_euler("z", 60.0, degrees=True).as_matrix(),
        system=system,
    )
    found = window.get_block(readout)
    for gradient in wanted:
        _same_event(getattr(found, f"g{gradient.channel}"), gradient, f"rotated g{gradient.channel}")


def test_an_rf_shim_arrives_spread_across_the_channels(turned):
    """Every channel over the same interval, so upstream needs no pTx awareness."""
    window = turned._upstream_window(1, len(turned))
    base = turned.get_block(len(turned)).rf
    shimmed = window.get_block(len(turned)).rf

    weights = np.array([1.0, 0.5j, -0.25])
    assert np.allclose(shimmed.signal, (weights[:, None] * base.signal[None, :]).ravel())
    assert np.allclose(shimmed.t, np.tile(base.t, weights.size))


def test_the_extensions_upstream_cannot_read_are_gone_from_the_window(turned):
    """`get_block` raises on an unknown extension id; these would be two."""
    window = turned._upstream_window(1, len(turned))

    assert set(window.extension_string_idx) <= {"TRIGGERS", "LABELSET", "LABELINC", "DELAYS"}
    for index in window.block_events:
        window.get_block(index)  # would raise if a ROTATIONS node had survived


def test_a_custom_label_is_dropped_rather_than_read_as_another(system, events):
    """Upstream resolves a label by indexing its own fixed list of names."""
    built = pp.Sequence(system=system)
    built.add_block(events["gx"], pp.make_label(type="SET", label="BIN", value=3))
    built.add_block(events["gx"], pp.make_label(type="SET", label="LIN", value=1))

    window = to_upstream(built)
    assert window.get_block(1).label is None
    assert [label.label for label in window.get_block(2).label.values()] == ["LIN"]


# %% the plot


def test_unstacked_is_one_window_of_three_rows_by_two_columns(plain):
    plot = plain.plot(plot_now=False)

    assert plot.fig1 is plot.fig2
    assert [axis.get_ylabel() for axis in plot.fig1.axes] == [
        "ADC",
        "RF mag (Hz)",
        "RF/ADC phase (rad)",
        "Gx (kHz/m)",
        "Gy (kHz/m)",
        "Gz (kHz/m)",
    ]
    assert {axis.get_subplotspec().get_gridspec().get_geometry() for axis in plot.fig1.axes} == {(3, 2)}


def test_the_two_columns_share_their_time_axis(plain):
    """Moving an axis between figures drops it out of the shared group."""
    plot = plain.plot(plot_now=False)

    plot.ax1[0].set_xlim(0.001, 0.004)
    assert plot.ax2[2].get_xlim() == pytest.approx((0.001, 0.004))


def test_stacked_is_upstreams_single_figure(plain):
    plot = plain.plot(plot_now=False, stacked=True)

    assert plot.fig2 is None
    assert len(plot.fig1.axes) == 6


def test_an_overlay_draws_over_a_merged_plot(plain, turned):
    """Comparing two sequences still works, and still ends up in one window."""
    first = plain.plot(plot_now=False)
    lines = sum(len(axis.get_lines()) for axis in first.fig1.axes)

    second = turned.plot(plot_now=False, overlay=first)

    assert second.fig1 is second.fig2
    assert len(second.fig1.axes) == 6
    assert sum(len(axis.get_lines()) for axis in second.fig1.axes) > lines


def test_a_windowed_plot_is_drawn_where_it_is_played(plain):
    edges = np.concatenate(([0.0], np.cumsum(plain.block_durations)))
    plot = plain.plot(plot_now=False, time_range=(edges[4], edges[6]))

    assert plot.ax1[0].get_xlim() == pytest.approx((edges[4], edges[6]))


def test_drawing_a_whole_long_sequence_says_so_first(plain, monkeypatch):
    monkeypatch.setattr(sequence_module, "_LOUD_ABOVE", 4)

    with pytest.warns(UserWarning, match="pass time_range"):
        plot = plain.plot(plot_now=False)

    assert plot.fig1 is not None  # said out loud, then drawn anyway


def test_a_time_range_that_ends_before_it_begins_is_refused(plain):
    with pytest.raises(ValueError, match="end after it begins"):
        plain.plot(plot_now=False, time_range=(0.02, 0.01))


# %% the canonical TR
#
# The claim under test throughout is one claim: what is drawn under `tr` is
# what the C safety core built, not something rebuilt here that resembles it.
# So these compare the picture against the core's own arrays rather than
# against a fixture -- if the viewer and the interpreter could disagree, they
# would disagree here.

NUM_INSTANCES = 16


@pytest.fixture
def encoded(system):
    """A phase-encoded gradient echo: one structural TR, a varying gradient.

    The phase encode sweeps through zero, so no instance is the envelope over
    them and ``zero_variable`` has something to zero. Readout and RF are
    constant, which is what makes the three canonical views tell apart.
    """
    built = pp.Sequence(system=system)
    rf = pp.make_block_pulse(flip_angle=np.pi / 6, duration=1e-3, system=system)
    gx = pp.make_trapezoid(channel="x", area=2000, system=system)
    adc = pp.make_adc(
        num_samples=128, duration=gx.flat_time, delay=gx.rise_time, system=system
    )
    for index in range(NUM_INSTANCES):
        gy = pp.make_trapezoid(
            channel="y", area=-1000 + 125 * index, duration=1e-3, system=system
        )
        built.add_block(rf)
        built.add_block(gy)
        built.add_block(gx, adc)
        built.add_block(pp.make_delay(5e-3))
    built.remove_duplicates(in_place=True)
    return built


def _drawn(axis) -> np.ndarray:
    """Every point on ``axis``, as one ``(N, 2)`` array of x against y."""
    points = [np.stack(line.get_data(), axis=-1) for line in axis.get_lines()]
    return np.concatenate(points) if points else np.zeros((0, 2))


def _panels(plot) -> dict:
    """The six panels of a merged plot, by the channel each one carries."""
    return dict(
        zip(("adc", "rf_mag", "rf_phase", "gx", "gy", "gz"), plot.fig1.axes, strict=True)
    )


@pytest.mark.parametrize("tr", ["worst_case", "zero_variable", 0], ids=str)
def test_a_canonical_tr_is_drawn_whole_rather_than_gradients_only(encoded, tr):
    """RF, ADC and gradients all arrive: the core reports them in one call.

    The binding used to hand back the three gradient axes and drop everything
    else, which would have left three of the six panels empty however the TR
    was asked for.
    """
    panels = _panels(encoded.plot(plot_now=False, tr=tr))

    assert len(_drawn(panels["adc"])), "no ADC was drawn"
    assert len(_drawn(panels["rf_mag"])), "no RF magnitude was drawn"
    assert len(_drawn(panels["gx"])), "no readout gradient was drawn"


@pytest.mark.parametrize("tr", ["worst_case", "zero_variable", 0, NUM_INSTANCES - 1], ids=str)
def test_what_is_drawn_is_the_waveform_the_safety_core_handed_over(encoded, tr):
    """The viewer and the analyses read the same array, to the sample.

    ``calculate_pns(tr=...)`` and ``calculate_gradient_spectrum(tr=...)`` reach
    their waveform through ``TRSequence.waveforms()``; the plot reaches it
    through ``get_block``. This says the two cuts of it agree, which is what
    stops a picture and a predownload verdict from disagreeing.
    """
    panels = _panels(encoded.plot(plot_now=False, tr=tr))
    extracted = encoded._structure_for("test").waveform(tr)

    for axis, channel in zip("xyz", extracted.waveforms(), strict=True):
        times, amplitudes = channel
        carries = np.any(amplitudes)
        points = _drawn(panels[f"g{axis}"])
        if not carries:
            assert not len(points), f"g{axis} carries nothing but was drawn"
            continue
        # kHz/m is upstream's default gradient unit; the time axis is seconds.
        drawn = points[np.argsort(points[:, 0])]
        wanted = np.interp(drawn[:, 0], times, amplitudes) * 1e-3
        assert np.allclose(drawn[:, 1], wanted, atol=1e-6), f"g{axis}"


def test_the_worst_case_tr_drawn_bounds_every_instance_drawn(encoded):
    """The claim ``worst_case`` exists to make, read off the picture.

    Its gradient at each position is the largest any instance reaches there,
    so no instance can be drawn outside it.
    """
    structure = encoded._structure_for("test")
    envelope = [channel[1] for channel in structure.waveform("worst_case").waveforms()]

    for index in range(NUM_INSTANCES):
        played = structure.waveform(index).waveforms()
        for axis, bound, channel in zip("xyz", envelope, played, strict=True):
            assert np.all(np.abs(channel[1]) <= np.abs(bound) + 1e-3), f"tr={index} g{axis}"


def test_zeroing_the_variable_gradients_keeps_the_constant_ones(encoded):
    """``zero_variable`` is the structure the constant gradients make.

    The readout is the same in every instance and survives; the phase encode
    differs between them and goes.
    """
    panels = _panels(encoded.plot(plot_now=False, tr="zero_variable"))

    assert len(_drawn(panels["gx"])), "the constant readout was zeroed too"
    assert not len(_drawn(panels["gy"])), "the varying phase encode survived"


def test_an_instance_is_drawn_as_it_really_plays(encoded):
    """Signed amplitudes and all -- which is what makes it a check on the envelope.

    The phase encode sweeps through zero, so one instance has no gradient at
    all and the ones either side of it are drawn with opposite signs.
    """
    structure = encoded._structure_for("test")
    encodes = [structure.waveform(index).waveforms()[1][1] for index in range(NUM_INSTANCES)]
    extremes = [channel[np.argmax(np.abs(channel))] for channel in encodes]

    assert min(extremes) < 0 < max(extremes)
    assert not np.any(_drawn(_panels(encoded.plot(plot_now=False, tr=8))["gy"])[:, 1])


def test_the_blocks_drawn_are_the_blocks_the_core_reported(encoded):
    """Not blocks recovered from the timeline: the core's own descriptors.

    Their durations have to sum to the TR the same core reports, or the
    picture is a different length from the thing being analysed.
    """
    extracted = encoded._structure_for("test").waveform("worst_case")

    assert len(extracted.block_events) == len(extracted.block_durations) > 1
    assert sum(extracted.block_durations.values()) == pytest.approx(
        encoded._structure_for("test").tr_duration
    )


def test_a_tr_is_drawn_from_its_own_clock(encoded):
    """A TR is not on the timeline, so its axis starts where the TR does."""
    plot = encoded.plot(plot_now=False, tr="worst_case")
    duration = encoded._structure_for("test").tr_duration

    assert plot.ax1[0].get_xlim() == pytest.approx((0.0, duration))


def test_a_time_range_zooms_into_the_tr(encoded):
    """Zooming works under `tr` too; it just means time within the TR."""
    plot = encoded.plot(plot_now=False, tr="worst_case", time_range=(0.002, 0.004))

    assert plot.ax1[0].get_xlim() == pytest.approx((0.002, 0.004))


def test_the_timeline_is_still_drawn_when_no_tr_is_asked_for(encoded):
    """`tr=None` is untouched: the whole sequence, not one repetition of it."""
    plot = encoded.plot(plot_now=False)
    structure = encoded._structure_for("test")

    assert plot.ax1[0].get_xlim()[1] == pytest.approx(
        structure.tr_duration * NUM_INSTANCES
    )


# %% what cannot be asked for


def test_a_label_cannot_be_marked_on_a_canonical_tr(encoded):
    """The core builds the TR out of waveforms and carries no label values."""
    with pytest.raises(ValueError, match="carries none"):
        encoded.plot(plot_now=False, tr="worst_case", label="LIN")


def test_an_instance_past_the_last_one_is_refused(encoded):
    with pytest.raises(ValueError, match="out of range"):
        encoded.plot(plot_now=False, tr=NUM_INSTANCES)


def test_a_tr_named_something_the_core_has_no_mode_for_is_refused(encoded):
    with pytest.raises(ValueError, match="worst_case"):
        encoded.plot(plot_now=False, tr="steady_state")


def test_every_instance_the_core_indexes_can_be_asked_for(encoded):
    """The bound is the core's own, not the count of structural TRs."""
    structure = encoded._structure_for("test")

    assert structure.num_instances == NUM_INSTANCES
    structure.resolve(NUM_INSTANCES - 1)  # would raise if the bound were short


# %% many transmit channels
#
# The core writes a block's RF channel-major *within the block*, repeating the
# time base once per channel, so a pTx TR's time array runs backwards at every
# channel boundary and cannot be searched -- only bucketed by block and then
# cut into equal parts. These pin that down directly, because the shimmed
# sequences reachable from here come back single-channel and would not
# exercise it.

RF_RASTER = 1e-6


def _two_channel_tr(system, num_channels=3):
    """A two-block TR whose second block transmits on ``num_channels``."""
    samples = 8
    times = (np.arange(samples) + 0.5) * RF_RASTER + 0.002
    return _safety.TRSequence(
        system,
        [np.zeros((2, 2)) for _ in range(3)],
        duration=0.004,
        blocks=[
            {"start": 0.0, "duration": 0.002, "segment": 0},
            {"start": 0.002, "duration": 0.002, "segment": 0},
        ],
        rf=(
            np.tile(times, num_channels),
            np.concatenate([np.full(samples, 10.0 * (c + 1)) for c in range(num_channels)]),
            np.concatenate([np.full(samples, 0.5 * c) for c in range(num_channels)]),
        ),
        num_rf_channels=num_channels,
    )


def test_a_ptx_block_is_cut_into_one_pulse_per_transmit_channel(system):
    """Read off the repeated time base, not off the TR-wide channel count."""
    channels = _two_channel_tr(system).rf_over(2)

    assert len(channels) == 3
    for index, (times, magnitudes, phases) in enumerate(channels):
        assert times.size == magnitudes.size == phases.size == 8
        assert np.allclose(magnitudes, 10.0 * (index + 1))
        assert np.allclose(phases, 0.5 * index)
    # Every channel transmits over the same interval, which is the point.
    assert all(np.array_equal(channels[0][0], other[0]) for other in channels[1:])


def test_a_block_that_does_not_transmit_reports_no_rf(system):
    assert _two_channel_tr(system).rf_over(1) == []
    assert _two_channel_tr(system).get_block(1).rf is None


def test_the_block_drawn_by_upstream_is_the_channel_that_was_asked_for(system):
    """Upstream's block model has one RF event, so one channel goes through it."""
    tr = _two_channel_tr(system)

    tr.rf_channel = 2
    assert np.allclose(np.abs(tr.get_block(2).rf.signal), 30.0)
    tr.rf_channel = 0
    assert np.allclose(np.abs(tr.get_block(2).rf.signal), 10.0)


def test_the_remaining_transmit_channels_are_added_to_the_panels(system):
    """The two upstream drew nothing on, drawn from the same arrays."""
    tr = _two_channel_tr(system)
    _, (magnitude, phase) = plt.subplots(2)

    _safety.overlay_rf_channels(tr, magnitude, phase, time_factor=1.0)

    assert len(magnitude.get_lines()) == len(phase.get_lines()) == 2
    assert sorted(float(line.get_ydata()[0]) for line in magnitude.get_lines()) == [20.0, 30.0]
    assert [text.get_text() for text in magnitude.get_legend().get_texts()] == ["Tx1", "Tx2"]


def test_a_single_transmit_tr_is_left_exactly_as_upstream_drew_it(system):
    """The overlay is a pTx affordance and must not touch the ordinary case."""
    tr = _two_channel_tr(system, num_channels=1)
    _, (magnitude, phase) = plt.subplots(2)

    _safety.overlay_rf_channels(tr, magnitude, phase, time_factor=1.0)

    assert magnitude.get_lines() == [] and magnitude.get_legend() is None
