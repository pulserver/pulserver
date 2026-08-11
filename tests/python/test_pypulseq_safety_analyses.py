"""``calculate_pns`` and ``calculate_gradient_spectrum``: the timeline and the TR.

Two claims are under test and they pull in opposite directions.

``tr=None`` must be upstream PyPulseq, to the bit -- a script that ran against
``pypulseq.Sequence`` has to get the same numbers here, so those tests compare
against upstream running on the same sequence rather than against a fixture.

``tr=`` must be the C safety core, because the point of it is that a plot and
a predownload verdict cannot disagree. Those tests check the properties that
make the core's answer meaningful: that its object is the per-sample envelope
over every TR instance rather than any one of them, that the lines it reports
sit at harmonics of the repetition time, and that the SAFE model now in C is
the one upstream implements in Python.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest
from pulserver.pypulseq import _safety
from pypulseq.utils.safe_pns_prediction import safe_example_hw, safe_pns_model

matplotlib.use("Agg")

#: Recorded sequences with a real repetition structure, shared with the C
#: engine's own tests.
EXPECTED = Path(__file__).resolve().parents[1] / "utils" / "expected"

#: An Irnich model in the form ``calculate_pns`` recognises, with GE-shaped
#: constants. Not a calibrated table -- it exists to exercise the branch.
IRNICH = {"chronaxie": 334e-6, "rheobase": 23.4}


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


@pytest.fixture
def num_trs():
    return 16


@pytest.fixture
def seq(system, num_trs):
    """A phase-encoded gradient echo: one structural TR, varying amplitude.

    The phase encode sweeps through zero, so the instances genuinely differ
    and the per-sample maximum over them is a waveform that no single TR
    plays. That is the whole distinction ``tr=`` exists to make.
    """
    built = pp.Sequence(system=system)
    # `use` is not decoration: what a pulse does to k -- reset it, negate it,
    # leave it -- is what it is for, and the trajectory refuses to guess.
    rf = pp.make_block_pulse(
        flip_angle=np.pi / 6, duration=1e-3, system=system, use="excitation"
    )
    gx = pp.make_trapezoid(channel="x", area=2000, system=system)
    for index in range(num_trs):
        gy = pp.make_trapezoid(
            channel="y", area=-1000 + 125 * index, duration=1e-3, system=system
        )
        built.add_block(rf)
        built.add_block(gy)
        built.add_block(gx)
        built.add_block(pp.make_delay(5e-3))
    built.remove_duplicates()
    return built


@pytest.fixture
def mirror(seq, system, tmp_path):
    """The same sequence, held by upstream PyPulseq, via a file."""
    path = tmp_path / "mirror.seq"
    seq.write(path)
    read = upstream.Sequence(system=system)
    read.read(str(path))
    return read


# %% tr=None is upstream, to the bit


def test_pns_over_the_timeline_is_upstreams_answer_exactly(seq, mirror):
    ours = seq.calculate_pns(safe_example_hw(), do_plots=False)
    theirs = mirror.calculate_pns(safe_example_hw(), do_plots=False)

    assert ours[0] == theirs[0]
    for ours_array, theirs_array in zip(ours[1:], theirs[1:], strict=True):
        assert np.array_equal(ours_array, theirs_array)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"time_range": [0.02, 0.06]},
        {"combine_mode": "rss", "use_derivative": True},
        {"combine_mode": "none"},
        {"window_width": 0.01, "frequency_oversampling": 2.0},
    ],
    ids=["defaults", "time_range", "rss_derivative", "uncombined", "window"],
)
def test_gradient_spectrum_over_the_timeline_is_upstreams_answer_exactly(seq, mirror, options):
    ours = seq.calculate_gradient_spectrum(plot=False, max_frequency=3000.0, **options)
    theirs = mirror.calculate_gradient_spectrum(plot=False, max_frequency=3000.0, **options)

    assert len(ours) == len(theirs) == 4
    ours_spectrograms, theirs_spectrograms = ours[0], theirs[0]
    assert len(ours_spectrograms) == len(theirs_spectrograms)
    for ours_axis, theirs_axis in zip(ours_spectrograms, theirs_spectrograms, strict=True):
        assert np.array_equal(ours_axis, theirs_axis)
    for ours_array, theirs_array in zip(ours[1:], theirs[1:], strict=True):
        assert np.array_equal(np.asarray(ours_array), np.asarray(theirs_array))


def test_the_timeline_answer_never_grows_a_fifth_element(seq):
    """The return shape is upstream's until something upstream cannot express."""
    assert len(seq.calculate_gradient_spectrum(plot=False)) == 4
    assert len(seq.calculate_gradient_spectrum(plot=False, tr="worst_case")) == 4
    assert len(seq.calculate_pns(safe_example_hw(), do_plots=False)) == 4
    assert len(seq.calculate_pns(safe_example_hw(), do_plots=False, tr="worst_case")) == 4


# %% tr= is the safety core


def test_the_worst_case_tr_bounds_every_instance_it_stands_for(seq, num_trs):
    """The claim the acoustic and PNS gates both rest on.

    ``pulseg_check_safety`` judges a waveform assembled from the per-sample
    maximum over every TR instance, and calls the result safe for all of
    them. If any instance read higher than the envelope, that reasoning would
    be unsound -- so this is the assertion protecting the gate, not a
    regression baseline.
    """
    envelope = seq.calculate_pns(safe_example_hw(), do_plots=False, tr="worst_case")[1].max()
    instances = [
        seq.calculate_pns(safe_example_hw(), do_plots=False, tr=index)[1].max()
        for index in range(num_trs)
    ]

    assert max(instances) <= envelope * (1 + 1e-6)
    # ... and the envelope is not simply the first instance's answer repeated.
    assert len(set(np.round(instances, 6))) > 1


def test_the_instance_index_selects_a_real_and_different_tr(seq, num_trs):
    """``tr=<int>`` reads AMP_ACTUAL, so the phase encode really varies.

    The sweep passes through zero at instance 8, which no envelope could show:
    it is the one TR whose phase-encode lobe is absent.
    """
    structure = seq._structure_for("test")
    peaks = []
    for index in range(num_trs):
        waveform = structure.waveform(index)
        peaks.append(np.abs(waveform.waveforms()[1][1]).max())

    assert peaks[8] == pytest.approx(0.0, abs=1e-9)
    assert max(peaks) > 0.0
    assert np.abs(structure.waveform("worst_case").waveforms()[1][1]).max() == pytest.approx(
        max(peaks), rel=1e-6
    )


def test_a_repetition_time_is_transformed_whole_into_one_spectrum(seq):
    """Under ``tr`` the window is the TR, so this is a spectrum, not a spectrogram.

    A periodic waveform has no time-varying spectrum to chart -- it has energy
    only at multiples of ``1 / T_TR`` -- and cutting it into shorter windows
    would smear neighbouring harmonics together instead of resolving them.
    """
    structure = seq._structure_for("test")
    uncombined, _, _, times = seq.calculate_gradient_spectrum(
        plot=False, max_frequency=3000.0, tr="worst_case", combine_mode="none"
    )
    assert np.asarray(times).size == 1
    assert np.asarray(uncombined[0]).shape[1] == 1

    # With one window there is nothing to combine, so the collapsed answer is
    # that single column unchanged.
    combined, _, frequencies, _ = seq.calculate_gradient_spectrum(
        plot=False, max_frequency=3000.0, tr="worst_case"
    )
    assert np.asarray(combined[0]) == pytest.approx(np.asarray(uncombined[0])[:, 0])

    # A window of exactly one period puts the bins at k / (oversampling *
    # T_TR), so every third one falls on a TR harmonic exactly.
    assert np.diff(frequencies) == pytest.approx(1.0 / (3.0 * structure.tr_duration))


def test_choosing_a_window_under_tr_is_refused_out_loud_rather_than_ignored(seq):
    with pytest.warns(UserWarning, match="window_width"):
        seq.calculate_gradient_spectrum(plot=False, tr="worst_case", window_width=0.01)


def test_leaving_the_window_alone_under_tr_says_nothing(seq, recwarn):
    seq.calculate_gradient_spectrum(plot=False, tr="worst_case")
    assert not [w for w in recwarn if "window_width" in str(w.message)]


def test_the_reported_lines_are_harmonics_of_the_repetition_time(seq):
    *_, resonances = seq.calculate_gradient_spectrum(
        plot=False, max_frequency=3000.0, tr="worst_case", resonance_lines=True
    )

    fundamental = 1.0 / resonances.tr_duration
    expected = fundamental * np.arange(1, resonances.line_freqs.size + 1)
    assert resonances.line_freqs == pytest.approx(expected, rel=1e-5)
    assert resonances.line_a_eq.shape == (resonances.line_freqs.size, 3)
    assert resonances.line_freqs.max() <= 3000.0


def test_a_band_over_a_strong_line_is_flagged_and_one_over_nothing_is_not(seq):
    """The verdict follows the bands, and ``ok`` follows the verdict."""
    *_, quiet = seq.calculate_gradient_spectrum(
        plot=False, max_frequency=3000.0, tr="worst_case", resonance_lines=True, bands=[]
    )
    assert quiet.candidate_freqs.size == 0
    assert quiet.ok

    strongest = float(quiet.line_freqs[quiet.line_a_eq.max(axis=-1).argmax()])
    *_, flagged = seq.calculate_gradient_spectrum(
        plot=False,
        max_frequency=3000.0,
        tr="worst_case",
        resonance_lines=True,
        bands=[(strongest - 1.0, strongest + 1.0, 1.0)],
    )
    assert flagged.candidate_freqs.size > 0
    assert not flagged.ok
    assert bool(flagged.violations.any())


@pytest.mark.parametrize(
    "name", ["epi_2d_1sl_1avg", "bssfp_2d_1sl_1avg", "gre_2d_1sl_1avg"]
)
def test_the_drawn_lines_and_the_predownload_gate_reach_the_same_verdict(name):
    """The whole point of ``resonance_lines``, on recorded sequences.

    A plot exists to be trusted, so a band that the overlay draws in red must
    be a band ``pulseg_check_safety`` refuses, and a band the overlay leaves
    alone must be one it passes. Both directions are checked, by placing a
    band on the strongest line the analysis found and then on the weakest.

    The gate is called here directly, not through anything this module owns,
    so agreement is between two independent paths into the C engine rather
    than between a value and a copy of itself.
    """
    from pulserver._ext._pulseg_wrapper import _check_safety

    system = pp.Opts(max_grad=50.0, grad_unit="mT/m", max_slew=350.0, slew_unit="T/m/s")
    recorded = pp.Sequence(system=system)
    recorded.read(EXPECTED / f"{name}.seq")

    def lines_for(bands):
        return recorded.calculate_gradient_spectrum(
            plot=False, max_frequency=3000.0, tr="worst_case", resonance_lines=True, bands=bands
        )[4]

    unbanded = lines_for([])
    assert unbanded.line_freqs.size > 1
    ranked = np.argsort(unbanded.line_a_eq.max(axis=-1))

    for expected_ok, index in ((False, ranked[-1]), (True, ranked[0])):
        band = (float(unbanded.line_freqs[index]) - 5.0, float(unbanded.line_freqs[index]) + 5.0, 0.0)
        overlay = lines_for([band])

        gate_ok = True
        try:
            # PNS skipped: this is the acoustic verdict, and a slew or
            # amplitude violation would short-circuit before reaching it.
            _check_safety(
                recorded._structure_for("test").collection, [band], 0.0, 0.0, 100.0, True
            )
        except Exception:  # the engine's own error type is not public
            gate_ok = False

        assert overlay.ok is gate_ok
        assert overlay.ok is expected_ok


def test_the_line_spectrum_is_not_returned_unless_it_was_asked_for(seq):
    with pytest.raises(ValueError, match="resonance_lines needs tr"):
        seq.calculate_gradient_spectrum(plot=False, resonance_lines=True)


@pytest.mark.parametrize("hardware", [safe_example_hw(), IRNICH], ids=["safe", "irnich"])
def test_both_nerve_models_are_reachable_and_disagree(seq, hardware):
    ok, norm, components, times = seq.calculate_pns(hardware, do_plots=False, tr="worst_case")

    assert isinstance(ok, bool)
    assert components.shape == (norm.size, 3)
    assert times.size == norm.size
    assert norm == pytest.approx(np.sqrt((components**2).sum(axis=1)))
    assert norm.max() > 0.0


def test_the_two_models_are_not_the_same_calculation(seq):
    safe = seq.calculate_pns(safe_example_hw(), do_plots=False, tr="worst_case")[1].max()
    irnich = seq.calculate_pns(IRNICH, do_plots=False, tr="worst_case")[1].max()
    assert safe != pytest.approx(irnich, rel=1e-3)


# %% the SAFE model in C is the one upstream has in Python


def test_the_c_safe_model_matches_upstreams_python_one(seq, system):
    """Cross-check across the language boundary, on identical input.

    ``SafeParams`` in ``cxx/include/pulseg/types.hpp`` is a second
    implementation of ``safe_pns_prediction.safe_pns_model``, and a second
    implementation that is allowed to drift is worse than none: a plot would
    stop meaning what the gate means. So the C response is rebuilt here from
    the same extracted waveform, fed through upstream's Python model, and the
    two are required to agree to float precision.
    """
    from pulserver._ext._pulseg_wrapper import _calc_pns_safe, _get_tr_waveforms

    hardware = safe_example_hw()
    structure = seq._structure_for("test")
    coefficients = _safety.safe_coefficients(hardware)
    theirs = _calc_pns_safe(structure.collection, 0, 0, *coefficients)

    raster = _safety.SAFETY_RASTER_FRACTION * float(system.grad_raster_time)
    waveform = _get_tr_waveforms(structure.collection, 0, 0, 0)
    num_uniform = round(float(waveform["total_duration_us"]) * 1e-6 / raster) + 1
    padding = int(theirs["num_samples"]) - num_uniform + 1
    grid = np.arange(num_uniform) * raster

    assert padding > 0, "the model asked for no warm-up history; nothing would be tested"

    for axis, coefficient_axis in zip("xyz", (hardware.x, hardware.y, hardware.z), strict=True):
        channel = waveform[f"g{axis}"]
        times = np.asarray(channel["time_us"], dtype=float) * 1e-6
        values = np.asarray(channel["amplitude"], dtype=float)
        uniform = np.interp(grid, times, values) if times.size else np.zeros(num_uniform)
        # The core wraps the waveform round rather than zero-padding: a TR
        # that repeats has no silence in front of it.
        padded = np.concatenate([uniform, uniform[np.arange(padding) % num_uniform]])
        slew = np.diff(padded) / raster / float(system.gamma)

        ours = safe_pns_model(slew, raster, coefficient_axis)
        scale = max(np.abs(ours).max(), 1e-12)
        assert np.abs(ours - np.asarray(theirs[f"slew_{axis}"])).max() / scale < 1e-5


def test_safe_hardware_is_recognised_however_it_is_given(tmp_path):
    hardware = safe_example_hw()
    assert _safety.is_safe_hardware(hardware)
    assert _safety.is_safe_hardware(tmp_path / "table.asc")
    assert not _safety.is_safe_hardware(IRNICH)

    coefficients = _safety.safe_coefficients(hardware)
    assert coefficients[0] == pytest.approx(
        (
            hardware.x.a1,
            hardware.x.a2,
            hardware.x.a3,
            hardware.x.tau1,
            hardware.x.tau2,
            hardware.x.tau3,
            hardware.x.stim_limit,
            hardware.x.g_scale,
        )
    )
    assert _safety.irnich_coefficients(IRNICH) == pytest.approx((334.0, 23.4, 1.0))

    with pytest.raises(ValueError, match="missing axis"):
        _safety.safe_hardware(pp.Opts.default)
    with pytest.raises(ValueError, match="chronaxie"):
        _safety.irnich_coefficients({"rheobase": 23.4})


# %% the structure cache


def test_the_structure_is_derived_once_and_reused(seq, num_trs):
    first = seq._structure_for("test")
    assert seq._structure_for("test") is first
    assert first.num_trs == num_trs
    # The TR the C library found is the one the block durations imply.
    assert first.tr_duration == pytest.approx(seq.duration()[0] / num_trs, rel=1e-6)
    assert len(first.segments) > 0


def test_the_structure_does_not_outlive_the_sequence_it_describes(seq, system):
    """A cache that survived a mutation would answer about blocks that are gone."""
    first = seq._structure_for("test")
    before = seq.calculate_pns(safe_example_hw(), do_plots=False, tr="worst_case")[1].max()

    seq.add_block(pp.make_trapezoid(channel="z", area=1500, duration=2e-3, system=system))

    assert seq._structure_for("test") is not first
    after = seq.calculate_pns(safe_example_hw(), do_plots=False, tr="worst_case")[1].max()
    assert after != pytest.approx(before, rel=1e-9)


@pytest.mark.parametrize(
    ("mutate", "arguments"),
    [
        ("add_block", ()),
        ("set_block", (1,)),
        ("remove_duplicates", ()),
        ("set_definition", ("Name", "x")),
        ("register_grad_event", None),
        ("register_adc_event", None),
        ("transform_fov", ((1.0, 0.0, 0.0),)),
    ],
)
def test_every_mutator_bumps_the_revision(seq, system, mutate, arguments):
    if arguments is None:
        arguments = (pp.make_trapezoid(channel="x", area=100, duration=1e-3, system=system),)
        if mutate == "register_adc_event":
            arguments = (pp.make_adc(num_samples=8, duration=1e-3, system=system),)

    before = seq._revision
    getattr(seq, mutate)(*arguments)
    assert seq._revision > before


def test_the_structure_is_handed_the_binary_format(seq, monkeypatch):
    """Written only to be parsed straight back, so it never becomes text.

    The text writer would spend number formatting on one side and ``sscanf``
    on the other, and round every gradient amplitude to the six significant
    digits ``%12g`` keeps -- a needless loss on the way into a safety
    analysis.
    """
    seen = []
    original = pp.Sequence._to_binary

    def record_binary(self):
        seen.append("binary")
        return original(self)

    def record_text(*args, **kwargs):
        seen.append(("text", args[1:], kwargs))
        return b""

    monkeypatch.setattr(pp.Sequence, "_to_binary", record_binary)
    monkeypatch.setattr(pp.Sequence, "_to_text", record_text)

    seq._structure_for("test")
    assert seen == ["binary"]


def test_a_clone_starts_with_no_structure_of_its_own(seq):
    seq._structure_for("test")
    clone = seq._clone()
    assert clone._structure is None
    assert clone._revision == 0


def test_an_empty_sequence_says_so_rather_than_failing_inside_the_c_library(system):
    empty = pp.Sequence(system=system)
    with pytest.raises(ValueError, match="holds no blocks"):
        empty.calculate_pns(safe_example_hw(), tr="worst_case")


# %% arguments that cannot both be meant


@pytest.mark.parametrize("method", ["calculate_pns", "calculate_gradient_spectrum"])
def test_a_time_range_and_a_tr_are_refused_together(seq, method):
    """A repetition time is not a stretch of the timeline; asking for both is a mistake."""
    arguments = (safe_example_hw(),) if method == "calculate_pns" else ()
    with pytest.raises(ValueError, match="one or the other"):
        getattr(seq, method)(*arguments, time_range=[0.0, 0.05], tr="worst_case")


@pytest.mark.parametrize("bad", ["envelope", "canonical", -1, 999])
def test_an_unusable_tr_is_named_in_the_error(seq, bad):
    with pytest.raises(ValueError, match="tr"):
        seq.calculate_pns(safe_example_hw(), do_plots=False, tr=bad)


# %% the plots are upstream's


def test_the_line_overlay_gets_its_own_axis(seq):
    """A_eq and a short-time Fourier magnitude are different quantities.

    Drawing them against one vertical scale would invite reading a crossing
    as meaningful, so the overlay twins the axis.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    *_, resonances = seq.calculate_gradient_spectrum(
        plot=True, max_frequency=3000.0, tr="worst_case", resonance_lines=True
    )
    figure = plt.gcf()
    assert len(figure.axes) == 2, "expected the spectrogram axis plus a twin"
    assert figure.axes[1].get_ylabel() == "$A_{eq}$ (Hz/m)"
    assert resonances.line_freqs.size > 0
    plt.close("all")


def test_asking_for_the_pns_plots_draws_upstreams_pair(seq):
    import matplotlib.pyplot as plt

    plt.close("all")
    seq.calculate_pns(safe_example_hw(), do_plots=True, tr="worst_case")
    assert len(plt.get_fignums()) == 2
    plt.close("all")


@pytest.mark.parametrize(
    ("hardware", "options"),
    [
        (safe_example_hw(), {}),
        (safe_example_hw(), {"tr": "worst_case"}),
        (IRNICH, {"tr": "worst_case"}),
    ],
    ids=["upstream_safe", "c_safe", "irnich"],
)
def test_the_pns_panel_is_marked_at_the_threshold_and_the_margin(seq, hardware, options):
    """Both models, both routes. Upstream marks only the peak it happened to reach.

    A peak alone says how high the sequence got, not how close that is to
    being a problem -- so the level it has to beat, and the margin below it,
    are drawn on every PNS panel this class produces.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    seq.calculate_pns(hardware, do_plots=True, **options)
    axes = plt.gcf().axes[0]

    constant = {
        round(float(np.asarray(line.get_ydata())[0]), 3)
        for line in axes.lines
        if np.ptp(np.asarray(line.get_ydata(), dtype=float)) == 0
    }
    assert {100.0, 80.0} <= constant
    assert {text.get_text().strip() for text in axes.texts} == {
        "100% threshold",
        "80% margin",
    }
    plt.close("all")


def test_a_pns_plot_that_overshoots_is_not_cropped_at_upstreams_ceiling(seq):
    """``safe_plot`` pins the axis at 0-120 %, hiding the one case that matters.

    A sequence that overshoots leaves the panel entirely and its shape cannot
    be read, which is exactly backwards. The axis grows to fit -- and only
    grows, so a passing plot keeps upstream's framing.
    """
    import matplotlib.pyplot as plt

    plt.close("all")
    _, passing, *_ = seq.calculate_pns(safe_example_hw(), do_plots=True, tr="worst_case")
    assert passing.max() * 100 < 120.0
    assert plt.gcf().axes[0].get_ylim()[1] == pytest.approx(120.0)

    plt.close("all")
    ok, failing, *_ = seq.calculate_pns(IRNICH, do_plots=True, tr="worst_case")
    assert not ok
    assert failing.max() * 100 > 120.0
    assert plt.gcf().axes[0].get_ylim()[1] >= failing.max() * 100
    plt.close("all")
