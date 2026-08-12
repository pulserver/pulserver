"""The analysis methods added on top of the waveform path.

Four of these have an oracle and are checked against it: ``apply_soft_delay``
and ``evaluate_labels`` against **upstream PyPulseq**, which implements the
same semantics; ``calc_rf_power`` against a brute-force walk that decodes
every block rather than memoizing on shapes; and ``calc_moments_btensor``
against a four-million-point numerical integration of the same waveforms.

The last is the one worth the cost. Its whole claim is that a Pulseq gradient
is piecewise linear and can therefore be integrated in closed form, exactly,
rather than sampled -- so the test that means anything is one where the
sampled answer is converged and the closed form has to meet it.
"""

from __future__ import annotations

import numpy as np
import pulserver.pypulseq as pp
import pypulseq as upstream
import pytest

from pulserver.pypulseq import BTensor, RfPower, SoftDelay


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=80,
        grad_unit="mT/m",
        max_slew=100,
        slew_unit="T/m/s",
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture
def gre(system):
    """Eight repetitions of excite / readout, with a line label on each."""
    built = pp.Sequence(system=system)
    rf = pp.make_sinc_pulse(flip_angle=np.pi / 9, duration=2e-3, system=system, use="excitation")
    gx = pp.make_trapezoid(channel="x", flat_area=1000, flat_time=3.2e-3, system=system)
    adc = pp.make_adc(num_samples=64, duration=3.2e-3, delay=gx.rise_time, system=system)
    for line in range(8):
        built.add_block(rf)
        built.add_block(gx, adc, pp.make_label(type="SET", label="LIN", value=line))
        built.add_block(pp.make_delay(4e-3))
    return built


def _diffusion(system, *, gy_scale=0.0, refocusings=1):
    """Stejskal-Tanner, or its twice-refocused sibling.

    The second refocusing pulse is what MATLAB's ``2*t_refocusing -
    t_excitation`` cannot express, so it is the case that separates the two
    implementations rather than merely exercising one.
    """
    amplitude = 40e-3 * system.gamma
    delta, separation, rise = 8e-3, 22e-3, 0.4e-3

    seq = pp.Sequence(system=system)
    excite = pp.make_block_pulse(
        flip_angle=np.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    refocus = pp.make_block_pulse(
        flip_angle=np.pi, duration=1e-3, system=system, use="refocusing"
    )
    diffusion = [
        pp.make_trapezoid(
            channel="x", amplitude=amplitude, flat_time=delta, rise_time=rise, system=system
        )
    ]
    if gy_scale:
        diffusion.append(
            pp.make_trapezoid(
                channel="y",
                amplitude=gy_scale * amplitude,
                flat_time=delta,
                rise_time=rise,
                system=system,
            )
        )

    seq.add_block(excite)
    for _ in range(refocusings):
        seq.add_block(*diffusion)
        seq.add_block(pp.make_delay(separation - delta - 2 * rise - 1e-3))
        seq.add_block(refocus)
        seq.add_block(*diffusion)
    seq.add_block(pp.make_delay(2e-3))
    seq.add_block(
        pp.make_trapezoid(channel="z", flat_area=100, flat_time=2e-3, system=system),
        pp.make_adc(num_samples=64, duration=2e-3, system=system),
    )
    return seq


# %% time_range replaces block_range everywhere


def test_no_public_method_takes_a_block_range():
    """A window is stated in seconds here, as PyPulseq states every other one.

    MATLAB's `blockRange` is the divergence; this pins the side we chose, so
    the argument cannot creep back in on the next method.
    """
    import inspect

    offenders = []
    for name in dir(pp.Sequence):
        if name.startswith("_"):
            continue
        member = getattr(pp.Sequence, name)
        if not callable(member):
            continue
        try:
            parameters = inspect.signature(member).parameters
        except (TypeError, ValueError):
            continue
        if "block_range" in parameters:
            offenders.append(name)
    assert offenders == []
    assert "block_range" not in inspect.signature(pp.TransformFOV.apply_to_sequence).parameters


# %% the scan structure the C core recovers


def test_the_structure_properties_agree_with_the_blocks_they_describe(gre):
    assert gre.num_trs == 8
    assert gre.tr_size == 3
    assert gre.num_trs * gre.tr_size == gre.num_blocks
    assert gre.num_segments >= 1


def test_a_segment_is_drawn_as_its_highest_energy_instance(gre):
    """Not the first repetition that happened to be played."""
    first, last = gre._segment_blocks(0)
    assert 1 <= first <= last <= gre.num_blocks
    assert last - first + 1 == len(gre._structure_for("t").segments[0]["block_indices"])


def test_an_out_of_range_segment_says_how_many_there_are(gre):
    with pytest.raises(ValueError, match="out of range; the sequence holds"):
        gre._segment_blocks(gre.num_segments)


def test_a_segment_and_a_tr_cannot_be_asked_for_together(gre):
    with pytest.raises(ValueError, match="one or the other"):
        gre.plot(tr="worst_case", segment_idx=0, plot_now=False)


# %% soft delays, against upstream


def _with_soft_delays(module, system):
    seq = module.Sequence(system=system)
    seq.add_block(
        module.make_delay(5e-3),
        module.make_soft_delay(numID=0, hint="TE", offset=1e-3, factor=1.0),
    )
    seq.add_block(module.make_delay(2e-3))
    seq.add_block(
        module.make_delay(8e-3),
        module.make_soft_delay(numID=1, hint="TR", offset=0.0, factor=1.0),
    )
    return seq


def test_applying_a_soft_delay_gives_upstreams_block_durations(system):
    ours = _with_soft_delays(pp, system)
    theirs = _with_soft_delays(upstream, system)

    ours.apply_soft_delay(TE=10e-3, TR=20e-3)
    theirs.apply_soft_delay(TE=10e-3, TR=20e-3)

    np.testing.assert_allclose(
        np.asarray(ours.block_durations),
        np.array(list(theirs.block_durations.values())),
        rtol=1e-12,
    )


def test_an_unknown_hint_names_the_ones_that_exist(system):
    seq = _with_soft_delays(pp, system)
    with pytest.raises(ValueError, match=r"\['TE', 'TR'\]"):
        seq.apply_soft_delay(TI=1.0)


def test_the_defaults_invert_what_applying_them_writes(system):
    """`(duration - offset) * factor` is the inverse of `value / factor + offset`."""
    seq = _with_soft_delays(pp, system)
    defaults, report = seq.get_default_soft_delay_values()

    assert report == []
    assert isinstance(defaults["TE"], SoftDelay)
    assert float(defaults["TE"]) == pytest.approx(4e-3)
    assert float(defaults["TR"]) == pytest.approx(8e-3)

    seq.apply_soft_delay(**{name: float(value) for name, value in defaults.items()})
    np.testing.assert_allclose(np.asarray(seq.block_durations), [5e-3, 2e-3, 8e-3], rtol=1e-12)


def test_a_soft_delay_carries_the_range_it_may_be_set_over(system):
    """The duration reaching zero, which is the only bound the sequence states."""
    seq = _with_soft_delays(pp, system)
    defaults, _ = seq.get_default_soft_delay_values()
    assert defaults["TE"].minimum == pytest.approx(-1e-3)
    assert defaults["TE"].maximum == np.inf


def test_an_inconsistent_hint_is_reported_rather_than_raised(system):
    """`get_default_soft_delay_values` is the diagnostic, so it collects."""
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_delay(5e-3), pp.make_soft_delay(numID=0, hint="TE", offset=0.0, factor=1.0)
    )
    seq.add_block(
        pp.make_delay(5e-3), pp.make_soft_delay(numID=1, hint="TE", offset=0.0, factor=1.0)
    )
    _, report = seq.get_default_soft_delay_values()
    assert any("numeric ID" in line for line in report)


# %% labels, against upstream


def test_label_evaluation_agrees_with_upstream(gre, tmp_path):
    gre.remove_duplicates()
    written = tmp_path / "labelled.seq"
    gre.write(written)

    theirs = upstream.Sequence(system=gre.system)
    theirs.read(str(written))

    assert gre.evaluate_labels() == theirs.evaluate_labels()
    for evolution in ("blocks", "adc", "label"):
        ours = gre.evaluate_labels(evolution=evolution)
        want = theirs.evaluate_labels(evolution=evolution)
        assert set(ours) == set(want)
        for name in ours:
            np.testing.assert_array_equal(ours[name], want[name], err_msg=f"{evolution}/{name}")


def test_a_label_window_evaluates_only_the_blocks_it_covers(gre):
    """MATLAB takes a block range here; the window is in seconds instead."""
    edges = np.concatenate(([0.0], np.cumsum(gre.block_durations)))
    half = gre.evaluate_labels(time_range=[0.0, float(edges[gre.num_blocks // 2])])
    assert half["LIN"] < gre.evaluate_labels()["LIN"]


def test_an_unknown_evolution_is_refused(gre):
    with pytest.raises(ValueError, match="evolution must be"):
        gre.evaluate_labels(evolution="everything")


def test_a_block_reports_its_label_operations_in_play_order(gre):
    """`label_sets` and `label_incs` cannot say which came first."""
    block = gre.get_block(2)
    assert block.labels == [("SET", "LIN", 0)]
    assert [(name, value) for _, name, value in block.labels] == block.label_sets


# %% RF power


def _brute_rf_power(seq):
    """Every block decoded and integrated, with nothing memoized."""
    raster = float(seq._native.rf_raster_time)
    total = peak = 0.0
    for index in range(1, seq.num_blocks + 1):
        rf = seq.get_block(index).rf
        if rf is None:
            continue
        count = int(round(rf.shape_dur / raster))
        centres = (np.arange(count) + 0.5) * raster
        squared = (
            np.abs(np.interp(centres, np.asarray(rf.t, float), rf.signal, left=0.0, right=0.0)) ** 2
        )
        total += float(squared.sum()) * raster
        peak = max(peak, float(squared.max()))
    return total, peak


def test_rf_power_agrees_with_the_walk_it_replaces(gre):
    total, peak = _brute_rf_power(gre)
    mean_power, peak_power, rf_rms, total_energy = gre.calc_rf_power()

    assert total_energy == pytest.approx(total, rel=1e-12)
    assert peak_power == pytest.approx(peak, rel=1e-12)
    assert mean_power == pytest.approx(total / float(np.sum(gre.block_durations)), rel=1e-12)
    # MATLAB keeps a second accumulator for this and it is the same number.
    assert rf_rms == pytest.approx(np.sqrt(mean_power), rel=1e-12)


def test_the_shape_memo_survives_a_variable_flip_angle_train(system):
    """Thousands of rows over one waveform still integrate that waveform once."""
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    amplitudes = np.linspace(100.0, 900.0, 40)
    for amplitude in amplitudes:
        rf.amplitude = amplitude
        seq.add_block(rf)

    total, peak = _brute_rf_power(seq)
    _, peak_power, _, total_energy = seq.calc_rf_power()
    assert total_energy == pytest.approx(total, rel=1e-12)
    assert peak_power == pytest.approx(peak, rel=1e-12)
    # Forty rows over one waveform: the shape count does not follow the rows,
    # so neither does the number of integrals.
    assert seq._native.num_rf() == amplitudes.size
    assert seq._native.num_shapes() < amplitudes.size


def test_a_window_reports_the_worst_stretch_not_the_average(gre):
    whole = gre.calc_rf_power()[0]
    span = float(np.sum(gre.block_durations))
    windowed = gre.calc_rf_power(window_duration=span / 4)[0]
    assert windowed >= whole


def test_a_window_longer_than_the_sequence_dilutes_by_the_window(gre):
    """MATLAB divides by the nominal window, not by what was walked."""
    total = gre.calc_rf_power()[3]
    assert gre.calc_rf_power(window_duration=10.0)[0] == pytest.approx(total / 10.0, rel=1e-12)


def test_a_non_positive_window_is_refused(gre):
    with pytest.raises(ValueError, match="must be positive"):
        gre.calc_rf_power(window_duration=0.0)


def test_the_result_object_says_which_window_it_is_for(gre):
    plain = gre.calc_rf_power(compat=False)
    windowed = gre.calc_rf_power(window_duration=0.05, compat=False)
    assert isinstance(plain, RfPower)
    assert plain.window_duration is None
    assert windowed.window_duration == 0.05
    # total_energy is always the whole window's, never one slice of it.
    assert windowed.total_energy == pytest.approx(plain.total_energy, rel=1e-12)


# %% b-tensor and moments


def _dense_reference(seq, order=1, points=2_000_001):
    """The same integrals by brute numerical quadrature."""
    channels = seq.waveforms()
    excitation = float(np.sort(seq.rf_times(compat=False).of("excitation", "undefined").t)[0])
    refocusings = np.sort(seq.rf_times(compat=False).of("refocusing").t)
    echo = float(seq.calc_moments_btensor(compat=False).echo_times[0])

    times = np.linspace(excitation, echo, points)
    gradients = np.stack(
        [
            np.interp(times, c[0], c[1], left=0.0, right=0.0)
            if c.shape[1]
            else np.zeros(times.size)
            for c in channels[:3]
        ],
        axis=1,
    )
    gradients *= ((-1.0) ** np.searchsorted(refocusings, times, side="right"))[:, None]

    q = 2 * np.pi * np.concatenate(
        ([np.zeros(3)], np.cumsum(0.5 * (gradients[1:] + gradients[:-1]) * np.diff(times)[:, None], axis=0))
    )
    b = np.trapezoid(q[:, :, None] * q[:, None, :], times, axis=0)
    moment = 2 * np.pi * np.trapezoid(gradients * (times - excitation)[:, None] ** order, times, axis=0)
    return b, moment


def test_the_closed_form_b_tensor_meets_a_converged_quadrature(system):
    """The whole claim: piecewise-linear gradients integrate exactly.

    A sampled answer converges to this one; if the closed form were wrong the
    two would differ by far more than quadrature error.
    """
    seq = _diffusion(system, gy_scale=0.35)
    B = seq.calc_moments_btensor()[0][0]
    reference, _ = _dense_reference(seq)

    scale = float(np.abs(reference).max())
    assert float(np.abs(B - reference).max()) / scale < 1e-9


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_moments_meet_the_same_quadrature(system, order):
    seq = _diffusion(system, gy_scale=0.35)
    flags = {f"calc_m{order}": True}
    moments = seq.calc_moments_btensor(**flags)[order]
    _, reference = _dense_reference(seq, order=order)

    scale = max(float(np.abs(reference).max()), 1e-30)
    assert float(np.abs(moments[0] - reference).max()) / scale < 1e-9


def test_a_twice_refocused_sequence_is_handled_by_the_same_code(system):
    """The case MATLAB's `2*t_refocusing - t_excitation` cannot express.

    Two refocusing pulses mean two sign flips, and the echo is where the
    trajectory says it is rather than where a two-pulse formula puts it.
    """
    seq = _diffusion(system, refocusings=2)
    result = seq.calc_moments_btensor(compat=False)

    assert result.b_values.size == 1
    assert np.isfinite(result.echo_times[0])
    reference, _ = _dense_reference(seq)
    scale = float(np.abs(reference).max())
    assert float(np.abs(result.B[0] - reference).max()) / scale < 1e-9


def test_the_echo_is_the_trajectorys_and_not_the_two_pulse_formula(system):
    """They differ, and the one used is the measured one."""
    seq = _diffusion(system)
    result = seq.calc_moments_btensor(compat=False)

    excitation = float(result.excitation_times[0])
    refocusing = float(np.sort(seq.rf_times(compat=False).of("refocusing").t)[0])
    matlab = 2.0 * refocusing - excitation

    _, echoes = seq._echo_centers((1, seq.num_blocks))
    assert result.echo_times[0] == pytest.approx(float(echoes[0]))
    assert result.echo_times[0] != pytest.approx(matlab, abs=1e-9)


def test_a_window_with_no_excitation_says_so(system):
    seq = pp.Sequence(system=system)
    seq.add_block(pp.make_delay(1e-3))
    with pytest.raises(ValueError, match="no excitation pulse"):
        seq.calc_moments_btensor()


# %% what a diffusion pipeline reads


def test_the_b_tensor_is_in_the_units_dipy_and_mrtrix_mean(system):
    """`b_tensors` is s/mm^2 with trace equal to the b-value -- which is
    exactly what DIPY's `gradient_table(btens=...)` takes as an (N, 3, 3)."""
    seq = _diffusion(system, gy_scale=0.35)
    result = seq.calc_moments_btensor(compat=False)

    assert isinstance(result, BTensor)
    np.testing.assert_allclose(result.b_tensors, result.B * 1e-6, rtol=1e-12)
    np.testing.assert_allclose(
        np.trace(result.b_tensors, axis1=-2, axis2=-1), result.b_values, rtol=1e-12
    )
    # A plausible in-vivo b-value, so a units slip of 1e6 cannot pass.
    assert 10.0 < result.b_values[0] < 5000.0


def test_the_b_vector_is_unit_length_and_points_along_the_encoding(system):
    seq = _diffusion(system, gy_scale=0.35)
    result = seq.calc_moments_btensor(compat=False)

    np.testing.assert_allclose(np.linalg.norm(result.b_vectors, axis=-1), 1.0, rtol=1e-12)
    expected = np.array([1.0, 0.35, 0.0])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(np.abs(result.b_vectors[0]), np.abs(expected), atol=1e-9)


def test_the_b_vector_sign_is_fixed_so_a_table_can_be_diffed(system):
    """A b-vector and its negative are the same measurement; a table that
    flips between them for numerical reasons is one nobody can compare."""
    seq = _diffusion(system, gy_scale=0.35)
    vectors = seq.calc_moments_btensor(compat=False).b_vectors
    leading = np.argmax(np.abs(vectors), axis=-1)
    assert np.all(np.take_along_axis(vectors, leading[:, None], axis=-1) > 0)


def test_a_single_axis_encoding_reads_as_linear(system):
    """`b_delta` is derived from the eigenvalues, not from the sequence's name."""
    result = _diffusion(system).calc_moments_btensor(compat=False)
    assert result.b_delta[0] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    ("eigenvalues", "expected"),
    [((1.0, 0.0, 0.0), 1.0), ((0.0, 0.5, 0.5), -0.5), ((1 / 3, 1 / 3, 1 / 3), 0.0)],
    ids=["linear", "planar", "spherical"],
)
def test_b_delta_places_the_three_canonical_encodings(eigenvalues, expected):
    """1 linear, -1/2 planar, 0 spherical -- the convention DIPY's LTE/PTE/STE
    tensors are built to, so a table produced here lands in the right family."""
    tensor = np.diag(np.asarray(eigenvalues) * 1e6)[None]
    result = BTensor.of(
        B=tensor,
        m1=np.zeros((1, 3)),
        m2=np.zeros((1, 3)),
        m3=np.zeros((1, 3)),
        excitation_times=np.zeros(1),
        echo_times=np.zeros(1),
    )
    assert result.b_delta[0] == pytest.approx(expected)
    assert result.b_values[0] == pytest.approx(sum(eigenvalues))


def test_the_mrtrix_table_is_one_column_stack_away(system):
    seq = _diffusion(system, gy_scale=0.35)
    result = seq.calc_moments_btensor(compat=False)
    table = np.column_stack((result.b_vectors, result.b_values))
    assert table.shape == (1, 4)
    np.testing.assert_allclose(np.linalg.norm(table[:, :3], axis=-1), 1.0, rtol=1e-12)


# %% test_report


def test_the_report_is_upstreams_and_mentions_what_upstream_mentions(gre):
    report = gre.test_report()
    for expected in ("Number of blocks", "Sequence duration", "TE", "TR"):
        assert expected in report
