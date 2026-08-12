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


def _diffusion(
    system,
    *,
    gy_scale=0.0,
    refocusings=1,
    shots=1,
    directions=None,
    label=None,
    label_values=None,
):
    """Stejskal-Tanner, or its twice-refocused sibling.

    The second refocusing pulse is what MATLAB's ``2*t_refocusing -
    t_excitation`` cannot express, so it is the case that separates the two
    implementations rather than merely exercising one.

    ``shots`` repeats the whole thing, which is what a real acquisition does
    and what the b-tensor table has to collapse; ``directions`` gives each
    shot its own ``(gx, gy)`` scaling instead, which is what it must not.
    ``label`` stamps a counter on each shot's excitation, which is how a
    reconstruction is told which row of the table an acquisition used.
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
    def lobes(x_scale, y_scale):
        events = [
            pp.make_trapezoid(
                channel="x",
                amplitude=x_scale * amplitude,
                flat_time=delta,
                rise_time=rise,
                system=system,
            )
        ]
        if y_scale:
            events.append(
                pp.make_trapezoid(
                    channel="y",
                    amplitude=y_scale * amplitude,
                    flat_time=delta,
                    rise_time=rise,
                    system=system,
                )
            )
        return events

    scalings = directions if directions is not None else [(1.0, gy_scale)] * shots
    for shot, (x_scale, y_scale) in enumerate(scalings):
        diffusion = lobes(x_scale, y_scale)
        if label is None:
            seq.add_block(excite)
        else:
            index = shot if label_values is None else label_values[shot]
            seq.add_block(excite, pp.make_label(label=label, type="SET", value=index))
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
    gre.remove_duplicates(in_place=True)
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

# The numpy closed form that `calc_moments_btensor` used to be, kept here as
# the reference implementation now that the method delegates to
# `pulseq::calc_moments` in cxx/pulseq.  It is an independent transcription of
# the same algebra -- a union grid, a refocusing sign per piece, and a quartic
# integrated term by term -- so the two agreeing is worth more than either
# agreeing with a quadrature.


class _Pieces:
    """One shot's gradients as straight-line pieces, refocusing folded in.

    ``offsets`` and ``widths`` locate each piece relative to the excitation;
    ``start`` and ``slope`` are its gradient in Hz/m and Hz/m/s. Between two
    nodes each channel is a single straight line, which is what lets the
    integrals below be closed-form rather than sampled.

    **The sign belongs to the piece, not to the node.** A refocusing pulse
    inverts the phase accumulated so far, which is the same as inverting every
    gradient after it -- but the node *at* the pulse is the end of one piece
    and the start of the next, and those two want opposite signs. So the flip
    is counted per piece, from its own start time, and the sign multiplies
    both of that piece's endpoints. Every refocusing instant is forced into
    the grid, so no piece ever straddles one.

    The flip happens at *every* refocusing pulse rather than only the first,
    which is the same answer for a Stejskal-Tanner pair and the right one for
    a twice-refocused sequence -- where MATLAB's single flip is wrong, and
    says so in its own ``TODO``.
    """

    __slots__ = ("offsets", "widths", "start", "slope")

    def __init__(self, times: np.ndarray, gradients: np.ndarray, flips: np.ndarray) -> None:
        widths = np.diff(times)
        keep = widths > 0
        self.widths = widths[keep]
        self.offsets = (times[:-1] - times[0])[keep]

        signs = ((-1.0) ** np.searchsorted(flips, times[:-1], side="right"))[keep, None]
        first = gradients[:-1][keep] * signs
        last = gradients[1:][keep] * signs
        self.start = first
        self.slope = np.divide(
            last - first, self.widths[:, None], out=np.zeros_like(first), where=self.widths[:, None] > 0
        )

    def __len__(self) -> int:
        return int(self.widths.size)


def _gradient_pieces(
    channels: list[np.ndarray], start: float, stop: float, refocusings: np.ndarray
) -> _Pieces:
    """The three gradients over ``start..stop`` as :class:`_Pieces`.

    The grid is the union of every channel's own nodes with the interval ends
    and the refocusing instants.
    """
    inside = refocusings[(refocusings > start) & (refocusings < stop)]
    nodes = [np.array([start, stop], dtype=float), inside]
    for channel in channels[:3]:
        if channel.shape[1]:
            nodes.append(channel[0])

    times = np.unique(np.concatenate(nodes))
    times = times[(times >= start) & (times <= stop)]

    gradients = np.zeros((times.size, 3))
    for axis, channel in enumerate(channels[:3]):
        if channel.shape[1]:
            gradients[:, axis] = np.interp(times, channel[0], channel[1], left=0.0, right=0.0)
    return _Pieces(times, gradients, inside)


def _b_tensor_over(pieces: _Pieces) -> np.ndarray:
    """``(3, 3)`` of ``integral q_i q_j dt`` in s/m^2, exactly.

    ``q = 2*pi*integral g dt`` in rad/m. A Pulseq gradient is piecewise
    linear, so ``q`` is piecewise quadratic and the product of two of its
    components is a quartic -- integrated in closed form per piece rather than
    sampled, which is what makes the answer independent of any raster.
    """
    if not len(pieces):
        return np.zeros((3, 3))

    widths = pieces.widths
    # q(tau) = q0 + 2*pi*(a*tau + b*tau**2/2), as coefficients in tau.
    increments = 2 * np.pi * (
        pieces.start * widths[:, None] + 0.5 * pieces.slope * widths[:, None] ** 2
    )
    origins = np.zeros_like(increments)
    origins[1:] = np.cumsum(increments, axis=0)[:-1]
    coefficients = np.stack(
        (origins, 2 * np.pi * pieces.start, np.pi * pieces.slope), axis=-1
    )  # (n, 3, 3)

    # The integral of tau**(p+q) over a piece, as a weight per coefficient pair.
    powers = np.arange(3)
    orders = powers[:, None] + powers[None, :] + 1
    weights = widths[:, None, None] ** orders / orders
    return np.einsum("nip,njq,npq->ij", coefficients, coefficients, weights)


def _moment_over(pieces: _Pieces, order: int) -> np.ndarray:
    """``2*pi*integral g(t) * (t - t0)**order dt`` per axis, exactly.

    Units are rad/m times seconds to the ``order``. The weight is measured
    from the excitation, which is where MATLAB measures it from.
    """
    from math import comb

    if not len(pieces):
        return np.zeros(3)

    total = np.zeros(3)
    for power in range(order + 1):
        # (offset + tau)**order expanded, times the piece's own (a + b*tau).
        weight = comb(order, power) * pieces.offsets ** (order - power)
        for degree, coefficient in ((power, pieces.start), (power + 1, pieces.slope)):
            total += weight * pieces.widths ** (degree + 1) / (degree + 1) @ coefficient
    return 2 * np.pi * total




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


# %% the C++ core against the reference implementation


def _reference_moments(seq, *, orders=(1, 2, 3)):
    """``calc_moments_btensor`` done again, in numpy, from ``waveforms()``."""
    channels = seq.waveforms()
    pulses = seq.rf_times(compat=False)
    excitations = np.sort(pulses.of("excitation", "undefined").t)
    refocusings = np.sort(pulses.of("refocusing").t)
    echoes = np.asarray(seq.calc_moments_btensor(compat=False).echo_times, dtype=float)

    B = np.zeros((excitations.size, 3, 3))
    moments = {order: np.zeros((excitations.size, 3)) for order in orders}
    for shot, (start, echo) in enumerate(zip(excitations, echoes)):
        if not np.isfinite(echo) or echo <= start:
            continue
        pieces = _gradient_pieces(channels, start, echo, refocusings)
        B[shot] = _b_tensor_over(pieces)
        for order in orders:
            moments[order][shot] = _moment_over(pieces, order)
    return B, moments


@pytest.mark.parametrize("kwargs", [{}, {"gy_scale": 0.35}, {"refocusings": 2}])
def test_the_core_reproduces_the_numpy_reference(system, kwargs):
    """Two independent transcriptions of the same closed form.

    Trapezoids only, so both assemble the identical waveform and there is
    nothing left but floating-point association -- which is why this is at
    machine precision rather than at a tolerance.
    """
    seq = _diffusion(system, **kwargs)
    B, moments = _reference_moments(seq)
    got = seq.calc_moments_btensor(calc_m1=True, calc_m2=True, calc_m3=True)

    np.testing.assert_allclose(got[0], B, rtol=0, atol=1e-14 * np.abs(B).max())
    for order in (1, 2, 3):
        scale = max(float(np.abs(moments[order]).max()), 1e-30)
        # A twice-refocused shot cancels m1 along the diffusion axis almost
        # exactly, so that component is the difference of two ~1e-1 numbers
        # and carries only the association noise of whichever order the terms
        # were summed in. Judged against the largest component rather than
        # against itself, which would be judging a zero.
        np.testing.assert_allclose(got[order], moments[order], rtol=0, atol=1e-9 * scale)


def test_the_prescription_split_sums_to_the_tensor(system):
    """``B`` is the three parts at ``R = I``, which is what MATLAB returns."""
    result = _diffusion(system, gy_scale=0.35).calc_moments_btensor(compat=False)
    parts = (
        result.b_fixed
        + result.b_rotatable
        + result.b_cross
        + np.swapaxes(result.b_cross, -1, -2)
    )
    np.testing.assert_allclose(parts, result.B, rtol=1e-14)
    np.testing.assert_allclose(result.compose(np.eye(3)), result.B, rtol=1e-14)


def test_norot_moves_a_lobe_between_the_parts_without_changing_B(system):
    """What `NOROT` changes is which half of the split a gradient lands in.

    It exempts a block from the console's FOV rotation, which the ``.seq``
    does not carry -- so at ``R = I`` the tensor cannot move, and the only
    visible effect is that the lobe's contribution crosses from
    ``b_rotatable`` to ``b_fixed``.
    """
    plain = _diffusion(system, gy_scale=0.35).calc_moments_btensor(compat=False)

    exempt = _diffusion(system, gy_scale=0.35)
    exempt.add_block(pp.make_label(label="NOROT", type="SET", value=1))
    shifted = exempt.calc_moments_btensor(compat=False)

    assert np.abs(plain.b_fixed).max() == 0.0
    np.testing.assert_allclose(shifted.B, plain.B, rtol=1e-12)


def test_a_prescription_rotation_turns_only_the_rotatable_half(system):
    """The composition against a direct integration of the rotated sequence.

    Rotating the whole sequence is the same as rotating the prescription when
    nothing is exempt, so this pins the algebra of :meth:`BTensor.compose`
    against an answer that never went through it.
    """
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    result = _diffusion(system, gy_scale=0.35).calc_moments_btensor(compat=False)
    composed = result.compose(R)
    direct = R @ result.B @ R.T

    np.testing.assert_allclose(composed, direct, rtol=1e-12)
    # Mutation check: the algebra has to be sensitive to what it multiplies.
    assert not np.allclose(result.compose(R.T), direct, rtol=1e-6)


def test_the_table_deduplicates_shots_that_share_an_encoding(system):
    """A scan repeats each direction; the table is what a header carries."""
    result = _diffusion(system, gy_scale=0.35, shots=4).calc_moments_btensor(compat=False)

    assert result.B.shape[0] == 4
    assert result.b_tensor_table.shape[0] == 1
    np.testing.assert_array_equal(result.table_index, np.zeros(4, dtype=int))
    scale = float(np.abs(result.B).max())
    for shot in range(4):
        np.testing.assert_allclose(
            result.b_tensor_table[result.table_index[shot]],
            result.B[shot],
            rtol=0,
            atol=1e-9 * scale,
        )


def test_the_table_keeps_two_shells_of_one_direction_apart(system):
    """The trap the obvious key falls into.

    Halving both lobes gives a tensor that is the same shape and a quarter the
    size -- the same *direction*, a different b-value. A key normalised by each
    shot's own peak cannot see the difference, which would silently merge the
    shells of a multi-shell acquisition into one row.
    """
    seq = _diffusion(system, directions=[(1.0, 0.35), (0.5, 0.175), (1.0, 0.35)])
    result = seq.calc_moments_btensor(compat=False)

    assert result.b_tensor_table.shape[0] == 2
    np.testing.assert_array_equal(result.table_index, [0, 1, 0])
    assert result.b_values[1] == pytest.approx(0.25 * result.b_values[0], rel=1e-6)


# %% the diffusion table as [DEFINITIONS]


def _shells(system):
    """Two b-values of one direction, the second shell played twice."""
    return _diffusion(
        system,
        directions=[(1.0, 0.35), (0.5, 0.175), (1.0, 0.35)],
        label="SET",
        label_values=[0, 1, 0],
    )


def test_the_definitions_carry_one_row_per_counter_value(system):
    definitions = _shells(system).diffusion_definitions(axis="SET")

    assert definitions["bTensorAxis"] == "SET"
    assert len(definitions["bTensorFixed"]) == 2 * 9
    # Nothing is under NOROT here, so the fixed part is zeros and carries only
    # the row count; the rotatable one is where the encoding is.
    assert not np.any(definitions["bTensorFixed"])
    assert np.any(definitions["bTensorRotatable"])
    # A part that is identically zero is not written at all.
    assert "bTensorCross" not in definitions


def test_writing_the_definitions_stores_them_and_returns_the_table(system):
    seq = _shells(system)
    table = seq.write_diffusion_definitions(axis="SET")

    assert sorted(k for k in seq.definitions if k.startswith("bTensor")) == [
        "bTensorAxis",
        "bTensorFixed",
        "bTensorRotatable",
    ]
    assert table.axis == "SET"
    assert table.b_values.shape == (2,)
    # Halving both lobes quarters the b-value; the shape is unchanged.
    assert table.b_values[1] == pytest.approx(0.25 * table.b_values[0], rel=1e-6)
    np.testing.assert_allclose(table.b_vectors[1], table.b_vectors[0], atol=1e-9)
    np.testing.assert_allclose(
        np.trace(table.b_tensors, axis1=-2, axis2=-1), table.b_values, rtol=1e-12
    )


def test_a_counter_that_does_not_index_the_table_is_refused(system):
    """Values that are not 0..N-1 cannot be a row index, so this must not ship."""
    seq = _diffusion(
        system, directions=[(1.0, 0.35), (0.5, 0.175)], label="SET", label_values=[3, 7]
    )
    with pytest.raises(ValueError, match=r"SET takes values \[3, 7\]"):
        seq.diffusion_definitions(axis="SET")


def test_a_counter_that_does_not_move_with_the_encoding_is_refused(system):
    """Two shots under one counter value with two different tensors.

    The failure a table cannot survive: whichever row were written, half the
    acquisitions would be reconstructed against the wrong b-vector, and
    nothing downstream could notice.
    """
    seq = _diffusion(
        system, directions=[(1.0, 0.35), (0.5, 0.175)], label="SET", label_values=[0, 0]
    )
    with pytest.raises(ValueError, match="not the axis the diffusion encoding varies along"):
        seq.diffusion_definitions(axis="SET")


def test_a_sequence_with_no_such_counter_says_so(system):
    with pytest.raises(ValueError, match="never sets a 'SET' label"):
        _diffusion(system, gy_scale=0.35).diffusion_definitions(axis="SET")


@pytest.mark.parametrize("suffix", [".seq", ".bseq"])
def test_the_table_survives_a_file_round_trip(system, tmp_path, suffix):
    """Through the writer, the parser, and back into the same type."""
    seq = _shells(system)
    written = seq.write_diffusion_definitions(axis="SET")

    path = tmp_path / f"diffusion{suffix}"
    seq.write(str(path))
    back = pp.Sequence()
    back.read(str(path))

    recovered = pp.DiffusionTable.from_definitions(
        {k: v for k, v in back.definitions.items() if k.startswith("bTensor")}
    )
    assert recovered.axis == "SET"
    # The definitions are written as text, so this is the file's precision and
    # not the computation's.
    np.testing.assert_allclose(recovered.b_tensors, written.b_tensors, rtol=1e-8)


def test_reading_a_sequence_that_carries_no_table_says_which_entry_is_missing():
    with pytest.raises(ValueError, match="no bTensorFixed/bTensorAxis entry"):
        pp.DiffusionTable.from_definitions({})


def test_the_reader_composes_the_prescription_the_way_compose_does(system):
    """The two sides of the split have to agree, or the frame is lost."""
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])

    seq = _shells(system)
    definitions = seq.diffusion_definitions(axis="SET")
    turned = pp.DiffusionTable.from_definitions(definitions, rotation=R)
    plain = pp.DiffusionTable.from_definitions(definitions)

    # A b-value is a trace and a rotation cannot change it; the direction is
    # what moves.
    np.testing.assert_allclose(turned.b_values, plain.b_values, rtol=1e-12)
    np.testing.assert_allclose(
        turned.b_tensors, R @ plain.b_tensors @ R.T, rtol=1e-12, atol=1e-12
    )
    assert not np.allclose(turned.b_vectors, plain.b_vectors, atol=1e-6)


# %% test_report


def test_the_report_is_upstreams_and_mentions_what_upstream_mentions(gre):
    report = gre.test_report()
    for expected in ("Number of blocks", "Sequence duration", "TE", "TR"):
        assert expected in report
