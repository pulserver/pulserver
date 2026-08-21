"""``Sequence.calculate_kspace`` and ``Sequence.auto_label``.

The trajectory is checked against **upstream PyPulseq**, the independent
implementation of the same quantity, and against the sequences' own
prescriptions where PyPulseq has nothing to say -- it computes no echo
position and reads no rotation extension at all.

The file fixtures come from the corpus in ``tests/python/fixtures/`` (see
``fixture_corpus.py``). Upstream can only read the corpus entries that carry
no revision-2 labels, so the upstream comparisons run on the EPI collection
files and the label-free scans ``_cartesian_scan`` builds in memory.

``auto_label`` is asserted on what the sequences state about themselves --
a fully sampled scan visits each line once, an EPI's navigators repeat one
line -- and the arithmetic underneath is covered in tests/cpptests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fixture_corpus import FIXTURES_DIR
from pulserver.pypulseq import Sequence


def load(stem: str) -> Sequence:
    seq = Sequence()
    seq.read(str(FIXTURES_DIR / f"{stem}.seq"))
    return seq


@pytest.fixture
def gre() -> Sequence:
    return load("gre_2d_3sl")


def _cartesian_scan(lines, n_y: int = 16, fov: float = 0.22) -> Sequence:
    """A single-slice gradient echo acquiring exactly ``lines``, and no others.

    Carries ``FOV`` and no labels, so upstream PyPulseq reads its file too.
    """
    import pulserver.pypulseq as pp

    system = pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(10),
        duration=1e-3,
        slice_thickness=5e-3,
        system=system,
        return_gz=True,
        use="excitation",
        delay=system.rf_dead_time,
    )
    gx = pp.make_trapezoid(
        channel="x", flat_area=32 / fov, flat_time=1.6e-3, system=system
    )
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, system=system)
    adc = pp.make_adc(
        num_samples=32, duration=gx.flat_time, delay=gx.rise_time, system=system
    )

    seq = pp.Sequence(system=system)
    seq.set_definition(key="FOV", value=[fov, fov, 5e-3])
    for line in lines:
        gy = pp.make_trapezoid(channel="y", area=(line - n_y // 2) / fov, system=system)
        seq.add_block(rf, gz)
        seq.add_block(gz_reph)
        seq.add_block(gx_pre, gy)
        seq.add_block(gx, adc)
    return seq


# --------------------------------------------------------------------------
# Against upstream
# --------------------------------------------------------------------------

#: Corpus files both readers parse, and the relative agreement to demand.
#: The EPI blips ride the read ramps as shaped gradients, which is the
#: hardest overlap the two implementations share.
UPSTREAM_CASES = [
    ("epi_2d", 1e-12),
    ("epi_2d_main", 1e-12),
    ("epi_3d", 1e-12),
    ("epi_3d_main", 1e-12),
]


@pytest.mark.parametrize(
    ("stem", "tolerance"), UPSTREAM_CASES, ids=[s for s, _ in UPSTREAM_CASES]
)
def test_the_trajectory_agrees_with_upstream_pypulseq(stem, tolerance):
    pp = pytest.importorskip("pypulseq")

    ours = load(stem).calculate_kspace(dense=False)[0]

    theirs_seq = pp.Sequence()
    theirs_seq.read(str(FIXTURES_DIR / f"{stem}.seq"))
    theirs = theirs_seq.calculate_kspace()[0]

    assert ours.shape == theirs.shape
    scale = max(float(np.abs(theirs).max()), 1e-12)
    assert float(np.abs(ours - theirs).max()) / scale < tolerance


def test_it_returns_upstreams_five_tuple(gre):
    """Shapes and orientation, so a PyPulseq script unpacks it unchanged."""
    k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = gre.calculate_kspace()

    assert k_traj_adc.ndim == 2 and k_traj_adc.shape[0] == 3
    assert k_traj.ndim == 2 and k_traj.shape[0] == 3
    assert t_adc.shape == (k_traj_adc.shape[1],)
    assert t_excitation.ndim == 1
    assert t_refocusing.ndim == 1
    # A gradient-echo sequence has excitations and no refocusing pulses.
    assert t_excitation.size > 0
    assert t_refocusing.size == 0
    assert np.all(np.diff(t_adc[: k_traj_adc.shape[1]]) > -1e-12)


# --------------------------------------------------------------------------
# What upstream does not compute
# --------------------------------------------------------------------------


def test_the_echo_of_a_symmetric_readout_is_at_half_n(gre):
    """PyPulseq derives no echo position at all; this is the point of the core.

    A fully sampled readout straddling the origin puts DC at ``N/2`` -- the
    convention it is reconstructed under, and the reason the search resolves a
    tie towards the later sample rather than towards whichever way the file's
    printed digits rounded.
    """
    result = gre._kspace(dense=False)
    samples = result["readout_samples"]
    centers = result["readout_center_sample"]

    assert samples.size > 0
    assert np.array_equal(centers, samples // 2)


def test_the_repeat_memo_fires_on_a_repeating_scan(gre):
    """A scan that repeats and reports one key group per readout has a wrong key.

    Worth asserting rather than merely measuring: every wrong version of the
    key still produced right answers, it just never fired.
    """
    result = gre._kspace(dense=False)
    assert 0 < result["key_groups"] < len(result["readout_block"])


def test_the_dense_trajectory_is_optional(gre):
    """It is the one output that grows with the scan rather than the acquisition."""
    with_dense = gre.calculate_kspace(dense=True)
    without = gre.calculate_kspace(dense=False)

    assert with_dense[1].shape[1] > 0
    assert without[1].shape[1] == 0
    # Same ADC samples either way: the dense grid is an extra, not a mode.
    assert np.array_equal(with_dense[0], without[0])


def test_the_whole_tuple_matches_upstream():
    """Every element, not just the ADC samples -- this is the drop-in claim.

    ``k_traj`` is upstream's own, computed by upstream, because the tuple's
    contract is upstream's and the C core's breakpoint grid is a different
    (better, sparser) representation of the same curve. The other four come
    from the C core and agree to ~1e-13.
    """
    pp = pytest.importorskip("pypulseq")
    scan = load("epi_2d_main")
    theirs_seq = pp.Sequence()
    theirs_seq.read(str(FIXTURES_DIR / "epi_2d_main.seq"))

    for name, ours, theirs in zip(
        ("k_traj_adc", "k_traj", "t_excitation", "t_refocusing", "t_adc"),
        scan.calculate_kspace(),
        theirs_seq.calculate_kspace(),
        strict=False,
    ):
        ours = np.atleast_1d(np.asarray(ours, dtype=float))
        theirs = np.atleast_1d(np.asarray(theirs, dtype=float))
        assert ours.shape == theirs.shape, name
        if not theirs.size:
            continue
        # Upstream marks excitation boundaries in k_traj with NaN, so the
        # gaps have to line up before the numbers between them can be
        # compared at all.
        assert np.array_equal(np.isnan(ours), np.isnan(theirs)), name
        finite = ~np.isnan(theirs)
        if not finite.any():
            continue
        scale = max(float(np.abs(theirs[finite]).max()), 1e-12)
        assert float(np.abs(ours[finite] - theirs[finite]).max()) / scale < 1e-9, name


def test_the_dense_trajectory_of_a_rotated_sequence_is_the_physical_one():
    """Upstream has no vocabulary for a rotation, so the window is replayed
    with it resolved into the gradients -- and what comes back is then the
    same trajectory the C core reports for the ADC samples, in the same
    frame."""
    seq = load("gre_radial_2d")
    dense = np.asarray(seq.calculate_kspace()[1])

    assert dense.size > 0
    # Every spoke of a radial scan is the base spoke turned, so the resolved
    # trajectory reaches both in-plane axes where the stored one reaches one.
    assert np.nanmax(np.abs(dense[1])) > 0.5 * np.nanmax(np.abs(dense[0]))


def test_the_resolved_window_and_the_core_agree_on_a_rotated_trajectory():
    """The differential check the materialisation is worth having: upstream,
    given the replayed window, lands the ADC samples where the C core says
    they are."""
    seq = load("gre_radial_2d")
    ours = np.asarray(seq.calculate_kspace(dense=False)[0])
    theirs = np.asarray(seq._upstream_window(1, seq.num_blocks).calculate_kspace()[0])

    assert ours.shape == theirs.shape
    scale = max(float(np.abs(ours).max()), 1e-12)
    assert float(np.abs(ours - theirs).max()) / scale < 1e-9


def test_the_dense_trajectory_of_a_rotated_sequence_can_be_asked_for_unrotated():
    """``frame="logical"`` is the file's own frame, so there the rotation is
    exactly what should be left out: every spoke is the one that was stored."""
    seq = load("gre_radial_2d")
    dense = np.asarray(seq.calculate_kspace(frame="logical")[1])

    assert dense.size > 0
    assert float(np.nanmax(np.abs(dense[1]))) == pytest.approx(0.0, abs=1e-9)


def test_plotting_the_kspace_of_a_rotated_sequence_needs_no_special_argument():
    """The path a reader reaches for first, on the scans that carry rotations."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    figure = load("gre_radial_2d").plot_kspace(plot_now=False)
    assert figure is not None


def test_the_two_plots_draw_the_same_resolved_sequence():
    """``plot`` and ``plot_kspace`` reach upstream through one materialisation,
    so what one draws the gradients of is what the other draws the trajectory
    of -- not the base waveform beside the turned one."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    seq = load("gre_radial_2d")
    drawn = seq.plot_kspace(plot_now=False).axes[0].lines[0].get_xydata()
    dense = np.asarray(seq.calculate_kspace()[1])

    assert np.array_equal(np.nan_to_num(drawn.T), np.nan_to_num(dense[:2]))
    # And that trajectory is the one the resolved window produces, which is
    # the sequence ``plot`` hands upstream.
    resolved = seq._upstream_window(1, seq.num_blocks).calculate_kspace()[1]
    assert np.array_equal(np.nan_to_num(dense), np.nan_to_num(np.asarray(resolved)))


# -- the sampling order the ordering view is drawn from ------------------


def per_readout(sequence, values):
    """``values`` is per sample; take the one each readout starts with."""
    stride = values.size // sequence._num_adc(None)
    return values[::stride]


def test_an_authored_echo_label_is_the_echo_index():
    """A train that writes ``ECO`` has already said which echo each line is,
    and the shot is where that count returns to zero."""
    seq = load("fse_3d")
    shot, echo = seq._sampling_order(None)

    assert np.array_equal(per_readout(seq, echo), np.tile(np.arange(8), 4))
    assert np.array_equal(per_readout(seq, shot), np.repeat(np.arange(4), 8))


def test_a_train_under_one_excitation_is_one_shot_of_many_echoes():
    """No ``ECO`` to read, so the index is the readout's rank in its train --
    which is what makes a whole EPI plane one shot rather than sixteen."""
    seq = load("epi_2d")
    shot, echo = seq._sampling_order(None)
    echo = per_readout(seq, echo)

    assert per_readout(seq, shot).max() + 1 == 2
    # The navigator triplet opens the shot; the imaging train follows it.
    assert np.array_equal(echo[:3], [0, 1, 2])
    assert echo.max() == 15


def test_one_readout_per_excitation_is_one_shot_each():
    seq = load("gre_2d")
    shot, echo = seq._sampling_order(None)

    assert np.array_equal(per_readout(seq, shot), np.arange(seq._num_adc(None)))
    assert not np.any(echo)


def test_the_sampling_order_has_one_value_per_sample():
    """It colours ``k_traj_adc``, so it has to be the same length as it."""
    seq = load("fse_3d")
    shot, echo = seq._sampling_order(None)
    samples = np.asarray(seq.calculate_kspace(dense=False, compat=False).k_traj_adc)

    assert shot.shape == echo.shape == (samples.shape[1],)


def kspace_panels(figure):
    """The panels drawing k-space, which is every axis but the colourbars."""
    return [axis for axis in figure.axes if axis.get_xlabel().startswith("$k_")]


def test_the_ordering_view_draws_both_hierarchies():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    figure = load("fse_3d").plot_kspace(
        plane="yz", color_by="order", show_trajectory=False, plot_now=False
    )
    assert len(kspace_panels(figure)) == 2


def test_the_ordering_view_drops_a_panel_whose_index_never_varies():
    """One line per excitation has no echo axis, so there is no echo panel."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    figure = load("gre_2d").plot_kspace(color_by="order", plot_now=False)
    assert len(kspace_panels(figure)) == 1


def test_colouring_draws_the_same_samples_as_not_colouring():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    plain = load("gre_radial_2d").plot_kspace(plot_now=False)
    coloured = load("gre_radial_2d").plot_kspace(color_by="shot", plot_now=False)

    assert np.array_equal(
        kspace_panels(plain)[0].collections[0].get_offsets(),
        kspace_panels(coloured)[0].collections[0].get_offsets(),
    )


def test_an_unknown_colour_index_is_refused():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    with pytest.raises(ValueError, match="color_by"):
        load("gre_2d").plot_kspace(color_by="slice", plot_now=False)


def test_the_logical_frame_leaves_rotations_out():
    """Two frames, and the difference is a rotation extension.

    ``TransformFOV`` works in the logical frame because ``dr . k`` is invariant
    when both are rotated; a reconstruction wants the physical one.
    """
    stem = "gre_radial_2d"
    physical = load(stem).calculate_kspace(dense=False)[0]
    logical = load(stem).calculate_kspace(dense=False, frame="logical")[0]

    assert physical.shape == logical.shape
    assert not np.allclose(physical, logical)


def test_an_unknown_frame_is_refused(gre):
    with pytest.raises(ValueError, match="physical"):
        gre.calculate_kspace(frame="scanner")


@pytest.mark.parametrize("raster", [4e-6, 10e-6, 20e-6, 25e-6])
def test_the_times_do_not_depend_on_which_raster_the_sequence_was_built_on(raster):
    """A sequence never written to disk still knows its own raster.

    The trajectory core is reached by serialising and reparsing, and block
    durations cross as raster ticks, so a reader that cannot see the raster
    recovers seconds scaled by the ratio of the real raster to its default.
    """
    import pulserver.pypulseq as pp

    system = pp.Opts(
        grad_raster_time=raster, block_duration_raster=raster, rf_dead_time=100e-6
    )
    seq = pp.Sequence(system=system)
    excite = pp.make_block_pulse(
        flip_angle=np.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    spacing = 40 * raster
    for _ in range(3):
        seq.add_block(excite)
        seq.add_block(pp.make_delay(spacing))
        seq.add_block(
            pp.make_trapezoid(
                channel="x", flat_area=100, flat_time=2e-3, system=system
            ),
            pp.make_adc(num_samples=32, duration=2e-3, system=system),
        )

    per_shot = float(np.sum(seq.block_durations[:3]))
    times = np.asarray(seq.calculate_kspace(dense=False)[2])

    assert len(times) == 3
    assert np.allclose(np.diff(times), per_shot, rtol=0, atol=1e-12)


def test_window_averaging_is_off_by_default_and_only_moves_a_curved_readout():
    """Off by default so the answer stays comparable with PyPulseq and mrpro."""
    flat = load("fse_2d")
    assert np.allclose(
        flat.calculate_kspace(dense=False)[0],
        flat.calculate_kspace(dense=False, sample_window_average=True)[0],
    )

    curved = load("gre_spiral_2d")
    midpoint = curved.calculate_kspace(dense=False)[0]
    averaged = curved.calculate_kspace(dense=False, sample_window_average=True)[0]
    assert float(np.abs(midpoint - averaged).max()) > 1e-3


def test_a_time_range_bounds_the_answer(gre):
    whole = gre.calculate_kspace(dense=False)[0]
    half = float(np.sum(gre.block_durations)) / 2.0
    part = gre.calculate_kspace(dense=False, time_range=[0.0, half])[0]
    assert part.shape[1] < whole.shape[1]


@pytest.mark.parametrize("time_range", [[1.0, 0.0], [0.0, 1.0, 2.0]])
def test_a_bad_time_range_is_refused(gre, time_range):
    with pytest.raises(ValueError):
        gre.calculate_kspace(time_range=time_range)


@pytest.mark.parametrize("offsets", [[1e-6, 2e-6], [0.0] * 4])
def test_a_per_axis_argument_takes_one_value_or_three(gre, offsets):
    with pytest.raises(ValueError, match="one value or three"):
        gre.calculate_kspace(trajectory_delay=offsets)


def test_a_scalar_delay_means_the_same_on_every_axis(gre):
    assert np.array_equal(
        gre.calculate_kspace(dense=False, trajectory_delay=1e-6)[0],
        gre.calculate_kspace(dense=False, trajectory_delay=[1e-6] * 3)[0],
    )


# --------------------------------------------------------------------------
# auto_label
# --------------------------------------------------------------------------


def test_it_recovers_the_encoding_counters(gre):
    """Three slices inside eight lines, acquired slice-inner."""
    labels, aux = gre.auto_label(skip_apply=True, boundary_flags=False)

    assert set(labels) == {"SLC", "LIN"}
    assert np.array_equal(labels["SLC"], np.tile([0, 1, 2], 8))
    assert np.array_equal(labels["LIN"], np.repeat(np.arange(8), 3))

    assert aux["kSpaceCenterLine"] == 4
    assert aux["kSpaceCenterSample"] == 32
    assert np.allclose(aux["SlicePositions"], [-5e-3, 0.0, 5e-3])


def test_the_boundaries_come_with_the_counters(gre):
    """A slice ends where its last line is, wherever the loop put it.

    Slices are the inner loop here, so every ``(LIN, SLC)`` pair occurs once
    and the three slices each finish in the last three acquisitions -- which is
    what a boundary read off the loop nesting would get wrong.
    """
    labels, _ = gre.auto_label(skip_apply=True)

    assert np.array_equal(np.flatnonzero(labels["LASTSLC"]), [21, 22, 23])
    assert np.array_equal(np.flatnonzero(labels["FIRSTSLC"]), [0, 1, 2])
    # A line is one acquisition per slice, so it both starts and ends there.
    assert np.array_equal(labels["FIRSTLIN"], np.ones(24, dtype=int))
    assert np.array_equal(np.flatnonzero(labels["LASTSCAN"]), [23])


def test_the_counters_are_grid_positions_when_the_file_says_what_the_grid_is(gre):
    """``FOV`` and ``Matrix`` together say what one step of the counter is.

    The fixture is fully sampled, so both routes agree on it. What changes is
    where zero is: without ``Matrix`` the lowest line acquired becomes line 0,
    and with it the counter is a position on the prescribed matrix.
    """
    gre.set_definition(key="Matrix", value=[64, 16, 3])

    labels, aux = gre.auto_label(skip_apply=True, boundary_flags=False)

    assert np.array_equal(labels["LIN"], np.repeat(np.arange(8), 3) + 4)
    assert aux["kSpaceCenterLine"] == 8


@pytest.mark.parametrize(
    ("acceleration", "first_line"),
    [(2, 0), (4, 0), (1, 6)],
    ids=["R=2", "R=4", "partial-fourier"],
)
def test_a_scan_that_skips_lines_is_still_placed_on_the_grid(acceleration, first_line):
    """The case the prescription is needed for.

    An accelerated scan has no adjacent pair to read the step off, so the step
    inferred from the data is the accelerated one and the counter steps by one
    where it should step by ``acceleration``. A partial-Fourier scan never
    reaches the low edge, so the lowest line it did acquire becomes line 0.
    """
    lines = list(range(first_line, 16, acceleration))
    scan = _cartesian_scan(lines)

    inferred, _ = scan.auto_label(skip_apply=True, boundary_flags=False)
    assert np.array_equal(inferred["LIN"], np.arange(len(lines)))

    scan.set_definition(key="Matrix", value=[32, 16, 1])
    placed, aux = scan.auto_label(skip_apply=True, boundary_flags=False)

    assert np.array_equal(placed["LIN"], lines)
    assert aux["kSpaceCenterLine"] == 8


def test_a_matrix_the_readouts_do_not_land_on_is_not_believed(gre):
    """The guard: a prescription that disagrees with the trajectory loses.

    A grid three times too coarse puts every readout between its points, which
    is what a non-Cartesian sequence carrying a ``Matrix`` would look like.
    """
    gre.set_definition(key="FOV", value=[0.22, 0.22 / 3.0, 0.015])
    gre.set_definition(key="Matrix", value=[64, 16, 3])

    labels, _ = gre.auto_label(skip_apply=True, boundary_flags=False)

    assert np.array_equal(labels["LIN"], np.repeat(np.arange(8), 3))


def test_the_grid_is_only_read_when_both_definitions_are_there():
    """With ``FOV`` alone the lowest line acquired becomes line 0."""
    scan = _cartesian_scan(range(4, 12))
    assert scan.get_definition("FOV") is not None
    assert scan.get_definition("Matrix") is None

    labels, aux = scan.auto_label(skip_apply=True, boundary_flags=False)

    assert np.array_equal(labels["LIN"], np.arange(8))
    assert aux["kSpaceCenterLine"] == 4


def test_the_slice_thickness_matches_the_prescription(gre):
    """Recovered from the RF spectrum over the slice-select amplitude.

    ``FOV`` and the ``Matrix`` slice count state the answer independently --
    they come from the prescription, not from the pulse -- and the sequence
    records no thickness of its own. The recovery is only as sharp as the
    pulse's spectrum, which bounds the tolerance.
    """
    _, aux = gre.auto_label(skip_apply=True)
    prescribed = gre.definitions["FOV"][2] / gre.definitions["Matrix"][2]
    assert aux["SliceThickness"] == pytest.approx(prescribed, rel=0.05)
    assert aux["SliceGap"] == pytest.approx(0.0, abs=0.05 * prescribed)


def test_the_navigators_of_an_epi_repeat_one_line():
    labels, aux = load("epi_2d").auto_label(skip_apply=True)

    assert set(labels) >= {"LIN", "REV", "REP"}
    assert labels["LIN"][0] == labels["LIN"][1] == labels["LIN"][2]
    assert list(labels["REP"][:3]) == [0, 1, 2]
    assert aux["kSpaceCenterLine"] == 7


def test_the_epi_echo_index_is_quoted_after_mirroring():
    """One number for a scan whose two polarities disagree about it by one.

    The forward readouts put the echo at 16 of 32 and the reverse ones at 15,
    because they reach the same point in k from opposite ends. A recon mirrors
    the reverse lines -- that is what ``REV`` is for -- and then both are at
    16, so 16 is the number that is true of the reconstructed data.
    """
    seq = load("epi_2d")
    _, aux = seq.auto_label(skip_apply=True)
    assert aux["kSpaceCenterSample"] == 16

    labels, _ = seq.auto_label(skip_apply=True)
    result = seq._kspace(dense=False)
    echo = np.asarray(result["readout_center_sample"])
    samples = np.asarray(result["readout_samples"])
    rev = np.asarray(labels["REV"], dtype=bool)

    assert set(echo.tolist()) == {15, 16}, "the fixture is not bipolar"
    assert np.array_equal(
        np.where(rev, samples - 1 - echo, echo), np.full(echo.shape, 16)
    )


def test_a_single_named_dimension_takes_the_repeat_count():
    """One name is always safe: there is nothing to work out."""
    seq = load("epi_2d")
    plain, _ = seq.auto_label(skip_apply=True)

    named, _ = seq.auto_label(repeat_dims=["SET"], skip_apply=True)
    assert "REP" not in named
    assert np.array_equal(named["SET"], plain["REP"])


def test_ragged_repeats_are_refused_rather_than_split_anyway():
    """The EPI's navigators revisit one line; the other lines are acquired once.

    That is not a dimension, and splitting it into two would put different
    acquisitions in one slot with every surrounding label looking ordinary.
    """
    seq = load("epi_2d")
    with pytest.raises(RuntimeError, match="not a rectangle"):
        seq.auto_label(repeat_dims=["REP", "ECO"], skip_apply=True)

    # Declared outright it is arithmetic again, and goes through.
    split, _ = seq.auto_label(repeat_dims=[("REP", 2), ("ECO", 2)], skip_apply=True)
    assert list(split["REP"][:3]) == [0, 0, 1]
    assert list(split["ECO"][:3]) == [0, 1, 0]


def test_repeat_dimensions_that_contradict_the_scan_are_refused():
    seq = load("epi_2d")
    with pytest.raises(RuntimeError, match="were found"):
        seq.auto_label(repeat_dims=[("SET", 3)], skip_apply=True)
    with pytest.raises(RuntimeError, match="derived from the trajectory"):
        seq.auto_label(repeat_dims=["LIN"], skip_apply=True)
    with pytest.raises(ValueError, match="names, or"):
        seq.auto_label(repeat_dims=[("SET", 2, 1)], skip_apply=True)


def _stamp_label(seq, name, value_of):
    """Attach a LABELSET to every ADC block, the way a design loop would."""
    native = seq._native
    labelset = native.extension_type_id("LABELSET")
    label = native.label_id(name)
    count = 0
    for block in range(1, native.num_blocks() + 1):
        rf, gx, gy, gz, adc, ext, duration = native.get_block(block)
        if adc == 0:
            continue
        row = native.register_label_set(int(value_of(count)), label)
        native.set_block(
            block,
            rf,
            gx,
            gy,
            gz,
            adc,
            native.chain_extension(labelset, row, ext),
            duration,
        )
        count += 1
    seq._touch()
    return count


def _replay_labels(seq):
    """What a reader sees at each ADC: the running LABELSET state."""
    native = seq._native
    labelset = native.find_extension_type_id("LABELSET")
    state = {
        native.label_name(int(native.label_set_row(i)[1])): 0
        for i in range(1, native.num_label_set() + 1)
    }
    out = {}
    for block in range(1, native.num_blocks() + 1):
        _, _, _, _, adc, ext, _ = native.get_block(block)
        link = ext
        while link:
            row = native.extension_row(link)
            if labelset and int(row[0]) == labelset and int(row[1]):
                entry = native.label_set_row(int(row[1]))
                state[native.label_name(int(entry[1]))] = int(entry[0])
            link = int(row[2])
        if adc:
            for name, value in state.items():
                out.setdefault(name, []).append(value)
    return {name: np.asarray(values) for name, values in out.items()}


def test_labels_the_sequence_already_carries_survive_an_auto_label_pass():
    """Stamp the axes only the design loop knows, then fill in the rest.

    A label ``auto_label`` never derives comes through untouched with nothing
    asked for. ``REP`` is the exception -- it *is* derived by default -- so a
    loop that separated its own repeats has to hand it back with ``skip``.
    """
    seq = load("gre_2d_3sl")
    n = _stamp_label(seq, "ECO", lambda i: i % 2)
    seq.auto_label()

    replayed = _replay_labels(seq)
    assert np.array_equal(replayed["ECO"], np.array([i % 2 for i in range(n)]))
    assert "SLC" in replayed and "LIN" in replayed


def test_skip_hands_a_derived_counter_back_to_the_sequence():
    seq = load("epi_2d")
    n = _stamp_label(seq, "REP", lambda i: i // 8)
    want = np.array([i // 8 for i in range(n)])

    derived, _ = seq.auto_label(skip=["REP"])

    assert "REP" not in derived
    assert "LIN" in derived and "REV" in derived
    assert np.array_equal(_replay_labels(seq)["REP"], want)


def test_skipping_something_not_derived_is_refused(gre):
    """It reads as protection, and it is not -- so it must not look accepted."""
    with pytest.raises(RuntimeError, match="not a counter this derives"):
        gre.auto_label(skip=["ECO"], skip_apply=True)
    with pytest.raises(RuntimeError, match="either yours or mine"):
        gre.auto_label(skip=["REP"], repeat_dims=["REP"], skip_apply=True)


def test_a_non_cartesian_sequence_is_refused():
    """These are Cartesian counters; there is no honest value otherwise."""
    seq = load("gre_radial_2d")
    with pytest.raises(RuntimeError, match="do not share a direction"):
        seq.auto_label(skip_apply=True)


def test_skip_apply_leaves_the_sequence_alone(gre):
    before = gre.definitions.copy()
    gre.auto_label(skip_apply=True)
    assert gre.definitions == before


def test_applying_writes_the_labels_and_the_definitions(tmp_path):
    scan = _cartesian_scan(range(16))
    labels_before = scan._native.num_label_set()

    _labels, aux = scan.auto_label()

    assert scan._native.num_label_set() > labels_before
    assert scan.definitions["kSpaceCenterLine"] == aux["kSpaceCenterLine"]
    assert scan.definitions["kSpaceCenterSample"] == aux["kSpaceCenterSample"]
    # And what was written survives a round trip through the file.
    path = tmp_path / "labelled.seq"
    scan.write(str(path))
    text = path.read_text()
    assert "LABELSET" in text
    assert "kSpaceCenterLine" in text
    assert "SliceThickness" in text


@pytest.mark.parametrize("reorder", [[0, 0], [1, 2, 3], [0, 1]])
def test_a_reorder_must_be_a_permutation(gre, reorder):
    if reorder == [0, 1]:
        pytest.skip("[0, 1] is the identity and is allowed")
    with pytest.raises(ValueError, match="permute"):
        gre.auto_label(reorder=reorder, skip_apply=True)


def test_reflecting_an_axis_flips_the_counter_it_carries(gre):
    """The line index runs the other way; the set of lines does not change."""
    plain, _ = gre.auto_label(skip_apply=True)
    flipped, _ = gre.auto_label(reflect=[1], skip_apply=True)

    assert sorted(plain["LIN"]) == sorted(flipped["LIN"])
    assert np.array_equal(flipped["LIN"], plain["LIN"].max() - plain["LIN"])


# --------------------------------------------------------------------------
# The autoLabel surface
# --------------------------------------------------------------------------
#
# `auto_label` is meant to be a drop-in for MATLAB Pulseq's `autoLabel`: every
# one of its parameters accepted, under the Python spelling of its name, with
# Pulserver's own additions after them rather than mixed in. Two defaults
# differ deliberately and both are asserted here, so a change to either is a
# decision rather than a drift.

#: `autoLabel`'s parameter list, and what each is called here.
AUTOLABEL_PARAMETERS = {
    "blockRange": "time_range",
    "useLabels": "use_labels",
    "useAux": "use_aux",
    "skipApply": "skip_apply",
    "mirrorFourier": "mirror_fourier",
    "reflect": "reflect",
    "reorder": "reorder",
    "sortSlices": "sort_slices",
    "noPlots": "no_plots",
}


def test_every_autolabel_parameter_is_accepted():
    import inspect

    ours = inspect.signature(Sequence.auto_label).parameters
    missing = [m for m, p in AUTOLABEL_PARAMETERS.items() if p not in ours]
    assert not missing, f"autoLabel parameters with no equivalent here: {missing}"

    # MATLAB's come first, in MATLAB's order; ours are appended.
    names = [n for n in ours if n != "self"]
    assert names[: len(AUTOLABEL_PARAMETERS)] == list(AUTOLABEL_PARAMETERS.values())
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in ours.values()
        if p.name != "self"
    )


def test_the_two_defaults_that_differ_from_matlab():
    import inspect

    defaults = inspect.signature(Sequence.auto_label).parameters
    # MATLAB defaults to 'acquisition'; a geometric index is what makes
    # SlicePositions[SLC] usable as a stack.
    assert defaults["sort_slices"].default == "ascending"
    # MATLAB defaults to False and draws figures; nothing here draws any.
    assert defaults["no_plots"].default is True


def test_asking_for_plots_is_refused_rather_than_ignored(gre):
    with pytest.raises(ValueError, match="no_plots"):
        gre.auto_label(no_plots=False)


@pytest.mark.parametrize("mode", ["ascending", "descending", "acquisition"])
def test_slice_positions_index_by_slc_under_every_sorting(mode):
    """The invariant that makes the numbering a free choice."""
    labels, aux = load("gre_2d_3sl").auto_label(skip_apply=True, sort_slices=mode)
    positions = np.asarray(aux["SlicePositions"])
    assert positions.size == 3
    assert set(labels["SLC"]) == {0, 1, 2}
    # Whatever the order, the table and the counters name the same three slices.
    assert np.allclose(np.sort(positions), sorted([-5e-3, 0.0, 5e-3]))


def test_descending_is_the_reverse_of_ascending():
    up, aux_up = load("gre_2d_3sl").auto_label(skip_apply=True, sort_slices="ascending")
    down, aux_down = load("gre_2d_3sl").auto_label(
        skip_apply=True, sort_slices="descending"
    )

    assert np.allclose(
        np.asarray(aux_down["SlicePositions"]),
        np.asarray(aux_up["SlicePositions"])[::-1],
    )
    assert np.array_equal(down["SLC"], 2 - np.asarray(up["SLC"]))
    # The gap is geometry: it cannot depend on which end the counting starts.
    assert aux_up["SliceGap"] == aux_down["SliceGap"]


def test_an_unknown_sorting_is_refused(gre):
    with pytest.raises(ValueError, match="sort_slices"):
        gre.auto_label(sort_slices="interleaved")


def test_mirror_fourier_turns_the_encoding_over_and_leaves_the_slices():
    plain, _ = load("gre_2d_3sl").auto_label(skip_apply=True)
    mirrored, _ = load("gre_2d_3sl").auto_label(skip_apply=True, mirror_fourier=True)

    # Line order reverses ...
    assert np.array_equal(
        mirrored["LIN"], plain["LIN"].max() - np.asarray(plain["LIN"])
    )
    # ... and the slice stack does not, which is what separates this from
    # reflect=[0, 1, 2].
    assert np.array_equal(mirrored["SLC"], plain["SLC"])


def test_use_labels_applies_what_detection_would_have():
    """Detect once, apply anywhere -- the same counters and definitions.

    Compared as labels rather than as bytes: where a sequence already sets a
    counter itself, applying a detection leaves that one alone and detecting
    rewrites it, so the two files can number their extension chains
    differently while saying exactly the same thing.
    """
    labels, aux = load("gre_2d_3sl").auto_label(skip_apply=True)

    reused = load("gre_2d_3sl")
    reused.auto_label(use_labels=labels, use_aux=aux)

    direct = load("gre_2d_3sl")
    direct.auto_label()

    from_reused = reused.evaluate_labels(evolution="adc")
    from_direct = direct.evaluate_labels(evolution="adc")
    assert set(from_reused) == set(from_direct)
    for name, values in from_direct.items():
        assert np.array_equal(from_reused[name], values), name
    for name in aux:
        assert reused.get_definition(name) == direct.get_definition(name), name


def test_use_labels_carries_a_hand_corrected_counter_through():
    labels, aux = load("gre_2d_3sl").auto_label(skip_apply=True)
    edited = dict(labels)
    edited["LIN"] = np.asarray(labels["LIN"])[::-1].copy()

    seq = load("gre_2d_3sl")
    got, _ = seq.auto_label(use_labels=edited, use_aux=aux)
    assert np.array_equal(got["LIN"], edited["LIN"])
    assert "SlicePositions" in seq.definitions


def test_use_labels_with_a_detection_only_option_is_refused(gre):
    labels, _ = gre.auto_label(skip_apply=True)
    for option in ({"reflect": [0]}, {"reorder": [1, 0]}, {"mirror_fourier": True}):
        with pytest.raises(ValueError, match="only affect detection"):
            load("gre_2d_3sl").auto_label(use_labels=labels, **option)


def test_use_labels_of_the_wrong_length_is_refused(gre):
    with pytest.raises(ValueError, match="values for"):
        gre.auto_label(use_labels={"LIN": [1, 2, 3]})
