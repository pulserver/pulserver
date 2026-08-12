"""``Sequence.calculate_kspace`` and ``Sequence.auto_label``.

The trajectory is checked against **upstream PyPulseq**, which is the only
independent implementation of the same quantity available here, and against the
sequences' own prescriptions where PyPulseq has nothing to say -- it computes
no echo position and reads no rotation extension at all.

``auto_label`` has no comparable oracle in Python: Pulseq's ``autoLabel`` is
MATLAB-only, and running it is a one-off validation rather than a dependency.
So the assertions below are on what the sequences state about themselves --
a fully sampled scan visits each line once, an EPI's navigators repeat one
line -- and the arithmetic underneath is covered in tests/cpptests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pulserver.pypulseq import Sequence

FIXTURES = Path(__file__).resolve().parents[1] / "utils" / "expected"


def load(stem: str) -> Sequence:
    seq = Sequence()
    seq.read(str(FIXTURES / f"{stem}.seq"))
    return seq


@pytest.fixture
def gre() -> Sequence:
    return load("gre_2d_3sl_3avg")


# --------------------------------------------------------------------------
# Against upstream
# --------------------------------------------------------------------------

#: Sequences whose trajectory upstream computes the same way we do, and the
#: relative agreement to demand of each.
#:
#: ``fse_2d_1sl_1avg`` is absent on purpose. Its readout is an arbitrary
#: gradient with an explicit time shape, and there PyPulseq's own gradient
#: spline disagrees with its own waveform: it ramps 194552.5924 to 194553.0
#: across a gradient ``gx.waveform`` reports flat at 194553, so its k step
#: comes out 3.891051895538112 where the file says 194553 * 20 us =
#: 3.8910600000000004. We match the file. Asserting against PyPulseq there
#: would be asserting its error, so that case is covered in
#: tests/ctests/test_pulseq_ktraj.c against the sequence's own numbers instead.
UPSTREAM_CASES = [
    ("gre_2d_1sl_1avg", 1e-12),
    ("gre_2d_3sl_3avg", 1e-12),
    ("bssfp_2d_1sl_1avg", 1e-12),
    ("mprage_2d_3sl_3avg", 1e-12),
    ("epi_2d_1sl_1avg", 1e-11),
    # Blipped phase encoding, whose blips are arbitrary gradients: the same
    # spline effect as the FSE case, three orders of magnitude smaller.
    ("gre_32x32_pe_blip", 1e-8),
]


@pytest.mark.parametrize(("stem", "tolerance"), UPSTREAM_CASES, ids=[s for s, _ in UPSTREAM_CASES])
def test_the_trajectory_agrees_with_upstream_pypulseq(stem, tolerance):
    pp = pytest.importorskip("pypulseq")

    ours = load(stem).calculate_kspace(dense=False)[0]

    theirs_seq = pp.Sequence()
    theirs_seq.read(str(FIXTURES / f"{stem}.seq"))
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


def test_the_whole_tuple_matches_upstream(gre):
    """Every element, not just the ADC samples -- this is the drop-in claim.

    ``k_traj`` is upstream's own, computed by upstream, because the tuple's
    contract is upstream's and the C core's breakpoint grid is a different
    (better, sparser) representation of the same curve. The other four come
    from the C core and agree to ~1e-13.
    """
    pp = pytest.importorskip("pypulseq")
    theirs_seq = pp.Sequence()
    theirs_seq.read(str(FIXTURES / "gre_2d_3sl_3avg.seq"))

    for name, ours, theirs in zip(
        ("k_traj_adc", "k_traj", "t_excitation", "t_refocusing", "t_adc"),
        gre.calculate_kspace(),
        theirs_seq.calculate_kspace(),
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


def test_the_dense_trajectory_is_refused_where_upstream_cannot_read_the_sequence():
    """A rotation extension would come back in the wrong frame, silently."""
    seq = load("mprage_noncart_3d_3sl_3avg_userotext1")
    with pytest.raises(NotImplementedError, match="rotation or RF-shim"):
        seq.calculate_kspace()
    # The ADC samples are still available, and they do resolve the rotation.
    assert seq.calculate_kspace(dense=False)[0].shape[1] > 0


def test_the_logical_frame_leaves_rotations_out():
    """Two frames, and the difference is a rotation extension.

    ``TransformFOV`` works in the logical frame because ``dr . k`` is invariant
    when both are rotated; a reconstruction wants the physical one.
    """
    stem = "mprage_noncart_3d_3sl_3avg_userotext1"
    physical = load(stem).calculate_kspace(dense=False)[0]
    logical = load(stem).calculate_kspace(dense=False, frame="logical")[0]

    assert physical.shape == logical.shape
    assert not np.allclose(physical, logical)


def test_an_unknown_frame_is_refused(gre):
    with pytest.raises(ValueError, match="physical"):
        gre.calculate_kspace(frame="scanner")


def test_window_averaging_is_off_by_default_and_only_moves_a_curved_readout():
    """Off by default so the answer stays comparable with PyPulseq and mrpro."""
    flat = load("fse_2d_1sl_1avg")
    assert np.allclose(
        flat.calculate_kspace(dense=False)[0],
        flat.calculate_kspace(dense=False, sample_window_average=True)[0],
    )

    curved = load("mprage_noncart_3d_3sl_3avg_userotext1")
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
    labels, aux = gre.auto_label(skip_apply=True)

    assert set(labels) == {"SLC", "LIN"}
    assert np.array_equal(labels["SLC"], np.tile([0, 1, 2], 8))
    assert np.array_equal(labels["LIN"], np.repeat(np.arange(8), 3))

    assert aux["kSpaceCenterLine"] == 4
    assert aux["kSpaceCenterSample"] == 32
    assert np.allclose(aux["SlicePositions"], [-5e-3, 0.0, 5e-3])


def test_the_slice_thickness_matches_the_prescription(gre):
    """Recovered from the RF spectrum over the slice-select amplitude.

    ``FOV`` and ``NumSlices`` state the answer independently -- they come from
    the prescription, not from the pulse -- and the sequence records no
    thickness of its own.
    """
    _, aux = gre.auto_label(skip_apply=True)
    prescribed = gre.definitions["FOV"][2] / gre.definitions["NumSlices"][0]
    assert aux["SliceThickness"] == pytest.approx(prescribed, rel=0.03)
    assert aux["SliceGap"] == pytest.approx(0.0, abs=0.03 * prescribed)


def test_the_navigators_of_an_epi_repeat_one_line():
    labels, aux = load("epi_2d_1sl_1avg").auto_label(skip_apply=True)

    assert set(labels) >= {"LIN", "REV", "REP"}
    assert labels["LIN"][0] == labels["LIN"][1] == labels["LIN"][2]
    assert list(labels["REP"][:3]) == [0, 1, 2]
    assert aux["kSpaceCenterLine"] == 64


def test_the_epi_echo_index_is_quoted_after_mirroring():
    """One number for a scan whose two polarities disagree about it by one.

    The forward readouts put the echo at 64 of 128 and the reverse ones at 63,
    because they reach the same point in k from opposite ends. A recon mirrors
    the reverse lines -- that is what ``REV`` is for -- and then both are at
    64, so 64 is the number that is true of the reconstructed data.
    """
    seq = load("epi_2d_1sl_1avg")
    _, aux = seq.auto_label(skip_apply=True)
    assert aux["kSpaceCenterSample"] == 64

    labels, _ = seq.auto_label(skip_apply=True)
    result = seq._kspace(dense=False)
    echo = np.asarray(result["readout_center_sample"])
    samples = np.asarray(result["readout_samples"])
    rev = np.asarray(labels["REV"], dtype=bool)

    assert set(echo.tolist()) == {63, 64}, "the fixture is not bipolar"
    assert np.array_equal(np.where(rev, samples - 1 - echo, echo), np.full(echo.shape, 64))


def test_a_single_named_dimension_takes_the_repeat_count():
    """One name is always safe: there is nothing to work out."""
    seq = load("epi_2d_1sl_1avg")
    plain, _ = seq.auto_label(skip_apply=True)

    named, _ = seq.auto_label(repeat_dims=["SET"], skip_apply=True)
    assert "REP" not in named
    assert np.array_equal(named["SET"], plain["REP"])


def test_ragged_repeats_are_refused_rather_than_split_anyway():
    """The EPI's navigators revisit one line; the other 127 are acquired once.

    That is not a dimension, and splitting it into two would put different
    acquisitions in one slot with every surrounding label looking ordinary.
    """
    seq = load("epi_2d_1sl_1avg")
    with pytest.raises(RuntimeError, match="not a rectangle"):
        seq.auto_label(repeat_dims=["REP", "ECO"], skip_apply=True)

    # Declared outright it is arithmetic again, and goes through.
    split, _ = seq.auto_label(repeat_dims=[("REP", 2), ("ECO", 2)], skip_apply=True)
    assert list(split["REP"][:3]) == [0, 0, 1]
    assert list(split["ECO"][:3]) == [0, 1, 0]


def test_repeat_dimensions_that_contradict_the_scan_are_refused():
    seq = load("epi_2d_1sl_1avg")
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
            block, rf, gx, gy, gz, adc, native.chain_extension(labelset, row, ext), duration
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
    seq = load("gre_2d_3sl_3avg")
    n = _stamp_label(seq, "ECO", lambda i: i % 2)
    seq.auto_label()

    replayed = _replay_labels(seq)
    assert np.array_equal(replayed["ECO"], np.array([i % 2 for i in range(n)]))
    assert "SLC" in replayed and "LIN" in replayed


def test_skip_hands_a_derived_counter_back_to_the_sequence():
    seq = load("epi_2d_1sl_1avg")
    n = _stamp_label(seq, "REP", lambda i: i // 40)
    want = np.array([i // 40 for i in range(n)])

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
    seq = load("mprage_noncart_3d_3sl_3avg_userotext1")
    with pytest.raises(RuntimeError, match="do not share a direction"):
        seq.auto_label(skip_apply=True)


def test_skip_apply_leaves_the_sequence_alone(gre):
    before = gre.definitions.copy()
    gre.auto_label(skip_apply=True)
    assert gre.definitions == before


def test_applying_writes_the_labels_and_the_definitions(gre):
    labels_before = gre._native.num_label_set()

    labels, aux = gre.auto_label()

    assert gre._native.num_label_set() > labels_before
    assert gre.definitions["kSpaceCenterLine"] == aux["kSpaceCenterLine"]
    assert gre.definitions["kSpaceCenterSample"] == aux["kSpaceCenterSample"]
    # And what was written survives a round trip through the file.
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "labelled.seq"
        gre.write(str(path))
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
# Against Pulseq's MATLAB autoLabel
# --------------------------------------------------------------------------

#: What `mr.Sequence.autoLabel` reports for the same fixtures, captured once
#: under Octave (`conda run -n octave2`) and checked in. Nothing in the shipped
#: code calls MATLAB; this is a frozen oracle, and regenerating it is a
#: deliberate act -- see the header of tests/utils/expected/.
ORACLE = FIXTURES / "autolabel_matlab_oracle.json"

#: The two places we deliberately disagree with MATLAB, and why.
#:
#: ``gre_2d_3sl_3avg`` differs on both counts at once. It plays five dummy
#: excitations at 0 mm before acquiring, and MATLAB uniques slice offsets over
#: *every* excitation, so those dummies take slice 0 and put 0 mm at the front
#: of ``SlicePositions``; we build the table from the slices actually acquired,
#: so every entry is a slice that exists. MATLAB then labels in acquisition
#: order (``sliceCountersAcquisitionOrder``, line 122) while we rank by
#: position, so that ``SlicePositions[SLC]`` is where slice ``SLC`` sits
#: whatever order the scan visited them in -- an interleaved acquisition is
#: what separates the two, and MATLAB computes that ordering as well
#: (``sliceCountersSorted``, line 128) before choosing the other one.
#:
#: ``epi_2d_1sl_1avg`` has two readout polarities whose echoes are one sample
#: apart -- the same point in k reached from opposite ends, 64 and 63 of 128.
#: One number for the scan is only true in one frame. MATLAB quotes the raw
#: index of the first readout it meets, a reverse navigator, giving 63; we
#: quote the frame a reconstruction reads it in, after ``REV`` has been
#: honoured, where every line sits at 64.
KNOWN_DIVERGENCES = {
    ("gre_2d_3sl_3avg", "SLC"),
    ("gre_2d_3sl_3avg", "SlicePositions"),
    ("epi_2d_1sl_1avg", "kSpaceCenterSample"),
}


def _oracle():
    import json

    if not ORACLE.exists():
        pytest.skip(f"no MATLAB oracle at {ORACLE}")
    return json.loads(ORACLE.read_text())


@pytest.mark.parametrize("stem", sorted(_oracle()))
def test_the_counters_match_matlab_autolabel(stem):
    """Element-wise equality on every counter. They are integers: exact or a bug."""
    expected = _oracle()[stem]["labels"]
    labels, _ = load(stem).auto_label(skip_apply=True)

    for name, want in expected.items():
        if (stem, name) in KNOWN_DIVERGENCES:
            continue
        assert name in labels, f"{stem}: MATLAB found {name} and we did not"
        assert list(map(int, labels[name])) == list(want), f"{stem} {name}"


@pytest.mark.parametrize("stem", sorted(_oracle()))
def test_the_definitions_match_matlab_autolabel(stem):
    """Reals, so compared with a tolerance rather than for equality.

    ``SliceThickness`` agrees to about 5e-6 relative -- the spectrum is carried
    in float32 here against MATLAB's double -- which on a 5 mm slice is 25 nm.

    ``SliceGap`` is judged in absolute terms rather than relative, because it
    is the difference of two millimetre-scale numbers: it inherits the
    thickness's *absolute* error while being two orders of magnitude smaller
    than it, so a relative tolerance on the gap would really be a tolerance of
    1e-4 on 25 nm and would mean nothing. 1 um on a 5 mm slice is 0.02%.
    """
    #: Absolute metres, where a relative tolerance would be meaningless.
    absolute = {"SliceGap": 1e-6}

    expected = _oracle()[stem]["aux"]
    _, aux = load(stem).auto_label(skip_apply=True)

    for name, want in expected.items():
        if (stem, name) in KNOWN_DIVERGENCES:
            continue
        assert name in aux, f"{stem}: MATLAB derived {name} and we did not"
        got = np.atleast_1d(np.asarray(aux[name], dtype=float))
        want = np.atleast_1d(np.asarray(want, dtype=float))
        assert got.shape == want.shape, f"{stem} {name}"
        if name in absolute:
            assert np.allclose(got, want, rtol=0.0, atol=absolute[name]), (
                f"{stem} {name}: {got} vs {want}"
            )
        else:
            assert np.allclose(got, want, rtol=1e-4, atol=1e-8), f"{stem} {name}: {got} vs {want}"


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
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in ours.values() if p.name != "self")


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
    labels, aux = load("gre_2d_3sl_3avg").auto_label(skip_apply=True, sort_slices=mode)
    positions = np.asarray(aux["SlicePositions"])
    assert positions.size == 3
    assert set(labels["SLC"]) == {0, 1, 2}
    # Whatever the order, the table and the counters name the same three slices.
    assert np.allclose(np.sort(positions), sorted([-5e-3, 0.0, 5e-3]))


def test_descending_is_the_reverse_of_ascending():
    up, aux_up = load("gre_2d_3sl_3avg").auto_label(skip_apply=True, sort_slices="ascending")
    down, aux_down = load("gre_2d_3sl_3avg").auto_label(skip_apply=True, sort_slices="descending")

    assert np.allclose(np.asarray(aux_down["SlicePositions"]),
                       np.asarray(aux_up["SlicePositions"])[::-1])
    assert np.array_equal(down["SLC"], 2 - np.asarray(up["SLC"]))
    # The gap is geometry: it cannot depend on which end the counting starts.
    assert aux_up["SliceGap"] == aux_down["SliceGap"]


def test_an_unknown_sorting_is_refused(gre):
    with pytest.raises(ValueError, match="sort_slices"):
        gre.auto_label(sort_slices="interleaved")


def test_mirror_fourier_turns_the_encoding_over_and_leaves_the_slices():
    plain, _ = load("gre_2d_3sl_3avg").auto_label(skip_apply=True)
    mirrored, _ = load("gre_2d_3sl_3avg").auto_label(skip_apply=True, mirror_fourier=True)

    # Line order reverses ...
    assert np.array_equal(mirrored["LIN"], plain["LIN"].max() - np.asarray(plain["LIN"]))
    # ... and the slice stack does not, which is what separates this from
    # reflect=[0, 1, 2].
    assert np.array_equal(mirrored["SLC"], plain["SLC"])


def test_use_labels_applies_what_detection_would_have():
    """Detect once, apply anywhere -- byte for byte the same file."""
    labels, aux = load("gre_2d_3sl_3avg").auto_label(skip_apply=True)

    reused = load("gre_2d_3sl_3avg")
    reused.auto_label(use_labels=labels, use_aux=aux)

    direct = load("gre_2d_3sl_3avg")
    direct.auto_label()

    assert reused._to_text(create_signature=False) == direct._to_text(create_signature=False)


def test_use_labels_carries_a_hand_corrected_counter_through():
    labels, aux = load("gre_2d_3sl_3avg").auto_label(skip_apply=True)
    edited = dict(labels)
    edited["LIN"] = np.asarray(labels["LIN"])[::-1].copy()

    seq = load("gre_2d_3sl_3avg")
    got, _ = seq.auto_label(use_labels=edited, use_aux=aux)
    assert np.array_equal(got["LIN"], edited["LIN"])
    assert "SlicePositions" in seq.definitions


def test_use_labels_with_a_detection_only_option_is_refused(gre):
    labels, _ = gre.auto_label(skip_apply=True)
    for option in ({"reflect": [0]}, {"reorder": [1, 0]}, {"mirror_fourier": True}):
        with pytest.raises(ValueError, match="only affect detection"):
            load("gre_2d_3sl_3avg").auto_label(use_labels=labels, **option)


def test_use_labels_of_the_wrong_length_is_refused(gre):
    with pytest.raises(ValueError, match="values for"):
        gre.auto_label(use_labels={"LIN": [1, 2, 3]})
