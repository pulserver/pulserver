"""The checks a chain passes before its IR is built, and its SAR against a reference."""

import json
import re

import numpy as np
import pypulseqpp as pp
import pytest
from pypulseqpp import safety

from pulserver import ir

SYSTEM = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
#: A quarter turn about z: logical x plays on physical y.
QUARTER = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
EIGHTH = np.array(
    [[np.sqrt(0.5), -np.sqrt(0.5), 0.0], [np.sqrt(0.5), np.sqrt(0.5), 0.0], [0, 0, 1]]
)
BAND_X = safety.ForbiddenBand("x", 900.0, 1100.0, 5.0)
BAND_Y = safety.ForbiddenBand("y", 900.0, 1100.0, 5.0)


def _written(tmp_path, blocks):
    seq = pp.Sequence(SYSTEM)
    for events in blocks:
        seq.add_block(*events)
    path = tmp_path / "sequence.seq"
    seq.write(path)
    return path


def _mT(axis, amplitude, rise_time, flat_time=1e-3):
    return pp.make_trapezoid(
        axis,
        amplitude=amplitude * 1e-3 * SYSTEM.gamma,
        rise_time=rise_time,
        flat_time=flat_time,
        system=SYSTEM,
    )


@pytest.fixture
def diagonal(tmp_path):
    """32 mT/m on x and y at once, at 100 T/m/s: sqrt(2) of that on one axis at 45 degrees."""
    return _written(tmp_path, [(_mT("x", 32, 320e-6), _mT("y", 32, 320e-6))])


@pytest.fixture
def train(tmp_path):
    """60 ms of 15 mT/m lobes alternating on x every 0.5 ms: 1 kHz."""
    lobes = [(_mT("x", sign * 15, 100e-6, 300e-6),) for sign in (1, -1)]
    return _written(tmp_path, lobes * 60)


def test_a_gradient_within_the_limit_on_each_logical_axis_can_exceed_it_on_a_physical_one(
    diagonal,
):
    assert ir.check(diagonal, SYSTEM) == []
    assert ir.check(diagonal, SYSTEM, rotation=EIGHTH) == [
        "gradient amplitude of 45.3 mT/m on y in block 1 exceeds 40.0 mT/m"
    ]


def test_a_block_labelled_norot_is_checked_as_it_plays_unrotated(tmp_path):
    unturned = pp.make_label(type="SET", label="NOROT", value=True)
    path = _written(tmp_path, [(_mT("x", 32, 320e-6), _mT("y", 32, 320e-6), unturned)])
    assert ir.check(path, SYSTEM, rotation=EIGHTH) == []


def test_a_gradient_train_in_a_forbidden_band_is_refused_on_the_axis_it_plays(train):
    # Every window of the train reads alike, so which one is worst is the FFT's
    # rounding; the window it names is not part of the claim.
    (problem,) = ir.check(train, SYSTEM, limits=ir.CheckLimits(bands=(BAND_X,)))
    assert re.fullmatch(
        r"17\.9 mT/m at 1000 Hz on x in the window at \d+\.\d{3} s exceeds the "
        r"5\.0 mT/m of the forbidden band 900-1100 Hz on x",
        problem,
    )
    assert ir.check(train, SYSTEM, limits=ir.CheckLimits(bands=(BAND_Y,))) == []


def test_the_prescription_moves_a_train_into_the_forbidden_band_of_another_axis(train):
    on_x, on_y = (ir.CheckLimits(bands=(band,)) for band in (BAND_X, BAND_Y))
    assert ir.check(train, SYSTEM, rotation=QUARTER, limits=on_x) == []
    assert len(ir.check(train, SYSTEM, rotation=QUARTER, limits=on_y)) == 1


def test_a_reflected_prescription_is_checked_as_it_plays(train):
    """Reflected onto -y, the train drives the band on y as the quarter turn does."""
    mirrored = np.diag([1.0, -1.0, 1.0]) @ QUARTER
    on_x, on_y = (ir.CheckLimits(bands=(band,)) for band in (BAND_X, BAND_Y))
    assert np.linalg.det(mirrored) == pytest.approx(-1.0)
    assert ir.check(train, SYSTEM, rotation=mirrored, limits=on_x) == []
    assert ir.check(train, SYSTEM, rotation=mirrored, limits=on_y) == ir.check(
        train, SYSTEM, rotation=QUARTER, limits=on_y
    )


@pytest.mark.parametrize(("pns_limit", "refused"), [(0.8, True), (1.0, False)])
def test_pns_is_refused_from_the_fraction_of_threshold_the_limits_allow(
    tmp_path, pns_limit, refused
):
    # 150 T/m/s for 200 us: 84% of the threshold, sampled at the raster centres.
    path = _written(tmp_path, [(_mT("x", 30, 200e-6),), (pp.make_delay(5e-3),)])
    nerve = safety.ChronaxieModel(chronaxie=360e-6, rheobase=20.0, alpha=0.324)
    problems = ir.check(
        path, SYSTEM, limits=ir.CheckLimits(pns=nerve, pns_limit=pns_limit)
    )
    assert problems == (
        [
            "PNS of 84% of the chronaxie threshold, largest on x, at 0.0002 s in "
            "block 1 reaches 80%"
        ]
        if refused
        else []
    )


def _pulse(degrees, duration=1e-3):
    return pp.make_block_pulse(
        flip_angle=np.deg2rad(degrees), duration=duration, system=SYSTEM
    )


#: Head and local SAR limits that put the reference pulse's local term at 1
#: in a VOP twice its head matrix.
HEAD_LIMIT, LOCAL_LIMIT = 5.0, 10.0


def _limits(vops, **more):
    return ir.CheckLimits(
        vops=vops, vop_head_limit=HEAD_LIMIT, vop_local_limit=LOCAL_LIMIT, **more
    )


def _one_channel_file(tmp_path, metadata=None, global_matrix=((0.5,),)):
    path = tmp_path / "vops.npz"
    np.savez(
        path,
        vops=np.ones((1, 1, 1)),
        global_matrix=np.asarray(global_matrix),
        metadata=np.array(json.dumps(metadata or {})),
    )
    return path


@pytest.fixture
def one_channel(tmp_path):
    """One channel's VOP and head matrix: every ratio is a ratio of RF energy."""
    return _limits(_one_channel_file(tmp_path))


@pytest.mark.parametrize(
    ("pulses", "ratio"),
    [
        ((180,), 1.0),
        ((90,), 0.25),
        ((180, 90), 0.625),
    ],
)
def test_a_repetitions_sar_is_measured_against_the_same_repetition_of_reference_pulses(
    tmp_path, one_channel, pulses, ratio
):
    # A 1 ms hard pulse's energy goes as its flip angle squared, so each pulse
    # counts (flip / 180)^2 of the reference's.
    repetition = [(_pulse(flip),) for flip in pulses] + [(pp.make_delay(9e-3),)]
    path = _written(tmp_path, repetition * 20)
    (found,) = ir.sar_ratios(path, SYSTEM, one_channel)
    assert (found.local_sar, found.global_sar) == pytest.approx((ratio, ratio))


def test_a_longer_pulse_of_the_same_flip_angle_deposits_less_than_the_reference(
    tmp_path, one_channel
):
    # Half the amplitude for twice as long: half the energy.
    path = _written(tmp_path, [(_pulse(180, 2e-3),), (pp.make_delay(9e-3),)] * 20)
    (found,) = ir.sar_ratios(path, SYSTEM, one_channel)
    assert (found.local_sar, found.global_sar) == pytest.approx((0.5, 0.5))


def test_a_sequence_without_rf_has_no_sar_ratio(tmp_path, one_channel):
    path = _written(tmp_path, [(pp.make_delay(1e-3),)] * 5)
    assert ir.sar_ratios(path, SYSTEM, one_channel) == [ir.SarRatio(0.0, 0.0)]


@pytest.mark.parametrize(("default_shim", "ratio"), [(None, 0.5), ((1, 0), 1.0)])
def test_the_reference_pulse_is_played_in_the_default_shim(
    tmp_path, default_shim, ratio
):
    # Two uncoupled channels: a pulse on the first alone deposits half of one
    # on both.
    vops = tmp_path / "vops.npz"
    np.savez(vops, vops=np.eye(2)[None], global_matrix=0.5 * np.eye(2))
    on_first = pp.make_rf_shim([1.0, 0.0])
    path = _written(tmp_path, [(_pulse(180), on_first), (pp.make_delay(9e-3),)] * 20)
    (found,) = ir.sar_ratios(path, SYSTEM, _limits(vops, default_shim=default_shim))
    assert (found.local_sar, found.global_sar) == pytest.approx((ratio, ratio))


def test_vops_given_as_a_model_or_as_their_file_give_the_same_ratios(
    tmp_path, one_channel
):
    path = _written(tmp_path, [(_pulse(90),), (pp.make_delay(9e-3),)] * 20)
    model = _limits(safety.read_vops(one_channel.vops))
    assert ir.sar_ratios(path, SYSTEM, model) == ir.sar_ratios(
        path, SYSTEM, one_channel
    )


def test_the_file_s_safety_factor_raises_the_local_ratio_and_not_the_global(
    tmp_path,
):
    vops = _one_channel_file(tmp_path, {"safety_factor": 1.5})
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    (found,) = ir.sar_ratios(path, SYSTEM, _limits(vops))
    assert (found.local_sar, found.global_sar) == pytest.approx((1.5, 1.0))


@pytest.mark.parametrize("head_limit", [3.2, 6.4])
def test_the_local_ratio_goes_with_the_head_limit_over_the_local_limit(
    tmp_path, head_limit
):
    vops = _one_channel_file(tmp_path)
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    limits = ir.CheckLimits(vops=vops, vop_head_limit=head_limit, vop_local_limit=10.0)
    (found,) = ir.sar_ratios(path, SYSTEM, limits)
    assert (found.local_sar, found.global_sar) == pytest.approx(
        (2.0 * head_limit / 10.0, 1.0)
    )


def test_the_local_ratio_is_over_the_body_model_the_reference_heats_least(tmp_path):
    # Two body models, one heated half as much: the drive that puts the
    # reference at the head limit there is twice as large.
    vops = _one_channel_file(tmp_path, global_matrix=[[[0.5]], [[0.25]]])
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    (found,) = ir.sar_ratios(path, SYSTEM, _limits(vops))
    assert (found.local_sar, found.global_sar) == pytest.approx((2.0, 1.0))


@pytest.mark.parametrize(
    ("written_for", "refused"),
    [("Head8Tx/8/0x1a2b3c4d", False), ("Head8Tx/8/0x00000000", True), (None, True)],
)
def test_the_vop_file_must_name_the_transmit_configuration_the_scanner_reports(
    tmp_path, written_for, refused
):
    metadata = {} if written_for is None else {"transmit": written_for}
    vops = _one_channel_file(tmp_path, metadata)
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    limits = _limits(vops, vop_coil="Head8Tx/8/0x1a2b3c4d")
    if refused:
        with pytest.raises(ValueError, match="transmit configuration"):
            ir.sar_ratios(path, SYSTEM, limits)
    else:
        (found,) = ir.sar_ratios(path, SYSTEM, limits)
        assert found.local_sar == pytest.approx(1.0)


def test_sar_ratios_need_the_head_and_local_limits(tmp_path):
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    limits = ir.CheckLimits(vops=_one_channel_file(tmp_path), vop_head_limit=3.2)
    with pytest.raises(ValueError, match="head and local SAR limits"):
        ir.sar_ratios(path, SYSTEM, limits)


def test_sar_ratios_need_the_head_matrices(tmp_path):
    vops = tmp_path / "vops.npz"
    np.savez(vops, vops=np.ones((1, 1, 1)))
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(9e-3),)] * 20)
    with pytest.raises(ValueError, match="global SAR matrices"):
        ir.sar_ratios(path, SYSTEM, _limits(vops))


def test_a_file_that_cannot_be_parsed_is_refused_by_the_checks_and_the_ratios(
    tmp_path, one_channel
):
    path = tmp_path / "sequence.seq"
    path.write_text("[VERSION]\nmajor 1\nminor 5\nrevision 1\n\n[BLOCKS]\n1 a b\n")
    with pytest.raises(ValueError, match="cannot read"):
        ir.check(path, SYSTEM)
    with pytest.raises(ValueError, match="cannot read"):
        ir.sar_ratios(path, SYSTEM, one_channel)


def test_sar_ratios_need_vops(train):
    with pytest.raises(ValueError, match="need VOPs"):
        ir.sar_ratios(train, SYSTEM, ir.CheckLimits())


def test_the_checks_apply_no_sar_limit(tmp_path, one_channel):
    # Thousands of W/kg by the VOP at a drive of 1 per Hz: no SAR limit applies.
    path = _written(tmp_path, [(_pulse(180),), (pp.make_delay(49e-3),)] * 20)
    assert ir.check(path, SYSTEM, limits=one_channel) == []


def test_a_rotation_that_is_not_orthonormal_is_refused(diagonal):
    with pytest.raises(ValueError, match="not orthonormal"):
        ir.check(diagonal, SYSTEM, rotation=np.diag([1.0, 1.0, 1.01]))


def _heard_on(axis, gain=1.0):
    """An acoustic response of ``gain`` Pa per mT/m on one physical axis, to 20 kHz."""
    transfer = np.zeros((3, 4001), complex)
    transfer["xyz".index(axis)] = gain
    return safety.AcousticResponse(transfer, 5.0)


def test_a_repetition_too_loud_on_the_physical_axis_it_plays_on_is_refused(train):
    limits = ir.CheckLimits(acoustic=_heard_on("y"))
    assert ir.check(train, SYSTEM, limits=limits) == []
    problems = ir.check(train, SYSTEM, rotation=QUARTER, limits=limits)
    assert [p.split(" of ")[0] for p in problems] == [
        "A-weighted average sound pressure level"
    ]
    assert "played without end exceeds 99 dB(A)" in problems[0]


def test_the_levels_the_cache_carries_are_those_the_check_holds_to_its_limits(train):
    limits = ir.CheckLimits(acoustic=_heard_on("y"))
    (carried,) = ir.spl_levels(train, SYSTEM, limits, rotation=QUARTER)
    (sequence,) = (s for _, s in ir._checks._read(train))
    pp.TransformFOV(rotation=QUARTER).apply_to_sequence(sequence, in_place=True)
    _, checked = safety.check_spl(sequence, limits.acoustic, system=SYSTEM)
    assert carried.peak_db == pytest.approx(checked.peak, abs=1e-9)
    assert carried.average_dba == pytest.approx(checked.average, abs=1e-9)


def test_an_acoustic_file_is_read_once_per_check_at_its_interval(tmp_path, train):
    import h5py

    path = tmp_path / "response.h5"
    with h5py.File(path, "w") as held:
        for axis in "XYZ":
            held[f"{axis}_AXIS_TRANSFER_FUNCTION"] = np.stack(
                [np.full(4096, 10.0), np.zeros(4096)]
            )
    limits = ir.CheckLimits(acoustic=path, acoustic_interval=30e-6)
    assert limits.acoustic_response().frequency_step == pytest.approx(
        1 / (4096 * 30e-6)
    )
    assert ir.check(train, SYSTEM, limits=limits)


def test_an_acoustic_file_without_its_interval_is_refused(tmp_path):
    with pytest.raises(ValueError, match="acoustic_interval"):
        ir.CheckLimits(acoustic=tmp_path / "response.h5")


def test_levels_handed_to_the_check_are_held_to_its_limits_in_place_of_a_response(
    train,
):
    levels = [ir.SplLevels(150.0, 80.0, (1, 2))]
    assert ir.check(train, SYSTEM, spl_levels=levels) == [
        "peak sound pressure level of 150.0 dB over blocks 1-2 played without end "
        "exceeds 140 dB"
    ]
