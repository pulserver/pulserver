"""The checks a chain passes before its IR is built, in the physical frame."""

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
    assert ir.check(train, SYSTEM, limits=ir.CheckLimits(bands=(BAND_X,))) == [
        "17.9 mT/m at 1000 Hz on x in the window at 0.000 s exceeds the 5.0 mT/m "
        "of the forbidden band 900-1100 Hz on x"
    ]
    assert ir.check(train, SYSTEM, limits=ir.CheckLimits(bands=(BAND_Y,))) == []


def test_the_prescription_moves_a_train_into_the_forbidden_band_of_another_axis(train):
    on_x, on_y = (ir.CheckLimits(bands=(band,)) for band in (BAND_X, BAND_Y))
    assert ir.check(train, SYSTEM, rotation=QUARTER, limits=on_x) == []
    assert len(ir.check(train, SYSTEM, rotation=QUARTER, limits=on_y)) == 1


def test_a_reflected_prescription_is_checked_as_the_rotation_it_mirrors(train):
    mirrored = np.diag([1.0, -1.0, 1.0]) @ QUARTER
    on_y = ir.CheckLimits(bands=(BAND_Y,))
    assert np.linalg.det(mirrored) == pytest.approx(-1.0)
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


@pytest.mark.parametrize(("local_limit", "refused"), [(2.0, True), (3.0, False)])
def test_local_sar_over_the_vops_is_refused_above_its_limit(
    tmp_path, local_limit, refused
):
    # One channel at 0.01 drive per Hz: (0.01 * 500)^2 for 1 ms in 10 is 2.5 W/kg.
    rf = pp.make_block_pulse(flip_angle=np.pi, duration=1e-3, system=SYSTEM)
    repetition = [(rf,), (pp.make_delay(9e-3),)]
    path = _written(tmp_path, repetition * 20)
    vops = tmp_path / "vops.npz"
    np.savez(vops, vops=np.ones((1, 1, 1), dtype=complex))
    limits = ir.CheckLimits(vops=vops, drive_per_hz=0.01, local_sar_limit=local_limit)
    assert ir.check(path, SYSTEM, limits=limits) == (
        ["local SAR of 2.50 W/kg at VOP 0 over blocks 1-2 exceeds 2.00 W/kg"]
        if refused
        else []
    )


def test_vops_without_a_channel_drive_are_refused():
    with pytest.raises(ValueError, match="drive per Hz"):
        ir.CheckLimits(vops="vops.npz")


def test_a_rotation_that_is_not_orthonormal_is_refused(diagonal):
    with pytest.raises(ValueError, match="not orthonormal"):
        ir.check(diagonal, SYSTEM, rotation=np.diag([1.0, 1.0, 1.01]))
