"""The limits a design call carries: scanner limits, conversion options and check limits."""

import cmath
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from pypulseqpp import safety

from pulserver import ir
from pulserver.host import DesignStore, call
from pulserver.host._blocks import format_import
from pulserver.host._limits import check_limits, split_limits
from pulserver.protocol import FOV_ROTATION, prescribed_rotation

SCANNER = {
    "max_grad": 40.0,
    "grad_unit": "mT/m",
    "max_slew": 150.0,
    "slew_unit": "T/m/s",
}
SAFE = {
    "a1": 0.4,
    "a2": 0.1,
    "a3": 0.5,
    "tau1": 0.2,
    "tau2": 0.03,
    "tau3": 3.0,
    "stim_limit": 30.0,
    "g_scale": 0.35,
}


def _safe_model():
    return {f"pns_{axis}_{name}": v for axis in "xyz" for name, v in SAFE.items()}


def _rotation_entries(matrix, digits=6):
    return {
        name: float(f"{value:.{digits}g}")
        for name, value in zip(FOV_ROTATION, np.ravel(matrix), strict=True)
    }


def test_the_check_limits_are_read_apart_from_the_scanner_limits():
    limits = {
        **SCANNER,
        "pns_chronaxie": 360e-6,
        "pns_rheobase": 20.0,
        "pns_alpha": 0.324,
        "pns_limit": 0.8,
        "forbidden_band_10": "all 590 650",
        "forbidden_band_2": "x 1100 1200 5.5",
        "vop_file": "/data/vops.mat",
        "vop_drive_per_hz": "0.01 0.02",
        "vop_default_shim": "1 0 0.5 1.5",
    }
    system, _, checked = split_limits(limits)
    assert system.max_grad == pytest.approx(40e-3 * system.gamma)
    assert checked == ir.CheckLimits(
        pns=safety.ChronaxieModel(360e-6, 20.0, 0.324),
        pns_limit=0.8,
        bands=(
            safety.ForbiddenBand("x", 1100.0, 1200.0, 5.5),
            safety.ForbiddenBand(None, 590.0, 650.0, 0.0),
        ),
        vops=Path("/data/vops.mat"),
        drive_per_hz=(0.01, 0.02),
        default_shim=(1 + 0j, cmath.rect(0.5, 1.5)),
    )


def test_a_vop_file_alone_is_read_with_a_unit_drive_and_equal_weights():
    assert check_limits({"vop_file": "/data/vops.mat"}) == ir.CheckLimits(
        vops=Path("/data/vops.mat")
    )


def test_a_design_under_a_vop_file_carries_its_sar_ratios_in_the_cache(tmp_path):
    vops = tmp_path / "vops.npz"
    np.savez(vops, vops=np.ones((1, 1, 1)))
    system = pp.Opts(**SCANNER)
    seq = pp.Sequence(system)
    rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    for _ in range(10):
        seq.add_block(rf)
        seq.add_block(pp.make_delay(9e-3))
    path = tmp_path / "sequence.seq"
    seq.write(path)
    store = DesignStore(tmp_path / "designs")
    status, reply = call(
        "import",
        limits={**SCANNER, "vop_file": str(vops)},
        block=format_import(path),
        store=store,
    )
    assert status == 0, reply
    stored = store.directory(reply.split()[1]) / "sequence.seq"
    (loaded,) = ir.summary(stored, system, cache_ext=".pseg")["subsequences"]
    # A 90 degree, 1 ms hard pulse deposits a quarter of the reference's energy.
    assert loaded["vop_sar_ratio"] == pytest.approx(0.25)
    assert loaded["vop_global_sar_ratio"] == 0.0


def test_limits_without_check_limits_check_timing_and_gradients_only():
    assert split_limits(SCANNER)[2] == ir.CheckLimits()


def test_a_safe_model_is_read_for_every_axis_as_pypulseqpp_takes_it():
    model = check_limits(_safe_model()).pns
    assert [getattr(model, axis).stim_limit for axis in "xyz"] == [30.0] * 3
    seq = pp.Sequence(pp.Opts(**SCANNER))
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=seq.system))
    _, report = safety.check_pns(seq, model)
    assert report.model == "safe"


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"pns_chronaxy": 360e-6}, "not check limits"),
        ({**_safe_model(), "pns_chronaxie": 360e-6, "pns_rheobase": 20}, "not both"),
        ({k: v for k, v in _safe_model().items() if k != "pns_z_tau3"}, "pns_z_tau3"),
        ({"pns_chronaxie": 360e-6, "pns_rheobase": 20, "pns_limit": 0}, "positive"),
        ({"forbidden_band_1": "w 590 650"}, "forbidden band"),
        ({"forbidden_band_1": "x 590"}, "forbidden band"),
        ({"forbidden_band_1": "x 590 high"}, "forbidden band"),
        ({"vop_drive_per_hz": 0.01}, "vop_file"),
        ({"vop_file": "/data/vops.mat", "vop_local_limit": 20}, "not check limits"),
        ({"vop_file": "/data/vops.mat", "vop_default_shim": "1 0 1"}, "a phase"),
        ({"vop_file": "/data/vops.mat", "vop_default_shim": "1 zero"}, "a phase"),
    ],
)
def test_check_limits_that_cannot_be_read_are_refused(limits, message):
    with pytest.raises(ValueError, match=message):
        check_limits(limits)


def test_an_absent_rotation_entry_is_the_identitys():
    np.testing.assert_array_equal(prescribed_rotation({}), np.eye(3))


def test_a_rotation_carried_at_six_digits_is_made_orthonormal():
    exact = np.array(
        [
            [np.cos(0.3), -np.sin(0.3), 0.0],
            [np.sin(0.3) * np.cos(0.7), np.cos(0.3) * np.cos(0.7), -np.sin(0.7)],
            [np.sin(0.3) * np.sin(0.7), np.cos(0.3) * np.sin(0.7), np.cos(0.7)],
        ]
    )
    rotation = prescribed_rotation(_rotation_entries(exact))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(rotation, exact, atol=1e-6)


def test_a_reflected_rotation_stays_a_reflection():
    mirrored = np.diag([1.0, -1.0, 1.0])
    assert np.linalg.det(prescribed_rotation(_rotation_entries(mirrored))) == (
        pytest.approx(-1.0)
    )


def test_a_rotation_that_is_not_orthonormal_is_refused():
    with pytest.raises(ValueError, match="not orthonormal"):
        prescribed_rotation(_rotation_entries(np.diag([1.0, 1.0, 1.01])))
