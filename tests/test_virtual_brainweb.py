"""BrainWeb's normal brain as the virtual scanner's tissue: its tissues, where the head lies, and its download."""

import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

from pulserver import virtual
from pulserver.virtual import _brainweb, _command

GREY, WHITE, FAT = 2, 3, 4


@pytest.fixture
def model(monkeypatch):
    """A small fuzzy model brainweb-dl hands over, and the calls it was asked."""
    fractions = np.zeros((4, 6, 8, len(_brainweb.TISSUES)), dtype=np.float32)
    calls = []

    def get_mri(sub_id, contrast, brainweb_dir=None):
        calls.append((sub_id, contrast, brainweb_dir))
        return fractions

    monkeypatch.setitem(
        sys.modules, "brainweb_dl", types.SimpleNamespace(get_mri=get_mri)
    )
    return types.SimpleNamespace(fractions=fractions, calls=calls)


def test_each_tissue_of_a_voxel_is_an_entry_where_a_head_first_supine_head_puts_it(
    model,
):
    model.fractions[2, 3, 1, GREY] = 0.25
    model.fractions[2, 3, 1, WHITE] = 0.75

    brain = virtual.BrainWeb(directory="cache", susceptibility=False).tissue(
        field_t=3.0
    )

    assert model.calls == [(0, "fuzzy", "cache")]
    left, posterior, superior = -(1 - 90.0), -(3 - 126.0), 2 - 72.0
    np.testing.assert_allclose(
        brain.positions, 1e-3 * np.array([[left, posterior, superior]] * 2)
    )
    np.testing.assert_allclose(brain.density, [0.86 * 0.25e-9, 0.77 * 0.75e-9])
    # Rooney et al.'s T1 and Cox and Gowland's T2 of grey and white matter at 3 T.
    np.testing.assert_allclose(brain.t1, [0.857 * 3.0**0.376, 0.583 * 3.0**0.382])
    np.testing.assert_allclose(brain.t2, [0.076, 0.071])
    np.testing.assert_allclose(brain.frequency, 0.0)
    assert brain.receive(brain.positions) is None


def test_cubes_average_the_fractions_of_their_voxels_at_their_centres(model):
    model.fractions[0:2, 0:2, 0:2, WHITE] = 0.5
    model.fractions[0, 0, 0, WHITE] = 1.0

    brain = virtual.BrainWeb().tissue(2e-3, field_t=3.0)

    np.testing.assert_allclose(
        brain.positions, 1e-3 * np.array([[-(0.5 - 90.0), -(0.5 - 126.0), 0.5 - 72.0]])
    )
    np.testing.assert_allclose(
        brain.density, [0.77 * (1.0 + 7 * 0.5) / 8 * (2e-3) ** 3]
    )


def test_fat_precesses_at_its_shift_at_the_field_beside_the_off_resonance(model):
    model.fractions[1, 1, 1, FAT] = 1.0
    model.fractions[1, 1, 2, WHITE] = 1.0

    brain = virtual.BrainWeb(susceptibility=False).tissue(
        field_t=3.0, off_resonance_hz=20.0
    )

    fat = 1e-6 * pp.Opts().gamma * 3.0 * FAT_SHIFT_PPM
    np.testing.assert_allclose(brain.frequency, [20.0, fat + 20.0])


def test_each_entry_precesses_at_the_mean_field_its_head_adds_over_its_cube(model):
    model.fractions[..., WHITE] = 1.0
    field = np.arange(model.fractions[..., 0].size, dtype=np.float32).reshape(
        model.fractions.shape[:3]
    )
    brain = virtual.BrainWeb()
    brain.field_ppm = field

    spins = brain.tissue(2e-3, field_t=3.0, off_resonance_hz=20.0)

    cubes = field.reshape(2, 2, 3, 2, 4, 2).mean(axis=(1, 3, 5)).reshape(-1)
    np.testing.assert_allclose(
        spins.frequency, 20.0 + 1e-6 * pp.Opts().gamma * 3.0 * cubes, rtol=1e-6
    )


def test_a_sphere_of_tissue_in_air_has_no_field_inside_once_shimmed_and_a_dipole_outside():
    centre, radius = 23.5, 10.0
    z, y, x = np.indices((48, 48, 48)) - centre
    distance = np.sqrt(x**2 + y**2 + z**2)

    field = _brainweb._susceptibility_field((distance > radius).astype(np.float32))

    contrast = _brainweb.WATER_PPM - _brainweb.AIR_PPM
    assert np.abs(field[distance < radius - 2.0]).max() < 0.02 * abs(contrast)
    pole, equator = field[39, 23, 23], field[23, 23, 39]
    assert pole == pytest.approx(2.0 / 3.0 * contrast * (radius / 15.5) ** 3, rel=0.1)
    assert equator == pytest.approx(-0.5 * pole, rel=0.05)


def _sphere_background(size, radius):
    """Background fraction of a sphere of tissue centred in a cube, each voxel sampled at 3 x 3 x 3 points."""
    centre = 0.5 * (size - 1)
    offsets = np.array([-1.0, 0.0, 1.0]) / 3.0
    z, y, x = np.indices((size, size, size)) - centre
    inside = np.zeros((size, size, size))
    for dz in offsets:
        for dy in offsets:
            for dx in offsets:
                inside += (x + dx) ** 2 + (y + dy) ** 2 + (z + dz) ** 2 <= radius**2
    return (1.0 - inside / 27.0).astype(np.float32)


def test_tissue_in_a_voxel_the_surface_of_a_sphere_crosses_is_in_the_field_inside_the_sphere():
    background = _sphere_background(64, 20.0)

    field = _brainweb._susceptibility_field(background)

    contrast = _brainweb.WATER_PPM - _brainweb.AIR_PPM
    assert np.abs(field[background < 1.0]).max() < 0.02 * abs(contrast)


def test_background_scattered_inside_the_tissue_of_a_sphere_adds_no_field():
    background = _sphere_background(64, 20.0)
    speckled = background.copy()
    speckled[26:38:3, 26:38:3, 26:38:3] = 0.05

    np.testing.assert_array_equal(
        _brainweb._susceptibility_field(speckled),
        _brainweb._susceptibility_field(background),
    )


def test_a_region_keeps_the_entries_it_answers_for_from_their_positions_and_frequencies(
    model,
):
    model.fractions[..., WHITE] = 1.0
    model.fractions[..., FAT] = 1.0
    x = 1e-3 * -(np.arange(8) - 90.0)

    def region(positions, frequencies):
        across = (positions[:, 0] >= x[5] - 1e-4) & (positions[:, 0] <= x[3] + 1e-4)
        return across & (frequencies > -100.0)

    brain = virtual.BrainWeb(susceptibility=False)
    kept = brain.tissue(field_t=3.0, region=region)

    assert len(kept.positions) == 4 * 6 * 3
    assert np.all(
        (kept.positions[:, 0] >= x[5] - 1e-4) & (kept.positions[:, 0] <= x[3] + 1e-4)
    )
    np.testing.assert_allclose(kept.t1, brain.relaxation(3.0)["white matter"][0])


def test_the_entries_hold_their_density_and_are_received_by_each_coil(model):
    model.fractions[1, 2, 3, GREY] = 1.0
    model.fractions[3, 4, 5, WHITE] = 0.5

    brain = virtual.BrainWeb(coils=3).tissue(field_t=1.5)

    assert brain.coils == 3
    np.testing.assert_allclose(brain.density, [0.86e-9, 0.77 * 0.5e-9])
    assert brain.receive(brain.positions).shape == (2, 3)


@pytest.mark.parametrize(
    ("spacing", "field_t", "match"),
    [(1.5e-3, 3.0, "whole millimetres"), (1e-3, None, "field_t")],
)
def test_a_fractional_spacing_or_a_missing_field_is_refused(
    model, spacing, field_t, match
):
    with pytest.raises(ValueError, match=match):
        virtual.BrainWeb().tissue(spacing, field_t=field_t)


def test_a_brain_received_by_coils_of_its_own_is_not_scanned_with_a_coil(model):
    with pytest.raises(ValueError, match="coils of its own"):
        virtual.BrainWeb(coils=3).tissue(field_t=3.0, coil=virtual.COILS["body/head48"])


def test_a_model_of_other_tissues_is_refused(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "brainweb_dl",
        types.SimpleNamespace(get_mri=lambda *a, **k: np.zeros((2, 2, 2, 12))),
    )
    with pytest.raises(ValueError, match="10 tissues"):
        virtual.BrainWeb().tissue(field_t=3.0)


def test_the_scan_command_scans_brainweb_by_name():
    args = argparse.Namespace(phantom=Path("brainweb"), coils=3, coil=None)

    brain = _command._phantom(args)

    assert isinstance(brain, virtual.BrainWeb)
    assert brain.coils == 3


def test_a_tissue_class_relaxes_with_the_t2_prime_it_is_given(model):
    model.fractions[2, 3, 1, GREY] = 0.25
    model.fractions[2, 3, 1, WHITE] = 0.75
    brain = virtual.BrainWeb(susceptibility=False, t2_prime={"grey matter": 0.05})

    sampled = brain.tissue(field_t=3.0)

    np.testing.assert_allclose(
        sampled.t2_prime, [0.05, brain.relaxation(3.0)["white matter"][2]]
    )


def test_tissues_without_in_vivo_fits_keep_brainweb_s_relaxation_at_1_5_t():
    relaxed = virtual.BrainWeb().relaxation(1.5)

    for name in ("fat", "muscle and skin", "background", "skull"):
        assert relaxed[name][:2] == _brainweb.TISSUES[name][:2]
    assert relaxed["CSF"][1] == _brainweb.TISSUES["CSF"][1]
    # R2' = R2* - R2 at BrainWeb's T2 and T2*, 70 ms and 30 ms for muscle.
    assert relaxed["muscle and skin"][2] == pytest.approx(
        1.0 / (1.0 / 0.030 - 1.0 / 0.047)
    )
    assert relaxed["skin"][2] == pytest.approx(1.0 / (1.0 / 0.058 - 1.0 / 0.329))
    assert relaxed["skull"][2] == math.inf


@pytest.mark.parametrize("field_t", [0.5, 1.5, 3.0, 7.0, 9.4])
def test_csf_has_no_static_dephasing_at_any_field(field_t):
    assert virtual.BrainWeb().relaxation(field_t)["CSF"][2] == math.inf


@pytest.mark.parametrize(
    ("field_t", "grey", "white"),
    [(1.5, 0.0840, 0.0662), (3.0, 0.0660, 0.0532), (7.0, 0.0332, 0.0268)],
)
def test_brain_t2_star_matches_peters_at_the_fields_they_measured(field_t, grey, white):
    relaxed = virtual.BrainWeb().relaxation(field_t)

    for name, t2_star in (("grey matter", grey), ("white matter", white)):
        t2, t2_prime = relaxed[name][1:]
        assert 1.0 / (1.0 / t2 + 1.0 / t2_prime) == pytest.approx(t2_star)
    assert relaxed["glial matter"][2] == relaxed["grey matter"][2]
    assert relaxed["connective tissue"][2] == relaxed["white matter"][2]


def test_grey_matter_outlasts_white_matter_in_t2_star_at_7_t():
    relaxed = virtual.BrainWeb().relaxation(7.0)

    def t2_star(name):
        return 1.0 / (1.0 / relaxed[name][1] + 1.0 / relaxed[name][2])

    assert t2_star("grey matter") > t2_star("white matter")


@pytest.mark.parametrize("field_t", [0.2, 1.5, 3.0, 7.0])
def test_brain_t1_follows_rooney_s_fits_and_csf_s_holds_at_4_3_s(field_t):
    relaxed = virtual.BrainWeb().relaxation(field_t)

    assert relaxed["white matter"][0] == pytest.approx(0.583 * field_t**0.382)
    assert relaxed["connective tissue"][0] == relaxed["white matter"][0]
    assert relaxed["grey matter"][0] == pytest.approx(0.857 * field_t**0.376)
    assert relaxed["glial matter"][0] == relaxed["grey matter"][0]
    assert relaxed["CSF"][0] == relaxed["skin"][0] == 4.3


def test_csf_takes_twice_grey_matter_s_t1_at_7_t():
    relaxed = virtual.BrainWeb().relaxation(7.0)

    assert relaxed["CSF"][0] > 2 * relaxed["grey matter"][0]
    assert relaxed["grey matter"][0] > relaxed["white matter"][0] > 1.0


@pytest.mark.parametrize(
    ("field_t", "grey", "white"),
    [(1.5, 0.087, 0.080), (3.0, 0.076, 0.071), (7.0, 0.047, 0.047)],
)
def test_brain_t2_matches_cox_and_gowland_at_the_fields_they_measured(
    field_t, grey, white
):
    relaxed = virtual.BrainWeb().relaxation(field_t)

    assert relaxed["grey matter"][1] == pytest.approx(grey)
    assert relaxed["white matter"][1] == pytest.approx(white)


def test_brain_t2_falls_linearly_between_measured_fields_and_holds_outside():
    brain = virtual.BrainWeb()

    assert brain.relaxation(5.0)["grey matter"][1] == pytest.approx(
        0.076 + (0.047 - 0.076) * (5.0 - 3.0) / 4.0
    )
    assert brain.relaxation(9.4)["white matter"][1] == pytest.approx(0.047)
    assert brain.relaxation(0.5)["white matter"][1] == pytest.approx(0.080)


def test_muscle_and_fat_t1_follow_their_power_laws_and_r2_prime_the_field():
    brain = virtual.BrainWeb(t2_prime={"white matter": 0.2})
    low, high = brain.relaxation(1.5), brain.relaxation(7.0)

    ratio = 7.0 / 1.5
    assert high["muscle and skin"][0] == pytest.approx(
        low["muscle and skin"][0] * ratio**0.4203
    )
    assert high["fat"][0] == pytest.approx(low["fat"][0] * ratio**0.1743)
    assert high["fat"][1] == low["fat"][1]
    assert high["muscle and skin"][2] == pytest.approx(
        low["muscle and skin"][2] / ratio
    )
    assert high["white matter"][2] == low["white matter"][2] == 0.2
    with pytest.raises(ValueError, match="above zero"):
        brain.relaxation(0.0)
