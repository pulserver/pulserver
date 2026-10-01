"""BrainWeb's normal brain as isochromats: its tissues, where the head lies, and its download."""

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


@pytest.fixture
def made(monkeypatch):
    """The arguments the isochromats are made with."""
    arguments = []

    def isochromats(positions, **fields):
        arguments.append(types.SimpleNamespace(positions=positions, **fields))
        return arguments[-1]

    monkeypatch.setattr(_brainweb, "Isochromats", isochromats)
    return arguments


def test_each_tissue_of_a_voxel_is_an_isochromat_where_a_head_first_supine_head_puts_it(
    model, made
):
    model.fractions[2, 3, 1, GREY] = 0.25
    model.fractions[2, 3, 1, WHITE] = 0.75

    virtual.BrainWeb(directory="cache", susceptibility=False).isochromats(field_t=3.0)

    assert model.calls == [(0, "fuzzy", "cache")]
    (brain,) = made
    left, posterior, superior = -(1 - 90.0), -(3 - 126.0), 2 - 72.0
    np.testing.assert_allclose(
        brain.positions, 1e-3 * np.array([[left, posterior, superior]] * 2)
    )
    np.testing.assert_allclose(brain.proton_density, [0.86 * 0.25e-9, 0.77 * 0.75e-9])
    np.testing.assert_allclose(brain.t1, [0.833, 0.5])
    np.testing.assert_allclose(brain.t2, [0.083, 0.07])
    np.testing.assert_allclose(brain.off_resonance, 0.0)
    assert brain.receive is None


def test_cubes_average_the_fractions_of_their_voxels_at_their_centres(model, made):
    model.fractions[0:2, 0:2, 0:2, WHITE] = 0.5
    model.fractions[0, 0, 0, WHITE] = 1.0

    virtual.BrainWeb().isochromats(2e-3, field_t=3.0)

    (brain,) = made
    np.testing.assert_allclose(
        brain.positions, 1e-3 * np.array([[-(0.5 - 90.0), -(0.5 - 126.0), 0.5 - 72.0]])
    )
    np.testing.assert_allclose(
        brain.proton_density, [0.77 * (1.0 + 7 * 0.5) / 8 * (2e-3) ** 3]
    )


def test_fat_precesses_at_its_shift_at_the_field_beside_the_off_resonance(model, made):
    model.fractions[1, 1, 1, FAT] = 1.0
    model.fractions[1, 1, 2, WHITE] = 1.0

    virtual.BrainWeb(susceptibility=False).isochromats(
        field_t=3.0, off_resonance_hz=20.0
    )

    (brain,) = made
    fat = 1e-6 * pp.Opts().gamma * 3.0 * FAT_SHIFT_PPM
    np.testing.assert_allclose(brain.off_resonance, [20.0, fat + 20.0])


def test_each_isochromat_precesses_at_the_mean_field_its_head_adds_over_its_cube(
    model, made
):
    model.fractions[..., WHITE] = 1.0
    field = np.arange(model.fractions[..., 0].size, dtype=np.float32).reshape(
        model.fractions.shape[:3]
    )
    brain = virtual.BrainWeb()
    brain.field_ppm = field

    brain.isochromats(2e-3, field_t=3.0, off_resonance_hz=20.0)

    (spins,) = made
    cubes = field.reshape(2, 2, 3, 2, 4, 2).mean(axis=(1, 3, 5)).reshape(-1)
    np.testing.assert_allclose(
        spins.off_resonance, 20.0 + 1e-6 * pp.Opts().gamma * 3.0 * cubes, rtol=1e-6
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


def test_a_region_keeps_the_isochromats_it_answers_for_from_their_positions_and_frequencies(
    model, made
):
    model.fractions[..., WHITE] = 1.0
    model.fractions[..., FAT] = 1.0
    x = 1e-3 * -(np.arange(8) - 90.0)

    def region(positions, frequencies):
        across = (positions[:, 0] >= x[5] - 1e-4) & (positions[:, 0] <= x[3] + 1e-4)
        return across & (frequencies > -100.0)

    brain = virtual.BrainWeb(susceptibility=False)
    brain.isochromats(field_t=3.0, region=region)

    (kept,) = made
    assert len(kept.positions) == 4 * 6 * 3 == brain.count(field_t=3.0, region=region)
    assert np.all(
        (kept.positions[:, 0] >= x[5] - 1e-4) & (kept.positions[:, 0] <= x[3] + 1e-4)
    )
    np.testing.assert_allclose(kept.t1, _brainweb.TISSUES["white matter"][0])


@pytest.mark.parametrize("spacing", [1e-3, 2e-3])
def test_slabs_keep_in_order_the_isochromats_a_test_of_every_voxel_keeps(
    model, made, spacing
):
    rng = np.random.default_rng(0)
    model.fractions[...] = rng.uniform(size=model.fractions.shape) * (
        rng.uniform(size=model.fractions.shape) < 0.5
    )
    # The model lies 83 to 90 mm along x and -72 to -69 mm along z. At 300 Hz
    # off resonance, the first two slabs hold water two or three voxels thick
    # and the third fat, which precesses 447 Hz lower at 3 T.
    slabs = virtual.Slabs(
        gradients=((1e4, 0.0, 0.0), (0.0, 0.0, 2e4), (1e4, 0.0, 0.0)),
        bounds=((1145.0, 1172.0), (-1125.0, -1095.0), (698.0, 725.0)),
    )
    brain = virtual.BrainWeb(susceptibility=False)

    brain.isochromats(spacing, field_t=3.0, off_resonance_hz=300.0, region=slabs)
    brain.isochromats(
        spacing, field_t=3.0, off_resonance_hz=300.0, region=lambda *at: slabs(*at)
    )
    brain.isochromats(spacing, field_t=3.0, off_resonance_hz=300.0)

    through_slabs, everywhere, whole = made
    assert 0 < len(through_slabs.positions) < len(whole.positions)
    for field in ("positions", "proton_density", "t1", "t2", "off_resonance"):
        np.testing.assert_array_equal(
            getattr(through_slabs, field), getattr(everywhere, field)
        )


@pytest.mark.parametrize("spacing", [1e-3, 2e-3])
def test_a_count_is_how_many_isochromats_the_brain_is_sampled_as(model, made, spacing):
    model.fractions[1:3, 2:5, 3:8, GREY] = 0.4
    model.fractions[0:3, 1:4, 2:6, FAT] = 0.6
    brain = virtual.BrainWeb(susceptibility=False)

    def region(positions, frequencies):
        return frequencies > -100.0

    counted = [
        brain.count(spacing, field_t=3.0),
        brain.count(spacing, field_t=3.0, region=region),
    ]
    brain.isochromats(spacing, field_t=3.0)
    brain.isochromats(spacing, field_t=3.0, region=region)

    assert counted == [len(sampled.positions) for sampled in made]
    assert counted[1] < counted[0]


def test_the_isochromats_start_at_rest_at_their_density_and_are_received_by_each_coil(
    model,
):
    model.fractions[1, 2, 3, GREY] = 1.0
    model.fractions[3, 4, 5, WHITE] = 0.5

    brain = virtual.BrainWeb(coils=3).isochromats(field_t=1.5)

    assert brain.coils == 3
    np.testing.assert_allclose(
        brain.magnetization, [[0.0, 0.0, 0.86e-9], [0.0, 0.0, 0.77 * 0.5e-9]]
    )


@pytest.mark.parametrize(
    ("spacing", "field_t", "match"),
    [(1.5e-3, 3.0, "whole millimetres"), (1e-3, None, "field_t")],
)
def test_a_fractional_spacing_or_a_missing_field_is_refused(
    model, spacing, field_t, match
):
    with pytest.raises(ValueError, match=match):
        virtual.BrainWeb().isochromats(spacing, field_t=field_t)
    with pytest.raises(ValueError, match=match):
        virtual.BrainWeb().count(spacing, field_t=field_t)


def test_a_brain_received_by_coils_of_its_own_is_not_scanned_with_a_coil(model):
    with pytest.raises(ValueError, match="coils of its own"):
        virtual.BrainWeb(coils=3).isochromats(
            field_t=3.0, coil=virtual.COILS["body/head48"]
        )


def test_a_model_of_other_tissues_is_refused(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "brainweb_dl",
        types.SimpleNamespace(get_mri=lambda *a, **k: np.zeros((2, 2, 2, 12))),
    )
    with pytest.raises(ValueError, match="10 tissues"):
        virtual.BrainWeb().isochromats(field_t=3.0)


def test_without_brainweb_dl_the_brain_names_the_extra_that_downloads_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "brainweb_dl", None)
    with pytest.raises(ImportError, match=r"pulserver\[brainweb\]"):
        virtual.BrainWeb().isochromats(field_t=3.0)


@pytest.mark.parametrize("diffusion", [False, True])
def test_the_scan_command_scans_brainweb_by_name(diffusion):
    args = argparse.Namespace(
        phantom=Path("brainweb"), coils=3, coil=None, diffusion=diffusion
    )

    brain = _command._phantom(args)

    assert isinstance(brain, virtual.BrainWeb)
    assert brain.coils == 3
    assert brain.diffusion == (dict(virtual.BrainWeb.DIFFUSION) if diffusion else {})


def test_each_tissue_of_a_voxel_is_spread_over_its_spins_with_its_t2_prime_and_diffusion(
    model, made
):
    model.fractions[2, 3, 1, GREY] = 0.25
    model.fractions[2, 3, 1, WHITE] = 0.75
    brain = virtual.BrainWeb(
        susceptibility=False,
        t2_prime={"grey matter": 0.05},
        diffusion=virtual.BrainWeb.DIFFUSION,
    )

    brain.isochromats(field_t=3.0, spins=8, voxel="box", seed=0)

    (spins,) = made
    centre = 1e-3 * np.array([-(1 - 90.0), -(3 - 126.0), 2 - 72.0])
    offsets = spins.positions - centre
    np.testing.assert_allclose(np.unique(np.round(offsets, 12)), [-2.5e-4, 2.5e-4])
    np.testing.assert_allclose(
        spins.proton_density, np.repeat([0.86 * 0.25e-9, 0.77 * 0.75e-9], 8) / 8
    )
    np.testing.assert_allclose(spins.diffusion, np.repeat([0.89e-9, 0.70e-9], 8))
    for line, t2_prime in zip(
        np.split(spins.off_resonance, 2),
        (0.05, brain.t2_prime["white matter"]),
        strict=True,
    ):
        assert len(np.unique(line)) == 8
        assert np.abs(line).max() <= 32.0 / (2.0 * np.pi * t2_prime)


def test_each_tissue_takes_the_t2_prime_its_t2_and_t2_star_in_brainweb_s_simulator_give():
    brain = virtual.BrainWeb(t2_prime={"white matter": 0.2})

    # R2' = R2* - R2 at BrainWeb's T2 and T2*, 83 ms and 69 ms for grey matter.
    assert brain.t2_prime["grey matter"] == pytest.approx(
        1.0 / (1.0 / 0.069 - 1.0 / 0.083)
    )
    assert brain.t2_prime["CSF"] == pytest.approx(1.0 / (1.0 / 0.058 - 1.0 / 0.329))
    assert brain.t2_prime["white matter"] == 0.2
    assert brain.t2_prime["skull"] == math.inf
