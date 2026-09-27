"""BrainWeb's normal brain as isochromats: its tissues, where the head lies, and its download."""

import argparse
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
    """The arguments pypulseqpp's isochromats are made with."""
    arguments = []

    def isochromats(positions, **fields):
        arguments.append(types.SimpleNamespace(positions=positions, **fields))
        return arguments[-1]

    monkeypatch.setattr(pp, "Isochromats", isochromats)
    return arguments


def test_each_tissue_of_a_voxel_is_an_isochromat_where_a_head_first_supine_head_puts_it(
    model, made
):
    model.fractions[2, 3, 1, GREY] = 0.25
    model.fractions[2, 3, 1, WHITE] = 0.75

    virtual.BrainWeb(directory="cache").isochromats(field_t=3.0)

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

    virtual.BrainWeb().isochromats(field_t=3.0, off_resonance_hz=20.0)

    (brain,) = made
    fat = 1e-6 * pp.Opts().gamma * 3.0 * FAT_SHIFT_PPM
    np.testing.assert_allclose(brain.off_resonance, [20.0, fat + 20.0])


def test_a_region_keeps_the_isochromats_inside_it(model, made):
    model.fractions[..., WHITE] = 1.0
    x = 1e-3 * -(np.arange(8) - 90.0)
    region = [[x[5] - 1e-4, x[3] + 1e-4], [-1.0, 1.0], [-1.0, 1.0]]

    virtual.BrainWeb().isochromats(field_t=3.0, region=region)

    (brain,) = made
    assert len(brain.positions) == 4 * 6 * 3
    assert np.all(
        (brain.positions[:, 0] >= x[5] - 1e-4) & (brain.positions[:, 0] <= x[3] + 1e-4)
    )


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


def test_the_scan_command_scans_brainweb_by_name():
    args = argparse.Namespace(phantom=Path("brainweb"), coils=3)

    brain = _command._phantom(args)

    assert isinstance(brain, virtual.BrainWeb)
    assert brain.coils == 3
