"""The virtual scanner on isochromats: the cache played through pypulseqpp's Bloch simulation."""

import math
import shutil
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import OFFSET, ORIENTATIONS, posed
from pypulseqpp import sequences
from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM

from pulserver import ir, virtual
from pulserver.mrd import read_chain

FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SEQUENCES = [
    "dedup_gre_pair.seq",
    "epi_2d_main.seq",
    "gre_2d_3sl.seq",
    "mprage_stack_of_spirals_3d.seq",
    "zte_3d.seq",
]
SYSTEM = pp.Opts(B0=3.0)
RNG = np.random.default_rng(7)
#: Isochromats across the fixtures' fields of view, relaxing, each at its own off-resonance.
POSITIONS = RNG.uniform(-0.05, 0.05, size=(40, 3))
TISSUE = {"t1": 0.8, "t2": 0.08, "off_resonance": RNG.uniform(-40.0, 40.0, 40)}
#: The phase the cache's single-precision gradients accrue over a fixture, relative to the signal.
PRECISION = 1e-4
#: Isochromat spacing, in metres: a seventh of the EPI's pixel.
SPACING = 1e-3


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """Each fixture beside the cache it converts to, at the magnet's field."""
    directory = tmp_path_factory.mktemp("fixtures")
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    for name in SEQUENCES:
        ir.convert(directory / name, SYSTEM)
    return directory


def _designed(seq, rotation):
    """The samples pypulseqpp simulates of each file's design, turned as the checks turn it, one scan across the chain.

    The files carry no field, so their ppm offsets are resolved at the one the
    caches are converted at.
    """
    spins = pp.Isochromats(POSITIONS, **TISSUE)
    parts = []
    for _, sequence in read_chain(seq):
        sequence.system.B0 = SYSTEM.B0
        turned = pp.TransformFOV(rotation=rotation).apply_to_sequence(sequence)
        parts.append(turned.simulate(spins))
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
@pytest.mark.parametrize("name", SEQUENCES)
def test_the_cache_played_on_isochromats_samples_what_its_design_simulated_samples(
    name, rotation, converted
):
    played = virtual.simulate(
        converted / name, pp.Isochromats(POSITIONS, **TISSUE), rotation=rotation
    )
    assert all(readout.dtype == np.complex64 for readout in played)
    played = np.concatenate(played, axis=1)
    designed = _designed(converted / name, rotation)
    assert played.shape == designed.shape
    assert np.abs(played - designed).max() < PRECISION * np.abs(designed).max()


def _single_shot(path, fov_offset=None):
    """A single-shot 2D EPI of one 90 degree excitation, converted with ``fov_offset``."""
    sequences.epi2D_sequence(
        n_x=32, n_y=32, n_dummy=0, flip_angle_deg=90.0, system=pp.Opts(B0=3.0)
    ).write(str(path))
    ir.convert(path, SYSTEM, fov_offset=fov_offset)
    return path


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_the_phantom_as_isochromats_is_acquired_as_the_analytic_phantom(
    rotation, tmp_path
):
    """A 90 degree excitation leaves the phantom's density at the phase the analytic model states."""
    seq = _single_shot(tmp_path / "epi.seq", fov_offset=OFFSET)
    tissue = posed(rotation)
    analytic = np.concatenate(virtual.acquire(seq, tissue, rotation=rotation), axis=1)
    bloch = np.concatenate(
        virtual.simulate(seq, tissue.isochromats(SPACING), rotation=rotation), axis=1
    )
    gain = np.vdot(analytic, bloch) / np.vdot(analytic, analytic)
    residual = np.linalg.norm(bloch - gain * analytic) / np.linalg.norm(analytic)
    assert gain == pytest.approx(1.0, abs=5e-3)
    assert residual < 1e-2


def _excited(tissue, **given):
    """The phantom's isochromats, turned by 90 degrees about x in 0.1 us."""
    spins = tissue.isochromats(SPACING, **given)
    spins.play(1e-7, rf=(0.0, 1e-7, [2.5e6]))
    return spins


@pytest.mark.parametrize(("t1", "t2"), [(0.3, 0.02), (1.2, 0.08)])
def test_an_ellipse_relaxes_with_its_own_times(t1, t2):
    ellipse = virtual.Ellipse((0.01, -0.02, 0.0), (0.01, 0.008), t1=t1, t2=t2)
    spins = _excited(virtual.Phantom([ellipse]))
    spins.play(0.05)
    m = spins.magnetization / SPACING**2
    np.testing.assert_allclose(
        np.hypot(m[:, 0], m[:, 1]), math.exp(-0.05 / t2), rtol=1e-4
    )
    np.testing.assert_allclose(m[:, 2], 1.0 - math.exp(-0.05 / t1), rtol=1e-4)


def test_an_ellipse_precesses_at_its_chemical_shift_at_the_field():
    ellipse = virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01), shift_ppm=FAT_SHIFT_PPM)
    spins = _excited(virtual.Phantom([ellipse]), field_t=3.0, off_resonance_hz=25.0)
    spins.play(1e-3)
    m = spins.magnetization / SPACING**2
    frequency = 1e-6 * pp.Opts().gamma * 3.0 * FAT_SHIFT_PPM + 25.0
    # A hard 90 degree pulse leaves the magnetization precessing as from
    # 2/pi of its duration before its end.
    precessed = 1e-3 + 2.0 / math.pi * 1e-7
    np.testing.assert_allclose(
        m[:, 0] + 1j * m[:, 1],
        1j * np.exp(-2j * math.pi * frequency * precessed),
        atol=1e-7,
    )


def test_the_isochromats_of_an_ellipse_fill_its_area_at_its_density():
    ellipse = virtual.Ellipse((0.013, 0.004, 0.0), (0.03, 0.017), 0.4, intensity=0.7)
    spins = _excited(virtual.Phantom([ellipse]))
    total = spins.magnetization[:, 1].sum()
    assert total == pytest.approx(0.7 * math.pi * 0.03 * 0.017, rel=1e-2)


@pytest.mark.parametrize(
    ("ellipse", "message"),
    [
        (
            virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01), intensity=1j),
            "complex intensity",
        ),
        (
            virtual.Ellipse((0.0, 0.0, 0.0), (0.01, 0.01), shift_ppm=FAT_SHIFT_PPM),
            "field_t",
        ),
    ],
)
def test_isochromats_carry_a_real_density_and_a_shift_at_a_field(ellipse, message):
    with pytest.raises(ValueError, match=message):
        virtual.Phantom([ellipse]).isochromats(SPACING)
