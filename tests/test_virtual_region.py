"""The slabs a scan's excitation pulses excite, and the isochromats kept in them."""

import math

import numpy as np
import pypulseqpp as pp
import pytest
from _virtual import ORIENTATIONS

from pulserver import ir, virtual
from pulserver.virtual import _phantom

SYSTEM = pp.Opts(B0=3.0)
THICKNESS = 5e-3


def _written(path, *blocks):
    """A sequence file of ``blocks`` beside its cache."""
    sequence = pp.Sequence(SYSTEM)
    for block in blocks:
        sequence.add_block(*block)
    sequence.write(str(path))
    ir.convert(path, SYSTEM)
    return path


def _slice(offset=0.0):
    """A 90 degree excitation of a slice ``offset`` metres along the logical z axis, and its gradient."""
    rf, gz, _ = pp.make_sinc_pulse(
        math.pi / 2,
        duration=2e-3,
        slice_thickness=THICKNESS,
        apodization=0.5,
        time_bw_product=4,
        system=SYSTEM,
        return_gz=True,
        use="excitation",
    )
    rf.freq_offset = gz.amplitude * offset
    return rf, gz


def _readout():
    return (pp.make_adc(64, duration=3.2e-3, system=SYSTEM),)


@pytest.mark.parametrize("rotation", ORIENTATIONS.values(), ids=ORIENTATIONS.keys())
def test_a_slice_selective_excitation_excites_its_slice_along_its_turned_gradient(
    rotation, tmp_path
):
    rf, gz = _slice(offset=0.01)
    seq = _written(tmp_path / "slice.seq", (rf, gz), _readout())

    slabs = virtual.excited(seq, rotation)

    (gradient,) = slabs.gradients
    np.testing.assert_allclose(gradient, gz.amplitude * rotation[:, 2], atol=1e-6)
    normal = rotation[:, 2]
    inside = 0.01 + np.array([0.0, -0.45, 0.45]) * THICKNESS
    outside = 0.01 + np.array([-3.0, 3.0]) * THICKNESS
    assert slabs(np.outer(inside, normal), 0.0).all()
    assert not slabs(np.outer(outside, normal), 0.0).any()


def test_off_resonance_moves_an_isochromats_slab_as_it_moves_the_pulses(tmp_path):
    rf, gz = _slice()
    seq = _written(tmp_path / "slice.seq", (rf, gz), _readout())
    slabs = virtual.excited(seq)
    shift = 2.0 * THICKNESS * gz.amplitude

    centre = np.zeros((1, 3))
    moved = np.array([[0.0, 0.0, -2.0 * THICKNESS]])

    assert slabs(centre, 0.0).all()
    assert not slabs(centre, shift).any()
    assert slabs(moved, shift).all()


def test_each_slice_a_scan_excites_is_a_slab_of_its_own(tmp_path):
    offsets = (-0.02, 0.0, 0.03)
    blocks = []
    for offset in offsets:
        blocks += [_slice(offset), _readout()]
    seq = _written(tmp_path / "slices.seq", *blocks)

    slabs = virtual.excited(seq)

    assert len(slabs.bounds) == len(offsets)
    assert len(set(slabs.gradients)) == 1
    normal = np.array([0.0, 0.0, 1.0])
    for offset in offsets:
        assert slabs(np.outer([offset], normal), 0.0).all()
    between = np.outer([0.5 * (offsets[1] + offsets[2])], normal)
    assert not slabs(between, 0.0).any()


def test_an_excitation_without_a_gradient_excites_every_isochromat(tmp_path):
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=5e-4, system=SYSTEM, use="excitation"
    )
    seq = _written(tmp_path / "hard.seq", (pulse,), _readout())

    assert virtual.excited(seq) is None


def test_an_excitation_under_a_changing_gradient_excites_every_isochromat(tmp_path):
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=5e-4, system=SYSTEM, use="excitation"
    )
    ramp = pp.make_trapezoid(
        "z", amplitude=2e5, rise_time=4e-4, flat_time=4e-4, system=SYSTEM
    )
    seq = _written(tmp_path / "ramp.seq", (pulse, ramp), _readout())

    assert virtual.excited(seq) is None


def test_pulses_other_than_excitations_leave_the_slabs_to_the_excitations(tmp_path):
    rf, gz = _slice()
    inversion = pp.make_block_pulse(
        math.pi, duration=5e-4, system=SYSTEM, use="inversion"
    )
    seq = _written(tmp_path / "inverted.seq", (inversion,), (rf, gz), _readout())

    slabs = virtual.excited(seq)

    assert slabs is not None
    assert len(slabs.bounds) == 1


def test_a_phantom_keeps_the_isochromats_its_region_answers_for_and_counts_them(
    monkeypatch,
):
    made = []
    monkeypatch.setattr(
        _phantom,
        "Isochromats",
        lambda positions, **fields: made.append((positions, fields)),
    )
    tissue = virtual.Phantom(
        [
            virtual.Ellipse((0.0, 0.0, 0.0), (0.05, 0.04)),
            virtual.Ellipse((0.01, 0.0, 0.0), (0.01, 0.01), shift_ppm=-3.45),
        ]
    )

    def region(positions, frequencies):
        return (positions[:, 0] > 0.0) & (frequencies > -100.0)

    tissue.isochromats(2e-3, field_t=3.0, region=region)

    ((positions, fields),) = made
    assert len(positions) == tissue.count(2e-3, field_t=3.0, region=region)
    assert 0 < len(positions) < tissue.count(2e-3, field_t=3.0)
    assert (positions[:, 0] > 0.0).all()
    assert (fields["off_resonance"] == 0.0).all()
