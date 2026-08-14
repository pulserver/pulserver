"""Within-shot EPI orderings.

The ordering is checked against the thing it is supposed to produce: the CAIPI
scheme against the lattice :func:`make_caipirinha_mask` defines, and the
partition blips against the two values Stirnberg's appendix predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp


def tile(ny, nz, ry, rz, shift, segments):
    """Every shot of a full CAIPI acquisition, as a mask and a sample count."""
    etl = ny // (segments * ry)
    order = pp.calc_epi_order(
        etl,
        scheme="caipi",
        acceleration=ry,
        segments=segments,
        partition_acceleration=rz,
        caipi_shift=shift,
    )
    mask = np.zeros((ny, nz), dtype=bool)
    acquired = 0
    for residue in range(segments):
        for family in range(nz // rz):
            mask[
                (residue * ry + order[:, 0]) % ny,
                (family * rz + residue * shift + order[:, 1]) % nz,
            ] = True
            acquired += etl
    return mask, acquired


# ----------------------------------------------------------------------
# Shape and the shot origin
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"scheme": "caipi", "partition_acceleration": 4},
        {"scheme": "zigzag", "acceleration": 4, "extent": 12},
    ],
)
def test_every_scheme_starts_at_the_shot_origin(kwargs):
    """The pattern says where a shot goes *from* its origin, so it starts at zero."""
    order = pp.calc_epi_order(7, **kwargs)
    assert order.shape == (7, 2)
    assert np.issubdtype(order.dtype, np.integer)
    assert tuple(order[0]) == (0, 0)


def test_a_linear_train_steps_by_the_segmented_blip():
    order = pp.calc_epi_order(5, acceleration=3, segments=2)
    assert list(order[:, 0]) == [0, 6, 12, 18, 24]
    assert not order[:, 1].any()


def test_segmenting_widens_the_blip_without_moving_the_lattice():
    """S shots of etl/S lines visit exactly what one shot of etl visited."""
    whole = pp.calc_epi_order(12, acceleration=2)[:, 0]
    parts = [
        residue * 2 + pp.calc_epi_order(4, acceleration=2, segments=3)[:, 0]
        for residue in range(3)
    ]
    assert sorted(np.concatenate(parts)) == sorted(whole)


# ----------------------------------------------------------------------
# CAIPI
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ny", "nz", "ry", "rz", "shift", "segments"),
    [
        (32, 8, 2, 2, 1, 1),
        (32, 8, 2, 2, 1, 2),
        (32, 16, 1, 8, 2, 2),
        (54, 6, 3, 2, 1, 3),
        (64, 8, 2, 4, 1, 4),
        (60, 10, 2, 5, 2, 3),
    ],
)
def test_a_caipi_train_tiles_its_lattice_exactly(ny, nz, ry, rz, shift, segments):
    """Every point of the CAIPI mask, once each -- no gap and no repeat."""
    mask, acquired = tile(ny, nz, ry, rz, shift, segments)
    expected = pp.make_caipirinha_mask((ny, nz), ry, rz, delta=shift)
    assert np.array_equal(mask, expected)
    assert acquired == int(expected.sum())


@pytest.mark.parametrize(
    ("rz", "shift", "segments"),
    [(4, 1, 1), (4, 1, 2), (6, 2, 1), (5, 2, 3), (3, 1, 2)],
)
def test_the_partition_blips_take_the_two_values_the_pattern_allows(rz, shift, segments):
    """``b1 = (S dz) mod Rz`` up, ``b2 = (Rz - b1) mod Rz`` down, and nothing else."""
    order = pp.calc_epi_order(
        16,
        scheme="caipi",
        partition_acceleration=rz,
        caipi_shift=shift,
        segments=segments,
    )
    up = (segments * shift) % rz
    down = (rz - up) % rz
    assert set(np.unique(np.diff(order[:, 1]))) <= {up, -down}


def test_a_train_segmented_by_the_blip_cycle_needs_no_partition_blips():
    """``S = n1`` is shot-selective CAIPI: the partition never moves."""
    order = pp.calc_epi_order(
        8, scheme="caipi", partition_acceleration=4, caipi_shift=1, segments=4
    )
    assert not order[:, 1].any()


def test_a_caipi_train_stays_inside_its_partition_cycle():
    """Bounded blips are the point: the pattern wraps rather than climbing."""
    order = pp.calc_epi_order(
        32, scheme="caipi", partition_acceleration=5, caipi_shift=2
    )
    assert order[:, 1].min() == 0
    assert order[:, 1].max() == 4


# ----------------------------------------------------------------------
# Zigzag
# ----------------------------------------------------------------------


def test_a_zigzag_turns_inside_its_segment():
    order = pp.calc_epi_order(20, scheme="zigzag", acceleration=4, extent=12)
    assert order[:, 0].min() == 0
    assert order[:, 0].max() == 12
    assert not order[:, 1].any()


def test_a_zigzag_never_steps_further_than_one_blip():
    """Short blips are what keep the echo spacing short."""
    order = pp.calc_epi_order(40, scheme="zigzag", acceleration=6, extent=24)
    assert np.abs(np.diff(order[:, 0])).max() == 6


def test_the_return_pass_samples_between_the_outward_one():
    """Half a blip of offset is what makes adjacent passes complementary."""
    order = pp.calc_epi_order(7, scheme="zigzag", acceleration=4, extent=12)
    outward, inward = order[:4, 0], order[4:, 0]
    assert list(outward) == [0, 4, 8, 12]
    assert list(inward) == [10, 6, 2]


def test_a_zigzag_revisits_the_same_lines_at_later_echoes():
    """Sampling one line at many echo times is the whole point of the traversal."""
    order = pp.calc_epi_order(21, scheme="zigzag", acceleration=4, extent=12)
    visits = np.bincount(order[:, 0])
    assert visits.max() > 1
    assert len(np.unique(order[:, 0])) == 7


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match="scheme must be"):
        pp.calc_epi_order(8, scheme="spiral")


def test_a_zigzag_without_a_segment_to_turn_in_is_refused():
    with pytest.raises(ValueError, match="extent"):
        pp.calc_epi_order(8, scheme="zigzag")


def test_an_extent_given_to_a_scheme_that_crosses_k_space_is_refused():
    with pytest.raises(ValueError, match="extent"):
        pp.calc_epi_order(8, extent=12)


@pytest.mark.parametrize("kwargs", [{"etl": 0}, {"acceleration": 0}, {"segments": -1}])
def test_a_count_out_of_range_is_refused(kwargs):
    etl = kwargs.pop("etl", 8)
    with pytest.raises(ValueError, match="positive"):
        pp.calc_epi_order(etl, **kwargs)


# --------------------------------------------------------------------------
# Uniform undersampling, an ACS block, and partial Fourier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("partial_fourier", [1.0, 0.75, 0.6])
def test_partial_fourier_keeps_the_far_side_of_k_space(partial_fourier):
    """The centre stays in; conjugate symmetry covers what is dropped."""
    lines = pp.calc_sampled_lines(32, 1, 0, partial_fourier=partial_fourier)

    assert lines == list(range(32 - round(partial_fourier * 32), 32))
    assert 16 in lines


def test_partial_fourier_truncates_the_calibration_block_with_everything_else():
    """A view that is not played is not in the list, whoever asked for it."""
    lines = pp.calc_sampled_lines(32, 2, 32, partial_fourier=0.75)

    assert min(lines) == 8
    assert lines == sorted(lines)


def test_partial_fourier_and_acceleration_compose():
    lines = pp.calc_sampled_lines(32, 2, 0, partial_fourier=0.75)

    assert lines == [line for line in range(8, 32) if line % 2 == 0]


@pytest.mark.parametrize("partial_fourier", [0.5, 0.0, 1.5])
def test_a_fraction_that_loses_the_centre_is_refused(partial_fourier):
    with pytest.raises(ValueError, match=r"partial_fourier must be in \(0.5, 1\]"):
        pp.calc_sampled_lines(32, 1, 0, partial_fourier=partial_fourier)
