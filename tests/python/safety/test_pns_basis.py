"""Sequences with more distinct waveforms than the sweep can group.

The gate prices a groupable scan with the sweep, unchanged. Past the group
cap it prices multiplicity with the occurrence score and settles anything the
bound cannot clear by cold exact assembly -- so a multishot scan written out
arm by arm must reach the same verdict as the same scan encoded with a
``ROTATIONS`` extension, on both sides of the threshold, however many arms it
plays.
"""

import pytest
from pulserver._ext.pulseg import _check_safety
from pulserver.pypulseq import Opts

from .conftest import build_collection

IRNICH_CHRONAXIE_US = 360.0
IRNICH_STIM_THRESHOLD = 4.25e8 / 0.333

#: Far above and far below every corpus peak, so one threshold exercises the
#: instant pass and the other drives the offender assembly end to end.
PASSING_THRESHOLD = 100.0
FAILING_THRESHOLD = 1e-7


def _system() -> Opts:
    return Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=20e-6,
        block_duration_raster=20e-6,
    )


def _stack(tmp_path, arms: int, use_rotation_ext: bool):
    from pulserver.app import mprage_stack_of_spirals3D_sequence

    seq = mprage_stack_of_spirals3D_sequence(
        n_x=128,
        n_z=8,
        slab_thickness=0.192,
        n_arms=arms,
        etl=1,
        design_interleaves=arms,
        ti=0.9,
        tr_outer=3.0,
        use_rotation_ext=use_rotation_ext,
        plot=False,
        write_seq=False,
    )
    seq.declare_tr()
    path = tmp_path / f"stack_{arms}_{int(use_rotation_ext)}.seq"
    path.write_bytes(seq.remove_duplicates()._to_text(create_signature=False))
    return build_collection(path, _system())


def _verdict(collection, threshold: float) -> bool:
    try:
        _check_safety(
            collection,
            [],
            IRNICH_STIM_THRESHOLD,
            IRNICH_CHRONAXIE_US,
            threshold,
            False,
        )
        return True
    except RuntimeError as err:
        assert "PNS" in str(err) or "pns" in str(err), str(err)
        return False


@pytest.mark.parametrize("arms", [96, 256])
def test_written_out_arms_above_the_group_cap_reach_the_rotated_verdict(tmp_path, arms):
    written = _stack(tmp_path, arms, use_rotation_ext=False)
    rotated = _stack(tmp_path, arms, use_rotation_ext=True)

    assert _verdict(written, PASSING_THRESHOLD), (
        "a written-out multishot scan was refused above every real peak"
    )
    assert _verdict(rotated, PASSING_THRESHOLD)

    assert not _verdict(written, FAILING_THRESHOLD), (
        "the exact assembly missed a violation the rotated encoding reports"
    )
    assert not _verdict(rotated, FAILING_THRESHOLD)
