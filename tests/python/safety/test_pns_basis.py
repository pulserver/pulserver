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


@pytest.mark.parametrize("arms", [72, 96, 256])
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


def _hardware():
    from pypulseq.utils.safe_pns_prediction import safe_example_hw

    return safe_example_hw()


def test_a_repetition_of_a_written_out_scan_has_its_own_curve():
    """Past the group cap ``tr=<int>`` is that repetition, played as it stands,
    and ``tr="worst_case"`` is the repetition holding the scan's exact peak
    -- a witness, not an envelope."""
    import numpy as np

    from pulserver.pypulseq import (
        Sequence,
        make_adc,
        make_arbitrary_grad,
        make_block_pulse,
    )

    system = Opts(
        max_grad=50.0,
        grad_unit="mT/m",
        max_slew=350.0,
        slew_unit="T/m/s",
        B0=3.0,
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
        rf_raster_time=2e-6,
    )
    t = np.linspace(0.0, 1.0, 2048)
    taper = 4.0 * t * (1.0 - t)
    rf = make_block_pulse(flip_angle=0.17, duration=200e-6, system=system)
    adc = make_adc(num_samples=2048, dwell=4e-6, system=system)
    seq = Sequence(system)
    arms = 72
    for k in range(arms):
        phase = 2.0 * np.pi * k / arms
        scale = 0.6 * system.max_grad * (0.5 + 0.5 * k / arms)
        gx = scale * np.sin(20 * np.pi * t + phase) * taper
        gy = scale * np.cos(20 * np.pi * t + phase) * taper
        seq.add_block(rf)
        seq.add_block(
            make_arbitrary_grad(channel="x", waveform=gx, system=system),
            make_arbitrary_grad(channel="y", waveform=gy, system=system),
            adc,
        )
    assert seq.num_trs == arms
    first = seq.calculate_pns(_hardware(), do_plots=False, tr=0, compat=False)
    last = seq.calculate_pns(_hardware(), do_plots=False, tr=arms - 1, compat=False)
    assert 0.0 < first.total.max() < last.total.max(), (
        "a stronger arm plays a stronger curve"
    )
    witness = seq.calculate_pns(
        _hardware(), do_plots=False, tr="worst_case", compat=False
    )
    assert witness.total.max() >= last.total.max() * (1.0 - 1e-6), (
        "the witness is the strongest repetition"
    )


def test_a_repetition_is_bounded_by_the_envelope_within_the_cap():
    """Under the cap the worst case is an envelope: no repetition exceeds it."""
    from pathlib import Path

    from pulserver.pypulseq import Sequence

    seq = Sequence(_system())
    seq.read(str(Path(__file__).parent.parent / "fixtures" / "gre_2d.seq"))
    envelope = seq.calculate_pns(
        _hardware(), do_plots=False, tr="worst_case", compat=False
    )
    instance = seq.calculate_pns(_hardware(), do_plots=False, tr=3, compat=False)
    assert instance.total.max() <= envelope.total.max() * (1.0 + 1e-6)
