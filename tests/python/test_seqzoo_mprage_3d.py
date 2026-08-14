"""The 3D MPRAGE of the sequence zoo.

What MPRAGE owes beyond a plain 3D gradient echo: an inversion per segment,
the centre view at the requested TI, a recovery that makes the
inversion-to-inversion interval exact, and the ordering machinery shared
with the 3D fast spin echo.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import UIParam, params
from pulserver.seqzoo import mprage_3d

N_Y, N_Z = 8, 4
TI = 200e-3
TR_OUTER = 600e-3


def design(**kwargs):
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_acs_z", 2)
    kwargs.setdefault("views_per_segment", 8)
    kwargs.setdefault("ti", TI)
    kwargs.setdefault("tr_outer", TR_OUTER)
    return mprage_3d.main(n_x=32, n_y=N_Y, n_z=N_Z, slab_thickness=64e-3, **kwargs)


def pulse_centres(seq):
    """Absolute centre time of every pulse, split by use."""
    inversions, excitations = [], []
    t = 0.0
    for index in range(1, seq.num_blocks + 1):
        block = seq.get_block(index)
        if block.rf is not None:
            centre = t + block.rf.delay + block.rf.center
            (inversions if block.rf.use == "i" else excitations).append(centre)
        t += block.block_duration
    return np.asarray(inversions), np.asarray(excitations)


def test_the_sequence_is_valid_pulseq():
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


def test_every_view_is_acquired_once():
    labels = design().evaluate_labels(evolution="adc")
    assert len(set(zip(labels["LIN"].tolist(), labels["PAR"].tolist()))) == N_Y * N_Z


def test_the_centre_view_is_excited_at_the_inversion_time():
    seq = design()
    inversions, excitations = pulse_centres(seq)
    labels = seq.evaluate_labels(evolution="adc")
    pairs = list(zip(labels["LIN"].tolist(), labels["PAR"].tolist()))
    n_center = int(labels["ECO"][pairs.index((N_Y // 2, N_Z // 2))])

    first_segment = excitations[
        (excitations > inversions[0]) & (excitations < inversions[1])
    ]
    measured = first_segment[n_center] - inversions[0]
    assert measured == pytest.approx(TI, abs=2e-5)


def test_the_inversions_repeat_at_the_outer_tr():
    inversions, _ = pulse_centres(design())
    assert np.diff(inversions) == pytest.approx(
        np.full(len(inversions) - 1, TR_OUTER), abs=1e-5
    )


def test_a_too_short_ti_is_refused_by_name():
    with pytest.raises(ValueError, match="TI"):
        design(ti=5e-3)


def test_a_too_short_outer_tr_is_refused_by_name():
    with pytest.raises(ValueError, match="TR"):
        design(tr_outer=100e-3)


def test_the_orderings_place_the_centre_where_they_promise():
    def centre_echo(seq):
        labels = seq.evaluate_labels(evolution="adc")
        pairs = list(zip(labels["LIN"].tolist(), labels["PAR"].tolist()))
        return int(labels["ECO"][pairs.index((N_Y // 2, N_Z // 2))])

    assert centre_echo(design(ordering="linear")) == 4
    assert centre_echo(design(ordering="radial")) == 0
    assert centre_echo(design(ordering="radial_adaptive")) == 4


def test_the_declarations_feed_a_subspace_basis():
    seq = design(ordering="shuffling")
    assert seq.get_definition("TI") == pytest.approx(TI)
    assert seq.get_definition("InnerTR") > 0
    assert seq.get_definition("ViewsPerSegment") == 8
    assert seq.get_definition("ViewOrdering") == "shuffling"


def test_the_protocol_round_trip_is_feasible():
    system = pp.Opts()
    protocol = mprage_3d.PLUGIN.get_default_protocol(system)
    report = mprage_3d.PLUGIN.validate_protocol(system, protocol)
    assert report["valid"] is True, report["info"]

    params.set_protocol_value(protocol, UIParam.PREP_TIME, 1.0)
    refused = mprage_3d.PLUGIN.validate_protocol(system, protocol)
    assert refused["valid"] is False
    assert "TI" in refused["info"]
