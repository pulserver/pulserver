"""The excitation and preparation modules.

What these pin is the part a caller cannot check by eye: that the slice really
is rephased, that the slab merge changes how the gradient is delivered without
changing its moment, and that `center` points at the isodelay a TE is measured
from rather than at the start of the pulse.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.design as design
import pulserver.pypulseq as pp

GAMMA_TOLERANCE = 1e-6


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _moment(event) -> float:
    """Zeroth moment of a gradient event, whichever kind it is."""
    if event.type == "trap":
        return float(event.area)
    return float(np.trapezoid(np.asarray(event.waveform), np.asarray(event.tt)))


# ----------------------------------------------------------------------
# Non-selective
# ----------------------------------------------------------------------


def test_hard_pulse_is_one_block_centred_on_its_middle(system):
    excitation = design.NonSelectiveExcitation(system, flip_angle_deg=10.0, duration_s=1e-3)
    assert len(excitation.blocks) == 1
    assert excitation.blocks[0] == (excitation.rf,)
    assert excitation.center == pytest.approx(0.5e-3)
    assert excitation.rf.use == "excitation"


def test_hard_pulse_flip_angle_is_the_envelope_integral(system):
    excitation = design.NonSelectiveExcitation(system, flip_angle_deg=90.0, duration_s=1e-3)
    signal = np.asarray(excitation.rf.signal)
    times = np.asarray(excitation.rf.t)
    flip_deg = np.rad2deg(2 * np.pi * np.abs(np.trapezoid(signal, times)))
    assert flip_deg == pytest.approx(90.0, rel=1e-2)


def test_inversion_is_adiabatic_and_carries_no_gradient(system):
    inversion = design.Inversion(system, duration_s=8e-3)
    assert len(inversion.blocks) == 1
    assert inversion.rf.use == "inversion"
    assert inversion.duration == pytest.approx(8e-3)
    # An adiabatic sweep is not symmetric, so its reference is not the middle.
    assert inversion.center != pytest.approx(4e-3)


# ----------------------------------------------------------------------
# Spatially selective
# ----------------------------------------------------------------------


def test_a_slice_keeps_its_rephaser_in_a_second_block(system):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    assert len(excitation.blocks) == 2
    assert excitation.blocks[0] == (excitation.rf, excitation.gz_select)
    assert excitation.blocks[1] == (excitation.gz_reph,)


def _moment_after(module, since: float, axis: int = 2) -> float:
    """Integral of the module's gradient on ``axis`` from ``since`` to its end.

    Read off the played waveform rather than off the events, so ramps count
    the way the scanner plays them.
    """
    times, amplitudes = np.asarray(module.waveforms_and_times()[0][axis])
    grid = np.unique(np.concatenate([times, [since]]))
    grid = grid[grid >= since]
    return float(np.trapezoid(np.interp(grid, times, amplitudes), grid))


def test_the_rephaser_undoes_the_selection_moment_after_isocentre(system):
    """Half the selection lobe dephases the slice; the rephaser puts it back.

    Measured from the RF isodelay, because that is where the transverse
    magnetisation starts accruing phase, and over the whole module, because
    the rephaser is in a block of its own.
    """
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)
    scale = abs(excitation.gz_select.area)
    assert _moment_after(excitation, excitation.center) == pytest.approx(0.0, abs=1e-3 * scale)


def test_without_rephasing_the_slice_is_left_dephased(system):
    """The counterpart: what the rephaser was cancelling is a real moment."""
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3, rephase=False)
    scale = abs(excitation.gz_select.area)
    assert abs(_moment_after(excitation, excitation.center)) > 0.4 * scale


def test_a_slab_merges_the_rephaser_into_one_gradient(system):
    slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
    assert len(slab.blocks) == 1
    assert slab.blocks[0] == (slab.rf, slab.gz_slab)
    assert slab.gz_slab.type == "grad"


def test_merging_moves_the_rephaser_without_changing_the_net_moment(system):
    """The slab and the slice deliver the same total moment, differently."""
    slice_ = design.SpatialSelectiveExcitation(system, 8.0, 0.12)
    slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)

    separate = _moment(slice_.gz_select) + _moment(slice_.gz_reph)
    assert _moment(slab.gz_slab) == pytest.approx(separate, rel=1e-6)


def test_a_slab_without_rephasing_is_the_selection_lobe_alone(system):
    """Nothing to concatenate, so the trapezoid is handed over as it stands."""
    rephased = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True)
    slab = design.SpatialSelectiveExcitation(system, 8.0, 0.12, is_slab=True, rephase=False)

    assert len(slab.blocks) == 1
    assert slab.gz_slab.type == "trap"
    assert _moment(slab.gz_slab) > _moment(rephased.gz_slab)


def test_a_slice_without_rephasing_drops_the_second_block(system):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3, rephase=False)
    assert len(excitation.blocks) == 1
    assert not hasattr(excitation.events, "gz_reph")


def test_center_is_the_isodelay_not_the_start_of_the_envelope(system):
    """The pulse sits on the gradient's flat top, so its delay counts."""
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3, duration_s=3e-3)
    assert excitation.rf.delay > 0
    assert excitation.center == pytest.approx(excitation.rf.delay + excitation.rf.center)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_the_selection_axis_is_honoured(system, axis):
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3, axis=axis)
    assert excitation.gz_select.channel == axis
    assert excitation.gz_reph.channel == axis


def test_a_thicker_slab_needs_a_weaker_gradient(system):
    """Bandwidth is fixed by the pulse, so thickness sets the amplitude."""
    thin = design.SpatialSelectiveExcitation(system, 15.0, 3e-3)
    thick = design.SpatialSelectiveExcitation(system, 15.0, 12e-3)
    assert abs(thick.gz_select.amplitude) == pytest.approx(
        abs(thin.gz_select.amplitude) / 4.0, rel=1e-6
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"thickness_m": 0.0}, "thickness_m must be positive"),
        ({"duration_s": -1e-3}, "duration_s must be positive"),
        ({"axis": "w"}, "axis must be one of"),
    ],
)
def test_an_impossible_excitation_is_refused(system, kwargs, message):
    arguments = {"flip_angle_deg": 15.0, "thickness_m": 5e-3, **kwargs}
    with pytest.raises(ValueError, match=message):
        design.SpatialSelectiveExcitation(system, **arguments)


# ----------------------------------------------------------------------
# Preparation
# ----------------------------------------------------------------------


def test_inversion_preparation_is_a_pulse_and_a_crusher(system):
    prep = design.InversionPreparation(system, duration_s=8e-3)
    assert len(prep.blocks) == 2
    assert prep.blocks[0] == (prep.rf,)
    assert prep.blocks[1] == (prep.gcrush,)


def test_the_crusher_winds_the_cycles_it_was_asked_for(system):
    """Cycles across a voxel is an area: `cycles / voxel_size`."""
    prep = design.InversionPreparation(system, spoiling_cycles=6.0, voxel_size_m=1e-3)
    assert _moment(prep.gcrush) == pytest.approx(6.0 / 1e-3, rel=1e-3)


def test_labels_get_one_slot_each_on_the_inversion_block(system):
    prep = design.InversionPreparation(system, labels=("REP", "SET"))
    assert [event.label for event in prep.prep_labels] == ["REP", "SET"]
    assert prep.blocks[0] == (prep.rf, *prep.prep_labels)


def test_one_label_is_still_a_list(system):
    """Registered explicitly, so indexing does not depend on how many there are."""
    prep = design.InversionPreparation(system, labels=("REP",))
    assert isinstance(prep.prep_labels, list)
    assert len(prep.prep_labels) == 1


def test_no_labels_is_an_empty_list(system):
    prep = design.InversionPreparation(system)
    assert prep.prep_labels == []


def test_writing_a_label_shows_through_the_block(system):
    prep = design.InversionPreparation(system, labels=("REP",))
    prep.prep_labels[0].value = 7
    assert prep.blocks[0][1].value == 7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"voxel_size_m": 0.0}, "voxel_size_m must be positive"),
        ({"spoiling_cycles": -1.0}, "spoiling_cycles must be >= 0"),
        ({"axis": "w"}, "axis must be one of"),
    ],
)
def test_an_impossible_preparation_is_refused(system, kwargs, message):
    with pytest.raises(ValueError, match=message):
        design.InversionPreparation(system, **kwargs)


# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------


def test_modules_compose_into_a_plain_pypulseq_loop(system):
    """The whole authoring style, in six lines: read events, write add_block."""
    prep = design.InversionPreparation(system, duration_s=8e-3)
    excitation = design.SpatialSelectiveExcitation(system, 15.0, 5e-3)

    seq = pp.Sequence(system)
    for block in prep.blocks:
        seq.add_block(*block)
    for shot in range(3):
        excitation.rf.phase_offset = 0.5 * shot
        for block in excitation.blocks:
            seq.add_block(*block)

    assert seq.num_blocks == len(prep.blocks) + 3 * len(excitation.blocks)
    assert seq.check_timing()[0]


def test_sim_rf_answers_for_the_published_pulse(system):
    excitation = design.SpatialSelectiveExcitation(system, 90.0, 5e-3, pulse_type="ex")
    response = excitation.sim_rf(compat=False)
    assert response.mz_z.shape == response.frequency.shape
    # On resonance a 90 leaves essentially nothing along z.
    on_resonance = int(np.argmin(np.abs(response.frequency)))
    assert abs(response.mz_z[on_resonance]) < 0.2
