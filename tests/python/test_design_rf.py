"""Contracts for the consolidated pulserver private RF package."""

from __future__ import annotations

import numpy as np
import pypulseq as pp
import pytest
from pulserver.pypulseq import _readout as readout
from pulserver.pypulseq import _rf as rf
from scipy.spatial.transform import Rotation


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def test_hard_and_slr_aliases_are_the_expected_factories(system):
    assert rf.make_slr_pulse is rf.make_sigpy_pulse
    hard = rf.make_hard_pulse(np.pi / 2, system=system)
    assert isinstance(hard, rf.RfPulse)
    assert hard.rf.type == "rf"
    pulse = rf.make_slr_pulse(
        np.pi / 2,
        duration=2e-3,
        dwell=10e-6,
        pulse_type="ex",
        system=system,
        use="excitation",
    )
    assert pulse.rf.type == "rf"
    assert pulse.rf.use == "excitation"
    assert pulse.rf.signal.size == 200
    assert np.iscomplexobj(pulse.rf.signal)
    achieved_flip = 2.0 * np.pi * abs(np.sum(pulse.rf.signal) * 10e-6)
    assert achieved_flip == pytest.approx(np.pi / 2)


@pytest.mark.parametrize("pulse_type", ["st", "ex", "se", "inv", "sat"])
def test_slr_supports_all_pauly_pulse_types(system, pulse_type):
    pulse = rf.make_slr_pulse(
        np.pi / 2,
        duration=1.28e-3,
        dwell=10e-6,
        pulse_type=pulse_type,
        system=system,
    )
    assert pulse.rf.signal.shape == (128,)
    assert np.all(np.isfinite(pulse.rf.signal))


def test_slice_selective_slr_returns_selection_and_rephasing_lobes(system):
    pulse = rf.make_slice_selective_pulse(
        np.pi / 2,
        5e-3,
        duration=3e-3,
        dwell=10e-6,
        system=system,
    )
    select = pulse.gradients[0]
    rephase = pulse.rephasers[0]
    assert pulse.rf.use == "excitation"
    assert select.channel == rephase.channel == "z"
    assert rephase.area == pytest.approx(-0.5 * select.area)


def test_frequency_selective_uses_bandwidth_as_the_intuitive_duration_knob(system):
    pulse = rf.make_frequency_selective_pulse(
        np.pi / 2,
        500.0,
        freq_offset=-440.0,
        time_bw_product=2.0,
        system=system,
    )
    assert pp.calc_duration(pulse.rf) == pytest.approx(4e-3)
    assert pulse.rf.freq_offset == pytest.approx(-440.0)


def test_refocusing_is_nominal_pi_and_supports_nonselective_and_selective(system):
    nonselective = rf.make_refocusing_pulse(duration=1.28e-3, dwell=10e-6, system=system)
    selective = rf.make_refocusing_pulse(
        slice_thickness=5e-3,
        duration=3e-3,
        dwell=10e-6,
        system=system,
    )
    assert nonselective.rf.use == "refocusing"
    assert nonselective.rf.phase_offset == pytest.approx(np.pi / 2)
    assert len(selective.gradients) == len(selective.rephasers) == 1
    assert selective.rf.use == "refocusing"
    assert selective.rephasers[0].channel == "z"


def test_inversion_supports_nonselective_and_selective_slr_and_adiabatic(system):
    slr = rf.make_inversion_pulse(
        adiabatic=False,
        duration=1.28e-3,
        dwell=10e-6,
        system=system,
    )
    selective = rf.make_inversion_pulse(
        adiabatic=False,
        slice_thickness=5e-3,
        duration=3e-3,
        dwell=10e-6,
        system=system,
    )
    adiabatic = rf.make_inversion_pulse(duration=4e-3, dwell=10e-6, system=system)
    assert slr.rf.use == adiabatic.rf.use == "inversion"
    assert len(selective.gradients) == len(selective.rephasers) == 1
    assert selective.rephasers[0].channel == "z"


def test_spsp_uses_slr_sublobes_and_returns_rephaser(system):
    pulse = rf.make_spsp_pulse(
        np.pi / 4,
        5e-3,
        150.0,
        n_subpulses=8,
        system=system,
    )
    assert pulse.rf.use == "excitation"
    assert len(pulse.gradients) == len(pulse.rephasers) == 1
    assert pulse.gradients[0].channel == pulse.rephasers[0].channel == "z"
    assert not pulse.self_refocused
    assert pulse.kspace.shape == (pulse.rf.signal.size, 1)


def _smooth_gradient(samples: int, dimensions: int) -> np.ndarray:
    time = np.linspace(0.0, np.pi, samples)
    columns = [1e-3 * np.sin(time), 0.5e-3 * np.sin(time), 0.25e-3 * np.sin(time)]
    return np.column_stack(columns[:dimensions])


@pytest.mark.parametrize("dimensions", [2, 3])
def test_nd_selective_pulse_consumes_existing_trajectory_and_balances_each_axis(system, dimensions):
    gradient = _smooth_gradient(100, dimensions)
    pulse = rf.make_spatially_selective_pulse(
        np.pi / 12,
        gradient,
        (0.2,) * dimensions,
        (8,) * dimensions,
        selective_size=(0.08,) * dimensions,
        system=system,
    )
    assert pulse.kspace.shape == gradient.shape
    assert len(pulse.gradients) == len(pulse.rephasers) == dimensions
    rephasers = {event.channel: event for event in pulse.rephasers}
    for event in pulse.gradients:
        assert event.area + rephasers[event.channel].area == pytest.approx(0.0, abs=1e-8)


def test_explicit_3d_factory_rejects_2d_trajectory(system):
    with pytest.raises(ValueError, match="samples, 3"):
        rf.make_3d_selective_pulse(np.pi / 12, _smooth_gradient(40, 2), (0.2,) * 3, (8,) * 3, system=system)


def test_automatic_2d_factory_reuses_arbgrad_spiral(system):
    pulse = rf.make_2d_selective_pulse(
        np.pi / 12,
        0.16,
        16,
        selective_size=0.06,
        system=system,
    )
    assert pulse.kspace.shape[1] == 2
    assert len(pulse.gradients) == len(pulse.rephasers) == 2


def test_refocusing_output_is_directly_accepted_by_fse_constructor(system):
    module = rf.make_refocusing_pulse(
        slice_thickness=5e-3,
        duration=3e-3,
        dwell=10e-6,
        system=system,
    )
    train = readout.Fse2D(system, (0.22, 0.22), (32, 32), 2, module.rf, module.gradients[0])
    train.set_state(lin_idx=np.array([10, 11]))
    assert len(train) > 0


def test_fat_saturation_scopes_and_resets_fov_exemption_labels(system):
    pulse = rf.make_fat_saturation_pulse(b0=3.0, voxel_size=3e-3, system=system)
    assert pulse.rf.freq_offset == pytest.approx(rf.fat_shift_hz(3.0, system.gamma))
    assert [event.label for event in pulse.labels_on] == ["NOPOS", "NOROT"]
    assert [event.value for event in pulse.labels_on] == [1, 1]
    assert [event.value for event in pulse.labels_off] == [0, 0]
    assert [(event.label, event.value) for event in pulse.blocks[0][-2:]] == [("NOPOS", 1), ("NOROT", 1)]
    assert [(event.label, event.value) for event in pulse.blocks[-1][-2:]] == [("NOPOS", 0), ("NOROT", 0)]


def test_mt_and_ihmt_offsets(system):
    mt = rf.make_mt_pulse(freq_offset=-1500.0, voxel_size=3e-3, system=system)
    ihmt = rf.make_ihmt_pulse(freq_offset=7000.0, repetitions=4, voxel_size=3e-3, system=system)
    assert mt.rf.freq_offset == -1500.0
    assert ihmt.rf_positive.freq_offset == 7000.0
    assert ihmt.rf_negative.freq_offset == -7000.0
    assert [block[0].freq_offset for block in ihmt.blocks[:-1]] == [7000.0, -7000.0, 7000.0, -7000.0]


def test_bloch_siegert_reports_direct_kbs_integral(system):
    pulse = rf.make_bloch_siegert_pulse(freq_offset=4000.0, peak_b1=500.0, system=system)
    envelope = np.abs(pulse.rf.signal)
    expected = np.sum((2 * np.pi * envelope) ** 2 / (2 * 2 * np.pi * 4000.0)) * 10e-6
    assert pulse.kbs == pytest.approx(expected)
    assert pulse.kbs_per_gauss2 > 0


def test_t2_and_hybrid_t1t2_preps_differ_in_final_tip(system):
    t2 = rf.make_t2prep_pulse(30e-3, adiabatic=False, voxel_size=3e-3, system=system)
    hybrid = rf.make_t1t2_prep_pulse(30e-3, adiabatic=False, voxel_size=3e-3, system=system)
    assert t2.final_tip == "up"
    assert hybrid.final_tip == "down"
    assert (t2.rf90_up.phase_offset - hybrid.rf90_up.phase_offset) % (2 * np.pi) == pytest.approx(np.pi)


def test_adiabatic_t2prep_uses_half_passages_for_both_90s(system):
    prep = rf.make_t2prep_pulse(
        50e-3,
        adiabatic=True,
        half_passage_duration=2e-3,
        refocusing_duration=4e-3,
        voxel_size=3e-3,
        system=system,
    )
    assert prep.rf90_down.signal.size == prep.rf90_up.signal.size
    assert prep.rf90_down.use == prep.rf90_up.use == "preparation"
    assert prep.rf180.use == "refocusing"


def test_diffusion_prep_is_full_scale_rotatable_and_scopes_fov_labels(system):
    pulse = rf.make_diffusion_prep(800.0, voxel_size=3e-3, system=system)
    assert pulse.gradient.channel == "z"
    assert pulse.gradient_scale(200.0) == pytest.approx(0.5)
    assert pulse.gradient_for_b_value(200.0).amplitude == pytest.approx(0.5 * pulse.gradient.amplitude)
    assert [event.label for event in pulse.labels_on] == ["NOPOS", "NOROT"]
    assert [event.value for event in pulse.labels_off] == [0, 0]
    assert [(event.label, event.value) for event in pulse.blocks[0][-2:]] == [("NOPOS", 1), ("NOROT", 1)]
    assert [(event.label, event.value) for event in pulse.blocks[-1][-2:]] == [("NOPOS", 0), ("NOROT", 0)]


def test_rf_modules_are_iterable_indexable_and_materialize_standalone_sequences(system):
    module = rf.make_t2prep_pulse(30e-3, adiabatic=False, voxel_size=3e-3, system=system)
    assert module.num_blocks == len(module)
    assert module[0] == module.blocks[0]
    assert module[:] == module.blocks
    assert all(isinstance(block, tuple) for block in module)

    standalone = module.get()
    assert len(standalone.block_events) == len(module)

    from pulserver.pypulseq import Sequence

    assembled = Sequence(system)
    for block in module:
        assembled.add_block(*block)
    assert len(assembled.block_events) == len(module)


def test_set_state_replaces_offsets_and_scaling_without_mutating_templates(system):
    module = rf.make_ihmt_pulse(freq_offset=7000.0, repetitions=4, voxel_size=3e-3, system=system)
    template_signal = module.rf_positive.signal.copy()
    before = module.blocks

    returned = module.set_state(freq_offset_hz=250.0, phase_offset_rad=0.3, amplitude_scale=0.4)
    current_rf = [event for block in module for event in block if getattr(event, "type", None) == "rf"]

    assert returned is module
    assert module.blocks is not before
    assert [event.freq_offset for event in current_rf] == [7250.0, -6750.0, 7250.0, -6750.0]
    assert all(event.phase_offset == pytest.approx(0.3) for event in current_rf)
    assert np.allclose(current_rf[0].signal, 0.4 * template_signal)
    assert np.array_equal(module.rf_positive.signal, template_signal)

    module.set_state(freq_offset_hz=-100.0, amplitude_scale=0.5)
    replaced_rf = [event for block in module for event in block if getattr(event, "type", None) == "rf"]
    assert replaced_rf[0].freq_offset == pytest.approx(6900.0)
    assert np.allclose(replaced_rf[0].signal, 0.5 * template_signal)


def test_diffusion_state_rotation_is_explicit_despite_norot_scope(system):
    module = rf.make_diffusion_prep(800.0, voxel_size=3e-3, system=system)
    module.set_state(b_value=200.0, rotation=Rotation.from_euler("y", 90.0, degrees=True))

    gradient_blocks = [
        block for block in module if any(getattr(event, "type", None) in ("grad", "trap") for event in block)
    ]
    assert gradient_blocks
    assert all(any(getattr(event, "type", None) == "rot3D" for event in block) for block in gradient_blocks)
    diffusion_lobes = [
        block[0]
        for block in module
        if len([event for event in block if getattr(event, "type", None) in ("grad", "trap")]) == 1
        and getattr(block[0], "channel", None) == "z"
    ]
    assert module.requested_b_value == 200.0
    assert len(diffusion_lobes) == 2
    assert all(event.amplitude == pytest.approx(0.5 * module.gradient.amplitude) for event in diffusion_lobes)
    assert [(event.label, event.value) for event in module.blocks[0][-2:]] == [("NOPOS", 1), ("NOROT", 1)]
    assert module.get().rotation_library.data


def test_set_state_validates_scaling_and_rotation(system):
    module = rf.make_hard_pulse(np.pi / 2, system=system)
    with pytest.raises(ValueError, match="amplitude_scale"):
        module.set_state(amplitude_scale=-1.0)
    with pytest.raises(TypeError, match="rotation"):
        module.set_state(rotation="z")
