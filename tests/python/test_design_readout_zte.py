"""Continuous-gradient basic-ZTE readout contracts."""

from __future__ import annotations

import numpy as np
import pypulseq as pp
import pytest
from pulserver.design import _readout as readout
from pulserver.design import _rf as rf
from pulserver.design import _sampling as sampling
from scipy.spatial.transform import Rotation

OPTS_KW = dict(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _system(**kwargs):
    return pp.Opts(**OPTS_KW, **kwargs)


def _hard(system, duration=20e-6):
    return rf.make_hard_pulse(np.deg2rad(4.0), duration=duration, system=system, use="excitation")


def _gradient_vector(block, field: str) -> np.ndarray:
    values = {event.channel: getattr(event, field) for event in block if getattr(event, "type", None) == "grad"}
    return np.array([values.get(axis, 0.0) for axis in ("x", "y", "z")])


def test_zte_accepts_caller_rf_and_keeps_gradient_live_between_ordered_views():
    system = _system()
    excitation = _hard(system)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    module = readout.Zte(system, 0.22, 64, directions, excitation.rf, tr_s=2e-3)
    module.set_state(lin_idx=np.arange(len(directions)), phase_offset_rad=np.arange(len(directions)) * 0.1)

    assert len(module) == module.num_views + 1  # one ramp-up + one block per acquired view
    assert _gradient_vector(module[0], "first") == pytest.approx(np.zeros(3))
    assert _gradient_vector(module[0], "last") == pytest.approx(
        module.gradient_amplitude * module.directions[0]
    )

    for view in range(module.num_views):
        block = module[view + 1]
        start = _gradient_vector(block, "first")
        end = _gradient_vector(block, "last")
        expected_end = (
            module.gradient_amplitude * module.directions[view + 1]
            if view + 1 < module.num_views
            else np.zeros(3)
        )
        assert start == pytest.approx(module.gradient_amplitude * module.directions[view])
        assert end == pytest.approx(expected_end)
        if view + 1 < module.num_views:
            assert np.linalg.norm(end) == pytest.approx(module.gradient_amplitude)

        gradients = [event for event in block if getattr(event, "type", None) == "grad"]
        assert gradients
        # Every direction update begins only after the RF/ADC plateau.
        assert all(event.tt[1] == pytest.approx(module.view_duration_s) for event in gradients)
        rf_event = next(event for event in block if getattr(event, "type", None) == "rf")
        adc_event = next(event for event in block if getattr(event, "type", None) == "adc")
        assert rf_event.phase_offset == pytest.approx(view * 0.1)
        assert adc_event.phase_offset == pytest.approx(rf_event.phase_offset)
        assert adc_event.delay + adc_event.num_samples * adc_event.dwell <= module.view_duration_s + 1e-12

    # Dynamic phase state is applied to copies, never the caller's RF event.
    assert excitation.rf.phase_offset == pytest.approx(0.0)


def test_zte_dead_time_omits_integer_center_samples_without_rescaling_the_grid():
    system = _system(rf_dead_time=20e-6, rf_ringdown_time=10e-6, adc_dead_time=10e-6)
    module = readout.Zte(system, 0.22, 64, [0.0], _hard(system).rf)
    module.set_state()

    assert module.num_missing_samples == 2
    assert module.n_samples == module.nominal_samples - module.num_missing_samples
    assert module.gap_s == pytest.approx(module.num_missing_samples * module.adc_dwell_s)

    seq = pp.Sequence(system)
    for block in module:
        seq.add_block(*block)
    assert seq.check_timing()[0]
    k_adc = seq.calculate_kspace()[0]
    expected_first = (module.num_missing_samples + 0.5) * module.delta_k
    expected_last = (module.nominal_samples - 0.5) * module.delta_k
    assert k_adc[:, 0] == pytest.approx(module.directions[0] * expected_first)
    assert k_adc[:, -1] == pytest.approx(module.directions[0] * expected_last)


def test_zte_accepts_planar_disk_order_and_rotates_the_complete_disk():
    system = _system()
    angles = sampling.calc_uniform_angles(6)
    module = readout.Zte(system, (0.22,) * 3, (32,) * 3, angles, _hard(system).rf, tr_s=2e-3)
    rotation = Rotation.from_euler("x", 90.0, degrees=True)
    module.set_state(lin_idx=10, rotation=rotation)

    assert module.directions.shape == (6, 3)
    assert np.allclose(np.linalg.norm(module.directions, axis=1), 1.0)
    assert np.allclose(module.directions[:, 2], 0.0)  # base disk lies in xy
    rotated_disk = rotation.apply(np.array(module.directions, copy=True))
    assert np.allclose(rotated_disk[:, 1], 0.0, atol=1e-12)  # same disk rotated into xz
    assert all(any(getattr(event, "type", None) == "rot3D" for event in block) for block in module)
    label_values = [
        next(event.value for event in block if getattr(event, "type", None) == "labelset")
        for block in module[1:]
    ]
    assert label_values == [10] * module.num_views
    assert module.get().rotation_library.data


def test_zte_rejects_nonuniform_rf_bandwidth_and_tr_that_cannot_fit_a_transition():
    system = _system()
    with pytest.raises(ValueError, match="too long"):
        readout.Zte(system, 0.22, 64, [0.0], _hard(system, duration=40e-6).rf)

    excitation = _hard(system)
    opposite_views = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="minimum required"):
        readout.Zte(system, 0.22, 64, opposite_views, excitation.rf, tr_s=0.5e-3)


def test_zte_validates_isotropy_view_order_and_state_schedules():
    system = _system()
    excitation = _hard(system)
    with pytest.raises(ValueError, match="isotropic fov"):
        readout.Zte(system, (0.22, 0.24), (64, 64), [0.0], excitation.rf)
    with pytest.raises(ValueError, match="view_order"):
        readout.Zte(system, 0.22, 64, np.empty((0, 3)), excitation.rf)

    module = readout.Zte(system, 0.22, 64, sampling.calc_uniform_angles(4), excitation.rf)
    with pytest.raises(ValueError, match="length 4"):
        module.set_state(phase_offset_rad=[0.0, 0.1])
