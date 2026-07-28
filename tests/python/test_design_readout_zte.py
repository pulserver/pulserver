"""Continuous-gradient basic-ZTE readout contracts."""

from __future__ import annotations

import numpy as np
import pypulseq as pp
import pytest
from pulserver.design import _readout as readout
from pulserver.design import _rf as rf
from pulserver.design import _sampling as sampling
from scipy.spatial.transform import Rotation

OPTS_KW = {"max_grad": 40, "grad_unit": "mT/m", "max_slew": 150, "slew_unit": "T/m/s"}


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


def test_rotation_encoded_zte_designs_one_waveform_and_steps_it_with_block_rotations():
    """A long segment must cost a fixed number of gradient shapes, not one per view.

    Every ZTE transition lands in the same gradient definition (same sample
    count, same delay), and an interpreter holds a bounded number of shapes per
    definition, so a segment that encodes each direction change in its own
    waveform is rejected past a handful of views. Rotation encoding keeps two
    waveforms however long the segment is.
    """
    system = _system()
    views = 24
    module = readout.Zte(
        system,
        (0.22,) * 3,
        (32,) * 3,
        sampling.calc_uniform_angles(views),
        _hard(system).rf,
        tr_s=2e-3,
        rotation_encoded=True,
    )

    # An interior waveform and a closing one, shared by every view.
    assert len(module._view_gradients) == 2
    assert module.num_views == views

    module.set_state(lin_idx=np.arange(views))
    blocks = module.blocks
    assert len(blocks) == views + 1  # ramp-up, then one per view

    # Each view carries its own rotation, and they advance by a constant step.
    quaternions = [
        next(event.rot_quaternion for event in block if getattr(event, "type", None) == "rot3D")
        for block in blocks[1:]
    ]
    step = quaternions[1]  # view 0 sits in the canonical frame, so its rotation is the identity
    for index, got in enumerate(quaternions):
        assert np.allclose(np.abs(got.as_quat() @ (step**index).as_quat()), 1.0)

    # The rotated canonical waveform reproduces the requested directions: view k
    # holds its own direction, then slews to the next one.
    for index in range(views - 1):
        rotation = quaternions[index]
        held = rotation.apply(_gradient_vector(blocks[index + 1], "first"))
        assert np.allclose(held / np.linalg.norm(held), module.directions[index], atol=1e-9)
        arrived = rotation.apply(_gradient_vector(blocks[index + 1], "last"))
        assert np.allclose(arrived / np.linalg.norm(arrived), module.directions[index + 1], atol=1e-9)

    # ...and the segment still ends at zero, so shots stay independent.
    assert np.allclose(_gradient_vector(blocks[-1], "last"), 0.0, atol=1e-9)


def test_rotation_encoded_zte_composes_the_segment_rotation_with_the_view_step():
    system = _system()
    views = 8
    module = readout.Zte(
        system, (0.22,) * 3, (32,) * 3, sampling.calc_uniform_angles(views),
        _hard(system).rf, tr_s=2e-3, rotation_encoded=True,
    )
    segment = Rotation.from_euler("x", 37.0, degrees=True)
    module.set_state(lin_idx=0, rotation=segment)

    quaternions = [
        next(event.rot_quaternion for event in block if getattr(event, "type", None) == "rot3D")
        for block in module.blocks[1:]
    ]
    for index, rotation in enumerate(quaternions):
        placed = rotation.apply(np.array(module.directions[0], copy=True))
        assert np.allclose(placed, segment.apply(np.array(module.directions[index], copy=True)), atol=1e-9)


def test_rotation_encoded_zte_rejects_a_view_order_whose_angular_step_wanders():
    """The condition is a constant step, not a circle -- say which one failed, and by how much."""
    system = _system()
    base, _ = sampling.make_rotated_projection_sampling(
        (32,) * 3, shots=5, views=12, scheme="phyllotaxis", require_fibonacci=False
    )
    with pytest.raises(ValueError, match="constant angle apart"):
        readout.Zte(system, (0.22,) * 3, (32,) * 3, base.flatten(), _hard(system).rf,
                    tr_s=4e-3, rotation_encoded=True)


def test_rotation_encoded_zte_accepts_a_non_circular_constant_step_shell():
    """A pole-to-pole spiral is not a circle, but it steps by a constant angle -- so it encodes.

    This is the case that makes segmented ZTE useful: the shell spans the whole
    polar range the way a phyllotaxis interleaf does, while still costing the
    two gradient shapes a circle would.
    """
    system = _system()
    base, _ = sampling.make_rotated_projection_sampling((32,) * 3, shots=8, views=24, scheme="spiral")
    directions = np.asarray(base.flatten())
    assert np.ptp(directions[:, 2]) > 1.5  # spans pole to pole, so not a circle about z
    assert not np.allclose(directions[:, 1], 0.0)  # nor a planar disk

    module = readout.Zte(system, (0.22,) * 3, (32,) * 3, directions, _hard(system).rf,
                         tr_s=4e-3, rotation_encoded=True)
    assert len(module._view_gradients) == 2

    module.set_state(lin_idx=np.arange(24))
    blocks = module.blocks
    rotations = [
        next(event.rot_quaternion for event in block if getattr(event, "type", None) == "rot3D")
        for block in blocks[1:]
    ]
    # Not a cyclic R**k orbit -- which is exactly what the disk would have been.
    step = rotations[1]
    assert not np.allclose(rotations[2].as_matrix(), (step ** 2).as_matrix())
    # ...yet every view still lands on its intended direction.
    for index, rotation in enumerate(rotations):
        placed = rotation.apply(np.array(directions[0], copy=True))
        assert np.allclose(placed, directions[index], atol=1e-9)


def test_rotated_projection_sampling_shots_are_rigid_copies_stepping_by_a_constant_angle():
    base, rotations = sampling.make_rotated_projection_sampling((32,) * 3, shots=13, views=16)
    assert base.positions.shape == (16, 3)
    assert rotations.shape == (13, 3, 3)
    assert np.allclose(np.linalg.norm(base.positions, axis=1), 1.0)

    spokes = np.einsum("sij,vj->svi", rotations, base.positions)
    assert np.allclose(np.linalg.norm(spokes, axis=-1), 1.0)
    # Congruence is the whole point: pairwise angles inside a shot are invariant.
    reference = base.positions @ base.positions.T
    for shot in spokes:
        assert np.allclose(shot @ shot.T, reference, atol=1e-9)

    for scheme in ("spiral", "disk"):
        positions = sampling.make_rotated_projection_sampling(
            (32,) * 3, shots=6, views=10, scheme=scheme
        )[0].positions
        steps = np.arccos(np.clip(np.sum(positions[:-1] * positions[1:], axis=1), -1.0, 1.0))
        assert np.ptp(steps) < 1e-9, scheme

    disk = sampling.make_rotated_projection_sampling((32,) * 3, shots=6, views=10, scheme="disk")[0]
    assert np.allclose(disk.positions[:, 1], 0.0)  # great circle in xz
    with pytest.raises(ValueError, match="Fibonacci"):
        sampling.make_rotated_projection_sampling((32,) * 3, shots=10, views=8, scheme="phyllotaxis")
    with pytest.raises(ValueError, match="below the"):
        sampling.make_rotated_projection_sampling((32,) * 3, shots=6, views=64, scheme="spiral", step_deg=0.5)


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
