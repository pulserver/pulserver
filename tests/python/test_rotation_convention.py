"""What a ``ROTATIONS`` extension means, pinned against the reference toolbox.

A block's gradients are written in a logical frame; the physical vector the
amplifiers play is the extension's quaternion matrix **applied to** that
vector -- ``g_physical = R @ g_logical``. That is the sense of
``mr.rotate3D`` in the reference toolbox, which bakes a rotation into the
gradients at design time and so defines what resolving the extension has to
produce.

Every consumer resolves it: the trajectory core, the canonical-TR waveform
composer the safety analyses run on, and the gradient-continuity check. They
must agree, and a rotation about a single axis cannot tell ``R`` from its
transpose -- the two differ only in the sign of one component -- so the
rotations here are deliberately general.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pulserver.pypulseq as pp

#: General rotations: no axis aligned with a gradient channel, so R and R.T
#: give genuinely different vectors rather than a sign flip.
ROTATIONS = [
    Rotation.from_euler("xyz", [0.3, -0.7, 1.1]),
    Rotation.from_euler("xyz", [-1.2, 0.45, 0.9]),
    Rotation.from_euler("xyz", [2.0, 1.3, -0.6]),
]

AMPLITUDE = 1.0e5


def _constant(vector, duration=2e-4, system=None):
    """One extended trapezoid per axis, holding ``vector`` for ``duration``."""
    vector = np.asarray(vector, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.array([0.0, duration]),
            amplitudes=np.array([vector[index], vector[index]]),
            system=system,
        )
        for index, axis in enumerate(("x", "y", "z"))
    ]


def _ramp(start, stop, duration=1e-4, system=None):
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    return [
        pp.make_extended_trapezoid(
            channel=axis,
            times=np.array([0.0, duration]),
            amplitudes=np.array([start[index], stop[index]]),
            system=system,
        )
        for index, axis in enumerate(("x", "y", "z"))
    ]


@pytest.mark.parametrize("rotation", ROTATIONS, ids=range(len(ROTATIONS)))
def test_a_rotated_gradient_plays_along_the_rotated_direction(rotation):
    """``R @ logical`` is what the trajectory reports, as ``mr.rotate3D`` defines."""
    system = pp.Opts()
    logical = np.array([1.0, 0.0, 0.0])
    expected = rotation.as_matrix() @ logical

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=AMPLITUDE,
        flat_time=4e-4,
        rise_time=1e-4,
        fall_time=1e-4,
        system=system,
    )
    adc = pp.make_adc(num_samples=64, duration=4e-4, delay=1e-4, system=system)
    seq = pp.Sequence(system)
    seq.add_block(gx, adc, pp.make_rotation(rotation))
    seq.add_block(pp.make_delay(1e-3))

    k = np.asarray(seq.calculate_kspace(dense=False)[0])[:, -1]
    np.testing.assert_allclose(k / np.linalg.norm(k), expected, atol=1e-4)


@pytest.mark.parametrize("rotation", ROTATIONS, ids=range(len(ROTATIONS)))
def test_blocks_meeting_in_the_physical_frame_are_continuous(rotation):
    """Two blocks holding one physical vector do not step, whatever frame states it.

    The second block states the same physical gradient in its own rotated
    frame, so the amplifiers see no change at the boundary. A checker that
    resolved the rotation the wrong way round would see the difference
    between ``R @ v`` and ``R.T @ v`` and refuse a sequence that is in fact
    continuous -- which is how a rotation-encoded readout like ZTE, where
    every view hands its gradient to the next, gets rejected.
    """
    system = pp.Opts()
    matrix = rotation.as_matrix()
    physical = np.array([AMPLITUDE, 0.0, 0.0])
    logical = matrix.T @ physical
    event = pp.make_rotation(rotation)

    seq = pp.Sequence(system)
    seq.add_block(*_ramp(np.zeros(3), physical, system=system))
    seq.add_block(*_constant(physical, system=system))
    seq.add_block(*_constant(logical, system=system), event)
    seq.add_block(*_ramp(logical, np.zeros(3), system=system), event)

    is_ok, message = seq.check_gradient_continuity()
    assert is_ok, message


@pytest.mark.parametrize("rotation", ROTATIONS, ids=range(len(ROTATIONS)))
def test_the_composed_tr_waveform_is_the_rotated_gradient(rotation):
    """The window the safety analyses run on resolves the rotation the same way.

    Checked against ``R @ logical`` computed here rather than against the
    oracle, so the composer and its reference cannot drift together.
    """
    from pulserver._ext.pulseg import _PulseqCollection, _get_tr_waveforms

    system = pp.Opts()
    expected = rotation.as_matrix() @ np.array([1.0, 0.0, 0.0])

    gx = pp.make_trapezoid(
        channel="x",
        amplitude=AMPLITUDE,
        flat_time=4e-4,
        rise_time=1e-4,
        fall_time=1e-4,
        system=system,
    )
    adc = pp.make_adc(num_samples=64, duration=4e-4, delay=1e-4, system=system)
    seq = pp.Sequence(system)
    seq.add_block(gx, adc, pp.make_rotation(rotation))
    seq.add_block(pp.make_delay(1e-3))

    payload = seq._prepared_for_write(
        check_timing=False, check_gradients=False
    )._to_text(create_signature=False)
    collection = _PulseqCollection(
        [payload],
        float(system.gamma),
        float(system.B0),
        float(system.max_grad),
        float(system.max_slew),
        float(system.rf_raster_time),
        float(system.grad_raster_time),
        float(system.adc_raster_time),
        float(system.block_duration_raster),
        True,
        [],
    )
    window = _get_tr_waveforms(collection, 0, 1, 0, False)
    amplitudes = np.array(
        [np.asarray(window[axis]["amplitude"]) for axis in ("gx", "gy", "gz")]
    )
    peak = amplitudes[:, int(np.argmax(np.abs(amplitudes).sum(axis=0)))]
    np.testing.assert_allclose(peak / np.linalg.norm(peak), expected, atol=1e-4)
