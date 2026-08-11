"""Tests for SimpleITK registration and rigid-motion EKF tracking."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("SimpleITK")

from pulserver.recon.motion import (
    RigidMotionEKF,
    RigidMotionEstimate,
    RigidRegistration,
)


def _gaussian(shape):
    coordinates = np.meshgrid(
        *[np.arange(size) for size in shape],
        indexing="ij",
    )
    first = np.exp(
        -sum(
            ((axis - 0.35 * size) / (size / 9)) ** 2
            for axis, size in zip(coordinates, shape, strict=True)
        )
    )
    second = 0.7 * np.exp(
        -sum(
            ((axis - 0.68 * size) / (size / 12)) ** 2
            for axis, size in zip(coordinates, shape, strict=True)
        )
    )
    return (first + second).astype(np.float32)


@pytest.mark.parametrize("shape", [(40, 48), (16, 20, 24)])
def test_rigid_registration_supports_identity_in_2d_and_3d(shape):
    image = _gaussian(shape)
    model = RigidRegistration(
        iterations=10,
        sampling_percentage=1.0,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
    )

    estimate = model(image, image)

    assert estimate.dimension == len(shape)
    np.testing.assert_allclose(estimate.parameters, 0.0, atol=1e-8)
    np.testing.assert_allclose(
        estimate.matrix,
        np.eye(len(shape) + 1),
        atol=1e-8,
    )


def test_rigid_registration_recovers_2d_translation_and_resamples():
    fixed = _gaussian((64, 64))
    moving = np.roll(np.roll(fixed, 3, axis=1), -2, axis=0)
    model = RigidRegistration(
        iterations=80,
        sampling_percentage=1.0,
        shrink_factors=(2, 1),
        smoothing_sigmas=(1, 0),
    )

    estimate = model(fixed, moving)
    corrected = model.resample(moving, estimate, reference=fixed)

    np.testing.assert_allclose(estimate.translation, (3.0, -2.0), atol=0.1)
    assert np.mean((corrected - fixed) ** 2) < 1e-4 * np.mean((moving - fixed) ** 2)


def test_rigid_motion_ekf_uses_prediction_to_initialize_registration():
    measurements = [
        RigidMotionEstimate(np.array([0.0, 1.0, -1.0]), np.zeros(2)),
        RigidMotionEstimate(np.array([0.02, 2.0, -2.0]), np.zeros(2)),
    ]

    class Registration:
        def __init__(self):
            self.initials = []

        def estimate(self, _fixed, _moving, *, initial, spacing):
            del spacing
            self.initials.append(initial)
            return measurements.pop(0)

    registration = Registration()
    tracker = RigidMotionEKF(
        registration,
        dimension=2,
        process_noise=1e-3,
        measurement_noise=1e-2,
    )

    first = tracker.step(object(), object())
    second = tracker.step(object(), object())

    np.testing.assert_allclose(first.parameters, (0.0, 1.0, -1.0))
    assert registration.initials[0] is None
    assert isinstance(registration.initials[1], RigidMotionEstimate)
    assert first.parameters[1] < second.parameters[1] < 2.0
    assert tracker.velocity[1] > 0.0


def test_rigid_motion_ekf_wraps_angle_innovations():
    tracker = RigidMotionEKF(
        RigidRegistration(),
        dimension=2,
        process_noise=1e-3,
        measurement_noise=1e-3,
    )
    tracker.update(
        RigidMotionEstimate(
            np.array([np.pi - 0.01, 0.0, 0.0]),
            np.zeros(2),
        )
    )
    tracker.predict()

    result = tracker.update(
        RigidMotionEstimate(
            np.array([-np.pi + 0.01, 0.0, 0.0]),
            np.zeros(2),
        )
    )

    assert abs(abs(result.angles[0]) - np.pi) < 0.02
