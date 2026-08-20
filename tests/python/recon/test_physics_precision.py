"""Every operator works in single precision, whatever precision it was given.

A trajectory is what a NUFFT plans on and sensitivities are what it applies,
so a double-precision one plans a double-precision transform that then meets
single-precision data. The backend reports that from inside a plan, as a dtype
mismatch far from the call that caused it, which is why it is worth holding
here rather than leaving to whoever hits it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pulserver.recon import Cartesian2D, Cartesian3D, NonCartesian2D, NonCartesian3D


def radial(spokes: int = 8, samples: int = 32) -> np.ndarray:
    """A float64 radial trajectory, which is what plain NumPy hands back."""
    angles = np.linspace(0, np.pi, spokes, endpoint=False)
    radius = np.linspace(-0.5, 0.5, samples)
    return np.stack(
        [np.outer(np.cos(angles), radius), np.outer(np.sin(angles), radius)], -1
    ).reshape(-1, 2)


def maps(coils: int = 4, shape: tuple[int, ...] = (16, 16), dtype=torch.complex64):
    return torch.ones(1, coils, *shape, dtype=dtype) / coils**0.5


@pytest.mark.parametrize(
    "trajectory",
    [
        pytest.param(radial(), id="float64-numpy"),
        pytest.param(radial().astype(np.float32), id="float32-numpy"),
        pytest.param(torch.as_tensor(radial()), id="float64-torch"),
        pytest.param(torch.as_tensor(radial()).float(), id="float32-torch"),
    ],
)
def test_a_noncartesian_operator_takes_a_trajectory_in_either_precision(trajectory):
    physics = NonCartesian2D(trajectory, (16, 16), coil_maps=maps())
    measurement = physics.A(torch.zeros(1, 16, 16, dtype=torch.complex64))

    assert measurement.dtype == torch.complex64
    assert physics.A_adjoint(measurement).dtype == torch.complex64


def test_double_precision_sensitivities_do_not_reach_the_plan():
    physics = NonCartesian2D(radial(), (16, 16), coil_maps=maps(dtype=torch.complex128))
    assert physics.A(torch.zeros(1, 16, 16, dtype=torch.complex64)).dtype == (
        torch.complex64
    )


def test_a_three_dimensional_trajectory_is_taken_in_either_precision():
    rng = np.random.default_rng(0)
    trajectory = rng.uniform(-0.5, 0.5, size=(256, 3))
    physics = NonCartesian3D(trajectory, (8, 8, 8), coil_maps=maps(2, (8, 8, 8)))
    assert physics.A(torch.zeros(1, 8, 8, 8, dtype=torch.complex64)).dtype == (
        torch.complex64
    )


def test_a_cartesian_operator_takes_a_double_precision_mask_and_maps():
    physics = Cartesian2D(
        torch.ones(1, 1, 16, 16, dtype=torch.float64),
        maps(dtype=torch.complex128),
    )
    assert physics.A(torch.zeros(1, 16, 16, dtype=torch.complex64)).dtype == (
        torch.complex64
    )


def test_a_three_dimensional_cartesian_operator_does_the_same():
    physics = Cartesian3D(
        np.ones((1, 1, 8, 8, 8)), maps(2, (8, 8, 8), dtype=torch.complex128)
    )
    assert physics.A(torch.zeros(1, 8, 8, 8, dtype=torch.complex64)).dtype == (
        torch.complex64
    )


def test_single_precision_input_is_not_copied():
    """The coercion is a no-op where there is nothing to coerce, so the common
    case pays nothing for it."""
    from pulserver.recon.physics import _single_precision

    trajectory = torch.as_tensor(radial()).float()
    assert _single_precision(trajectory) is trajectory

    array = radial().astype(np.float32)
    assert _single_precision(array) is array
