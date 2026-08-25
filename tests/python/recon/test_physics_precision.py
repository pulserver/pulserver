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


def frames(count: int = 8) -> np.ndarray:
    """One radial trajectory per frame, float64 as NumPy hands it back."""
    return np.repeat(radial(24, 48)[None], count, 0)


@pytest.mark.parametrize(
    "basis",
    [
        pytest.param(
            np.linalg.qr(np.random.default_rng(0).normal(size=(8, 2)))[0].T,
            id="float64",
        ),
        pytest.param(
            np.linalg.qr(np.random.default_rng(0).normal(size=(8, 2)))[0].T.astype(
                np.complex128
            ),
            id="complex128",
        ),
    ],
)
def test_a_double_precision_basis_does_not_reach_the_kernel(basis):
    """A transfer is single precision whatever precision it was built from.

    Every applier -- the AVX kernel, the Triton kernels, the dense host
    matvec -- reads single-precision words, so a double-precision transfer is
    not a more accurate answer but a differently-interpreted one. A basis is
    double unless it was asked not to be, which is what NumPy hands back from
    a decomposition, so this is the ordinary case and not the exotic one.
    """
    from pulserver.recon import NonCartesian2D, Subspace

    rank = basis.shape[0]
    trajectory = frames()
    coil_maps = np.ones((4, 16, 16), dtype=np.complex128) / 2.0

    def built(toeplitz):
        return Subspace(
            NonCartesian2D(
                trajectory,
                (16, 16),
                coil_maps=coil_maps,
                n_coils=4,
                toeplitz=toeplitz,
            ),
            basis,
        )

    accelerated = built({"compress": False})
    image = torch.randn(1, rank, 16, 16, dtype=torch.complex64)
    applied = accelerated.A_adjoint_A(image)

    # The kernel is built on first use, so it exists only now.
    assert accelerated.toeplitz_kernel.values.dtype in (
        torch.float32,
        torch.complex64,
    )

    exact = built(False)
    reference = exact.A_adjoint(exact.A(image))
    error = (applied - reference).abs().max()
    assert float(error / reference.abs().max()) < 1e-3


def test_a_double_precision_field_model_does_not_reach_the_kernel():
    """The same, for the segment transfers an off-resonance model builds."""
    from pulserver.recon import NonCartesian2D, OffResonance

    trajectory = radial(24, 48)
    coil_maps = np.ones((4, 16, 16), dtype=np.complex128) / 2.0
    field_map = np.zeros((16, 16), dtype=np.float64)
    field_map[:, 8:] = 200.0
    readout_time = np.tile(np.linspace(0, 8e-3, 48, dtype=np.float64), 24)

    def built(toeplitz):
        return OffResonance(
            NonCartesian2D(
                trajectory,
                (16, 16),
                coil_maps=coil_maps,
                n_coils=4,
                toeplitz=toeplitz,
            ),
            field_map,
            readout_time,
        )

    accelerated = built({"compress": False})
    image = torch.randn(1, 1, 16, 16, dtype=torch.complex64)
    applied = accelerated.A_adjoint_A(image)

    # The kernel is built on first use, so it exists only now.
    assert accelerated.toeplitz_kernel.values.dtype in (
        torch.float32,
        torch.complex64,
    )

    exact = built(False)
    reference = exact.A_adjoint(exact.A(image))
    error = (applied - reference).abs().max()
    assert float(error / reference.abs().max()) < 1e-3
