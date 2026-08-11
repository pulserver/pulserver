"""CUDA integration tests for the private Torch-native CUFINUFFT adapter."""

from __future__ import annotations

import pytest

from pulserver.recon.physics import NonCartesian2D, Toeplitz


torch = pytest.importorskip("torch")
pytest.importorskip("cufinufft")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_public_auto_backend_uses_torch_cufinufft_without_cupy():
    generator = torch.Generator(device="cuda").manual_seed(83)
    trajectory = torch.rand(29, 2, generator=generator, device="cuda") - 0.5
    image = torch.randn(
        2,
        1,
        4,
        4,
        generator=generator,
        dtype=torch.complex64,
        device="cuda",
    )
    physics = NonCartesian2D(
        trajectory,
        (4, 4),
        n_batchs=2,
        viewed_as_real=False,
    )

    kspace = physics.A(image)
    exact = physics.A_adjoint(kspace)
    accelerated = Toeplitz(
        physics,
        cuda_transfer_precision="float32",
    ).A_adjoint_A(image)

    assert physics.native_operator.backend == "cufinufft-torch"
    assert kspace.device.type == "cuda"
    torch.testing.assert_close(accelerated, exact, atol=2e-5, rtol=2e-5)
