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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_either_sensitivity_layout_reaches_the_plan_as_one_map_per_coil():
    """A leading batch axis on the maps is the same operator without it.

    Calibration hands back ``(batch, coil, *grid)`` and a plugin that
    reconstructs one slice at a time hands back ``(coil, *grid)``; both
    describe one map per coil, and both backends take them.
    """
    maps = torch.ones(4, 8, 8, dtype=torch.complex64) / 2

    for backend in ("finufft", "cufinufft-torch"):
        trajectory = torch.rand(37, 2).float() - 0.5
        image = torch.zeros(1, 8, 8, dtype=torch.complex64)
        plain = NonCartesian2D(trajectory, (8, 8), coil_maps=maps, backend=backend)
        batched = NonCartesian2D(
            trajectory, (8, 8), coil_maps=maps[None], backend=backend
        )

        assert plain.native_operator.smaps.shape == (4, 8, 8)
        assert batched.native_operator.smaps.shape == (4, 8, 8)
        torch.testing.assert_close(batched.A(image), plain.A(image))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_the_adapter_describes_itself():
    trajectory = torch.rand(11, 2).float() - 0.5
    physics = NonCartesian2D(trajectory, (4, 4))

    assert "n_coils: 1" in repr(physics.native_operator)
