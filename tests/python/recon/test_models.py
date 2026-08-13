"""Tests for native DeepInverse interoperability and MRI tensor adapters."""

from __future__ import annotations

import pytest
import torch
from deepinv.models import ArtifactRemoval, DnCNN, MoDL, RAM, UNet, VarNet

from pulserver.recon import models
from pulserver.recon.learned import (
    ComplexAdapter,
    GradientDataConsistency,
    UnrolledReconstructor,
    as_complex_channels,
    as_real_channels,
)
from pulserver.recon.physics import Cartesian2D


class _Scale(torch.nn.Module):
    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(factor))

    def forward(self, value, *_args, **_kwargs):
        return self.factor * value


class _RecordingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []

    def forward(self, value, *, sigma=None, gain=None):
        self.calls.append(
            (
                value.detach().clone(),
                None if sigma is None else sigma.detach().clone(),
                None if gain is None else gain.detach().clone(),
            )
        )
        if sigma is not None:
            sigma = sigma.reshape(sigma.shape[0], *([1] * (value.ndim - 1)))
            value = value + sigma
        return value if gain is None else value + gain


class _FakeBatchStreaming:
    def __init__(self) -> None:
        self.max_batch_size = None
        self.calls = []

    def wrap_batch_denoiser(self, denoiser, *, max_batch_size):
        self.max_batch_size = max_batch_size

        def execute(value, *, sigma=None, **kwargs):
            self.calls.append((value.shape, sigma, kwargs))
            return denoiser(value, sigma=sigma, **kwargs)

        return execute


def test_models_expose_a_small_native_selection_and_mri_adapter():
    assert set(models.__all__) == {"RAM", "MoDL", "VarNet", "ContextAgnosticDenoiser"}
    assert models.RAM is RAM
    assert models.MoDL is MoDL
    assert models.VarNet is VarNet
    assert not hasattr(models, "UNet")
    assert not hasattr(models, "DnCNN")


def test_context_adapter_aligns_conditions_and_bounds_flattened_batch():
    denoiser = _RecordingDenoiser()
    model = models.ContextAgnosticDenoiser(
        denoiser,
        spatial_ndim=2,
        max_batch_size=5,
    )
    value = torch.arange(2 * 1 * 3 * 2 * 4 * 5, dtype=torch.float32).reshape(
        2, 1, 3, 2, 4, 5
    )
    gain = torch.full_like(value, 2.0)
    sigma = torch.tensor([0.25, 0.75])

    result = model(value, sigma, gain=gain)

    expected = value + gain + sigma.reshape(2, 1, 1, 1, 1, 1)
    torch.testing.assert_close(result, expected)
    assert [call[0].shape[0] for call in denoiser.calls] == [5, 5, 2]
    torch.testing.assert_close(
        torch.cat([call[1] for call in denoiser.calls]),
        sigma.repeat_interleave(6),
    )
    expected_gain = gain.permute(0, 2, 3, 1, 4, 5).reshape(12, 1, 4, 5)
    torch.testing.assert_close(
        torch.cat([call[2] for call in denoiser.calls]), expected_gain
    )


def test_context_adapter_accepts_context_wise_noise_levels():
    denoiser = _RecordingDenoiser()
    model = models.ContextAgnosticDenoiser(denoiser, spatial_ndim=2)
    value = torch.zeros(2, 1, 3, 2, 4, 5)
    sigma = torch.arange(12, dtype=value.dtype).reshape(2, 3, 2)

    result = model(value, sigma)

    torch.testing.assert_close(
        result, sigma[:, None, :, :, None, None].expand_as(value)
    )


def test_context_adapter_applies_two_channel_prior_to_complex_channel_groups():
    denoiser = _RecordingDenoiser()
    model = models.ContextAgnosticDenoiser(
        denoiser,
        spatial_ndim=2,
        channels_per_group=2,
    )
    value = torch.arange(1 * 6 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 6, 3, 4, 5)
    gain = torch.ones(1, 1, 3, 4, 5)

    result = model(value, gain=gain)

    assert len(denoiser.calls) == 1
    assert denoiser.calls[0][0].shape == (9, 2, 4, 5)
    assert denoiser.calls[0][2].shape == (9, 1, 4, 5)
    torch.testing.assert_close(result, value + gain)


def test_context_adapter_rejects_incomplete_channel_group():
    model = models.ContextAgnosticDenoiser(
        _RecordingDenoiser(),
        channels_per_group=2,
    )

    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 5, 4, 4))


def test_context_adapter_streams_the_exact_flattened_inference_batch():
    streaming = _FakeBatchStreaming()
    denoiser = _RecordingDenoiser()
    model = models.ContextAgnosticDenoiser(
        denoiser,
        channels_per_group=2,
        max_batch_size=3,
        streaming=streaming,
    ).eval()
    value = torch.zeros(1, 4, 5, 6, 7)
    sigma = torch.tensor([0.25])

    result = model(value, sigma)

    assert result.shape == value.shape
    assert streaming.max_batch_size == 3
    assert streaming.calls[0][0] == (10, 2, 6, 7)
    torch.testing.assert_close(
        streaming.calls[0][1],
        torch.full((10,), 0.25),
    )


def test_context_adapter_preserves_gradients_across_chunks():
    model = models.ContextAgnosticDenoiser(
        _Scale(2.0),
        spatial_ndim=2,
        max_batch_size=2,
    )
    value = torch.randn(1, 1, 5, 4, 4, requires_grad=True)

    model(value).sum().backward()

    torch.testing.assert_close(value.grad, torch.full_like(value, 2.0))
    torch.testing.assert_close(
        model.denoiser.factor.grad,
        value.detach().sum(),
    )


def test_complex_channel_conversion_is_invertible_and_differentiable():
    value = torch.tensor(
        [[[[1.0 + 2.0j]], [[3.0 + 4.0j]]]],
        requires_grad=True,
    )

    packed = as_real_channels(value)
    restored = as_complex_channels(packed)

    assert packed.shape == (1, 4, 1, 1)
    torch.testing.assert_close(
        packed[:, :, 0, 0],
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(restored, value)
    restored.abs().sum().backward()
    assert value.grad is not None


def test_complex_adapter_preserves_native_complex_layout():
    model = ComplexAdapter(_Scale(2.0))
    value = torch.randn(2, 5, 4, 4, 4, dtype=torch.complex64)

    result = model(value)

    assert result.shape == value.shape
    assert result.is_complex()
    torch.testing.assert_close(result, 2.0 * value)


@pytest.mark.parametrize("architecture", ["artifact-removal", "modl", "varnet"])
def test_native_reconstructors_accept_pulserver_cartesian_physics(architecture):
    denoiser = DnCNN(
        in_channels=2,
        out_channels=2,
        depth=3,
        nf=4,
        pretrained=None,
    )
    if architecture == "artifact-removal":
        model = ArtifactRemoval(denoiser)
    elif architecture == "modl":
        model = MoDL(denoiser, num_iter=2)
    else:
        model = VarNet(denoiser, num_cascades=2, mode="varnet")
    shape = (8, 8)
    physics = Cartesian2D(
        torch.ones(1, 1, *shape),
        torch.ones(1, 1, *shape, dtype=torch.complex64),
    )
    image = torch.randn(1, 2, *shape)

    result = model(physics.A(image), physics)

    assert result.shape == image.shape


def test_pulserver_unroll_accepts_a_native_deepinverse_denoiser():
    denoiser = UNet(
        in_channels=2,
        out_channels=2,
        channels_per_scale=(4, 8),
        batch_norm=False,
    )
    model = UnrolledReconstructor(
        denoiser,
        iterations=2,
        data_consistency=GradientDataConsistency(
            2,
            stepsize=[0.1, 0.2],
            trainable=False,
        ),
        denoiser_strength=0.05,
    )

    assert model.iterations == 2
    torch.testing.assert_close(
        model.data_consistency.stepsizes,
        torch.tensor([0.1, 0.2]),
    )
