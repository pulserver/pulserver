"""Tests for the DeepInverse adapters: models and algorithm hooks."""

from __future__ import annotations

import deepinv
import pytest
import torch
from deepinv.models import ArtifactRemoval, DnCNN, MoDL, UNet, VarNet

from pulserver.recon import adapters
from pulserver.recon.adapters import (
    Checkpointed,
    ComplexDenoiser,
    NormalEquationL2,
    ScaledAdjoint,
    StepwiseUnroll,
    _as_complex_channels,
    _as_real_channels,
)
from pulserver.recon.physics import Cartesian2D
from pulserver.recon.physics._common import _measurement_to_channels


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


def test_the_models_module_exposes_only_original_code():
    """Native DeepInverse models are imported from deepinv, never re-exported."""
    assert adapters.__all__ == [
        "Checkpointed",
        "ComplexDenoiser",
        "ContextAgnosticDenoiser",
        "NoiseConditioned",
        "NormalEquationL2",
        "ScaledAdjoint",
        "StepwiseUnroll",
    ]
    assert not hasattr(adapters, "UNet")
    assert not hasattr(adapters, "DnCNN")


def test_context_adapter_aligns_conditions_and_bounds_flattened_batch():
    denoiser = _RecordingDenoiser()
    model = adapters.ContextAgnosticDenoiser(
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
    model = adapters.ContextAgnosticDenoiser(denoiser, spatial_ndim=2)
    value = torch.zeros(2, 1, 3, 2, 4, 5)
    sigma = torch.arange(12, dtype=value.dtype).reshape(2, 3, 2)

    result = model(value, sigma)

    torch.testing.assert_close(
        result, sigma[:, None, :, :, None, None].expand_as(value)
    )


def test_context_adapter_applies_two_channel_prior_to_complex_channel_groups():
    denoiser = _RecordingDenoiser()
    model = adapters.ContextAgnosticDenoiser(
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
    model = adapters.ContextAgnosticDenoiser(
        _RecordingDenoiser(),
        channels_per_group=2,
    )

    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 5, 4, 4))


def test_context_adapter_streams_the_exact_flattened_inference_batch():
    streaming = _FakeBatchStreaming()
    denoiser = _RecordingDenoiser()
    model = adapters.ContextAgnosticDenoiser(
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
    model = adapters.ContextAgnosticDenoiser(
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

    packed = _as_real_channels(value)
    restored = _as_complex_channels(packed)

    assert packed.shape == (1, 4, 1, 1)
    torch.testing.assert_close(
        packed[:, :, 0, 0],
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(restored, value)
    restored.abs().sum().backward()
    assert value.grad is not None


def test_complex_adapter_preserves_native_complex_layout():
    model = ComplexDenoiser(_Scale(2.0))
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
    # DeepInverse's own reconstructors work in the two-channel real layout, so
    # this opts the physics back into it from the native-complex default.
    physics = Cartesian2D(
        torch.ones(1, 1, *shape),
        torch.ones(1, 1, *shape, dtype=torch.complex64),
        viewed_as_real=True,
    )
    image = torch.randn(1, 2, *shape)
    measurement = physics.A(image)
    if architecture == "varnet":
        # VarNet is DeepInverse's own class and refines the measurement, so it
        # reads the channel-first layout rather than Pulserver's trailing one.
        measurement = _measurement_to_channels(measurement)
        physics = physics.operator.operator

    result = model(measurement, physics)

    assert result.shape == image.shape


def test_a_native_deepinverse_denoiser_reaches_a_complex_volume():
    denoiser = UNet(
        in_channels=2,
        out_channels=2,
        channels_per_scale=(4, 8),
        batch_norm=False,
    )
    adapted = adapters.ContextAgnosticDenoiser(
        ComplexDenoiser(denoiser),
        spatial_ndim=2,
        channels_per_group=1,
    )
    # a complex 3D+t image: batch, coefficient, frame, slice, and two spatial axes
    image = torch.randn(1, 1, 2, 3, 16, 16, dtype=torch.complex64)

    result = adapted(image, 0.01)

    assert result.shape == image.shape
    assert result.is_complex()


def test_checkpointing_returns_the_models_own_value_and_gradient():
    torch.manual_seed(0)
    network = torch.nn.Conv2d(2, 2, 3, padding=1)
    image = torch.randn(1, 2, 8, 8, requires_grad=True)

    plain = network(image)
    plain.pow(2).sum().backward()
    reference = image.grad.clone()
    image.grad = None

    checkpointed = Checkpointed(network)(image)
    checkpointed.pow(2).sum().backward()

    torch.testing.assert_close(checkpointed, plain)
    torch.testing.assert_close(image.grad, reference)


def test_checkpointing_recomputes_the_model_during_the_backward_pass():
    calls = []

    class _Counting(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value, *_args, **_kwargs):
            calls.append(None)
            return self.scale * value

    model = Checkpointed(_Counting())
    result = model(torch.ones(1, 2, 4, 4, requires_grad=True))
    assert len(calls) == 1

    result.sum().backward()

    assert len(calls) == 2


def test_checkpointing_is_a_passthrough_without_gradients():
    calls = []

    class _Counting(torch.nn.Module):
        def forward(self, value, *_args, **_kwargs):
            calls.append(None)
            return value

    with torch.no_grad():
        Checkpointed(_Counting())(torch.ones(1, 2, 4, 4))

    assert len(calls) == 1


def test_the_noise_level_reaches_a_conditioned_network():
    """A blind denoiser leaves a plug-and-play prior's parameter inert."""
    torch.manual_seed(0)
    conditioned = ComplexDenoiser(
        adapters.NoiseConditioned(torch.nn.Conv2d(3, 2, 3, padding=1))
    )
    image = torch.randn(1, 1, 16, 16, dtype=torch.complex64)

    quiet = conditioned(image, 0.01)
    loud = conditioned(image, 0.20)

    assert not torch.allclose(quiet, loud)
    assert quiet.shape == image.shape and quiet.is_complex()


def test_a_conditioned_network_takes_one_channel_more_than_the_data():
    recorded = []

    class _Recorder(torch.nn.Module):
        def forward(self, value, **_kwargs):
            recorded.append(value)
            return value

    adapters.NoiseConditioned(_Recorder())(torch.zeros(2, 4, 8, 8), 0.05)

    seen = recorded[0]
    assert seen.shape == (2, 5, 8, 8)
    torch.testing.assert_close(seen[:, 4], torch.full((2, 8, 8), 0.05))


def test_a_conditioned_network_takes_a_level_per_batch_entry():
    recorded = []

    class _Recorder(torch.nn.Module):
        def forward(self, value, **_kwargs):
            recorded.append(value)
            return value

    levels = torch.tensor([0.01, 0.2, 0.05])
    adapters.NoiseConditioned(_Recorder())(torch.zeros(3, 2, 4, 4), levels)

    torch.testing.assert_close(recorded[0][:, 2, 0, 0], levels)


def test_a_residual_conditioned_network_drops_its_extra_output_channel():
    """A residual body must be as wide out as in; the level channel is not data."""
    wide = adapters.NoiseConditioned(torch.nn.Conv2d(3, 3, 3, padding=1))

    assert wide(torch.zeros(1, 2, 8, 8), 0.05).shape == (1, 2, 8, 8)


class _IdentityPhysics:
    def __init__(self) -> None:
        self.adjoint_calls = 0
        self.normal_calls = 0

    def A(self, value):
        return value

    def A_adjoint(self, value):
        self.adjoint_calls += 1
        return value

    def A_adjoint_A(self, value):
        self.normal_calls += 1
        return value


class _ScaledPhysics:
    def A(self, value):
        return 2.0 * value

    def A_adjoint(self, value):
        return 2.0 * value


class _ChainRulePhysics(deepinv.physics.LinearPhysics):
    """A physics with no normal operator of its own."""

    def __init__(self) -> None:
        super().__init__(
            A=lambda value, **_: 2.0 * value, A_adjoint=lambda value, **_: 2.0 * value
        )
        self.A_adjoint_A = None


def _cartesian(size=16, coils=2):
    maps = torch.ones(1, coils, size, size, dtype=torch.complex64) / coils**0.5
    from pulserver.recon import Cartesian2D

    return Cartesian2D(torch.ones(1, 1, size, size), maps, viewed_as_real=True)


def _unfolded(physics_fidelity, *, max_iter=4):
    return deepinv.optim.PGD(
        data_fidelity=physics_fidelity,
        prior=deepinv.optim.PnP(deepinv.models.MedianFilter()),
        params_algo={"stepsize": 1.0, "g_param": 0.01},
        max_iter=max_iter,
        unfold=True,
    )


def test_scaled_adjoint_aligns_reencoded_measurement_norms():
    measurements = torch.ones(2, 3, 4, 4)

    estimate = ScaledAdjoint()(measurements, _ScaledPhysics())

    torch.testing.assert_close(estimate, 0.5 * measurements)
    encoded = _ScaledPhysics().A(estimate)
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoded, dim=(1, 2, 3)),
        torch.linalg.vector_norm(measurements, dim=(1, 2, 3)),
    )


def test_scaled_adjoint_initializes_an_unfolded_optimizer():
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = _unfolded(NormalEquationL2(), max_iter=1)
    model.custom_init = ScaledAdjoint()

    assert model(measured, physics).shape == (1, 2, 16, 16)


def test_the_normal_equation_gradient_is_the_chain_rule_gradient():
    physics = _cartesian()
    truth = torch.randn(1, 2, 16, 16)
    measured = physics.A(truth)
    estimate = truth + 0.1 * torch.randn_like(truth)

    ours = NormalEquationL2().grad(estimate, measured, physics)
    theirs = deepinv.optim.data_fidelity.L2().grad(estimate, measured, physics)

    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-4)


def test_the_normal_equation_gradient_scales_with_the_noise_level():
    physics = _cartesian()
    truth = torch.randn(1, 2, 16, 16)
    measured = physics.A(truth)
    estimate = truth + 0.1 * torch.randn_like(truth)

    ours = NormalEquationL2(sigma=2.0).grad(estimate, measured, physics)
    theirs = deepinv.optim.data_fidelity.L2(sigma=2.0).grad(estimate, measured, physics)

    torch.testing.assert_close(ours, theirs, atol=1e-5, rtol=1e-4)


def test_the_adjoint_is_computed_once_across_every_step():
    physics = _IdentityPhysics()
    fidelity = NormalEquationL2()
    measured = torch.ones(1, 2, 4, 4)

    for _ in range(5):
        fidelity.grad(torch.zeros(1, 2, 4, 4), measured, physics)

    assert physics.adjoint_calls == 1
    assert physics.normal_calls == 5


def test_a_new_measurement_recomputes_the_adjoint():
    physics = _IdentityPhysics()
    fidelity = NormalEquationL2()

    fidelity.grad(torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 4, 4), physics)
    fidelity.grad(torch.zeros(1, 2, 4, 4), torch.zeros(1, 2, 4, 4), physics)

    assert physics.adjoint_calls == 2


def test_a_physics_without_a_normal_operator_falls_back_to_the_chain_rule():
    physics = _ChainRulePhysics()
    measured = torch.ones(1, 2, 4, 4)
    estimate = torch.zeros(1, 2, 4, 4)

    ours = NormalEquationL2().grad(estimate, measured, physics)
    theirs = deepinv.optim.data_fidelity.L2().grad(estimate, measured, physics)

    torch.testing.assert_close(ours, theirs)


def test_stepping_reproduces_the_optimizers_own_run():
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = _unfolded(NormalEquationL2())

    steps = list(StepwiseUnroll(model).steps(measured, physics))

    assert [index for index, _ in steps] == [1, 2, 3, 4]
    torch.testing.assert_close(steps[-1][1], model(measured, physics))


def test_a_truncated_run_stops_where_it_was_asked_to():
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = _unfolded(NormalEquationL2())

    steps = list(StepwiseUnroll(model).steps(measured, physics, iterations=2))

    assert len(steps) == 2


def test_every_step_carries_a_graph_reaching_the_trainable_parameters():
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = _unfolded(NormalEquationL2())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable

    for _, image in StepwiseUnroll(model).steps(measured, physics, detach_every=1):
        image.pow(2).sum().backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_detaching_cuts_the_graph_at_the_requested_stride():
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = _unfolded(NormalEquationL2())

    grad_fns = [
        image.grad_fn
        for _, image in StepwiseUnroll(model).steps(measured, physics, detach_every=1)
    ]

    # every step starts from a detached state, so each graph is one step deep
    assert all(fn is not None for fn in grad_fns)
    for image_index in range(1, len(grad_fns)):
        assert grad_fns[image_index] is not grad_fns[image_index - 1]


def test_something_that_is_not_an_optimizer_is_refused():
    with pytest.raises(TypeError, match="DeepInverse optimizer"):
        StepwiseUnroll(object())


def test_a_folded_optimizer_can_be_stepped_for_inspection():
    """Stepping is also how an iteration is looked at, not only trained."""
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    model = deepinv.optim.PGD(
        data_fidelity=NormalEquationL2(),
        prior=deepinv.optim.PnP(deepinv.models.MedianFilter()),
        params_algo={"stepsize": 1.0, "g_param": 0.01},
        max_iter=3,
    )

    steps = list(StepwiseUnroll(model).steps(measured, physics))

    assert [index for index, _ in steps] == [1, 2, 3]
    torch.testing.assert_close(steps[-1][1], model(measured, physics))


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_an_invalid_detach_stride_is_refused(value):
    physics = _cartesian()
    measured = physics.A(torch.randn(1, 2, 16, 16))
    unroll = StepwiseUnroll(_unfolded(NormalEquationL2()))

    with pytest.raises(ValueError, match="detach_every"):
        next(unroll.steps(measured, physics, detach_every=value))


def test_a_discarded_measurement_is_never_served_a_stale_right_hand_side():
    """The cache is keyed on the measurement itself, not on where it sat."""
    physics = _IdentityPhysics()
    fidelity = NormalEquationL2()

    first = torch.ones(1, 2, 4, 4)
    fidelity.grad(torch.zeros(1, 2, 4, 4), first, physics)
    del first

    second = torch.full((1, 2, 4, 4), 3.0)
    gradient = fidelity.grad(torch.zeros(1, 2, 4, 4), second, physics)

    torch.testing.assert_close(gradient, -second)
    assert physics.adjoint_calls == 2
