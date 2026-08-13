"""Tests for stateful learned reconstruction and memory policies."""

from __future__ import annotations

import torch

from pulserver.recon.learned import (
    GradientDataConsistency,
    ScaledAdjoint,
    StatefulReconstructor,
    UnrollResult,
    UnrolledReconstructor,
)


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


class _ConditionedDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, value, sigma, *, iteration, acceleration):
        self.calls.append((iteration, acceleration, sigma))
        return self.scale * value


class _CountingDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.9))
        self.calls = 0

    def forward(self, value, _sigma=None):
        self.calls += 1
        return self.scale * value


class _ImageOnlyDenoiser(torch.nn.Module):
    def forward(self, value):
        return value


def test_scaled_adjoint_aligns_reencoded_measurement_norms():
    measurements = torch.ones(2, 3, 4, 4)

    estimate, rhs = ScaledAdjoint()(measurements, _ScaledPhysics())

    torch.testing.assert_close(rhs, 2.0 * measurements)
    torch.testing.assert_close(estimate, 0.5 * measurements)
    encoded = _ScaledPhysics().A(estimate)
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoded, dim=(1, 2, 3)),
        torch.linalg.vector_norm(measurements, dim=(1, 2, 3)),
    )


def test_default_initializer_computes_adjoint_once():
    physics = _IdentityPhysics()
    model = UnrolledReconstructor(_CountingDenoiser(), iterations=1)

    state = model.init_state(torch.ones(1, 2, 4, 4), physics)

    assert state.iteration == 0
    assert physics.adjoint_calls == 1


def test_unroll_accepts_plain_torch_modules_without_strength_argument():
    model = UnrolledReconstructor(_ImageOnlyDenoiser(), iterations=1)

    result = model(torch.ones(1, 2, 4, 4), _IdentityPhysics())

    assert result.shape == (1, 2, 4, 4)


def test_unroll_forwards_iteration_and_acquisition_context():
    denoiser = _ConditionedDenoiser()
    model = UnrolledReconstructor(
        denoiser,
        iterations=3,
        data_consistency=GradientDataConsistency(
            3,
            stepsize=0.5,
            trainable=False,
        ),
        denoiser_strength=[0.1, 0.2, 0.3],
        conditioning="all",
    )
    data = torch.ones(1, 10, 4, 4, 4)

    result = model(
        data,
        _IdentityPhysics(),
        context={"acceleration": 12},
        return_info=True,
        record_iterations=(0, 2, 3),
    )

    assert isinstance(model, StatefulReconstructor)
    assert isinstance(result, UnrollResult)
    assert result.reconstruction.shape == (1, 10, 4, 4, 4)
    assert set(result.history) == {0, 2, 3}
    assert [(item[0], item[1]) for item in denoiser.calls] == [
        (0, 12),
        (1, 12),
        (2, 12),
    ]
    torch.testing.assert_close(
        torch.stack([item[2] for item in denoiser.calls]),
        torch.tensor([0.1, 0.2, 0.3]),
    )


def test_greedy_iteration_detaches_between_local_backward_calls():
    denoiser = _CountingDenoiser()
    model = UnrolledReconstructor(
        denoiser,
        iterations=3,
        data_consistency=GradientDataConsistency(3, stepsize=0.25),
    )
    data = torch.full((1, 2, 4, 4), 2.0)
    states = []

    for state in model.iterate(
        data,
        _IdentityPhysics(),
        init=torch.zeros_like(data),
        detach_between=True,
    ):
        states.append(state)
        state.estimate.square().mean().backward()

    assert [state.iteration for state in states] == [1, 2, 3]
    assert denoiser.calls == 3
    assert denoiser.scale.grad is not None
    assert torch.isfinite(denoiser.scale.grad)
    assert model.data_consistency.stepsizes.grad is not None
    assert torch.isfinite(model.data_consistency.stepsizes.grad).all()


def test_checkpointing_recomputes_data_consistency_and_denoiser_on_backward():
    physics = _IdentityPhysics()
    denoiser = _CountingDenoiser()
    model = UnrolledReconstructor(
        denoiser,
        iterations=2,
        data_consistency=GradientDataConsistency(2, stepsize=0.25),
    )
    data = torch.full((1, 2, 4, 4), 2.0, requires_grad=True)

    output = model(
        data,
        physics,
        init=torch.zeros_like(data),
        checkpoint={"data_consistency", "denoiser"},
    )
    assert physics.normal_calls == 2
    assert denoiser.calls == 2

    output.square().mean().backward()

    assert physics.normal_calls == 4
    assert denoiser.calls == 4
    assert data.grad is not None


def test_denoiser_sequences_support_grouped_weight_sharing():
    first = _CountingDenoiser()
    second = _CountingDenoiser()
    model = UnrolledReconstructor(
        [first, first, second],
        iterations=3,
        data_consistency=GradientDataConsistency(
            3,
            stepsize=0.5,
            trainable=False,
        ),
    )

    model(torch.ones(1, 2, 4, 4), _IdentityPhysics())

    assert first.calls == 2
    assert second.calls == 1
    parameters = list(model.parameters())
    assert sum(parameter is first.scale for parameter in parameters) == 1


def test_truncated_backprop_keeps_final_segment_graph():
    denoisers = [_CountingDenoiser() for _ in range(3)]
    model = UnrolledReconstructor(
        denoisers,
        iterations=3,
        data_consistency=GradientDataConsistency(
            3,
            stepsize=0.5,
            trainable=False,
        ),
    )

    output = model(
        torch.ones(1, 2, 4, 4),
        _IdentityPhysics(),
        detach_every=1,
    )
    output.square().mean().backward()

    assert denoisers[0].scale.grad is None
    assert denoisers[1].scale.grad is None
    assert denoisers[2].scale.grad is not None
