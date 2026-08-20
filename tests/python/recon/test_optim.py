"""Tests for DeepInverse-style MRI optimizers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
deepinv = pytest.importorskip("deepinv")

import pulserver.recon as recon
from pulserver.recon.optim import (
    ADMM,
    FISTA,
    IRGNM,
    PDHG,
    ConjugateGradient,
    OptimResult,
    StackedPrior,
)


class _IdentityPhysics:
    def __init__(self) -> None:
        self.normal_calls = 0

    def A(self, value):
        return value

    def A_adjoint(self, value):
        return value

    def A_adjoint_A(self, value):
        self.normal_calls += 1
        return value

    def A_vjp(self, _value, vector):
        return vector


class _LearnedPrior(deepinv.optim.Prior):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def prox(self, value, _g_param=None, *, gamma=1.0, **_kwargs):
        return value / (1.0 + gamma * self.weight)


class _LearnedTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def A(self, value):
        return self.gain * value

    def A_adjoint(self, value):
        return self.gain * value


class _ScaledIdentity:
    def __init__(self, scale) -> None:
        self.scale = scale

    def A(self, value):
        return self.scale * value

    def A_adjoint(self, value):
        return self.scale * value

    def A_adjoint_A(self, value):
        return self.scale.square() * value


class _QuadraticPhysics:
    def A(self, value):
        return value.square()

    def A_adjoint(self, value):
        return value

    def linearize(self, value):
        return self.A(value), _ScaledIdentity(2.0 * value)


class _AutomaticQuadraticPhysics(deepinv.physics.Physics):
    def A(self, value):
        return value.square()


def test_public_optim_namespace_is_deepinverse_shaped():
    assert recon.optim.__all__ == [
        "ADMM",
        "CGInfo",
        "ConjugateGradient",
        "FISTA",
        "IRGNM",
        "OptimResult",
        "OptimState",
        "PDHG",
        "PolynomialPreconditioner",
        "StackedPrior",
        "pics",
    ]
    assert issubclass(FISTA, deepinv.optim.FISTA)


def test_implicit_cg_matches_dense_solution_and_manual_backward():
    scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor(
        [[2.0 + 1.0j, 4.0 - 2.0j], [1.0 - 3.0j, -2.0 + 1.0j]],
        dtype=torch.complex128,
        requires_grad=True,
    )
    solver = ConjugateGradient(max_iter=4, rtol=1e-12, batch_dim=0)

    solution, info = solver(
        lambda value: scale * value,
        rhs,
        parameters=[scale],
        return_info=True,
    )
    loss = solution.abs().square().sum()
    loss.backward()

    torch.testing.assert_close(solution, rhs.detach() / scale.detach())
    torch.testing.assert_close(rhs.grad, solution.detach())
    expected_scale_grad = -rhs.detach().abs().square().sum() / 4.0
    torch.testing.assert_close(scale.grad, expected_scale_grad)
    assert info.converged
    assert info.iterations == 1
    assert info.residual_norm.shape == (2,)


def test_pics_uses_implicit_cg_end_to_end():
    physics = _IdentityPhysics()
    data = torch.full((1, 1, 4), 3.0, requires_grad=True)

    result = recon.pics(
        data,
        physics,
        regularization=0.5,
        iterations=3,
        tolerance=1e-8,
    )
    result.sum().backward()

    torch.testing.assert_close(result, torch.full_like(data, 2.0))
    torch.testing.assert_close(data.grad, torch.full_like(data, 2.0 / 3.0))


def test_stacked_prior_is_simultaneous_and_fista_rejects_proximal_sum():
    stack = StackedPrior(
        [deepinv.optim.L1Prior(), deepinv.optim.L1Prior()],
        weights=[0.2, 0.3],
    )
    assert len(stack) == 2
    with pytest.raises(ValueError, match="PDHG or ADMM"):
        FISTA(prior=stack)


@pytest.mark.parametrize("inner_kind", ["cg", "admm"])
def test_irgnm_reuses_linear_inner_solvers(inner_kind):
    if inner_kind == "cg":
        inner = ConjugateGradient(max_iter=3, rtol=1e-8, batch_dim=0)
    else:
        inner = ADMM(
            max_iter=1,
            cg_max_iter=3,
            cg_rtol=1e-8,
        )
    model = IRGNM(inner, damping=0.0, max_iter=5)
    data = torch.full((1, 1, 4), 4.0)

    result = model(
        data,
        _QuadraticPhysics(),
        init=torch.ones_like(data),
        return_info=True,
        record_iterations=(1, 5),
    )

    torch.testing.assert_close(
        result.reconstruction,
        torch.full_like(data, 2.0),
        atol=1e-5,
        rtol=1e-5,
    )
    assert set(result.history) == {1, 5}


def test_irgnm_linearizes_an_ordinary_deepinverse_physics():
    model = IRGNM(
        ConjugateGradient(max_iter=3, rtol=1e-8, batch_dim=0),
        damping=0.0,
        max_iter=5,
    )
    data = torch.full((1, 1, 4), 4.0)

    result = model(
        data,
        _AutomaticQuadraticPhysics(),
        init=torch.ones_like(data),
    )

    torch.testing.assert_close(
        result,
        torch.full_like(data, 2.0),
        atol=1e-5,
        rtol=1e-5,
    )


def test_irgnm_iteration_callback_can_apply_a_state_projection():
    completed = []

    def project(estimate, _physics, iteration):
        completed.append(iteration)
        return estimate.clamp_max(1.8)

    model = IRGNM(
        ConjugateGradient(max_iter=3, rtol=1e-8, batch_dim=0),
        damping=0.0,
        max_iter=3,
        iteration_callback=project,
    )
    data = torch.full((1, 1, 4), 4.0)

    result = model(data, _QuadraticPhysics(), init=torch.ones_like(data))

    assert completed == [1, 2, 3]
    assert bool(torch.all(result <= 1.8))


@pytest.mark.parametrize("algorithm", ["fista", "pdhg", "admm"])
def test_optimizers_match_identity_l1_solution(algorithm):
    physics = _IdentityPhysics()
    data = torch.full((2, 1, 8), 2.0)
    prior = deepinv.optim.L1Prior()
    if algorithm == "fista":
        model = FISTA(
            data_fidelity=deepinv.optim.L2(),
            prior=prior,
            lambda_reg=0.5,
            stepsize=0.9,
            max_iter=40,
        )
    elif algorithm == "pdhg":
        model = PDHG(
            data_fidelity=deepinv.optim.L2(),
            prior=StackedPrior([prior]),
            lambda_reg=0.5,
            stepsize=0.4,
            stepsize_dual=0.4,
            max_iter=100,
        )
    else:
        model = ADMM(
            data_fidelity=deepinv.optim.L2(),
            prior=StackedPrior([prior]),
            lambda_reg=0.5,
            stepsize=1.0,
            max_iter=20,
            cg_max_iter=4,
            cg_rtol=1e-7,
        )

    result = model(data, physics)
    torch.testing.assert_close(
        result,
        torch.full_like(data, 1.5),
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.parametrize("algorithm", ["pdhg", "admm"])
def test_product_solvers_add_multiple_regularizer_weights(algorithm):
    physics = _IdentityPhysics()
    data = torch.full((1, 1, 4), 2.0)
    stack = StackedPrior(
        [deepinv.optim.L1Prior(), deepinv.optim.L1Prior()],
        weights=[0.2, 0.3],
    )
    if algorithm == "pdhg":
        model = PDHG(
            prior=stack,
            stepsize=0.3,
            stepsize_dual=0.3,
            max_iter=150,
        )
    else:
        model = ADMM(
            prior=stack,
            max_iter=30,
            cg_max_iter=5,
            cg_rtol=1e-7,
        )

    result = model(data, physics)
    torch.testing.assert_close(
        result,
        torch.full_like(data, 1.5),
        atol=3e-4,
        rtol=3e-4,
    )


def test_admm_image_update_uses_fast_normal_operator():
    physics = _IdentityPhysics()
    model = ADMM(
        prior=deepinv.optim.L1Prior(),
        lambda_reg=0.1,
        max_iter=3,
        cg_max_iter=2,
        cg_rtol=0.0,
    )
    model(torch.ones(1, 1, 4), physics)
    assert physics.normal_calls == 9


@pytest.mark.parametrize("algorithm", ["fista", "pdhg", "admm"])
def test_advanced_state_interface_records_only_requested_iterations(algorithm):
    physics = _IdentityPhysics()
    data = torch.full((1, 1, 4), 2.0)
    kwargs = {"prior": deepinv.optim.L1Prior(), "max_iter": 5, "unfold": True}
    if algorithm == "fista":
        kwargs["stepsize"] = 0.9
        model = FISTA(**kwargs)
    elif algorithm == "pdhg":
        kwargs.update({"stepsize": 0.4, "stepsize_dual": 0.4})
        model = PDHG(**kwargs)
    else:
        kwargs["cg_max_iter"] = 3
        model = ADMM(**kwargs)

    initial = model.init_state(data, physics)
    updated = model.step(initial, data, physics)
    assert initial.iteration == 0
    assert updated.iteration == 1
    assert model.get_output(updated) is updated.estimate

    result = model(
        data,
        physics,
        return_info=True,
        record_iterations=(0, 2, 5),
        detach_every=2,
    )
    assert isinstance(result, OptimResult)
    assert result.state.iteration == 5
    assert set(result.history) == {0, 2, 5}
    assert result.history[2].requires_grad

    truncated = model(
        data,
        physics,
        iterations=2,
        return_info=True,
        detach_every=2,
    )
    assert not truncated.state.estimate.requires_grad


def test_training_controls_address_prior_transform_and_algorithm_roles():
    prior = _LearnedPrior()
    transform = _LearnedTransform()
    stack = StackedPrior(
        [prior],
        transforms=[transform],
        weights=[0.5],
        g_params=[0.1],
        trainable_weights=True,
        trainable_g_params=True,
    )
    model = ADMM(
        prior=stack,
        max_iter=2,
        unfold=True,
    )

    assert model.prior_parameters() == [prior.weight]
    assert model.transform_parameters() == [transform.gain]
    algorithm_parameters = model.algorithm_parameters()
    assert any(parameter is stack.weights for parameter in algorithm_parameters)
    assert any(parameter is stack.g_params[0] for parameter in algorithm_parameters)
    assert {group["name"] for group in model.parameter_groups()} == {
        "prior",
        "transform",
        "algorithm",
    }

    model.set_trainable("prior", enabled=False, stages=[0])
    model.set_trainable("transform", enabled=False, stages=[0])
    assert not prior.weight.requires_grad
    assert not transform.gain.requires_grad
    model.set_trainable("prior", enabled=True)
    assert prior.weight.requires_grad


def test_the_optim_namespace_is_derived_from_the_recon_map():
    """The facade keeps no second copy of the layout: its members are exactly
    the recon map's optim entries."""
    import pulserver.recon.optim as optim

    derived = {
        name for name, target in recon._MEMBERS.items() if target.startswith("optim.")
    }
    assert set(optim.__all__) == derived


def test_every_stepwise_solver_satisfies_the_stateful_reconstructor_protocol():
    """One step-wise contract: iterations, init_state, step, get_output."""
    from pulserver.recon.learned import StatefulReconstructor, UnrolledReconstructor

    class _Identity(torch.nn.Module):
        def forward(self, value, *_args, **_kwargs):
            return value

    solvers = [
        FISTA(max_iter=2),
        ADMM(max_iter=2),
        PDHG(max_iter=2),
        IRGNM(ConjugateGradient(max_iter=2), max_iter=2),
        UnrolledReconstructor(_Identity(), iterations=1),
    ]
    for solver in solvers:
        assert isinstance(solver, StatefulReconstructor), type(solver).__name__
