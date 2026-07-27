"""Tests for the high-level reconstruction API."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pulserver.recon.algorithms as algorithms
import pulserver.recon.denoisers as denoisers
import pulserver.recon.density as density
import pulserver.recon.physics as physics
from pulserver.recon.preprocessing import (
    cartesian_3d_to_2d,
    correct_epi_eddy_currents,
    epi_ramp_interpolate,
    noise_prewhiten,
    remove_readout_oversampling,
)


class _IdentityPhysics:
    def A(self, value):
        return value

    def A_adjoint(self, value):
        return value

    def A_adjoint_A(self, value):
        return value


def test_pics_selects_cg_without_a_denoiser(monkeypatch):
    calls = {}

    def conjugate_gradient(operator, rhs, **kwargs):
        calls["normal"] = operator(np.ones_like(rhs))
        calls.update(kwargs)
        return rhs

    module = SimpleNamespace(conjugate_gradient=conjugate_gradient)
    monkeypatch.setattr(
        algorithms,
        "import_module",
        lambda name: module if name == "deepinv.optim.linear" else None,
    )
    data = np.ones((1, 2, 4, 4))
    result = algorithms.pics(
        data,
        _IdentityPhysics(),
        regularization=0.25,
        iterations=7,
    )
    assert result is data
    np.testing.assert_allclose(calls["normal"], 1.25)
    assert calls["max_iter"] == 7


def test_pics_selects_fista_with_a_denoiser(monkeypatch):
    calls = {}

    class FISTA:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def __call__(self, data, selected_physics, **kwargs):
            calls["data"] = data
            calls["physics"] = selected_physics
            calls["call_kwargs"] = kwargs
            return "reconstructed"

    module = SimpleNamespace(
        FISTA=FISTA,
        L2=lambda: "l2",
        PnP=lambda model: ("pnp", model),
    )
    monkeypatch.setattr(algorithms, "import_module", lambda _name: module)
    model = object()
    selected_physics = _IdentityPhysics()
    assert (
        algorithms.pics(
            object(),
            selected_physics,
            model,
            regularization=0.05,
            iterations=9,
            stepsize=0.2,
        )
        == "reconstructed"
    )
    assert calls["g_param"] == 0.05
    assert calls["stepsize"] == 0.2
    assert calls["max_iter"] == 9
    assert calls["prior"] == ("pnp", model)


def test_pics_averages_a_sequence_of_denoisers(monkeypatch):
    calls = {}
    averaged = object()

    class FISTA:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def __call__(self, *_args, **_kwargs):
            return "reconstructed"

    models = [object(), object()]
    monkeypatch.setattr(
        algorithms,
        "average",
        lambda selected: calls.setdefault("models", selected) and averaged,
    )
    monkeypatch.setattr(
        algorithms,
        "import_module",
        lambda _name: SimpleNamespace(
            FISTA=FISTA,
            L2=lambda: "l2",
            PnP=lambda model: ("pnp", model),
        ),
    )
    assert (
        algorithms.pics(
            object(),
            _IdentityPhysics(),
            models,
            regularization=0.1,
            iterations=2,
            stepsize=0.5,
        )
        == "reconstructed"
    )
    assert calls["models"] is models
    assert calls["prior"] == ("pnp", averaged)


def test_polynomial_preconditioned_fista_uses_the_normal_operator():
    class CountingPhysics(_IdentityPhysics):
        def __init__(self):
            self.normal_calls = 0

        def A_adjoint_A(self, value):
            self.normal_calls += 1
            return value

    selected_physics = CountingPhysics()
    result = algorithms.pics(
        np.ones(4),
        selected_physics,
        lambda value, _sigma: value,
        regularization=0.1,
        iterations=3,
        stepsize=0.5,
        polynomial_degree=2,
        init=np.zeros(4),
    )
    assert selected_physics.normal_calls == 9
    assert np.isfinite(result).all()
    assert np.linalg.norm(result - 1) < 0.1


def test_denoiser_factories_hide_deepinverse_class_names(monkeypatch):
    models = SimpleNamespace(
        WaveletDenoiser=lambda **kwargs: ("wavelet", kwargs),
        TVDenoiser=lambda **kwargs: ("tv", kwargs),
        TGVDenoiser=lambda **kwargs: ("tgv", kwargs),
    )
    monkeypatch.setattr(denoisers, "import_module", lambda _name: models)
    assert denoisers.denoiser("wavelet", dimension=3)[1]["wvdim"] == 3
    assert denoisers.tv(n_it_max=4) == ("tv", {"n_it_max": 4})
    assert denoisers.tgv(crit=1e-4) == ("tgv", {"crit": 1e-4})
    with pytest.raises(ValueError, match="Unknown denoiser"):
        denoisers.denoiser("invented")


def test_average_denoiser_is_a_registered_torch_model(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        denoisers,
        "_models",
        lambda: SimpleNamespace(Denoiser=torch.nn.Module),
    )

    class Scale(torch.nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor

        def forward(self, value, sigma):
            return self.factor * value + sigma

    model = denoisers.AverageDenoiser([Scale(1), Scale(3)])
    result = model(torch.ones(2), 0.5)
    torch.testing.assert_close(result, torch.full((2,), 2.5))
    assert len(model.models) == 2


def test_llr_matches_direct_singular_value_thresholding(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        denoisers,
        "_models",
        lambda: SimpleNamespace(Denoiser=torch.nn.Module),
    )
    generator = torch.Generator().manual_seed(8)
    value = torch.randn(2, 5, 4, 4, generator=generator, dtype=torch.complex64)
    threshold = torch.tensor([0.1, 0.2])
    model = denoisers.llr(
        dimension=2,
        block_size=4,
        cycle_spins=False,
        block_batch_size=1,
    )

    result = model(value, threshold)
    matrix = value.reshape(2, 5, -1)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    reference = (
        u * (singular_values - threshold[:, None]).clamp_min(0).unsqueeze(-2)
    ) @ vh
    torch.testing.assert_close(
        result,
        reference.reshape_as(value),
        atol=1e-5,
        rtol=1e-5,
    )


def test_llr_chunked_3d_cycle_spins_preserve_zero_threshold(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        denoisers,
        "_models",
        lambda: SimpleNamespace(Denoiser=torch.nn.Module),
    )
    value = torch.randn(1, 5, 4, 4, 4, dtype=torch.complex64)
    model = denoisers.LLR(
        dimension=3,
        block_size=2,
        cycle_spins=True,
        block_batch_size=1,
    )
    model(value, 0.0)
    shifted_result = model(value, 0.0)
    torch.testing.assert_close(
        shifted_result,
        value,
        atol=1e-5,
        rtol=1e-5,
    )


def test_pipe_menon_dcf_delegates_to_mrinufft(monkeypatch):
    calls = {}

    def pipe(trajectory, image_shape, **kwargs):
        calls["trajectory"] = trajectory
        calls["image_shape"] = image_shape
        calls.update(kwargs)
        return "weights"

    monkeypatch.setattr(
        density,
        "import_module",
        lambda name: SimpleNamespace(pipe=pipe) if name == "mrinufft.density" else None,
    )
    trajectory = np.zeros((32, 2))
    assert (
        density.pipe_menon_dcf(
            trajectory,
            (64, 64),
            backend="finufft",
            max_iter=12,
        )
        == "weights"
    )
    assert calls["trajectory"] is trajectory
    assert calls["image_shape"] == (64, 64)
    assert calls["backend"] == "finufft"
    assert calls["max_iter"] == 12


def test_cartesian_factory_returns_uniform_facade(monkeypatch):
    calls = {}

    class Cartesian:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    module = SimpleNamespace(MultiCoilMRI=Cartesian)
    monkeypatch.setattr(physics, "import_module", lambda _name: module)
    mask = SimpleNamespace(device="cpu")
    coil_maps = SimpleNamespace(device="cuda:0")
    result = physics.Cartesian2D(mask, coil_maps, toeplitz=True)
    assert isinstance(result, physics.MRIPhysics)
    assert result.kind == "cartesian2d"
    assert result.normal_mode == "exact-fft"
    assert calls["device"] == "cuda:0"


def test_noncartesian_factory_owns_mrinufft_construction(monkeypatch):
    calls = {}
    native = SimpleNamespace()

    def constructor(**kwargs):
        calls.update(kwargs)
        return native

    mrinufft = SimpleNamespace(get_operator=lambda _backend: constructor)
    adapter = SimpleNamespace(use_toeplitz=False)
    monkeypatch.setattr(physics, "_require_mrinufft", lambda: mrinufft)
    monkeypatch.setattr(
        physics,
        "_native_linear_physics",
        lambda *_args, **_kwargs: adapter,
    )

    trajectory = np.zeros((3, 8, 2), dtype=np.float32)
    result = physics.NonCartesian2D(
        trajectory,
        (32, 32),
        backend="finufft",
        density=np.ones(24),
        toeplitz=True,
    )
    assert result.native_operator is native
    assert result.normal_mode == "toeplitz"
    assert calls["samples"] is trajectory
    assert calls["shape"] == (32, 32)
    assert calls["squeeze_dims"] is False


def test_epi_ramp_interpolation_handles_complex_batches():
    source = np.array([-1.0, 0.0, 1.0])
    target = np.array([-0.5, 0.5])
    data = np.array([[0.0 + 0.0j, 1.0 + 2.0j, 2.0 + 4.0j]])
    result = epi_ramp_interpolate(data, source, target)
    np.testing.assert_allclose(result, [[0.5 + 1.0j, 1.5 + 3.0j]])


def test_readout_hybrid_transform_and_oversampling_crop():
    image = np.zeros((2, 8), dtype=np.complex64)
    image[:, 2:6] = 1
    kspace = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(image, axes=-1), axis=-1, norm="ortho"),
        axes=-1,
    )
    cropped = remove_readout_oversampling(kspace, 4)
    assert cropped.shape == (2, 4)
    hybrid = cartesian_3d_to_2d(kspace)
    np.testing.assert_allclose(hybrid, image, atol=1e-6)


def test_noise_prewhitening_decorrelates_coils():
    rng = np.random.default_rng(4)
    mixing = np.array([[2.0, 0.0], [0.7, 1.0]])
    independent = rng.normal(size=(2, 20000)) + 1j * rng.normal(size=(2, 20000))
    noise = mixing @ independent
    whitened = noise_prewhiten(noise, noise, coil_axis=0)
    covariance = whitened @ whitened.conj().T / whitened.shape[-1]
    np.testing.assert_allclose(covariance, np.eye(2), atol=2e-2)


def test_symmetric_epi_eddy_correction_removes_known_phase():
    phase = np.linspace(-0.6, 0.6, 16)
    signal = np.ones((2, 16), dtype=np.complex64)
    positive = signal * np.exp(0.5j * phase)
    negative = signal * np.exp(-0.5j * phase)
    corrected_positive, corrected_negative, returned_phase = correct_epi_eddy_currents(
        positive, negative, phase
    )
    np.testing.assert_allclose(corrected_positive, signal, atol=1e-6)
    np.testing.assert_allclose(corrected_negative, signal, atol=1e-6)
    np.testing.assert_allclose(returned_phase, phase)
