"""Tests for the high-level reconstruction API."""

from __future__ import annotations

import inspect
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import pulserver.recon as recon
import pulserver.recon.optim as algorithms
import pulserver.recon.optim._algorithms as _algorithms
import pulserver.recon.denoisers as denoisers
import pulserver.recon.preprocessing as preprocessing
import pulserver.recon.physics as physics
from pulserver.recon._mrd import metadata
from pulserver.recon.preprocessing import (
    EPIPhaseCorrection,
    Homodyne,
    POCS,
    cartesian_3d_to_2d,
    correct_epi_eddy_currents,
    epi_ramp_interpolate,
    noise_prewhiten,
    remove_readout_oversampling,
)


#: Where the public names actually live. Reaching them is not supposed to
#: require knowing this -- that is what the test below is about.
RECON_MODULES = (
    "calibration",
    "datasets",
    "denoisers",
    "execution",
    "learned",
    "models",
    "motion",
    "optim",
    "physics",
    "plugin",
    "postprocessing",
    "preprocessing",
    "simulation",
    "weights",
)


def test_the_public_namespace_is_flat():
    """One access point. A reconstruction reaches everything as
    ``pulserver.recon.<name>`` and never has to know which file it is in, so no
    submodule is part of the public namespace."""
    assert not set(recon.__all__) & set(RECON_MODULES)
    assert not any("." in name or name.startswith("_") for name in recon.__all__)


def test_every_name_a_submodule_publishes_is_reachable_flat():
    """The map cannot rot: a name added to any submodule and not surfaced here
    would be a name only reachable the long way round."""
    import importlib

    flat = set(recon.__all__)
    for module in RECON_MODULES:
        published = set(
            getattr(importlib.import_module(f"pulserver.recon.{module}"), "__all__", ())
        )
        assert published <= flat, f"{module}: {sorted(published - flat)}"


def test_every_public_name_resolves():
    for name in recon.__all__:
        assert getattr(recon, name) is not None


def test_the_flat_name_is_the_same_object_as_the_module_one():
    assert recon.pics is algorithms.pics
    assert recon.diffusion_table is metadata.diffusion_table
    assert recon.ReconPlugin is recon.plugin.ReconPlugin
    assert recon.calibration_extent is recon.calibration.calibration_extent


def test_the_transport_stays_private():
    assert "Connection" not in dir(recon)
    assert "Server" not in dir(recon)
    assert "MrdMetadata" not in dir(recon)


def test_importing_public_root_does_not_load_private_mrd_stack():
    code = """
import sys
import pulserver.recon

loaded = [name for name in sys.modules if name.startswith("pulserver.recon._mrd")]
if loaded:
    raise SystemExit(f"private MRD modules imported: {loaded}")
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_importing_algorithm_module_does_not_require_deepinverse():
    code = """
import sys
sys.modules["deepinv"] = None
import pulserver.recon.optim
"""
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_scientific_apis_expose_only_deepinverse_style_classes():
    assert "Cartesian2D" in physics.__all__
    assert "Cartesian3D" in physics.__all__
    assert "NonCartesian2D" in physics.__all__
    assert "WaveEncoding" in physics.__all__
    assert "WaveShuffling" in physics.__all__
    assert "SMS" in physics.__all__
    assert "LLR" in denoisers.__all__
    assert "Positive" in denoisers.__all__
    assert inspect.isclass(physics.Cartesian2D)
    assert inspect.isclass(physics.NonCartesian2D)
    assert inspect.isclass(denoisers.LLR)
    assert "noncartesian_2d" not in physics.__all__
    assert "llr" not in denoisers.__all__
    assert "PipeMenonDCF" not in preprocessing.__all__


class _IdentityPhysics:
    def A(self, value):
        return value

    def A_adjoint(self, value):
        return value

    def A_adjoint_A(self, value):
        return value


def test_pics_selects_cg_without_a_denoiser(monkeypatch):
    calls = {}

    class ConjugateGradient:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def __call__(self, operator, rhs, **kwargs):
            calls["normal"] = operator(np.ones_like(rhs))
            calls["call_kwargs"] = kwargs
            return rhs

    monkeypatch.setattr(
        _algorithms,
        "_optim_class",
        lambda name: ConjugateGradient if name == "ConjugateGradient" else None,
    )
    data = np.ones((1, 2, 4, 4))
    result = _algorithms.pics(
        data,
        _IdentityPhysics(),
        regularization=0.25,
        iterations=7,
        init=np.zeros_like(data),
    )
    # NumPy in, NumPy back out through the native-complex boundary.
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)
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
        L2=lambda: "l2",
        PnP=lambda model: ("pnp", model),
    )
    monkeypatch.setattr(
        _algorithms,
        "_optim_class",
        lambda name: FISTA if name == "FISTA" else None,
    )
    monkeypatch.setattr(_algorithms, "import_module", lambda _name: module)
    model = object()
    selected_physics = _IdentityPhysics()
    assert (
        _algorithms.pics(
            object(),
            selected_physics,
            model,
            regularization=0.05,
            iterations=9,
            stepsize=0.2,
        )
        == "reconstructed"
    )
    from pulserver.recon.learned import _ComplexAdapter

    assert calls["g_param"] == 0.05
    assert calls["stepsize"] == 0.2
    assert calls["max_iter"] == 9
    # The denoiser is routed through the one complex adapter so it can act on
    # the native-complex image; the wrapped model is the one that was passed.
    tag, wrapped = calls["prior"]
    assert tag == "pnp"
    assert isinstance(wrapped, _ComplexAdapter)
    assert wrapped.model is model


def test_pics_rejects_ambiguous_denoiser_sequences():
    models = [object(), object()]
    with pytest.raises(TypeError, match="StackedPrior"):
        algorithms.pics(
            object(),
            _IdentityPhysics(),
            models,
            regularization=0.1,
            iterations=2,
            stepsize=0.5,
        )


def test_polynomial_preconditioned_fista_uses_the_normal_operator():
    class CountingPhysics(_IdentityPhysics):
        def __init__(self):
            self.normal_calls = 0

        def A_adjoint_A(self, value):
            self.normal_calls += 1
            return value

    selected_physics = CountingPhysics()
    result = _algorithms.pics(
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


def test_denoiser_classes_wrap_deepinverse_models(monkeypatch):
    models = SimpleNamespace(
        WaveletDenoiser=lambda **kwargs: ("wavelet", kwargs),
        TVDenoiser=lambda **kwargs: ("tv", kwargs),
        TGVDenoiser=lambda **kwargs: ("tgv", kwargs),
    )
    monkeypatch.setattr(denoisers, "import_module", lambda _name: models)
    assert denoisers.Wavelet(dimension=3).model[1]["wvdim"] == 3
    assert denoisers.TV(n_it_max=4).model == ("tv", {"n_it_max": 4})
    assert denoisers.TGV(crit=1e-4).model == ("tgv", {"crit": 1e-4})


def test_positive_prior_projects_native_and_paired_complex_values():
    torch = pytest.importorskip("torch")
    value = torch.tensor([[-2.0 + 3.0j, 4.0 - 1.0j]])
    native = denoisers.Positive()
    torch.testing.assert_close(
        native.prox(value),
        torch.tensor([[0.0 + 0.0j, 4.0 + 0.0j]]),
    )

    paired = torch.view_as_real(value).reshape(1, 4)
    projected = denoisers.Positive(viewed_as_real=True).prox(paired)
    restored = torch.view_as_complex(projected.reshape(1, 2, 2).contiguous())
    torch.testing.assert_close(restored, native.prox(value))


def test_positive_prior_projects_onto_a_fixed_phase_ray():
    torch = pytest.importorskip("torch")
    phase = torch.tensor(torch.pi / 2)
    model = denoisers.Positive(phase)
    value = torch.tensor([[2.0 + 3.0j, -1.0 - 2.0j]])

    result = model.prox(value)

    torch.testing.assert_close(result, torch.tensor([[0.0 + 3.0j, 0.0 + 0.0j]]))
    assert torch.isfinite(model(result, None)).all()
    assert torch.isinf(model(value, None)).all()


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_partial_fourier_reconstructors_preserve_fully_sampled_data(library):
    generator = np.random.default_rng(42)
    image = generator.normal(size=(2, 12, 14)) + 1j * generator.normal(size=(2, 12, 14))
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1), norm="ortho"
        ),
        axes=(-2, -1),
    )
    if library == "torch":
        torch = pytest.importorskip("torch")
        image = torch.as_tensor(image)
        kspace = torch.as_tensor(kspace)

    homodyne = Homodyne(dimension=2, partial_axis=-2)(kspace)
    pocs = POCS(dimension=2, partial_axis=-2, iterations=2)(kspace)

    if library == "torch":
        torch.testing.assert_close(homodyne, image, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(pocs, image, atol=1e-10, rtol=1e-10)
    else:
        np.testing.assert_allclose(homodyne, image, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(pocs, image, atol=1e-10, rtol=1e-10)


def test_pocs_keeps_acquired_partial_fourier_lines_exact():
    generator = np.random.default_rng(17)
    image = np.exp(1j * 0.25) * generator.random((10, 12))
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image), norm="ortho"))
    mask = np.zeros(10, dtype=bool)
    mask[:8] = True
    partial = kspace.copy()
    partial[~mask] = 0

    result = POCS(iterations=6)(partial, mask)
    recovered = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(result), norm="ortho"))

    np.testing.assert_allclose(recovered[mask], kspace[mask], atol=1e-10)


@pytest.mark.parametrize("library", ["numpy", "torch"])
def test_epi_phase_correction_tracks_independent_shot_drift(library):
    coordinate = np.linspace(-1, 1, 32)
    true_phase = np.stack(
        [0.1 + 0.3 * coordinate, -0.25 + 0.15 * coordinate],
    )
    signal = np.ones((2, 3, 32), dtype=np.complex64)
    positive = signal * np.exp(0.5j * true_phase[:, None])
    negative = signal * np.exp(-0.5j * true_phase[:, None])
    if library == "torch":
        torch = pytest.importorskip("torch")
        positive = torch.as_tensor(positive)
        negative = torch.as_tensor(negative)
        expected = torch.as_tensor(true_phase, dtype=positive.real.dtype)
    else:
        expected = true_phase

    model = EPIPhaseCorrection(shot_axis=0, readout_axis=-1)
    phase = model.fit(positive, negative)
    corrected_positive, corrected_negative, _ = model.correct(
        positive,
        negative,
        phase,
    )

    if library == "torch":
        torch.testing.assert_close(phase, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            corrected_positive,
            corrected_negative,
            atol=2e-6,
            rtol=2e-6,
        )
    else:
        np.testing.assert_allclose(phase, expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            corrected_positive,
            corrected_negative,
            atol=2e-6,
            rtol=2e-6,
        )


def test_epi_phase_correction_supports_nonleading_shot_axis():
    coordinate = np.linspace(-1, 1, 24)
    phase = np.stack([0.2 * coordinate, 0.1 - 0.3 * coordinate])
    positive = np.exp(0.5j * phase[:, None]).transpose(2, 1, 0)
    negative = np.exp(-0.5j * phase[:, None]).transpose(2, 1, 0)
    model = EPIPhaseCorrection(shot_axis=2, readout_axis=0)

    fitted = model.fit(positive, negative)
    corrected_positive, corrected_negative, broadcast_phase = model(
        positive,
        negative,
        fitted,
    )

    assert broadcast_phase.shape == (24, 1, 2)
    np.testing.assert_allclose(corrected_positive, corrected_negative, atol=1e-7)


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
    assert isinstance(model, denoisers.AverageDenoiser)
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
    model = denoisers.LLR(
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
    assert isinstance(model, denoisers.LLR)
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
        preprocessing,
        "import_module",
        lambda name: SimpleNamespace(pipe=pipe) if name == "mrinufft.density" else None,
    )
    trajectory = np.zeros((32, 2))
    assert (
        preprocessing.pipe_menon_dcf(
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
    assert isinstance(result, physics.Cartesian2D)
    assert result.kind == "cartesian2d"
    assert result.normal_mode == "exact-fft"
    assert calls["device"] == "cuda:0"


def test_cartesian_without_coil_maps_keeps_the_coils():
    """No sensitivities makes a coil-wise operator, not a coil sum.

    The adjoint returns one image per coil -- matching the non-Cartesian
    convention -- so the caller combines them explicitly rather than being
    handed a meaningless sum of the coils.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepinv")
    from pulserver.recon.postprocessing import coil_combine

    n, coils = 8, 4
    axes = (-2, -1)
    coil_images = torch.randn(coils, n, n, dtype=torch.complex64)
    kspace = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(coil_images, dim=axes), dim=axes, norm="ortho"
        ),
        dim=axes,
    )
    operator = physics.Cartesian2D(torch.ones(1, 1, n, n), img_size=(n, n))
    measurement = kspace[None]  # complex (1, coils, n, n)

    adjoint = operator.A_adjoint(measurement)
    assert adjoint.is_complex()
    assert tuple(adjoint.shape) == (1, coils, n, n)  # coils kept, not summed

    recovered = adjoint[0]
    assert torch.allclose(recovered, coil_images, atol=1e-4)

    combined = coil_combine(recovered.numpy(), coil_axis=0)
    rss = np.sqrt((np.abs(coil_images.numpy()) ** 2).sum(axis=0))
    assert np.allclose(np.abs(combined), rss, atol=1e-4)


def test_cartesian_gridder_matches_grid_cartesian():
    """Placing acquisitions one at a time equals gridding them all at once."""
    from pulserver.recon.preprocessing import CartesianGridder, grid_cartesian

    rng = np.random.default_rng(0)
    data = (
        rng.standard_normal((6, 3, 12)) + 1j * rng.standard_normal((6, 3, 12))
    ).astype(np.complex64)
    lines = [0, 2, 4, 5, 8, 11]
    buffer = CartesianGridder((12, 12), coils=3)
    for line, acquisition in zip(lines, data, strict=True):
        buffer.add(acquisition, line)
    grid, mask = buffer.result()
    reference, reference_mask = grid_cartesian(data, lines, 12)
    assert np.array_equal(grid, reference)
    assert np.array_equal(mask, reference_mask)


def test_cartesian_gridder_right_aligns_a_partial_echo():
    """A readout shorter than the grid ends where a full one would."""
    from pulserver.recon.preprocessing import CartesianGridder

    buffer = CartesianGridder((4, 16), coils=2)
    buffer.add(np.ones((2, 12), dtype=np.complex64), 1)
    assert not buffer.mask[1, :4].any()
    assert buffer.mask[1, 4:].all()
    assert not buffer.mask[0].any()


def test_cartesian_gridder_indexes_one_position_of_the_volume():
    """Indexing returns one slice's k-space and mask, without its leading axis."""
    from pulserver.recon.preprocessing import CartesianGridder

    buffer = CartesianGridder((3, 8, 16), coils=2)
    buffer.add(np.full((2, 16), 2.0, dtype=np.complex64), 2, 5)
    kspace, mask = buffer[2]
    assert kspace.shape == (2, 8, 16)
    assert mask.shape == (8, 16)
    assert mask[5].all()
    assert not buffer[0][1].any()


def test_cartesian_gridder_refuses_what_does_not_fit():
    from pulserver.recon.preprocessing import CartesianGridder

    buffer = CartesianGridder((3, 8, 16), coils=2)
    with pytest.raises(ValueError, match="coils"):
        buffer.add(np.ones((3, 16)), 0, 0)
    with pytest.raises(ValueError, match="index values"):
        buffer.add(np.ones((2, 16)), 0)
    with pytest.raises(ValueError, match="readout axis holds"):
        buffer.add(np.ones((2, 20)), 0, 0)


def test_center_crop_takes_the_middle_of_the_trailing_axes():
    from pulserver.recon.postprocessing import center_crop

    image = np.arange(2 * 4 * 8).reshape(2, 4, 8)
    cropped = center_crop(image, (2, 4))
    assert cropped.shape == (2, 2, 4)
    assert np.array_equal(cropped, image[:, 1:3, 2:6])
    with pytest.raises(ValueError, match="cannot crop"):
        center_crop(image, (16,))


def test_noncartesian_factory_owns_mrinufft_construction(monkeypatch):
    calls = {}
    native = SimpleNamespace(gram_op=lambda value, **_kwargs: value)

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


def test_noncartesian_frame_rebuild_slices_flattened_density(monkeypatch):
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    mrinufft = SimpleNamespace(get_operator=lambda _backend: constructor)
    monkeypatch.setattr(physics, "_require_mrinufft", lambda: mrinufft)
    monkeypatch.setattr(
        physics,
        "_native_linear_physics",
        lambda *_args, **_kwargs: SimpleNamespace(use_toeplitz=False),
    )
    trajectory = np.zeros((3, 8, 2), dtype=np.float32)
    density_weights = np.arange(24, dtype=np.float32)
    result = physics.NonCartesian2D(
        trajectory,
        (32, 32),
        density=density_weights,
    )

    result.rebuild(trajectory[1], frame_index=1)

    np.testing.assert_array_equal(calls[-1]["density"], density_weights[8:16])


def test_streamed_noncartesian_factory_only_builds_first_dynamic_frame(monkeypatch):
    calls = []

    def constructor(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    mrinufft = SimpleNamespace(get_operator=lambda _backend: constructor)
    monkeypatch.setattr(physics, "_require_mrinufft", lambda: mrinufft)
    monkeypatch.setattr(
        physics,
        "_native_linear_physics",
        lambda *_args, **_kwargs: SimpleNamespace(use_toeplitz=False),
    )
    trajectory = np.zeros((5, 3, 8, 2), dtype=np.float32)
    density_weights = np.ones((5, 3, 8), dtype=np.float32)
    policy = SimpleNamespace(frame_cache_size=2)

    result = physics.NonCartesian2D(
        trajectory,
        (32, 32),
        density=density_weights,
        streaming=policy,
    )

    np.testing.assert_array_equal(calls[0]["samples"], trajectory[0])
    np.testing.assert_array_equal(calls[0]["density"], density_weights[0])
    assert result.trajectory is trajectory
    assert result.streaming_policy is policy


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


def test_the_api_page_documents_every_public_name():
    """The namespace is flat, so the only place the library is grouped by what
    it is for is the API page -- which makes an undocumented name invisible."""
    import re
    from pathlib import Path

    page = Path(__file__).resolve().parents[3] / "docs/api/python/recon.md"
    listed: set[str] = set()
    for block in re.findall(
        r"autosummary::\n(?:\s+:\w+:.*\n)*\n((?:   \S+\n)+)", page.read_text()
    ):
        listed |= {line.strip() for line in block.splitlines() if line.strip()}

    assert set(recon.__all__) - listed == set(), "undocumented"
    assert listed - set(recon.__all__) == set(), "documented but not public"


def test_the_acquisition_flags_are_the_ismrmrd_ones():
    """They are defined here so this module imports without ``ismrmrd``, which
    makes agreeing with it something to check rather than something given."""
    import ismrmrd

    from pulserver.recon import AcquisitionFlag

    for member in AcquisitionFlag:
        assert getattr(ismrmrd, member.flag) == member.position, member.name

    declared = {name for name in dir(ismrmrd) if name.startswith("ACQ_")}
    assert declared == {member.flag for member in AcquisitionFlag}


def test_the_app_page_documents_every_reconstruction():
    """The zoo is named for the samplings it undoes, and the page is where a
    reader finds which one fits their scan."""
    import re
    from pathlib import Path

    import pulserver.app.recon as family

    page = Path(__file__).resolve().parents[3] / "docs/api/python/app_recon.md"
    listed: set[str] = set()
    for block in re.findall(
        r"autosummary::\n(?:\s+:\w+:.*\n)*\n((?:   \S+\n)+)", page.read_text()
    ):
        listed |= {line.strip() for line in block.splitlines() if line.strip()}

    assert listed == set(family.__all__)
