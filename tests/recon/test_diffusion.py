"""Trace-weighted images and ADC maps of a diffusion-weighted series."""

from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.recon._diffusion import (
    ADC_SERIES,
    ADC_UNIT,
    TRACE_SERIES,
    DiffusionMaps,
    adc_map,
    trace_weighted,
)

#: Two b = 0 volumes and the six directions of a 1000 s/mm² shell.
B_VALUES = [0.0, 0.0, *[1000.0] * 6]
DIRECTIONS = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float
)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


def anisotropic_volumes(tensor, s0=1.0, b=1000.0):
    """Images of every volume of :data:`B_VALUES` for one voxel's tensor (mm²/s)."""
    weighted = [s0 * np.exp(-b * g @ tensor @ g) for g in DIRECTIONS]
    return [np.full((4, 4), s0), np.full((4, 4), s0)] + [
        np.full((4, 4), w) for w in weighted
    ]


def test_trace_weighting_of_three_orthogonal_directions_is_the_mean_diffusivity():
    tensor = np.diag([1.7e-3, 0.3e-3, 0.3e-3])
    b = 1000.0
    images = [np.full((2, 2), np.exp(-b * d)) for d in np.diag(tensor)]
    expected = np.exp(-b * np.trace(tensor) / 3)
    np.testing.assert_allclose(trace_weighted(images), expected, rtol=1e-12)


def test_one_shell_gives_the_log_ratio_over_its_b_value():
    s0, sb = np.full((3, 3), 2.0), np.full((3, 3), 0.8)
    np.testing.assert_allclose(
        adc_map([0.0, 1000.0], [s0, sb]), np.log(2.0 / 0.8) / 1000.0
    )


def test_mono_exponential_shells_fit_their_diffusivity_exactly():
    adc = 0.9e-3
    b = [0.0, 500.0, 1000.0, 2000.0]
    shells = [np.full((2, 2), 3.0 * np.exp(-v * adc)) for v in b]
    np.testing.assert_allclose(adc_map(b, shells), adc, rtol=1e-10)


def test_no_map_is_made_until_the_slice_holds_every_volume():
    maps = DiffusionMaps(B_VALUES)
    volumes = anisotropic_volumes(np.eye(3) * 1e-3)
    made = [
        maps.add(0, r, image, SimpleNamespace(repetition=r))
        for r, image in enumerate(volumes)
    ]
    assert all(not m for m in made[:-1])
    assert [result.series_index for result in made[-1]] == [TRACE_SERIES, ADC_SERIES]


def test_the_adc_map_of_an_isotropic_voxel_is_its_diffusivity_in_adc_units():
    diffusivity = 0.8e-3
    maps = DiffusionMaps(B_VALUES)
    volumes = anisotropic_volumes(np.eye(3) * diffusivity)
    for r, image in enumerate(volumes):
        made = maps.add(2, r, image, SimpleNamespace(repetition=r))
    (adc,) = [result for result in made if result.series_index == ADC_SERIES]
    np.testing.assert_allclose(adc.data, diffusivity / ADC_UNIT, rtol=1e-5)


def test_each_map_takes_the_geometry_of_a_volume_whose_b_value_it_states():
    maps = DiffusionMaps(B_VALUES)
    references = [SimpleNamespace(repetition=r) for r in range(len(B_VALUES))]
    for r, image in enumerate(anisotropic_volumes(np.eye(3) * 1e-3)):
        made = maps.add(0, r, image, references[r])
    trace, adc = made
    assert B_VALUES[trace.reference.repetition] == 1000.0
    assert B_VALUES[adc.reference.repetition] == 0.0


def test_background_noise_is_fitted_not_zeroed():
    rng = np.random.default_rng(0)
    maps = DiffusionMaps(B_VALUES)
    for r in range(len(B_VALUES)):
        noise = np.abs(rng.standard_normal((8, 8)))
        made = maps.add(0, r, noise, SimpleNamespace(repetition=r))
    for result in made:
        assert np.all(np.isfinite(result.data))
        assert np.count_nonzero(result.data == 0) == 0


def test_a_series_without_diffusion_weighting_makes_no_maps():
    header = SimpleNamespace(
        userParameters=SimpleNamespace(
            userParameterString=[
                SimpleNamespace(name="diffusion_table", value="0 0 0 0\n0 0 0 0")
            ],
            userParameterLong=[],
            userParameterDouble=[],
            userParameterBase64=[],
        )
    )
    assert DiffusionMaps.from_header(header) is None


@pytest.mark.parametrize("frames", [1, 2])
def test_the_volumes_of_a_repeated_table_make_one_set_of_maps_per_slice(frames):
    maps = DiffusionMaps(B_VALUES * frames)
    volumes = anisotropic_volumes(np.eye(3) * 1e-3) * frames
    made = [
        maps.add(0, r, image, SimpleNamespace(repetition=r))
        for r, image in enumerate(volumes)
    ]
    assert sum(bool(m) for m in made) == 1
