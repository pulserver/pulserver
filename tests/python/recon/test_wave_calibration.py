"""Tests for Wave-CAIPI PSF synthesis and Wave-Shuffling calibration."""

from __future__ import annotations

from math import pi

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("deepinv")

from pulserver.recon.calibration import (
    WavePSF,
    WavePSFCalibration,
    WavePSFResult,
)
from pulserver.recon.physics import WaveEncoding


def _axes():
    return torch.linspace(-0.012, 0.012, 5), torch.linspace(-0.01, 0.01, 3)


def test_wave_psf_matches_definition_and_differentiates():
    phase_axis, partition_axis = _axes()
    phase = torch.randn(2, 8, generator=torch.Generator().manual_seed(4))
    phase.requires_grad_()

    result = WavePSF(phase_axis, partition_axis)(phase)
    angle = (
        phase[0, :, None, None] * phase_axis[None, :, None]
        + phase[1, :, None, None] * partition_axis[None, None, :]
    )
    reference = torch.complex(torch.cos(angle), -torch.sin(angle))

    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(result.abs(), torch.ones_like(result.real))
    result.real.sum().backward()
    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()


def test_phase_from_gradients_integrates_piecewise_linear_waveforms():
    gradient = torch.tensor([[2.0] * 5, [-0.5] * 5])
    adc_times = torch.tensor([0.0, 0.05, 0.2, 0.4])

    phase = WavePSF.phase_from_gradients(
        gradient,
        0.1,
        adc_times,
        gyromagnetic_ratio=1.0 / (2.0 * pi),
    )

    torch.testing.assert_close(phase, gradient[:, :1] * adc_times)


def test_sinusoidal_phase_matches_bart_wavepsf_prephasing():
    readout = 8
    duration = 80e-6
    raster = 10e-6
    cycles = 1.0
    maximum = 0.08
    gamma = 42.5756e6
    omega = 2.0 * pi * cycles / duration
    amplitude = maximum
    times = torch.arange(readout) * raster
    gradients = torch.stack(
        [amplitude * torch.sin(omega * times), amplitude * torch.cos(omega * times)]
    )
    moments = (gradients.cumsum(-1) - 0.5 * gradients) * raster
    moments = moments - amplitude / omega
    reference = 2.0 * pi * gamma * moments

    phase = WavePSF.sinusoidal_phase(
        readout,
        duration,
        cycles,
        maximum,
        1e5,
        variant="wave-caipi",
        gradient_raster_time=raster,
    )
    shuffling = WavePSF.sinusoidal_phase(
        readout,
        duration,
        cycles,
        maximum,
        1e5,
        variant="wave-shuffling",
        gradient_raster_time=raster,
    )

    torch.testing.assert_close(phase, reference, atol=2e-4, rtol=2e-5)
    expected_offset = 2.0 * pi * gamma * amplitude / omega
    torch.testing.assert_close(
        shuffling[1] - phase[1],
        torch.full((readout,), expected_offset),
        atol=2e-4,
        rtol=2e-5,
    )


def test_wave_shuffling_calibration_returns_bounded_fit_and_corrected_images():
    generator = torch.Generator().manual_seed(19)
    phase_axis, partition_axis = _axes()
    phase = 200.0 * torch.randn(2, 8, generator=generator)
    reconstruction = torch.randn(
        2,
        8,
        phase_axis.numel(),
        partition_axis.numel(),
        generator=generator,
        dtype=torch.complex64,
    )
    calibrator = WavePSFCalibration(
        phase_axis,
        partition_axis,
        frequency_indices=torch.tensor([[1], [1]]),
        alternations=1,
        max_iter=3,
        estimate_offsets=False,
    )

    result = calibrator(reconstruction, phase, return_info=True)

    assert isinstance(result, WavePSFResult)
    assert result.psf.shape == (8, 5, 3)
    assert result.phase.shape == (2, 8)
    assert result.phase_error.shape == (2, 8)
    assert result.spatial_offset.shape == (2,)
    assert result.corrected.shape == reconstruction.shape
    assert result.frequency_indices.tolist() == [[1], [1]]
    assert len(result.objective_history) == 2
    assert result.objective_history[-1] <= result.objective_history[0] + 1e-6
    torch.testing.assert_close(result.psf.abs(), torch.ones_like(result.psf.real))


def test_coefficient_specific_calibration_and_physics_result_composition():
    generator = torch.Generator().manual_seed(23)
    phase_axis, partition_axis = _axes()
    rank = 2
    reconstruction = torch.randn(
        2,
        rank,
        6,
        5,
        3,
        generator=generator,
        dtype=torch.complex64,
    )
    phase = torch.randn(rank, 2, 8, generator=generator) * 100.0
    calibrator = WavePSFCalibration(
        phase_axis,
        partition_axis,
        frequency_indices=torch.tensor([[1], [1]]),
        alternations=1,
        max_iter=1,
        estimate_offsets=False,
        coefficient_specific=True,
    )

    result = calibrator(reconstruction, phase, return_info=True)

    assert result.psf.shape == (rank, 8, 5, 3)
    assert result.phase_error.shape == (rank, 2, 8)
    assert result.corrected.shape == reconstruction.shape

    sampling = torch.tensor([[0, 0], [2, 1], [4, 2]])
    coil_maps = torch.ones(1, 6, 5, 3, dtype=torch.complex64)
    physics_from_result = WaveEncoding(
        sampling,
        coil_maps,
        WavePSFResult(
            psf=result.psf[:1],
            phase=result.phase[:1],
            phase_error=result.phase_error[:1],
            spatial_offset=result.spatial_offset[:1],
            frequency_indices=result.frequency_indices,
            corrected=result.corrected,
            objective_history=result.objective_history,
        ),
        viewed_as_real=False,
    )
    physics_from_tensor = WaveEncoding(
        sampling,
        coil_maps,
        result.psf[0],
        viewed_as_real=False,
    )
    image = torch.randn(1, 1, 6, 5, 3, dtype=torch.complex64)
    torch.testing.assert_close(
        physics_from_result.A(image),
        physics_from_tensor.A(image),
    )


def test_the_fitted_harmonics_are_the_ones_the_corkscrew_turns_at():
    """A phase that integrates a one-sided gradient has its largest Fourier
    bin at zero, and a constant readout phase displaces the image rather than
    blurring it -- so the search that picks what to fit must skip it, or every
    scan fits a term the objective cannot move."""
    cycles = 4
    times = (torch.arange(24) + 0.5) / 24
    # A one-sided corkscrew: nonzero mean, so its phase carries a ramp.
    phase = torch.stack(
        [
            torch.cumsum(0.5 * (1 - torch.cos(2 * pi * cycles * times)), dim=0),
            torch.cumsum(torch.sin(2 * pi * cycles * times), dim=0),
        ]
    )[None]
    phase_axis, partition_axis = _axes()

    calibrator = WavePSFCalibration(phase_axis, partition_axis, n_harmonics=3)
    indices = calibrator._resolved_frequency_indices(phase)

    assert torch.fft.rfft(phase[0], dim=-1).abs().argmax(dim=-1).tolist() == [0, 0]
    assert (indices > 0).all()
    assert cycles in indices[0].tolist() and cycles in indices[1].tolist()
