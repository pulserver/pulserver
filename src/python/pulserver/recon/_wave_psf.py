"""Wave-CAIPI PSF synthesis and Wave-Shuffling PSF calibration."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from math import ceil, pi
from typing import Any

import torch

from .._accelerators import require


@dataclass(frozen=True)
class WavePSFResult:
    """Outputs of data-driven Wave PSF calibration.

    Parameters
    ----------
    psf
        Calibrated PSF with shape ``(readout, phase, partition)`` or one PSF
        per subspace coefficient.
    phase
        Corrected phase accrual in radians per spatial-axis unit.
    phase_error
        Estimated sparse harmonic phase error.
    spatial_offset
        Estimated phase/partition isocenter offsets, in the units of the
        supplied spatial axes.
    frequency_indices
        Non-negative readout Fourier indices fitted by the calibrator.
    corrected
        Preliminary coefficient images after the estimated error
        deconvolution.
    objective_history
        Objective values before and after each bounded optimization stage.
    """

    psf: torch.Tensor
    phase: torch.Tensor
    phase_error: torch.Tensor
    spatial_offset: torch.Tensor
    frequency_indices: torch.Tensor
    corrected: torch.Tensor
    objective_history: tuple[float, ...]


class WavePSF(torch.nn.Module):
    r"""Synthesize a hybrid-space Wave point-spread function.

    The phase input contains phase accrual along the phase and partition
    directions, with shape ``(..., 2, readout)``. For physical calibration,
    use SI units: axes in metres, gradients in T/m, time in seconds, and the
    default proton gyromagnetic ratio in Hz/T.

    Parameters
    ----------
    phase_axis, partition_axis
        Physical voxel coordinates along the two Wave-encoding directions.
    cuda_kernel
        Use the fused Triton inference kernel when available. Autograd always
        uses the native Torch formulation.
    """

    def __init__(
        self,
        phase_axis: Any,
        partition_axis: Any,
        *,
        cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        phase_axis = torch.as_tensor(phase_axis, dtype=torch.float32)
        partition_axis = torch.as_tensor(partition_axis, dtype=torch.float32)
        if phase_axis.ndim != 1 or phase_axis.numel() == 0:
            raise ValueError("phase_axis must be a non-empty vector")
        if partition_axis.ndim != 1 or partition_axis.numel() == 0:
            raise ValueError("partition_axis must be a non-empty vector")
        self.register_buffer("phase_axis", phase_axis.contiguous())
        self.register_buffer("partition_axis", partition_axis.contiguous())
        self.cuda_kernel = bool(cuda_kernel)

    def forward(self, phase: Any) -> torch.Tensor:
        """Return ``exp(-i (phase_y y + phase_z z))`` as complex64."""
        phase = torch.as_tensor(phase, dtype=torch.float32)
        if phase.ndim < 2 or phase.shape[-2] != 2 or phase.shape[-1] == 0:
            raise ValueError("phase must have shape (..., 2, readout)")
        phase_axis = self.phase_axis.to(phase.device)
        partition_axis = self.partition_axis.to(phase.device)
        leading_shape = tuple(int(size) for size in phase.shape[:-2])
        flat_phase = phase.reshape(-1, 2, phase.shape[-1]).contiguous()
        output = _accelerated_wave_psf(
            flat_phase,
            phase_axis,
            partition_axis,
            cuda_kernel=self.cuda_kernel,
        )
        if output is None:
            angle = (
                flat_phase[:, 0, :, None, None] * phase_axis[None, None, :, None]
                + flat_phase[:, 1, :, None, None] * partition_axis[None, None, None, :]
            )
            output = torch.complex(torch.cos(angle), -torch.sin(angle))
        return output.reshape(
            *leading_shape,
            phase.shape[-1],
            phase_axis.numel(),
            partition_axis.numel(),
        )

    @staticmethod
    def phase_from_gradients(
        gradients: Any,
        gradient_raster_time: float,
        adc_times: Any,
        *,
        initial_moment: Any | None = None,
        gyromagnetic_ratio: float = 42.5756e6,
    ) -> torch.Tensor:
        """Integrate measured or designed Wave gradients at ADC times.

        Parameters
        ----------
        gradients
            Gradient samples with shape ``(..., 2, samples)`` in T/m.
        gradient_raster_time
            Time between gradient samples in seconds.
        adc_times
            ADC sample times relative to the first gradient sample in seconds.
        initial_moment
            Phase/partition gradient moment before the first sample in T s/m.
        gyromagnetic_ratio
            Gyromagnetic ratio in Hz/T.

        Returns
        -------
        torch.Tensor
            Phase accrual with shape ``(..., 2, readout)`` in rad/m.
        """
        gradients = torch.as_tensor(gradients, dtype=torch.float32)
        if gradients.ndim < 2 or gradients.shape[-2] != 2:
            raise ValueError("gradients must have shape (..., 2, samples)")
        if gradients.shape[-1] < 2:
            raise ValueError("at least two gradient samples are required")
        if gradient_raster_time <= 0.0:
            raise ValueError("gradient_raster_time must be positive")
        if gyromagnetic_ratio <= 0.0:
            raise ValueError("gyromagnetic_ratio must be positive")
        adc_times = torch.as_tensor(
            adc_times,
            dtype=torch.float32,
            device=gradients.device,
        )
        if adc_times.ndim != 1 or adc_times.numel() == 0:
            raise ValueError("adc_times must be a non-empty vector")
        duration = (gradients.shape[-1] - 1) * gradient_raster_time
        time_tolerance = max(duration * 1e-6, torch.finfo(adc_times.dtype).eps)
        if (
            float(adc_times.min()) < -time_tolerance
            or float(adc_times.max()) > duration + time_tolerance
        ):
            raise ValueError("adc_times must lie inside the gradient waveform")
        adc_times = adc_times.clamp(0.0, duration)

        if initial_moment is None:
            initial = torch.zeros(
                gradients.shape[:-1],
                dtype=gradients.dtype,
                device=gradients.device,
            )
        else:
            initial = torch.as_tensor(
                initial_moment,
                dtype=gradients.dtype,
                device=gradients.device,
            )
            try:
                initial = torch.broadcast_to(initial, gradients.shape[:-1])
            except RuntimeError as error:
                raise ValueError(
                    "initial_moment must broadcast to gradients[..., :, 0]"
                ) from error

        interval_moments = (
            0.5 * (gradients[..., 1:] + gradients[..., :-1]) * gradient_raster_time
        )
        node_moments = torch.cat(
            [
                initial[..., None],
                initial[..., None] + interval_moments.cumsum(dim=-1),
            ],
            dim=-1,
        )
        left_indices = torch.floor(adc_times / gradient_raster_time).to(torch.int64)
        left_indices = left_indices.clamp_max(gradients.shape[-1] - 2)
        left_times = left_indices.to(adc_times.dtype) * gradient_raster_time
        delta = adc_times - left_times
        left_gradient = gradients.index_select(-1, left_indices)
        right_gradient = gradients.index_select(-1, left_indices + 1)
        left_moment = node_moments.index_select(-1, left_indices)
        slope = (right_gradient - left_gradient) / gradient_raster_time
        moment = left_moment + left_gradient * delta + 0.5 * slope * delta.square()
        return (2.0 * pi * gyromagnetic_ratio * moment).to(torch.float32)

    @staticmethod
    def sinusoidal_phase(
        readout_size: int,
        readout_duration: float,
        cycles: float,
        max_gradient: Any,
        max_slew: Any,
        *,
        variant: str = "wave-caipi",
        gradient_raster_time: float = 10e-6,
        delays: Any = (0.0, 0.0),
        scales: Any = (1.0, 1.0),
        gyromagnetic_ratio: float = 42.5756e6,
        device: Any | None = None,
    ) -> torch.Tensor:
        """Return a slew/gradient-limited sine/cosine theoretical estimate.

        The first direction uses a sine gradient and the second a quadrature
        cosine. ``variant="wave-caipi"`` matches BART ``wavepsf`` prephasing;
        ``"wave-shuffling"`` uses the quadrature cosine convention from the
        reference Wave-Shuffling calibration. Both use BART's centered Fourier
        resampling. Actual scanner waveforms should use
        :meth:`phase_from_gradients`.
        """
        if readout_size <= 0:
            raise ValueError("readout_size must be positive")
        if readout_duration <= 0.0 or cycles <= 0.0:
            raise ValueError("readout_duration and cycles must be positive")
        if gradient_raster_time <= 0.0:
            raise ValueError("gradient_raster_time must be positive")
        if variant not in {"wave-caipi", "wave-shuffling"}:
            raise ValueError("variant must be 'wave-caipi' or 'wave-shuffling'")
        maximum = _as_pair(max_gradient, "max_gradient", device=device)
        slew = _as_pair(max_slew, "max_slew", device=device)
        delay = _as_pair(delays, "delays", device=device)
        scale = _as_pair(scales, "scales", device=device)
        if bool((maximum <= 0.0).any()) or bool((slew <= 0.0).any()):
            raise ValueError("gradient and slew limits must be positive")
        omega = 2.0 * pi * cycles / readout_duration
        amplitude = torch.minimum(maximum, slew / omega) * scale
        intervals = max(2, ceil(readout_duration / gradient_raster_time))
        raster_time = readout_duration / intervals
        times = (
            torch.arange(
                intervals,
                dtype=torch.float32,
                device=maximum.device,
            )
            * raster_time
        )
        arguments = omega * (times[None] - delay[:, None])
        gradients = torch.stack(
            [
                amplitude[0] * torch.sin(arguments[0]),
                amplitude[1] * torch.cos(arguments[1]),
            ]
        )
        initial_z = (
            -amplitude[1] / omega if variant == "wave-caipi" else amplitude[1] * 0.0
        )
        initial_moment = torch.stack([-amplitude[0] / omega, initial_z])
        half_sample_moment = (gradients.cumsum(dim=-1) - 0.5 * gradients) * raster_time
        phase = (
            2.0
            * pi
            * gyromagnetic_ratio
            * (half_sample_moment + initial_moment[:, None])
        )
        return _fourier_resize_phase(phase, readout_size)


class WavePSFCalibration(torch.nn.Module):
    """Refine a theoretical Wave PSF from preliminary coefficient images.

    The calibration model follows Wave-Shuffling PSF calibration: sparse
    readout Fourier harmonics describe gradient delay/scale errors, while two
    bounded spatial offsets describe isocenter error. Optimization uses only
    normalized central phase/partition profiles, so it does not replicate the
    full reconstruction inside each closure.

    Parameters
    ----------
    phase_axis, partition_axis
        Physical voxel coordinates. SI metres are recommended.
    n_harmonics
        Number of dominant non-negative readout frequencies per direction.
    frequency_indices
        Optional explicit indices with shape ``(2, n_harmonics)``. A vector is
        shared by both directions. By default indices are inferred from the
        theoretical phase.
    alternations
        Number of bounded harmonic/offset optimization pairs.
    max_iter
        Maximum LBFGS iterations in each stage.
    tolerance
        Gradient and parameter-change tolerance.
    sigmoid_scale
        Positive edge-magnitude sigmoid scale, or zero for an L1 objective.
    estimate_offsets
        Estimate phase/partition isocenter offsets.
    coefficient_specific
        Fit independent errors per subspace coefficient. The default fits the
        shared hardware PSF used by Wave-Shuffling and Wave-CAIPI.
    cuda_kernel
        Use fused Triton PSF synthesis outside autograd when available.
    """

    def __init__(
        self,
        phase_axis: Any,
        partition_axis: Any,
        *,
        n_harmonics: int = 1,
        frequency_indices: Any | None = None,
        alternations: int = 2,
        max_iter: int = 25,
        tolerance: float = 1e-6,
        sigmoid_scale: float = 0.0,
        estimate_offsets: bool = True,
        coefficient_specific: bool = False,
        cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        if n_harmonics <= 0:
            raise ValueError("n_harmonics must be positive")
        if alternations <= 0 or max_iter <= 0:
            raise ValueError("iteration counts must be positive")
        if tolerance < 0.0 or sigmoid_scale < 0.0:
            raise ValueError("tolerance and sigmoid_scale must be non-negative")
        self.synthesizer = WavePSF(
            phase_axis,
            partition_axis,
            cuda_kernel=cuda_kernel,
        )
        if frequency_indices is None:
            indices = torch.empty((0,), dtype=torch.int64)
        else:
            indices = torch.as_tensor(frequency_indices, dtype=torch.int64)
            if indices.ndim == 1:
                indices = indices[None].expand(2, -1).clone()
            if indices.ndim != 2 or indices.shape != (2, n_harmonics):
                raise ValueError("frequency_indices must have shape (2, n_harmonics)")
            if bool((indices < 0).any()):
                raise ValueError("frequency_indices must be non-negative")
        self.register_buffer("_frequency_indices", indices.contiguous())
        self.n_harmonics = int(n_harmonics)
        self.alternations = int(alternations)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.sigmoid_scale = float(sigmoid_scale)
        self.estimate_offsets = bool(estimate_offsets)
        self.coefficient_specific = bool(coefficient_specific)

    def forward(
        self,
        reconstruction: torch.Tensor,
        initial_phase: Any,
        *,
        coefficient_axis: int = -4,
        return_info: bool = False,
    ) -> torch.Tensor | WavePSFResult:
        """Calibrate a shared or coefficient-specific Wave PSF.

        Parameters
        ----------
        reconstruction
            Preliminary complex images with spatial dimensions
            ``(readout, phase, partition)`` last. For coefficient-specific
            calibration, ``coefficient_axis`` identifies the subspace rank.
        initial_phase
            Theoretical phase with shape ``(2, wave_readout)`` or
            ``(rank, 2, wave_readout)``.
        coefficient_axis
            Subspace coefficient dimension when fitting independent PSFs.
        return_info
            Return fitted parameters and corrected preliminary images.

        Returns
        -------
        torch.Tensor or WavePSFResult
            Calibrated complex64 PSF, or all calibration outputs.
        """
        if (
            not isinstance(reconstruction, torch.Tensor)
            or not reconstruction.is_complex()
        ):
            raise TypeError("reconstruction must be a complex torch.Tensor")
        if reconstruction.ndim < 3:
            raise ValueError("reconstruction must have three spatial dimensions")
        phase = torch.as_tensor(
            initial_phase,
            dtype=torch.float32,
            device=reconstruction.device,
        )
        phase_was_banked = phase.ndim == 3
        if phase.ndim == 2:
            phase = phase[None]
        if phase.ndim != 3 or phase.shape[1] != 2 or phase.shape[2] == 0:
            raise ValueError(
                "initial_phase must have shape (2, readout) or (rank, 2, readout)"
            )
        wave_readout = int(phase.shape[-1])
        if reconstruction.shape[-3] > wave_readout:
            raise ValueError("wave readout cannot be smaller than the image readout")
        phase_axis = self.synthesizer.phase_axis.to(reconstruction.device)
        partition_axis = self.synthesizer.partition_axis.to(reconstruction.device)
        if tuple(reconstruction.shape[-2:]) != (
            phase_axis.numel(),
            partition_axis.numel(),
        ):
            raise ValueError("spatial axes must match reconstruction dimensions")

        profiles_y, profiles_z = _central_profiles(
            reconstruction.detach(),
            wave_readout,
            phase_axis,
            partition_axis,
            coefficient_axis=coefficient_axis,
            coefficient_specific=self.coefficient_specific,
        )
        models = int(profiles_y.shape[0])
        if self.coefficient_specific:
            if phase.shape[0] == 1:
                reference_phase = phase.expand(models, -1, -1)
            elif phase.shape[0] == models:
                reference_phase = phase
            else:
                raise ValueError(
                    "banked initial_phase must match the subspace coefficient count"
                )
        else:
            reference_phase = phase.mean(dim=0, keepdim=True)

        indices = self._resolved_frequency_indices(reference_phase)
        basis = _harmonic_basis(indices, wave_readout, reference_phase.device)
        phase_scale = reference_phase.abs().amax().detach().clamp_min(1.0)
        offset_scale = torch.stack(
            [_axis_spacing(phase_axis), _axis_spacing(partition_axis)]
        )
        normalized_coefficients = torch.zeros(
            (models, 2, 2, self.n_harmonics),
            dtype=torch.float32,
            device=reconstruction.device,
            requires_grad=True,
        )
        normalized_offsets = torch.zeros(
            (models, 2),
            dtype=torch.float32,
            device=reconstruction.device,
            requires_grad=True,
        )
        spectra_y = _fftc(profiles_y, -2)
        spectra_z = _fftc(profiles_z, -2)
        history: list[float] = []

        def objective(*, cross_edges: bool) -> torch.Tensor:
            coefficients = normalized_coefficients * phase_scale
            phase_error = _phase_from_coefficients(coefficients, basis)
            corrected_phase = reference_phase - phase_error
            offsets = normalized_offsets * offset_scale
            error_y, error_z = _profile_error_psfs(
                phase_error,
                corrected_phase,
                offsets,
                phase_axis,
                partition_axis,
            )
            return _edge_objective(
                spectra_y,
                spectra_z,
                error_y,
                error_z,
                sigmoid_scale=self.sigmoid_scale,
                cross_edges=cross_edges,
            )

        with torch.enable_grad():
            history.append(float(objective(cross_edges=False).detach()))
            for _ in range(self.alternations):
                normalized_coefficients.requires_grad_(True)
                normalized_offsets.requires_grad_(False)
                self._minimize(
                    [normalized_coefficients],
                    lambda: objective(cross_edges=False),
                )
                history.append(float(objective(cross_edges=False).detach()))
                if self.estimate_offsets:
                    normalized_coefficients.requires_grad_(False)
                    normalized_offsets.requires_grad_(True)
                    self._minimize(
                        [normalized_offsets],
                        lambda: objective(cross_edges=True),
                    )
                    history.append(float(objective(cross_edges=True).detach()))

        coefficients = (normalized_coefficients.detach() * phase_scale).contiguous()
        phase_error = _phase_from_coefficients(coefficients, basis).detach()
        offsets = (normalized_offsets.detach() * offset_scale).contiguous()
        calibration_reference = reference_phase - phase_error
        output_phase_error = (
            phase_error if self.coefficient_specific else phase_error[:1]
        )
        output_offsets = offsets if self.coefficient_specific else offsets[:1]
        if self.coefficient_specific:
            base_phase = phase.expand(models, -1, -1) if phase.shape[0] == 1 else phase
        else:
            base_phase = phase
        corrected_phase = base_phase - output_phase_error
        psf = self.synthesizer(corrected_phase)
        output_offset_phase = (
            corrected_phase[:, 0] * output_offsets[:, 0, None]
            + corrected_phase[:, 1] * output_offsets[:, 1, None]
        )
        psf = (
            psf
            * torch.complex(
                torch.cos(output_offset_phase),
                torch.sin(output_offset_phase),
            )[..., None, None]
        )

        error_psf = self.synthesizer(phase_error)
        error_offset_phase = (
            calibration_reference[:, 0] * offsets[:, 0, None]
            + calibration_reference[:, 1] * offsets[:, 1, None]
        )
        error_psf = (
            error_psf
            * torch.complex(
                torch.cos(error_offset_phase),
                -torch.sin(error_offset_phase),
            )[..., None, None]
        )
        corrected = _apply_error_psf(
            reconstruction,
            error_psf,
            coefficient_axis=coefficient_axis if self.coefficient_specific else None,
        )

        if not phase_was_banked and not self.coefficient_specific:
            psf = psf[0]
            corrected_phase = corrected_phase[0]
        result_phase_error = (
            phase_error if self.coefficient_specific else phase_error[0]
        )
        result_offsets = offsets if self.coefficient_specific else offsets[0]
        if not return_info:
            return psf.to(torch.complex64)
        return WavePSFResult(
            psf=psf.to(torch.complex64),
            phase=corrected_phase,
            phase_error=result_phase_error,
            spatial_offset=result_offsets,
            frequency_indices=indices.detach(),
            corrected=corrected,
            objective_history=tuple(history),
        )

    # %% private module subroutines

    def _resolved_frequency_indices(self, phase: torch.Tensor) -> torch.Tensor:
        if self._frequency_indices.numel():
            indices = self._frequency_indices.to(phase.device)
            if int(indices.max()) > phase.shape[-1] // 2:
                raise ValueError("frequency_indices exceed the real-FFT support")
            return indices
        spectrum = torch.fft.rfft(phase.mean(dim=0), dim=-1).abs()
        if self.n_harmonics > spectrum.shape[-1]:
            raise ValueError("n_harmonics exceeds the real-FFT support")
        return (
            torch.topk(spectrum, self.n_harmonics, dim=-1).indices.sort(dim=-1).values
        )

    def _minimize(
        self,
        parameters: list[torch.Tensor],
        objective: Any,
    ) -> None:
        optimizer = torch.optim.LBFGS(
            parameters,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance,
            tolerance_change=self.tolerance,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Wave PSF calibration objective became non-finite")
            loss.backward()
            return loss

        optimizer.step(closure)


# %% private module subroutines


def _as_pair(value: Any, name: str, *, device: Any | None) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.ndim == 0:
        result = result.repeat(2)
    if result.shape != (2,):
        raise ValueError(f"{name} must be a scalar or a pair")
    return result


def _accelerated_wave_psf(
    phase: torch.Tensor,
    phase_axis: torch.Tensor,
    partition_axis: torch.Tensor,
    *,
    cuda_kernel: bool,
) -> torch.Tensor | None:
    if phase.requires_grad or phase_axis.requires_grad or partition_axis.requires_grad:
        return None
    output = torch.empty(
        (
            phase.shape[0],
            phase.shape[-1],
            phase_axis.numel(),
            partition_axis.numel(),
        ),
        dtype=torch.complex64,
        device=phase.device,
    )
    if phase.device.type == "cpu":
        implementation = require("recon_cpu", attribute="wave_psf")
        implementation(
            phase.contiguous().numpy(),
            phase_axis.contiguous().numpy(),
            partition_axis.contiguous().numpy(),
            output.numpy(),
        )
        return output
    if phase.device.type == "cuda" and cuda_kernel:
        try:
            from ._triton_wave import wave_psf
        except ImportError as error:
            raise RuntimeError(
                "optimized CUDA Wave PSF execution requires Triton. Install a "
                "Linux PyTorch distribution that includes Triton, or run the "
                "PSF on the CPU; Pulserver will not silently select a slower "
                "CUDA path."
            ) from error
        wave_psf(
            phase.contiguous(),
            phase_axis.contiguous(),
            partition_axis.contiguous(),
            output,
        )
        return output
    return None


def _axis_spacing(axis: torch.Tensor) -> torch.Tensor:
    if axis.numel() < 2:
        return axis.new_tensor(1.0)
    return axis.diff().abs().median().clamp_min(torch.finfo(axis.dtype).eps)


def _fourier_resize_phase(phase: torch.Tensor, readout: int) -> torch.Tensor:
    source = int(phase.shape[-1])
    if source == readout:
        return phase
    spectrum = torch.fft.fftshift(
        torch.fft.fft(
            torch.fft.ifftshift(phase, dim=-1),
            dim=-1,
            norm="ortho",
        ),
        dim=-1,
    )
    resized = torch.zeros(
        (*phase.shape[:-1], readout),
        dtype=spectrum.dtype,
        device=phase.device,
    )
    count = min(source, readout)
    source_start = (source - count) // 2
    target_start = (readout - count) // 2
    resized[..., target_start : target_start + count] = spectrum[
        ..., source_start : source_start + count
    ]
    interpolated = torch.fft.fftshift(
        torch.fft.ifft(
            torch.fft.ifftshift(resized, dim=-1),
            dim=-1,
            norm="ortho",
        ),
        dim=-1,
    ).real
    return interpolated * (readout / source) ** 0.5


def _central_profiles(
    reconstruction: torch.Tensor,
    wave_readout: int,
    phase_axis: torch.Tensor,
    partition_axis: torch.Tensor,
    *,
    coefficient_axis: int,
    coefficient_specific: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = _pad_readout(reconstruction, wave_readout)
    if coefficient_specific:
        axis = coefficient_axis % padded.ndim
        if axis >= padded.ndim - 3:
            raise ValueError("coefficient_axis cannot be a spatial dimension")
        padded = torch.movedim(padded, axis, 0)
        models = int(padded.shape[0])
        padded = padded.reshape(models, -1, *padded.shape[-3:])
    else:
        padded = padded.reshape(1, -1, *padded.shape[-3:])
    center_y = int(torch.argmin(phase_axis.abs()))
    center_z = int(torch.argmin(partition_axis.abs()))
    profile_y = padded[..., center_z]
    profile_z = padded[..., center_y, :]
    return _normalize_profiles(profile_y), _normalize_profiles(profile_z)


def _normalize_profiles(profiles: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(profiles, dim=(-2, -1), keepdim=True)
    return profiles / norm.clamp_min(1e-12)


def _harmonic_basis(
    indices: torch.Tensor,
    readout: int,
    device: Any,
) -> torch.Tensor:
    samples = torch.arange(readout, dtype=torch.float32, device=device)
    angle = 2.0 * pi * indices.to(torch.float32)[..., None] * samples / readout
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)


def _phase_from_coefficients(
    coefficients: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("maqh,aqhx->max", coefficients, basis)


def _profile_error_psfs(
    phase_error: torch.Tensor,
    corrected_phase: torch.Tensor,
    offsets: torch.Tensor,
    phase_axis: torch.Tensor,
    partition_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    angle_y = (
        phase_error[:, 0, :, None] * phase_axis[None, None]
        + corrected_phase[:, 0, :, None] * offsets[:, 0, None, None]
    )
    angle_z = (
        phase_error[:, 1, :, None] * partition_axis[None, None]
        + corrected_phase[:, 1, :, None] * offsets[:, 1, None, None]
    )
    error_y = torch.complex(torch.cos(angle_y), -torch.sin(angle_y))
    error_z = torch.complex(torch.cos(angle_z), -torch.sin(angle_z))
    return error_y, error_z


def _edge_objective(
    spectra_y: torch.Tensor,
    spectra_z: torch.Tensor,
    error_y: torch.Tensor,
    error_z: torch.Tensor,
    *,
    sigmoid_scale: float,
    cross_edges: bool,
) -> torch.Tensor:
    projected_y = _ifftc(spectra_y * error_y[:, None], -2)
    projected_z = _ifftc(spectra_z * error_z[:, None], -2)
    edges = [
        projected_y - torch.roll(projected_y, 1, dims=-2),
        projected_z - torch.roll(projected_z, 1, dims=-2),
    ]
    if cross_edges:
        edges = [edge - torch.roll(edge, 1, dims=-1) for edge in edges]
    magnitudes = torch.cat([edge.abs().flatten() for edge in edges])
    if sigmoid_scale > 0.0:
        magnitudes = torch.sigmoid(sigmoid_scale * magnitudes) - 0.5
    return magnitudes.mean()


def _apply_error_psf(
    reconstruction: torch.Tensor,
    error_psf: torch.Tensor,
    *,
    coefficient_axis: int | None,
) -> torch.Tensor:
    original_readout = int(reconstruction.shape[-3])
    padded = _pad_readout(reconstruction, int(error_psf.shape[-3]))
    if coefficient_axis is None:
        multiplier = error_psf[0] if error_psf.ndim == 4 else error_psf
        corrected = _ifftc(_fftc(padded, -3) * multiplier, -3)
    else:
        axis = coefficient_axis % padded.ndim
        moved = torch.movedim(padded, axis, -4)
        if moved.shape[-4] != error_psf.shape[0]:
            raise ValueError("coefficient count and calibrated PSF bank disagree")
        shape = (*([1] * (moved.ndim - 4)), *error_psf.shape)
        corrected = _ifftc(
            _fftc(moved, -3) * error_psf.reshape(shape),
            -3,
        )
        corrected = torch.movedim(corrected, -4, axis)
    return _crop_readout(corrected, original_readout)


def _pad_readout(value: torch.Tensor, readout: int) -> torch.Tensor:
    if value.shape[-3] == readout:
        return value
    if value.shape[-3] > readout:
        raise ValueError("target readout must not be smaller than the input")
    shape = (*value.shape[:-3], readout, *value.shape[-2:])
    output = torch.zeros(shape, dtype=value.dtype, device=value.device)
    start = (readout - value.shape[-3]) // 2
    output[..., start : start + value.shape[-3], :, :] = value
    return output


def _crop_readout(value: torch.Tensor, readout: int) -> torch.Tensor:
    if value.shape[-3] == readout:
        return value
    start = (value.shape[-3] - readout) // 2
    return value[..., start : start + readout, :, :]


def _fftc(value: torch.Tensor, dim: int) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=dim)
    value = torch.fft.fft(value, dim=dim, norm="ortho")
    return torch.fft.fftshift(value, dim=dim)


def _ifftc(value: torch.Tensor, dim: int) -> torch.Tensor:
    value = torch.fft.ifftshift(value, dim=dim)
    value = torch.fft.ifft(value, dim=dim, norm="ortho")
    return torch.fft.fftshift(value, dim=dim)
