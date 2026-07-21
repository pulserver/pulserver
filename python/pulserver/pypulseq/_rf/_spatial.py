"""Spectral-spatial and arbitrary-trajectory spatial RF pulse design."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pypulseq as pp

from ._base import RfModule
from ._slr import design_slr


class SpatialPulse(RfModule):
    """RF event, simultaneous encoding gradients, and their rephasers."""

    def __init__(self, system, rf, gradients, rephasers, kspace, self_refocused) -> None:
        self.rf = rf
        self.gradients = tuple(gradients)
        self.rephasers = tuple(rephasers)
        self.kspace = np.asarray(kspace)
        self.self_refocused = bool(self_refocused)
        blocks = [(rf, *self.gradients)]
        if self.rephasers:
            blocks.append(self.rephasers)
        super().__init__(system, blocks)


def _as_tuple(value, ndim: int, name: str, cast=float) -> tuple:
    result = (cast(value),) * ndim if np.isscalar(value) else tuple(cast(item) for item in value)
    if len(result) != ndim:
        raise ValueError(f"{name} must have {ndim} entries")
    return result


def _target_and_coordinates(
    matrix: tuple[int, ...],
    fov: tuple[float, ...],
    selective_size: tuple[float, ...] | None,
    target: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    axes = [(np.arange(n) - (n - 1) / 2.0) * (extent / n) for n, extent in zip(matrix, fov, strict=True)]
    mesh = np.meshgrid(*axes, indexing="ij")
    coordinates = np.stack([axis.ravel() for axis in mesh], axis=1)
    if target is not None:
        target = np.asarray(target, dtype=np.complex128)
        if target.shape != matrix:
            raise ValueError(f"target shape {target.shape} does not match matrix {matrix}")
        return target.ravel(), coordinates

    if selective_size is None:
        selective_size = tuple(0.5 * extent for extent in fov)
    radius_squared = np.zeros(coordinates.shape[0])
    for dim, diameter in enumerate(selective_size):
        if not 0 < diameter <= fov[dim]:
            raise ValueError("each selective_size entry must lie in (0, fov]")
        radius_squared += (coordinates[:, dim] / (diameter / 2.0)) ** 2
    return (radius_squared <= 1.0).astype(np.complex128), coordinates


def _small_tip_weights(target: np.ndarray, coordinates: np.ndarray, kspace: np.ndarray) -> np.ndarray:
    """Evaluate the target's Fourier transform along an excitation trajectory."""
    weights = np.empty(kspace.shape[0], dtype=np.complex128)
    # Bound the temporary matrix to about 64 MiB even for long trajectories.
    chunk = max(1, int(4_000_000 / max(1, coordinates.shape[0])))
    for start in range(0, kspace.shape[0], chunk):
        stop = min(start + chunk, kspace.shape[0])
        phase = coordinates @ kspace[start:stop].T
        weights[start:stop] = target @ np.exp(-2j * np.pi * phase) / target.size
    if np.allclose(weights, 0.0):
        raise ValueError("the requested target has zero response on the supplied trajectory")
    return weights


def _gradient_events_and_rephasers(
    gradient_t_per_m: np.ndarray,
    axes: tuple[str, ...],
    system: pp.Opts,
) -> tuple[tuple, tuple]:
    gradient_hz_per_m = gradient_t_per_m * system.gamma
    gradients = []
    rephasers = []
    for dim, axis in enumerate(axes):
        waveform = np.ascontiguousarray(gradient_hz_per_m[:, dim])
        if np.allclose(waveform, 0.0):
            continue
        event = pp.make_arbitrary_grad(
            channel=axis,
            waveform=waveform,
            first=0.0,
            last=0.0,
            system=system,
        )
        gradients.append(event)
        if not np.isclose(event.area, 0.0, atol=1e-9):
            rephasers.append(pp.make_trapezoid(channel=axis, area=-event.area, system=system))
    return tuple(gradients), tuple(rephasers)


def make_spatially_selective_pulse(
    flip_angle: float,
    gradient: np.ndarray,
    fov: float | Sequence[float],
    matrix: int | Sequence[int],
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    axes: Sequence[str] | None = None,
    system: pp.Opts | None = None,
    use: str = "excitation",
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
) -> SpatialPulse:
    """Design a 2D or 3D small-tip pulse on an existing gradient trajectory.

    ``gradient`` is in T/m with shape ``(samples, 2)`` or ``(samples, 3)``.
    It is intentionally supplied by the caller so EPI trajectories can come
    from pypulseq and spiral trajectories from :mod:`pulserver.pypulseq.arbgrad`.
    ``target`` may specify a complex desired profile; otherwise an ellipse or
    ellipsoid with diameter ``selective_size`` is used.
    """
    system = pp.Opts.default if system is None else system
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] not in (2, 3) or gradient.shape[0] < 2:
        raise ValueError("gradient must have shape (samples, 2) or (samples, 3)")
    ndim = gradient.shape[1]
    fov = _as_tuple(fov, ndim, "fov")
    matrix = _as_tuple(matrix, ndim, "matrix", int)
    size = None if selective_size is None else _as_tuple(selective_size, ndim, "selective_size")
    axes = tuple(("x", "y", "z")[:ndim] if axes is None else axes)
    if len(axes) != ndim or len(set(axes)) != ndim or any(axis not in ("x", "y", "z") for axis in axes):
        raise ValueError("axes must contain two or three distinct gradient channels")
    if any(n < 2 for n in matrix) or any(extent <= 0 for extent in fov):
        raise ValueError("matrix entries must be >= 2 and fov entries must be > 0")
    if not np.allclose(gradient[[0, -1]], 0.0, atol=1e-9):
        raise ValueError("gradient must ramp from and return to zero")

    dwell = system.grad_raster_time
    gradient_hz_per_m = gradient * system.gamma
    # Excitation k-space at an RF sample is the gradient moment remaining
    # after that sample. This convention directly includes the final phase.
    kspace = -np.cumsum(gradient_hz_per_m[::-1], axis=0)[::-1] * dwell
    desired, coordinates = _target_and_coordinates(matrix, fov, size, target)
    weights = _small_tip_weights(desired, coordinates, kspace)
    rf = pp.make_arbitrary_rf(
        signal=weights,
        flip_angle=flip_angle,
        dwell=dwell,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        system=system,
        use=use,
    )
    gradients, rephasers = _gradient_events_and_rephasers(gradient, axes, system)
    return SpatialPulse(
        system=system,
        rf=rf,
        gradients=gradients,
        rephasers=rephasers,
        kspace=kspace,
        self_refocused=len(rephasers) == 0,
    )


def make_spiral_selective_pulse(
    flip_angle: float,
    fov: float,
    matrix: int,
    *,
    selective_size: float | Sequence[float] | None = None,
    target: np.ndarray | None = None,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 2D selective pulse using ``pulserver.pypulseq.arbgrad.spiral``."""
    # Keep the compiled arbgrad extension optional for all other RF designs.
    from .. import _arbgrad as arbgrad

    system = pp.Opts.default if system is None else system
    native = arbgrad.spiral(
        fov=fov,
        n_pix=matrix,
        slew_limit=system.max_slew * fov / matrix,
        grad_limit=system.max_grad * fov / matrix,
        dt=system.grad_raster_time,
    )
    gradient = arbgrad.to_gradient_tesla_per_meter(native, fov, matrix, system.gamma)
    return make_spatially_selective_pulse(
        flip_angle,
        gradient[:, :2],
        (fov, fov),
        (matrix, matrix),
        selective_size=selective_size,
        target=target,
        system=system,
        **kwargs,
    )


def make_2d_selective_pulse(
    flip_angle: float,
    fov: float | Sequence[float],
    matrix: int | Sequence[int],
    *,
    gradient: np.ndarray | None = None,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 2D selective pulse on a supplied trajectory or a spiral.

    Omitting ``gradient`` selects the ``pulserver.pypulseq.arbgrad`` spiral path and
    therefore requires square ``fov`` and ``matrix`` values.
    """
    if gradient is None:
        if not np.isscalar(fov) or not np.isscalar(matrix):
            raise ValueError("the automatic spiral path requires scalar fov and matrix")
        return make_spiral_selective_pulse(
            flip_angle,
            float(fov),
            int(matrix),
            system=system,
            **kwargs,
        )
    return make_spatially_selective_pulse(
        flip_angle,
        gradient,
        fov,
        matrix,
        system=system,
        **kwargs,
    )


def make_3d_selective_pulse(
    flip_angle: float,
    gradient: np.ndarray,
    fov: Sequence[float],
    matrix: Sequence[int],
    *,
    system: pp.Opts | None = None,
    **kwargs,
) -> SpatialPulse:
    """Create a 3D selective pulse on an existing EPI/spiral trajectory."""
    gradient = np.asarray(gradient)
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError("gradient must have shape (samples, 3)")
    return make_spatially_selective_pulse(
        flip_angle,
        gradient,
        fov,
        matrix,
        system=system,
        **kwargs,
    )


def make_spsp_pulse(
    flip_angle: float,
    slice_thickness: float,
    spectral_bandwidth: float,
    *,
    freq_offset: float = 0.0,
    spatial_time_bandwidth_product: float = 4.0,
    spectral_time_bandwidth_product: float = 3.0,
    n_subpulses: int = 10,
    system: pp.Opts | None = None,
    use: str = "excitation",
) -> SpatialPulse:
    """Create an alternating-gradient SLR spectral-spatial pulse.

    The construction follows Peder Larson's compact SPSP example, replacing
    both its spectral and spatial sinc envelopes with SLR designs.
    """
    system = pp.Opts.default if system is None else system
    if slice_thickness <= 0 or spectral_bandwidth <= 0:
        raise ValueError("slice_thickness and spectral_bandwidth must be > 0")
    if n_subpulses < 4:
        raise ValueError("n_subpulses must be >= 4")
    if n_subpulses % 2:
        n_subpulses += 1

    dwell = system.grad_raster_time
    total_duration = spectral_time_bandwidth_product / spectral_bandwidth
    samples_per_lobe = int(round(total_duration / (n_subpulses * dwell)))
    if samples_per_lobe < 8:
        raise ValueError("spectral bandwidth is too large for the requested subpulse count")

    # Reserve 20% of each lobe for ramps. If the resulting amplitude cannot
    # meet the system slew limit, the requested selectivity is infeasible.
    ramp_samples = max(1, int(round(0.1 * samples_per_lobe)))
    flat_samples = samples_per_lobe - 2 * ramp_samples
    if flat_samples < 8:
        raise ValueError("SPSP sublobes leave fewer than 8 RF samples")
    if flat_samples % 2:
        flat_samples -= 1
        samples_per_lobe -= 1
    flat_time = flat_samples * dwell
    amplitude_hz_per_m = spatial_time_bandwidth_product / (flat_time * slice_thickness)
    if amplitude_hz_per_m > system.max_grad:
        raise ValueError("SPSP slice-selection amplitude exceeds system.max_grad")
    slew = amplitude_hz_per_m / (ramp_samples * dwell)
    if slew > system.max_slew * (1.0 + 1e-9):
        raise ValueError("SPSP slice-selection ramps exceed system.max_slew")

    ramp_up = amplitude_hz_per_m * (np.arange(ramp_samples) + 0.5) / ramp_samples
    flat = np.full(flat_samples, amplitude_hz_per_m)
    ramp_down = ramp_up[::-1]
    positive_lobe = np.concatenate((ramp_up, flat, ramp_down))
    spatial = design_slr(flat_samples, spatial_time_bandwidth_product, pulse_type="st", filter_type="ls")
    spectral = design_slr(n_subpulses, spectral_time_bandwidth_product, pulse_type="st", filter_type="ls")

    gradient_hz_per_m = np.empty(n_subpulses * positive_lobe.size)
    rf_shape = np.zeros_like(gradient_hz_per_m, dtype=np.complex128)
    for index in range(n_subpulses):
        start = index * positive_lobe.size
        stop = start + positive_lobe.size
        sign = 1.0 if index % 2 == 0 else -1.0
        gradient_hz_per_m[start:stop] = sign * positive_lobe
        rf_shape[start + ramp_samples : start + ramp_samples + flat_samples] = spectral[index] * spatial

    rf = pp.make_arbitrary_rf(
        signal=rf_shape,
        flip_angle=flip_angle,
        dwell=dwell,
        freq_offset=freq_offset,
        system=system,
        use=use,
    )
    gz = pp.make_arbitrary_grad(
        channel="z",
        waveform=np.ascontiguousarray(gradient_hz_per_m),
        first=0.0,
        last=0.0,
        system=system,
    )
    # Rephase from the center of the final spatial subpulse. The alternating
    # full-lobe areas cancel for even n_subpulses, but this half-lobe remains.
    last_sign = -1.0
    rephase_area = -last_sign * 0.5 * amplitude_hz_per_m * (flat_time + ramp_samples * dwell)
    gz_reph = pp.make_trapezoid(channel="z", area=rephase_area, system=system)
    kspace = -np.cumsum(gradient_hz_per_m[::-1])[::-1, None] * dwell
    return SpatialPulse(
        system=system,
        rf=rf,
        gradients=(gz,),
        rephasers=(gz_reph,),
        kspace=kspace,
        self_refocused=False,
    )
