"""Torch-native phase-pole detection used by nonlinear calibration."""

from __future__ import annotations

__all__: list[str] = []

from math import ceil, cos, pi, sin

import torch


class PhasePoleCorrection(torch.nn.Module):
    """Remove common phase singularities from coil sensitivity maps.

    The detector evaluates coil phase winding on finite-radius circular paths,
    following the phase-pole model used by BART NLINV. Coil windings are
    combined by local signal energy before signed regions are clustered. In
    3D, each plane is processed independently so curved pole lines remain
    representable.

    Parameters
    ----------
    diameter
        Loop diameter as a fraction of the larger in-plane matrix dimension.
    threshold
        Absolute winding-number threshold used for pole detection.
    segments
        Number of samples around each detection loop.
    max_poles
        Maximum number of signed pole regions corrected in each plane.
    spatial_ndim
        Coil-map spatial dimensionality. It is inferred from ordinary batched
        2D/3D tensors when omitted.
    """

    def __init__(
        self,
        *,
        diameter: float = 0.05,
        threshold: float = 0.5,
        segments: int = 12,
        max_poles: int = 16,
        spatial_ndim: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < diameter < 1.0:
            raise ValueError("diameter must lie in (0, 1)")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")
        if segments < 4:
            raise ValueError("segments must be at least four")
        if max_poles < 1:
            raise ValueError("max_poles must be positive")
        if spatial_ndim not in {None, 2, 3}:
            raise ValueError("spatial_ndim must be two, three, or None")
        self.diameter = float(diameter)
        self.threshold = float(threshold)
        self.segments = int(segments)
        self.max_poles = int(max_poles)
        self.spatial_ndim = spatial_ndim
        self.detected_poles: list[list[tuple[float, float, int]]] = []

    def curl(self, sensitivities: torch.Tensor) -> torch.Tensor:
        """Return the coil-energy-weighted winding map for each 2D plane."""
        planes, leading_shape, coil_axis = _coil_planes(
            sensitivities,
            self.spatial_ndim,
        )
        curl, _ = self._plane_curl(planes)
        return _restore_planes(curl, leading_shape, coil_axis, insert_coil=False)

    def phase(self, sensitivities: torch.Tensor) -> torch.Tensor:
        """Estimate a unit-magnitude phase field containing detected poles."""
        planes, leading_shape, coil_axis = _coil_planes(
            sensitivities,
            self.spatial_ndim,
        )
        curl, radius = self._plane_curl(planes)
        phases = torch.ones(
            (planes.shape[0], *planes.shape[-2:]),
            dtype=planes.dtype,
            device=planes.device,
        )
        self.detected_poles = []
        for plane_index in range(planes.shape[0]):
            poles = _cluster_poles(
                curl[plane_index],
                radius,
                self.threshold,
                self.max_poles,
            )
            self.detected_poles.append(poles)
            phases[plane_index] = _sample_poles(
                phases[plane_index],
                poles,
            )
        return _restore_planes(phases, leading_shape, coil_axis, insert_coil=True)

    def forward(
        self,
        image: torch.Tensor,
        sensitivities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move detected pole phase from sensitivities into the image."""
        if not image.is_complex() or not sensitivities.is_complex():
            raise TypeError("phase-pole correction requires complex tensors")
        phase = self.phase(sensitivities)
        try:
            torch.broadcast_shapes(image.shape, phase.shape, sensitivities.shape)
        except RuntimeError as error:
            raise ValueError(
                "image and sensitivities must share their spatial dimensions"
            ) from error
        weight = image.abs().square()
        spatial_axes = (-2, -1)
        average = (weight * phase).sum(dim=spatial_axes, keepdim=True)
        average = average / average.abs().clamp_min(torch.finfo(average.real.dtype).eps)
        phase = phase * average.conj()
        return image * phase, sensitivities * phase.conj()

    # %% private module subroutines

    def _plane_curl(
        self,
        planes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        height, width = planes.shape[-2:]
        diameter = max(2, ceil(self.diameter * max(height, width)))
        radius = max(1, diameter // 2)
        offsets = _circle_offsets(radius, self.segments)
        winding = torch.zeros_like(planes.real)
        valid = torch.ones_like(planes.real, dtype=torch.bool)
        for first, second in zip(offsets, (*offsets[1:], offsets[0]), strict=True):
            first_value, first_valid = _shift_without_wrap(planes, first)
            second_value, second_valid = _shift_without_wrap(planes, second)
            product = first_value.conj() * second_value
            winding = winding + torch.angle(product)
            epsilon = torch.finfo(planes.real.dtype).eps
            valid = valid & first_valid & second_valid & (product.abs() > epsilon)
        winding = winding / (2 * pi)
        energy = planes.abs().square()
        weights = energy / energy.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(energy.dtype).eps
        )
        weighted = weights * winding * valid
        denominator = (
            (weights * valid).sum(dim=1).clamp_min(torch.finfo(weights.dtype).eps)
        )
        return weighted.sum(dim=1) / denominator, radius


# %% private module subroutines


def _coil_planes(
    sensitivities: torch.Tensor,
    spatial_ndim: int | None,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    if not isinstance(sensitivities, torch.Tensor) or not sensitivities.is_complex():
        raise TypeError("sensitivities must be a complex torch.Tensor")
    if sensitivities.ndim < 4:
        raise ValueError("sensitivities must have batch, coil, and two spatial axes")
    if spatial_ndim is None:
        spatial_ndim = 2 if sensitivities.ndim == 4 else 3
    if sensitivities.ndim < spatial_ndim + 2:
        raise ValueError("sensitivities have too few axes for spatial_ndim")
    coil_axis = sensitivities.ndim - spatial_ndim - 1
    if coil_axis >= sensitivities.ndim - 2:
        raise ValueError("coil dimension must precede the two in-plane dimensions")
    leading_axes = [
        index for index in range(sensitivities.ndim - 2) if index != coil_axis
    ]
    permutation = (
        *leading_axes,
        coil_axis,
        sensitivities.ndim - 2,
        sensitivities.ndim - 1,
    )
    ordered = sensitivities.permute(permutation)
    leading_shape = tuple(int(size) for size in ordered.shape[:-3])
    planes = ordered.reshape(-1, *ordered.shape[-3:])
    return planes, leading_shape, coil_axis


def _restore_planes(
    value: torch.Tensor,
    leading_shape: tuple[int, ...],
    coil_axis: int,
    *,
    insert_coil: bool,
) -> torch.Tensor:
    result = value.reshape(*leading_shape, *value.shape[-2:])
    return result.unsqueeze(coil_axis) if insert_coil else result


def _circle_offsets(radius: int, segments: int) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for index in range(segments):
        angle = 2 * pi * index / segments
        offset = (round(radius * sin(angle)), round(radius * cos(angle)))
        if not offsets or offset != offsets[-1]:
            offsets.append(offset)
    if len(offsets) > 1 and offsets[0] == offsets[-1]:
        offsets.pop()
    if len(offsets) < 4:
        return ((0, radius), (radius, 0), (0, -radius), (-radius, 0))
    return tuple(offsets)


def _shift_without_wrap(
    value: torch.Tensor,
    offset: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    dy, dx = offset
    shifted = torch.roll(value, shifts=(-dy, -dx), dims=(-2, -1))
    valid = torch.ones_like(value.real, dtype=torch.bool)
    if dy > 0:
        valid[..., -dy:, :] = False
    elif dy < 0:
        valid[..., :-dy, :] = False
    if dx > 0:
        valid[..., -dx:] = False
    elif dx < 0:
        valid[..., :-dx] = False
    return shifted, valid


def _cluster_poles(
    curl: torch.Tensor,
    radius: int,
    threshold: float,
    max_poles: int,
) -> list[tuple[float, float, int]]:
    result: list[tuple[float, float, int]] = []
    suppression = max(2.5 * radius, 2.0)
    for sign in (1, -1):
        coordinates = torch.nonzero(sign * curl >= threshold, as_tuple=False)
        strengths = sign * curl[coordinates[:, 0], coordinates[:, 1]]
        while coordinates.numel() and len(result) < max_poles:
            seed = coordinates[strengths.argmax()]
            distances = (coordinates - seed).square().sum(dim=1).sqrt()
            selected = distances <= suppression
            weights = strengths[selected]
            cluster = coordinates[selected].to(curl.dtype)
            center = (cluster * weights[:, None]).sum(dim=0) / weights.sum()
            result.append((float(center[0]), float(center[1]), sign))
            coordinates = coordinates[~selected]
            strengths = strengths[~selected]
    return result


def _sample_poles(
    phase: torch.Tensor,
    poles: list[tuple[float, float, int]],
) -> torch.Tensor:
    if not poles:
        return phase
    y = torch.arange(phase.shape[-2], device=phase.device, dtype=phase.real.dtype)
    x = torch.arange(phase.shape[-1], device=phase.device, dtype=phase.real.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    epsilon = torch.finfo(phase.real.dtype).eps
    for center_y, center_x, charge in poles:
        pole = (grid_x - center_x) + 1j * (grid_y - center_y)
        magnitude = pole.abs()
        pole = torch.where(
            magnitude > epsilon,
            pole / magnitude.clamp_min(epsilon),
            torch.ones_like(pole),
        )
        if charge < 0:
            pole = pole.conj()
        phase = phase * pole
    return phase
