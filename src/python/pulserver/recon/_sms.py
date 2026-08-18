"""Simultaneous-multislice data contracts and private encoding machinery."""

from __future__ import annotations

__all__ = ["SmsEpiInputs"]

from dataclasses import dataclass
from typing import Any

import torch

import deepinv


@dataclass(frozen=True)
class SmsEpiInputs:
    """Inputs a simultaneous-multislice reconstruction needs, checked together.

    Parameters
    ----------
    imaging
        Multiband data, encoded by whatever trajectory the sequence played.
    coil_maps
        Per-slice coil maps for the model-based separation. Keep this tensor
        on the selected Torch CPU/CUDA device.
    single_band_reference
        Single-band prescan the coil maps can be derived from, when they were
        not estimated another way.
    caipi_encoding
        The CAIPI phase/modulation model that was played. Required whenever
        the multiband factor is greater than one, since it is what tells the
        simultaneously excited slices apart.

    Notes
    -----
    This class collects and checks the inputs; the separation itself is
    :class:`pulserver.recon.physics.SMS`, which composes with any in-plane
    physics and so does not restrict the acquisition to a Cartesian encode.
    """

    imaging: Any
    coil_maps: Any | None = None
    single_band_reference: Any | None = None
    caipi_encoding: Any | None = None

    def validate(self, multiband_factor: int) -> None:
        """Raise ``ValueError`` when an SMS backend lacks essential inputs."""
        if multiband_factor < 1:
            raise ValueError("multiband_factor must be at least one")
        if multiband_factor > 1 and self.caipi_encoding is None:
            raise ValueError(
                "SMS reconstruction requires the acquired CAIPI encoding model"
            )
        if (
            multiband_factor > 1
            and self.coil_maps is None
            and self.single_band_reference is None
        ):
            raise ValueError(
                "SMS reconstruction requires coil maps or a single-band reference"
            )


# %% private module subroutines


class _SMSLinearPhysics(deepinv.physics.LinearPhysics):
    """Collapse and unalias slices around one or more linear MRI operators."""

    def __init__(
        self,
        physics: Any | list[Any],
        caipi_encoding: Any | None,
        n_slices: int | None,
    ) -> None:
        super().__init__()
        if isinstance(physics, (list, tuple)):
            if not physics:
                raise ValueError("SMS physics cannot be empty")
            self.physics = torch.nn.ModuleList(physics)
            self.shared = False
            inferred_slices = len(physics)
        else:
            self.physics = torch.nn.ModuleList([physics])
            self.shared = True
            inferred_slices = n_slices
        if caipi_encoding is not None:
            encoding = torch.as_tensor(caipi_encoding)
            if encoding.ndim == 0:
                raise ValueError("caipi_encoding must start with the slice axis")
            if inferred_slices is None:
                inferred_slices = int(encoding.shape[0])
            if encoding.shape[0] != inferred_slices:
                raise ValueError(
                    "caipi_encoding slice count does not match the SMS physics"
                )
            if not encoding.is_complex():
                if not encoding.is_floating_point():
                    encoding = encoding.to(torch.float32)
                encoding = torch.polar(torch.ones_like(encoding), encoding)
            else:
                epsilon = torch.finfo(encoding.real.dtype).eps
                if bool(torch.any(encoding.abs() <= epsilon)):
                    raise ValueError("complex caipi_encoding cannot contain zeros")
                encoding = encoding / encoding.abs().clamp_min(epsilon)
        else:
            if inferred_slices is None:
                raise ValueError("n_slices is required without caipi_encoding")
            encoding = torch.ones(inferred_slices, dtype=torch.complex64)
        if inferred_slices is None or inferred_slices < 1:
            raise ValueError("n_slices must be positive")
        viewed_as_real = {
            bool(getattr(item, "viewed_as_real", False)) for item in self.physics
        }
        if len(viewed_as_real) != 1:
            raise ValueError("all SMS physics must use the same complex representation")
        self.register_buffer("caipi_encoding", encoding.contiguous())
        self.n_slices = int(inferred_slices)
        self.viewed_as_real = viewed_as_real.pop()

    def A(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode slices and collapse them into one multiband measurement."""
        if value.ndim < 3 or value.shape[1] != self.n_slices:
            raise ValueError("SMS input must have shape (batch, slices, ...)")
        batch = value.shape[0]
        if self.shared:
            flattened = value.reshape(batch * self.n_slices, *value.shape[2:])
            encoded = self.physics[0].A(flattened, **kwargs)
            encoded = encoded.reshape(batch, self.n_slices, *encoded.shape[1:])
        else:
            encoded = torch.stack(
                [
                    selected.A(value[:, index], **kwargs)
                    for index, selected in enumerate(self.physics)
                ],
                dim=1,
            )
        return self._modulate(encoded, conjugate=False).sum(dim=1)

    def A_adjoint(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the slice-separated Hermitian adjoint."""
        expanded = value[:, None].expand(
            value.shape[0],
            self.n_slices,
            *value.shape[1:],
        )
        demodulated = self._modulate(expanded, conjugate=True)
        if self.shared:
            flattened = demodulated.reshape(
                value.shape[0] * self.n_slices,
                *value.shape[1:],
            )
            decoded = self.physics[0].A_adjoint(flattened, **kwargs)
            return decoded.reshape(
                value.shape[0],
                self.n_slices,
                *decoded.shape[1:],
            )
        return torch.stack(
            [
                selected.A_adjoint(demodulated[:, index], **kwargs)
                for index, selected in enumerate(self.physics)
            ],
            dim=1,
        )

    def A_adjoint_A(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the exact coupled slice normal operator."""
        return self.A_adjoint(self.A(value, **kwargs), **kwargs)

    def enable_streaming(self, policy: Any) -> None:
        """Forward bounded/multi-GPU execution to each base physics object."""
        for selected in self.physics:
            method = getattr(selected, "enable_streaming", None)
            if method is not None:
                method(policy)

    @property
    def streaming_methods(self) -> tuple[str, ...]:
        """Declare that slice batching is already stream-aware."""
        return ("A", "A_adjoint", "A_adjoint_A")

    def _modulate(
        self,
        value: torch.Tensor,
        *,
        conjugate: bool,
    ) -> torch.Tensor:
        if self.viewed_as_real:
            if value.shape[-1] != 2:
                raise ValueError("paired-real SMS measurements must end in length two")
            complex_value = torch.view_as_complex(value.contiguous())
        elif value.is_complex():
            complex_value = value
        else:
            raise TypeError("complex SMS measurements require a complex tensor")
        encoding = self.caipi_encoding.to(
            device=complex_value.device,
            dtype=complex_value.dtype,
        )
        trailing = encoding.ndim - 1
        if trailing > complex_value.ndim - 2:
            raise ValueError("caipi_encoding has too many measurement dimensions")
        encoding = encoding.reshape(
            1,
            self.n_slices,
            *([1] * (complex_value.ndim - 2 - trailing)),
            *encoding.shape[1:],
        )
        result = complex_value * (encoding.conj() if conjugate else encoding)
        return torch.view_as_real(result) if self.viewed_as_real else result
