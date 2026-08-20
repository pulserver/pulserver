"""Private simultaneous-multislice encoding machinery."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import torch

import deepinv

# %% private module subroutines


def _fits_one_call(physics: Any, batch: int) -> bool:
    """Whether ``physics`` encodes ``batch`` images in a single call.

    An mri-nufft operator plans for a fixed batch size and refuses any other;
    a physics that carries no such plan takes whatever batch it is given.
    """
    planned = getattr(getattr(physics, "native_operator", None), "n_batchs", None)
    return planned is None or int(planned) == batch


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
        if self.shared and _fits_one_call(self.physics[0], batch * self.n_slices):
            flattened = value.reshape(batch * self.n_slices, *value.shape[2:])
            encoded = self.physics[0].A(flattened, **kwargs)
            encoded = encoded.reshape(batch, self.n_slices, *encoded.shape[1:])
        else:
            encoded = torch.stack(
                [
                    self._for_slice(index).A(value[:, index], **kwargs)
                    for index in range(self.n_slices)
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
        batch = value.shape[0]
        if self.shared and _fits_one_call(self.physics[0], batch * self.n_slices):
            flattened = demodulated.reshape(
                batch * self.n_slices,
                *value.shape[1:],
            )
            decoded = self.physics[0].A_adjoint(flattened, **kwargs)
            return decoded.reshape(
                batch,
                self.n_slices,
                *decoded.shape[1:],
            )
        return torch.stack(
            [
                self._for_slice(index).A_adjoint(demodulated[:, index], **kwargs)
                for index in range(self.n_slices)
            ],
            dim=1,
        )

    def _for_slice(self, index: int) -> Any:
        """The physics that encodes slice ``index``."""
        return self.physics[0] if self.shared else self.physics[index]

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
