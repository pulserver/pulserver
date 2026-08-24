"""Packed Hermitian matrix fields shared by MRI normal operators."""

from __future__ import annotations

__all__ = ["PackedHermitianField", "packed_hermitian_matvec"]

from typing import Any

import torch

from ._toeplitz import (
    _cpu_packed_matvec,
    _cuda_transfer_values,
    _packed_cuda_matvec,
    _resolved_cuda_precision,
    as_torch,
)


class PackedHermitianField(torch.nn.Module):
    """Apply tiny Hermitian matrices over many independent locations.

    The upper triangle is stored by rows with shape
    ``(rank * (rank + 1) // 2, locations)``. CPU execution uses Pulserver's
    precompiled runtime-dispatched SIMD extension. CUDA execution keeps
    complex64 data intact, stores real fields as BF16 when supported, and
    dispatches one fused Triton program over batch-location tiles.

    Parameters
    ----------
    values
        Packed upper-triangular matrix values.
    rank
        Matrix rank.
    cuda_transfer_precision
        ``"auto"`` selects native BF16 for real fields and complex64 for
        complex fields. Explicit complex reduced precision is not supported.
    """

    def __init__(
        self,
        values: Any,
        rank: int,
        *,
        cuda_transfer_precision: str = "auto",
    ) -> None:
        super().__init__()
        values = as_torch(values)
        packed_rank = rank * (rank + 1) // 2
        if rank <= 0:
            raise ValueError("rank must be positive")
        if values.ndim != 2 or values.shape[0] != packed_rank:
            raise ValueError(
                f"values must have shape ({packed_rank}, locations), "
                f"got {tuple(values.shape)}"
            )
        if cuda_transfer_precision not in {"auto", "float32", "bfloat16"}:
            raise ValueError(
                "cuda_transfer_precision must be 'auto', 'float32', or 'bfloat16'"
            )
        if values.is_complex() and cuda_transfer_precision not in {
            "auto",
            "float32",
        }:
            raise ValueError("complex fields require complex64 CUDA storage")

        rows, columns = torch.triu_indices(rank, rank, device=values.device)
        self.register_buffer("values", values.contiguous())
        self.register_buffer("_rows", rows, persistent=False)
        self.register_buffer("_columns", columns, persistent=False)
        self.rank = int(rank)
        self.cuda_transfer_precision = cuda_transfer_precision
        self._coordinates = tuple(zip(rows.tolist(), columns.tolist(), strict=True))
        self._cuda_value_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._last_cuda_precision: str | None = None

    @property
    def is_real(self) -> bool:
        """Whether field coefficients use real storage."""
        return not self.values.is_complex()

    @property
    def n_locations(self) -> int:
        """Number of independent field locations."""
        return int(self.values.shape[1])

    @property
    def storage_nbytes(self) -> int:
        """Persistent packed-field storage in bytes."""
        return self.values.numel() * self.values.element_size()

    @property
    def last_cuda_precision(self) -> str | None:
        """Precision selected by the latest optimized CUDA call."""
        return self._last_cuda_precision

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Multiply data of shape ``(batch, rank, locations)`` by the field."""
        if not isinstance(spectrum, torch.Tensor) or not spectrum.is_complex():
            raise TypeError("spectrum must be a complex torch.Tensor")
        if spectrum.ndim != 3:
            raise ValueError("spectrum must have shape (batch, rank, locations)")
        if spectrum.shape[1:] != (self.rank, self.n_locations):
            raise ValueError(
                f"expected spectrum shape (batch, {self.rank}, "
                f"{self.n_locations}), got {tuple(spectrum.shape)}"
            )
        if self.values.device != spectrum.device:
            self.to(spectrum.device)

        differentiable = spectrum.requires_grad or self.values.requires_grad
        if spectrum.device.type != "cuda" or differentiable:
            return packed_hermitian_matvec(
                self.values,
                spectrum,
                coordinates=self._coordinates,
            )
        if spectrum.dtype != torch.complex64:
            raise TypeError("optimized CUDA field execution requires complex64 data")

        precision = self._cuda_precision(spectrum.device)
        encoded = self._encoded_values(spectrum.device, precision)
        result = spectrum.clone()
        _packed_cuda_matvec(
            result,
            encoded,
            location_start=0,
            count=self.n_locations,
        )
        self._last_cuda_precision = precision
        return result

    def _cuda_precision(self, device: torch.device) -> str:
        if self.values.is_complex():
            return "float32"
        return _resolved_cuda_precision(self.cuda_transfer_precision, device)

    def _encoded_values(self, device: torch.device, precision: str) -> torch.Tensor:
        key = (
            device,
            precision,
            self.values.data_ptr(),
            self.values._version,
        )
        encoded = self._cuda_value_cache.get(key)
        if encoded is None:
            encoded = _cuda_transfer_values(self.values, precision, device)
            self._cuda_value_cache[key] = encoded
        return encoded

    def _apply(self, function: Any, recurse: bool = True) -> PackedHermitianField:
        self._cuda_value_cache.clear()
        return super()._apply(function, recurse=recurse)


# %% private module subroutines


def packed_hermitian_matvec(
    values: torch.Tensor,
    spectrum: torch.Tensor,
    *,
    coordinates: tuple[tuple[int, int], ...] | None = None,
) -> torch.Tensor:
    """Apply one packed field without materializing dense tiny matrices."""
    rank = int(spectrum.shape[1])
    if values.shape != (rank * (rank + 1) // 2, spectrum.shape[2]):
        raise ValueError("packed values and spectrum shapes disagree")
    accelerated = _cpu_packed_matvec(values, spectrum)
    if accelerated is not None:
        return accelerated
    if coordinates is None:
        rows, columns = torch.triu_indices(rank, rank)
        coordinates = tuple(zip(rows.tolist(), columns.tolist(), strict=True))

    terms: list[list[torch.Tensor]] = [[] for _ in range(rank)]
    is_real = not values.is_complex()
    for packed_index, (row, column) in enumerate(coordinates):
        weight = values[packed_index].to(spectrum.dtype)
        terms[row].append(spectrum[:, column] * weight)
        if row != column:
            reverse = weight if is_real else weight.conj()
            terms[column].append(spectrum[:, row] * reverse)
    return torch.stack(
        [torch.stack(row_terms).sum(dim=0) for row_terms in terms],
        dim=1,
    )
