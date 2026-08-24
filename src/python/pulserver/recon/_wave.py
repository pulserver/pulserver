"""Memory-bounded Wave and Wave-Shuffling linear physics."""

from __future__ import annotations

__all__: list[str] = []

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from ..mrd._fourier import fftc as _fftc
from ..mrd._fourier import ifftc as _ifftc
from ..mrd._fourier import resize_centered_axis as _resize_centered

import deepinv

from ._packed import PackedHermitianField


class _WaveLinearPhysics(deepinv.physics.LinearPhysics):
    """Fused SENSE, Wave, temporal-basis, and line-sampling operator."""

    def __init__(
        self,
        sampling: Any,
        coil_maps: Any,
        wave_psf: Any,
        basis: Any,
        *,
        line_weights: Any | None = None,
        viewed_as_real: bool = False,
        coil_batch_size: int = 1,
        cuda_transfer_precision: str = "auto",
    ) -> None:
        super().__init__()
        sampling = torch.as_tensor(sampling)
        coil_maps = torch.as_tensor(coil_maps)
        wave_psf = torch.as_tensor(getattr(wave_psf, "psf", wave_psf))
        basis = torch.as_tensor(basis)
        if sampling.ndim != 2 or sampling.shape[1] not in {2, 3}:
            raise ValueError("sampling must have shape (lines, 2) or (lines, 3)")
        if sampling.is_floating_point() or sampling.is_complex():
            rounded = sampling.real.round()
            if not torch.equal(sampling.real, rounded):
                raise ValueError("sampling entries must be integer indices")
            sampling = rounded
        sampling = sampling.to(torch.int64)
        if sampling.shape[1] == 2:
            sampling = torch.cat(
                [
                    sampling,
                    torch.zeros(
                        (sampling.shape[0], 1),
                        dtype=sampling.dtype,
                        device=sampling.device,
                    ),
                ],
                dim=1,
            )
        if basis.ndim != 2:
            raise ValueError("basis must have shape (rank, echoes)")
        if coil_maps.ndim == 4:
            coil_maps = coil_maps[:, None]
        if coil_maps.ndim != 5:
            raise ValueError(
                "coil_maps must have shape (coils, *image_shape) or "
                "(coils, maps, *image_shape)"
            )
        if wave_psf.ndim not in {3, 4}:
            raise ValueError(
                "wave_psf must have shape (readout, phase, partition) or "
                "(rank, readout, phase, partition)"
            )
        if coil_batch_size <= 0:
            raise ValueError("coil_batch_size must be positive")

        rank, echoes = map(int, basis.shape)
        n_coils, n_maps, sx, sy, sz = map(int, coil_maps.shape)
        if wave_psf.ndim == 4 and wave_psf.shape[0] not in {1, rank}:
            raise ValueError(
                "rank-dependent wave_psf must have one PSF per coefficient"
            )
        wx = int(wave_psf.shape[-3])
        if tuple(wave_psf.shape[-2:]) != (sy, sz):
            raise ValueError("wave_psf phase dimensions must match coil_maps")
        if sampling.numel():
            if int(sampling[:, 0].min()) < 0 or int(sampling[:, 0].max()) >= sy:
                raise ValueError("sampling contains an out-of-bounds phase index")
            if int(sampling[:, 1].min()) < 0 or int(sampling[:, 1].max()) >= sz:
                raise ValueError("sampling contains an out-of-bounds partition index")
            if int(sampling[:, 2].min()) < 0 or int(sampling[:, 2].max()) >= echoes:
                raise ValueError("sampling contains an out-of-bounds echo index")

        if line_weights is None:
            line_weights = torch.ones(
                sampling.shape[0],
                dtype=torch.float32,
                device=sampling.device,
            )
        else:
            line_weights = torch.as_tensor(line_weights)
            if line_weights.ndim != 1 or line_weights.numel() != sampling.shape[0]:
                raise ValueError("line_weights must contain one value per sampled line")

        device = coil_maps.device
        sampling = sampling.to(device)
        wave_psf = wave_psf.to(device)
        basis = basis.to(device)
        line_weights = line_weights.to(device)

        self.register_buffer("sampling", sampling.contiguous())
        self.register_buffer("coil_maps", coil_maps.to(torch.complex64).contiguous())
        self.register_buffer("wave_psf", wave_psf.to(torch.complex64).contiguous())
        self.register_buffer(
            "basis",
            basis.to(
                torch.complex64 if basis.is_complex() else torch.float32
            ).contiguous(),
        )
        self.register_buffer(
            "line_weights",
            line_weights.to(
                torch.complex64 if line_weights.is_complex() else torch.float32
            ).contiguous(),
        )
        self.rank = rank
        self.n_coils = n_coils
        self.n_maps = n_maps
        self.image_shape = (sx, sy, sz)
        self.wave_shape = (wx, sy, sz)
        self.viewed_as_real = bool(viewed_as_real)
        self.coil_batch_size = int(coil_batch_size)
        self.cuda_transfer_precision = cuda_transfer_precision
        self.field = PackedHermitianField(
            self._build_field_values(),
            rank,
            cuda_transfer_precision=cuda_transfer_precision,
        )
        self.streaming_policy = None
        self.streaming_methods = {"A", "A_adjoint", "A_adjoint_A"}
        self._stream_replicas: list[tuple[Any, Any, _WaveLinearPhysics, int, int]] = []

    def A(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Encode coefficient images directly into the acquired line table."""
        del kwargs
        if self._should_stream(x):
            return self._streamed_call("A", x)
        coefficients = self._image_as_complex(x)
        chunks = []
        for start in range(0, self.n_coils, self.coil_batch_size):
            maps = self.coil_maps[start : start + self.coil_batch_size]
            coil_coefficients = self._sense_forward(coefficients, maps)
            hybrid = self._wave_forward(coil_coefficients)
            chunks.append(self._sample(hybrid))
        result = torch.cat(chunks, dim=1)
        return torch.view_as_real(result) if self.viewed_as_real else result

    def A_adjoint(self, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply indexed temporal collapse and the adjoint Wave-SENSE chain."""
        del kwargs
        if self._should_stream(y):
            return self._streamed_call("A_adjoint", y)
        table = self._kspace_as_complex(y)
        result = None
        for start in range(0, self.n_coils, self.coil_batch_size):
            stop = min(start + self.coil_batch_size, self.n_coils)
            hybrid = self._sample_adjoint(table[:, start:stop])
            coil_coefficients = self._wave_adjoint(hybrid)
            contribution = self._sense_adjoint(
                coil_coefficients,
                self.coil_maps[start:stop],
            )
            result = contribution if result is None else result + contribution
        assert result is not None
        return self._image_from_complex(result)

    def A_adjoint_A(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Apply the exact packed temporal normal in hybrid k-space."""
        del kwargs
        if self._should_stream(x):
            return self._streamed_call("A_adjoint_A", x)
        if torch.is_grad_enabled() and (
            self.basis.requires_grad or self.line_weights.requires_grad
        ):
            return self.A_adjoint(self.A(x))
        coefficients = self._image_as_complex(x)
        result = torch.zeros_like(coefficients)
        for start in range(0, self.n_coils, self.coil_batch_size):
            maps = self.coil_maps[start : start + self.coil_batch_size]
            hybrid = self._wave_forward(self._sense_forward(coefficients, maps))
            batch, coils, rank, wx, sy, sz = hybrid.shape
            bank = hybrid.permute(0, 1, 3, 2, 4, 5).reshape(
                batch * coils * wx,
                rank,
                sy * sz,
            )
            bank = self.field(bank)
            hybrid = bank.reshape(batch, coils, wx, rank, sy, sz).permute(
                0,
                1,
                3,
                2,
                4,
                5,
            )
            result += self._sense_adjoint(self._wave_adjoint(hybrid), maps)
        return self._image_from_complex(result)

    def enable_streaming(self, policy: Any) -> None:
        """Enable host-resident, coil-parallel CUDA execution."""
        self.streaming_policy = policy
        self._stream_replicas.clear()

    # %% private module subroutines

    def _build_field_values(self) -> torch.Tensor:
        rows, columns = torch.triu_indices(
            self.rank,
            self.rank,
            device=self.basis.device,
        )
        echoes = self.sampling[:, 2]
        coefficients = (
            self.basis[rows[:, None], echoes[None]]
            * self.basis[columns[:, None], echoes[None]].conj()
        )
        coefficients = coefficients * self.line_weights.abs().square()[None]
        locations = self.sampling[:, 0] * self.image_shape[2] + self.sampling[:, 1]
        values = torch.zeros(
            (rows.numel(), self.image_shape[1] * self.image_shape[2]),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        values.index_add_(1, locations, coefficients)
        if not self.basis.is_complex() and not self.line_weights.is_complex():
            values = values.real
        return values.contiguous()

    def _image_as_complex(self, value: torch.Tensor) -> torch.Tensor:
        if self.viewed_as_real:
            batch, channels, *spatial = value.shape
            if channels % 2:
                raise ValueError("real-view images require channel pairs")
            value = value.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
            value = torch.view_as_complex(value.contiguous())
        if value.ndim == 5:
            if value.shape[1] == self.n_maps * self.rank:
                value = value.reshape(
                    value.shape[0],
                    self.n_maps,
                    self.rank,
                    *value.shape[2:],
                )
            elif self.n_maps == 1:
                value = value[:, None]
        if value.ndim != 6:
            raise ValueError(
                "coefficient images must have shape (batch, maps, rank, x, y, z)"
            )
        if value.shape[1:3] != (self.n_maps, self.rank):
            raise ValueError("coefficient map/rank dimensions do not match physics")
        if tuple(value.shape[-3:]) != self.image_shape:
            raise ValueError("coefficient image shape does not match coil_maps")
        return value

    def _image_from_complex(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(1, 2)
        if not self.viewed_as_real:
            return value
        batch, channels, *spatial = value.shape
        value = torch.view_as_real(value).movedim(-1, 2)
        return value.reshape(batch, channels * 2, *spatial)

    def _kspace_as_complex(self, value: torch.Tensor) -> torch.Tensor:
        if self.viewed_as_real:
            if value.shape[-1] != 2:
                raise ValueError("real-view k-space requires a trailing pair")
            value = torch.view_as_complex(value.contiguous())
        if value.ndim != 4 or value.shape[1:] != (
            self.n_coils,
            self.sampling.shape[0],
            self.wave_shape[0],
        ):
            raise ValueError("k-space table shape does not match Wave physics")
        return value

    @staticmethod
    def _sense_forward(coefficients: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bmkxyz,cmxyz->bckxyz", coefficients, maps)

    @staticmethod
    def _sense_adjoint(coil_images: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bckxyz,cmxyz->bmkxyz", coil_images, maps.conj())

    def _wave_forward(self, value: torch.Tensor) -> torch.Tensor:
        value = _resize_centered(value, self.wave_shape[0], axis=-3)
        value = _fftc(value, (-3,))
        psf = self.wave_psf
        if psf.ndim == 3:
            value = value * psf[None, None, None]
        else:
            value = value * psf[None, None]
        return _fftc(value, (-2, -1))

    def _wave_adjoint(self, value: torch.Tensor) -> torch.Tensor:
        value = _ifftc(value, (-2, -1))
        psf = self.wave_psf
        if psf.ndim == 3:
            value = value * psf.conj()[None, None, None]
        else:
            value = value * psf.conj()[None, None]
        value = _ifftc(value, (-3,))
        return _resize_centered(value, self.image_shape[0], axis=-3)

    def _sample(self, hybrid: torch.Tensor) -> torch.Tensor:
        phase = self.sampling[:, 0]
        partition = self.sampling[:, 1]
        echo = self.sampling[:, 2]
        selected = hybrid[..., phase, partition]
        result = torch.einsum(
            "bckxn,kn->bcnx",
            selected,
            self.basis[:, echo].conj().to(selected.dtype),
        )
        return result * self.line_weights[None, None, :, None]

    def _sample_adjoint(self, table: torch.Tensor) -> torch.Tensor:
        table = table * self.line_weights.conj()[None, None, :, None]
        source = torch.einsum(
            "bcnx,kn->bckxn",
            table,
            self.basis[:, self.sampling[:, 2]].to(table.dtype),
        )
        batch, coils, rank, wx, lines = source.shape
        locations = self.sampling[:, 0] * self.image_shape[2] + self.sampling[:, 1]
        grid = torch.zeros(
            (batch * coils * rank * wx, self.image_shape[1] * self.image_shape[2]),
            dtype=source.dtype,
            device=source.device,
        )
        grid.scatter_add_(
            1,
            locations.expand(grid.shape[0], lines),
            source.reshape(grid.shape[0], lines),
        )
        return grid.reshape(
            batch,
            coils,
            rank,
            wx,
            self.image_shape[1],
            self.image_shape[2],
        )

    def _should_stream(self, value: torch.Tensor) -> bool:
        return (
            self.streaming_policy is not None
            and isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
        )

    def _streamed_call(self, name: str, value: torch.Tensor) -> torch.Tensor:
        policy = self.streaming_policy
        if not self._stream_replicas:
            devices = tuple(policy.execution_devices)
            slots = min(len(devices), self.n_coils)
            boundaries = [index * self.n_coils // slots for index in range(slots + 1)]
            for index in range(slots):
                start, stop = boundaries[index], boundaries[index + 1]
                device = devices[index]
                stream = torch.cuda.Stream(device=device)
                replica = _WaveLinearPhysics(
                    self.sampling,
                    self.coil_maps[start:stop],
                    self.wave_psf,
                    self.basis,
                    line_weights=self.line_weights,
                    viewed_as_real=self.viewed_as_real,
                    coil_batch_size=self.coil_batch_size,
                    cuda_transfer_precision=self.cuda_transfer_precision,
                ).to(device)
                self._stream_replicas.append((device, stream, replica, start, stop))

        def apply_part(
            item: tuple[Any, Any, _WaveLinearPhysics, int, int],
        ) -> torch.Tensor:
            device, stream, replica, start, stop = item
            selected = value
            if name == "A_adjoint":
                selected = value[:, start:stop]
            if policy.pin_memory and not selected.is_pinned():
                host = torch.empty_like(selected, pin_memory=True)
                host.copy_(selected)
                selected = host
            with torch.cuda.device(device), torch.cuda.stream(stream):
                device_value = selected.to(
                    device,
                    non_blocking=policy.pin_memory,
                )
                result = getattr(replica, name)(device_value)
                host_result = result.to("cpu", non_blocking=policy.pin_memory)
            stream.synchronize()
            return host_result

        with ThreadPoolExecutor(max_workers=len(self._stream_replicas)) as executor:
            results = list(executor.map(apply_part, self._stream_replicas))
        if name == "A":
            return torch.cat(results, dim=1)
        return torch.stack(results).sum(dim=0)
