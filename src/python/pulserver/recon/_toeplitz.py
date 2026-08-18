"""Memory-efficient Torch application of matrix-valued Toeplitz kernels."""

from __future__ import annotations

__all__: list[str] = []

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from math import prod
from typing import Any

from .._accelerators import require


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise ImportError(
            "Compact Toeplitz kernels require Torch, which ships with "
            "pulserver; reinstall the package to restore it."
        ) from error


def _triton_matvecs() -> tuple[Any, Any]:
    """Import required fused kernels with a clear deployment error."""
    try:
        from ._triton_toeplitz import (
            packed_complex_matvec_inplace,
            packed_real_matvec_inplace,
        )
    except ImportError as error:
        raise RuntimeError(
            "optimized CUDA Toeplitz execution requires Triton. Install a "
            "Linux PyTorch distribution that includes Triton, or use the CPU "
            "operator; Pulserver will not silently select a slower CUDA path."
        ) from error
    return packed_real_matvec_inplace, packed_complex_matvec_inplace


def _triton_direct_matvecs() -> tuple[Any, Any]:
    """Import the full-residency kernels with the same hard-fail contract."""
    try:
        from ._triton_toeplitz import (
            packed_complex_matvec_direct,
            packed_real_matvec_direct,
        )
    except ImportError as error:
        raise RuntimeError(
            "optimized CUDA Toeplitz execution requires Triton. Install a "
            "Linux PyTorch distribution that includes Triton, or use the CPU "
            "operator; Pulserver will not silently select a slower CUDA path."
        ) from error
    return packed_real_matvec_direct, packed_complex_matvec_direct


def _cuda_supports_bfloat16(device: Any) -> bool:
    """Return whether ``device`` has native CUDA BF16 support."""
    torch = _torch()
    device = torch.device(device)
    if device.type != "cuda":
        return False
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    if checker is None:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8
    with torch.cuda.device(device):
        try:
            return bool(checker(including_emulation=False))
        except TypeError:
            return bool(checker())


def _resolved_cuda_precision(requested: str, device: Any) -> str:
    """Select BF16 automatically when the CUDA device supports it natively."""
    if requested == "auto":
        return "bfloat16" if _cuda_supports_bfloat16(device) else "float32"
    if requested == "bfloat16" and not _cuda_supports_bfloat16(device):
        raise RuntimeError(f"native CUDA BF16 is unavailable on {device}; use float32")
    return requested


def _cuda_transfer_values(values: Any, precision: str, device: Any) -> Any:
    """Encode real values while retaining complex kernels as complex64."""
    torch = _torch()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]
    target_dtype = torch.complex64 if values.is_complex() else dtype
    return values.to(device=device, dtype=target_dtype).contiguous()


def _cuda_transfer_spec(
    values: Any,
    precision: str,
    locations: int,
) -> tuple[tuple[int, ...], Any]:
    """Return shape and dtype for a packed CUDA transfer buffer."""
    torch = _torch()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]
    if values.is_complex():
        dtype = torch.complex64
    return (values.shape[0], locations), dtype


def _cpu_packed_matvec(packed: Any, spectrum: Any) -> Any | None:
    """Use the precompiled CPU kernel when its zero-copy contract is met."""
    torch = _torch()
    if (
        packed.device.type != "cpu"
        or spectrum.device.type != "cpu"
        or spectrum.dtype != torch.complex64
        or packed.dtype not in {torch.float32, torch.complex64}
        or packed.requires_grad
        or spectrum.requires_grad
    ):
        return None
    matvec = require("recon_cpu", attribute="packed_hermitian_matvec")
    packed = packed.contiguous()
    spectrum = spectrum.contiguous()
    result = torch.empty_like(spectrum)
    matvec(
        packed.numpy(),
        spectrum.numpy(),
        result.numpy(),
    )
    return result


def _packed_cuda_matvec(
    supported: Any,
    packed: Any,
    *,
    location_start: int,
    count: int,
) -> None:
    """Dispatch the required fused CUDA kernel."""
    packed_real_matvec_inplace, packed_complex_matvec_inplace = _triton_matvecs()
    implementation = (
        packed_complex_matvec_inplace
        if packed.is_complex()
        else packed_real_matvec_inplace
    )
    implementation(
        supported,
        packed,
        location_start=location_start,
        count=count,
    )


def _packed_cuda_matvec_direct(
    output: Any,
    spectrum: Any,
    packed: Any,
    indices: Any,
) -> str:
    """Dispatch the Julia-style direct padded-bank multiplication."""
    packed_real_matvec_direct, packed_complex_matvec_direct = _triton_direct_matvecs()
    implementation = (
        packed_complex_matvec_direct
        if packed.is_complex()
        else packed_real_matvec_direct
    )
    return implementation(output, spectrum, packed, indices)


def as_torch(value: Any, *, device: Any | None = None) -> Any:
    """Convert NumPy, CuPy, or Torch data without a needless device copy."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        result = value
    elif hasattr(value, "__dlpack__"):
        result = torch.utils.dlpack.from_dlpack(value)
    else:
        result = torch.as_tensor(value)
    return result.to(device=device) if device is not None else result


def support_indices(
    spatial_shape: tuple[int, ...],
    *,
    support: str = "radial",
    radius: float = 1.0,
    device: Any | None = None,
) -> Any:
    """Return flat unshifted-FFT indices for full or radial support."""
    torch = _torch()
    if support == "full":
        return torch.arange(prod(spatial_shape), dtype=torch.int32, device=device)
    if support != "radial":
        raise ValueError("support must be 'full' or 'radial'")
    if radius <= 0.0 or radius > 1.0:
        raise ValueError("radius must be in the interval (0, 1]")

    axes = [torch.fft.fftfreq(size, device=device) / 0.5 for size in spatial_shape]
    coordinates = torch.meshgrid(*axes, indexing="ij")
    radius_squared = sum(axis.square() for axis in coordinates)
    return (
        torch.nonzero(
            (radius_squared <= radius * radius).flatten(),
            as_tuple=False,
        )
        .flatten()
        .to(torch.int32)
    )


class CompactToeplitzKernel:
    """Packed Hermitian transfer matrices over selected k-space locations.

    ``values`` has shape ``(rank * (rank + 1) // 2, n_locations)`` and stores
    the upper triangle by rows. Real-valued transfers remain real. During
    application, only a bounded location chunk is unpacked for Torch batched
    matrix multiplication; the dense matrix-valued spatial kernel is never
    retained.
    """

    def __init__(
        self,
        values: Any,
        indices: Any,
        spatial_shape: tuple[int, ...],
        rank: int,
        *,
        image_shape: tuple[int, ...] | None = None,
        chunk_size: int = 65536,
        cuda_mode: str = "auto",
        cuda_max_device_fraction: float = 0.85,
        cuda_transfer_precision: str = "auto",
    ) -> None:
        torch = _torch()
        values = as_torch(values)
        indices = as_torch(indices, device=values.device).to(torch.int32)
        packed_rank = rank * (rank + 1) // 2
        if values.ndim != 2 or values.shape[0] != packed_rank:
            raise ValueError(
                f"values must have shape ({packed_rank}, n_locations), "
                f"got {tuple(values.shape)}"
            )
        if indices.ndim != 1 or indices.numel() != values.shape[1]:
            raise ValueError("indices must have one entry per packed location")
        if any(size <= 0 for size in spatial_shape):
            raise ValueError("spatial_shape entries must be positive")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= prod(spatial_shape)
        ):
            raise ValueError("indices contain an out-of-bounds spatial location")
        if image_shape is None:
            image_shape = tuple(size // 2 for size in spatial_shape)
        if len(image_shape) != len(spatial_shape) or any(
            image > transfer
            for image, transfer in zip(image_shape, spatial_shape, strict=True)
        ):
            raise ValueError("image_shape must fit within spatial_shape")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if cuda_mode not in {"auto", "resident", "compact"}:
            raise ValueError("cuda_mode must be 'auto', 'resident', or 'compact'")
        if not 0.0 < cuda_max_device_fraction <= 1.0:
            raise ValueError("cuda_max_device_fraction must be in (0, 1]")
        if cuda_transfer_precision not in {
            "auto",
            "float32",
            "float16",
            "bfloat16",
        }:
            raise ValueError(
                "cuda_transfer_precision must be 'auto', 'float32', "
                "'float16', or 'bfloat16'"
            )
        if values.is_complex() and cuda_transfer_precision not in {
            "auto",
            "float32",
        }:
            raise ValueError("complex Toeplitz kernels require complex64 CUDA storage")
        rows, columns = torch.triu_indices(rank, rank, device=values.device)
        self.values = values.contiguous()
        self.indices = indices.contiguous()
        self.spatial_shape = tuple(int(size) for size in spatial_shape)
        self.image_shape = tuple(int(size) for size in image_shape)
        self.rank = int(rank)
        self.chunk_size = int(chunk_size)
        self.cuda_mode = cuda_mode
        self.cuda_max_device_fraction = float(cuda_max_device_fraction)
        self.cuda_transfer_precision = cuda_transfer_precision
        self._rows = rows
        self._columns = columns
        self._packed_coordinates = tuple(
            zip(rows.tolist(), columns.tolist(), strict=True)
        )
        self._off_diagonal = rows != columns
        self._stream_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._resident_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._cuda_value_cache: dict[tuple[Any, ...], Any] = {}
        self._last_cuda_mode: str | None = None
        self._last_cuda_algorithm: str | None = None

    @property
    def is_real(self) -> bool:
        """Whether the packed transfer uses real rather than complex values."""
        return not self.values.is_complex()

    @property
    def n_locations(self) -> int:
        """Number of retained spatial-frequency locations."""
        return int(self.indices.numel())

    @property
    def storage_nbytes(self) -> int:
        """Persistent bytes used by packed values and location indices."""
        return (
            self.values.numel() * self.values.element_size()
            + self.indices.numel() * self.indices.element_size()
        )

    @property
    def dense_nbytes(self) -> int:
        """Bytes required by an equivalent dense complex transfer kernel."""
        complex_size = 8 if self.values.element_size() <= 4 else 16
        return prod(self.spatial_shape) * self.rank**2 * complex_size

    @property
    def compression_ratio(self) -> float:
        """Dense-complex bytes divided by compact persistent bytes."""
        if self.storage_nbytes == 0:
            return float("inf")
        return self.dense_nbytes / self.storage_nbytes

    @property
    def last_cuda_mode(self) -> str | None:
        """CUDA implementation selected by the most recent ``apply`` call."""
        return self._last_cuda_mode

    @property
    def last_cuda_algorithm(self) -> str | None:
        """Autotuned direct-matvec organization used by resident CUDA."""
        return self._last_cuda_algorithm

    def to(self, device: Any) -> CompactToeplitzKernel:
        """Move packed storage to ``device`` and return ``self``."""
        torch = _torch()
        device = torch.device(device)
        if self.values.device != device:
            self.values = self.values.to(device)
            self._cuda_value_cache.clear()
            self._stream_workspaces.clear()
            self._resident_workspaces.clear()
        self._metadata_to(device)
        return self

    def _metadata_to(self, device: Any) -> None:
        """Move the small index metadata without moving canonical FP32 values."""
        torch = _torch()
        device = torch.device(device)
        if self.indices.device != device:
            self.indices = self.indices.to(device)
            self._rows = self._rows.to(device)
            self._columns = self._columns.to(device)
            self._off_diagonal = self._off_diagonal.to(device)

    def _cuda_values_for(
        self,
        device: Any,
        requested: str,
    ) -> tuple[str, Any]:
        """Return the cached CUDA representation for the precision policy."""
        torch = _torch()
        device = torch.device(device)
        precision = self._cuda_precision(requested, device)
        return precision, self._encoded_values_for(device, precision)

    def _cuda_precision(
        self,
        requested: str,
        device: Any,
    ) -> str:
        """Use automatic BF16 only for real packed coefficient kernels."""
        torch = _torch()
        device = torch.device(device)
        if self.values.is_complex():
            if requested not in {"auto", "float32"}:
                raise ValueError(
                    "complex Toeplitz kernels require complex64 CUDA storage"
                )
            return "float32"
        return _resolved_cuda_precision(requested, device)

    def _encoded_values_for(self, device: Any, precision: str) -> Any:
        """Cache one encoded representation on a host or CUDA device."""
        torch = _torch()
        device = torch.device(device)
        key = (
            device,
            precision,
            self.values.device,
            self.values.data_ptr(),
            self.values._version,
        )
        encoded = self._cuda_value_cache.get(key)
        if encoded is None:
            encoded = _cuda_transfer_values(self.values, precision, device)
            self._cuda_value_cache[key] = encoded
        return encoded

    def _cuda_storage_nbytes(self, precision: str) -> int:
        """Bytes occupied by the encoded CUDA transfer."""
        torch = _torch()
        itemsize = {
            "float32": torch.float32.itemsize,
            "float16": torch.float16.itemsize,
            "bfloat16": torch.bfloat16.itemsize,
        }[precision]
        components = 2 if self.values.is_complex() else 1
        return self.values.numel() * itemsize * components

    def _unpack(self, packed: Any, dtype: Any) -> Any:
        """Unpack one location chunk to Hermitian matrices."""
        torch = _torch()
        count = packed.shape[1]
        matrix = torch.zeros(
            (count, self.rank, self.rank),
            dtype=dtype,
            device=packed.device,
        )
        matrix[:, self._rows, self._columns] = packed.T.to(dtype)
        off = self._off_diagonal
        matrix[:, self._columns[off], self._rows[off]] = packed[off].T.conj().to(dtype)
        return matrix

    def _matmul(self, packed: Any, spectrum: Any) -> Any:
        """Multiply packed matrices by ``(batch, rank, locations)`` data."""
        from ._packed import packed_hermitian_matvec

        return packed_hermitian_matvec(
            packed,
            spectrum,
            coordinates=self._packed_coordinates,
        )

    def apply(self, image: Any) -> Any:
        """Apply the zero-padded matrix-valued convolution to coefficient images."""
        torch = _torch()
        if not isinstance(image, torch.Tensor) or not image.is_complex():
            raise TypeError("image must be a complex torch.Tensor")
        expected_ndim = len(self.image_shape) + 2
        if image.ndim != expected_ndim:
            raise ValueError(
                f"image must have shape (batch, rank, *image_shape), "
                f"got {tuple(image.shape)}"
            )
        if image.shape[1] != self.rank or tuple(image.shape[2:]) != self.image_shape:
            raise ValueError(
                f"expected image shape (batch, {self.rank}, {self.image_shape}), "
                f"got {tuple(image.shape)}"
            )
        if image.device.type == "cuda":
            _triton_matvecs()
        if image.device.type == "cuda":
            self._metadata_to(image.device)
        else:
            self.to(image.device)
        if image.device.type == "cuda" and self.values.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.complex64,
        }:
            if image.dtype != torch.complex64:
                raise TypeError(
                    "optimized CUDA Toeplitz execution requires complex64 images"
                )
            if self._select_cuda_mode(image) == "resident":
                self._last_cuda_mode = "resident"
                return self._apply_cuda_resident(image)
            self._last_cuda_mode = "compact"
            return self._apply_cuda_packed(image)

        padded = torch.zeros(
            (image.shape[0], self.rank, *self.spatial_shape),
            dtype=image.dtype,
            device=image.device,
        )
        image_slices = (
            slice(None),
            slice(None),
            *(slice(0, size) for size in self.image_shape),
        )
        padded[image_slices] = image
        axes = tuple(range(2, padded.ndim))
        spectrum = torch.fft.fftn(padded, dim=axes)
        spectrum_flat = spectrum.flatten(start_dim=2)
        output_flat = torch.zeros_like(spectrum_flat)

        for start in range(0, self.n_locations, self.chunk_size):
            stop = min(start + self.chunk_size, self.n_locations)
            index = self.indices[start:stop]
            selected = torch.index_select(spectrum_flat, 2, index)
            transformed = self._matmul(self.values[:, start:stop], selected)
            output_flat.index_copy_(2, index.to(torch.int64), transformed)

        output = torch.fft.ifftn(output_flat.reshape_as(spectrum), dim=axes)
        return output[image_slices]

    def _resident_workspace_key(self, image: Any) -> tuple[Any, ...]:
        return (
            image.device,
            image.dtype,
            image.shape[0],
            self._cuda_precision(
                self.cuda_transfer_precision,
                image.device,
            ),
        )

    def _resident_additional_bytes(self, image: Any) -> int:
        """Conservative peak bytes for two banks plus a cuFFT work bank."""
        bank_bytes = (
            image.shape[0] * self.rank * prod(self.spatial_shape) * image.element_size()
        )
        result_bytes = image.numel() * image.element_size()
        transfer_precision = self._cuda_precision(
            self.cuda_transfer_precision,
            image.device,
        )
        transfer_bytes = self._cuda_storage_nbytes(transfer_precision)
        already_encoded = (
            transfer_precision == "float32" and self.values.device == image.device
        )
        transfer_conversion = 0 if already_encoded else transfer_bytes
        transfer_residency = (
            0
            if self.indices.device == image.device
            else self.indices.numel() * self.indices.element_size()
        )
        # cuFFT work areas depend on shape and CUDA version. Reserving one
        # additional bank keeps automatic selection conservative and prevents
        # a fast-path choice from turning into an allocator OOM.
        return 3 * bank_bytes + result_bytes + transfer_conversion + transfer_residency

    def _resident_fits(self, image: Any) -> tuple[bool, int, int]:
        """Return whether the Julia-style banks fit the configured VRAM budget."""
        torch = _torch()
        required = self._resident_additional_bytes(image)
        free, total = torch.cuda.mem_get_info(image.device)
        allocated = torch.cuda.memory_allocated(image.device)
        reserved = torch.cuda.memory_reserved(image.device)
        reclaimable = max(0, reserved - allocated)
        physical_headroom = free + reclaimable
        fraction_headroom = max(
            0,
            int(total * self.cuda_max_device_fraction) - allocated,
        )
        available = min(physical_headroom, fraction_headroom)
        return required <= available, required, available

    def _select_cuda_mode(self, image: Any) -> str:
        """Select resident banks when they already exist or safely fit."""
        if self.cuda_mode == "compact":
            return "compact"
        if self._resident_workspace_key(image) in self._resident_workspaces:
            return "resident"
        fits, required, available = self._resident_fits(image)
        if fits:
            return "resident"
        if self.cuda_mode == "resident":
            raise MemoryError(
                "resident Toeplitz banks require approximately "
                f"{required / 2**30:.2f} GiB of additional CUDA memory, but "
                f"only {available / 2**30:.2f} GiB is available within the "
                "configured device fraction"
            )
        return "compact"

    def _apply_cuda_resident(
        self,
        image: Any,
        *,
        right_factor: Any | None = None,
        left_factor: Any | None = None,
        output: Any | None = None,
    ) -> Any:
        """Apply persistent batched banks, optionally fusing spatial factors."""
        torch = _torch()
        self._metadata_to(image.device)
        key = self._resident_workspace_key(image)
        workspace = self._resident_workspaces.get(key)
        if workspace is None:
            _, transfer_values = self._cuda_values_for(
                image.device,
                self.cuda_transfer_precision,
            )
            workspace = {
                "input": torch.empty(
                    (image.shape[0], self.rank, *self.spatial_shape),
                    dtype=image.dtype,
                    device=image.device,
                ),
                "output": torch.empty(
                    (image.shape[0], self.rank, *self.spatial_shape),
                    dtype=image.dtype,
                    device=image.device,
                ),
                "indices": self.indices.to(image.device, dtype=torch.int32),
                "values": transfer_values,
            }
            self._resident_workspaces[key] = workspace
        input_bank = workspace["input"]
        output_bank = workspace["output"]
        image_slices = (
            slice(None),
            slice(None),
            *(slice(0, size) for size in self.image_shape),
        )
        axes = tuple(range(2, input_bank.ndim))

        input_bank.zero_()
        input_crop = input_bank[image_slices]
        if right_factor is None:
            input_crop.copy_(image)
        else:
            torch.mul(image, right_factor, out=input_crop)
        torch.fft.fftn(input_bank, dim=axes, out=input_bank)
        output_bank.zero_()
        self._last_cuda_algorithm = _packed_cuda_matvec_direct(
            output_bank,
            input_bank,
            workspace["values"],
            workspace["indices"],
        )
        torch.fft.ifftn(output_bank, dim=axes, out=output_bank)
        output_crop = output_bank[image_slices]
        if output is None:
            result = output_crop.clone()
            if left_factor is not None:
                result.mul_(left_factor)
            return result
        if left_factor is None:
            output.add_(output_crop)
        else:
            output.addcmul_(output_crop, left_factor)
        return output

    def _apply_cuda_packed(self, image: Any) -> Any:
        """Apply a resident packed transfer with bounded CUDA buffers."""
        torch = _torch()
        batch = image.shape[0]
        supported = torch.empty(
            (batch, self.rank, self.n_locations),
            dtype=image.dtype,
            device=image.device,
        )
        padded = torch.empty(
            (batch, *self.spatial_shape),
            dtype=image.dtype,
            device=image.device,
        )
        result = torch.empty_like(image)
        image_slices = (
            slice(None),
            *(slice(0, size) for size in self.image_shape),
        )
        axes = tuple(range(1, padded.ndim))
        for coefficient in range(self.rank):
            padded.zero_()
            padded[image_slices].copy_(image[:, coefficient])
            torch.fft.fftn(padded, dim=axes, out=padded)
            torch.index_select(
                padded.flatten(start_dim=1),
                1,
                self.indices,
                out=supported[:, coefficient],
            )

        _, transfer_values = self._cuda_values_for(
            image.device,
            self.cuda_transfer_precision,
        )
        _packed_cuda_matvec(
            supported,
            transfer_values,
            location_start=0,
            count=self.n_locations,
        )

        spectrum_flat = padded.flatten(start_dim=1)
        for coefficient in range(self.rank):
            spectrum_flat.zero_()
            spectrum_flat.index_copy_(
                1,
                # Packed indices are int32 to halve resident storage;
                # index_copy_ is the one consumer that insists on int64.
                self.indices.to(torch.int64),
                supported[:, coefficient],
            )
            torch.fft.ifftn(padded, dim=axes, out=padded)
            result[:, coefficient].copy_(padded[image_slices])
        return result

    def _streamed_multiply(
        self,
        spectrum: Any,
        policy: Any,
    ) -> Any:
        """Multiply host spectra by sample-parallel packed CUDA kernels."""
        torch = _torch()
        batch = spectrum.shape[0]
        chunk_size = min(policy.transfer_chunk_size, self.n_locations)
        precision = self._cuda_precision(
            policy.transfer_precision,
            policy.torch_device,
        )
        encoded_host = self._encoded_values_for("cpu", precision)
        packed_shape, packed_dtype = _cuda_transfer_spec(
            self.values,
            precision,
            chunk_size,
        )
        output = torch.empty_like(
            spectrum,
            device="cpu",
        )
        streams = [
            torch.cuda.Stream(device=policy.torch_device) for _ in range(policy.streams)
        ]
        host_inputs = [
            torch.empty(
                (batch, self.rank, chunk_size),
                dtype=spectrum.dtype,
                pin_memory=policy.pin_memory,
            )
            for _ in streams
        ]
        host_values = [
            torch.empty(
                packed_shape,
                dtype=packed_dtype,
                pin_memory=policy.pin_memory,
            )
            for _ in streams
        ]
        host_outputs = [torch.empty_like(item) for item in host_inputs]
        pending: list[tuple[int, int] | None] = [None] * len(streams)

        def finish(slot: int) -> None:
            location = pending[slot]
            if location is None:
                return
            streams[slot].synchronize()
            start, stop = location
            output[:, :, start:stop].copy_(host_outputs[slot][:, :, : stop - start])
            pending[slot] = None

        for chunk_index, start in enumerate(range(0, self.n_locations, chunk_size)):
            stop = min(start + chunk_size, self.n_locations)
            count = stop - start
            slot = chunk_index % policy.streams
            finish(slot)
            host_inputs[slot][:, :, :count].copy_(spectrum[:, :, start:stop])
            host_values[slot][:, :count].copy_(encoded_host[:, start:stop])
            stream = streams[slot]
            with torch.cuda.stream(stream):
                selected = host_inputs[slot][:, :, :count].to(
                    policy.torch_device,
                    non_blocking=policy.pin_memory,
                )
                packed = host_values[slot][:, :count].to(
                    policy.torch_device,
                    non_blocking=policy.pin_memory,
                )
                _packed_cuda_matvec(
                    selected,
                    packed,
                    location_start=0,
                    count=count,
                )
                host_outputs[slot][:, :, :count].copy_(
                    selected,
                    non_blocking=policy.pin_memory,
                )
            pending[slot] = (start, stop)
        for slot in range(policy.streams):
            finish(slot)
        return output

    def _packed_index(self, row: int, column: int) -> tuple[int, bool]:
        """Return packed upper-triangle index and whether conjugation is needed."""
        conjugate = row > column
        upper_row, upper_column = (column, row) if conjugate else (row, column)
        start = upper_row * self.rank - (upper_row * (upper_row - 1)) // 2
        return start + upper_column - upper_row, conjugate

    def _apply_device_spectra(
        self,
        image: Any,
        policy: Any,
        *,
        right_factor: Any | None = None,
        left_factor: Any | None = None,
    ) -> Any:
        """Keep supported spectra on CUDA while streaming packed matrix rows."""
        torch = _torch()
        if self.values.device.type != "cpu":
            self.to("cpu")
        chunk_size = min(policy.transfer_chunk_size, self.n_locations)
        fused_packed = self.values.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.complex64,
        }
        transfer_precision = self._cuda_precision(
            policy.transfer_precision,
            policy.torch_device,
        )
        encoded_host = self._encoded_values_for("cpu", transfer_precision)
        packed_shape, transfer_dtype = _cuda_transfer_spec(
            self.values,
            transfer_precision,
            chunk_size,
        )

        # Select a resident representation before allocating the workspace.
        # Automatic BF16 is encoded once and cached, rather than recast inside
        # every normal-operator application.
        _, total_device_bytes = torch.cuda.mem_get_info(policy.torch_device)
        complex_size = image.element_size()
        transfer_chunk_bytes = (
            self._cuda_storage_nbytes(transfer_precision)
            * chunk_size
            // max(1, self.n_locations)
        )
        base_workspace_bytes = (
            image.shape[0] * self.rank * self.n_locations * complex_size
            # One persistent padded volume plus one cuFFT work/output volume.
            # The latter is not visible in the Python tensor graph but is a
            # real peak allocation even for an aliased ``out`` transform.
            + 2 * image.shape[0] * prod(self.spatial_shape) * complex_size
            + self.indices.numel() * torch.int32.itemsize
            + policy.streams * transfer_chunk_bytes
        )
        if not fused_packed:
            base_workspace_bytes += (
                image.shape[0]
                * (prod(self.spatial_shape) + self.n_locations)
                * complex_size
            )
        if right_factor is not None or left_factor is not None:
            base_workspace_bytes += (
                image.shape[0] * prod(self.image_shape) * complex_size
            )
        resident_precision = None
        if policy.kernel_residency != "host":
            capacity = int(total_device_bytes * policy.max_device_fraction)
            kernel_bytes = self._cuda_storage_nbytes(transfer_precision)
            if base_workspace_bytes + kernel_bytes <= capacity:
                resident_precision = transfer_precision
            elif policy.kernel_residency == "device":
                minimum = base_workspace_bytes + kernel_bytes
                raise MemoryError(
                    "resident Toeplitz transfer requires approximately "
                    f"{minimum / 2**30:.2f} GiB, above the configured "
                    "CUDA memory fraction"
                )
        workspace_key = (
            policy.torch_device,
            image.dtype,
            image.shape[0],
            chunk_size,
            policy.streams,
            policy.pin_memory,
            transfer_precision,
            resident_precision,
        )
        workspace = self._stream_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "indices": self.indices.to(
                    policy.torch_device,
                    dtype=torch.int32,
                ),
                "supported": torch.empty(
                    (image.shape[0], self.rank, self.n_locations),
                    dtype=image.dtype,
                    device=policy.torch_device,
                ),
                "padded": torch.empty(
                    (image.shape[0], *self.spatial_shape),
                    dtype=image.dtype,
                    device=policy.torch_device,
                ),
                # cuFFT accepts an aliased ``out`` tensor.  A real packed
                # transfer therefore needs only one padded FFT workspace;
                # the complex-Hermitian path retains a distinct spectrum
                # because it cannot transform its inputs in place.
                "spectrum": (
                    None
                    if fused_packed
                    else torch.empty(
                        (image.shape[0], *self.spatial_shape),
                        dtype=image.dtype,
                        device=policy.torch_device,
                    )
                ),
                "output_supported": (
                    None
                    if fused_packed
                    else torch.empty(
                        (image.shape[0], self.n_locations),
                        dtype=image.dtype,
                        device=policy.torch_device,
                    )
                ),
                "streams": [
                    torch.cuda.Stream(device=policy.torch_device)
                    for _ in range(policy.streams)
                ],
                "host_rows": [
                    torch.empty(
                        (self.rank, chunk_size),
                        dtype=self.values.dtype,
                        pin_memory=policy.pin_memory,
                    )
                    for _ in range(policy.streams)
                ],
                "host_packed": [
                    torch.empty(
                        packed_shape,
                        dtype=transfer_dtype,
                        pin_memory=policy.pin_memory,
                    )
                    for _ in range(policy.streams)
                ],
                "resident_packed": (
                    None
                    if resident_precision is None
                    else self._encoded_values_for(
                        policy.torch_device,
                        resident_precision,
                    )
                ),
                "device_packed": (
                    None
                    if resident_precision is not None
                    else [
                        torch.empty(
                            packed_shape,
                            dtype=transfer_dtype,
                            device=policy.torch_device,
                        )
                        for _ in range(policy.streams)
                    ]
                ),
                "host_factor": (
                    None
                    if right_factor is None and left_factor is None
                    else torch.empty(
                        (image.shape[0], *self.image_shape),
                        dtype=image.dtype,
                        pin_memory=policy.pin_memory,
                    )
                ),
                "device_factor": (
                    None
                    if right_factor is None and left_factor is None
                    else torch.empty(
                        (image.shape[0], *self.image_shape),
                        dtype=image.dtype,
                        device=policy.torch_device,
                    )
                ),
            }
            self._stream_workspaces[workspace_key] = workspace
        elif (right_factor is not None or left_factor is not None) and workspace[
            "host_factor"
        ] is None:
            workspace["host_factor"] = torch.empty(
                (image.shape[0], *self.image_shape),
                dtype=image.dtype,
                pin_memory=policy.pin_memory,
            )
            workspace["device_factor"] = torch.empty(
                (image.shape[0], *self.image_shape),
                dtype=image.dtype,
                device=policy.torch_device,
            )
        indices_device = workspace["indices"]
        supported = workspace["supported"]
        padded = workspace["padded"]
        spectrum = workspace["spectrum"]
        fft_output = padded if fused_packed else spectrum
        assert fft_output is not None
        image_slices = (
            slice(None),
            *(slice(0, size) for size in self.image_shape),
        )
        axes = tuple(range(1, len(self.spatial_shape) + 1))

        def stage_factor(factor: Any, coefficient: int) -> Any:
            host_factor = workspace["host_factor"]
            device_factor = workspace["device_factor"]
            assert host_factor is not None and device_factor is not None
            if factor.ndim == len(self.image_shape):
                selected = factor[None].expand(image.shape[0], *self.image_shape)
            elif factor.ndim == len(self.image_shape) + 1:
                selected = factor[coefficient][None].expand(
                    image.shape[0],
                    *self.image_shape,
                )
            else:
                selected = factor[:, coefficient]
            host_factor.copy_(selected)
            device_factor.copy_(
                host_factor,
                non_blocking=policy.pin_memory,
            )
            return device_factor

        for coefficient in range(self.rank):
            padded.zero_()
            padded[image_slices].copy_(
                image[:, coefficient],
                non_blocking=policy.pin_memory and image.is_pinned(),
            )
            if right_factor is not None:
                padded[image_slices].mul_(stage_factor(right_factor, coefficient))
            torch.fft.fftn(padded, dim=axes, out=fft_output)
            torch.index_select(
                fft_output.flatten(start_dim=1),
                1,
                indices_device,
                out=supported[:, coefficient],
            )

        result = torch.empty(
            (image.shape[0], self.rank, *self.image_shape),
            dtype=image.dtype,
            device="cpu",
            pin_memory=policy.pin_memory,
        )
        streams = workspace["streams"]
        host_rows = workspace["host_rows"]
        pending = [False] * policy.streams
        if fused_packed:
            resident_packed = workspace["resident_packed"]
            if resident_packed is not None:
                _packed_cuda_matvec(
                    supported,
                    resident_packed,
                    location_start=0,
                    count=self.n_locations,
                )
            else:
                host_packed = workspace["host_packed"]
                device_packed = workspace["device_packed"]
                assert device_packed is not None
                for chunk_index, start in enumerate(
                    range(0, self.n_locations, chunk_size)
                ):
                    stop = min(start + chunk_size, self.n_locations)
                    count = stop - start
                    slot = chunk_index % policy.streams
                    if pending[slot]:
                        streams[slot].synchronize()
                    host_packed[slot][:, :count].copy_(encoded_host[:, start:stop])
                    with torch.cuda.stream(streams[slot]):
                        packed = device_packed[slot][:, :count]
                        packed.copy_(
                            host_packed[slot][:, :count],
                            non_blocking=policy.pin_memory,
                        )
                        _packed_cuda_matvec(
                            supported,
                            packed,
                            location_start=start,
                            count=count,
                        )
                    pending[slot] = True
                for slot in range(policy.streams):
                    if pending[slot]:
                        streams[slot].synchronize()
                        pending[slot] = False
        else:
            assert spectrum is not None
            for output_coefficient in range(self.rank):
                output_supported = workspace["output_supported"]
                assert output_supported is not None
                for chunk_index, start in enumerate(
                    range(0, self.n_locations, chunk_size)
                ):
                    stop = min(start + chunk_size, self.n_locations)
                    count = stop - start
                    slot = chunk_index % policy.streams
                    if pending[slot]:
                        streams[slot].synchronize()
                    for input_coefficient in range(self.rank):
                        packed_index, conjugate = self._packed_index(
                            output_coefficient,
                            input_coefficient,
                        )
                        values = self.values[packed_index, start:stop]
                        host_rows[slot][input_coefficient, :count].copy_(
                            values.conj() if conjugate else values
                        )
                    with torch.cuda.stream(streams[slot]):
                        row = host_rows[slot][:, :count].to(
                            policy.torch_device,
                            non_blocking=policy.pin_memory,
                        )
                        output_supported[:, start:stop] = (
                            supported[:, :, start:stop] * row[None].to(image.dtype)
                        ).sum(dim=1)
                    pending[slot] = True
                for slot in range(policy.streams):
                    if pending[slot]:
                        streams[slot].synchronize()
                        pending[slot] = False
                spectrum_flat = spectrum.flatten(start_dim=1)
                spectrum_flat.zero_()
                for start in range(0, self.n_locations, chunk_size):
                    stop = min(start + chunk_size, self.n_locations)
                    spectrum_flat.index_copy_(
                        1,
                        indices_device[start:stop].to(torch.int64),
                        output_supported[:, start:stop],
                    )
                torch.fft.ifftn(
                    spectrum,
                    dim=axes,
                    out=padded,
                )
                if left_factor is not None:
                    padded[image_slices].mul_(
                        stage_factor(left_factor, output_coefficient)
                    )
                result[:, output_coefficient].copy_(
                    padded[image_slices],
                    non_blocking=policy.pin_memory,
                )
            torch.cuda.current_stream(policy.torch_device).synchronize()
            return result

        for output_coefficient in range(self.rank):
            spectrum_flat = padded.flatten(start_dim=1)
            spectrum_flat.zero_()
            for start in range(0, self.n_locations, chunk_size):
                stop = min(start + chunk_size, self.n_locations)
                spectrum_flat.index_copy_(
                    1,
                    indices_device[start:stop].to(torch.int64),
                    supported[:, output_coefficient, start:stop],
                )
            torch.fft.ifftn(
                padded,
                dim=axes,
                out=padded,
            )
            if left_factor is not None:
                padded[image_slices].mul_(stage_factor(left_factor, output_coefficient))
            result[:, output_coefficient].copy_(
                padded[image_slices],
                non_blocking=policy.pin_memory,
            )
        torch.cuda.current_stream(policy.torch_device).synchronize()
        return result

    def apply_streamed(
        self,
        image: Any,
        policy: Any,
        *,
        right_factor: Any | None = None,
        left_factor: Any | None = None,
    ) -> Any:
        """Apply the kernel with CPU-resident state and bounded CUDA buffers.

        Only one padded coefficient volume is transformed at a time. Supported
        spectra are retained on the host; packed matrix multiplication is
        double-buffered across the policy's two CUDA streams.
        """
        torch = _torch()
        policy.ensure_available()
        if self.values.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.complex64,
        }:
            _triton_matvecs()
        if (
            not isinstance(image, torch.Tensor)
            or image.device.type != "cpu"
            or not image.is_complex()
        ):
            raise TypeError(
                "streamed Toeplitz input must be a complex CPU torch.Tensor"
            )
        expected = (image.shape[0], self.rank, *self.image_shape)
        if tuple(image.shape) != expected:
            raise ValueError(
                f"expected image shape {expected}, got {tuple(image.shape)}"
            )
        factor_shapes = {
            self.image_shape,
            (self.rank, *self.image_shape),
            (image.shape[0], self.rank, *self.image_shape),
        }
        for name, factor in (
            ("right_factor", right_factor),
            ("left_factor", left_factor),
        ):
            if factor is not None and (
                not isinstance(factor, torch.Tensor)
                or factor.device.type != "cpu"
                or tuple(factor.shape) not in factor_shapes
            ):
                raise ValueError(
                    f"{name} must be a CPU tensor with spatial, rank-spatial, "
                    "or batch-rank-spatial shape"
                )
        if policy.device_count > 1 and image.shape[0] > 1:
            # Each leading batch item is an independent matrix-valued
            # convolution. Keep the packed kernel on the host and let one
            # single-device policy per worker own its streams/workspaces.
            self.to("cpu")
            device_count = min(policy.device_count, image.shape[0])
            boundaries = [
                (index * image.shape[0]) // device_count
                for index in range(device_count + 1)
            ]

            def factor_slice(factor: Any | None, start: int, stop: int) -> Any | None:
                if factor is None:
                    return None
                if factor.ndim == len(self.image_shape) + 2:
                    return factor[start:stop]
                return factor

            def apply_part(index: int) -> Any:
                start, stop = boundaries[index], boundaries[index + 1]
                return self.apply_streamed(
                    image[start:stop],
                    policy.for_device(policy.torch_devices[index]),
                    right_factor=factor_slice(right_factor, start, stop),
                    left_factor=factor_slice(left_factor, start, stop),
                )

            with ThreadPoolExecutor(max_workers=device_count) as executor:
                parts = list(executor.map(apply_part, range(device_count)))
            return torch.cat(parts, dim=0)
        if image.shape[0] > policy.physics_batch_size:
            chunks = [
                self.apply_streamed(
                    image[start : start + policy.physics_batch_size],
                    policy,
                    right_factor=(
                        None
                        if right_factor is None
                        else (
                            right_factor[start : start + policy.physics_batch_size]
                            if right_factor.ndim == len(self.image_shape) + 2
                            else right_factor
                        )
                    ),
                    left_factor=(
                        None
                        if left_factor is None
                        else (
                            left_factor[start : start + policy.physics_batch_size]
                            if left_factor.ndim == len(self.image_shape) + 2
                            else left_factor
                        )
                    ),
                )
                for start in range(0, image.shape[0], policy.physics_batch_size)
            ]
            return torch.cat(chunks, dim=0)
        chunk_size = min(policy.transfer_chunk_size, self.n_locations)
        complex_size = image.element_size()
        transfer_precision = self._cuda_precision(
            policy.transfer_precision,
            policy.torch_device,
        )
        transfer_chunk_bytes = (
            self._cuda_storage_nbytes(transfer_precision)
            * chunk_size
            // max(1, self.n_locations)
        )
        fused_packed = self.values.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.complex64,
        }
        fft_buffers = 1 if fused_packed else 2
        fft_bytes = (
            fft_buffers * image.shape[0] * prod(self.spatial_shape) * complex_size
        )
        free_device_bytes, total_device_bytes = torch.cuda.mem_get_info(
            policy.torch_device
        )
        reclaimable_cache = torch.cuda.memory_reserved(
            policy.torch_device
        ) - torch.cuda.memory_allocated(policy.torch_device)
        available_device_bytes = free_device_bytes + reclaimable_cache
        device_spectrum_bytes = (
            image.shape[0]
            * (self.rank + (0 if fused_packed else 1))
            * self.n_locations
            * complex_size
            + fft_bytes
            + policy.streams * transfer_chunk_bytes
        )
        workspace_prefix = (
            policy.torch_device,
            image.dtype,
            image.shape[0],
            chunk_size,
            policy.streams,
            policy.pin_memory,
        )
        reusable_device_spectra = any(
            key[: len(workspace_prefix)] == workspace_prefix
            for key in self._stream_workspaces
        )
        device_spectra = (
            reusable_device_spectra
            or policy.spectrum_residency == "device"
            or (
                policy.spectrum_residency == "auto"
                and device_spectrum_bytes
                <= min(
                    available_device_bytes,
                    int(total_device_bytes * policy.max_device_fraction),
                )
            )
        )
        if device_spectra:
            return self._apply_device_spectra(
                image,
                policy,
                right_factor=right_factor,
                left_factor=left_factor,
            )
        if right_factor is not None or left_factor is not None:

            def expanded_factor(factor: Any) -> Any:
                if factor.ndim == len(self.image_shape):
                    return factor[None, None].expand_as(image)
                if factor.ndim == len(self.image_shape) + 1:
                    return factor[None].expand_as(image)
                return factor

            factored = (
                image if right_factor is None else image * expanded_factor(right_factor)
            )
            transformed = self.apply_streamed(factored, policy)
            return (
                transformed
                if left_factor is None
                else transformed * expanded_factor(left_factor)
            )

        if self.values.device.type != "cpu":
            self.to("cpu")

        indices_device = self.indices.to(
            policy.torch_device,
            dtype=torch.int32,
        )
        supported = torch.empty(
            (image.shape[0], self.rank, self.n_locations),
            dtype=image.dtype,
            device="cpu",
        )
        image_slices = (
            slice(None),
            *(slice(0, size) for size in self.image_shape),
        )
        axes = tuple(range(1, len(self.spatial_shape) + 1))

        for coefficient in range(self.rank):
            device_image = image[:, coefficient].to(
                policy.torch_device,
                non_blocking=policy.pin_memory and image.is_pinned(),
            )
            padded = torch.zeros(
                (image.shape[0], *self.spatial_shape),
                dtype=image.dtype,
                device=policy.torch_device,
            )
            padded[image_slices] = device_image
            spectrum = torch.fft.fftn(padded, dim=axes)
            selected = torch.index_select(
                spectrum.flatten(start_dim=1),
                1,
                indices_device,
            )
            supported[:, coefficient].copy_(selected.to("cpu"))
            del device_image, padded, spectrum, selected

        transformed = self._streamed_multiply(supported, policy)
        result = torch.empty(
            (image.shape[0], self.rank, *self.image_shape),
            dtype=image.dtype,
            device="cpu",
            pin_memory=policy.pin_memory,
        )
        chunk_size = policy.transfer_chunk_size
        for coefficient in range(self.rank):
            spectrum = torch.zeros(
                (image.shape[0], prod(self.spatial_shape)),
                dtype=image.dtype,
                device=policy.torch_device,
            )
            for start in range(0, self.n_locations, chunk_size):
                stop = min(start + chunk_size, self.n_locations)
                values = transformed[:, coefficient, start:stop].to(
                    policy.torch_device,
                    non_blocking=(policy.pin_memory and transformed.is_pinned()),
                )
                spectrum.index_copy_(
                    1,
                    indices_device[start:stop].to(torch.int64),
                    values,
                )
            spatial = torch.fft.ifftn(
                spectrum.reshape(image.shape[0], *self.spatial_shape),
                dim=axes,
            )
            result[:, coefficient].copy_(spatial[image_slices].to("cpu"))
            del spectrum, spatial
        return result
