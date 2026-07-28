"""Profile direct and compact Toeplitz MRI normal operators.

This benchmark uses the actual mri-nufft FINUFFT/CUFINUFFT backends for
non-Cartesian forward/adjoint operations and the Torch implementation shipped
by pulserver for compact Toeplitz application. Cartesian cases use exact
centered Torch FFTs. Each row runs in a fresh subprocess so peak RSS/VRAM
figures are comparable.

Examples
--------
Run the representative matrix on CPU and CUDA::

    python docs/_bench/bench_recon_physics.py --profile quick

Run one worker directly (normally used by the controller)::

    python docs/_bench/bench_recon_physics.py --worker-json '{...}'
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "refcode/recon/mri-nufft/src"))
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__module__ != "_editable_skbc_pulserver"
]

import numpy as np
import psutil
import torch

from pulserver.recon._toeplitz import CompactToeplitzKernel, as_torch, support_indices
import pulserver.recon.physics as physics


@dataclass(frozen=True)
class Case:
    family: str
    layout: str
    size: int
    slices: int
    decorator: str
    method: str
    device: str
    rank: int = 3
    frames: int = 4
    segments: int = 2
    coils: int = 2

    @property
    def ndim(self) -> int:
        return 3 if self.layout == "3d" else 2

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,) * self.ndim

    @property
    def label(self) -> str:
        return (
            f"{self.family}-{self.layout}-{self.size}-s{self.slices}-"
            f"{self.decorator}-{self.method}-{self.device}"
        )


class MemorySampler:
    def __init__(self, device: str) -> None:
        self.device = device
        self.process = psutil.Process()
        self.rss_peak = self.process.memory_info().rss
        self.vram_peak = self._vram()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _vram(self) -> int:
        if self.device != "cuda":
            return 0
        free, total = torch.cuda.mem_get_info()
        return total - free

    def _sample(self) -> None:
        while not self._stop.wait(0.002):
            self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
            self.vram_peak = max(self.vram_peak, self._vram())

    def __enter__(self) -> MemorySampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self._sample()


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()


def _time_call(function: Any, device: str, repeats: int) -> float:
    function()
    _sync(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        _sync(device)
        timings.append(time.perf_counter() - start)
    return min(timings)


def _complex_random(
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return (real + 1j * imag).astype(np.complex64)


def _coil_maps(
    rng: np.random.Generator,
    coils: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    maps = _complex_random(rng, (coils, *shape))
    scale = np.sqrt(np.square(np.abs(maps)).sum(axis=0, keepdims=True))
    return maps / np.maximum(scale, 1e-6)


def _trajectory(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    frames: int,
) -> np.ndarray:
    samples_per_frame = max(64, int(np.prod(shape) // 8))
    direction = rng.standard_normal((frames, samples_per_frame, len(shape)))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    radius = rng.random((frames, samples_per_frame, 1))
    radius **= 1.0 / len(shape)
    return (0.48 * radius * direction).astype(np.float32)


def _array_module(device: str) -> Any:
    if device == "cpu":
        return np
    import cupy as cp

    return cp


def _to_backend(value: np.ndarray, device: str) -> Any:
    return _array_module(device).asarray(value)


def _native_operator(
    samples: np.ndarray,
    shape: tuple[int, ...],
    maps: np.ndarray,
    slices: int,
    device: str,
) -> Any:
    import mrinufft

    backend = "finufft" if device == "cpu" else "cufinufft"
    return mrinufft.get_operator(backend)(
        samples=_to_backend(samples, device),
        shape=shape,
        smaps=_to_backend(maps, device),
        n_coils=maps.shape[0],
        n_batchs=slices,
        squeeze_dims=False,
    )


def _base_compact(native: Any, options: dict[str, Any]) -> CompactToeplitzKernel:
    transfer = as_torch(native.compute_toeplitz_kernel())
    indices = support_indices(
        tuple(transfer.shape),
        support=options["support"],
        radius=options["radius"],
        device=transfer.device,
    )
    values = torch.index_select(transfer.flatten(), 0, indices)[None]
    return CompactToeplitzKernel(
        values,
        indices,
        tuple(transfer.shape),
        1,
        image_shape=tuple(native.shape),
        chunk_size=options["chunk_size"],
    )


def _orc_factors(
    rng: np.random.Generator,
    samples: int,
    segments: int,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    temporal = _complex_random(rng, (samples, segments)) / np.sqrt(segments)
    spatial = _complex_random(rng, (segments, *shape)) / np.sqrt(segments)
    return temporal, spatial


def _orc_normal(
    native: Any,
    temporal: Any,
    spatial: Any,
    image: Any,
) -> Any:
    xp = _array_module("cuda" if type(image).__module__.startswith("cupy") else "cpu")
    kspace = xp.zeros(
        (image.shape[0], native.n_coils, native.n_samples),
        dtype=xp.complex64,
    )
    for segment in range(temporal.shape[1]):
        encoded = native.op(spatial[segment] * image)
        kspace += temporal[:, segment] * encoded
    result = xp.zeros_like(image)
    for segment in range(temporal.shape[1]):
        weighted = temporal[:, segment].conj() * kspace
        result += spatial[segment].conj() * native.adj_op(weighted)
    return result


def _build_noncartesian(case: Case, rng: np.random.Generator) -> tuple[Any, Any, Any]:
    xp = _array_module(case.device)
    shape = case.shape
    maps_np = _coil_maps(rng, case.coils, shape)
    trajectories = _trajectory(
        rng,
        shape,
        case.frames if "subspace" in case.decorator else 1,
    )
    natives = [
        _native_operator(samples, shape, maps_np, case.slices, case.device)
        for samples in trajectories
    ]
    basis_np = rng.standard_normal(
        (case.rank, case.frames),
        dtype=np.float32,
    )
    options = physics._toeplitz_options(
        support="radial",
        radius=1.0,
        chunk_size=32768,
        coil_batch_size=1,
    )
    input_rank = case.rank if "subspace" in case.decorator else 1
    image_np = _complex_random(rng, (case.slices, input_rank, *shape))

    if case.method == "direct":
        image = xp.asarray(image_np)
        basis = xp.asarray(basis_np)
        if case.decorator == "base":
            normal = lambda: natives[0].adj_op(natives[0].op(image))
        elif case.decorator == "subspace":

            def normal() -> Any:
                expanded = xp.einsum("kt,bk...->bt...", basis.conj(), image)
                frames = []
                for frame, native in enumerate(natives):
                    value = expanded[:, frame : frame + 1]
                    frames.append(native.adj_op(native.op(value)))
                return xp.einsum(
                    "kt,bt...->bk...",
                    basis,
                    xp.concatenate(frames, axis=1),
                )

        elif case.decorator == "orc":
            temporal_np, spatial_np = _orc_factors(
                rng,
                natives[0].n_samples,
                case.segments,
                shape,
            )
            temporal = xp.asarray(temporal_np)
            spatial = xp.asarray(spatial_np)
            normal = lambda: _orc_normal(natives[0], temporal, spatial, image)
        else:
            factors = [
                _orc_factors(rng, native.n_samples, case.segments, shape)
                for native in natives
            ]
            factors_backend = [
                (xp.asarray(temporal), xp.asarray(spatial))
                for temporal, spatial in factors
            ]

            def normal() -> Any:
                expanded = xp.einsum("kt,bk...->bt...", basis.conj(), image)
                frames = []
                for frame, native in enumerate(natives):
                    temporal, spatial = factors_backend[frame]
                    frames.append(
                        _orc_normal(
                            native,
                            temporal,
                            spatial,
                            expanded[:, frame : frame + 1],
                        )
                    )
                return xp.einsum(
                    "kt,bt...->bk...",
                    basis,
                    xp.concatenate(frames, axis=1),
                )

        return normal, image, None

    torch_device = torch.device(case.device)
    image = torch.as_tensor(image_np, device=torch_device)
    frame_physics = [
        SimpleNamespace(native_operator=native) for native in natives
    ]
    if case.decorator == "base":
        kernel = _base_compact(natives[0], options)
        normal = lambda: physics._apply_sense_toeplitz(
            kernel,
            image,
            natives[0],
            coil_batch_size=1,
        )
    elif case.decorator == "subspace":
        kernel = physics._build_subspace_toeplitz(
            frame_physics,
            basis_np,
            options,
        )
        normal = lambda: physics._apply_sense_toeplitz(
            kernel,
            image,
            natives[0],
            coil_batch_size=1,
        )
    elif case.decorator == "orc":
        temporal_np, spatial_np = _orc_factors(
            rng,
            natives[0].n_samples,
            case.segments,
            shape,
        )
        corrected = SimpleNamespace(
            _fourier_op=natives[0],
            B=_to_backend(temporal_np, case.device),
            C=_to_backend(spatial_np, case.device),
            n_shots=1,
        )
        kernel, factors = physics._build_off_resonance_toeplitz(
            corrected,
            options,
        )
        normal = lambda: physics._apply_sense_toeplitz(
            kernel,
            image,
            corrected,
            right_factors=factors,
            left_factors=factors,
            coil_batch_size=1,
        )
    else:
        spatial_np = _orc_factors(
            rng,
            natives[0].n_samples,
            case.segments,
            shape,
        )[1]
        spatial = _to_backend(spatial_np, case.device)
        corrected = []
        for native in natives:
            temporal_np = _orc_factors(
                rng,
                native.n_samples,
                case.segments,
                shape,
            )[0]
            corrected.append(
                SimpleNamespace(
                    _fourier_op=native,
                    B=_to_backend(temporal_np, case.device),
                    C=spatial,
                    n_shots=1,
                    shape=shape,
                )
            )
        corrected_physics = [
            SimpleNamespace(native_operator=native) for native in corrected
        ]
        kernel, factors = physics._build_subspace_off_resonance_toeplitz(
            corrected_physics,
            basis_np,
            options,
        )
        normal = lambda: physics._apply_subspace_off_resonance_toeplitz(
            kernel,
            image,
            corrected[0],
            factors,
            coefficient_rank=case.rank,
            coil_batch_size=1,
        )
    return normal, image, kernel


def _centered_fft(value: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(2, value.ndim))
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(value, dim=axes),
            dim=axes,
            norm="ortho",
        ),
        dim=axes,
    )


def _centered_ifft(value: torch.Tensor) -> torch.Tensor:
    axes = tuple(range(2, value.ndim))
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(value, dim=axes),
            dim=axes,
            norm="ortho",
        ),
        dim=axes,
    )


def _cartesian_frame_normal(
    image: torch.Tensor,
    mask: torch.Tensor,
    maps: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(image)
    for coil_map in maps:
        coil_image = image * coil_map
        result += coil_map.conj() * _centered_ifft(mask * _centered_fft(coil_image))
    return result


def _build_cartesian(case: Case, rng: np.random.Generator) -> tuple[Any, Any, Any]:
    device = torch.device(case.device)
    shape = (case.size, case.size)
    maps = torch.as_tensor(
        _coil_maps(rng, case.coils, shape),
        device=device,
    )
    frame_count = case.frames if case.decorator == "subspace" else 1
    masks = torch.as_tensor(
        rng.random((frame_count, *shape)) > 0.35,
        dtype=torch.float32,
        device=device,
    )
    rank = case.rank if case.decorator == "subspace" else 1
    image = torch.as_tensor(
        _complex_random(rng, (case.slices, rank, *shape)),
        device=device,
    )
    if case.layout == "hybrid3d":
        volume = torch.as_tensor(
            _complex_random(rng, (1, rank, case.slices, *shape)),
            device=device,
        )
        image = torch.fft.fft(volume, dim=2, norm="ortho").movedim(2, 0)[
            :, 0
        ]

    if case.decorator == "base" or case.method == "direct":
        if case.decorator == "base":
            normal = lambda: _cartesian_frame_normal(
                image,
                masks[0],
                maps,
            )
        else:
            basis = torch.as_tensor(
                rng.standard_normal(
                    (case.rank, case.frames),
                    dtype=np.float32,
                ),
                device=device,
            )

            def normal() -> torch.Tensor:
                expanded = torch.einsum(
                    "kt,bk...->bt...",
                    basis.to(image.dtype).conj(),
                    image,
                )
                frames = [
                    _cartesian_frame_normal(
                        expanded[:, frame : frame + 1],
                        masks[frame],
                        maps,
                    )
                    for frame in range(case.frames)
                ]
                return torch.einsum(
                    "kt,bt...->bk...",
                    basis.to(image.dtype),
                    torch.cat(frames, dim=1),
                )

        return normal, image, None

    basis = rng.standard_normal((case.rank, case.frames), dtype=np.float32)
    operator = SimpleNamespace(mask=masks, coil_maps=maps)
    frames = [
        SimpleNamespace(operator=operator) for _ in range(case.frames)
    ]
    options = physics._toeplitz_options(
        support="full",
        chunk_size=32768,
        coil_batch_size=1,
    )
    kernel, proxy = physics._build_cartesian_subspace_toeplitz(
        frames,
        basis,
        options,
    )
    normal = lambda: physics._apply_sense_toeplitz(
        kernel,
        image,
        proxy,
        coil_batch_size=1,
    )
    return normal, image, kernel


def run_worker(case: Case, repeats: int) -> dict[str, Any]:
    if case.device == "cuda" and not torch.cuda.is_available():
        return {**asdict(case), "label": case.label, "error": "CUDA unavailable"}
    rng = np.random.default_rng(20260728)
    process = psutil.Process()
    rss_start = process.memory_info().rss
    vram_start = 0
    if case.device == "cuda":
        _sync("cuda")
        free, total = torch.cuda.mem_get_info()
        vram_start = total - free

    start = time.perf_counter()
    with MemorySampler(case.device) as setup_memory:
        if case.family == "noncartesian":
            normal, image, kernel = _build_noncartesian(case, rng)
        else:
            normal, image, kernel = _build_cartesian(case, rng)
        _sync(case.device)
    setup_seconds = time.perf_counter() - start

    with MemorySampler(case.device) as apply_memory:
        normal_seconds = _time_call(normal, case.device, repeats)

    rss_peak = max(
        setup_memory.rss_peak,
        apply_memory.rss_peak,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    kernel_bytes = kernel.storage_nbytes if kernel is not None else 0
    dense_kernel_bytes = kernel.dense_nbytes if kernel is not None else 0
    result = {
        **asdict(case),
        "label": case.label,
        "shape": list(case.shape),
        "setup_s": setup_seconds,
        "normal_s": normal_seconds,
        "rss_peak_mb": rss_peak / 2**20,
        "rss_delta_mb": (rss_peak - rss_start) / 2**20,
        "vram_peak_mb": max(
            setup_memory.vram_peak,
            apply_memory.vram_peak,
        )
        / 2**20,
        "vram_delta_mb": (
            max(setup_memory.vram_peak, apply_memory.vram_peak) - vram_start
        )
        / 2**20,
        "kernel_mb": kernel_bytes / 2**20,
        "dense_kernel_mb": dense_kernel_bytes / 2**20,
        "kernel_compression": (
            kernel.compression_ratio if kernel is not None else None
        ),
        "input_mb": (
            image.numel() * image.element_size()
            if isinstance(image, torch.Tensor)
            else image.nbytes
        )
        / 2**20,
    }
    return result


def cases(profile: str, devices: list[str]) -> list[Case]:
    if profile == "subspace":
        result = []
        for layout, size, slices, frames in (
            ("2d-single", 64, 1, 16),
            ("2d-single", 128, 1, 16),
            ("2d-single", 256, 1, 16),
            ("2d-multislice", 128, 8, 16),
            ("3d", 24, 1, 12),
            ("3d", 32, 1, 12),
            ("3d", 48, 1, 12),
        ):
            for method in ("direct", "toeplitz"):
                for device in devices:
                    result.append(
                        Case(
                            "noncartesian",
                            layout,
                            size,
                            slices,
                            "subspace",
                            method,
                            device,
                            rank=6,
                            frames=frames,
                        )
                    )
        return result
    sizes = {
        "quick": [
            ("cartesian", "2d-single", 64, 1),
            ("cartesian", "2d-multislice", 64, 8),
            ("cartesian", "hybrid3d", 64, 16),
            ("noncartesian", "2d-single", 64, 1),
            ("noncartesian", "2d-multislice", 64, 8),
            ("noncartesian", "3d", 24, 1),
        ],
        "full": [
            ("cartesian", "2d-single", 128, 1),
            ("cartesian", "2d-multislice", 128, 16),
            ("cartesian", "hybrid3d", 128, 32),
            ("noncartesian", "2d-single", 128, 1),
            ("noncartesian", "2d-multislice", 128, 16),
            ("noncartesian", "3d", 48, 1),
        ],
    }[profile]
    result = []
    for family, layout, size, slices in sizes:
        decorators = (
            ["base", "subspace"]
            if family == "cartesian"
            else ["base", "orc", "subspace", "orc-subspace"]
        )
        for decorator in decorators:
            for method in ("direct", "toeplitz"):
                for device in devices:
                    result.append(
                        Case(
                            family,
                            layout,
                            size,
                            slices,
                            decorator,
                            method,
                            device,
                        )
                    )
    if profile == "quick":
        for size, layout in ((128, "2d-single"), (32, "3d")):
            for method in ("direct", "toeplitz"):
                for device in devices:
                    result.append(
                        Case(
                            "noncartesian",
                            layout,
                            size,
                            1,
                            "subspace",
                            method,
                            device,
                        )
                    )
    return result


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Case | Normal (s) | Setup (s) | Peak RSS (MiB) | Peak VRAM (MiB) | Kernel (MiB) | Dense/compact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['label']} | error: {row['error']} | | | | | |")
            continue
        ratio = row["kernel_compression"]
        lines.append(
            f"| {row['label']} | {row['normal_s']:.5f} | "
            f"{row['setup_s']:.3f} | {row['rss_peak_mb']:.1f} | "
            f"{row['vram_peak_mb']:.1f} | {row['kernel_mb']:.2f} | "
            f"{ratio:.2f}x |" if ratio is not None else
            f"| {row['label']} | {row['normal_s']:.5f} | "
            f"{row['setup_s']:.3f} | {row['rss_peak_mb']:.1f} | "
            f"{row['vram_peak_mb']:.1f} | 0 | — |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("quick", "full", "subspace"),
        default="quick",
    )
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("recon_physics_results.json"),
    )
    parser.add_argument("--worker-json")
    args = parser.parse_args(argv)

    if args.worker_json:
        case = Case(**json.loads(args.worker_json))
        print(json.dumps(run_worker(case, args.repeats)))
        return 0

    selected = cases(args.profile, args.devices.split(","))
    if args.only:
        selected = [case for case in selected if args.only in case.label]
    rows = []
    for index, case in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {case.label}", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-json",
            json.dumps(asdict(case)),
            "--repeats",
            str(args.repeats),
        ]
        environment = os.environ.copy()
        process = subprocess.run(  # noqa: S603 - fixed interpreter/script
            command,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        if process.returncode:
            row = {
                **asdict(case),
                "label": case.label,
                "error": process.stderr.strip().splitlines()[-1],
            }
        else:
            row = json.loads(process.stdout.strip().splitlines()[-1])
        rows.append(row)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
        args.out.with_suffix(".md").write_text(_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
