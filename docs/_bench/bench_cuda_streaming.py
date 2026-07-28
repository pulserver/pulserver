"""Benchmark host-backed compact Toeplitz execution at scanner-scale sizes."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__module__ != "_editable_skbc_pulserver"
]

import torch

import pulserver
from pulserver.recon import CudaStreaming
from pulserver.recon._toeplitz import CompactToeplitzKernel, support_indices

SOURCE = Path(pulserver.__file__).resolve()
if REPO_ROOT not in SOURCE.parents:
    raise RuntimeError(
        f"benchmark imported {SOURCE}, not the requested checkout {REPO_ROOT}"
    )


def worker(
    size: int,
    rank: int,
    streams: int,
    chunk_size: int,
    transfer_precision: str = "auto",
    repeats: int = 3,
) -> dict[str, Any]:
    image_shape = (size,) * 3
    spatial_shape = (2 * size,) * 3
    indices = support_indices(spatial_shape, support="radial")
    packed_rank = rank * (rank + 1) // 2
    values = torch.zeros((packed_rank, indices.numel()), dtype=torch.float32)
    kernel = CompactToeplitzKernel(
        values,
        indices,
        spatial_shape,
        rank,
        image_shape=image_shape,
    )
    image = torch.zeros((1, rank, *image_shape), dtype=torch.complex64)
    policy = CudaStreaming(
        streams=streams,
        transfer_chunk_size=chunk_size,
        transfer_precision=transfer_precision,
    )

    # Warm cuFFT plans, CUDA allocator pools, and pinned staging buffers.
    warm = kernel.apply_streamed(image, policy)
    del warm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = kernel.apply_streamed(image, policy)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    elapsed = min(timings)
    assert result is not None
    workspace = next(iter(kernel._stream_workspaces.values()))
    resident_packed = workspace["resident_packed"]
    return {
        "size": size,
        "rank": rank,
        "streams": streams,
        "transfer_precision": transfer_precision,
        "normal_s": elapsed,
        "timings_s": timings,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "kernel_gib": kernel.storage_nbytes / 2**30,
        "dense_kernel_gib": kernel.dense_nbytes / 2**30,
        "compression": kernel.compression_ratio,
        "input_gib": image.numel() * image.element_size() / 2**30,
        "locations": kernel.n_locations,
        "resident_transfer_dtype": (
            None if resident_packed is None else str(resident_packed.dtype)
        ),
        "gpu": torch.cuda.get_device_name(policy.torch_device),
        "torch": torch.__version__,
        "output_shape": list(result.shape),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    speedup = 1.0 - rows[1]["normal_s"] / rows[0]["normal_s"]
    comparison = (
        f"{speedup:.1%} faster"
        if speedup >= 0
        else f"{-speedup:.1%} slower"
    )
    lines = [
        "| Streams | Normal (s) | Peak VRAM (GiB) | Peak RSS (GiB) | "
        "Kernel (GiB) | Dense kernel (GiB) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['streams']} | {row['normal_s']:.3f} | "
            f"{row['peak_vram_gib']:.2f} | {row['peak_rss_gib']:.2f} | "
            f"{row['kernel_gib']:.2f} | {row['dense_kernel_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"The {rows[0]['size']}³, rank-{rows[0]['rank']} packed radial real "
            f"transfer is {rows[0]['kernel_gib']:.2f} GiB versus "
            f"{rows[0]['dense_kernel_gib']:.2f} GiB for dense complex storage, "
            f"a {rows[0]['compression']:.2f}x reduction. Timings are warmed "
            "compact-normal applications with the transfer resident in host RAM "
            "and support-restricted spectra selected automatically for the GPU. "
            f"The two-stream path is {comparison} than one stream on "
            f"{rows[0]['gpu']} with Torch {rows[0]['torch']} while using the same "
            f"{rows[1]['peak_vram_gib']:.2f} GiB peak allocated VRAM. These "
            "numbers are one coefficient-space transfer application, equivalent "
            "to one coil pass; they are not a complete multi-coil SENSE normal.",
            "",
            "The transfer values are zero-filled to make the scanner-scale "
            "allocation reproducible; FFTs, support gathering/scattering, "
            "host transfers, packed matrix multiplication, and output storage "
            "are unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1048576)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("cuda_streaming_256_results.json"),
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--transfer-precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    args = parser.parse_args()
    if args.worker:
        print(
            json.dumps(
                worker(
                    args.size,
                    args.rank,
                    args.streams,
                    args.chunk_size,
                    args.transfer_precision,
                    args.repeats,
                )
            )
        )
        return 0

    rows = []
    for streams in (1, 2):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--size",
            str(args.size),
            "--rank",
            str(args.rank),
            "--chunk-size",
            str(args.chunk_size),
            "--streams",
            str(streams),
            "--transfer-precision",
            args.transfer_precision,
            "--repeats",
            str(args.repeats),
        ]
        process = subprocess.run(  # noqa: S603 - fixed interpreter/script
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        rows.append(json.loads(process.stdout.strip().splitlines()[-1]))
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    args.out.with_suffix(".md").write_text(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
