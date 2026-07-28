"""Benchmark a complete host-backed subspace SENSE Toeplitz normal."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
from pulserver.recon.physics import _apply_sense_toeplitz

SOURCE = Path(pulserver.__file__).resolve()
if REPO_ROOT not in SOURCE.parents:
    raise RuntimeError(
        f"benchmark imported {SOURCE}, not the requested checkout {REPO_ROOT}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--coils", type=int, default=8)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("streamed", "full"),
        default="streamed",
    )
    parser.add_argument("--chunk-size", type=int, default=1048576)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--transfer-precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-mode",
        choices=("auto", "resident", "compact"),
        default="auto",
    )
    parser.add_argument(
        "--cuda-transfer-precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    args = parser.parse_args()

    image_shape = (args.size,) * 3
    spatial_shape = (2 * args.size,) * 3
    indices = support_indices(spatial_shape, support="radial")
    packed_rank = args.rank * (args.rank + 1) // 2
    kernel = CompactToeplitzKernel(
        torch.zeros((packed_rank, indices.numel()), dtype=torch.float32),
        indices,
        spatial_shape,
        args.rank,
        image_shape=image_shape,
        cuda_mode=args.cuda_mode,
        cuda_transfer_precision=args.cuda_transfer_precision,
    )
    tensor_device = "cpu" if args.mode == "streamed" else "cuda"
    image = torch.zeros(
        (1, args.rank, *image_shape),
        dtype=torch.complex64,
        device=tensor_device,
        pin_memory=args.mode == "streamed",
    )
    maps = torch.zeros(
        (args.coils, *image_shape),
        dtype=torch.complex64,
        device=tensor_device,
    )
    native = SimpleNamespace(shape=image_shape, smaps=maps)
    policy = CudaStreaming(
        streams=args.streams,
        transfer_chunk_size=args.chunk_size,
        transfer_precision=args.transfer_precision,
    )

    def normal() -> torch.Tensor:
        return _apply_sense_toeplitz(
            kernel,
            image,
            native,
            coil_batch_size=1,
            streaming=policy if args.mode == "streamed" else None,
        )

    for _ in range(args.warmups):
        warm = normal()
        del warm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    result = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = normal()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    elapsed = min(timings)
    assert result is not None
    workspace = next(iter(kernel._stream_workspaces.values()), None)
    resident = None if workspace is None else workspace["resident_packed"]
    print(
        json.dumps(
            {
                "size": args.size,
                "rank": args.rank,
                "coils": args.coils,
                "streams": args.streams,
                "mode": args.mode,
                "cuda_mode": args.cuda_mode,
                "selected_cuda_mode": kernel.last_cuda_mode,
                "selected_cuda_algorithm": kernel.last_cuda_algorithm,
                "transfer_precision": args.transfer_precision,
                "cuda_transfer_precision": args.cuda_transfer_precision,
                "normal_s": elapsed,
                "median_s": statistics.median(timings),
                "timings_s": timings,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_vram_gib": (
                    torch.cuda.max_memory_reserved() / 2**30
                ),
                "peak_rss_gib": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                ),
                "resident_transfer_dtype": (
                    (
                        str(
                            next(iter(kernel._resident_workspaces.values()))[
                                "values"
                            ].dtype
                        )
                        if kernel._resident_workspaces
                        else None
                    )
                    if resident is None
                    else str(resident.dtype)
                ),
                "kernel_gib": kernel.storage_nbytes / 2**30,
                "output_shape": list(result.shape),
                "gpu": torch.cuda.get_device_name(policy.torch_device),
                "torch": torch.__version__,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
