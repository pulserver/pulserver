"""Phase-level profiler for the host-backed compact Toeplitz CUDA path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def main() -> int:
    source = Path(pulserver.__file__).resolve()
    if REPO_ROOT not in source.parents:
        raise RuntimeError(
            f"benchmark imported {source}, not the requested checkout {REPO_ROOT}"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1048576)
    args = parser.parse_args()

    spatial_shape = (2 * args.size,) * 3
    image_shape = (args.size,) * 3
    indices = support_indices(spatial_shape, support="radial")
    packed_rank = args.rank * (args.rank + 1) // 2
    kernel = CompactToeplitzKernel(
        torch.zeros((packed_rank, indices.numel()), dtype=torch.float32),
        indices,
        spatial_shape,
        args.rank,
        image_shape=image_shape,
    )
    image = torch.zeros(
        (1, args.rank, *image_shape),
        dtype=torch.complex64,
    )
    policy = CudaStreaming(
        streams=2,
        transfer_chunk_size=args.chunk_size,
    )
    kernel.apply_streamed(image, policy)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as profile:
        kernel.apply_streamed(image, policy)
        torch.cuda.synchronize()
    print(
        profile.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
