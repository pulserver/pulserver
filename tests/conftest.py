"""Shared pytest fixtures."""

import os

import pytest


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# Without a GPU, Triton's kernels run on the CPU under its interpreter, which
# is chosen when triton is first imported.
if not _cuda_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def device(request):
    """Device name, ``"cpu"`` or ``"cuda"``; the CUDA leg is skipped without a device."""
    return request.param
