"""Execution policies shared by reconstruction operators and algorithms."""

from importlib import import_module
from typing import Any

from ._cuda_streaming import CudaStreaming

__all__ = ["CudaStreaming"]


def _resolve_device(device: Any) -> Any:
    """Settle a ``device=`` argument, ``"auto"`` meaning CUDA when there is one.

    ``None`` leaves the data where it is, which is what a caller that has
    already placed its tensors wants. ``"auto"`` is what a reconstruction
    defaults to: a host with a GPU reconstructs on it, and one without is
    unchanged.
    """
    if device != "auto":
        return device
    try:
        torch = import_module("torch")
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None
