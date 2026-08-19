"""Complex-to-real channel views at the DeepInverse boundary."""

from __future__ import annotations

__all__: list[str] = [
    "image_as_cpx",
    "image_as_real",
    "kspace_as_cpx",
    "kspace_as_real",
]

from importlib import import_module
from typing import Any


def image_as_real(value: Any) -> Any:
    """Pack complex image channels as DeepInverse real channels."""
    torch = import_module("torch")
    batch, channels, *spatial = value.shape
    value = torch.view_as_real(value).movedim(-1, 2)
    return value.reshape(batch, channels * 2, *spatial)


def image_as_cpx(value: Any) -> Any:
    """Unpack DeepInverse real image channels to complex channels."""
    torch = import_module("torch")
    batch, channels, *spatial = value.shape
    if channels % 2:
        raise ValueError("real-view images require pairs of complex channels")
    value = value.reshape(batch, channels // 2, 2, *spatial).movedim(2, -1)
    return torch.view_as_complex(value.contiguous())


def kspace_as_real(value: Any) -> Any:
    """Expose complex k-space through a trailing real/imaginary dimension."""
    torch = import_module("torch")
    return torch.view_as_real(value)


def kspace_as_cpx(value: Any) -> Any:
    """Restore complex k-space from its trailing real/imaginary dimension."""
    torch = import_module("torch")
    return torch.view_as_complex(value.contiguous())
