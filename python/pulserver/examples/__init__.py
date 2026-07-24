"""Importable examples shipped with Pulserver."""

from __future__ import annotations

import importlib

__all__ = ["recon"]


def __getattr__(name: str):
    if name == "recon":
        return importlib.import_module(f"{__name__}.recon")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
