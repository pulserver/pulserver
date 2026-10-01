"""FINUFFT's entry points, handed to the isochromat engine."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
import warnings


def install(kernels) -> bool:
    """Hand FINUFFT's entry points in both precisions and its options' layout to ``kernels``; False where they are not taken.

    The layout is read from the package the library comes from, so that a
    release that moves a field is read where it is. Without them the engine
    reads every window under a changing gradient sample by sample, and a
    warning says so.
    """
    _one_openmp_runtime()
    try:
        from finufft import _finufft

        names = [
            f"{prefix}_{entry}"
            for prefix in ("finufft", "finufftf")
            for entry in ("makeplan", "setpts", "execute", "destroy")
        ]
        entries = {
            name: ctypes.cast(getattr(_finufft.lib, name), ctypes.c_void_p).value
            for name in [*names, "finufft_default_opts"]
        }
        options = _finufft.FinufftOpts
        layout = {
            field: getattr(options, field).offset
            for field in ("nthreads", "fftw", "upsampfac", "showwarn")
        }
        layout["size"] = ctypes.sizeof(options)
    except (ImportError, AttributeError) as error:
        warnings.warn(
            f"FINUFFT's entry points were not found ({error}): windows under a "
            "changing gradient are read sample by sample",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return kernels.use_finufft(entries, layout)


def _one_openmp_runtime() -> None:
    """On macOS, point FINUFFT's library at torch's OpenMP runtime before it loads, where bartorch is installed to do it.

    torch and the finufft wheel each carry LLVM's OpenMP runtime, which ends a
    process that loads a second copy of it.
    """
    if sys.platform != "darwin" or importlib.util.find_spec("bartorch") is None:
        return
    from bartorch import _macos_openmp

    _macos_openmp.ensure()
