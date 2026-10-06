"""FINUFFT's entry points, handed to the isochromat engine."""

from __future__ import annotations

import ctypes
import sys
import warnings


def install(kernels) -> bool:
    """Hand FINUFFT's entry points in both precisions and its options' layout to ``kernels``; False where they are not taken.

    The layout is read from the package the library comes from, so that a
    release that moves a field is read where it is. Without them the engine
    reads every window under a changing gradient sample by sample, and a
    warning says so. They are not taken on macOS, where the finufft wheel
    carries its own copy of LLVM's OpenMP runtime and torch another, and a
    process that initialises a second copy ends.
    """
    if sys.platform == "darwin":
        warnings.warn(
            "FINUFFT is not loaded beside torch on macOS (two OpenMP runtimes): "
            "windows under a changing gradient are read sample by sample",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
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
