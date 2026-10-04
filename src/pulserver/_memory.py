"""Process-wide memory settings for the pulserver services."""

from __future__ import annotations

import os
import sys


def without_huge_page_advice() -> None:
    """Stop NumPy advising the kernel to back large arrays with huge pages.

    Under ``transparent_hugepage/defrag = madvise`` the first touch of an
    advised array compacts memory, which then dominates building the readout
    tables. Sets ``NUMPY_MADVISE_HUGEPAGE=0`` unless the environment already
    chooses, and applies it to a NumPy already imported.
    """
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    numpy = sys.modules.get("numpy")
    if numpy is None or os.environ["NUMPY_MADVISE_HUGEPAGE"] != "0":
        return
    core = getattr(numpy, "_core", None) or numpy.core
    setter = getattr(core.multiarray, "_set_madvise_hugepage", None)
    if setter is not None:
        setter(False)
