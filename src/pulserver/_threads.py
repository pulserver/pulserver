"""Thread counts of the numerical libraries a reconstruction process loads."""

from __future__ import annotations

__all__ = ["BLAS_THREAD_VARIABLES", "serial_blas"]

import os
from collections.abc import MutableMapping

#: Variables that set the threads of numpy's BLAS when it is first loaded.
BLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def serial_blas(environ: MutableMapping[str, str] | None = None) -> None:
    """Give numpy's BLAS one thread in processes that load it after this call.

    A reconstruction's parallel work is done by torch and bartorch through
    OpenMP; a multi-threaded BLAS answering numpy's small per-readout products
    between them contends for the same cores and is many times slower than one
    thread. A variable already set is kept. ``OMP_NUM_THREADS`` is not
    touched, so torch keeps its threads.

    Parameters
    ----------
    environ : MutableMapping[str, str], default=None
        The environment to set; ``None`` is ``os.environ``.
    """
    environ = os.environ if environ is None else environ
    for name in BLAS_THREAD_VARIABLES:
        environ.setdefault(name, "1")
