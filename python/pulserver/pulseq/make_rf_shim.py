"""RF shim extension event constructor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

__all__ = ["make_rf_shim"]


def make_rf_shim(shim_vector) -> SimpleNamespace:
    """Create an RF shim extension event.

    Parameters
    ----------
    shim_vector:
        Complex per-channel shim weights.
    """
    event = SimpleNamespace()
    event.type = "rf_shim"
    event.shim_vector = np.asarray(shim_vector, dtype=np.complex128)
    return event
