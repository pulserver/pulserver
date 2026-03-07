"""Educational buffered-style reference plugin inspired by Gadgetron course examples.

This plugin is intentionally simple and meant as a readable starting point.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..plugin_api import ReconPlugin


def ifft2_sigpy_style(kspace_ro_e1_e2_cha: np.ndarray) -> np.ndarray:
    """Apply 2D inverse FFT in a SigPy-compatible axis convention.

    Input convention: [RO, E1, E2, CHA]
    Internal convention for operator math: [CHA, E2, E1, RO]
    """
    if kspace_ro_e1_e2_cha.ndim != 4:
        raise ValueError('Expected 4D k-space array [RO, E1, E2, CHA]')

    # Match the educational transpose pattern used in the Gadgetron course material.
    ksp = np.transpose(kspace_ro_e1_e2_cha, (3, 2, 1, 0))

    try:
        import sigpy as sp

        op = sp.linop.FFT(ksp.shape, axes=(-1, -2))
        image = op.H * ksp
    except Exception:
        # Keep this plugin usable even when SigPy is unavailable in lightweight setups.
        image = np.fft.ifft2(ksp, axes=(-1, -2), norm='ortho')

    return np.transpose(image, (3, 2, 1, 0))


class SigPyReferencePlugin(ReconPlugin):
    """Reference plugin showing where SigPy-style reconstruction logic lives."""

    def process(self, connection: Any, config: Any, metadata: Any) -> None:
        # Minimal behavior for v1: drain incoming stream while keeping a clear extension point.
        logging.info('SigPyReferencePlugin started (metadata type: %s)', type(metadata).__name__)
        for _ in connection:
            pass


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Legacy function-style entrypoint for compatibility."""
    SigPyReferencePlugin().process(connection, config, metadata)
