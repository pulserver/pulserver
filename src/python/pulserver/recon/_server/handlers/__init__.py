"""Private built-in Gadgetron-compatible reconstruction handlers."""

from __future__ import annotations

import numpy as np
import numpy.fft as fft

from ....mrd._images import coil_combine
from ....mrd._metadata import max_stored_value


def _fft_combine_scaled(stacked: np.ndarray, header) -> np.ndarray:
    """IFFT stacked Cartesian lines, combine the coils, and scale to the
    stored bit depth."""
    data = fft.fftshift(stacked, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    data = coil_combine(data, coil_axis=0)

    maximum = float(data.max(initial=0.0))
    if maximum > 0.0:
        data *= max_stored_value(header) / maximum
    return np.around(data).astype(np.int16)
