"""
"""

__all__ = ['verse']

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


def verse(grad: NDArray[float], rf: NDArray[complex]) -> NDArray[complex]:
    m = rf.size

    # k = (m-1)*cumsum(g)/max(cumsum(g))
    k = np.cumsum(grad)
    k = (m - 1) * k / k.max()

    # Normalize g: g = m*g/sum(g)
    grad = m * deepcopy(grad) / grad.sum()

    # Interpolate: interp1([0:m-1], rf, k)
    x = np.arange(m)
    interpolated = np.interp(k, x, rf)

    # Multiply by g
    rfv = grad * interpolated

    return rfv
