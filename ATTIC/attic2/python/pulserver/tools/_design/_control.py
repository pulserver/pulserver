"""
"""

__all__ = [
    # "make_rf_freq_offset_control",
    'make_grad_amplitude_control_1d',
    'make_grad_amplitude_control_2d',
    'make_line_index_control_1d',
    'make_line_index_control_2d',
]

import numpy as np
from numpy.typing import NDArray

# def make_rf_freq_offset_control(n1: int, mb: int = 1) -> NDArray[float]:
#     scale = np.arange(0, n1) / n1
#     return scale[::mb]


def make_grad_amplitude_control_1d(n1: int) -> NDArray[float]:
    return np.arange(-n1 // 2, n1 // 2) / n1


def make_grad_amplitude_control_2d(n1: int, n2: int) -> NDArray[float]:
    ax1 = make_grad_amplitude_control_1d(n1)
    ax2 = make_grad_amplitude_control_1d(n2)
    ax1, ax2 = np.broadcast_arrays(ax1[:, None], ax2)
    return np.stack((ax1.ravel(), ax2.ravel()), axis=-1)


def make_line_index_control_1d(n1: int) -> NDArray[int]:
    return np.arange(n1)


def make_line_index_control_2d(n1: int, n2: int) -> NDArray[int]:
    ax1 = make_line_index_control_1d(n1)
    ax2 = make_line_index_control_1d(n2)
    ax1, ax2 = np.broadcast_arrays(ax1[:, None], ax2)
    return np.stack((ax1.ravel(), ax2.ravel()), axis=-1)
