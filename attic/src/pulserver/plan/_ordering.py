"""Acquisition reordering subroutines."""

__all__ = ["interleaved", "center_out"]


import numpy as np


def interleaved(acq_table: np.ndarray) -> np.ndarray:
    """
    Reorder the acquisition table so that all the even indexes are acquired after the odd.

    Parameters
    ----------
    acq_table : np.ndarray
        Input sequential acquisition table.

    Returns
    -------
    np.ndarray
        Acquisitions index table.

    """
    n_acq = len(acq_table)
    acq_idx = np.arange(n_acq)
    return acq_table[np.concatenate((acq_idx[::2], acq_idx[1::2]))]


def center_out(acq_table: np.ndarray) -> np.ndarray:
    """
    Reorder the acquisition table from center to periphery.

    Parameters
    ----------
    acq_table : np.ndarray
        Input sequential acquisition table.

    Returns
    -------
    np.ndarray
        Acquisitions index table.

    """
    n_acq = len(acq_table)
    acq_idx = np.arange(n_acq)

    # get ordering
    order = np.zeros_like(acq_idx)
    for n in range(int(n_acq // 2)):
        order[2 * n] = n_acq // 2 + n
        order[2 * n + 1] = n_acq // 2 - n - 1
    if n_acq % 2 != 0:
        order[-1] = n_acq - 1

    return acq_table[acq_idx[order.astype(int)]]
