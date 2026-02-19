"""Partial Fourier undersampling."""

__all__ = ["partial_fourier"]


import warnings


import numpy as np


def partial_fourier(shape: int, undersampling: float) -> np.ndarray:
    """
    Generate sampling pattern for Partial Fourier accelerated acquisition.

    Parameters
    ----------
    shape : int
        Image shape along partial fourier axis.
    undersampling : float
        Target undersampling factor.
        Must be > 0.5 (suggested > 0.7) and <= 1 (=1: no PF).

    Returns
    -------
    np.array
        Regular-grid sampling mask of shape ``(shape,)``.

    """
    # check
    if undersampling > 1 or undersampling <= 0.5:
        raise ValueError(
            "undersampling must be greater than 0.5 and lower than 1, got"
            f" {undersampling}"
        )
    # if undersampling == 1:
    #     warnings.warn("Undersampling factor set to 1 - no acceleration")
    if undersampling < 0.7:
        warnings.warn(
            f"Undersampling factor = {undersampling} < 0.7 - phase errors will"
            " likely occur."
        )

    # generate mask
    mask = np.ones(shape, dtype=bool)

    # cut mask
    edge = np.floor(np.asarray(shape) * np.asarray(undersampling))
    edge = int(edge)
    mask[edge:] = 0

    return mask
