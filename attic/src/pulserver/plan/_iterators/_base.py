"""
"""

from types import SimpleNamespace
import numpy as np


def sampling2labels(sampling_pattern: np.ndarray) -> np.ndarray:
    """
    Convert a boolean sampling pattern into an array of indices where the pattern is `True`.

    Given a boolean mask (sampling pattern), this function returns the indices of the `True`
    values in the mask. The indices are returned as an array with the first axis representing
    the dimension of the original array and the subsequent axes representing the coordinates
    of the `True` values in the original array.

    Parameters
    ----------
    sampling_pattern : np.ndarray
        A boolean NumPy array (or array-like) where each `True` value indicates the positions
        to be included in the output indices.

    Returns
    -------
    np.ndarray
        A NumPy array of shape `(sampling_pattern.ndim, N)`, where `N` is the number of `True`
        values in the input `sampling_pattern`. The array contains the indices of the `True` values
        along each axis. Each row in the array corresponds to the indices for one of the axes.

    See Also
    --------
    np.nonzero : A related function that returns the indices of the non-zero elements in an array.
    np.where : Return elements chosen from `x` or `y` depending on a condition.

    Examples
    --------
    >>> sampling_pattern = np.array([[True, False, True],
    ...                              [False, True, False],
    ...                              [True, False, True]])
    >>> sampling2labels(sampling_pattern)
    array([[0, 0, 1, 2, 2],
           [0, 2, 1, 0, 2]])

    In this example, the `True` values in the input array correspond to the following coordinates:
    (0, 0), (0, 2), (1, 1), (2, 0), and (2, 2). These coordinates are returned as rows of the
    output array, where the first row gives the indices along the first axis (rows) and the
    second row gives the indices along the second axis (columns).

    """
    # Get the indices array using the shape of the input array
    indices = np.indices(sampling_pattern.shape)
    return np.stack([idx[sampling_pattern] for idx in indices], axis=0).squeeze()


def _check_iterator_input(args):
    _type = [isinstance(arg, (list, tuple)) for arg in args]
    _type = np.asarray(_type)
    is_homogeneous_arg = np.all(_type == 1) or np.all(_type == 0)
    if not (is_homogeneous_arg):
        raise ValueError(
            "Input values must be either all scaling arrays or length-2 tuples (scaling array, labels array)"
        )
    if np.all(_type == 1):
        _len = [len(arg) for arg in args]
        _len = np.asarray(_len)
    if not np.all(_len == 2):
        raise ValueError(
            "Input values must be either all scaling arrays or length-2 tuples (scaling array, labels array)"
        )
