"""Iterator for 3D Cartesian sampling."""

__all__ = ["Cartesian3DIterator"]

from types import SimpleNamespace

import numpy as np

from ._base import _check_iterator_input


class Cartesian3DIterator:
    """
    Iterator for Cartesian 3D parameter combinations.

    This class generates combinations of two encoding axes, i.e., view encoding, slice encoding.
    It also supports an optional label system for associating each combination with corresponding
    label indices.

    Parameters
    ----------
    view_enc : np.ndarray or tuple
        The view encoding steps or a tuple where the first element is the scaling array and the
        second element is an associated label array. This parameter defines one axis of the
        Cartesian grid.

    slice_enc : np.ndarray or tuple
        The slice encoding steps or a tuple where the first element is the scaling array and the
        second element is an associated label array. This parameter defines the other axis of the
        Cartesian grid.

    dummy_shots : int, optional
        The number of dummy shots to perform before starting the iteration. During these dummy
        shots, the output will be zeros. Default is 0.

    Attributes
    ----------
    _gyamp : np.ndarray
        The 1D array of gy encoding amplitudes.

    _gz_amp : np.ndarray
        The 1D array of gz encoding amplitudes.

    _iy : np.ndarray or None
        The 1D array of labels corresponding to the view encoding axis (if labels are provided).

    _iz : np.ndarray or None
        The 1D array of labels corresponding to the slice encoding axis (if labels are provided).

    _haslabel : bool
        Indicates whether labels are provided for the view and slice encoding axes.

    count : int
        The current iteration count, starting from `-dummy_shots`.

    Methods
    -------
    __call__()
        Returns the next combination of parameters (`gy_amp`, `gz_amp`) and optional labels (`iy`, `iz`).

    Examples
    --------
    >>> view_enc = np.array([1.0, 2.0, 3.0])
    >>> slice_enc = np.array([0.1, 0.2])
    >>> iterator = Cartesian3DIterator(view_enc, slice_offset, dummy_shots=2)

    Iterate through the combinations:

    >>> for _ in range(5):
    ...     print(iterator())

    Output with `dummy_shots=2` will first yield two (gy_amp=0.0, gz_amp=0.0) dummy combinations,
    then real values such as (gy_amp=1.0, gz_amp=0.1), (gy_amp=2.0, gz_amp=0.1), and so on.

    """

    def __init__(
        self,
        view_enc: np.ndarray | tuple,
        slice_enc: np.ndarray | tuple,
        dummy_shots=0,
    ):
        args = [view_enc, slice_enc]

        # input checking
        _check_iterator_input([view_enc, slice_enc])

        # create combinations
        if isinstance(args[0], (list, tuple)):
            # combine scaling
            scaling = [arg[0] for arg in args]
            labels = [arg[1] for arg in args]
        else:
            # combine scaling
            scaling = [arg for arg in args]
            labels = None

        self._gy_amp = scaling[0]
        self._gz_amp = scaling[1]
        if labels is not None:
            self._haslabel = True
            self._iy = labels[0]
            self._iz = labels[1]
        else:
            self._haslabel = False

        self.count = -dummy_shots
        self.scanlength = self._gy_amp.shape[0]

    def __call__(self):  # noqa
        if self.count < 0:
            _gy_amp = 0.0
            _gz_amp = 0.0
        else:
            _gy_amp = self._gy_amp[self.count % self.scanlength]
            _gz_amp = self._gz_amp[self.count % self.scanlength]
        scale = SimpleNamespace(gy_amp=_gy_amp, gz_amp=_gz_amp)

        if self._haslabel:
            if self.count < 0:
                _iy = 0
                _iz = 0
            else:
                _iy = self._iy[self.count % self.scanlength]
                _iz = self._iz[self.count % self.scanlength]
            head = SimpleNamespace(iy=_iy, iz=_iz)

        self.count += 1

        if self._haslabel:
            return scale, head

        return scale

    def reset(self):
        """Reset count to 0."""
        self.count = 0
