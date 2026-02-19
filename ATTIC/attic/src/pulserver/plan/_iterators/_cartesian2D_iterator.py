"""Iterator for 2D Cartesian sampling."""

__all__ = ["Cartesian2DIterator"]

from types import SimpleNamespace

import numpy as np

from ._base import _check_iterator_input


class Cartesian2DIterator:
    """
    Iterator for Cartesian 2D parameter combinations, handling both inner and outer looping.

    This class generates combinations of two encoding axes, i.e., view encoding, slice offset,
    allowing for an "inner" or "outer" loop position to control which axis iterates faster.
    It also supports an optional label system for associating each combination with corresponding
    label indices.

    Parameters
    ----------
    view_enc : np.ndarray | tuple
        The view encoding steps or a tuple where the first element is the phase encoding gradient
        scaling array and the second element is an associated label array.
        This parameter defines one axis of the Cartesian grid.

    slice_offset : np.ndarray | tuple
        The slice offset steps or a tuple where the first element is the slice frequency offset array
        and the second element is an associated label array.
        This parameter defines the slice position.

    view_loop_position : str
        Specifies the looping behavior. Either 'inner' or 'outer', which determines whether the
        `view_enc` or `slice_offset` iterates faster. If 'inner', `view_enc` iterates faster
        than `slice_offset`; if 'outer', the opposite occurs.

    dummy_shots : int, optional
        The number of dummy shots to perform before starting the iteration. During these dummy
        shots, the output will be zeros. Default is 0.

    Raises
    ------
    ValueError
        If `view_loop_position` is neither 'inner' nor 'outer'.

    Attributes
    ----------
    _gy_amp : np.ndarray
        The 1D array of gy encoding amplitudes (based on the chosen loop position).

    _rf_freq : np.ndarray
        The 1D array of RF frequencies (based on the chosen loop position).

    _iy : np.ndarray or None
        The 1D array of labels corresponding to the view encoding axis (if labels are provided).

    _iz : np.ndarray or None
        The 1D array of labels corresponding to the slice offset axis (if labels are provided).

    _haslabel : bool
        Indicates whether labels are provided for the view encoding and slice offset axes.

    count : int
        The current iteration count, starting from `-dummy_shots`.

    Methods
    -------
    __call__()
        Returns the next combination of parameters (`gy_amp`, `rf_freq`) and optional labels (`iy`, `iz`).

    Examples
    --------
    >>> view_enc = np.array([1.0, 2.0, 3.0])
    >>> slice_offset = np.array([0.1, 0.2])
    >>> iterator = Cartesian2DIterator(view_enc, slice_offset, view_loop_position='inner', dummy_shots=2)

    Iterate through the combinations:

    >>> for _ in range(5):
    ...     print(iterator())

    Output with `dummy_shots=2` will first yield two (gy_amp=0.0, rf_freq=0.0) dummy combinations,
    then real values such as (gy_amp=1.0, rf_freq=0.1), (gy_amp=2.0, rf_freq=0.1), and so on.

    """

    def __init__(
        self,
        view_enc: np.ndarray | tuple,
        slice_offset: np.ndarray | tuple,
        view_loop_position: str,
        dummy_shots=0,
    ):
        if view_loop_position == "inner":
            args = [view_enc, slice_offset]
        elif view_loop_position == "outer":
            args = [slice_offset, view_enc]
        else:
            raise ValueError(
                f"Unrecognized view loop position: {view_loop_position} - must be either 'inner' or 'outer'."
            )

        # input checking
        _check_iterator_input(args)

        # create combinations
        if isinstance(args[0], (list, tuple)):
            # combine scaling
            _scaling = [arg[0] for arg in args]
            scaling = np.meshgrid(*_scaling, indexing="xy")
            scaling = [scale.ravel() for scale in scaling]

            _labels = [arg[1] for arg in args]
            labels = np.meshgrid(*_labels, indexing="xy")
            labels = [label.ravel() for label in labels]
        else:
            # combine scaling
            _scaling = [arg for arg in args]
            scaling = np.meshgrid(*_scaling, indexing="xy")
            scaling = [scale.ravel() for scale in scaling]
            labels = None

        if view_loop_position == "inner":
            self._gy_amp = scaling[0]
            self._rf_freq = scaling[1]
            if labels is not None:
                self._haslabel = True
                self._iy = labels[0]
                self._iz = labels[1]
            else:
                self._haslabel = False
        else:
            self._gy_amp = scaling[1]
            self._rf_freq = scaling[0]
            if labels is not None:
                self._haslabel = True
                self._iy = labels[1]
                self._iz = labels[0]
            else:
                self._haslabel = False

        self.count = -dummy_shots
        self.scanlength = self._gy_amp.shape[0]

    def __call__(self):  # noqa
        if self.count < 0:
            _gy_amp = 0.0
            _rf_freq = 0.0
        else:
            _gy_amp = self._gy_amp[self.count % self.scanlength]
            _rf_freq = self._rf_freq[self.count % self.scanlength]
        scale = SimpleNamespace(gy_amp=_gy_amp, rf_freq=_rf_freq)

        if self._haslabel:
            if self.count < 0:
                _iy = 0
                _iz = 0
            else:
                _iy = self._iy[self.count % self.scanlength]
                _iz = self._iz[self.count % self.scanlength]
            head = SimpleNamespace(iy=_iy, islice=_iz)

        self.count += 1

        if self._haslabel:
            return scale, head

        return scale

    def reset(self):
        """Reset count to 0."""
        self.count = 0
