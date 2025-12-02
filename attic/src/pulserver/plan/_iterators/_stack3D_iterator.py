"""Iterator for 3D Stack-of- Non Cartesian sampling."""

__all__ = ["Stack3DIterator"]

from types import SimpleNamespace

import numpy as np

from ._base import _check_iterator_input


class Stack3DIterator:
    """
    Iterator for 3D Stack-of- Non Cartesian parameter combinations, handling both inner and outer looping.

    This class generates combinations of two encoding axes, i.e., view encoding, slice encoding,
    allowing for an "inner" or "outer" loop position to control which axis iterates faster.
    It also supports an optional label system for associating each combination with corresponding
    label indices.

    Parameters
    ----------
    view_enc : np.ndarray | tuple
        The view rotation matrix or a tuple where the first element is the rotation matrix
        and the second element is an associated label array.
        This parameter defines the in-plane coordinates.

    slice_offset : np.ndarray | tuple
        The slice encoding steps or a tuple where the first element is the scaling array and the
        second element is an associated label array. This parameter defines the slice encoding position.

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
    _rotmat : np.ndarray
        The 3D array of rotation matrices (based on the chosen loop position).

    _gz_amp : np.ndarray
        The 1D array of gz encoding amplitudes (based on the chosen loop position).

    _irot : np.ndarray or None
        The 1D array of labels corresponding to the rotation matrices (if labels are provided).

    _iz : np.ndarray or None
        The 1D array of labels corresponding to the slice encoding axis (if labels are provided).

    _haslabel : bool
        Indicates whether labels are provided for the view encoding and slice offset axes.

    count : int
        The current iteration count, starting from `-dummy_shots`.

    Methods
    -------
    __call__()
        Returns the next combination of parameters (`rotmat`, `rf_freq`) and optional labels (`irot`, `iz`).

    Examples
    --------
    >>> view_enc = np.stack((np.eye(3), np.eye(3), np.eye(3)), axis=0)
    >>> slice_enc = np.array([0.1, 0.2])
    >>> iterator = Stack3DIterator(view_enc, slice_enc, view_loop_position='inner', dummy_shots=2)

    Iterate through the combinations:

    >>> for _ in range(5):
    ...     print(iterator())

    Output with `dummy_shots=2` will first yield two (rotmat=[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rf_freq=0.0)
    dummy combinations, then real values such as (rotmat=[[1.0, 0.0, 0.0],[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], rf_freq=0.1),
    and so on.

    """

    def __init__(
        self,
        view_enc: np.ndarray | tuple,
        slice_enc: np.ndarray | tuple,
        view_loop_position: str,
        dummy_shots=0,
    ):
        if view_loop_position == "inner":
            args = [view_enc, slice_enc]
        elif view_loop_position == "outer":
            args = [slice_enc, view_enc]
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
            self._rotmat = scaling[0]
            self._gz_amp = scaling[1]
            if labels is not None:
                self._haslabel = True
                self._irot = labels[0]
                self._iz = labels[1]
            else:
                self._haslabel = False
        else:
            self._rotmat = scaling[1]
            self._gz_amp = scaling[0]
            if labels is not None:
                self._haslabel = True
                self._irot = labels[1]
                self._iz = labels[0]
            else:
                self._haslabel = False

        self.count = -dummy_shots
        self.scanlength = self._rotmat.shape[0]

    def __call__(self):  # noqa
        if self.count < 0:
            _rotmat = 0.0
            _gz_amp = 0.0
        else:
            _rotmat = self._rotmat[self.count % self.scanlength]
            _gz_amp = self._gz_amp[self.count % self.scanlength]
        scale = SimpleNamespace(rotmat=_rotmat, gz_amp=_gz_amp)

        if self._haslabel:
            if self.count < 0:
                _irot = 0
                _iz = 0
            else:
                _irot = self._irot[self.count % self.scanlength]
                _iz = self._iz[self.count % self.scanlength]
            head = SimpleNamespace(irot=_irot, iz=_iz)

        self.count += 1

        if self._haslabel:
            return scale, head

        return scale

    def reset(self):
        """Reset count to 0."""
        self.count = 0
