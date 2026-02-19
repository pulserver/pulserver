"""Iterator for 3D Non Cartesian sampling."""

__all__ = ["NonCartesian3DIterator"]

from types import SimpleNamespace

import numpy as np

from ._base import _check_iterator_input


class NonCartesian3DIterator:
    """
    Iterator for Non Cartesian 3D parameter combinations.

    Parameters
    ----------
    view_enc : np.ndarray | tuple
        The view rotation matrix or a tuple where the first element is the rotation matrix
        and the second element is an associated label array.
        This parameter defines the in-plane coordinates.

    dummy_shots : int, optional
        The number of dummy shots to perform before starting the iteration. During these dummy
        shots, the output will be zeros. Default is 0.


    Attributes
    ----------
    _rotmat : np.ndarray
        The 3D array of rotation matrices(based on the chosen loop position).

    _irot : np.ndarray or None
        The 1D array of labels corresponding to the rotation matrices (if labels are provided).

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
    >>> iterator = NonCartesian3DIterator(view_enc, dummy_shots=2)

    Iterate through the combinations:

    >>> for _ in range(5):
    ...     print(iterator())

    """

    def __init__(
        self,
        view_enc: np.ndarray | tuple,
        dummy_shots=0,
    ):
        args = [view_enc]

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

        self._rotmat = scaling[0]
        if labels is not None:
            self._haslabel = True
            self._irot = labels[0]
        else:
            self._haslabel = False

        self.count = -dummy_shots
        self.scanlength = self._rotmat.shape[0]

    def __call__(self):  # noqa
        if self.count < 0:
            _rotmat = 0.0
        else:
            _rotmat = self._rotmat[self.count % self.scanlength]
        scale = SimpleNamespace(rotmat=_rotmat)

        if self._haslabel:
            if self.count < 0:
                _irot = 0
            else:
                _irot = self._irot[self.count % self.scanlength]
            head = SimpleNamespace(irot=_irot)

        self.count += 1

        if self._haslabel:
            return scale, head

        return scale

    def reset(self):
        """Reset count to 0."""
        self.count = 0
