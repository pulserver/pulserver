"""Image-space helpers and the NumPy reconstruction: centred on the centre of a centred FFT."""

from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.mrd import center_crop
from pulserver.recon.handlers import simplefft


@pytest.mark.parametrize(
    ("current", "size"), [(8, 3), (8, 4), (7, 3), (7, 4), (258, 129)]
)
def test_a_crop_keeps_the_centre_of_a_centred_fft_at_the_centre_of_the_window(
    current, size
):
    window = center_crop(np.arange(current), (size,))
    assert window[size // 2] == current // 2


def _header(x, y):
    matrix = SimpleNamespace(x=x, y=y)
    return SimpleNamespace(
        encoding=[SimpleNamespace(reconSpace=SimpleNamespace(matrixSize=matrix))],
        userParameters=None,
    )


@pytest.mark.parametrize(("n_x", "n_y"), [(8, 8), (9, 7)])
def test_an_object_at_the_isocentre_reconstructs_at_the_centre_of_the_matrix(n_x, n_y):
    """Flat k-space is a point at the isocentre, whatever the parity of the matrix."""
    lines = [SimpleNamespace(data=np.ones((2, n_x), dtype=np.complex64))] * n_y
    image = simplefft._reconstruct(lines, _header(n_x, n_y))
    peak = np.unravel_index(np.argmax(image), image.shape)
    assert peak == (n_x // 2, n_y // 2)
