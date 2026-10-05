"""Image-space helpers: centred on the centre of a centred FFT."""

import numpy as np
import pytest

from pulserver.mrd import center_crop


@pytest.mark.parametrize(
    ("current", "size"), [(8, 3), (8, 4), (7, 3), (7, 4), (258, 129)]
)
def test_a_crop_keeps_the_centre_of_a_centred_fft_at_the_centre_of_the_window(
    current, size
):
    window = center_crop(np.arange(current), (size,))
    assert window[size // 2] == current // 2
