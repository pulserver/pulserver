"""The phase-cycle reconstruction of balanced SSFP: each cycle solved over shared maps, the cycles combined into one image without bands."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import calibration_header, centred_kspace, coil_phantom, play, readouts

from pulserver.recon.handlers.phase_cycles import (
    COMBINATIONS,
    PhaseCycleRecon,
    combined,
)

pytest.importorskip("bartorch")
blochsim = pytest.importorskip("blochsim")

COILS, MATRIX, TR = 4, 32, 4e-3
#: Off-resonance across the disk, in cycles of 1/TR: two bands for one cycle.
BANDS = 2.0


class Images(PhaseCycleRecon):
    def __init__(self, **kwargs):
        super().__init__(
            kspace_filter=False,
            gradient_unwarp=False,
            intensity_correction=False,
            **kwargs,
        )
        self.results = []

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.append(result)
        return result


def _disk():
    axis = (np.arange(MATRIX) - MATRIX // 2) / (MATRIX / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return (x**2 + y**2 < 0.7**2), x


def _cycles(n_cycles):
    """Coil k-space ``(cycles, coils, y, x)`` of a disk in an off-resonance gradient, one balanced SSFP train per phase cycle."""
    import torch

    disk, x = _disk()
    b0 = torch.from_numpy(BANDS / TR * x / 1.4)
    _, maps, _ = coil_phantom(COILS, MATRIX)
    kspace = []
    for k in range(n_cycles):
        increment = (180.0 + 360.0 * k / n_cycles) % 360.0
        signal = blochsim.bssfp_sim(
            30.0, 1e3 * TR / 2, 1e3 * TR, 1200.0, 80.0, phase_inc=increment, B0=b0
        ).numpy()
        kspace.append(centred_kspace(maps * (disk * signal).astype(np.complex64)))
    return np.stack(kspace)


def _reconstructed(n_cycles, device=None, **kwargs):
    header = calibration_header(COILS, MATRIX)
    if n_cycles > 1:
        header.encoding[0].encodingLimits.set = SimpleNamespace(
            minimum=0, maximum=n_cycles - 1, center=0
        )
    acquisitions = []
    for k, kspace in enumerate(_cycles(n_cycles)):
        cycle = readouts(kspace, last=("ACQ_LAST_IN_SLICE",))
        for acquisition in cycle:
            acquisition.idx.set = k
        acquisitions += cycle
    plugin = Images(**kwargs)
    play(plugin, header, acquisitions, device=device)
    return np.asarray(plugin.results[-1].data)


def _band_depth(image):
    """Spread of the image over the disk: its standard deviation over its mean."""
    disk, _ = _disk()
    inside = np.abs(image)[disk]
    return float(inside.std() / inside.mean())


def test_the_cycles_combine_by_root_sum_of_squares_maximum_or_complex_sum():
    cycles = np.array([[3.0 + 0j], [4.0j]])
    assert [float(combined(cycles, c)[0]) for c in COMBINATIONS] == [5.0, 4.0, 5.0]


def test_an_unknown_combination_is_refused():
    with pytest.raises(ValueError, match="combination"):
        PhaseCycleRecon(combination="mean")


def test_four_phase_cycles_fill_in_the_bands_of_one(device):
    one = _reconstructed(1, device)
    four = _reconstructed(4, device)

    assert one.shape == four.shape == (MATRIX, MATRIX)
    assert _band_depth(four) < 0.5 * _band_depth(one)


def test_the_maximum_of_the_cycles_flattens_the_bands_further_than_their_root_sum_of_squares():
    rss = _reconstructed(4, combination="rss")
    maximum = _reconstructed(4, combination="maximum")

    assert _band_depth(maximum) < _band_depth(rss)
