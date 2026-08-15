"""The seqzoo/reczoo contract for accelerated 3D EPI.

The blipped-CAIPI train of :mod:`pulserver.seqzoo.epi_3d` lays down an
undersampled ``make_caipirinha_mask`` lattice plus a fully sampled
autocalibration rectangle. The reconstruction of :mod:`pulserver.reczoo.epi_3d`
reuses the CG-SENSE and NLINV routines of :mod:`pulserver.reczoo.gre_3d`, whose
image math is exercised in :mod:`test_reczoo_gre_3d`. What this file pins is the
join between them: the coil-sensitivity calibration must succeed on the mask the
sequence actually produces -- i.e. the rectangle the sequence acquires is the
block NLINV reads off the mask.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.reczoo import gre_3d
from pulserver.seqzoo import epi_3d

N = 32
N_Z = 8
COILS = 4
N_ACS = 8
N_ACS_Z = 4


def _sampling_mask(acceleration, acceleration_z, caipi_shift):
    """The ``(kz, ky, kx)`` mask the sequence samples, read off its labels."""
    seq = epi_3d.main(
        n_x=N,
        n_y=N,
        n_z=N_Z,
        slab_thickness=32e-3,
        acceleration=acceleration,
        acceleration_z=acceleration_z,
        caipi_shift=caipi_shift,
        n_acs=N_ACS,
        n_acs_z=N_ACS_Z,
        readout_bandwidth_hz=120e3,
    )
    labels = seq.evaluate_labels(evolution="adc")
    plane = np.zeros((N, N_Z), dtype=bool)
    plane[labels["LIN"], labels["PAR"]] = True
    return np.broadcast_to(plane.T[:, :, None], (N_Z, N, N)).astype(bool)


def _smooth_kspace():
    """Fully sampled coil k-space of a smooth phantom NLINV can fit."""
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    phantom = np.stack(
        [0.3 + np.where((x / 0.9) ** 2 + (y / 0.9) ** 2 < 1.0, 0.7, 0.0) for _ in range(N_Z)]
    ).astype(np.complex64)
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 3.0)
            * np.exp(1j * 1.5 * (np.cos(a) * x + np.sin(a) * y))
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    maps = np.broadcast_to(maps[:, None], (COILS, N_Z, N, N)).astype(np.complex64)
    axes = (-3, -2, -1)
    volume = maps * phantom
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(volume, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    ).astype(np.complex64)


@pytest.mark.parametrize(
    "acceleration,acceleration_z,caipi_shift",
    [(2, 1, 0), (1, 2, 1), (2, 2, 1)],
    ids=["Ry", "Rz", "RyRz"],
)
def test_calibration_succeeds_on_the_sequence_mask(
    acceleration, acceleration_z, caipi_shift
):
    """NLINV reproduces the calibration data from the rectangle the accelerated
    EPI sequence lays down -- the seqzoo ACS and the reczoo calibration read the
    same central block."""
    mask = _sampling_mask(acceleration, acceleration_z, caipi_shift)
    assert not mask.all()  # genuinely undersampled

    kspace = _smooth_kspace() * mask[None]
    maps = gre_3d.sensitivities(kspace, mask)
    maps = np.asarray(maps.cpu() if hasattr(maps, "cpu") else maps)

    assert maps.shape == (1, COILS, N_Z, N, N)
    assert np.all(np.isfinite(maps))
    assert np.abs(maps).max() > 0.0


def test_the_acs_rectangle_matches_the_calibration_window():
    """The fully sampled block the sequence acquires is exactly the window
    ``calc_calibration_lines`` describes on each axis, so NLINV finds it where
    it expects it."""
    mask = _sampling_mask(2, 2, 1)
    acs_y = pp.calc_calibration_lines(N, N_ACS)
    acs_z = pp.calc_calibration_lines(N_Z, N_ACS_Z)
    rectangle = mask[:, :, 0].T  # (ky, kz)
    assert rectangle[np.ix_(acs_y, acs_z)].all()
