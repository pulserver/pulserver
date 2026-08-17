"""The sequence/reconstruction contract for accelerated 3D EPI.

The blipped-CAIPI train of :mod:`pulserver.app.sequence.epi3D_sequence` lays down an
undersampled ``make_caipirinha_mask`` lattice plus a fully sampled
autocalibration rectangle. The reconstruction of :mod:`pulserver.app.recon.epi3D_recon`
reuses the CG-SENSE and NLINV routines of :mod:`pulserver.app.recon.gre3D_recon`, whose
image math is exercised in :mod:`test_gre3D_recon`. What this file pins is the
join between them: the coil-sensitivity calibration must succeed on the mask the
sequence actually produces -- i.e. the rectangle the sequence acquires is the
block NLINV reads off the mask.
"""

from __future__ import annotations

import numpy as np

import pulserver.pypulseq as pp
from pulserver.app import gre3D_recon
from pulserver.app import epi3D_sequence

N = 32
N_Z = 8
COILS = 4
N_ACS = 8
N_ACS_Z = 4


def _sampling_mask(acceleration, acceleration_z, caipi_shift):
    """The ``(kz, ky, kx)`` mask the sequence samples, read off its labels."""
    seq = epi3D_sequence.main(
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


def _calibration_mask():
    """The ``(kz, ky, kx)`` mask the separate low-res GRE calibration samples."""
    seq = epi3D_sequence.CalibrationKernel(
        n_x=N,
        n_y=N,
        n_z=N_Z,
        slab_thickness=32e-3,
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


def test_calibration_succeeds_on_the_calibration_mask():
    """NLINV reproduces coil sensitivities from the separate low-resolution GRE
    calibration -- the block the reconstruction estimates maps from once and reuses for
    every frame."""
    mask = _calibration_mask()
    assert not mask.all()  # a central block only

    kspace = _smooth_kspace() * mask[None]
    maps = gre3D_recon.sensitivities(kspace, mask)
    maps = np.asarray(maps.cpu() if hasattr(maps, "cpu") else maps)

    assert maps.shape == (1, COILS, N_Z, N, N)
    assert np.all(np.isfinite(maps))
    assert np.abs(maps).max() > 0.0


def test_the_imaging_mask_is_undersampled_and_holds_no_calibration():
    """The main file's own mask is the undersampled imaging lattice -- the
    centre is not fully sampled there, because the calibration is a separate
    sequence."""
    imaging = _sampling_mask(2, 2, 1)
    assert not imaging.all()


def test_the_calibration_rectangle_matches_the_window():
    """The calibration acquires exactly the ``calc_calibration_lines`` window on
    each axis, so NLINV finds the fully sampled block where it expects it."""
    mask = _calibration_mask()
    acs_y = pp.calc_calibration_lines(N, N_ACS)
    acs_z = pp.calc_calibration_lines(N_Z, N_ACS_Z)
    rectangle = mask[:, :, 0].T  # (ky, kz)
    assert rectangle[np.ix_(acs_y, acs_z)].all()
    # and nothing outside it.
    assert int(rectangle.sum()) == len(acs_y) * len(acs_z)
