"""The pics reconstruction of Cartesian streams: the lines it solves from, the maps it takes and the noise it whitens with."""

import numpy as np
import pytest
from conftest import (
    calibration_header,
    centred_kspace,
    coil_phantom,
    play,
    readouts,
    relative_difference,
)

from pulserver.recon.handlers.pics import PicsRecon

pytest.importorskip("bartorch")

COILS, MATRIX, ACS = 4, 32, 12
CENTRE = range(MATRIX // 2 - ACS // 2, MATRIX // 2 + ACS // 2)
LAST = ("ACQ_LAST_IN_SLICE",)
CALIBRATION = "ACQ_IS_PARALLEL_CALIBRATION"
CALIBRATION_AND_IMAGING = "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING"
NOISE = "ACQ_IS_NOISE_MEASUREMENT"

#: The tolerance on the image of every other phase encode against the image of
#: all of them, after a scale. The undersampled solve differs by what the
#: wavelet term shrinks of the vials' edges from fewer data, and by the maps
#: fitted to 12 lines: a few per cent. The zero-filled lines it is given are
#: several times further from the image of all of them, which the test checks.
UNDERSAMPLING = 7e-2


class Images(PicsRecon):
    """Keeps what each unit returns, in the list a stream's copy shares."""

    def __init__(self):
        super().__init__()
        self.results = []

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.append(result)
        return result


def reconstruct(acquisitions, header=None, *, device=None):
    """The context and the image of each unit of ``acquisitions`` that makes one, in the order the units close."""
    plugin = Images()
    context = play(
        plugin,
        calibration_header(COILS, MATRIX) if header is None else header,
        acquisitions,
        device=device,
    )
    images = [
        np.asarray(result.data) for result in plugin.results if result is not None
    ]
    return context, images


def vials():
    """The centred k-space ``(coils, y, x)`` of seven vials of different intensity on a ring around one in the middle, seen by ``COILS`` coils."""
    axis = (np.arange(MATRIX) - MATRIX // 2) / (MATRIX / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros((MATRIX, MATRIX), dtype=np.complex64)
    radius = 15 / 110
    for index, angle in enumerate(2 * np.pi * np.arange(7) / 7):
        vial = (x - 45 / 110 * np.cos(angle)) ** 2 + (y - 45 / 110 * np.sin(angle)) ** 2
        image[vial < radius**2] = 0.4 + 0.1 * index
    image[x**2 + y**2 < radius**2] = 1.0
    maps = coil_phantom(COILS, MATRIX)[1]
    return centred_kspace(maps * image)


def correlated_noise():
    """Complex Gaussian noise ``(coils, 16, MATRIX)`` of channels that are correlated and of unequal power."""
    generator = np.random.default_rng(3)
    shape = (COILS, 16, MATRIX)
    mixing = np.eye(COILS) + 0.6 * (
        generator.standard_normal((COILS, COILS))
        + 1j * generator.standard_normal((COILS, COILS))
    )
    white = generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    return np.einsum("ij,jls->ils", mixing, white).astype(np.complex64)


def acquired(lines):
    """``lines`` and the central ``ACS`` lines, in order."""
    return sorted({*lines, *CENTRE})


def stream(kspace, lines):
    """The readouts of ``lines`` of ``kspace``, the central ones also calibration; the slice closes with the last."""
    outside = [line for line in lines if line not in CENTRE]
    return readouts(kspace, CALIBRATION_AND_IMAGING, lines=CENTRE) + readouts(
        kspace, lines=outside, last=LAST
    )


def root_sum_of_squares(kspace):
    axes = (1, 2)
    images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes)), axes=axes
    )
    return np.sqrt((np.abs(images) ** 2).sum(axis=0))


def scaled_difference(image, reference):
    """The norm of ``image - reference`` over that of ``reference``, once ``image`` is scaled onto it in the least squares.

    ``pics`` leaves its image at the scale of the data it was given, which it
    estimates from the fully sampled centre of k-space, so the images of two
    sampling patterns agree up to a scale.
    """
    scale = (image * reference).sum() / (image * image).sum()
    return relative_difference(scale * image, reference)


def test_every_other_phase_encode_and_the_centre_reconstruct_to_the_image_of_all_lines(
    device,
):
    kspace = vials()
    lines = acquired(range(0, MATRIX, 2))
    zero_filled = np.zeros_like(kspace)
    zero_filled[:, lines] = kspace[:, lines]

    _, (full,) = reconstruct(stream(kspace, range(MATRIX)), device=device)
    _, (undersampled,) = reconstruct(stream(kspace, lines), device=device)

    assert scaled_difference(undersampled, full) < UNDERSAMPLING
    aliased = root_sum_of_squares(zero_filled)
    assert scaled_difference(aliased, full) > 2 * UNDERSAMPLING


def test_a_unit_of_calibration_readouts_makes_no_image_and_its_maps_serve_the_next_unit_of_its_slice(
    device, monkeypatch
):
    from bartorch import apps

    kspace = vials()
    lines = acquired(range(0, MATRIX, 2))
    _, (calibrated_by_its_readouts,) = reconstruct(stream(kspace, lines), device=device)
    nlinv_maps = apps.nlinv_maps
    estimates = []

    def counted(*args, **kwargs):
        estimates.append(args[0].shape)
        return nlinv_maps(*args, **kwargs)

    monkeypatch.setattr(apps, "nlinv_maps", counted)

    context, images = reconstruct(
        readouts(kspace, CALIBRATION, lines=CENTRE, last=LAST)
        + readouts(kspace, lines=lines, last=LAST),
        device=device,
    )

    (image,) = images
    assert len(estimates) == 1
    assert list(context.coil_maps) == [0]
    assert relative_difference(image, calibrated_by_its_readouts) < 1e-4


def test_noise_readouts_whiten_the_imaging_readouts_and_are_not_placed(device):
    kspace = vials()
    lines = acquired(range(0, MATRIX, 2))

    context, (image,) = reconstruct(
        readouts(correlated_noise(), NOISE) + stream(kspace, lines), device=device
    )
    whitened = np.einsum("ij,jls->ils", context.noise.matrix, kspace)
    _, (reference,) = reconstruct(
        stream(whitened.astype(kspace.dtype), lines), device=device
    )
    _, (unwhitened,) = reconstruct(stream(kspace, lines), device=device)

    assert relative_difference(image, reference) < 1e-4
    assert relative_difference(unwhitened, reference) > 1e-2


def test_a_partial_fourier_image_is_closer_to_the_full_one_than_the_solve_of_its_lines_alone(
    device, monkeypatch
):
    from bartorch import apps, tools

    kspace = tools.phantom(64, kspace=True, coils=8).reshape(8, 64, 64).numpy()

    def image():
        acquisitions = readouts(kspace, lines=range(16, 64), last=LAST)
        _, (image,) = reconstruct(
            acquisitions, calibration_header(8, 64), device=device
        )
        return image

    completed = image()
    monkeypatch.setattr(apps, "partial_fourier", lambda image, sampled: image)
    solved = image()

    full = root_sum_of_squares(kspace)
    assert scaled_difference(completed, full) < scaled_difference(solved, full)
