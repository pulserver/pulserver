"""The pics reconstruction of Cartesian streams: the lines it solves from, the maps it takes and the noise it whitens with."""

from types import SimpleNamespace

import ismrmrd
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

from pulserver.recon.handlers.pics import (
    IMAGINARY_SERIES,
    MAGNITUDE_SERIES,
    REAL_SERIES,
    PicsRecon,
    partial_fourier,
)
from pulserver.recon.handlers.pics_train import PicsTrainRecon

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


class Recording:
    """Keeps what each unit returns, in the list a stream's copy shares."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = []

    def recon(self, context, branch, data):
        result = super().recon(context, branch, data)
        self.results.append(result)
        return result


class Images(Recording, PicsRecon):
    pass


class TrainImages(Recording, PicsTrainRecon):
    pass


def reconstruct(acquisitions, header=None, *, device=None, plugin=None):
    """The context and the image of each unit of ``acquisitions`` that makes one, in the order the units close."""
    plugin = Images() if plugin is None else plugin
    context = play(
        plugin,
        calibration_header(COILS, MATRIX) if header is None else header,
        acquisitions,
        device=device,
    )
    images = [
        np.asarray(result.data)
        for result in results(plugin)
        if result.series_index == MAGNITUDE_SERIES
    ]
    return context, images


def results(plugin):
    """Every ``ReconResult`` the units of ``plugin`` returned, in order."""
    returned = []
    for result in plugin.results:
        if isinstance(result, list):
            returned += result
        elif result is not None:
            returned.append(result)
    return returned


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


def test_maps_of_a_plane_from_calibration_readouts_and_from_the_units_own_centre_are_fitted_with_phase_pole_correction(
    device, monkeypatch
):
    from bartorch import apps

    kspace = vials()
    lines = acquired(range(0, MATRIX, 2))
    nlinv_maps = apps.nlinv_maps
    options = []

    def recorded(*args, **kwargs):
        options.append(kwargs)
        return nlinv_maps(*args, **kwargs)

    monkeypatch.setattr(apps, "nlinv_maps", recorded)
    reconstruct(
        readouts(kspace, CALIBRATION, lines=CENTRE, last=LAST)
        + readouts(kspace, lines=lines, last=LAST),
        device=device,
    )
    reconstruct(stream(kspace, lines), device=device)

    assert [option["phase_pole"] for option in options] == [0, 0]


@pytest.mark.parametrize(
    ("shape", "trajectory", "image", "phase_pole"),
    [
        ((COILS, MATRIX, MATRIX), False, None, 0),
        ((COILS, 1, MATRIX, MATRIX), False, None, 0),
        ((COILS, 2, MATRIX, MATRIX), False, None, None),
        ((COILS, 3, MATRIX, MATRIX), False, None, 0),
        ((COILS, 3, 256, 256), False, None, None),
        ((COILS, 4, 256, 256), False, None, 0),
        ((COILS, 8, MATRIX), True, None, 0),
        ((COILS, 8, MATRIX), True, (MATRIX,) * 3, 0),
        ((COILS, 8, MATRIX), True, (2, MATRIX, MATRIX), None),
    ],
    ids=[
        "plane",
        "one partition",
        "2 partitions",
        "3 partitions",
        "3 of 256",
        "4 of 256",
        "trajectory",
        "3D trajectory",
        "3D trajectory of 2 partitions",
    ],
)
def test_phase_pole_correction_is_asked_unless_the_calibration_grid_is_too_thin(
    shape, trajectory, image, phase_pole, monkeypatch
):
    """The calibration grid is 48 points a side, capped at the matrix, and
    BART's detection circle is a twentieth of its longest side: 3 points
    along every axis are enough on a 32-point matrix, 4 on 256."""
    import torch
    from bartorch import apps

    from pulserver.recon.handlers.pics import nlinv_estimate

    asked = []
    monkeypatch.setattr(
        apps, "nlinv_maps", lambda *args, **kwargs: asked.append(kwargs)
    )
    traj = torch.zeros(shape[1], shape[2], 3) if trajectory else None
    nlinv_estimate(torch.zeros(shape, dtype=torch.complex64), traj, image)

    assert asked == [
        {
            "size": 48,
            "radius": 24.0,
            "shape": image,
            "phase_pole": phase_pole,
        }
    ]


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
    from bartorch import tools

    from pulserver.recon.handlers import pics

    kspace = tools.phantom(64, kspace=True, coils=8).reshape(8, 64, 64).numpy()

    def image():
        acquisitions = readouts(kspace, lines=range(16, 64), last=LAST)
        _, (image,) = reconstruct(
            acquisitions, calibration_header(8, 64), device=device
        )
        return image

    completed = image()
    monkeypatch.setattr(pics, "partial_fourier", lambda image, sampled: image)
    solved = image()

    full = root_sum_of_squares(kspace)
    assert scaled_difference(completed, full) < scaled_difference(solved, full)


def contrasts_in_header(count):
    """The header of ``reconstruct``'s default with ``count`` values of the ``contrast`` counter."""
    header = calibration_header(COILS, MATRIX)
    header.encoding[0].encodingLimits.contrast = SimpleNamespace(
        minimum=0, maximum=count - 1, center=0
    )
    return header


def at_places_of_a_train(acquisitions, places):
    """``acquisitions`` with each line's place in a train of ``places`` as its ``contrast``.

    The last readout of each place closes the slice, as the proxy flags a
    counter that selects images.
    """
    last = {}
    for index, acquisition in enumerate(acquisitions):
        acquisition.idx.contrast = acquisition.idx.kspace_encode_step_1 % places
        last[acquisition.idx.contrast] = index
    for index in last.values():
        acquisitions[index].setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    return acquisitions


def test_the_places_of_a_train_reconstruct_to_the_image_of_the_readouts_without_them(
    device,
):
    places = 4
    kspace = vials()
    lines = acquired(range(0, MATRIX, 2))
    _, (reference,) = reconstruct(stream(kspace, lines), device=device)

    _, (image,) = reconstruct(
        at_places_of_a_train(stream(kspace, lines), places),
        contrasts_in_header(places),
        device=device,
        plugin=TrainImages(),
    )

    assert relative_difference(image, reference) < 1e-4


def test_contrasts_that_are_images_reconstruct_to_one_image_each(device):
    echoes = [vials(), coil_phantom(COILS, MATRIX)[2]]
    acquisitions = []
    for echo, kspace in enumerate(echoes):
        for acquisition in stream(kspace, range(MATRIX)):
            acquisition.idx.contrast = echo
            acquisitions.append(acquisition)

    _, images = reconstruct(
        acquisitions, contrasts_in_header(len(echoes)), device=device
    )

    assert len(images) == len(echoes)
    for image, kspace in zip(images, echoes, strict=True):
        _, (expected,) = reconstruct(stream(kspace, range(MATRIX)), device=device)
        assert relative_difference(image, expected) < 1e-4


@pytest.mark.parametrize("partitions", [2, 3])
def test_a_volume_of_fewer_partitions_than_the_wavelet_needs_reconstructs(
    device, partitions
):
    kspace = vials()[:, None] * (1.0 + np.arange(partitions))[:, None, None]
    kspace = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(kspace, axes=1), axis=1, norm="ortho"), axes=1
    )
    acquisitions = []
    for partition in range(partitions):
        last = LAST if partition == partitions - 1 else ()
        acquisitions += readouts(kspace[:, partition], partition=partition, last=last)

    _, (image,) = reconstruct(
        acquisitions,
        calibration_header(COILS, MATRIX, partitions=partitions),
        device=device,
    )

    assert image.shape == (partitions, MATRIX, MATRIX)
    assert np.isfinite(image).all()
    assert image.max() > 0


#: Compression to at least the rank of the coil data discards nothing, so the
#: image of the virtual channels is that of all of them to single-precision
#: round-off.
COMPRESSION = 1e-5


def test_channels_beyond_the_virtual_coils_are_compressed_before_the_maps_and_the_solve(
    device,
):
    """Eight channels spanning three dimensions of coil space lose nothing in six virtual channels."""
    coils = 8
    _, _, kspace = coil_phantom(coils, MATRIX)
    lines = acquired(range(0, MATRIX, 2))
    header = calibration_header(coils, MATRIX)
    every_channel = Images(virtual_coils=None)

    context, (compressed,) = reconstruct(stream(kspace, lines), header, device=device)
    _, (full,) = reconstruct(
        stream(kspace, lines), header, device=device, plugin=every_channel
    )

    assert context.coil_compression.n_virtual == 6
    assert context.coil_maps[0].maps.shape == (6, MATRIX, MATRIX)
    assert context.coil_maps[0].basis == context.coil_compression.id
    assert relative_difference(compressed, full) < COMPRESSION


@pytest.mark.parametrize("partitions", [1, 8])
def test_the_solve_on_host_k_space_is_bartorchs_pics_with_the_image_on_the_host(
    device, partitions
):
    """The encoding computes on the context's device; the k-space, the maps and the image stay on the host."""
    import torch
    from bartorch import apps, priors

    from pulserver.recon.handlers.pics import nlinv_estimate, pics_image, wavelet_axes

    kspace = vials()[:, None] * (1.0 + np.arange(partitions))[:, None, None]
    kspace = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(kspace, axes=1), axis=1, norm="ortho"), axes=1
    )
    kspace[:, :, 1::2] = 0
    kspace = torch.from_numpy(np.ascontiguousarray(kspace, dtype=np.complex64))
    maps = nlinv_estimate(kspace)
    sampled = kspace.abs().sum(dim=0) > 0
    terms = priors.Wavelet(wavelet_axes(kspace.shape[1:], (-1, -2, -3)), 0.005)
    expected = apps.pics(kspace, maps, regularizers=terms, maxiter=30)

    image = pics_image(kspace, maps, sampled, 0.005, 30, device)

    assert image.device.type == "cpu"
    assert image.shape == kspace.shape[1:]
    assert (
        relative_difference(image.numpy(), expected.reshape(image.shape).numpy()) < 1e-5
    )


def contrasts_led_by_their_calibration(echoes, lines):
    """The readouts of each echo's k-space, the central lines of every echo first and flagged calibration, then the rest, echo by echo within a line."""
    outside = [line for line in lines if line not in CENTRE]
    acquisitions = []
    for flags, selected in ((CALIBRATION_AND_IMAGING,), CENTRE), ((), outside):
        for line in selected:
            for echo, kspace in enumerate(echoes):
                last = LAST if line == outside[-1] else ()
                (acquisition,) = readouts(kspace, *flags, lines=[line], last=last)
                acquisition.idx.contrast = echo
                acquisitions.append(acquisition)
    return acquisitions


class Channels(Images):
    """Keeps the channels each unit's imaging buffer holds, in the list a stream's copy shares."""

    def __init__(self):
        super().__init__()
        self.channels = []

    def recon(self, context, branch, data):
        self.channels.append(data.data.coils)
        return super().recon(context, branch, data)


def test_contrasts_open_together_are_buffered_with_the_virtual_channels_and_lose_nothing(
    device,
):
    """Eight channels spanning three dimensions of coil space lose nothing in six virtual channels from the first echo's calibration."""
    coils = 8
    image, maps, first = coil_phantom(coils, MATRIX)
    echoes = [first, centred_kspace(maps * image**2)]
    lines = acquired(range(0, MATRIX, 2))
    header = calibration_header(coils, MATRIX)
    header.encoding[0].encodingLimits.contrast = SimpleNamespace(
        minimum=0, maximum=1, center=0
    )
    streamed = Channels()

    context, images = reconstruct(
        contrasts_led_by_their_calibration(echoes, lines),
        header,
        device=device,
        plugin=streamed,
    )
    _, full = reconstruct(
        contrasts_led_by_their_calibration(echoes, lines),
        header,
        device=device,
        plugin=Images(virtual_coils=None),
    )

    assert streamed.channels == [6, 6]
    assert context.coil_compression.n_virtual == 6
    for image, reference in zip(images, full, strict=True):
        assert relative_difference(image, reference) < COMPRESSION


def test_a_multi_echo_scan_sends_the_real_and_imaginary_parts_of_each_echo_as_series_of_their_own(
    device,
):
    echoes = [vials(), coil_phantom(COILS, MATRIX)[2]]
    plugin = Images()

    reconstruct(
        contrasts_led_by_their_calibration(echoes, range(MATRIX)),
        contrasts_in_header(len(echoes)),
        device=device,
        plugin=plugin,
    )

    sent = results(plugin)
    series = {
        index: [result for result in sent if result.series_index == index]
        for index in (MAGNITUDE_SERIES, REAL_SERIES, IMAGINARY_SERIES)
    }
    assert [len(series[index]) for index in series] == [2, 2, 2]
    assert {result.image_type for result in series[REAL_SERIES]} == {"real"}
    assert {result.image_type for result in series[IMAGINARY_SERIES]} == {"imaginary"}
    for magnitude, real, imaginary in zip(*series.values(), strict=True):
        modulus = np.abs(real.data + 1j * imaginary.data)
        np.testing.assert_allclose(modulus, magnitude.data, rtol=1e-5, atol=1e-6)
        assert np.abs(imaginary.data).max() > 0


def test_a_single_echo_scan_sends_its_magnitude_alone(device):
    plugin = Images()

    reconstruct(stream(vials(), range(MATRIX)), device=device, plugin=plugin)

    assert [result.image_type for result in results(plugin)] == ["magnitude"]


def ellipsoid(n):
    """The magnitude and the centred k-space of a ``(n, n, n)`` ellipsoid of smoothly varying phase."""
    z, y, x = np.meshgrid(*(np.linspace(-1, 1, n),) * 3, indexing="ij")
    magnitude = (z**2 / 0.6 + y**2 / 0.5 + x**2 / 0.7 < 0.8) * (1 + 0.3 * x)
    image = magnitude * np.exp(1j * (0.8 * y + 0.5 * z + 0.3 * x * y))
    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image)))
    return magnitude, kspace


def truncated(kspace, axes):
    """The image of ``kspace`` with the last quarter of its lines along ``axes`` missing, and the mask of what is left."""
    import torch

    sampled = np.ones(kspace.shape, dtype=bool)
    for axis in axes:
        missing = [slice(None)] * kspace.ndim
        missing[axis] = slice(3 * kspace.shape[axis] // 4, None)
        sampled[tuple(missing)] = False
    image = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace * sampled)))
    return torch.from_numpy(image.astype(np.complex64)), torch.from_numpy(sampled)


def magnitude_error(image, magnitude):
    inside = magnitude > 0
    difference = np.abs(np.asarray(image)) - magnitude
    return np.linalg.norm(difference[inside]) / np.linalg.norm(magnitude[inside])


@pytest.mark.parametrize("n", [32, 33])
def test_homodyne_over_two_partial_fourier_axes_together_beats_one_axis_after_the_other(
    n,
):
    from bartorch import apps

    magnitude, kspace = ellipsoid(n)
    image, sampled = truncated(kspace, (0, 1))

    together = magnitude_error(partial_fourier(image, sampled), magnitude)
    in_turn = magnitude_error(apps.partial_fourier(image, sampled), magnitude)

    assert together < 0.8 * in_turn
    assert together < 0.8 * magnitude_error(image, magnitude)


@pytest.mark.parametrize("axis", [0, 1])
def test_homodyne_along_one_partial_fourier_axis_makes_a_real_image_near_the_full_magnitude(
    axis,
):
    magnitude, kspace = ellipsoid(32)
    image, sampled = truncated(kspace, (axis,))

    completed = partial_fourier(image, sampled)

    assert completed.imag.abs().max() == 0
    assert magnitude_error(completed, magnitude) < 0.5 * magnitude_error(
        image, magnitude
    )


def test_an_image_without_partial_fourier_axes_keeps_its_phase():
    import torch

    _, kspace = ellipsoid(16)
    image, _ = truncated(kspace, ())
    sampled = torch.ones(image.shape, dtype=torch.bool)
    sampled[:, ::2] = False

    assert partial_fourier(image, sampled) is image
