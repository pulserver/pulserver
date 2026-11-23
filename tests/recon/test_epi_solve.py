"""The EPI reconstruction's coil combination, partial-Fourier solve and stack coil maps."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("bartorch")
torch = pytest.importorskip("torch")


def _phantom(shape=(1, 16, 16)):
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    disc = ((y / 6.0) ** 2 + (x / 5.0) ** 2 < 1).astype(np.complex64)
    return disc * np.exp(0.4j * x / shape[-1] + 0.2j * y / shape[-2]).astype(
        np.complex64
    )


def _maps(shape=(1, 16, 16), coils=4):
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    maps = np.stack(
        [
            np.exp(
                -(((y - 4 * np.cos(c)) / 12.0) ** 2 + ((x - 4 * np.sin(c)) / 12.0) ** 2)
            )
            * np.exp(1j * c)
            for c in range(coils)
        ]
    ).astype(np.complex64)
    return maps / np.sqrt((np.abs(maps) ** 2).sum(axis=0, keepdims=True))


def _kspace(image, maps):
    import bartorch

    coils = torch.from_numpy(maps * image[None])
    return bartorch.fft(coils, (-1, -2), unitary=True).reshape(coils.shape)


def test_a_fully_sampled_unit_is_combined_into_the_object_its_maps_weight(device):
    from pulserver.recon.handlers.epi import sense_combined

    image, maps = _phantom(), _maps()
    kspace = _kspace(image, maps).to(device)

    combined = sense_combined(kspace, torch.from_numpy(maps).to(device))

    np.testing.assert_allclose(combined.cpu().numpy(), image, atol=1e-5)


def test_conjugate_gradients_unfold_a_twofold_undersampled_unit(device):
    from pulserver.recon.handlers.epi import cg_sense

    image, maps = _phantom(), _maps(coils=8)
    kspace = _kspace(image, maps)
    sampled = torch.zeros(1, 16, 16, dtype=torch.bool)
    sampled[:, ::2] = True

    solved = cg_sense(kspace * sampled, torch.from_numpy(maps), sampled, 30, device)

    error = np.linalg.norm(solved.numpy() - image) / np.linalg.norm(image)
    assert error < 0.02


def test_the_default_iterations_unfold_a_threefold_undersampled_object_filling_the_field_of_view(
    device,
):
    from pulserver.recon.handlers.epi import EpiRecon, cg_sense

    shape = (1, 48, 48)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    image = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1).astype(np.complex64)
    image += 0.5 * ((y / 8.0) ** 2 + ((x - 6) / 3.0) ** 2 < 1)
    maps = _maps(shape, coils=12)
    kspace = _kspace(image, maps)
    sampled = torch.zeros(shape, dtype=torch.bool)
    sampled[:, 1::3] = True

    solved = cg_sense(
        kspace * sampled,
        torch.from_numpy(maps),
        sampled,
        EpiRecon().iterations,
        device,
    )

    error = np.linalg.norm(solved.numpy() - image) / np.linalg.norm(image)
    assert error < 0.05


def _gradient_echo(shape=(1, 48, 48)):
    """A disc whose phase wraps several times within a few voxels of one spot, as a gradient echo's near an air cavity."""
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    magnitude = ((y / 18.0) ** 2 + (x / 15.0) ** 2 < 1).astype(np.float32)
    cavity = 6 * np.pi * np.exp(-((x**2 + (y - 10) ** 2) / (2 * 3.0**2)))
    return magnitude, cavity, x, y


def _partial_fourier(shape, ry=1):
    sampled = torch.zeros(shape, dtype=torch.bool)
    sampled[:, shape[1] // 4 :: ry] = True
    return sampled


def _magnitude_error(solved, magnitude):
    residual = np.abs(solved.cpu().numpy()) - magnitude
    return np.linalg.norm(residual) / np.linalg.norm(magnitude)


@pytest.mark.parametrize("ry", [1, 2])
def test_a_partial_fourier_unit_takes_the_phase_of_its_full_k_reference_and_its_drift(
    device, ry
):
    from pulserver.recon.handlers.epi import EpiRecon, cg_sense

    shape = (1, 48, 48)
    magnitude, cavity, x, y = _gradient_echo(shape)
    drift = 0.6 * np.pi * x / 48 - 0.4 * np.pi * y / 48
    image = (magnitude * np.exp(1j * (cavity + drift))).astype(np.complex64)
    reference = torch.from_numpy((magnitude * np.exp(1j * cavity)).astype(np.complex64))
    maps = torch.from_numpy(_maps(shape, coils=12))
    sampled = _partial_fourier(shape, ry)
    kspace = _kspace(image, maps.numpy()) * sampled
    plugin = EpiRecon(0.0)

    solved = plugin._solved(kspace, maps, SimpleNamespace(device=device), reference)
    zero_filled = cg_sense(kspace, maps, sampled, plugin.iterations, device)

    assert _magnitude_error(solved, magnitude) < 0.3 * _magnitude_error(
        zero_filled, magnitude
    )


def test_a_spin_echo_partial_fourier_unit_takes_its_own_smooth_phase(device):
    from pulserver.recon.handlers.epi import EpiRecon, cg_sense

    shape = (1, 48, 48)
    magnitude, _, x, _ = _gradient_echo(shape)
    image = (magnitude * np.exp(1j * np.pi * (x / 48) ** 2)).astype(np.complex64)
    maps = torch.from_numpy(_maps(shape, coils=12))
    sampled = _partial_fourier(shape)
    kspace = _kspace(image, maps.numpy()) * sampled
    plugin = EpiRecon(0.0)

    solved = plugin._solved(kspace, maps, SimpleNamespace(device=device))
    zero_filled = cg_sense(kspace, maps, sampled, plugin.iterations, device)

    assert _magnitude_error(solved, magnitude) < 0.3 * _magnitude_error(
        zero_filled, magnitude
    )


def test_a_gradient_echo_partial_fourier_unit_without_a_reference_is_left_zero_filled_with_a_warning(
    device, caplog
):
    import logging

    from pulserver.recon.handlers.epi import EpiRecon, cg_sense

    shape = (1, 48, 48)
    magnitude, cavity, _, _ = _gradient_echo(shape)
    image = (magnitude * np.exp(1j * cavity)).astype(np.complex64)
    maps = torch.from_numpy(_maps(shape, coils=12))
    sampled = _partial_fourier(shape)
    kspace = _kspace(image, maps.numpy()) * sampled
    plugin = EpiRecon(0.0)

    with caplog.at_level(logging.WARNING, logger="pulserver.recon"):
        solved = plugin._solved(kspace, maps, SimpleNamespace(device=device))

    zero_filled = cg_sense(
        kspace, maps, sampled, plugin.iterations, device, l2=plugin.l2
    )
    np.testing.assert_allclose(solved.cpu().numpy(), zero_filled.cpu().numpy())
    assert any("no full-k phase reference" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("rows", "fills"),
    [(range(16), True), (range(5, 16), True), (range(0, 16, 2), False)],
    ids=["full", "partial Fourier", "undersampled"],
)
def test_a_unit_fills_its_acquired_range_only_when_no_line_inside_it_is_missing(
    rows, fills
):
    from pulserver.recon.handlers.epi import _fills_acquired_range

    sampled = np.zeros((1, 16, 16), dtype=bool)
    sampled[:, list(rows)] = True

    assert _fills_acquired_range(sampled) is fills


def _calibration_unit(location, kspace, position):

    reference = SimpleNamespace(
        position=(0.0, 0.0, position),
        slice_dir=(0.0, 0.0, 1.0),
        read_dir=(1.0, 0.0, 0.0),
        phase_dir=(0.0, 1.0, 0.0),
        patient_table_position=(0.0, 0.0, 0.0),
    )
    space = SimpleNamespace(
        index=0,
        cartesian=True,
        extents={"partition": 1, "phase_encode": kspace.shape[1]},
        recon_fov=(0.005, 0.2, 0.2),
        recon_matrix=(1, kspace.shape[1], kspace.shape[2]),
    )
    ref = SimpleNamespace(
        space=space,
        kspace=kspace,
        axes=("coil", "phase_encode", "readout"),
        origin={"phase_encode": 0},
        coils=kspace.shape[0],
        reference=reference,
    )
    return SimpleNamespace(data=None, ref=ref, counters={"slice": location, "set": 0})


def test_the_stack_maps_are_estimated_once_in_slice_position_order_and_stored_per_slice():
    from pulserver.recon import ReconContext, volume_coil_maps

    context = ReconContext.offline()
    calls = []

    def estimate(kspace):
        calls.append(tuple(kspace.shape))
        # The image of each plane along the stack, in stacking order.
        import bartorch

        planes = bartorch.fft(kspace, (1,), inverse=True, unitary=True)
        return planes.reshape(kspace.shape)

    plane = np.ones((2, 4, 4), np.complex64)
    units = [
        _calibration_unit(location, plane * (location + 1), position)
        for location, position in ((0, 10.0), (1, -10.0), (2, 0.0))
    ]

    volume_coil_maps(context, units, estimate=estimate)

    assert calls == [(2, 3, 4, 4)]
    for location in range(3):
        stored = context.coil_maps[location].maps
        np.testing.assert_allclose(stored, plane * (location + 1), atol=1e-5)


def _multiband(device, coils=8, grid=(24, 24), maps_out=None):
    """Two bands' single-slice k-space and their blipped-CAIPI partition volume; each band's maps are appended to ``maps_out``."""
    import bartorch

    image = _phantom((1, *grid))[0]
    shifted = np.roll(image, 5, axis=-1) * 0.7
    maps = _maps((1, *grid), coils)[:, 0]
    other = np.roll(maps, coils // 2, axis=0) * np.exp(
        1.1j * np.arange(grid[1]) / grid[1]
    )
    if maps_out is not None:
        maps_out += [maps, other.astype(np.complex64)]
    singles = torch.stack(
        [_kspace(image, maps), _kspace(shifted, other.astype(np.complex64))]
    ).to(device)
    unit = torch.eye(2, dtype=torch.complex64, device=device)
    weights = bartorch.fft(unit, (1,), unitary=True).reshape(2, 2)
    volume = torch.zeros(coils, 2, *grid, dtype=torch.complex64, device=device)
    for line in range(grid[0]):
        encode = line % 2
        volume[:, encode, line] = (
            weights[:, encode, None, None] * singles[:, :, line]
        ).sum(0)
    return singles, volume


def _band_unit(position, partitions=2):
    reference = SimpleNamespace(
        position=(0.0, 0.0, position), slice_dir=(0.0, 0.0, 1.0)
    )
    space = SimpleNamespace(recon_fov=(0.005 * partitions, 0.2, 0.2))
    buffer = SimpleNamespace(reference=reference, space=space)
    return SimpleNamespace(data=buffer, counters={"set": 1})


def test_the_band_planes_stack_the_slices_at_each_band_position_preferring_the_unit_set():
    from pulserver.recon.handlers.epi import _band_planes

    plane = np.ones((3, 4, 4), np.complex64)
    stored = [
        (0, -2.5, plane * 1),
        (0, 2.5, plane * 2),
        (1, 2.5, plane * 3),
    ]

    planes = _band_planes(stored, _band_unit(2.5), (3, 2, 4, 4))

    assert planes.shape == (3, 2, 4, 4)
    np.testing.assert_array_equal(planes[0, :, 0, 0], [1, 3])


def test_the_band_planes_are_absent_when_a_band_slice_was_not_calibrated():
    from pulserver.recon.handlers.epi import _band_planes

    stored = [(0, -2.5, np.ones((3, 4, 4), np.complex64))]

    assert _band_planes(stored, _band_unit(2.5), (3, 2, 4, 4)) is None


def _calibrated_plugin(singles=None, **arguments):
    """An EPI reconstruction whose stream calibrated the slices at -2.5 and 2.5 mm."""
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi

    context = ReconContext.offline()
    plugin = epi.EpiRecon(
        gradient_unwarp=False, intensity_correction=False, **arguments
    )
    plugin.startup(context)
    planes = (None, None) if singles is None else singles.numpy()
    plugin.slice_calibrations = [(0, -2.5, planes[0]), (0, 2.5, planes[1])]
    return plugin, context


def test_each_band_of_a_multiband_unit_is_fermi_filtered_in_its_own_plane():
    """Filtered across the stack, each band of an MB2 unit took in a quarter of the other: two slices superimposed."""
    from pulserver.recon._correct import fermi_filtered

    maps = []
    singles, volume = _multiband("cpu", maps_out=maps)
    maps = torch.from_numpy(np.stack(maps, axis=1))
    images = {}
    for kspace_filter in (False, True):
        plugin, context = _calibrated_plugin(singles, kspace_filter=kspace_filter)
        plugin._maps = lambda context, data, volume: maps
        images[kspace_filter] = plugin.image(
            volume.numpy(), tuple(volume.shape[1:]), context, _band_unit(2.5)
        )

    for band in range(2):
        np.testing.assert_allclose(
            images[True][band],
            fermi_filtered(images[False][band]),
            atol=1e-6 * float(np.abs(images[False]).max()),
        )


def _caipi(ry, coils=6, grid=(48, 48)):
    """Two bands of steep sensitivities and their blipped-CAIPI partition volume, ``ry``-fold undersampled with partial Fourier 3/4.

    Returns the bands, their maps, their single-band calibration (the central
    third of k-space) and the volume.
    """
    import bartorch

    y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in grid), indexing="ij")
    first = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1) + 0.5 * (
        ((y - 6) / 5.0) ** 2 + ((x + 5) / 3.0) ** 2 < 1
    )
    second = ((y / 18.0) ** 2 + (x / 19.0) ** 2 < 1) - 0.4 * (
        ((y + 6) / 3.0) ** 2 + ((x - 4) / 6.0) ** 2 < 1
    )
    phase = np.exp(1j * (0.6 * np.sin(x / 7.0) + 0.4 * y / grid[0]))
    bands = (np.stack([first, second]) * phase).astype(np.complex64)
    steep = _steep_maps(grid, coils)
    maps = np.stack(
        [steep, np.roll(steep, 1, axis=0) * np.exp(1j * x / grid[1])], axis=1
    ).astype(np.complex64)
    singles = _kspace(bands, maps).transpose(0, 1)
    weights = bartorch.fft(
        torch.eye(2, dtype=torch.complex64), (1,), unitary=True
    ).reshape(2, 2)
    volume = torch.zeros(coils, 2, *grid, dtype=torch.complex64)
    centre = grid[0] // 2
    for line in range(grid[0] // 4, grid[0]):
        if (line - centre) % ry == 0:
            # The band encode steps by one partition every ry lines, the
            # CAIPI shift epi2D_sequence plays at MB2.
            encode = (1 + (line - centre) // ry) % 2
            volume[:, encode, line] = (
                weights[:, encode, None, None] * singles[:, :, line]
            ).sum(0)
    calibration = torch.zeros_like(singles)
    calibration[:, :, grid[0] // 3 : 2 * grid[0] // 3] = singles[
        :, :, grid[0] // 3 : 2 * grid[0] // 3
    ]
    return bands, maps, calibration, volume


@pytest.mark.parametrize(("ry", "bound"), [(3, 0.25), (4, 0.3)])
def test_a_caipi_multiband_unit_beyond_twofold_in_plane_is_unfolded_by_one_solve_of_its_bands(
    ry, bound
):
    """Separated first by split-slice GRAPPA and solved band by band, its bands erred by 0.34 of their norm at R3 and by 1.1 at R4."""
    bands, maps, calibration, volume = _caipi(ry)
    plugin, context = _calibrated_plugin(calibration, kspace_filter=False)
    plugin._maps = lambda context, data, volume: torch.from_numpy(maps)
    plugin.phase_references[0] = torch.from_numpy(bands)
    unit = _band_unit(2.5)
    unit.counters = {"set": 0, "slice": 0}

    image = plugin.image(volume.numpy(), tuple(volume.shape[1:]), context, unit)

    for band in range(2):
        error = np.linalg.norm(np.abs(image[band]) - np.abs(bands[band]))
        assert error / np.linalg.norm(bands[band]) < bound


def test_the_bands_of_a_multiband_unit_are_unwarped_and_divided_by_a_field_each_in_its_plane():
    plugin, context = _calibrated_plugin()
    calls = []

    class Corrections:
        def unwarped(self, image, context, data, planes=None):
            calls.append(("unwarped", planes))
            return image

        def applied(self):
            return []

        def intensity_corrected(self, results, data, by_plane=False):
            calls.append(("intensity_corrected", by_plane))
            return results

    plugin.corrections = Corrections()
    image = np.ones((2, 4, 4), np.float32)

    plugin._results(context, _band_unit(2.5), image)
    # A slab: its partitions lie on no calibrated slice.
    plugin._results(context, _band_unit(12.5), image)

    assert calls == [
        ("unwarped", [-2.5, 2.5]),
        ("intensity_corrected", True),
        ("unwarped", None),
        ("intensity_corrected", False),
    ]


def test_each_calibration_slice_of_a_stack_is_fitted_its_own_maps(monkeypatch):
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi

    calls = []

    def estimate(kspace, *args, **kwargs):
        calls.append(tuple(kspace.shape))
        return kspace.new_ones(kspace.shape)

    monkeypatch.setattr(epi, "nlinv_estimate", estimate)
    context = ReconContext.offline()
    plugin = epi.EpiRecon(
        gradient_unwarp=False, intensity_correction=False, kspace_filter=False
    )
    plugin.startup(context)
    plane = np.ones((2, 4, 4), np.complex64)
    units = [
        _calibration_unit(location, plane, position)
        for location, position in ((0, 10.0), (1, -10.0), (2, 0.0))
    ]
    for unit in units:
        unit.ref.mask = np.ones(plane.shape[1:], bool)
    plugin.calibrations = {0: units}

    plugin._calibrate(context)

    assert len(calls) == 3
    assert all(int(np.prod(shape[1:-2])) == 1 for shape in calls)
    assert sorted(context.coil_maps) == [0, 1, 2]
    assert len(plugin.slice_calibrations) == 3


@pytest.mark.parametrize("shape", [(2, 4, 4, 4), (2, 1, 4, 4), (2, 4, 4)])
def test_a_slab_or_slice_calibration_is_fitted_at_the_epi_sobolev_weight(
    monkeypatch, shape
):
    from pulserver.recon.handlers import epi

    given = {}
    monkeypatch.setattr(
        epi, "nlinv_estimate", lambda kspace, **options: given.update(options)
    )
    monkeypatch.setattr(epi, "EPI_SOBOLEV", 1.0)

    epi.epi_nlinv_estimate(np.ones(shape, np.complex64))

    assert given == {"sobolev": 1.0}


@pytest.mark.parametrize(
    ("wavelet", "partitions", "bands", "expected"),
    [
        (None, 2, True, "EPI_WAVELET_MULTIBAND"),
        (None, 1, False, "EPI_WAVELET_SLICE"),
        (0.0, 4, False, 0.0),
        (0.0, 2, True, 0.0),
    ],
)
def test_an_undersampled_unit_takes_the_wavelet_of_a_slab_multiband_unit_or_slice_unless_one_is_given(
    wavelet, partitions, bands, expected
):
    from pulserver.recon.handlers import epi

    plugin = epi.EpiRecon(wavelet)
    volume = np.zeros((2, partitions, 4, 4), np.complex64)

    want = getattr(epi, expected) if isinstance(expected, str) else expected
    assert plugin._wavelet(volume, bands) == want


@pytest.mark.parametrize(
    ("ry", "rz", "expected"),
    [
        (2, 2, "EPI_WAVELET_SLICE"),
        (4, 1, "EPI_WAVELET_SLICE"),
        (3, 2, "EPI_WAVELET_SLAB"),
        (2, 4, "EPI_WAVELET_SLAB"),
    ],
)
def test_a_slab_takes_the_slice_wavelet_up_to_fourfold_acceleration_and_the_slab_wavelet_beyond(
    ry, rz, expected
):
    from pulserver.recon.handlers import epi

    volume = np.zeros((2, 8, 24, 8), np.complex64)
    # A CAIPI lattice over a partial-Fourier extent: the lines before the
    # first read are not counted as skipped.
    for line in range(6, 24, ry):
        volume[:, ((line - 6) // ry) % rz :: rz, line] = 1.0

    assert epi.EpiRecon()._wavelet(volume) == getattr(epi, expected)


@pytest.mark.parametrize(
    ("step", "shots", "expected"),
    [
        (2, 1, "EPI_WAVELET_MULTIBAND"),
        (2, 2, "EPI_WAVELET_SLICE"),
        (2, 3, "EPI_WAVELET_SLICE"),
        (3, 1, "EPI_WAVELET_MULTIBAND"),
        (3, 2, "EPI_WAVELET_MULTIBAND"),
        (3, 3, "EPI_WAVELET_SLICE"),
    ],
)
def test_a_segmented_multiband_unit_takes_the_slice_wavelet_at_low_acceleration_per_shot(
    step, shots, expected
):
    from pulserver.recon.handlers import epi

    volume = np.zeros((2, 2, 24, 8), np.complex64)
    for line in range(0, 24, step):
        volume[:, (line // step) % 2, line] = 1.0

    assert epi.EpiRecon()._wavelet(volume, True, shots) == getattr(epi, expected)


def test_the_smooth_phase_has_unit_magnitude_and_keeps_a_slowly_varying_phase(device):
    from pulserver.recon.handlers.pics import smooth_phase

    shape = (2, 32, 32)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    slow = 0.5 * np.pi * (x / 16.0) ** 2 - 0.3 * np.pi * y / 16.0
    image = np.exp(-((x / 40.0) ** 2) - (y / 40.0) ** 2) * np.exp(1j * slow)

    phase = smooth_phase(torch.from_numpy(image.astype(np.complex64)).to(device))

    phase = phase.cpu().numpy()
    np.testing.assert_allclose(np.abs(phase), 1.0, atol=1e-5)
    inside = (np.abs(x) < 10) & (np.abs(y) < 10)
    assert np.abs(np.angle(phase * np.exp(-1j * slow)))[inside].max() < 0.05


def test_a_fast_image_phase_adds_little_to_the_error_of_the_wavelet_solve(device):
    from pulserver.recon.handlers.epi import EpiRecon

    shape = (1, 48, 48)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    magnitude = ((y / 18.0) ** 2 + (x / 15.0) ** 2 < 1).astype(np.float32)
    lobe = np.exp(-((x**2 + (y - 8) ** 2) / (2 * 6.0**2)))
    maps = _maps(shape, coils=12)
    sampled = torch.zeros(shape, dtype=torch.bool)
    sampled[:, ::3] = True
    plugin = EpiRecon(0.03)

    def error(cycles):
        image = (magnitude * np.exp(1j * cycles * np.pi * lobe)).astype(np.complex64)
        kspace = _kspace(image, maps) * sampled
        solved = plugin._solved(
            kspace, torch.from_numpy(maps), SimpleNamespace(device=device)
        )
        residual = np.abs(solved.cpu().numpy()) - magnitude
        return np.linalg.norm(residual) / np.linalg.norm(magnitude)

    assert error(6) < 1.25 * error(0)


def test_the_wavelet_solve_starts_from_the_conjugate_gradient_image(device):
    from pulserver.recon.handlers.epi import EpiRecon, cg_sense

    shape = (1, 48, 48)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    image = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1).astype(np.complex64)
    maps = _maps(shape, coils=12)
    sampled = torch.zeros(shape, dtype=torch.bool)
    sampled[:, 1::3] = True
    kspace = _kspace(image, maps) * sampled
    plugin = EpiRecon(1e-6, iterations=4)

    solved = plugin._solved(
        kspace, torch.from_numpy(maps), SimpleNamespace(device=device)
    )
    start = cg_sense(kspace, torch.from_numpy(maps), sampled, 4, device)

    error = np.linalg.norm(np.abs(solved.cpu().numpy()) - np.abs(start.numpy()))
    assert error / np.linalg.norm(np.abs(start.numpy())) < 0.05


def test_full_resolution_maps_of_a_threefold_undersampled_slice_follow_steep_coil_sensitivities(
    device,
):
    from pulserver.recon.handlers.epi import full_resolution_maps

    shape = (1, 48, 48)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    image = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1).astype(np.complex64)
    image += 0.5 * ((y / 8.0) ** 2 + ((x - 6) / 3.0) ** 2 < 1)
    angles = np.arange(8) * np.pi / 4
    true = np.stack(
        [
            1.0
            / ((y - 24 * np.cos(a)) ** 2 + (x - 20 * np.sin(a)) ** 2 + 9.0) ** 0.75
            * np.exp(1j * a)
            for a in angles
        ]
    ).astype(np.complex64)[:, 0]
    true /= np.sqrt((np.abs(true) ** 2).sum(axis=0, keepdims=True))
    kspace = _kspace(image[0], true)
    sampled = torch.zeros((48, 48), dtype=torch.bool)
    sampled[1::3] = True
    calibration = torch.zeros_like(kspace)
    calibration[:, 12:36] = kspace[:, 12:36]
    smooth = torch.from_numpy(_maps((1, 48, 48), coils=8)[:, 0])

    maps = full_resolution_maps(kspace * sampled, calibration, smooth, device).numpy()

    inside = np.abs(image[0]) > 0
    phase = np.exp(1j * np.angle((np.conj(maps) * true).sum(axis=0)))
    error = np.sqrt((np.abs(maps * phase - true) ** 2).sum(axis=0))
    assert np.median(error[inside]) < 0.05


def _steep_maps(shape=(48, 48), coils=8):
    y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    angles = np.arange(coils) * 2 * np.pi / coils
    maps = np.stack(
        [
            1.0
            / ((y - 24 * np.cos(a)) ** 2 + (x - 20 * np.sin(a)) ** 2 + 9.0) ** 0.75
            * np.exp(1j * a)
            for a in angles
        ]
    ).astype(np.complex64)
    return maps / np.sqrt((np.abs(maps) ** 2).sum(axis=0, keepdims=True))


def _missing(maps, true):
    """Fraction of each pixel's true sensitivity that ``maps`` do not hold, whatever their common phase."""
    overlap = np.abs((np.conj(maps) * true).sum(axis=0))
    return np.sqrt(np.clip(1.0 - overlap**2, 0.0, None))


def test_maps_drawn_from_fully_encoded_kspace_are_its_sensitivities_where_there_is_signal():
    from pulserver.recon.handlers.epi import fully_encoded_maps

    y, x = np.meshgrid(*(np.arange(48) - 24,) * 2, indexing="ij")
    image = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1) * np.exp(0.3j * np.sin(x / 3.0))
    true = _steep_maps()

    maps = fully_encoded_maps(_kspace(image.astype(np.complex64), true)).numpy()

    np.testing.assert_allclose((np.abs(maps) ** 2).sum(axis=0), 1.0, rtol=1e-5)
    assert _missing(maps, true)[np.abs(image) > 0].max() < 1e-3


def test_maps_drawn_from_fully_encoded_kspace_take_the_sensitivity_of_the_signal_nearby_where_there_is_none():
    import bartorch

    from pulserver.recon.handlers.epi import fully_encoded_maps

    y, x = np.meshgrid(*(np.arange(48) - 24,) * 2, indexing="ij")
    radius = np.sqrt((y / 15.0) ** 2 + (x / 12.0) ** 2)
    image = (radius < 1).astype(np.complex64)
    true = _maps((1, 48, 48), coils=8)[:, 0]
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((8, 48, 48)) + 1j * rng.standard_normal((8, 48, 48))
    kspace = _kspace(image, true) + 1e-3 * torch.from_numpy(noise.astype(np.complex64))

    maps = fully_encoded_maps(kspace).numpy()

    own = bartorch.fft(kspace, (-1, -2), inverse=True, unitary=True).reshape(8, 48, 48)
    own = (own / own.abs().square().sum(dim=0).sqrt()).numpy()
    beside = (radius > 1.05) & (radius < 1.3)
    assert np.median(_missing(own, true)[beside]) > 0.8
    assert np.median(_missing(maps, true)[beside]) < 0.4


def test_kspace_missing_a_phase_encode_line_is_not_fully_encoded():
    from pulserver.recon.handlers.epi import fully_encoded

    kspace = _kspace(_phantom()[0], _maps()[:, 0])
    partial = kspace.clone()
    partial[:, 3] = 0

    assert fully_encoded(kspace)
    assert not fully_encoded(partial)


def _reference_unit(unit_set):
    reference = SimpleNamespace(position=(0.0, 0.0, 4.0), slice_dir=(0.0, 0.0, 1.0))
    return SimpleNamespace(
        data=SimpleNamespace(reference=reference),
        counters={"slice": 0, "set": unit_set},
    )


def test_a_slice_takes_the_maps_drawn_from_its_fully_encoded_phase_reference_from_then_on(
    monkeypatch,
):
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi

    y, x = np.meshgrid(*(np.arange(48) - 24,) * 2, indexing="ij")
    image = ((y / 21.0) ** 2 + (x / 17.0) ** 2 < 1).astype(np.complex64)
    steep = _steep_maps()
    smooth = torch.from_numpy(_maps((1, 48, 48), coils=8))
    monkeypatch.setattr(epi, "coil_maps", lambda context, data, estimate: smooth)
    context = ReconContext.offline()
    plugin = epi.EpiRecon(
        gradient_unwarp=False, intensity_correction=False, kspace_filter=False
    )
    plugin.startup(context)
    reference = _kspace(image, steep)[:, None]
    imaging = torch.zeros_like(reference)
    imaging[:, :, 1::3] = reference[:, :, 1::3]
    inside = np.abs(image) > 0

    before = plugin._maps(context, _reference_unit(0), imaging)
    drawn = plugin._maps(context, _reference_unit(epi.FULL_K_REFERENCE_SET), reference)
    after = plugin._maps(context, _reference_unit(0), imaging)

    assert torch.equal(before, smooth)
    assert _missing(drawn[:, 0].numpy(), steep)[inside].max() < 1e-3
    assert after is drawn


def _segmented(phase=None, coils=8, shape=(1, 48, 48), shots=2):
    """A disc read in ``shots`` interleaved shots, shot ``s`` at ``exp(i phase[s])`` times the image; the image, maps, k-space and shot masks."""
    magnitude, _, x, _ = _gradient_echo(shape)
    image = (magnitude * np.exp(0.3j * x / shape[-1])).astype(np.complex64)
    maps = torch.from_numpy(_maps(shape, coils=coils))
    masks = torch.zeros((shots, *shape[:2]), dtype=torch.bool)
    for shot in range(shots):
        masks[shot, :, shot::shots] = True
    phase = np.zeros((shots, *shape)) if phase is None else phase
    kspace = sum(
        _kspace((image * np.exp(1j * phase[shot])).astype(np.complex64), maps.numpy())
        * masks[shot][..., None]
        for shot in range(shots)
    )
    return image, maps, kspace.to(torch.complex64), masks


def _shot_phase(shape=(1, 48, 48), constant=0.9):
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    second = constant + 0.8 * x / shape[-1] - 0.6 * y / shape[-2]
    return np.stack([np.zeros(shape), second]).astype(np.float32)


def test_the_phase_a_second_shot_adds_is_fitted_against_a_reference_image():
    from pulserver.recon.handlers.epi import fit_shot_phases, phase_basis

    phase = _shot_phase()
    image, maps, kspace, masks = _segmented(phase)
    reference = torch.from_numpy(image)
    start = torch.zeros((2, *image.shape))

    fitted = fit_shot_phases(
        kspace, maps, masks, reference, start, phase_basis(image.shape, 3)
    )

    inside = np.abs(image) > 0
    error = np.angle(np.exp(1j * (fitted.numpy() - phase)))[:, inside]
    assert np.abs(error).max() < 1e-3


def test_the_default_phase_basis_fits_a_local_shot_phase_that_degree_three_misses():
    """A blob of phase a fifth of the object wide: the frame-to-frame shot phase of breathing below the head."""
    from pulserver.recon.handlers.epi import (
        SHOT_PHASE_ORDER,
        fit_shot_phases,
        phase_basis,
    )

    shape = (1, 48, 48)
    _, y, x = np.meshgrid(*(np.arange(n) - n // 2 for n in shape), indexing="ij")
    blob = 0.4 * np.exp(-((x - 4) ** 2 + (y + 3) ** 2) / (2 * 5.0**2))
    phase = np.stack([np.zeros(shape), 0.5 + 0.6 * x / 48 + blob]).astype(np.float32)
    image, maps, kspace, masks = _segmented(phase)
    inside = np.abs(image) > 0

    def error(order):
        fitted = fit_shot_phases(
            kspace,
            maps,
            masks,
            torch.from_numpy(image),
            torch.zeros((2, *shape)),
            phase_basis(shape, order),
        )
        return np.abs(np.angle(np.exp(1j * (fitted.numpy() - phase)))[1][inside]).mean()

    assert error(SHOT_PHASE_ORDER) < 0.02
    assert error(3) > 0.03


def test_a_segmented_unit_is_unaliased_by_the_joint_solve_with_its_shot_phases(
    device,
):
    from pulserver.recon.handlers.epi import cg_sense

    phase = _shot_phase()
    image, maps, kspace, masks = _segmented(phase)
    sampled = masks.any(dim=0)[..., None].expand(kspace.shape[1:])
    shots = (masks, torch.exp(1j * torch.from_numpy(phase)).to(torch.complex64))

    joint = cg_sense(kspace, maps, sampled, 30, device, shots=shots)
    merged = cg_sense(kspace, maps, sampled, 30, device)

    assert _magnitude_error(joint, np.abs(image)) < 0.02
    assert _magnitude_error(merged, np.abs(image)) > 0.2


@pytest.mark.parametrize("reference", [True, False], ids=["reference", "search"])
def test_the_shot_phases_of_a_segmented_unit_are_fitted_with_or_without_a_reference(
    device, reference, caplog
):
    import logging

    from pulserver.recon.handlers.epi import EpiRecon

    phase = _shot_phase(constant=2.5)
    image, maps, kspace, masks = _segmented(phase)
    plugin = EpiRecon(0.0)
    held = torch.from_numpy(image) if reference else None

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        solved = plugin._solved(
            kspace, maps, SimpleNamespace(device=device), held, shots=masks
        )

    assert _magnitude_error(solved, np.abs(image)) < 0.03
    assert any(
        "shot phases of a unit of 2 shots kept" in r.getMessage()
        for r in caplog.records
    )


def test_every_template_is_tried_and_the_least_residual_kept(device, caplog):
    import logging

    from pulserver.recon.handlers.epi import EpiRecon

    phase = _shot_phase(constant=2.5)
    image, maps, kspace, masks = _segmented(phase)
    noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(0))
    templates = [
        ("noise", noise.to(torch.complex64)),
        ("the object", torch.from_numpy(image)),
    ]
    plugin = EpiRecon(0.0)

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        solved = plugin._solved(
            kspace,
            maps,
            SimpleNamespace(device=device),
            None,
            shots=masks,
            templates=templates,
        )

    messages = [r.getMessage() for r in caplog.records]
    assert _magnitude_error(solved, np.abs(image)) < 0.03
    assert any("fitted against noise:" in m for m in messages)
    assert any("fitted against the object:" in m for m in messages)
    assert not any("constant shot phases" in m for m in messages)


def test_the_shot_phases_are_removed_from_each_coil_of_a_fully_sampled_unit():
    from pulserver.recon.handlers.epi import shot_corrected

    phase = _shot_phase()
    image, maps, kspace, masks = _segmented(phase, shots=2)
    flat = _kspace(image, maps.numpy())

    corrected = shot_corrected(
        kspace, masks, torch.exp(1j * torch.from_numpy(phase)).to(torch.complex64)
    )

    error = (corrected - flat).norm() / flat.norm()
    assert float(error) < 1e-3
    assert float((kspace - flat).norm() / flat.norm()) > 0.2


def test_a_segmented_unit_whose_shots_agree_is_solved_as_one(device, caplog):
    import logging

    from pulserver.recon.handlers.epi import EpiRecon, sense_combined

    _, maps, kspace, masks = _segmented()
    plugin = EpiRecon(0.0)

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        solved = plugin._solved(
            kspace, maps, SimpleNamespace(device=device), None, shots=masks
        )

    np.testing.assert_allclose(
        solved.cpu().numpy(), sense_combined(kspace, maps).cpu().numpy()
    )
    assert not any("shot phases" in r.getMessage() for r in caplog.records)


def test_the_shot_masks_take_each_line_to_its_segment_and_a_single_segment_none():
    from pulserver.recon.handlers.epi import shot_masks

    def header(line, segment):
        return SimpleNamespace(
            idx=SimpleNamespace(kspace_encode_step_1=line, segment=segment)
        )

    def buffer(headers):
        return SimpleNamespace(
            axes=("coil", "phase_encode", "readout"),
            headers=headers,
            position=lambda h: (h.idx.kspace_encode_step_1,),
        )

    lines = [header(line, (line // 2) % 2) for line in range(0, 16, 2)]
    masks = shot_masks(buffer(lines), (1, 16, 16))

    assert masks.shape == (2, 1, 16)
    assert masks[0, 0].nonzero().flatten().tolist() == [0, 4, 8, 12]
    assert masks[1, 0].nonzero().flatten().tolist() == [2, 6, 10, 14]
    assert shot_masks(buffer([header(4, 1), header(8, 1)]), (1, 16, 16)) is None


def test_the_shot_phase_fit_is_the_same_whichever_number_of_basis_terms_it_carries_at_once(
    monkeypatch,
):
    from pulserver.recon.handlers import epi
    from pulserver.recon.handlers.epi import fit_shot_phases, phase_basis

    shape = (2, 32, 32)
    phase = _shot_phase(shape)
    image, maps, kspace, masks = _segmented(phase, shape=shape)
    reference = torch.from_numpy(image)
    start = torch.zeros((2, *shape))
    basis = phase_basis(shape, 3)

    fitted = {}
    for terms in (1, 7, basis.shape[0]):
        monkeypatch.setattr(epi, "SHOT_PHASE_TERMS_AT_ONCE", terms)
        fitted[terms] = fit_shot_phases(kspace, maps, masks, reference, start, basis)

    for terms in (1, 7):
        assert torch.allclose(fitted[terms], fitted[basis.shape[0]], atol=1e-5)
