"""The EPI reconstruction's coil combination, partial-Fourier completion and stack coil maps."""

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


def test_pocs_restores_the_missing_lines_of_a_partial_fourier_unit_and_keeps_its_phase(
    device,
):
    import bartorch

    from pulserver.recon.handlers.epi import pocs

    image = _phantom()
    kspace = bartorch.fft(torch.from_numpy(image), (-1, -2), unitary=True).reshape(
        image.shape
    )
    sampled = torch.zeros(1, 16, 16, dtype=torch.bool)
    sampled[:, 5:] = True
    zero_filled = bartorch.fft(
        kspace * sampled, (-1, -2), inverse=True, unitary=True
    ).reshape(image.shape)

    completed = pocs(zero_filled.to(device), sampled.to(device), 20).cpu().numpy()

    before = np.linalg.norm(zero_filled.numpy() - image)
    after = np.linalg.norm(completed - image)
    assert after < 0.3 * before
    inside = np.abs(image) > 0
    phase_error = np.angle(completed[inside] * np.conj(image[inside]))
    assert np.abs(phase_error).max() < 0.1


def test_a_unit_without_a_partial_fourier_axis_passes_pocs_unchanged():
    from pulserver.recon.handlers.epi import pocs

    image = torch.from_numpy(_phantom())
    sampled = torch.zeros(1, 16, 16, dtype=torch.bool)
    sampled[:, ::2] = True

    assert pocs(image, sampled, 8) is image


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


def _multiband(device, coils=8, grid=(24, 24)):
    """Two bands' single-slice k-space and their blipped-CAIPI partition volume."""
    import bartorch

    image = _phantom((1, *grid))[0]
    shifted = np.roll(image, 5, axis=-1) * 0.7
    maps = _maps((1, *grid), coils)[:, 0]
    other = np.roll(maps, coils // 2, axis=0) * np.exp(
        1.1j * np.arange(grid[1]) / grid[1]
    )
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


def test_split_slice_grappa_returns_each_caipi_band_of_a_multiband_unit_as_its_own_kspace(
    device,
):
    from pulserver.recon.handlers.epi import slice_grappa_bands

    singles, volume = _multiband(device)

    bands = slice_grappa_bands(volume, singles)

    inner = (slice(None), slice(None), slice(2, -2), slice(3, -3))
    expected = singles.transpose(0, 1)
    error = torch.linalg.vector_norm(
        bands[inner] - expected[inner]
    ) / torch.linalg.vector_norm(expected[inner])
    assert error < 0.1


def test_split_slice_grappa_declines_a_unit_with_fewer_calibrated_lines_than_the_kernel():
    from pulserver.recon.handlers.epi import slice_grappa_bands

    singles, volume = _multiband("cpu")
    training = torch.zeros_like(singles)
    training[:, :, 10:12] = singles[:, :, 10:12]

    assert slice_grappa_bands(volume, training) is None


def _band_unit(position, partitions=2):
    reference = SimpleNamespace(
        position=(0.0, 0.0, position), slice_dir=(0.0, 0.0, 1.0)
    )
    space = SimpleNamespace(recon_fov=(0.005 * partitions, 0.2, 0.2))
    buffer = SimpleNamespace(reference=reference, space=space)
    return SimpleNamespace(data=buffer, counters={"set": 1})


def test_the_band_calibration_stacks_the_slices_at_each_band_position_preferring_the_unit_set():
    from pulserver.recon.handlers.epi import band_calibration

    plane = np.ones((3, 4, 4), np.complex64)
    stored = [
        (0, -2.5, plane * 1),
        (0, 2.5, plane * 2),
        (1, 2.5, plane * 3),
    ]

    training = band_calibration(stored, _band_unit(2.5), (3, 2, 4, 4))

    assert training.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(training[:, 0, 0, 0], [1, 3])


def test_the_band_calibration_is_absent_when_a_band_slice_was_not_calibrated():
    from pulserver.recon.handlers.epi import band_calibration

    stored = [(0, -2.5, np.ones((3, 4, 4), np.complex64))]

    assert band_calibration(stored, _band_unit(2.5), (3, 2, 4, 4)) is None


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


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((2, 4, 4, 4), {"sobolev": 1.0}), ((2, 1, 4, 4), {}), ((2, 4, 4), {})],
)
def test_a_slab_calibration_is_fitted_at_the_epi_sobolev_weight_and_a_slice_at_bart_s(
    monkeypatch, shape, expected
):
    from pulserver.recon.handlers import epi

    given = {}
    monkeypatch.setattr(
        epi, "nlinv_estimate", lambda kspace, **options: given.update(options)
    )
    monkeypatch.setattr(epi, "EPI_SOBOLEV", 1.0)

    epi.epi_nlinv_estimate(np.ones(shape, np.complex64))

    assert given == expected
