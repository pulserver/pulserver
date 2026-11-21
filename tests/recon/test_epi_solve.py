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
