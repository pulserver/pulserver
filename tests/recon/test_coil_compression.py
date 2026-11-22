"""Compression of a unit's channels to virtual channels, against eigenvalues and images computed here."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import (
    acs_readouts,
    calibration_header,
    centred_kspace,
    coil_phantom,
    combine,
    play,
    readouts,
    relative_difference,
    stream_plugin,
)

from pulserver import recon
from pulserver.recon._calibration import _geometry

pytest.importorskip("bartorch")

COILS = 6
MATRIX = 32
ACS = 16

#: Compression to the rank of the coil data discards nothing, so the two
#: reconstructions differ by single-precision round-off.
ROUND_OFF = 1e-5

#: What the maps estimated from the data differ from the sensitivities by.
ESTIMATION = 2e-2


def estimate():
    from bartorch import apps

    return apps.nlinv_maps


def image_of(n_virtual=None):
    """A step that images a unit with maps from its calibration data, compressing it first where asked."""

    def step(context, data):
        if n_virtual is not None:
            data = recon.CoilCompression(n_virtual)(context, data)
        maps = recon.coil_maps(context, data, estimate=estimate()).cpu().numpy()
        return combine(data.data.kspace, maps)

    return step


def channel_covariance(kspace):
    flat = kspace.reshape(kspace.shape[0], -1)
    return flat @ flat.conj().T


def test_virtual_channels_are_uncorrelated_and_carry_the_leading_eigenvalues(device):
    """The compressed calibration k-space has the channel covariance diag(λ₁..λₙ) of the n leading eigenvalues of the original's."""
    _, _, kspace = coil_phantom(COILS, MATRIX)
    plugin = stream_plugin(
        lambda context, data: recon.CoilCompression(3)(context, data)
    )

    play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace, acs=ACS),
        device=device,
    )

    (unit,) = plugin.results
    centre = slice(MATRIX // 2 - ACS // 2, MATRIX // 2 + ACS // 2)
    eigenvalues = np.linalg.eigvalsh(channel_covariance(kspace[:, centre]))[::-1]
    covariance = channel_covariance(unit.ref.kspace)
    np.testing.assert_allclose(np.diag(covariance).real, eigenvalues[:3], rtol=1e-4)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    assert np.abs(off_diagonal).max() < 1e-5 * eigenvalues[0]


def test_a_unit_without_calibration_readouts_gives_its_basis_from_its_imaging_k_space(
    device,
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    plugin = stream_plugin(
        lambda context, data: recon.CoilCompression(3)(context, data)
    )

    play(
        plugin,
        calibration_header(COILS, MATRIX),
        readouts(kspace, last=("ACQ_LAST_IN_SLICE",)),
        device=device,
    )

    (unit,) = plugin.results
    eigenvalues = np.linalg.eigvalsh(channel_covariance(kspace))[::-1]
    covariance = channel_covariance(unit.data.kspace)
    np.testing.assert_allclose(np.diag(covariance).real, eigenvalues[:3], rtol=1e-4)
    assert unit.ref is None


def test_compression_leaves_the_unit_it_is_given_as_it_was(device):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    compressed = []

    def step(context, data):
        before = data.data.kspace.copy()
        compressed.append(recon.CoilCompression(3)(context, data))
        np.testing.assert_array_equal(data.data.kspace, before)
        return data

    plugin = stream_plugin(step)
    play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace),
        device=device,
    )

    (original,) = plugin.results
    (virtual,) = compressed
    assert original.data.kspace.shape[0] == COILS
    assert virtual.data.kspace.shape == (3, MATRIX, MATRIX)
    assert virtual.data.coils == 3
    assert virtual.counters is original.counters
    assert virtual.acquisitions is original.acquisitions


def _two_slices(by_slice, ids, first, second, *, device="cpu"):
    """Compress a unit of slice 0, one of slice 1 and slice 0 again; return the compressed units."""

    def step(context, data):
        compressed = recon.CoilCompression(3, by_slice=by_slice)(context, data)
        ids.append(context.coil_compression.id)
        return compressed

    plugin = stream_plugin(step)
    play(
        plugin,
        calibration_header(COILS, MATRIX, slices=2),
        acs_readouts(first, slice_index=0)
        + acs_readouts(second, slice_index=1)
        + acs_readouts(first, slice_index=0),
        device=device,
    )
    return plugin.results


def _on_channels(channels):
    """Calibration k-space of the phantom received by ``channels`` only."""
    _, _, kspace = coil_phantom(COILS, MATRIX)
    kept = np.zeros_like(kspace)
    kept[list(channels)] = kspace[list(channels)]
    return kept


def test_each_slice_location_keeps_a_basis_of_its_own(device):
    ids = []

    _two_slices(
        True, ids, _on_channels(range(3)), _on_channels([0, 4, 5]), device=device
    )

    assert ids[0] != ids[1]
    assert ids[2] == ids[0]


def test_a_slice_seen_by_channels_the_first_slice_does_not_see_keeps_its_signal(
    device,
):
    first, second = _on_channels(range(3)), _on_channels(range(3, COILS))

    _, kept, _ = _two_slices(True, [], first, second, device=device)

    np.testing.assert_allclose(
        np.linalg.norm(kept.data.kspace), np.linalg.norm(second), rtol=ROUND_OFF
    )


def test_one_basis_for_the_stream_is_that_of_the_first_unit(device):
    """A later unit whose channels lie in the span of the basis keeps its norm."""
    image, maps, kspace = coil_phantom(COILS, MATRIX)
    later = centred_kspace(maps * np.roll(image, 5, axis=0))
    ids = []

    _, second, _ = _two_slices(False, ids, kspace, later, device=device)

    assert ids[0] == ids[1] == ids[2]
    assert second.data.kspace.shape[0] == 3
    np.testing.assert_allclose(
        np.linalg.norm(second.data.kspace), np.linalg.norm(later), rtol=ROUND_OFF
    )


@pytest.mark.parametrize("n_virtual", [3, 4])
def test_compressed_data_and_maps_reconstruct_the_uncompressed_image(device, n_virtual):
    """SENSE over the virtual channels, with maps estimated from the compressed calibration data, gives the image of all the channels."""
    image, maps, kspace = coil_phantom(COILS, MATRIX)
    header = calibration_header(COILS, MATRIX)
    acquisitions = acs_readouts(kspace, acs=ACS)
    plain = stream_plugin(image_of())
    virtual = stream_plugin(image_of(n_virtual))

    play(plain, header, acquisitions, device=device)
    play(virtual, header, acquisitions, device=device)

    (full,), (compressed,) = plain.results, virtual.results
    sensitivity = np.sqrt((np.abs(maps) ** 2).sum(axis=0))
    truth = np.abs(image) * sensitivity
    assert relative_difference(np.abs(compressed), np.abs(full)) < ROUND_OFF
    assert relative_difference(np.abs(full), truth) < ESTIMATION
    assert relative_difference(np.abs(compressed), truth) < ESTIMATION


def test_maps_estimated_from_compressed_data_are_recorded_with_the_basis(device):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    plugin = stream_plugin(image_of(3))

    context = play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace),
        device=device,
    )

    stored = context.coil_maps[0]
    assert stored.basis == context.coil_compression.id
    assert stored.maps.shape == (3, MATRIX, MATRIX)
    assert stored.coils == tuple(f"H{index}" for index in range(COILS))


def publish_maps(n_virtual):
    """A calibration series' step: image with the maps, then leave them to the exam."""

    def step(context, data):
        image_of(n_virtual)(context, data)
        context.coil_sensitivities = context.coil_maps[0]

    return step


def request_maps(n_virtual):
    """A later series' step: ask for the maps of an unit that has no calibration readouts."""

    def step(context, data):
        if n_virtual is not None:
            data = recon.CoilCompression(n_virtual)(context, data)
        return recon.coil_maps(context, data, estimate=estimate())

    return step


def test_exam_maps_of_a_compressed_series_are_refused_by_an_uncompressed_one(
    tmp_path, device
):
    measured, asked = 3, None
    reason = r"basis differs: stored '\w+', this unit None"
    _, _, kspace = coil_phantom(COILS, MATRIX)
    header = calibration_header(COILS, MATRIX)
    play(
        stream_plugin(publish_maps(measured)),
        header,
        acs_readouts(kspace),
        exam=recon.ExamCache("exam-1", tmp_path),
        device=device,
    )
    imaging = stream_plugin(request_maps(asked))

    with pytest.raises(recon.MissingCalibration, match=reason):
        play(
            imaging,
            header,
            readouts(kspace, last=("ACQ_LAST_IN_SLICE",)),
            exam=recon.ExamCache("exam-1", tmp_path),
            device=device,
        )


def test_more_virtual_channels_than_the_unit_has_are_refused(device):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    with pytest.raises(ValueError, match="cannot keep 7 virtual channels of 6"):
        play(
            stream_plugin(
                lambda context, data: recon.CoilCompression(7)(context, data)
            ),
            calibration_header(COILS, MATRIX),
            acs_readouts(kspace),
            device=device,
        )


def test_a_stream_keeps_one_basis(device):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    def step(context, data):
        recon.CoilCompression(3)(context, data)
        recon.CoilCompression(4)(context, data)

    with pytest.raises(ValueError, match="keeps 3 virtual channels, not the 4"):
        play(
            stream_plugin(step),
            calibration_header(COILS, MATRIX),
            acs_readouts(kspace),
            device=device,
        )


@pytest.mark.parametrize("n_virtual", [0, -1])
def test_fewer_than_one_virtual_channel_is_refused(n_virtual):
    with pytest.raises(ValueError, match="at least 1"):
        recon.CoilCompression(n_virtual)


def test_uncompressed_exam_maps_are_projected_onto_the_basis_a_later_series_compresses_by(
    tmp_path, device
):
    """Projected maps times the object are the compressed coil images, as SENSE needs."""
    image, maps, kspace = coil_phantom(COILS, MATRIX)
    header = calibration_header(COILS, MATRIX)
    exam = recon.ExamCache("exam-1", tmp_path)
    exam[recon.COIL_SENSITIVITIES] = recon.CoilSensitivities(
        maps=maps,
        coils=tuple(f"H{index}" for index in range(COILS)),
        noise=None,
        basis=None,
        geometry=_geometry(header, capture_ref(header, kspace)),
        method="phantom",
        source="7",
    )

    def step(context, data):
        data = recon.CoilCompression(3)(context, data)
        return data, recon.coil_maps(context, data, estimate=estimate())

    plugin = stream_plugin(step)
    play(
        plugin,
        header,
        readouts(kspace, last=("ACQ_LAST_IN_SLICE",)),
        exam=exam,
        device=device,
    )
    ((data, projected),) = plugin.results

    coil_images = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(data.data.kspace, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1),
    )
    assert projected.shape == (3, MATRIX, MATRIX)
    assert relative_difference(projected.numpy() * image, coil_images) < 1e-4


def capture_ref(header, kspace):
    """The imaging k-space buffer a stream of ``kspace`` closes, for its geometry."""
    plugin = stream_plugin()
    play(plugin, header, readouts(kspace, last=("ACQ_LAST_IN_SLICE",)))
    (data,) = plugin.results
    return data.data


def streaming_context(coils=8):
    return recon.ReconContext(
        header=calibration_header(coils, MATRIX), exam=recon.ExamCache("exam")
    )


def started(gadget, context):
    gadget.startup(context)
    return gadget


def led_by_calibration(kspace, acs=ACS):
    """The readouts of ``kspace``, its central ``acs`` lines first and flagged calibration and imaging; the slice closes with the last."""
    lines = kspace.shape[1]
    centre = range(lines // 2 - acs // 2, lines // 2 + acs // 2)
    outside = [line for line in range(lines) if line not in centre]
    return readouts(
        kspace, "ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING", lines=centre
    ) + readouts(kspace, lines=outside, last=("ACQ_LAST_IN_SLICE",))


def test_calibration_readouts_are_held_then_given_on_projected_ahead_of_the_first_imaging_readout():
    _, _, kspace = coil_phantom(8, MATRIX)
    context = streaming_context()
    gadget = started(recon.CompressChannels(4), context)
    acquisitions = led_by_calibration(kspace)

    held = [gadget(a, np.asarray(a.data)) for a in acquisitions[:ACS]]
    given = gadget(acquisitions[ACS], np.asarray(acquisitions[ACS].data))

    assert held == [None] * ACS
    assert [acquisition for acquisition, _ in given] == acquisitions[: ACS + 1]
    basis = context.coil_compression
    virtual = basis.matrix[:4].conj()
    for acquisition, readout in given:
        np.testing.assert_allclose(
            readout, virtual @ np.asarray(acquisition.data), rtol=1e-5, atol=1e-6
        )
    later = acquisitions[ACS + 1]
    assert gadget(later, np.asarray(later.data)).shape == (4, MATRIX)


def test_the_basis_comes_from_the_calibration_readouts_of_the_first_contrast():
    """The first contrast's samples, projected, carry its three leading eigenvalues; noise of a later contrast takes no part."""
    _, _, kspace = coil_phantom(8, MATRIX)
    context = streaming_context()
    gadget = started(recon.CompressChannels(3), context)
    first = readouts(kspace, "ACQ_IS_PARALLEL_CALIBRATION", lines=range(8, 24))
    noise = np.random.default_rng(0).standard_normal(kspace.shape)
    later = readouts(
        100 * noise.astype(np.complex64),
        "ACQ_IS_PARALLEL_CALIBRATION",
        lines=range(8, 24),
    )
    for acquisition in later:
        acquisition.idx.contrast = 1
    for acquisition in [*first, *later]:
        gadget(acquisition, np.asarray(acquisition.data))
    gadget.release()

    samples = np.concatenate([np.asarray(a.data) for a in first], axis=1)
    eigenvalues = np.sort(np.linalg.eigvalsh(samples @ samples.conj().T))[::-1][:3]
    virtual = context.coil_compression.matrix[:3].conj() @ samples
    np.testing.assert_allclose(
        np.real(np.diag(virtual @ virtual.conj().T)), eigenvalues, rtol=1e-3
    )


def test_a_stream_led_by_imaging_readouts_is_held_to_its_kspace_centre_and_gives_the_basis():
    _, _, kspace = coil_phantom(8, MATRIX)
    context = streaming_context()
    gadget = started(recon.CompressChannels(4), context)
    lines = readouts(kspace, lines=range(MATRIX))

    given = [gadget(a, np.asarray(a.data)) for a in lines]

    centre = MATRIX // 2
    assert all(out is None for out in given[:centre])
    assert [acquisition for acquisition, _ in given[centre]] == lines[: centre + 1]
    assert all(data.shape == (4, MATRIX) for _, data in given[centre])
    assert all(out.shape == (4, MATRIX) for out in given[centre + 1 :])
    assert context.coil_compression.n_virtual == 4


@pytest.mark.parametrize(
    "flag",
    ["ACQ_IS_NAVIGATION_DATA", "ACQ_IS_RTFEEDBACK_DATA", "ACQ_IS_PHASECORR_DATA"],
)
def test_navigator_and_phase_reference_readouts_take_the_virtual_channels_in_their_place(
    flag,
):
    _, _, kspace = coil_phantom(8, MATRIX)
    gadget = started(recon.CompressChannels(4), streaming_context())
    (navigator,) = readouts(kspace, flag, lines=[0])
    calibration = readouts(kspace, "ACQ_IS_PARALLEL_CALIBRATION", lines=range(8, 24))
    (imaging,) = readouts(kspace, lines=[2])

    assert gadget(navigator, np.asarray(navigator.data)) is None
    for readout in calibration:
        assert gadget(readout, np.asarray(readout.data)) is None
    given = gadget(imaging, np.asarray(imaging.data))
    (later,) = readouts(kspace, flag, lines=[0])
    projected = gadget(later, np.asarray(later.data))

    assert [acquisition for acquisition, _ in given] == [
        navigator,
        *calibration,
        imaging,
    ]
    assert all(data.shape == (4, MATRIX) for _, data in given)
    assert projected.shape == (4, MATRIX)


def test_navigators_ahead_of_an_uncalibrated_stream_are_given_on_projected_at_its_centre():
    _, _, kspace = coil_phantom(8, MATRIX)
    gadget = started(recon.CompressChannels(4), streaming_context())
    (navigator,) = readouts(kspace, "ACQ_IS_NAVIGATION_DATA", lines=[0])
    lines = readouts(kspace, lines=range(MATRIX // 2 + 1))

    assert gadget(navigator, np.asarray(navigator.data)) is None
    *held, given = [gadget(a, np.asarray(a.data)) for a in lines]

    assert all(out is None for out in held)
    assert [acquisition for acquisition, _ in given] == [navigator, *lines]
    assert all(data.shape == (4, MATRIX) for _, data in given)


def test_readouts_held_when_the_stream_ends_reach_their_unit_projected(device):
    _, _, kspace = coil_phantom(8, MATRIX)
    plugin = stream_plugin(gadgets=[recon.CompressChannels(4)])
    calibration = readouts(
        kspace,
        "ACQ_IS_PARALLEL_CALIBRATION",
        lines=range(8, 24),
        last=("ACQ_LAST_IN_SLICE",),
    )

    context = play(plugin, calibration_header(8, MATRIX), calibration, device=device)

    (unit,) = plugin.results
    assert unit.ref.coils == 4
    assert len(unit.acquisitions) == 16
    assert context.coil_compression.n_virtual == 4


def test_a_unit_of_streamed_readouts_holds_the_virtual_channels_in_arrival_order(
    device,
):
    _, _, kspace = coil_phantom(8, MATRIX)
    plugin = stream_plugin(gadgets=[recon.CompressChannels(4)])
    acquisitions = led_by_calibration(kspace)

    play(plugin, calibration_header(8, MATRIX), acquisitions, device=device)

    (unit,) = plugin.results
    assert unit.data.coils == 4
    assert unit.acquisitions == acquisitions
    assert unit.data.mask.all()


def head_phantom(matrix=64, coils=16):
    """Channel images of a dim brain inside a bright scalp, the scalp seen mostly by the first two coils.

    Returns ``(images, brain, scalp)``: ``(coils, matrix, matrix)`` complex
    images and the boolean masks of the two tissues.
    """
    generator = np.random.default_rng(1)
    axis = (np.arange(matrix) - matrix // 2) / (matrix / 2)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x / 0.9, y)
    brain, scalp = radius < 0.7, (radius > 0.78) & (radius < 0.9)
    # Elements around the head, the first two right behind it.
    angles = np.concatenate(
        [[0.45 * np.pi, 0.55 * np.pi], np.linspace(0, 2 * np.pi, coils - 1)[1:]]
    )
    sensitivities = np.array(
        [
            np.exp(
                -((x - np.cos(a)) ** 2 + (y - np.sin(a)) ** 2) / (0.3 if c < 2 else 1.0)
            )
            * np.exp(1j * generator.uniform(-1, 1) * (x + 2 * y))
            for c, a in enumerate(angles)
        ]
    )
    images = sensitivities * (brain + 20.0 * scalp)
    return images.astype(np.complex64), brain, scalp


def scalp_to_brain(images, basis, brain, scalp):
    virtual = np.tensordot(basis.matrix[: basis.n_virtual].conj(), images, 1)
    energy = np.abs(virtual) ** 2
    return energy[:, scalp].sum() / energy[:, brain].sum()


def test_head_regions_keep_the_brain_deep_inside_the_outline_and_reject_its_outer_shell():
    from pulserver.recon import gadgets

    images, brain, scalp = head_phantom()
    rss = np.sqrt((np.abs(images) ** 2).sum(0))
    pixel = 0.24 / images.shape[-1]

    wanted, rejected = gadgets.head_regions(rss, (pixel, pixel))

    assert not (wanted & rejected).any()
    assert wanted[brain].mean() > 0.6 and not wanted[scalp].any()
    assert rejected[scalp].mean() > 0.9 and rejected[brain].mean() < 0.15


def test_region_optimized_channels_carry_less_of_the_scalp_than_the_strongest_channels():
    from pulserver.recon._calibration import basis_of, rovir_basis

    images, brain, scalp = head_phantom()
    strongest = basis_of(images.reshape(images.shape[0], -1), 3)
    optimized = rovir_basis(images, brain, scalp, 3, 6)

    assert scalp_to_brain(images, optimized, brain, scalp) < 0.5 * scalp_to_brain(
        images, strongest, brain, scalp
    )


def test_region_optimized_channels_are_orthonormal_so_whitened_noise_stays_white():
    from pulserver.recon._calibration import rovir_basis

    images, brain, scalp = head_phantom()
    virtual = rovir_basis(images, brain, scalp, 3, 6).matrix[:3]

    np.testing.assert_allclose(virtual @ virtual.conj().T, np.eye(3), atol=1e-5)


def test_a_scalp_rejecting_gadget_draws_its_basis_from_the_calibration_lines_it_grids():
    matrix = 64
    images, brain, scalp = head_phantom(matrix)
    kspace = centred_kspace(images)
    context = recon.ReconContext(
        header=calibration_header(16, matrix), exam=recon.ExamCache("exam")
    )
    gadget = started(recon.CompressChannels(3, reject_scalp=True), context)
    gadget.readout_grid = lambda acquisition, data: data
    for acquisition in led_by_calibration(kspace, acs=matrix // 2):
        gadget(acquisition, np.asarray(acquisition.data))
    gadget.release()

    plain = started(
        recon.CompressChannels(3),
        recon.ReconContext(
            header=calibration_header(16, matrix), exam=recon.ExamCache("exam")
        ),
    )
    for acquisition in led_by_calibration(kspace, acs=matrix // 2):
        plain(acquisition, np.asarray(acquisition.data))
    plain.release()

    assert scalp_to_brain(
        images, context.coil_compression, brain, scalp
    ) < 0.5 * scalp_to_brain(images, plain.context.coil_compression, brain, scalp)
