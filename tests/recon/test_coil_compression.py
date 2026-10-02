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


def test_the_basis_of_the_first_unit_is_applied_to_the_later_ones(device):
    """A later unit whose channels lie in the span of the basis keeps its norm."""
    image, maps, kspace = coil_phantom(COILS, MATRIX)
    later = centred_kspace(maps * np.roll(image, 5, axis=0))
    ids = []

    def step(context, data):
        compressed = recon.CoilCompression(3)(context, data)
        ids.append(context.coil_compression.id)
        return compressed

    plugin = stream_plugin(step)
    play(
        plugin,
        calibration_header(COILS, MATRIX, slices=2),
        acs_readouts(kspace, slice_index=0) + acs_readouts(later, slice_index=1),
        device=device,
    )

    _, second = plugin.results
    assert ids[0] == ids[1]
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


@pytest.mark.parametrize(
    ("measured", "asked", "reason"),
    [
        (None, 3, r"basis differs: stored None, this unit '\w+'"),
        (3, None, r"basis differs: stored '\w+', this unit None"),
    ],
)
def test_exam_maps_are_refused_by_a_series_in_another_basis(
    tmp_path, device, measured, asked, reason
):
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
