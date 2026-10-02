"""Coil sensitivities of a reconstruction unit: where they come from, and when stored maps serve another unit."""

from __future__ import annotations

import dataclasses
import re

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
from pulserver.recon._calibration import _geometry, coil_labels

COILS = 6
MATRIX = 32
ACS = 16
LAST = ("ACQ_LAST_IN_SLICE",)
CALIBRATION = "ACQ_IS_PARALLEL_CALIBRATION"

#: What the image combined with maps estimated from the calibration data
#: differs from the object times the coils' root sum of squares by.
ESTIMATION = 2e-2


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


class Estimate:
    """An estimate that records the k-space it is given and returns it scaled to unit peak."""

    def __init__(self):
        self.calls = []

    def __call__(self, kspace):
        self.calls.append(kspace)
        return kspace / kspace.abs().amax()


def maps_of(estimate, **options):
    """A step that asks for the maps of each unit and returns them."""
    return lambda context, data: recon.coil_maps(
        context, data, estimate=estimate, **options
    )


def leave_maps_to_the_exam(estimate):
    """A step that estimates the maps of each unit and leaves them to the exam."""

    def step(context, data):
        recon.coil_maps(context, data, estimate=estimate)
        context.coil_sensitivities = context.coil_maps[0]

    return step


def centre_lines(matrix=MATRIX, acs=ACS):
    return range(matrix // 2 - acs // 2, matrix // 2 + acs // 2)


def capture(header=None, acquisitions=None):
    """The context and the one unit a stream of ``acquisitions`` closes."""
    kspace = coil_phantom(COILS, MATRIX)[2]
    plugin = stream_plugin(lambda context, data: (context, data))
    play(
        plugin,
        header or calibration_header(COILS, MATRIX),
        acquisitions or readouts(kspace, last=LAST),
    )
    ((context, data),) = plugin.results
    return context, data


def stored_for(context, data, **changes):
    """Maps estimated by a calibration of ``data``'s coils and geometry, with ``changes``."""
    maps = recon.CoilSensitivities(
        maps=np.zeros((COILS, MATRIX, MATRIX), dtype=np.complex64),
        coils=coil_labels(context.header),
        noise=None,
        basis=None,
        geometry=_geometry(context.header, data.data),
        method="stub",
        source="3",
    )
    return dataclasses.replace(maps, **changes)


def test_a_reference_only_unit_stores_maps_for_its_slice(device):
    """Calibration readouts of a slice give maps the series keeps under its slice counter, estimated by bartorch on the grid of the unit's space."""
    pytest.importorskip("bartorch")
    from bartorch import apps

    image, maps, kspace = coil_phantom(COILS, MATRIX)
    units = []

    def calibrate(context, data):
        units.append(data)
        return recon.coil_maps(context, data, estimate=apps.nlinv_maps)

    plugin = stream_plugin(calibrate)
    context = play(
        plugin,
        calibration_header(COILS, MATRIX, slices=2),
        readouts(kspace, CALIBRATION, lines=centre_lines(), last=LAST, slice_index=1),
        device=device,
    )

    (unit,) = units
    assert unit.data is None
    assert list(context.coil_maps) == [1]
    stored = context.coil_maps[1]
    assert stored.maps.shape == (COILS, MATRIX, MATRIX)
    assert stored.maps.dtype == np.complex64
    assert stored.coils == tuple(f"H{index}" for index in range(COILS))
    assert (stored.noise, stored.basis) == (None, None)
    assert stored.geometry["frame_of_reference"] == "1.2.3"
    assert stored.geometry["matrix"] == (MATRIX, MATRIX)
    assert stored.geometry["grid"] == (MATRIX, MATRIX)
    assert (stored.method, stored.source) == ("nlinv_maps", "7")
    sensitivity = np.sqrt((np.abs(maps) ** 2).sum(axis=0))
    truth = np.abs(image) * sensitivity
    assert relative_difference(np.abs(combine(kspace, stored.maps)), truth) < ESTIMATION
    (returned,) = plugin.results
    assert returned.device.type == device
    np.testing.assert_array_equal(returned.cpu().numpy(), stored.maps)


def test_calibration_k_space_reaches_the_estimate_zero_filled_at_its_lines(
    device, torch
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    estimate = Estimate()
    lines = range(4, 4 + ACS)

    play(
        stream_plugin(maps_of(estimate)),
        calibration_header(COILS, MATRIX),
        readouts(kspace, CALIBRATION, lines=lines, last=LAST),
        device=device,
    )

    (given,) = estimate.calls
    assert given.shape == (COILS, MATRIX, MATRIX)
    assert given.dtype == torch.complex64
    assert given.device.type == device
    given = given.cpu().numpy()
    np.testing.assert_array_equal(given[:, lines.start : lines.stop], kspace[:, lines])
    assert not given[:, : lines.start].any()
    assert not given[:, lines.stop :].any()


def test_calibration_k_space_of_a_volume_is_zero_filled_onto_the_partition_grid(
    device, torch
):
    lines, partitions = 16, 8
    generator = np.random.default_rng(0)
    shape = (COILS, partitions, lines, lines)
    volume = (
        generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    ).astype(np.complex64)
    covered = range(partitions // 2 - 2, partitions // 2 + 2)
    estimate = Estimate()
    acquisitions = [
        acquisition
        for partition in covered
        for acquisition in readouts(
            volume[:, partition],
            CALIBRATION,
            lines=range(4, 12),
            partition=partition,
            last=LAST if partition == covered[-1] else (),
        )
    ]

    context = play(
        stream_plugin(maps_of(estimate)),
        calibration_header(COILS, lines, partitions=partitions),
        acquisitions,
        device=device,
    )

    expected = np.zeros_like(volume)
    expected[:, covered.start : covered.stop, 4:12] = volume[
        :, covered.start : covered.stop, 4:12
    ]
    (given,) = estimate.calls
    np.testing.assert_array_equal(given.cpu().numpy(), expected)
    stored = context.coil_maps[0]
    assert stored.maps.shape == shape
    assert stored.geometry["grid"] == (partitions, lines, lines)
    assert stored.geometry["matrix"] == (partitions, lines, lines)


def test_a_later_unit_without_calibration_readouts_is_given_the_maps_of_its_slice(
    device, torch
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    estimate = Estimate()
    plugin = stream_plugin(maps_of(estimate))

    play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace, repetition=0) + readouts(kspace, last=LAST, repetition=1),
        device=device,
    )

    first, later = plugin.results
    assert len(estimate.calls) == 1
    assert torch.equal(later, first)


def test_a_unit_with_calibration_readouts_replaces_the_maps_stored_for_its_slice(
    device, torch
):
    image, maps, kspace = coil_phantom(COILS, MATRIX)
    moved = centred_kspace(maps * np.roll(image, 5, axis=0))
    estimate = Estimate()
    plugin = stream_plugin(maps_of(estimate))

    context = play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace, repetition=0) + acs_readouts(moved, repetition=1),
        device=device,
    )

    first, second = plugin.results
    assert len(estimate.calls) == 2
    assert not torch.equal(first, second)
    np.testing.assert_array_equal(second.cpu().numpy(), context.coil_maps[0].maps)


def test_maps_handed_out_are_copies_of_the_stored_maps(device, torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    def zero_what_is_handed_out(context, data):
        maps = recon.coil_maps(context, data, estimate=Estimate())
        maps.zero_()
        return maps

    plugin = stream_plugin(zero_what_is_handed_out)
    context = play(
        plugin,
        calibration_header(COILS, MATRIX),
        acs_readouts(kspace, repetition=0) + readouts(kspace, last=LAST, repetition=1),
        device=device,
    )

    assert context.coil_maps[0].maps.any()
    assert plugin.results[0].abs().amax() == 0


def test_a_calibration_unit_of_another_encoding_space_gives_the_imaging_of_its_slice_its_maps(
    device, torch
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    estimate = Estimate()
    plugin = stream_plugin(maps_of(estimate))

    play(
        plugin,
        calibration_header(COILS, MATRIX, calibration=MATRIX),
        readouts(kspace, CALIBRATION, space=1, last=LAST)
        + readouts(kspace, space=0, last=LAST),
        device=device,
    )

    calibration, imaging = plugin.results
    assert len(estimate.calls) == 1
    assert torch.equal(imaging, calibration)


def test_imaging_without_maps_fails_naming_each_source(torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    plugin = stream_plugin(maps_of(Estimate()))

    with pytest.raises(recon.MissingCalibration) as failure:
        play(
            plugin,
            calibration_header(COILS, MATRIX, slices=2),
            readouts(kspace, last=LAST, slice_index=1),
        )

    assert str(failure.value) == (
        "no coil sensitivities for slice 1; sources consulted:\n"
        "  this unit: it holds no calibration readouts\n"
        "  this series, slice 1: no maps stored\n"
        "  this exam: no maps stored"
    )


def test_maps_are_none_where_they_are_not_required_and_no_source_has_them(torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    plugin = stream_plugin(maps_of(Estimate(), required=False))

    play(
        plugin,
        calibration_header(COILS, MATRIX),
        readouts(kspace, last=LAST),
    )

    assert plugin.results == [None]


def test_exam_maps_at_another_matrix_are_refused_not_resampled(tmp_path, torch):
    small = MATRIX // 2
    _, _, calibration = coil_phantom(COILS, small)
    _, _, imaging = coil_phantom(COILS, MATRIX)
    estimate = Estimate()

    play(
        stream_plugin(leave_maps_to_the_exam(estimate)),
        calibration_header(COILS, small, measurement="7"),
        readouts(calibration, CALIBRATION, last=LAST),
        exam=recon.ExamCache("exam-1", tmp_path),
    )
    exam = recon.ExamCache("exam-1", tmp_path)

    with pytest.raises(recon.MissingCalibration) as failure:
        play(
            stream_plugin(maps_of(estimate)),
            calibration_header(COILS, MATRIX, measurement="8"),
            readouts(imaging, last=LAST),
            exam=exam,
        )

    assert (
        f"this exam: geometry.matrix differs: stored ({small}, {small}), this unit "
        f"({MATRIX}, {MATRIX}) (estimated by Estimate from measurement 7)"
    ) in str(failure.value)
    assert len(estimate.calls) == 1, "the later series made maps of its own"
    assert exam[recon.COIL_SENSITIVITIES].maps.shape == (COILS, small, small)


@pytest.mark.parametrize(
    ("attribute", "component", "value"),
    [
        ("patient_table_position", "table_position", (0.0, 0.0, 10.0)),
        ("position", "position", (0.0, 0.0, 5.0)),
        ("read_dir", "read_dir", (1.0, 0.0, 0.0)),
        ("phase_dir", "phase_dir", (0.0, 1.0, 0.0)),
        ("slice_dir", "slice_dir", (0.0, 0.0, 1.0)),
    ],
)
def test_exam_maps_at_another_position_or_orientation_are_refused(
    tmp_path, torch, attribute, component, value
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    estimate = Estimate()
    play(
        stream_plugin(leave_maps_to_the_exam(estimate)),
        calibration_header(COILS, MATRIX),
        readouts(kspace, CALIBRATION, last=LAST),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    reason = (
        f"geometry.{component} differs: stored (0.0, 0.0, 0.0), this unit {value!r}"
    )
    with pytest.raises(recon.MissingCalibration, match=re.escape(reason)):
        play(
            stream_plugin(maps_of(estimate)),
            calibration_header(COILS, MATRIX),
            readouts(kspace, last=LAST, geometry={attribute: value}),
            exam=recon.ExamCache("exam-1", tmp_path),
        )


def test_a_value_in_the_exam_that_is_not_coil_sensitivities_is_rejected_naming_its_type(
    tmp_path, torch
):
    _, _, kspace = coil_phantom(COILS, MATRIX)
    exam = recon.ExamCache("exam-1", tmp_path)
    exam[recon.COIL_SENSITIVITIES] = [1.0, 2.0]

    with pytest.raises(
        recon.MissingCalibration,
        match="this exam: holds a list, not CoilSensitivities",
    ):
        play(
            stream_plugin(maps_of(Estimate())),
            calibration_header(COILS, MATRIX),
            readouts(kspace, last=LAST),
            exam=exam,
        )


def test_coil_labels_follow_the_coil_numbers_of_the_header():
    header = calibration_header(3, MATRIX)
    header.acquisitionSystemInformation.coilLabel.reverse()

    assert coil_labels(header) == ("H0", "H1", "H2")


def test_coils_without_labels_are_named_by_their_channel_index():
    assert coil_labels(calibration_header(3, MATRIX, labels=False)) == ("0", "1", "2")


def test_maps_serve_a_unit_acquired_as_they_were():
    context, data = capture()

    assert stored_for(context, data).incompatibility(data, context) is None


@pytest.mark.parametrize(
    ("field", "stored"),
    [
        ("coils", tuple(f"H{index}" for index in reversed(range(COILS)))),
        ("noise", "0123456789ab"),
        ("basis", "0123456789ab"),
        ("geometry.frame_of_reference", "9.9.9"),
        ("geometry.table_position", (0.0, 0.0, 10.0)),
        ("geometry.position", (0.0, 0.0, 5.0)),
        ("geometry.read_dir", (1.0, 0.0, 0.0)),
        ("geometry.phase_dir", (0.0, 1.0, 0.0)),
        ("geometry.slice_dir", (0.0, 0.0, 1.0)),
        ("geometry.fov", (0.2, 0.2)),
        ("geometry.matrix", (MATRIX // 2, MATRIX // 2)),
        ("geometry.grid", (MATRIX // 2, MATRIX // 2)),
    ],
)
def test_maps_that_differ_in_a_field_are_refused_naming_it_with_both_values(
    field, stored
):
    context, data = capture()
    wanted = {
        "coils": coil_labels(context.header),
        "noise": None,
        "basis": None,
        **{
            f"geometry.{name}": value
            for name, value in _geometry(context.header, data.data).items()
        },
    }[field]
    maps = stored_for(context, data)
    changes = (
        {"geometry": {**maps.geometry, field.removeprefix("geometry."): stored}}
        if field.startswith("geometry.")
        else {field: stored}
    )

    reason = dataclasses.replace(maps, **changes).incompatibility(data, context)

    assert reason == f"{field} differs: stored {stored!r}, this unit {wanted!r}"


def test_the_first_differing_field_is_the_one_named():
    context, data = capture()
    maps = stored_for(context, data)
    geometry = {**maps.geometry, "matrix": (8, 8)}

    names = dataclasses.replace(maps, coils=("x",), noise="n", basis="b")
    noise = dataclasses.replace(maps, noise="n", basis="b", geometry=geometry)
    basis = dataclasses.replace(maps, basis="b", geometry=geometry)

    assert names.incompatibility(data, context).startswith("coils differs")
    assert noise.incompatibility(data, context).startswith("noise differs")
    assert basis.incompatibility(data, context).startswith("basis differs")


def test_a_unit_holding_no_k_space_has_no_geometry_to_compare():
    context, data = capture()

    reason = stored_for(context, data).incompatibility(
        recon.ReconData("imaging"), context
    )

    assert reason == "the unit holds no k-space, so its geometry is not known"


def test_calibration_readouts_of_a_non_cartesian_space_are_refused(torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    with pytest.raises(ValueError, match="not Cartesian"):
        play(
            stream_plugin(maps_of(Estimate())),
            calibration_header(COILS, MATRIX, trajectory="radial"),
            readouts(kspace, CALIBRATION, last=LAST),
        )


def test_calibration_readouts_that_vary_along_another_axis_are_refused(torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    with pytest.raises(ValueError, match=r"varies along \['repetition'\]"):
        play(
            stream_plugin(maps_of(Estimate()), axes=("repetition",)),
            calibration_header(COILS, MATRIX, repetitions=2),
            readouts(kspace, CALIBRATION, repetition=0, last=LAST)
            + readouts(kspace, CALIBRATION, repetition=1, last=LAST),
        )


def test_an_estimate_returning_maps_on_another_grid_is_refused(torch):
    _, _, kspace = coil_phantom(COILS, MATRIX)

    with pytest.raises(ValueError, match="maps of shape"):
        play(
            stream_plugin(maps_of(lambda calibration: calibration[:1])),
            calibration_header(COILS, MATRIX),
            readouts(kspace, CALIBRATION, last=LAST),
        )
