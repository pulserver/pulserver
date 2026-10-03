"""The NUFFT reconstruction of a unit: the shots it solves from and the image it makes of them.

A segment, such as a PROPELLER blade, restarts the line counter, so the
readouts of one unit are placed by segment and line and every one of them is a
shot of the same solve.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import recon
from pulserver.recon.handlers.nufft import NufftRecon

MATRIX, FOV_MM, COILS, SAMPLES, LINES = 16, 220.0, 2, 24, 4


class Solves(NufftRecon):
    """Keeps what each solve of a stream is given in the list the stream's copy shares, and returns zeros."""

    def __init__(self):
        super().__init__()
        self.solved = []

    def _solved(self, samples, points, device):
        self.solved.append((samples, points))
        return np.zeros((MATRIX, MATRIX), dtype=np.float32)


def header(*, segments=1, averages=1, depth=1, partitions=1, lines=LINES):
    """One non-Cartesian encoding space of ``MATRIX`` voxels in plane and ``depth`` along z, counting what it is given."""

    def limit(extent):
        return SimpleNamespace(minimum=0, maximum=extent - 1, center=0)

    space = SimpleNamespace(
        matrixSize=SimpleNamespace(x=MATRIX, y=MATRIX, z=depth),
        fieldOfView_mm=SimpleNamespace(x=FOV_MM, y=FOV_MM, z=5.0 * depth),
    )
    limits = SimpleNamespace(
        kspace_encoding_step_1=limit(lines),
        kspace_encoding_step_2=limit(partitions),
        segment=limit(segments),
        average=limit(averages),
    )
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=space,
                reconSpace=space,
                encodingLimits=limits,
                trajectory="other",
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def blades(segments, *, samples=SAMPLES, lines=LINES):
    """The trajectory ``(segments, lines, samples, 2)`` of blades turned by half a turn, in grid units.

    A readout runs along the blade's direction through its line, which lies
    one voxel of k-space from the next across it.
    """
    angles = np.pi * np.arange(segments) / segments
    along = (np.arange(samples) - samples // 2) * (MATRIX / samples)
    across = np.arange(lines) - lines // 2
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    normal = np.stack([-np.sin(angles), np.cos(angles)], axis=-1)
    return (
        along[None, None, :, None] * direction[:, None, None, :]
        + across[None, :, None, None] * normal[:, None, None, :]
    )


def noise(*shape, seed=0):
    generator = np.random.default_rng(seed)
    return (
        generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    ).astype(np.complex64)


def readout(data, points, *, segment, line, **counters):
    """A readout of ``data`` ``(coils, samples)`` along ``points`` ``(samples, 2)`` in grid units."""
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(data.shape[-1], COILS, 2)
    acquisition.data[:] = data
    acquisition.traj[:] = points / (1e-3 * FOV_MM)
    acquisition.idx.segment = segment
    acquisition.idx.kspace_encode_step_1 = line
    for name, value in counters.items():
        setattr(acquisition.idx, name, value)
    return acquisition


def stream(data, points, segments=None, **counters):
    """The readouts of ``data`` ``(segments, lines, coils, samples)`` segment by segment.

    Each segment's last line carries ``LAST_IN_SEGMENT``, and the last readout
    ``LAST_IN_SLICE`` as the enrichment of a sequence flags them. ``segments``
    selects the segments played.
    """
    played = range(len(data)) if segments is None else segments
    acquisitions = []
    for segment in played:
        for line in range(data.shape[1]):
            acquisition = readout(
                data[segment, line],
                points[segment, line],
                segment=segment,
                line=line,
                **counters,
            )
            if line == data.shape[1] - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            acquisitions.append(acquisition)
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    return acquisitions


def play(plugin, header, acquisitions):
    """The ``(index, image)`` of each unit the stream closes, ``None`` for the index of one closed at its end."""
    context = recon.ReconContext.offline(header)
    plugin = plugin.spawn()
    plugin.startup(context)
    closed = []
    for index, acquisition in enumerate(acquisitions):
        closed += [
            (index, output) for _, output in plugin.receive(acquisition, context)
        ]
    return closed + [(None, output) for _, output in plugin.flush(context)]


def test_the_readouts_of_every_segment_are_shots_of_one_solve_in_segment_order():
    segments = 3
    points = blades(segments)
    data = noise(segments, LINES, COILS, SAMPLES)
    plugin = Solves()

    play(plugin, header(segments=segments), stream(data, points))

    ((samples, trajectory),) = plugin.solved
    expected = data.reshape(segments * LINES, COILS, SAMPLES).transpose(1, 0, 2)
    np.testing.assert_array_equal(samples, expected)
    np.testing.assert_allclose(
        trajectory[..., :2], points.reshape(-1, SAMPLES, 2), atol=1e-5
    )
    assert not trajectory[..., 2].any()


def test_a_unit_closes_with_the_last_readout_of_its_last_segment():
    segments = 3
    plugin = Solves()

    closed = play(
        plugin,
        header(segments=segments),
        stream(noise(segments, LINES, COILS, SAMPLES), blades(segments)),
    )

    assert [index for index, _ in closed] == [segments * LINES - 1]


def test_a_shot_that_no_readout_was_placed_at_is_not_a_sample():
    segments = 3
    points = blades(segments)
    data = noise(segments, LINES, COILS, SAMPLES)
    plugin = Solves()

    play(plugin, header(segments=segments), stream(data, points, segments=[0, 2]))

    ((samples, trajectory),) = plugin.solved
    kept = [0, 2]
    np.testing.assert_array_equal(
        samples, data[kept].reshape(-1, COILS, SAMPLES).transpose(1, 0, 2)
    )
    np.testing.assert_allclose(
        trajectory[..., :2], points[kept].reshape(-1, SAMPLES, 2), atol=1e-5
    )


def test_the_zero_fill_before_a_readout_shorter_than_the_matrix_is_not_a_sample():
    samples_per_readout = MATRIX - 6
    points = blades(2, samples=samples_per_readout)
    data = noise(2, LINES, COILS, samples_per_readout)
    plugin = Solves()

    play(plugin, header(segments=2), stream(data, points))

    ((samples, trajectory),) = plugin.solved
    assert samples.shape == (COILS, 2 * LINES, samples_per_readout)
    assert trajectory.shape == (2 * LINES, samples_per_readout, 3)
    np.testing.assert_array_equal(
        samples, data.reshape(-1, COILS, samples_per_readout).transpose(1, 0, 2)
    )


def test_averages_are_summed_and_the_trajectory_of_the_first_stands_for_them():
    segments = 2
    points = blades(segments)
    first, second = (
        noise(segments, LINES, COILS, SAMPLES, seed=1),
        noise(segments, LINES, COILS, SAMPLES, seed=2),
    )
    plugin = Solves()

    play(
        plugin,
        header(segments=segments, averages=2),
        stream(first, points, average=0) + stream(second, points + 5.0, average=1),
    )

    ((samples, trajectory),) = plugin.solved
    expected = (first + second).reshape(-1, COILS, SAMPLES).transpose(1, 0, 2)
    np.testing.assert_allclose(samples, expected, rtol=1e-6)
    np.testing.assert_allclose(
        trajectory[..., :2], points.reshape(-1, SAMPLES, 2), atol=1e-5
    )


def test_a_stack_of_segments_is_solved_partition_by_partition_over_all_its_shots():
    segments, partitions = 2, 3
    points = blades(segments)
    # The same shots at every partition: the transform along them leaves them
    # at the centre partition alone.
    data = noise(segments, LINES, COILS, SAMPLES)
    acquisitions = []
    for partition in range(partitions):
        for segment, line in np.ndindex(segments, LINES):
            acquisition = readout(
                data[segment, line],
                points[segment, line],
                segment=segment,
                line=line,
                kspace_encode_step_2=partition,
            )
            acquisitions.append(acquisition)
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    plugin = Solves()

    closed = play(
        plugin,
        header(segments=segments, depth=partitions, partitions=partitions),
        acquisitions,
    )

    assert len(plugin.solved) == partitions
    expected = data.reshape(-1, COILS, SAMPLES).transpose(1, 0, 2)
    for index, (samples, trajectory) in enumerate(plugin.solved):
        assert samples.shape == (COILS, segments * LINES, SAMPLES)
        assert trajectory.shape == (segments * LINES, SAMPLES, 3)
        centre = index == partitions // 2
        np.testing.assert_allclose(samples, expected if centre else 0, atol=1e-6)
    (_, image) = closed[-1]
    assert image.data.shape == (partitions, MATRIX, MATRIX)


def test_slices_counted_as_partitions_by_the_header_are_not_a_stack():
    slices = 3
    points = blades(2)
    acquisitions = []
    for index in range(slices):
        data = noise(2, LINES, COILS, SAMPLES, seed=index)
        acquisitions += stream(data, points, slice=index)
    plugin = Solves()

    closed = play(plugin, header(segments=2, depth=slices), acquisitions)

    assert len(plugin.solved) == slices
    assert {samples.shape for samples, _ in plugin.solved} == {
        (COILS, 2 * LINES, SAMPLES)
    }
    assert [output.data.shape for _, output in closed] == [(MATRIX, MATRIX)] * slices


def phantom(points):
    """The samples ``(segments, lines, coils, samples)`` along ``points`` of two coils seeing three blobs.

    The blobs lie on a diagonal through the centre of the field of view, and
    the coils' sensitivities are smooth, one rising along each axis.
    """
    axis = np.arange(MATRIX) - MATRIX // 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    blob = np.zeros((MATRIX, MATRIX))
    for shift in (-3.0, 0.0, 3.0):
        blob += np.exp(-((x - shift) ** 2 + (y - shift) ** 2) / 4.0)
    maps = np.stack([1.0 + 0.04 * x, 1.0 + 0.04 * y])
    phase = np.exp(
        -2j
        * np.pi
        * (points[..., 0, None] * x.ravel() + points[..., 1, None] * y.ravel())
        / MATRIX
    )
    samples = np.einsum("slkv,cv->slck", phase, (maps * blob).reshape(COILS, -1))
    return blob, samples.astype(np.complex64)


def test_blades_labelled_by_segment_and_line_make_the_image_of_the_same_readouts_labelled_by_one_running_line():
    pytest.importorskip("bartorch")
    segments = 4
    points = blades(segments)
    _, data = phantom(points)
    labelled = stream(data, points)
    running = [
        readout(
            data[segment, line],
            points[segment, line],
            segment=0,
            line=segment * LINES + line,
        )
        for segment, line in np.ndindex(segments, LINES)
    ]
    running[-1].setFlag(ismrmrd.ACQ_LAST_IN_SLICE)

    ((_, segmented),) = play(NufftRecon(), header(segments=segments), labelled)
    ((_, serial),) = play(NufftRecon(), header(lines=segments * LINES), running)

    assert segmented.data.shape == serial.data.shape == (MATRIX, MATRIX)
    difference = np.linalg.norm(segmented.data - serial.data)
    assert difference < 1e-3 * np.linalg.norm(serial.data)


def test_the_image_of_blades_is_the_object_they_sample():
    pytest.importorskip("bartorch")
    segments = 4
    points = blades(segments)
    blob, data = phantom(points)

    ((_, result),) = play(NufftRecon(), header(segments=segments), stream(data, points))

    image = np.asarray(result.data, dtype=float)
    correlation = np.corrcoef(image.ravel(), blob.ravel())[0, 1]
    assert correlation > 0.9
