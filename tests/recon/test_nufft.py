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
from pulserver.recon.handlers.nufft import NufftRecon, _within

MATRIX, FOV_MM, COILS, SAMPLES, LINES = 16, 220.0, 2, 24, 4


class Solves(NufftRecon):
    """Keeps what each solve of a stream is given in the list the stream's copy shares, and returns zeros; Fermi filter off unless asked for."""

    def __init__(self, kspace_filter=False):
        super().__init__(kspace_filter=kspace_filter)
        self.solved = []

    def _solved(self, samples, points, device, volume=None, partitions=1):
        self.solved.append((samples, points))
        return np.zeros(volume or (MATRIX, MATRIX), dtype=np.float32)


def header(
    *, segments=1, averages=1, depth=1, partitions=1, lines=LINES, fov_z_mm=None
):
    """One non-Cartesian encoding space of ``MATRIX`` voxels in plane and ``depth`` along z, counting what it is given.

    The field of view along z is 5 mm a voxel unless ``fov_z_mm`` gives it.
    """

    def limit(extent):
        return SimpleNamespace(minimum=0, maximum=extent - 1, center=0)

    space = SimpleNamespace(
        matrixSize=SimpleNamespace(x=MATRIX, y=MATRIX, z=depth),
        fieldOfView_mm=SimpleNamespace(
            x=FOV_MM, y=FOV_MM, z=5.0 * depth if fov_z_mm is None else fov_z_mm
        ),
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
    """A readout of ``data`` ``(coils, samples)`` along ``points`` ``(samples, dimensions)`` in grid units."""
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(data.shape[-1], COILS, points.shape[-1])
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


def test_the_corners_of_blades_past_the_matrix_are_not_samples_of_the_solve():
    points = np.zeros((3 * LINES, SAMPLES, 3), dtype=np.float32)
    points[..., :2] = blades(3).reshape(-1, SAMPLES, 2)
    data = noise(COILS, 3 * LINES, SAMPLES)
    radius = np.linalg.norm(points[..., :2], axis=-1)

    samples, kept = _within(data, points, (MATRIX, MATRIX))

    inside = radius <= MATRIX / 2
    assert not inside.all()
    np.testing.assert_array_equal(samples, data[:, inside][:, None])
    np.testing.assert_array_equal(kept, points[inside][None])


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


def test_a_stack_of_segments_is_one_solve_its_partitions_the_kz_encodes_of_its_shots():
    segments, partitions = 2, 3
    points = blades(segments)
    data = noise(partitions, segments, LINES, COILS, SAMPLES)
    acquisitions = []
    for partition in range(partitions):
        for segment, line in np.ndindex(segments, LINES):
            acquisition = readout(
                data[partition, segment, line],
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

    ((samples, trajectory),) = plugin.solved
    shots = segments * LINES
    assert samples.shape == (COILS, partitions * shots, SAMPLES)
    assert trajectory.shape == (partitions * shots, SAMPLES, 3)
    expected = data.reshape(partitions * shots, COILS, SAMPLES).transpose(1, 0, 2)
    np.testing.assert_allclose(samples, expected, atol=1e-6)
    kz = trajectory[:, 0, 2].reshape(partitions, shots)
    np.testing.assert_array_equal(
        kz, np.broadcast_to(np.arange(partitions)[:, None] - partitions // 2, kz.shape)
    )
    in_plane = trajectory[..., :2].reshape(partitions, shots, SAMPLES, 2)
    for partition in range(partitions):
        np.testing.assert_allclose(
            in_plane[partition], points.reshape(shots, SAMPLES, 2), atol=1e-5
        )
    (_, image) = closed[-1]
    assert image.data.shape == (partitions, MATRIX, MATRIX)


def stack_phantom(partitions, points):
    """The samples ``(partitions, shots, coils, samples)`` of a stack of ``points`` ``(shots, samples, 2)`` through three blobs on its centre partition and one beside it, with the object."""
    axis = np.arange(MATRIX) - MATRIX // 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    obj = np.zeros((partitions, MATRIX, MATRIX))
    for shift in (-3.0, 0.0, 3.0):
        obj[partitions // 2] += np.exp(-((x - shift) ** 2 + (y - shift) ** 2) / 4.0)
    obj[partitions // 2 + 1] = np.exp(-(x**2 + (y + 4) ** 2) / 6.0)
    maps = np.stack([1.0 + 0.04 * x, 1.0 + 0.04 * y])
    z = np.arange(partitions) - partitions // 2
    kz = z
    phase_z = np.exp(-2j * np.pi * np.outer(kz, z) / partitions)
    phase = np.exp(
        -2j
        * np.pi
        * (points[..., 0, None] * x.ravel() + points[..., 1, None] * y.ravel())
        / MATRIX
    )
    coil_images = (maps[:, None] * obj[None]).reshape(COILS, partitions, -1)
    hybrid = np.einsum("kz,czv->ckv", phase_z, coil_images)
    samples = np.einsum("lsv,ckv->klcs", phase, hybrid)
    return obj, samples.astype(np.complex64)


def test_the_image_of_a_stack_of_blades_is_the_object_it_samples_partition_by_partition():
    pytest.importorskip("bartorch")
    partitions = 8
    points = blades(4).reshape(-1, SAMPLES, 2)
    obj, data = stack_phantom(partitions, points)
    acquisitions = []
    for partition in range(partitions):
        for line in range(points.shape[0]):
            acquisitions.append(
                readout(
                    data[partition, line],
                    points[line],
                    segment=0,
                    line=line,
                    kspace_encode_step_2=partition,
                )
            )
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    stack = header(depth=partitions, partitions=partitions, lines=points.shape[0])

    ((_, result),) = play(NufftRecon(kspace_filter=False), stack, acquisitions)

    image = np.asarray(result.data, dtype=float)
    assert image.shape == (partitions, MATRIX, MATRIX)
    for plane in (partitions // 2, partitions // 2 + 1):
        correlation = np.corrcoef(image[plane].ravel(), obj[plane].ravel())[0, 1]
        assert correlation > 0.9


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


def test_channels_beyond_the_virtual_coils_are_compressed_before_the_solve():
    pytest.importorskip("bartorch")
    segments = 4
    points = blades(segments)
    _, data = phantom(points)
    plugin = Solves()
    plugin.virtual_coils = 1

    play(plugin, header(segments=segments), stream(data, points))

    ((samples, _),) = plugin.solved
    assert samples.shape[0] == 1


def test_the_image_of_blades_compressed_to_one_virtual_coil_is_the_object_they_sample():
    pytest.importorskip("bartorch")
    segments = 4
    points = blades(segments)
    blob, data = phantom(points)

    ((_, result),) = play(
        NufftRecon(virtual_coils=1), header(segments=segments), stream(data, points)
    )

    image = np.asarray(result.data, dtype=float)
    assert np.corrcoef(image.ravel(), blob.ravel())[0, 1] > 0.9


def spokes(shots, views, *, samples=SAMPLES, reach=1.0):
    """The trajectory ``(shots, views, samples, 3)`` of spokes through the centre, in grid units.

    The views of a shot are every ``shots``-th point of one Fibonacci set over
    the sphere, so that each shot samples it evenly and the shots together
    cover it. A spoke runs ``reach`` of the way to the edge of the matrix.
    """
    total = shots * views
    height = 1.0 - 2.0 * (np.arange(total) + 0.5) / total
    turn = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(total)
    ring = np.sqrt(1.0 - height**2)
    directions = np.stack([ring * np.cos(turn), ring * np.sin(turn), height], axis=-1)
    along = (np.arange(samples) - samples // 2) * (reach * MATRIX / samples)
    points = along[None, :, None] * directions[:, None, :]
    return points.reshape(views, shots, samples, 3).swapaxes(0, 1)


def volume_header(*, shots, views):
    """One non-Cartesian encoding space of ``MATRIX`` voxels along each axis, its views counted as lines and its shots as segments."""
    return header(segments=shots, depth=MATRIX, lines=views, fov_z_mm=FOV_MM)


def test_a_trajectory_that_encodes_kz_is_one_volume_solved_from_all_its_shots():
    shots, views = 3, 5
    points = spokes(shots, views)
    data = noise(shots, views, COILS, SAMPLES)
    plugin = Solves()

    closed = play(plugin, volume_header(shots=shots, views=views), stream(data, points))

    ((samples, trajectory),) = plugin.solved
    np.testing.assert_array_equal(
        samples, data.reshape(-1, COILS, SAMPLES).transpose(1, 0, 2)
    )
    np.testing.assert_allclose(trajectory, points.reshape(-1, SAMPLES, 3), atol=1e-5)
    ((_, image),) = closed
    assert image.data.shape == (MATRIX, MATRIX, MATRIX)


def test_a_shot_of_a_volume_that_no_readout_was_placed_at_is_not_a_sample():
    shots, views = 3, 5
    points = spokes(shots, views)
    data = noise(shots, views, COILS, SAMPLES)
    plugin = Solves()

    play(
        plugin,
        volume_header(shots=shots, views=views),
        stream(data, points, segments=[0, 2]),
    )

    ((samples, trajectory),) = plugin.solved
    np.testing.assert_array_equal(
        samples, data[[0, 2]].reshape(-1, COILS, SAMPLES).transpose(1, 0, 2)
    )
    assert trajectory.shape == (2 * views, SAMPLES, 3)


def sphere(points):
    """The samples ``(shots, views, coils, samples)`` along ``points`` of two coils seeing three blobs in a volume.

    The blobs differ in their position along every axis, and the coils'
    sensitivities are smooth, one rising along each of two axes.
    """
    axis = np.arange(MATRIX) - MATRIX // 2
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    blob = np.zeros((MATRIX,) * 3)
    for (at_z, at_y, at_x), weight in (
        ((-3, 0, 2), 1.0),
        ((0, 3, 0), 2.0),
        ((3, -2, -3), 1.5),
    ):
        blob += weight * np.exp(
            -((z - at_z) ** 2 + (y - at_y) ** 2 + (x - at_x) ** 2) / 4.0
        )
    maps = np.stack([1.0 + 0.04 * x, 1.0 + 0.04 * y])
    phase = np.exp(
        -2j
        * np.pi
        * (
            points[..., 0, None] * x.ravel()
            + points[..., 1, None] * y.ravel()
            + points[..., 2, None] * z.ravel()
        )
        / MATRIX
    )
    samples = np.einsum("slkv,cv->slck", phase, (maps * blob).reshape(COILS, -1))
    return blob, samples.astype(np.complex64)


def test_the_volume_of_spokes_is_the_object_they_sample():
    pytest.importorskip("bartorch")
    shots, views = 4, 40
    points = spokes(shots, views)
    blob, data = sphere(points)

    ((_, result),) = play(
        NufftRecon(),
        volume_header(shots=shots, views=views),
        stream(data, points),
    )

    image = np.asarray(result.data, dtype=float)
    assert image.shape == blob.shape
    assert np.corrcoef(image.ravel(), blob.ravel())[0, 1] > 0.9


def test_the_volume_of_spokes_that_stop_short_of_the_matrix_edge_is_on_the_matrix():
    pytest.importorskip("bartorch")
    shots, views = 4, 40
    points = spokes(shots, views, reach=0.85)
    blob, data = sphere(points)

    ((_, result),) = play(
        NufftRecon(),
        volume_header(shots=shots, views=views),
        stream(data, points),
    )

    image = np.asarray(result.data, dtype=float)
    assert image.shape == blob.shape
    assert np.corrcoef(image.ravel(), blob.ravel())[0, 1] > 0.9


def test_the_maps_of_one_coil_over_a_volume_are_unity_on_its_matrix():
    torch = pytest.importorskip("torch")
    pytest.importorskip("bartorch")
    from pulserver.recon.handlers.pics import nlinv_estimate

    points = torch.from_numpy(spokes(2, 6).reshape(-1, SAMPLES, 3))
    samples = torch.from_numpy(noise(1, 12, SAMPLES))

    maps = nlinv_estimate(samples, points, (MATRIX,) * 3)

    assert maps.shape == (1, MATRIX, MATRIX, MATRIX)
    assert torch.equal(maps, torch.ones_like(maps))


def test_the_solve_keeps_its_samples_kernels_and_weights_on_the_host_and_the_trajectory_on_the_device(
    device, monkeypatch
):
    pytest.importorskip("bartorch")
    from bartorch import linop

    points = blades(4)
    _, data = phantom(points)
    samples = np.moveaxis(data, 2, 0).reshape(COILS, -1, SAMPLES)
    flat = np.zeros((*points.shape[:-1], 3))
    flat[..., :2] = points
    traj = flat.reshape(-1, SAMPLES, 3)
    build = linop.NoncartesianSense
    given = {}

    def recorded(sensitivities, shape, **kwargs):
        given.update(
            kernels=(sensitivities.device.type, kwargs["kernels"]),
            weights=kwargs["weights"].device.type,
            traj=kwargs["traj"].device.type,
        )
        return build(sensitivities, shape, **kwargs)

    monkeypatch.setattr(linop, "NoncartesianSense", recorded)
    image = NufftRecon()._solved(samples, traj, device, (MATRIX, MATRIX))

    assert given == {"kernels": ("cpu", True), "weights": "cpu", "traj": device}
    assert image.shape == (MATRIX, MATRIX)


def counting(monkeypatch, module, name):
    """Replace ``module.name`` by a wrapper counting its calls; return the list of calls."""
    calls = []
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)
    return calls


def test_the_maps_density_and_eigenvalue_of_a_slice_are_computed_once_for_all_its_repetitions(
    monkeypatch,
):
    bartorch = pytest.importorskip("bartorch")
    from bartorch import optim

    from pulserver.recon.handlers import nufft

    maps = counting(monkeypatch, nufft, "nlinv_estimate")
    density = counting(monkeypatch, bartorch, "estimate_density")
    eigenvalue = counting(monkeypatch, optim, "maxeigen")
    segments = 4
    points = blades(segments)
    blob, data = phantom(points)
    acquisitions = [
        *stream(data, points, repetition=0),
        *stream(data, points, repetition=1),
        *stream(data, points, repetition=2),
    ]

    images = play(NufftRecon(), header(segments=segments), acquisitions)

    assert len(images) == 3
    assert (len(maps), len(density), len(eigenvalue)) == (1, 1, 1)
    for _, result in images:
        image = np.asarray(result.data, dtype=float)
        assert np.corrcoef(image.ravel(), blob.ravel())[0, 1] > 0.9


def test_each_slice_is_fitted_maps_of_its_own_and_shares_the_density_of_its_trajectory(
    monkeypatch,
):
    bartorch = pytest.importorskip("bartorch")

    from pulserver.recon.handlers import nufft

    maps = counting(monkeypatch, nufft, "nlinv_estimate")
    density = counting(monkeypatch, bartorch, "estimate_density")
    segments = 4
    points = blades(segments)
    _, data = phantom(points)
    acquisitions = [
        *stream(data, points, slice=0),
        *stream(data, points, slice=1),
    ]

    images = play(NufftRecon(), header(segments=segments), acquisitions)

    assert len(images) == 2
    assert (len(maps), len(density)) == (2, 1)


def test_a_stacks_corners_are_kept_and_its_shots_stay_one_per_partition_where_blades_reach_past_the_matrix():
    partitions = 4
    shots = blades(2).reshape(-1, SAMPLES, 2)
    points = np.zeros((partitions, *shots.shape[:-1], 3), dtype=np.float32)
    points[..., :2] = shots
    points[..., 2] = (np.arange(partitions) - partitions // 2)[:, None, None]
    points = points.reshape(-1, SAMPLES, 3)
    data = noise(COILS, points.shape[0], SAMPLES)

    samples, kept = _within(data, points, (partitions, MATRIX, MATRIX), partitions)

    inside = np.square(shots / (MATRIX / 2)).sum(axis=-1) <= 1.0 + 1e-6
    assert not inside.all()
    assert kept.shape == (partitions, int(inside.sum()), 3)
    assert samples.shape == (COILS, partitions, int(inside.sum()))
    np.testing.assert_array_equal(
        kept[:, 0, 2], np.arange(partitions) - partitions // 2
    )
