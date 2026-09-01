"""The 3D wave-encoded reconstruction of the recon zoo.

Driven over the acquisitions the runtime would stream: the wave-free
autocalibration rectangle first, flagged calibration, then the wave-encoded
imaging train, each readout carrying the trajectory the corkscrew traced.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest
import torch

from pulserver.recon import ReconContext, WaveEncoding, WavePSF, pics
from pulserver.app import wave3D_recon

N_X = 24
N_Y = 20
N_Z = 12
COILS = 6
FOV_Y_MM = 200.0
FOV_Z_MM = 200.0

N_ACS_Y = 8
N_ACS_Z = 6
ACCELERATION = 2

CYCLES = 4
#: Peak deviation of the corkscrew, in voxels of the axis it spreads.
SPREAD = 3.0

#: Fewer CG iterations than the deployed default: a CPU test suite pays for
#: every one, and the assertions compare against ignoring the corkscrew, not
#: against convergence.
PLUGIN = wave3D_recon.Wave3DRecon(iterations=25, calibration_iterations=24)

AXIS_Y = (np.arange(N_Y) - N_Y // 2) * (FOV_Y_MM * 1e-3 / N_Y)
AXIS_Z = (np.arange(N_Z) - N_Z // 2) * (FOV_Z_MM * 1e-3 / N_Z)


# ----------------------------------------------------------------------
# The scan
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def deviation():
    """The corkscrew, in cycles per metre, sampled at the ADC.

    A sine on the phase axis and a cosine on the partition axis, a quarter
    period apart, scaled so each spreads a voxel at the edge of the field of
    view by ``SPREAD`` voxels of its own axis.
    """
    times = (np.arange(N_X) + 0.5) / N_X
    peak = np.array([SPREAD / (FOV_Y_MM * 1e-3), SPREAD / (FOV_Z_MM * 1e-3)])
    shape = np.stack(
        [
            0.5 * (1.0 - np.cos(2 * np.pi * CYCLES * times)),
            np.sin(2 * np.pi * CYCLES * times),
        ]
    )
    return (peak[:, None] * shape).astype(np.float32)


@pytest.fixture(scope="module")
def psf(deviation):
    return WavePSF(AXIS_Y, AXIS_Z)(WavePSF.phase_from_trajectory(deviation))


@pytest.fixture(scope="module")
def phantom():
    """An object that varies along all three axes."""
    x = np.linspace(-1.0, 1.0, N_X)
    y = np.linspace(-1.0, 1.0, N_Y)
    z = np.linspace(-1.0, 1.0, N_Z)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    body = 0.3 + np.where(gx**2 + gy**2 + (1.4 * gz) ** 2 < 0.85, 0.7, 0.0)
    body[6:12, 8:16, 3:8] += 0.6
    body[14:18, 4:10, 5:10] += 0.4
    return body.astype(np.float32)


@pytest.fixture(scope="module")
def coil_maps():
    """Elements on a ring around the two encoded axes.

    Constant along the readout and varying in ``y`` and ``z``, so both of the
    axes the corkscrew spreads are also axes the array can resolve.
    """
    y = np.linspace(-1.0, 1.0, N_Y)
    z = np.linspace(-1.0, 1.0, N_Z)
    gy, gz = np.meshgrid(y, z, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((gy - 1.5 * np.cos(a)) ** 2 + (gz - 1.5 * np.sin(a)) ** 2) / 2.5)
            * np.exp(1j * 1.5 * (np.cos(a) * gy + np.sin(a) * gz))
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return np.broadcast_to(maps[:, None], (COILS, N_X, N_Y, N_Z)).astype(np.complex64)


def _pairs():
    """``(line, partition)`` the rectangle holds, and the train plays."""
    acs_y = range(N_Y // 2 - N_ACS_Y // 2, N_Y // 2 + N_ACS_Y // 2)
    acs_z = range(N_Z // 2 - N_ACS_Z // 2, N_Z // 2 + N_ACS_Z // 2)
    rectangle = [(line, partition) for partition in acs_z for line in acs_y]
    train = [
        (line, partition)
        for partition in range(0, N_Z, ACCELERATION)
        for line in range(0, N_Y, ACCELERATION)
    ]
    return rectangle, train


def _encode(phantom, coil_maps, pairs, point_spread):
    """What the scanner measures, ``(coils, lines, samples)``."""
    physics = WaveEncoding(
        np.asarray(pairs),
        torch.as_tensor(coil_maps),
        point_spread,
    )
    image = torch.as_tensor(phantom.astype(np.complex64))[None, None]
    return physics.A(image)[0].numpy()


@pytest.fixture(scope="module")
def measured(phantom, coil_maps, psf):
    """The wave-encoded train and the wave-free rectangle of one object."""
    rectangle, train = _pairs()
    return {
        "rectangle": _encode(phantom, coil_maps, rectangle, torch.ones_like(psf)),
        "train": _encode(phantom, coil_maps, train, psf),
    }


def header():
    """The encoded space and the geometry the plugin scales the corkscrew by."""
    matrix = SimpleNamespace(matrixSize=SimpleNamespace(x=N_X, y=N_Y, z=N_Z))
    encoded = SimpleNamespace(
        matrixSize=SimpleNamespace(x=N_X, y=N_Y, z=N_Z),
        fieldOfView_mm=SimpleNamespace(x=FOV_Y_MM, y=FOV_Y_MM, z=FOV_Z_MM),
    )
    return SimpleNamespace(
        encoding=[SimpleNamespace(encodedSpace=encoded, reconSpace=matrix)],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


@pytest.fixture
def context():
    return ReconContext.offline(header())


def _trajectory(line, partition, deviation, wave):
    """Where the samples of one readout were taken, ``(samples, 3)``."""
    traj = np.zeros((N_X, 3), dtype=np.float32)
    traj[:, 0] = (np.arange(N_X) - N_X // 2) / (FOV_Y_MM * 1e-3)
    traj[:, 1] = (line - N_Y // 2) / (FOV_Y_MM * 1e-3)
    traj[:, 2] = (partition - N_Z // 2) / (FOV_Z_MM * 1e-3)
    if wave:
        traj[:, 1:] += deviation.T
    return traj


def acquisitions(measured, deviation, *, calibration_only=False):
    """The stream as the native objects the server sees."""
    rectangle, train = _pairs()
    stream = []
    passes = [("rectangle", rectangle, False)]
    if not calibration_only:
        passes.append(("train", train, True))
    for name, pairs, wave in passes:
        for index, (line, partition) in enumerate(pairs):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(N_X, COILS, 3)
            acquisition.data[:] = measured[name][:, index].astype(np.complex64)
            acquisition.traj[:] = _trajectory(line, partition, deviation, wave)
            acquisition.idx.kspace_encode_step_1 = int(line)
            acquisition.idx.kspace_encode_step_2 = int(partition)
            acquisition.idx.segment = int(wave)
            acquisition.center_sample = N_X // 2
            if not wave:
                acquisition.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
                if index == len(pairs) - 1:
                    acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
            elif index == len(pairs) - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_SEGMENT)
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            stream.append(acquisition)
    return stream


def bucket(measured, deviation, **kwargs):
    from pulserver.recon._server.application import _make_bucket

    return _make_bucket(acquisitions(measured, deviation, **kwargs), [])


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(*args, **kwargs):
    """The reconstructed volume, back on the ``(x, y, z)`` grid.

    A result leaves a plugin as ``(partition, readout, phase)``: the volume is
    cropped on the buffer's own axes and the two in-plane ones are swapped
    into the order MRD reads them.
    """
    return PLUGIN(*args, **kwargs)[0].data.transpose(1, 2, 0)


# ----------------------------------------------------------------------
# The corkscrew is read off the trajectory
# ----------------------------------------------------------------------


def test_the_phase_off_a_trajectory_is_the_phase_the_gradients_accrued():
    """Integrating the gradients and differencing the k they traced agree.

    The reconstruction never sees the waveform: it reads what was played off
    the trajectory the readout carries. That is the same quantity, which is
    what makes the waveform's exact shape free.
    """
    gamma = 42.5756e6
    raster = 4e-6
    times = np.arange(N_X) * raster
    gradients = np.stack(
        [
            8e-3 * np.sin(2 * np.pi * CYCLES * times / times[-1]),
            8e-3 * np.cos(2 * np.pi * CYCLES * times / times[-1]),
        ]
    )
    # k is the gradient's running integral, in cycles per metre.
    moment = np.concatenate(
        [
            np.zeros((2, 1)),
            np.cumsum(0.5 * (gradients[:, 1:] + gradients[:, :-1]) * raster, axis=-1),
        ],
        axis=-1,
    )
    trajectory = gamma * moment

    from_gradients = WavePSF.phase_from_gradients(gradients, raster, times)
    from_trajectory = WavePSF.phase_from_trajectory(trajectory)
    assert torch.allclose(from_trajectory, from_gradients, atol=1e-3)


def test_a_line_the_wave_never_moved_accrues_no_phase(deviation):
    """A wave-free readout is a constant k, and a constant is no corkscrew."""
    flat = _trajectory(3, 2, deviation, wave=False)[:, 1:].T
    assert np.abs(WavePSF.phase_from_trajectory(flat).numpy()).max() == 0.0


def test_the_corkscrew_the_plugin_builds_is_the_one_that_was_played(
    measured, deviation, psf, context
):
    """The point-spread function is reconstructed, not declared."""
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in acquisitions(measured, deviation)[:-1]:
        plugin.receive(acquisition, context)

    buffer = plugin.buffers[0]
    first = np.argwhere(buffer.select(contrast=0)[1].any(axis=-1))[0]
    recovered = WavePSF(AXIS_Y, AXIS_Z)(
        WavePSF.phase_from_trajectory(
            buffer.points(contrast=0)[1:3, first[0], first[1]]
        )
    )
    assert torch.allclose(recovered, psf, atol=1e-4)


# ----------------------------------------------------------------------
# The reconstruction
# ----------------------------------------------------------------------


def test_the_wave_solve_recovers_the_object(measured, deviation, phantom, context):
    image = reconstruct(bucket(measured, deviation), context)
    assert image.shape == (N_X, N_Y, N_Z)
    assert relative_error(image, phantom) < 0.25


def test_ignoring_the_corkscrew_loses_the_object(
    measured, deviation, phantom, coil_maps, psf, context
):
    """The same data, solved as if the readout had been Cartesian.

    What the corkscrew buys is not a better solve of the same problem but a
    different one: the aliasing it spreads is aliasing the array can separate,
    and dropping it from the operator puts it back.
    """
    _, train = _pairs()
    flat = WaveEncoding(
        np.asarray(train),
        torch.as_tensor(coil_maps),
        torch.ones_like(psf),
    )
    cartesian = pics(
        torch.as_tensor(measured["train"])[None],
        flat,
        regularization=1e-3,
        iterations=25,
    )
    cartesian = np.asarray(cartesian).reshape(N_X, N_Y, N_Z)

    image = reconstruct(bucket(measured, deviation), context)
    assert relative_error(image, phantom) < relative_error(cartesian, phantom)


def test_the_matrix_comes_from_the_header_not_from_the_last_line(
    measured, deviation, context
):
    """Under acceleration the highest acquired line is not the matrix size."""
    _, train = _pairs()
    assert max(line for line, _ in train) < N_Y - 1
    result = PLUGIN(bucket(measured, deviation), context)[0]
    assert result.data.shape == (N_Z, N_X, N_Y)


# ----------------------------------------------------------------------
# The lifecycle the runtime drives
# ----------------------------------------------------------------------


def test_the_rectangle_calibrates_and_produces_no_image(measured, deviation, context):
    """The wave-free pass closing its segment is a calibration, not an image."""
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    emitted = [
        plugin.receive(acquisition, context)
        for acquisition in acquisitions(measured, deviation, calibration_only=True)
    ]
    assert all(output is None for output in emitted)
    assert plugin.coil_maps is not None


def test_the_train_does_not_recalibrate_off_its_own_wave_encoded_lines(
    measured, deviation, context
):
    """The train overwrites the rectangle, so only the flag may calibrate.

    Both passes fill the same buffer positions and the train closes segments
    of its own. Were the calibration routed off a segment boundary, it would
    run a second time over data the corkscrew had already spread.
    """
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    stream = acquisitions(measured, deviation)
    for acquisition in stream:
        if acquisition.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
            break
        plugin.receive(acquisition, context)
        if plugin.coil_maps is not None and not hasattr(plugin, "calibrated"):
            plugin.calibrated = np.asarray(plugin.coil_maps).copy()

    assert np.array_equal(np.asarray(plugin.coil_maps), plugin.calibrated)


def test_a_scan_that_sent_no_calibration_says_so(measured, deviation, context):
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    with pytest.raises(RuntimeError, match="wave-free rectangle"):
        for acquisition in acquisitions(measured, deviation):
            if acquisition.isFlagSet(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                continue
            plugin.receive(acquisition, context)


# ----------------------------------------------------------------------
# The corkscrew the gradients played, not the one they were asked for
# ----------------------------------------------------------------------


#: Samples between the gradient and the acquisition. The trajectory is
#: computed from the waveform, so it does not carry this.
GRADIENT_DELAY = 0.4


@pytest.fixture(scope="module")
def played(deviation):
    """The corkscrew the readout actually traced: the nominal one, late."""
    times = np.arange(N_X)
    return np.stack(
        [
            np.interp(times - GRADIENT_DELAY, times, row, left=row[0], right=row[-1])
            for row in deviation
        ]
    ).astype(np.float32)


@pytest.fixture(scope="module")
def measured_late(phantom, coil_maps, psf, played):
    """One object, encoded by the late corkscrew, described by the nominal."""
    rectangle, train = _pairs()
    late = WavePSF(AXIS_Y, AXIS_Z)(WavePSF.phase_from_trajectory(played))
    return {
        "rectangle": _encode(phantom, coil_maps, rectangle, torch.ones_like(psf)),
        "train": _encode(phantom, coil_maps, train, late),
    }


def test_calibrating_the_corkscrew_recovers_what_the_trajectory_alone_cannot(
    measured_late, deviation, phantom, context
):
    """A delay between the gradient and the acquisition is invisible to the
    trajectory and visible in the image, so the image is what measures it."""
    settings = {"iterations": 25, "calibration_iterations": 24}
    trusted = wave3D_recon.Wave3DRecon(**settings)
    fitted = wave3D_recon.Wave3DRecon(**settings, calibrate_psf=True)

    scan = bucket(measured_late, deviation)
    from_trajectory = trusted(scan, context)[0].data.transpose(1, 2, 0)
    from_image = fitted(bucket(measured_late, deviation), context)[0].data.transpose(
        1, 2, 0
    )

    assert relative_error(from_image, phantom) < relative_error(
        from_trajectory, phantom
    )


def test_a_corkscrew_the_trajectory_already_describes_survives_being_fitted(
    measured, deviation, phantom, context
):
    """Calibration is a refinement, not a second chance: with nothing to
    correct it must not walk away from the corkscrew it started from."""
    fitted = wave3D_recon.Wave3DRecon(
        iterations=25, calibration_iterations=24, calibrate_psf=True
    )

    image = fitted(bucket(measured, deviation), context)[0].data.transpose(1, 2, 0)

    assert relative_error(image, phantom) < 0.25
