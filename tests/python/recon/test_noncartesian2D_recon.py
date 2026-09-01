"""The non-Cartesian reconstruction of the recon zoo.

A radial stream from an analytic phantom, its samples computed by forward
NUFFT: fully sampled it must return the phantom through the direct branch,
undersampled it must go through the solve and beat the density-compensated
adjoint it would otherwise have been.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mrinufft")

from pulserver.recon import ReconContext
from pulserver.app import noncartesian2D_recon

N = 32
COILS = 4
NYQUIST = int(np.ceil(np.pi / 2 * N))


@pytest.fixture(scope="module")
def phantom():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = 0.3 + np.where((x / 0.9) ** 2 + (y / 0.9) ** 2 < 1.0, 0.7, 0.0)
    image[8:16, 6:20] += 0.5
    return image


@pytest.fixture(scope="module")
def coil_maps():
    axis = np.linspace(-1.0, 1.0, N)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    angles = np.linspace(0.0, 2 * np.pi, COILS, endpoint=False)
    maps = np.stack(
        [
            np.exp(-((x - 1.5 * np.cos(a)) ** 2 + (y - 1.5 * np.sin(a)) ** 2) / 3.0)
            for a in angles
        ]
    )
    maps /= np.sqrt(np.sum(np.abs(maps) ** 2, axis=0, keepdims=True))
    return maps.astype(np.complex64)


def spoke_trajectory(n_spokes):
    """Full spokes over half a turn, in MRI-NUFFT's [-0.5, 0.5) units."""
    radius = np.linspace(-0.5, 0.5, 2 * N, endpoint=False)
    angles = np.arange(n_spokes) * np.pi / n_spokes
    return np.stack(
        [
            np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=-1)
            for angle in angles
        ]
    ).astype(np.float32)


def sample(phantom, coil_maps, trajectory):
    """Forward-NUFFT the phantom onto the trajectory, per coil and view."""
    from mrinufft import get_operator

    operator = get_operator("finufft")(trajectory.reshape(-1, 2), (N, N), n_coils=COILS)
    data = operator.op(coil_maps * phantom)
    return data.reshape(COILS, trajectory.shape[0], trajectory.shape[1])


def header(n_spokes=NYQUIST):
    """What the scanner sends: spokes counted in ``kspace_encoding_step_1``,
    and the trajectory type that says the count is views and not a grid."""
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(
                    matrixSize=SimpleNamespace(x=2 * N, y=N, z=1)
                ),
                reconSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=N, y=N, z=1)),
                encodingLimits=SimpleNamespace(
                    kspace_encoding_step_1=SimpleNamespace(
                        minimum=0, maximum=n_spokes - 1, center=n_spokes // 2
                    )
                ),
                trajectory="radial",
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def relative_error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    image, reference = image / image.max(), reference / reference.max()
    return float(np.linalg.norm(image - reference) / np.linalg.norm(reference))


def reconstruct(kspace, trajectory, **kwargs):
    """Drive the plugin over the streamed views and return the magnitude."""
    from pulserver.recon._server.application import _make_bucket

    import ismrmrd

    acquisitions = []
    for view in range(trajectory.shape[0]):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(trajectory.shape[1], COILS, trajectory_dimensions=2)
        acquisition.data[:] = kspace[:, view]
        acquisition.traj[:] = trajectory[view]
        acquisition.idx.slice = 0
        # LIN carries the spoke: a radial plane has no phase-encode line, so
        # the counter a reconstruction grids by is the view.
        acquisition.idx.kspace_encode_step_1 = view
        if view == trajectory.shape[0] - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        acquisitions.append(acquisition)

    plugin = noncartesian2D_recon.NonCartesian2DRecon(**kwargs)
    results = plugin(
        _make_bucket(acquisitions, []),
        ReconContext.offline(header(trajectory.shape[0])),
    )
    return np.abs(np.asarray(results[0].data.T))


def test_a_fully_sampled_scan_is_the_phantom_through_the_direct_branch(
    phantom, coil_maps
):
    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)
    image = reconstruct(kspace, trajectory)
    # A 32x32 direct adjoint bottoms out near 0.2 relative error (DCF and
    # Gibbs at toy size); the claim under test is the branch, not convergence.
    assert relative_error(image, phantom) < 0.3


def test_an_undersampled_scan_goes_through_the_solve_and_beats_the_adjoint(
    phantom, coil_maps
):
    trajectory = spoke_trajectory(NYQUIST // 3)
    kspace = sample(phantom, coil_maps, trajectory)

    # At toy size the sixteen spokes only sample a calibration radius of
    # sixteen densely enough for NLINV to reproduce its own data, and the
    # sensitivities they yield leave part of the matrix unobserved -- a normal
    # operator with eigenvalues at zero, which a compressed kernel perturbs
    # below it. The solve has to damp past that to be a solve at all.
    solved = reconstruct(
        kspace,
        trajectory,
        mode="pics",
        iterations=25,
        calibration_width=16,
        regularization=0.3,
    )
    adjoint = reconstruct(kspace, trajectory, mode="direct")
    assert relative_error(solved, phantom) < relative_error(adjoint, phantom)


def test_the_plugin_reconstructs_a_streamed_measurement(phantom, coil_maps):
    from pulserver.recon._server.application import _make_bucket

    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)

    import ismrmrd

    acquisitions = []
    for view in range(trajectory.shape[0]):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(trajectory.shape[1], COILS, trajectory_dimensions=2)
        acquisition.data[:] = kspace[:, view]
        acquisition.traj[:] = trajectory[view]
        acquisition.idx.slice = 0
        acquisition.idx.kspace_encode_step_1 = view
        if view == trajectory.shape[0] - 1:
            acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        acquisitions.append(acquisition)

    results = noncartesian2D_recon.PLUGIN(
        _make_bucket(acquisitions, []),
        ReconContext.offline(header(trajectory.shape[0])),
    )
    assert len(results) == 1
    assert relative_error(results[0].data.T, phantom) < 0.3


# ----------------------------------------------------------------------
# Off-resonance, measured from the echoes and put in the operator
# ----------------------------------------------------------------------

#: Long enough for a 60 Hz offset to turn most of a cycle across it.
DWELL_S = 200e-6
ECHO_TIMES = (4e-3, 8e-3)


@pytest.fixture(scope="module")
def field():
    """A linear ramp of off-resonance across the object, in Hz."""
    axis = np.linspace(-1.0, 1.0, N)
    _, x = np.meshgrid(axis, axis, indexing="ij")
    return (60.0 * x).astype(np.float32)


def _multiecho(phantom, coil_maps, trajectory, field):
    """What the scanner measures at each echo, with the field acting.

    Off-resonance does two things and both are in here: between the echoes it
    turns the object, which is what a field map is read from, and along each
    readout it turns it again, which is what blurs.
    """
    from pulserver.recon import NonCartesian2D, OffResonance

    readout_time = (np.arange(trajectory.shape[1]) - trajectory.shape[1] // 2) * DWELL_S
    physics = NonCartesian2D(
        trajectory.reshape(-1, 2),
        (N, N),
        coil_maps=coil_maps,
        n_coils=COILS,
    )
    blurring = OffResonance(physics, field, np.tile(readout_time, trajectory.shape[0]))
    return [
        np.asarray(
            blurring.A(
                (phantom * np.exp(2j * np.pi * field * te)).astype(np.complex64)[None]
            )
        ).reshape(COILS, trajectory.shape[0], trajectory.shape[1])
        for te in ECHO_TIMES
    ]


def _multiecho_header(n_spokes):
    hdr = header(n_spokes)
    hdr.encoding[0].encodingLimits.contrast = SimpleNamespace(
        minimum=0, maximum=len(ECHO_TIMES) - 1, center=0
    )
    hdr.sequenceParameters = SimpleNamespace(TE=list(ECHO_TIMES))
    return hdr


def _reconstruct_multiecho(measured, trajectory, **kwargs):
    from pulserver.recon._server.application import _make_bucket

    import ismrmrd

    acquisitions = []
    for echo, kspace in enumerate(measured):
        for view in range(trajectory.shape[0]):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(trajectory.shape[1], COILS, trajectory_dimensions=2)
            acquisition.data[:] = kspace[:, view]
            acquisition.traj[:] = trajectory[view]
            acquisition.idx.slice = 0
            acquisition.idx.kspace_encode_step_1 = view
            acquisition.idx.contrast = echo
            acquisition.center_sample = trajectory.shape[1] // 2
            acquisition.sample_time_us = DWELL_S * 1e6
            if echo == len(measured) - 1 and view == trajectory.shape[0] - 1:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            acquisitions.append(acquisition)

    # Fewer iterations than the deployed default: the field pass solves every
    # echo before the corrected pass does, so a CPU suite pays for each one
    # twice, and the assertion is against ignoring the field, not convergence.
    settings = {"iterations": 8, "virtual_coils": 2, "calibration_width": 16}
    plugin = noncartesian2D_recon.NonCartesian2DRecon(**{**settings, **kwargs})
    return plugin(
        _make_bucket(acquisitions, []),
        ReconContext.offline(_multiecho_header(trajectory.shape[0])),
    )


def test_a_multiecho_scan_measures_its_own_field_and_unblurs_with_it(
    phantom, coil_maps, field
):
    """A long readout in an inhomogeneous field blurs, and what says how far
    off each voxel is, is how much its phase turned between the echoes."""
    trajectory = spoke_trajectory(NYQUIST)
    measured = _multiecho(phantom, coil_maps, trajectory, field)

    ignored = _reconstruct_multiecho(measured, trajectory, correct_off_resonance=False)
    modelled = _reconstruct_multiecho(measured, trajectory)

    assert len(ignored) == len(modelled) == len(ECHO_TIMES)
    assert [result.series_index for result in modelled] == [0, 1]
    for echo in range(len(ECHO_TIMES)):
        blurred = relative_error(ignored[echo].data.T, phantom)
        sharp = relative_error(modelled[echo].data.T, phantom)
        assert sharp < blurred, (echo, sharp, blurred)


def test_a_single_echo_scan_has_no_field_to_measure(phantom, coil_maps):
    """One echo states one phase, which is the object's own: asking for the
    correction anyway must reconstruct the scan, not refuse it."""
    trajectory = spoke_trajectory(NYQUIST)
    kspace = sample(phantom, coil_maps, trajectory)

    image = reconstruct(kspace, trajectory, correct_off_resonance=True)

    assert relative_error(image, phantom) < 0.3
