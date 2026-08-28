"""Prospective motion correction, from a navigator's planes to the wire.

Three things have to hold for a correction to mean anything on a scanner: the
pose a set of planes implies must be the pose that moved them, the correction
sent must be the change since the last one because the scanner accumulates
what it receives, and every readout must be answered because the transport
reads one reply for each it sends.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import affine_transform, gaussian_filter
from scipy.spatial.transform import Rotation

import pulserver.recon as recon
from pulserver.recon import ReconContext
from pulserver.app import pmc_recon

#: Where the three planes of a navigator lie, as the directions their images'
#: first and second axis run along: axial in x-y, sagittal in y-z, coronal in
#: x-z, with x left-right, y anterior-posterior and z inferior-superior.
AXES = (
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

#: Millimetres per pixel of the synthetic navigator planes.
SPACING = 5.0


def volume(size: int = 64) -> np.ndarray:
    """A textured ellipsoid, indexed ``[z, y, x]``.

    Registration measures whatever asymmetry a plane holds, so the object has
    to have some in every plane -- a smooth blob is rotationally ambiguous and
    a navigator of one carries no rotation at all.
    """
    grid = np.mgrid[0:size, 0:size, 0:size].astype(float)
    centre = (size - 1) / 2
    texture = gaussian_filter(np.random.default_rng(7).normal(size=grid[0].shape), 4.0)
    texture = (texture - texture.min()) / np.ptp(texture)
    radii = np.array([19.0, 17.0, 22.0])
    inside = np.sum(((grid - centre).T / radii).T ** 2, axis=0) <= 1.0
    return ((0.4 + texture) * inside).astype(np.float32)


#: Array indices run ``(z, y, x)`` and the pose is in ``(x, y, z)``.
FLIP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def moved(source: np.ndarray, rotation: np.ndarray, translation: np.ndarray):
    """``source`` after the object itself is rotated and translated."""
    centre = np.full(3, (source.shape[0] - 1) / 2)
    matrix = (FLIP @ Rotation.from_rotvec(rotation).as_matrix() @ FLIP.T).T
    offset = centre - matrix @ (centre + FLIP @ np.asarray(translation, float))
    return affine_transform(source, matrix, offset=offset, order=1)


def planes_of(source: np.ndarray) -> list[np.ndarray]:
    """The three central planes, each with its rows and columns along AXES."""
    middle = source.shape[0] // 2
    return [
        source[middle, :, :].T,
        source[:, :, middle].T,
        source[:, middle, :].T,
    ]


def measured(pose, axes):
    """What each plane would measure if the object were at ``pose``."""
    from pulserver.recon.motion import _project_onto_plane

    return [_project_onto_plane(pose, np.asarray(r), np.asarray(c)) for r, c in axes]


# %% the geometry


@pytest.mark.parametrize("seed", range(6))
def test_the_pose_is_the_one_every_plane_measured(seed):
    """The pose solved from the planes is the pose they were projected from.

    This is the whole of the geometry -- which plane sees which rotation, in
    which sense, and along which axes it sees the translation -- with no
    image, no registration and no approximation in the way, so a sign that is
    wrong anywhere is wrong here.
    """
    from pulserver.recon.motion import _plane_axes, _pose_from_planes

    rng = np.random.default_rng(seed)
    truth = recon.RigidMotionEstimate(
        parameters=np.concatenate(
            [rng.normal(size=3) * 0.05, rng.normal(size=3) * 5.0]
        ),
        center=np.zeros(3),
    )
    rows, columns = _plane_axes(AXES, len(AXES))
    solved = _pose_from_planes(measured(truth, AXES), rows, columns)
    assert np.allclose(solved.matrix, truth.matrix, atol=1e-9)


def test_planes_that_share_a_normal_leave_a_rotation_unmeasured():
    """Two parallel planes see the same rotation twice and the third never."""
    from pulserver.recon.motion import _plane_axes, _pose_from_planes

    parallel = (AXES[0], AXES[0], AXES[1])
    rows, columns = _plane_axes(parallel, len(parallel))
    pose = recon.RigidMotionEstimate(parameters=np.zeros(6), center=np.zeros(3))
    with pytest.raises(ValueError, match="do not span the rotation"):
        _pose_from_planes(measured(pose, parallel), rows, columns)


def test_the_first_navigator_is_the_reference_and_the_zero_pose():
    tracker = recon.NavigatorMotionTracker()
    planes = planes_of(volume(32))
    pose = tracker.track(planes, AXES, spacing=SPACING)
    assert np.allclose(pose.parameters, 0.0)
    assert tracker.reference is not None
    assert len(tracker.reference) == len(planes)


def test_a_navigator_of_a_different_shape_than_the_reference_is_refused():
    tracker = recon.NavigatorMotionTracker()
    planes = planes_of(volume(32))
    tracker.track(planes, AXES, spacing=SPACING)
    with pytest.raises(ValueError, match="reference"):
        tracker.track(planes[:2], AXES[:2], spacing=SPACING)


# %% the images


@pytest.mark.parametrize(
    ("axis", "angle"),
    [(0, 0.0), (2, 0.05)],
    ids=["translation", "rotation about z"],
)
def test_the_tracker_recovers_motion_it_can_see_in_the_planes(axis, angle):
    """Registered planes give back the pose that moved the object.

    The motion here stays close to in-plane, which is the regime a plane
    navigator measures: a plane can only report what it still cuts, so a
    large out-of-plane rotation is what bounds this, not the arithmetic.
    """
    source = volume()
    rotation = np.zeros(3)
    rotation[axis] = angle
    translation = np.array([1.5, -2.0, 1.0])

    tracker = recon.NavigatorMotionTracker(
        registration=recon.RigidRegistration(iterations=60, sampling_percentage=1.0),
        measurement_noise=(1e-5,) * 3 + (0.01,) * 3,
        initial_covariance=100.0,
    )
    tracker.track(planes_of(source), AXES, spacing=SPACING)
    pose = tracker.track(
        planes_of(moved(source, rotation, translation)), AXES, spacing=SPACING
    )

    recovered = Rotation.from_matrix(pose.matrix[:3, :3]).as_rotvec()
    assert np.rad2deg(np.linalg.norm(recovered - rotation)) < 1.0
    assert np.linalg.norm(pose.matrix[:3, 3] - translation * SPACING) < 1.5


# %% the real-time route


class Loopback:
    """A real-time connection whose stream is a list and whose peer is one.

    Satisfies what the RTP transport asks of a connection -- yield the
    readouts, take a payload -- without a socket, so the handler runs exactly
    as it runs on the scanner.
    """

    def __init__(self, acquisitions):
        self.acquisitions = list(acquisitions)
        self.sent = []

    def __iter__(self):
        return iter(self.acquisitions)

    def send(self, payload):
        self.sent.append(payload)


MATRIX = 48

#: A prescription whose logical axes are already the scanner's.
STRAIGHT = np.eye(3)


def spiral(matrix: int, samples: int = 6000) -> np.ndarray:
    """One Archimedean arm out to the matrix's Nyquist edge, normalised."""
    fraction = np.linspace(0.0, 1.0, samples)
    angle = np.pi * matrix * fraction
    return 0.5 * np.stack([fraction * np.cos(angle), fraction * np.sin(angle)], axis=-1)


def header(matrix: int = MATRIX):
    """A navigator-only header: one encoding space, isotropic in plane."""
    from types import SimpleNamespace

    extent = matrix * SPACING
    return SimpleNamespace(
        encoding=[
            SimpleNamespace(
                encodedSpace=SimpleNamespace(
                    matrixSize=SimpleNamespace(x=matrix, y=matrix, z=1)
                ),
                reconSpace=SimpleNamespace(
                    matrixSize=SimpleNamespace(x=matrix, y=matrix, z=1),
                    fieldOfView_mm=SimpleNamespace(x=extent, y=extent, z=SPACING),
                ),
            )
        ],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=1),
    )


def navigator(source, index, prescription=STRAIGHT):
    """One navigator as the readouts the scanner would have streamed."""
    import ismrmrd
    from mrinufft import get_operator

    arm = spiral(MATRIX)
    operator = get_operator("finufft")(arm, (MATRIX, MATRIX), n_coils=1)
    acquisitions = []
    for plane, (row, column) in zip(planes_of(source), AXES, strict=True):
        samples = operator.op(plane.astype(np.complex64))
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(arm.shape[0], 1, trajectory_dimensions=3)
        acquisition.data[:] = samples.reshape(1, -1)
        acquisition.traj[:] = np.outer(arm[:, 0], row) + np.outer(arm[:, 1], column)
        acquisition.acquisition_time_stamp = int(index * 40)
        acquisition.read_dir = tuple(float(v) for v in prescription[0])
        acquisition.phase_dir = tuple(float(v) for v in prescription[1])
        acquisition.slice_dir = tuple(float(v) for v in prescription[2])
        acquisitions.append(acquisition)
    return acquisitions


def accumulated(payloads):
    """What the scanner is holding after applying every payload it received.

    It adds each shift to the one it has and left-multiplies its rotation by
    each rotation, so this is the sequence's own arithmetic, written out.
    """
    shift, rotation = np.zeros(3), np.eye(3)
    for payload in payloads:
        shift = shift + np.asarray(payload.shift)
        rotation = np.asarray(payload.rotation).reshape(3, 3) @ rotation
    return shift, rotation


@pytest.fixture(scope="module")
def route():
    """One reference navigator and two of a head that moved once and held."""
    source = volume(MATRIX)
    rotation = np.array([0.0, 0.0, 0.04])
    translation = np.array([1.0, -1.5, 0.0])
    displaced = moved(source, rotation, translation)
    stream = navigator(source, 0) + navigator(displaced, 1) + navigator(displaced, 2)
    connection = Loopback(stream)
    pmc_recon.process(connection, "pmc_recon", header())
    return connection, rotation, translation


def test_every_readout_is_answered(route):
    """The transport reads one payload for each readout it sends."""
    connection, _, _ = route
    assert len(connection.sent) == len(connection.acquisitions)


def test_only_the_readout_that_completed_a_navigator_carries_a_correction(route):
    """A navigator is three readouts and one pose, so two replies are null."""
    connection, _, _ = route
    null = [
        index
        for index, payload in enumerate(connection.sent)
        if np.allclose(payload.shift, 0.0)
        and np.allclose(np.asarray(payload.rotation).reshape(3, 3), np.eye(3))
    ]
    # The first navigator is the reference, so all three of its replies are
    # null as well: it measures the zero pose, which is no change at all.
    assert null[:4] == [0, 1, 2, 3]
    assert 5 not in null


def test_the_correction_the_scanner_accumulates_is_the_pose_the_head_took(route):
    connection, rotation, translation = route
    shift, matrix = accumulated(connection.sent)
    recovered = Rotation.from_matrix(matrix).as_rotvec()
    assert np.rad2deg(np.linalg.norm(recovered - rotation)) < 1.5
    assert np.linalg.norm(shift - translation * SPACING * 1e-3) < 1.5e-3


def test_a_head_that_stopped_moving_stops_being_corrected(route):
    """The wire carries the change, so a pose already sent is not sent again.

    A payload repeating the pose would be applied twice, because the scanner
    adds what it receives to what it is already holding.
    """
    connection, _, _ = route
    first = accumulated(connection.sent[:6])[0]
    both = accumulated(connection.sent)[0]
    assert np.linalg.norm(both) < 1.4 * np.linalg.norm(first)


def test_no_navigator_leaves_the_scanner_uncorrected():
    """A stream that never completes a navigator is answered, in nulls."""
    source = volume(MATRIX)
    connection = Loopback(navigator(source, 0)[:2])
    pmc_recon.process(connection, "pmc_recon", header())
    assert len(connection.sent) == 2
    assert all(np.allclose(payload.shift, 0.0) for payload in connection.sent)


def test_the_correction_is_sent_on_the_scanner_axes_and_not_the_sequence_ones():
    """A prescription that is not the identity rotates the correction.

    The pose is measured on the trajectory's axes, which are the sequence's
    logical ones. The shift and the rotation on the wire are the scanner's, so
    an oblique prescription has to turn one into the other -- otherwise a head
    that nodded would be corrected sideways.
    """
    source = volume(MATRIX)
    displaced = moved(source, np.zeros(3), np.array([2.0, 0.0, 0.0]))
    prescription = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    corrections = []
    for rows in (STRAIGHT, prescription):
        connection = Loopback(
            navigator(source, 0, rows) + navigator(displaced, 1, rows)
        )
        pmc_recon.process(connection, "pmc_recon", header())
        corrections.append(accumulated(connection.sent)[0])

    logical, scanner = corrections
    assert np.linalg.norm(logical) > 5e-3
    # Two registrations of the same images agree to a small fraction of a
    # pixel rather than exactly: SimpleITK reduces its metric across threads.
    # A correction that had skipped the prescription would be a whole
    # centimetre away, not ten microns.
    assert np.allclose(scanner, prescription.T @ logical, atol=1e-5)


def test_calling_the_module_tracks_a_recorded_navigator_file(tmp_path):
    """The offline call is the same hooks over the same data, so it gives the
    same poses the stream gave."""
    import ismrmrd
    import ismrmrd.xsd

    source = volume(MATRIX)
    displaced = moved(source, np.zeros(3), np.array([1.0, -1.0, 0.0]))
    stream = navigator(source, 0) + navigator(displaced, 1)

    extent = MATRIX * SPACING
    space = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(x=MATRIX, y=MATRIX, z=1),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(x=extent, y=extent, z=SPACING),
    )
    document = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=63_750_000
        ),
        encoding=[
            ismrmrd.xsd.encodingType(
                encodedSpace=space,
                reconSpace=space,
                trajectory=ismrmrd.xsd.trajectoryType("spiral"),
                encodingLimits=ismrmrd.xsd.encodingLimitsType(),
            )
        ],
        acquisitionSystemInformation=ismrmrd.xsd.acquisitionSystemInformationType(
            receiverChannels=1
        ),
    )
    path = str(tmp_path / "navigators.h5")
    dataset = ismrmrd.Dataset(path, "dataset", create_if_needed=True)
    dataset.write_xml_header(document.toXML())
    for acquisition in stream:
        dataset.append_acquisition(acquisition)
    dataset.close()

    poses = pmc_recon(path)
    assert [pose.dimension for pose in poses] == [3, 3]
    assert np.allclose(poses[0].parameters, 0.0)

    connection = Loopback(stream)
    pmc_recon.process(connection, "pmc_recon", header())
    # The same registration run twice lands a fraction of a degree apart --
    # SimpleITK reduces its metric across threads -- so the two paths are the
    # same to well inside what a navigator can resolve, not to the bit.
    shift, rotation = accumulated(connection.sent)
    turned = Rotation.from_matrix(rotation).as_rotvec()
    offline = Rotation.from_matrix(poses[-1].matrix[:3, :3]).as_rotvec()
    assert np.linalg.norm(poses[-1].matrix[:3, 3] * 1e-3 - shift) < 5e-5
    assert np.rad2deg(np.linalg.norm(offline - turned)) < 0.2


# %% what the real-time path is allowed to claim


def test_the_real_time_path_claims_one_cpu_thread_and_no_accelerator():
    """A correction is computed beside the reconstruction of the images.

    It can claim neither the array nor the card, and it has a deadline: the
    residual recovery before the next inversion. Both halves of the work are
    asked for explicitly here rather than left to a default -- the gridding
    would otherwise plan on the GPU wherever there is one, and both the
    transform and the registration are faster on one thread at a navigator's
    size than on twenty, dispatching being the larger cost.
    """
    asked = {}
    original = pmc_recon.NonCartesian2D
    weights = pmc_recon.pipe_menon_dcf

    def record_operator(*args, **kwargs):
        asked["operator"] = kwargs
        return original(*args, **kwargs)

    def record_weights(*args, **kwargs):
        asked["weights"] = kwargs
        return weights(*args, **kwargs)

    pmc_recon.NonCartesian2D = record_operator
    pmc_recon.pipe_menon_dcf = record_weights
    try:
        connection = Loopback(navigator(volume(MATRIX), 0))
        pmc_recon.process(connection, "pmc_recon", header())
    finally:
        pmc_recon.NonCartesian2D = original
        pmc_recon.pipe_menon_dcf = weights

    assert asked["operator"]["backend"] == "finufft"
    assert asked["operator"]["nthreads"] == 1
    assert asked["weights"]["nthreads"] == 1


def test_the_registration_the_plugin_tracks_with_is_single_resolution_and_single_thread():
    plugin = pmc_recon.PLUGIN.spawn()
    plugin.startup(ReconContext.offline(header()))
    registration = plugin.tracker.registration
    assert registration.threads == 1
    assert registration.shrink_factors == (1,)


def test_a_registration_leaves_the_thread_count_it_found():
    """The count is ITK's, process-wide, and shared with everything else in
    the process -- so a registration sets it for its own call and puts it
    back."""
    sitk = pytest.importorskip("SimpleITK")
    before = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    planes = planes_of(volume(32))
    recon.RigidRegistration(iterations=2, threads=1).estimate(
        planes[0], planes[0], spacing=(SPACING, SPACING)
    )
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == before
