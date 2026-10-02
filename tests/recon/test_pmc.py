"""Pose from three-plane navigators of Gaussian blobs, whose k-space is closed-form."""

import numpy as np
import pytest

pytest.importorskip("bartorch")
pytest.importorskip("SimpleITK")
ismrmrd = pytest.importorskip("ismrmrd")

from pulserver.proxy._motion import pose_of  # noqa: E402
from pulserver.recon import ReconContext  # noqa: E402
from pulserver.recon.handlers.pmc import PmcRecon  # noqa: E402

RNG = np.random.default_rng(0)
CENTRES = RNG.uniform(-0.05, 0.05, (6, 3))
AMPLITUDES = RNG.uniform(0.5, 1.5, 6)
WIDTH = 0.012
SAMPLES = 4000

#: A 16-turn spiral reaching 50 /m: a 32 matrix over 0.32 m.
_t = np.linspace(0.0, 1.0, SAMPLES)
SPIRAL = (
    50.0
    * _t[:, None]
    * np.stack([np.cos(32 * np.pi * _t), np.sin(32 * np.pi * _t)], axis=1)
)
#: Axial, sagittal and coronal, as the physical axes each spiral spans.
PLANES = ((0, 1), (2, 1), (0, 2))


def _navigator(rotation):
    centres = CENTRES @ rotation.T
    readouts = []
    for a, b in PLANES:
        k = np.zeros((SAMPLES, 3))
        k[:, [a, b]] = SPIRAL
        # Through-plane k is zero: the slab sees the object's projection.
        signal = (
            AMPLITUDES
            * np.exp(-2 * np.pi**2 * WIDTH**2 * np.sum(k**2, axis=1))[:, None]
            * np.exp(-2j * np.pi * k @ centres.T)
        ).sum(axis=1)
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(SAMPLES, 2, 3)
        acquisition.data[:] = np.stack([signal, (0.5 + 0.2j) * signal])
        acquisition.traj[:] = k
        acquisition.set_flag(ismrmrd.ACQ_IS_NAVIGATION_DATA)
        readouts.append(acquisition)
    return readouts


def _about_z(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _poses(rotations):
    plugin = PmcRecon().spawn()
    context = ReconContext.offline()
    plugin.startup(context)
    poses = []
    for rotation in rotations:
        emitted = [plugin.receive(a, context) for a in _navigator(rotation)]
        assert emitted[:2] == [[], []]
        ((_, waveform),) = emitted[2]
        poses.append(pose_of(waveform))
    return poses


def test_the_first_navigator_states_the_identity_and_a_turned_object_its_rotation():
    first, turned = _poses([np.eye(3), _about_z(8.0)])

    np.testing.assert_allclose(np.reshape(first.rotation, (3, 3)), np.eye(3))
    np.testing.assert_allclose(
        np.reshape(turned.rotation, (3, 3)), _about_z(8.0), atol=0.05
    )
    np.testing.assert_allclose(turned.translation_m, 0.0, atol=2e-3)


def test_navigator_readouts_are_the_navigator_branch_and_the_rest_imaging():
    plugin = PmcRecon()
    imaging = ismrmrd.Acquisition()
    rtfeedback = ismrmrd.Acquisition()
    rtfeedback.set_flag(ismrmrd.ACQ_IS_RTFEEDBACK_DATA)
    noise = ismrmrd.Acquisition()
    noise.set_flag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)

    assert plugin.branch_for(_navigator(np.eye(3))[0]) == "navigator"
    assert plugin.branch_for(rtfeedback) == "navigator"
    assert plugin.branch_for(imaging) == "imaging"
    assert plugin.branch_for(noise) is None
