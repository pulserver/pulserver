"""The coronary MRA reconstruction: each segment's translation read from its image navigator and undone before the solve."""

import ismrmrd
import numpy as np
import pytest
from conftest import calibration_header, coil_phantom, play, readouts

from pulserver.recon.handlers.coronary import CoronaryRecon, SegmentMotion
from pulserver.recon.handlers.pics import PicsRecon

pytest.importorskip("bartorch")

COILS, MATRIX, PARTITIONS, SEGMENTS, NAV_LINES = 4, 32, 4, 8, 10
#: Displacement of each segment, voxels along (phase encodes, readout): a
#: breathing excursion mostly along the readout.
SHIFTS = [
    (0.4 * np.sin(s), 3.0 * np.sin(s)) for s in np.linspace(0, 2 * np.pi, SEGMENTS)
]


class Images:
    def __init__(self, cls, **kwargs):
        class Recording(cls):
            def recon(inner, context, branch, data):
                result = super().recon(context, branch, data)
                if result is not None:
                    self.results.append(result)
                    self.stream = inner
                return result

        self.results = []
        self.plugin = Recording(
            kspace_filter=False,
            gradient_unwarp=False,
            intensity_correction=False,
            **kwargs,
        )


def _volume_kspace(shift):
    """Coil k-space ``(coils, z, y, x)`` of the phantom moved by ``shift`` voxels ``(y, x)``."""
    image, maps, _ = coil_phantom(COILS, MATRIX)
    z = np.hanning(PARTITIONS + 2)[1:-1].astype(np.complex64)
    coil = maps[:, None] * (z[:, None, None] * image)[None]
    kspace = np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(coil, axes=(1, 2, 3)), axes=(1, 2, 3), norm="ortho"
        ),
        axes=(1, 2, 3),
    )
    k = (np.arange(MATRIX) - MATRIX // 2) / MATRIX
    phase = np.exp(-2j * np.pi * (k[:, None] * shift[0] + k[None, :] * shift[1]))
    return (kspace * phase[None, None]).astype(np.complex64)


def _stream(shifts, navigator=True):
    """The readouts of a segmented scan: each segment its navigator, then its interleave of views."""
    header = calibration_header(COILS, MATRIX, partitions=PARTITIONS)
    views = [(y, z) for z in range(PARTITIONS) for y in range(MATRIX)]
    acquisitions = []
    for s, shift in enumerate(shifts):
        kspace = _volume_kspace(shift)
        if navigator:
            lines = range(MATRIX // 2 - NAV_LINES // 2, MATRIX // 2 + NAV_LINES // 2)
            nav = readouts(
                kspace[:, PARTITIONS // 2],
                "ACQ_IS_NAVIGATION_DATA",
                lines=lines,
                partition=PARTITIONS // 2,
            )
            acquisitions += nav
        mine = views[s :: len(shifts)]
        for y, z in mine:
            [a] = readouts(kspace[:, z], lines=[y], partition=z)
            acquisitions.append(a)
        for a in acquisitions[-len(mine) - (NAV_LINES if navigator else 0) :]:
            a.idx.segment = s
    acquisitions[-1].setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    return header, acquisitions


def _reconstructed(cls, shifts, device=None, navigator=True, **kwargs):
    header, acquisitions = _stream(shifts, navigator)
    images = Images(cls, **kwargs)
    play(images.plugin, header, acquisitions, device=device)
    return np.asarray(images.results[-1].data), images.stream


def _error(image, reference):
    image, reference = np.abs(image), np.abs(reference)
    scale = np.vdot(image, reference).real / np.vdot(image, image).real
    return float(np.linalg.norm(scale * image - reference) / np.linalg.norm(reference))


def test_a_region_outside_the_field_of_view_is_refused():
    with pytest.raises(ValueError, match="region"):
        SegmentMotion(region=1.5)


def test_undoing_each_segments_navigator_displacement_removes_the_breathing_ghosts(
    device,
):
    still, _ = _reconstructed(PicsRecon, [(0.0, 0.0)] * SEGMENTS, device)
    moving, _ = _reconstructed(PicsRecon, SHIFTS, device)
    corrected, _ = _reconstructed(CoronaryRecon, SHIFTS, device)

    assert _error(corrected, still) < 0.3 * _error(moving, still)


def test_the_navigator_measures_each_segments_displacement_in_millimetres():
    _, plugin = _reconstructed(CoronaryRecon, SHIFTS)

    [motion] = [g for g in plugin.gadgets if isinstance(g, SegmentMotion)]
    voxel = 220.0 / MATRIX
    measured = np.array([motion.shifts[s] for s in range(SEGMENTS)])
    expected = np.array([(x * voxel, y * voxel) for y, x in SHIFTS])
    np.testing.assert_allclose(measured, expected, atol=0.1 * voxel)


def test_a_scan_without_navigators_is_reconstructed_as_pics_reconstructs_it():
    plain, _ = _reconstructed(PicsRecon, SHIFTS, navigator=False)
    coronary, _ = _reconstructed(CoronaryRecon, SHIFTS, navigator=False)

    np.testing.assert_allclose(coronary, plain, rtol=1e-5, atol=1e-6)
