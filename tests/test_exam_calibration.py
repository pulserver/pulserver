"""The names a calibration scan and the series after it agree on."""

import numpy as np
import pytest
from conftest import (
    acs_readouts,
    calibration_header,
    coil_phantom,
    exam_header,
    play,
    readouts,
    stream_plugin,
)

import pulserver.recon as recon
from pulserver.recon._runtime.exam import ExamCacheManager

NAMES = (
    recon.B0_MAP,
    recon.B1_MAP,
    recon.COIL_SENSITIVITIES,
    recon.NOISE_COVARIANCE,
)


def test_a_map_one_series_measures_is_read_by_the_next(tmp_path):
    """The point of naming them: two series of an exam mean the same thing."""
    calibration = recon.ExamCache("exam-1", tmp_path)
    calibration[recon.B1_MAP] = [1.0, 0.9, 0.8]
    calibration.close()

    later = recon.ExamCache("exam-1", tmp_path)
    assert later[recon.B1_MAP] == [1.0, 0.9, 0.8]
    later.close()


def test_the_maps_are_told_apart():
    assert len(set(NAMES)) == len(NAMES)


def test_every_name_is_public():
    """A plugin writing one has to be able to name it without reaching inside."""
    for name in ("B0_MAP", "B1_MAP", "COIL_SENSITIVITIES", "NOISE_COVARIANCE"):
        assert name in recon.__all__


def test_maps_from_a_calibration_series_reach_a_later_series(tmp_path):
    """The maps a calibration series leaves to its exam are the maps a unit of a later series without calibration readouts is given."""
    pytest.importorskip("torch")
    _, _, kspace = coil_phantom(6, 32)
    estimates = []

    def estimate(calibration):
        estimates.append(calibration)
        return calibration / calibration.abs().amax()

    def calibrate(context, data):
        recon.coil_maps(context, data, estimate=estimate)
        context.coil_sensitivities = context.coil_maps[0]

    def image(context, data):
        return recon.coil_maps(context, data, estimate=estimate)

    calibration, imaging = stream_plugin(calibrate), stream_plugin(image)
    first = ExamCacheManager(directory=tmp_path)
    second = ExamCacheManager(directory=tmp_path)
    with first.lease(exam_header("exam-7")) as cache:
        scan = play(
            calibration,
            calibration_header(6, 32, measurement="7"),
            acs_readouts(kspace),
            exam=cache,
        )
    with second.lease(exam_header("exam-7")) as cache:
        play(
            imaging,
            calibration_header(6, 32, measurement="8"),
            readouts(kspace, last=("ACQ_LAST_IN_SLICE",)),
            exam=cache,
        )
    first.close()
    second.close()

    (maps,) = imaging.results
    assert len(estimates) == 1, "the later series estimated maps of its own"
    np.testing.assert_array_equal(maps.numpy(), scan.coil_maps[0].maps)
