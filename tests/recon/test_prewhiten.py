"""Prewhitening of readouts by the stream's noise measurement, against a covariance computed here."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from conftest import calibration_header, play, readouts

from pulserver import recon
from pulserver.mrd import AcquisitionFlag

COILS = 4
LINES = 64
SAMPLES = 64
NOISE = "ACQ_IS_NOISE_MEASUREMENT"
LAST = ("ACQ_LAST_IN_SLICE",)

#: Sample covariances of ``LINES * SAMPLES`` samples differ from the covariance
#: they estimate by the order of ``1 / sqrt(samples)``.
STATISTICAL = 5 / np.sqrt(LINES * SAMPLES)


@pytest.fixture
def bartorch():
    return pytest.importorskip("bartorch")


needs_bartorch = pytest.mark.usefixtures("bartorch")


class Collect(recon.ReconPlugin):
    """Keeps the units it is given, and returns nothing."""

    def __init__(self, *gadgets):
        super().__init__(
            gadgets=gadgets, triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE}
        )
        self.units = []

    def recon(self, context, branch, data):
        self.units.append(data)


class NoiseScan(Collect):
    """A noise series: it reconstructs nothing and publishes its whitening."""

    def finish(self, context):
        self.gadget(recon.Prewhiten).publish()


def header(coils=COILS):
    return calibration_header(coils, SAMPLES)


def covariance(coils=COILS, seed=0):
    """A Hermitian positive-definite matrix, with correlated channels of unequal power."""
    generator = np.random.default_rng(seed)
    mix = generator.standard_normal((coils, coils)) + 1j * generator.standard_normal(
        (coils, coils)
    )
    return mix @ mix.conj().T / coils + 0.5 * np.eye(coils)


def noise(psi, lines=LINES, samples=SAMPLES, seed=1):
    """``(coils, lines, samples)`` complex Gaussian noise whose channel covariance is ``psi``."""
    generator = np.random.default_rng(seed)
    shape = (len(psi), lines * samples)
    white = (
        generator.standard_normal(shape) + 1j * generator.standard_normal(shape)
    ) / np.sqrt(2)
    colored = np.linalg.cholesky(psi) @ white
    return colored.reshape(len(psi), lines, samples).astype(np.complex64)


def channel_covariance(kspace):
    """The unbiased sample covariance of the channels of ``(coils, ...)`` data."""
    flat = kspace.reshape(kspace.shape[0], -1)
    return flat @ flat.conj().T / (flat.shape[1] - 1)


@needs_bartorch
def test_noise_readouts_whiten_the_imaging_readouts():
    """Noise measured with a covariance, then imaging readouts with the same: the whitened covariance is the identity."""
    psi = covariance()
    plugin = Collect(recon.Prewhiten())

    play(
        plugin,
        header(),
        readouts(noise(psi, seed=1), NOISE) + readouts(noise(psi, seed=2), last=LAST),
    )

    (unit,) = plugin.units
    np.testing.assert_allclose(
        channel_covariance(unit.data.kspace), np.eye(COILS), atol=STATISTICAL
    )
    assert len(unit.data.headers) == LINES, "no noise readout is placed"


@needs_bartorch
def test_the_whitening_matrix_maps_the_noise_covariance_to_the_identity():
    psi = covariance()
    measured = noise(psi)
    plugin = Collect(recon.Prewhiten())

    context = play(
        plugin,
        header(),
        readouts(measured, NOISE) + readouts(measured, last=LAST),
    )

    matrix = context.noise.matrix
    np.testing.assert_allclose(
        matrix @ channel_covariance(measured) @ matrix.conj().T,
        np.eye(COILS),
        atol=1e-4,
    )


@needs_bartorch
def test_a_readout_at_another_dwell_time_is_scaled_by_the_square_root_of_the_ratio():
    """The noise variance of a sample is inversely proportional to its dwell time."""
    psi = covariance()
    plugin = Collect(recon.Prewhiten())
    noise_dwell, imaging_dwell = 4.0, 16.0

    play(
        plugin,
        header(),
        readouts(noise(psi, seed=1), NOISE, dwell_us=noise_dwell)
        + readouts(
            noise(psi * noise_dwell / imaging_dwell, seed=2),
            last=LAST,
            dwell_us=imaging_dwell,
        ),
    )

    (unit,) = plugin.units
    np.testing.assert_allclose(
        channel_covariance(unit.data.kspace), np.eye(COILS), atol=STATISTICAL
    )


@needs_bartorch
def test_noise_readouts_after_the_first_imaging_readout_are_consumed_and_not_used():
    psi = covariance()
    other = covariance(seed=5)
    imaging = noise(psi, seed=2)
    plugin = Collect(recon.Prewhiten())

    context = play(
        plugin,
        header(),
        readouts(noise(psi, seed=1), NOISE)
        + readouts(imaging, lines=range(LINES // 2))
        + readouts(noise(other, seed=3), NOISE)
        + readouts(imaging, lines=range(LINES // 2, LINES), last=LAST),
    )

    assert len(plugin.units[0].data.headers) == LINES
    matrix = context.noise.matrix
    np.testing.assert_allclose(
        matrix @ psi @ matrix.conj().T, np.eye(COILS), atol=STATISTICAL
    )


def test_readouts_pass_unchanged_and_a_warning_is_logged_once_without_any_noise(caplog):
    imaging = noise(covariance())
    plugin = Collect(recon.Prewhiten())

    with caplog.at_level(logging.WARNING):
        context = play(plugin, header(), readouts(imaging, last=LAST))

    np.testing.assert_array_equal(plugin.units[0].data.kspace, imaging)
    assert context.noise is None
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "not prewhitened" in caplog.text


def test_a_required_prewhitening_without_noise_raises_naming_what_is_missing():
    plugin = Collect(recon.Prewhiten(required=True)).spawn()
    context = recon.ReconContext.offline(header())
    plugin.startup(context)
    (first,) = readouts(noise(covariance()), lines=[0])

    with pytest.raises(recon.MissingCalibration, match=r"noise readouts.*exam"):
        plugin.receive(first, context)


@needs_bartorch
def test_fewer_noise_samples_than_channels_are_refused():
    psi = covariance()

    with pytest.raises(ValueError, match="more samples than channels"):
        play(
            Collect(recon.Prewhiten()),
            header(),
            readouts(noise(psi, lines=1, samples=COILS), NOISE)
            + readouts(noise(psi), last=LAST),
        )


@needs_bartorch
def test_a_readout_of_other_channels_than_the_noise_is_refused():
    with pytest.raises(ValueError, match="channels"):
        play(
            Collect(recon.Prewhiten()),
            header(),
            readouts(noise(covariance()), NOISE)
            + readouts(noise(covariance(2)), last=LAST),
        )


@needs_bartorch
def test_a_noise_series_leaves_its_whitening_to_the_series_of_its_exam(tmp_path):
    psi = covariance()
    scan = play(
        NoiseScan(recon.Prewhiten()),
        header(),
        readouts(noise(psi), NOISE),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    plugin = Collect(recon.Prewhiten())
    context = play(
        plugin,
        header(),
        readouts(noise(psi, seed=2), last=LAST),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    assert context.noise.id == scan.noise.id
    np.testing.assert_allclose(
        channel_covariance(plugin.units[0].data.kspace),
        np.eye(COILS),
        atol=STATISTICAL,
    )


@needs_bartorch
def test_a_stored_whitening_is_loaded_only_by_a_series_of_the_same_coils(tmp_path):
    psi = covariance()
    play(
        NoiseScan(recon.Prewhiten()),
        header(),
        readouts(noise(psi), NOISE),
        exam=recon.ExamCache("exam-1", tmp_path),
    )
    renamed = header()
    renamed.acquisitionSystemInformation.coilLabel[0].coilName = "Body"

    context = play(
        Collect(recon.Prewhiten()),
        renamed,
        readouts(noise(psi, seed=2), last=LAST),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    assert context.noise is None


@needs_bartorch
def test_the_noise_a_series_measures_replaces_the_stored_whitening(tmp_path):
    own = covariance(seed=9)
    play(
        NoiseScan(recon.Prewhiten()),
        header(),
        readouts(noise(covariance(seed=1)), NOISE),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    context = play(
        Collect(recon.Prewhiten()),
        header(),
        readouts(noise(own), NOISE) + readouts(noise(own, seed=2), last=LAST),
        exam=recon.ExamCache("exam-1", tmp_path),
    )

    matrix = context.noise.matrix
    np.testing.assert_allclose(
        matrix @ own @ matrix.conj().T, np.eye(COILS), atol=STATISTICAL
    )


def test_a_series_without_noise_has_no_whitening_to_publish():
    gadget = recon.Prewhiten()
    gadget.startup(recon.ReconContext.offline(header()))

    with pytest.raises(recon.MissingCalibration, match="no noise to publish"):
        gadget.publish()
