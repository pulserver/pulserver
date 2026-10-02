"""The EPI reconstruction's per-readout steps, against readouts simulated here."""

from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.mrd import AcquisitionFlag
from pulserver.recon.handlers.epi import EpiPhaseCorrection, RampSampling

pytest.importorskip("bartorch")

COLUMNS, FOV = 32, 0.2


def _acquisition(flags=0, traj=None):
    return SimpleNamespace(
        flags=int(getattr(flags, "value", flags)), traj=traj, encoding_space_ref=0
    )


def _object():
    x = np.arange(COLUMNS) - COLUMNS // 2
    return np.exp(-((x / 5.0) ** 2)) * np.exp(0.3j * x / COLUMNS)


def _sampled(positions):
    """The readout of :func:`_object` at ``positions``, in cycles per pixel."""
    x = np.arange(COLUMNS) - COLUMNS // 2
    return np.exp(-2j * np.pi * np.outer(positions, x)) @ _object()


def test_a_ramp_sampled_readout_is_resampled_onto_the_readout_grid():
    # Twofold oversampled and denser on the ramps, as a trapezoid reads it.
    u = np.linspace(-1, 1, 2 * COLUMNS)
    positions = 0.5 * np.sin(0.5 * np.pi * u)
    gadget = RampSampling()
    gadget.geometry, gadget.operators = [(COLUMNS, FOV)], {}

    for order in (slice(None), slice(None, None, -1)):
        traj = (positions[order] * COLUMNS / FOV)[:, None]
        out = gadget(_acquisition(traj=traj), _sampled(positions[order])[None])

        grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
        np.testing.assert_allclose(out[0], _sampled(grid)[order], atol=1e-3)


def test_the_navigator_phase_is_removed_from_the_reversed_readouts():
    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    line = _sampled(grid)
    u = np.linspace(-1, 1, COLUMNS)

    def hybrid(data, inverse):
        transform = np.fft.ifft if inverse else np.fft.fft
        return np.fft.fftshift(transform(np.fft.ifftshift(data)))

    reversed_line = hybrid(hybrid(line, True) * np.exp(1j * (0.4 + 0.8 * u)), False)
    reversed_line = reversed_line[::-1][None]
    forward, backward = AcquisitionFlag(0), AcquisitionFlag.IS_REVERSE
    nav = AcquisitionFlag.IS_NAVIGATION_DATA
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))

    for flags, data in (
        (forward, line[None]),
        (backward, reversed_line),
        (forward, line[None]),
    ):
        assert gadget(_acquisition(flags | nav), data) is None
    corrected = gadget(_acquisition(backward), reversed_line)

    np.testing.assert_allclose(corrected[0], line, atol=1e-4 * np.abs(line).max())
