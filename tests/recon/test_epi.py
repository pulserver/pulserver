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


def _ramp_positions():
    """Twofold oversampled positions, denser on the ramps, in cycles per pixel."""
    u = np.linspace(-1, 1, 2 * COLUMNS)
    return 0.5 * np.sin(0.5 * np.pi * u)


def _ramp_gadget():
    gadget = RampSampling()
    gadget.geometry, gadget.operators = [(COLUMNS, FOV)], {}
    return gadget


def test_a_resampled_readout_is_a_full_echo_of_the_readout_grid():
    positions = _ramp_positions()
    acquisition = _acquisition(traj=(positions * COLUMNS / FOV)[:, None])
    acquisition.center_sample = 3
    acquisition.discard_pre, acquisition.discard_post = 2, 1

    out = _ramp_gadget()(acquisition, _sampled(positions)[None])

    assert out.shape == (1, COLUMNS)
    assert acquisition.center_sample == COLUMNS // 2
    assert (acquisition.discard_pre, acquisition.discard_post) == (0, 0)


def test_a_reversed_readout_is_resampled_with_the_grid_centre_counted_from_its_end():
    positions = _ramp_positions()[::-1]
    acquisition = _acquisition(
        AcquisitionFlag.IS_REVERSE, traj=(positions * COLUMNS / FOV)[:, None]
    )

    _ramp_gadget()(acquisition, _sampled(positions)[None])

    assert acquisition.center_sample == COLUMNS - 1 - COLUMNS // 2


def test_flipping_a_reversed_readout_mirrors_its_echo_and_its_discards():
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    acquisition = _acquisition(AcquisitionFlag.IS_REVERSE)
    acquisition.center_sample = 10
    acquisition.discard_pre, acquisition.discard_post = 4, 2

    gadget(acquisition, np.ones((1, COLUMNS), dtype=complex))

    assert acquisition.center_sample == COLUMNS - 1 - 10
    assert (acquisition.discard_pre, acquisition.discard_post) == (2, 4)


def test_a_forward_readout_keeps_its_echo_and_its_discards():
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    acquisition = _acquisition()
    acquisition.center_sample = 10
    acquisition.discard_pre, acquisition.discard_post = 4, 2

    gadget(acquisition, np.ones((1, COLUMNS), dtype=complex))

    assert acquisition.center_sample == 10
    assert (acquisition.discard_pre, acquisition.discard_post) == (4, 2)


def test_forward_and_reversed_lines_leave_the_gadgets_in_one_direction_with_one_echo():
    positions = _ramp_positions()
    ramp = _ramp_gadget()
    correction = EpiPhaseCorrection()
    correction.startup(SimpleNamespace(header=None))
    lines = []
    for flags, order in (
        (AcquisitionFlag(0), slice(None)),
        (AcquisitionFlag.IS_REVERSE, slice(None, None, -1)),
    ):
        acquisition = _acquisition(
            flags, traj=(positions[order] * COLUMNS / FOV)[:, None]
        )
        data = correction(
            acquisition, ramp(acquisition, _sampled(positions[order])[None])
        )
        lines.append((acquisition.center_sample, data[0]))

    assert [echo for echo, _ in lines] == [COLUMNS // 2, COLUMNS // 2]
    np.testing.assert_allclose(lines[1][1], lines[0][1], atol=1e-3)


class _Correction:
    """Stands in for bartorch's susceptibility correction: records what it is given and shifts ``blip_up`` by one along the phase encode."""

    def __init__(self):
        self.calls = []

    def __call__(self, blip_up, blip_down, *, voxel_size, phase_encoding_axis):
        self.calls.append((tuple(blip_up.shape), voxel_size, phase_encoding_axis))
        return SimpleNamespace(blip_up=blip_up.roll(1, dims=phase_encoding_axis))


@pytest.fixture
def correction(monkeypatch):
    from bartorch import tools

    stand_in = _Correction()
    monkeypatch.setattr(tools, "correct_susceptibility", stand_in)
    return stand_in


def test_a_slice_is_corrected_as_three_identical_slices_its_phase_encode_first(
    correction,
):
    from pulserver.recon.handlers.epi import undistorted

    image = np.random.default_rng(0).random((6, 8))

    corrected = undistorted(image, image, (2.0, 3.0))

    assert correction.calls == [((6, 8, 3), (2.0, 3.0, 1.0), 0)]
    np.testing.assert_allclose(corrected, np.roll(image, 1, axis=0))


def test_a_volume_is_corrected_as_one_its_phase_encode_first(correction):
    from pulserver.recon.handlers.epi import undistorted

    volume = np.random.default_rng(1).random((4, 6, 8))

    corrected = undistorted(volume, volume, (5.0, 2.0, 3.0))

    # (partition, phase, readout) goes in as (phase, readout, partition).
    assert correction.calls == [((6, 8, 4), (2.0, 3.0, 5.0), 0)]
    np.testing.assert_allclose(corrected, np.roll(volume, 1, axis=1))
