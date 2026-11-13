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


def test_an_epi_with_additional_adc_samples_reconstructs_the_prescribed_readout_grid():
    import pypulseqpp as pp
    from pypulseqpp import sequences

    system = pp.Opts(
        max_slew=50.0,
        slew_unit="T/m/s",
        grad_raster_time=4e-6,
        block_duration_raster=4e-6,
    )
    exc = sequences.SpatialSelectiveExcitation(system, 30.0, 0.005)
    epi = sequences.EpiReadout2D(
        system,
        exc.rf,
        exc.gz,
        exc.gz_reph,
        fov=FOV,
        matrix=(COLUMNS, 4),
        oversampling=2.0,
        readout_bandwidth_hz=500e3,
    )
    positions = (
        np.asarray(epi.calculate_kspace()[0])[0, : epi.n_samples] * FOV / COLUMNS
    )
    gadget = _ramp_gadget()

    assert epi.n_samples > 2 * COLUMNS
    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    for order in (slice(None), slice(None, None, -1)):
        acquisition = _acquisition(traj=(positions[order] * COLUMNS / FOV)[:, None])
        out = gadget(acquisition, _sampled(positions[order])[None])

        assert out.shape == (1, COLUMNS)
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


def _oversampled_header(columns, oversampling):
    """A header of one Cartesian space whose readout is sampled ``oversampling`` times over its reconstruction."""

    def fov(x):
        return SimpleNamespace(x=x, y=1e3 * FOV, z=5.0)

    space = SimpleNamespace(
        encodedSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(x=round(oversampling * columns), y=4, z=1),
            fieldOfView_mm=fov(oversampling * 1e3 * FOV),
        ),
        reconSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(x=columns, y=4, z=1),
            fieldOfView_mm=fov(1e3 * FOV),
        ),
        encodingLimits=SimpleNamespace(
            kspace_encoding_step_1=SimpleNamespace(minimum=0, maximum=3, center=2)
        ),
        trajectory="cartesian",
    )
    return SimpleNamespace(
        encoding=[space],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=1),
    )


def test_a_readout_without_a_trajectory_is_cropped_to_the_epi_readout_grid():
    # A gradient-echo calibration line, uniformly sampled twice over: it has to
    # land on the grid the ramp-sampled EPI lines are resampled onto, or the
    # coil sensitivities estimated from it serve none of them.
    from pulserver.recon import ReconContext

    gadget = RampSampling()
    gadget.startup(ReconContext.offline(_oversampled_header(COLUMNS, 2.0)))
    full = 2 * COLUMNS
    pixel = 3
    readout = np.exp(-2j * np.pi * (np.arange(full) - full // 2) * pixel / full)
    acquisition = _acquisition()
    acquisition.center_sample = full // 2
    acquisition.discard_pre = acquisition.discard_post = 0

    out = gadget(acquisition, readout[None])

    assert out.shape == (1, COLUMNS)
    assert acquisition.center_sample == COLUMNS // 2
    expected = np.exp(
        -2j * np.pi * (np.arange(COLUMNS) - COLUMNS // 2) * pixel / COLUMNS
    )
    np.testing.assert_allclose(out[0] / out[0, COLUMNS // 2], expected, atol=1e-4)


def test_without_pyhysco_undistorted_raises_rather_than_passing_the_image_on(
    monkeypatch,
):
    from bartorch import tools

    from pulserver.recon.handlers.epi import undistorted

    def missing(*args, **kwargs):
        raise ImportError("no PyHySCO")

    monkeypatch.setattr(tools, "correct_susceptibility", missing)

    with pytest.raises(ImportError):
        undistorted(np.ones((6, 8)), np.ones((6, 8)), (2.0, 3.0))


def _epi_recon(monkeypatch, image):
    """An EpiRecon whose solve returns ``image``, started on an offline stream."""
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi
    from pulserver.recon.handlers.pics import PicsRecon

    monkeypatch.setattr(PicsRecon, "image", lambda self, *args: image)
    plugin = epi.EpiRecon(gradient_unwarp=False, intensity_correction=False)
    plugin.startup(ReconContext.offline())
    return plugin


def _unit(set_index):
    space = SimpleNamespace(recon_fov=(0.2, 0.2), recon_matrix=(6, 8))
    return SimpleNamespace(
        counters={"slice": 0, "set": set_index}, data=SimpleNamespace(space=space)
    )


def _image(plugin, set_index):
    return plugin.image(None, (6, 8), None, _unit(set_index))


def test_an_image_without_its_reversed_reference_is_sent_on_with_a_warning(
    monkeypatch, caplog
):
    import logging

    plugin = _epi_recon(monkeypatch, np.ones((6, 8)))

    with caplog.at_level(logging.DEBUG, logger="pulserver.recon"):
        _image(plugin, 0)
        _image(plugin, 0)

    skipped = [
        r
        for r in caplog.records
        if "not corrected for susceptibility" in r.getMessage()
    ]
    assert [r.levelno for r in skipped] == [logging.WARNING, logging.DEBUG]
    assert (
        "no phase-encode-reversed reference (set 1) of slice 0"
        in skipped[0].getMessage()
    )


def test_an_image_left_uncorrected_for_want_of_pyhysco_says_so(monkeypatch, caplog):
    import logging

    from bartorch import tools

    def missing(*args, **kwargs):
        raise ImportError("no PyHySCO")

    monkeypatch.setattr(tools, "correct_susceptibility", missing)
    plugin = _epi_recon(monkeypatch, np.ones((6, 8)))

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        _image(plugin, 1)
        out = _image(plugin, 0)

    np.testing.assert_array_equal(out, np.ones((6, 8)))
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "PyHySCO is not installed" in warning.getMessage()


def test_a_corrected_image_is_logged_and_carries_hysco_in_its_history(
    monkeypatch, caplog, correction
):
    import logging

    plugin = _epi_recon(monkeypatch, np.ones((6, 8)))

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        _image(plugin, 1)
        _image(plugin, 0)

    assert len(correction.calls) == 1
    assert any(
        r.levelno == logging.INFO
        and r.getMessage().startswith("susceptibility distortion of a (6, 8) image")
        for r in caplog.records
    )
    assert plugin.corrections.applied() == ["HYSCO"]


def test_the_reversed_reference_is_reconstructed_and_kept_but_sent_as_no_image(
    monkeypatch,
):
    from pulserver.recon.handlers.pics import PicsRecon

    plugin = _epi_recon(monkeypatch, np.ones((6, 8)))
    reconstructed = []
    monkeypatch.setattr(
        plugin, "unit_image", lambda context, data: reconstructed.append(data)
    )
    monkeypatch.setattr(PicsRecon, "recon", lambda self, context, branch, data: "sent")

    assert plugin.recon(None, "", _unit(1)) is None
    assert len(reconstructed) == 1
    assert plugin.recon(None, "", _unit(0)) == "sent"
    assert len(reconstructed) == 1


def _hybrid(data, inverse):
    transform = np.fft.ifft if inverse else np.fft.fft
    return np.fft.fftshift(transform(np.fft.ifftshift(data)))


def _reversed(line, constant, slope):
    """``line`` read reversed, carrying ``constant + slope * u`` in hybrid space."""
    u = np.linspace(-1, 1, COLUMNS)
    phased = _hybrid(_hybrid(line, True) * np.exp(1j * (constant + slope * u)), False)
    return phased[::-1][None]


def _labelled(flags, echo, location=0):
    acquisition = _acquisition(flags)
    acquisition.idx = SimpleNamespace(user=[echo, 0, 0, 0, 0, 0, 0, 0], slice=location)
    return acquisition


def test_the_reference_phase_of_each_echo_is_removed_from_the_reversed_readouts_of_that_echo():
    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    line = _sampled(grid)
    phases = {1: (0.4, 0.8), 3: (-0.2, 1.1), 5: (0.0, 0.5)}
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    reference = AcquisitionFlag.IS_PHASECORR_DATA
    for echo in range(6):
        if echo in phases:
            data, flags = (
                _reversed(line, *phases[echo]),
                reference | AcquisitionFlag.IS_REVERSE,
            )
        else:
            data, flags = line[None], reference
        assert gadget(_labelled(flags, echo), data) is None

    for echo in (1, 3):
        corrected = gadget(
            _labelled(AcquisitionFlag.IS_REVERSE, echo), _reversed(line, *phases[echo])
        )
        np.testing.assert_allclose(corrected[0], line, atol=1e-3 * np.abs(line).max())
    # The last echo has a forward neighbour on one side only: it takes echo 3's.
    corrected = gadget(
        _labelled(AcquisitionFlag.IS_REVERSE, 5), _reversed(line, *phases[3])
    )
    np.testing.assert_allclose(corrected[0], line, atol=1e-3 * np.abs(line).max())


def test_a_slice_without_its_own_reference_takes_the_latest_and_navigators_are_ignored():
    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    line = _sampled(grid)
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    reference = AcquisitionFlag.IS_PHASECORR_DATA
    for echo in range(4):
        reverse = echo % 2 == 1
        flags = reference | (
            AcquisitionFlag.IS_REVERSE if reverse else AcquisitionFlag(0)
        )
        data = _reversed(line, 0.4, 0.8) if reverse else line[None]
        gadget(_labelled(flags, echo, location=2), data)
    # A navigator measuring another phase does not displace the reference.
    nav = AcquisitionFlag.IS_NAVIGATION_DATA
    for flags, data in (
        (AcquisitionFlag(0), line[None]),
        (AcquisitionFlag.IS_REVERSE, _reversed(line, -1.0, 0.1)),
        (AcquisitionFlag(0), line[None]),
    ):
        assert gadget(_labelled(flags | nav, 0), data) is None

    corrected = gadget(
        _labelled(AcquisitionFlag.IS_REVERSE, 1, location=5), _reversed(line, 0.4, 0.8)
    )

    np.testing.assert_allclose(corrected[0], line, atol=1e-3 * np.abs(line).max())


def test_reversed_readouts_without_reference_or_navigator_are_flipped_with_a_warning(
    caplog,
):
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    data = (np.arange(COLUMNS) + 1j).astype(complex)[None]

    with caplog.at_level("WARNING", logger="pulserver.recon"):
        first = gadget(_labelled(AcquisitionFlag.IS_REVERSE, 1), data)
        gadget(_labelled(AcquisitionFlag.IS_REVERSE, 3), data)

    np.testing.assert_allclose(first[0], data[0, ::-1])
    assert (
        sum("neither an odd/even phase reference" in r.message for r in caplog.records)
        == 1
    )


def test_the_epi_reconstruction_takes_in_the_phase_reference():
    from pulserver.recon.handlers.epi import EpiRecon

    recon = EpiRecon()
    acquisition = _labelled(AcquisitionFlag.IS_PHASECORR_DATA, 0)

    assert recon._accepts(acquisition)
