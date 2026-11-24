"""The EPI reconstruction's per-readout steps, against readouts simulated here."""

from types import SimpleNamespace

import numpy as np
import pytest

from pulserver.mrd import AcquisitionFlag
from pulserver.recon._buffers import ReconData
from pulserver.recon.handlers.epi import EpiPhaseCorrection, RampSampling, _polarity_of

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


def _hybrid(data, inverse):
    transform = np.fft.ifft if inverse else np.fft.fft
    return np.fft.fftshift(transform(np.fft.ifftshift(data)))


def _phased(line, constant, linear):
    """``line`` with the phase ``constant + linear * u`` across the readout, in hybrid space."""
    u = np.linspace(-1, 1, COLUMNS)
    return _hybrid(_hybrid(line, True) * np.exp(1j * (constant + linear * u)), False)


def _excitation(gadget, constant, linear):
    """Read one excitation's three-line navigator and a forward and a reversed imaging line, all carrying the phase ``constant + linear * u``; return the two imaging lines as corrected."""
    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    line = _phased(_sampled(grid), constant, linear)
    forward, backward = AcquisitionFlag(0), AcquisitionFlag.IS_REVERSE
    nav = AcquisitionFlag.IS_NAVIGATION_DATA
    for flags, data in (
        (forward, line[None]),
        (backward, line[::-1][None]),
        (forward, line[None]),
    ):
        assert gadget(_acquisition(flags | nav), data) is None
    return (
        gadget(_acquisition(forward), line[None])[0],
        gadget(_acquisition(backward), line[::-1][None])[0],
    )


@pytest.mark.parametrize("slab", [True, False])
def test_each_slab_excitation_has_its_navigator_phase_removed_from_every_readout(
    slab,
):
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    gadget.slabs = {0} if slab else set()
    first = _excitation(gadget, 0.0, 0.0)

    later = _excitation(gadget, 0.7, -0.5)

    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    expected = first if slab else [_phased(_sampled(grid), 0.7, -0.5)] * 2
    scale = np.abs(first[0]).max()
    for got, want in zip(later, expected, strict=True):
        np.testing.assert_allclose(got, want, atol=1e-4 * scale)


def test_the_excitation_phase_is_left_when_it_is_not_asked_for():
    gadget = EpiPhaseCorrection(excitation_phase=False)
    gadget.startup(SimpleNamespace(header=None))
    gadget.slabs = {0}
    _excitation(gadget, 0.0, 0.0)

    later = _excitation(gadget, 0.7, -0.5)

    grid = (np.arange(COLUMNS) - COLUMNS // 2) / COLUMNS
    np.testing.assert_allclose(later[0], _phased(_sampled(grid), 0.7, -0.5), atol=1e-4)


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
    """Stands in for bartorch's susceptibility estimate: records what it is given and returns a field displacing a slice by one voxel, and the k-th slice of a volume's last axis by k voxels."""

    def __init__(self):
        self.calls = []
        self.smoothness = []
        self.blip_up = []

    def __call__(
        self, blip_up, blip_down, *, voxel_size, phase_encoding_axis, alpha=None
    ):
        import torch

        self.calls.append((tuple(blip_up.shape), voxel_size, phase_encoding_axis))
        self.smoothness.append(alpha)
        self.blip_up.append(blip_up.numpy().copy())
        shape = list(blip_up.shape)
        shape[phase_encoding_axis] += 1
        if len(shape) == 2:
            field = torch.full(
                shape, voxel_size[phase_encoding_axis], dtype=torch.float64
            )
            return SimpleNamespace(field_map=field)
        steps = torch.arange(shape[-1], dtype=torch.float64)
        field = (voxel_size[phase_encoding_axis] * steps).expand(shape).clone()
        return SimpleNamespace(field_map=field)


@pytest.fixture
def correction(monkeypatch):
    from bartorch import tools

    stand_in = _Correction()
    monkeypatch.setattr(tools, "correct_susceptibility", stand_in)
    return stand_in


def test_a_slice_is_estimated_in_its_plane_its_phase_encode_first(
    correction,
):
    from pulserver.recon.handlers.epi import susceptibility_field, undistorted

    image = np.random.default_rng(0).random((6, 8))

    field = susceptibility_field(image, image, (2.0, 3.0))
    corrected = undistorted(image, field, (2.0, 3.0))

    assert correction.calls == [((6, 8), (2.0, 3.0), 0)]
    # A field of one voxel: the image is sampled at x + 1.
    assert field.shape == (7, 8)
    np.testing.assert_allclose(corrected, np.roll(image, -1, axis=0))


def test_a_volume_is_estimated_as_one_its_phase_encode_first_its_end_partitions_repeated(
    correction,
):
    from pulserver.recon.handlers.epi import (
        FIELD_EDGE_PARTITIONS,
        susceptibility_field,
        undistorted,
    )

    volume = np.random.default_rng(1).random((4, 6, 8))

    field = susceptibility_field(volume, volume, (5.0, 2.0, 3.0))
    corrected = undistorted(volume, field, (5.0, 2.0, 3.0))

    # (partition, phase, readout) goes in as (phase, readout, partition), with
    # the end partitions repeated, and the field comes back in the volume's
    # axes: partition k displaced by k plus the repeated partitions before it.
    padded = 4 + 2 * FIELD_EDGE_PARTITIONS
    assert correction.calls == [((6, 8, padded), (2.0, 3.0, 5.0), 0)]
    assert field.shape == (4, 7, 8)
    for k in range(4):
        shift = k + FIELD_EDGE_PARTITIONS
        np.testing.assert_allclose(corrected[k], np.roll(volume[k], -shift, axis=0))


def test_a_volume_field_is_fitted_at_the_slab_smoothness_and_a_slice_at_the_slice_smoothness(
    correction,
):
    from pulserver.recon.handlers.epi import (
        SLAB_FIELD_SMOOTHNESS,
        SLICE_FIELD_SMOOTHNESS,
        susceptibility_field,
    )

    susceptibility_field(np.ones((4, 6, 8)), np.ones((4, 6, 8)), (5.0, 2.0, 3.0))
    susceptibility_field(np.ones((6, 8)), np.ones((6, 8)), (2.0, 3.0))

    assert correction.smoothness == [SLAB_FIELD_SMOOTHNESS, SLICE_FIELD_SMOOTHNESS]


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


def test_without_pyhysco_the_field_raises_rather_than_passing_the_image_on(
    monkeypatch,
):
    from bartorch import tools

    from pulserver.recon.handlers.epi import susceptibility_field

    def missing(*args, **kwargs):
        raise ImportError("no PyHySCO")

    monkeypatch.setattr(tools, "correct_susceptibility", missing)

    with pytest.raises(ImportError):
        susceptibility_field(np.ones((6, 8)), np.ones((6, 8)), (2.0, 3.0))


def _stack_recon(monkeypatch, images, *, hysco=None, context=None):
    """An EpiRecon whose units are ``images[(slice, set)]``, started on an offline stream, or on ``context``.

    ``hysco`` is the reason the susceptibility correction is unavailable;
    ``None`` makes it available.
    """
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi

    context = ReconContext.offline() if context is None else context
    plugin = epi.EpiRecon(
        gradient_unwarp=False, intensity_correction=False, kspace_filter=False
    )
    plugin.startup(context)
    plugin.hysco = hysco

    def unit_image(context, data):
        key = (data.counters["slice"], data.counters["set"])
        return data.data, data, images[key]

    monkeypatch.setattr(plugin, "unit_image", unit_image)
    return plugin, context


def _unit(location, set_index):
    space = SimpleNamespace(recon_fov=(0.005, 0.2, 0.2), recon_matrix=(1, 6, 8))
    reference = SimpleNamespace(
        position=(0.0, 0.0, 5.0 * location), slice_dir=(0.0, 0.0, 1.0)
    )
    return SimpleNamespace(
        counters={"slice": location, "set": set_index},
        data=SimpleNamespace(
            space=space,
            reference=reference,
            mask=np.ones((1, 6, 8), dtype=bool),
            sample_time=2e-6,
        ),
    )


def _images(count, polarities=(0, 1)):
    rng = np.random.default_rng(2)
    return {
        (location, polarity): rng.random((6, 8))
        for location in range(count)
        for polarity in polarities
    }


def test_the_reversed_reference_volume_is_sent_as_no_image_and_gives_no_field(
    monkeypatch, correction
):
    plugin, context = _stack_recon(monkeypatch, _images(1))

    assert plugin.recon(context, "", _unit(0, 1)) is None
    (result,) = plugin.recon(context, "", _unit(0, 0))

    assert correction.calls == []
    assert plugin.field is None
    assert "HYSCO" not in result.attributes.get("ImageProcessingHistory", [])


def test_without_a_field_pair_each_image_is_sent_at_once_with_one_warning(
    monkeypatch, caplog
):
    import logging

    plugin, context = _stack_recon(monkeypatch, _images(1))

    with caplog.at_level(logging.DEBUG, logger="pulserver.recon"):
        first = plugin.recon(context, "", _unit(0, 0))
        second = plugin.recon(context, "", _unit(0, 0))

    assert len(first) == len(second) == 1
    skipped = [
        r
        for r in caplog.records
        if "not corrected for susceptibility" in r.getMessage()
    ]
    assert [r.levelno for r in skipped] == [logging.WARNING, logging.DEBUG]
    assert "no field pair" in skipped[0].getMessage()


def test_without_pyhysco_each_image_is_sent_at_once_uncorrected_saying_so(
    monkeypatch, caplog
):
    import logging

    images = _images(1)
    plugin, context = _stack_recon(
        monkeypatch, images, hysco="PyHySCO is not installed (stand-in)"
    )

    with caplog.at_level(logging.INFO, logger="pulserver.recon"):
        plugin.recon(context, "", _unit(0, 1))
        (result,) = plugin.recon(context, "", _unit(0, 0))

    np.testing.assert_array_equal(result.data, images[(0, 0)])
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "PyHySCO is not installed" in warning.getMessage()


def test_the_field_pair_of_a_stack_gives_one_field_that_corrects_every_slice(
    monkeypatch, correction
):
    from pulserver.recon.handlers.epi import FIELD_EDGE_PARTITIONS

    images = _images(2, polarities=(0, 2, 3))
    plugin, context = _stack_recon(monkeypatch, images)
    for location in (1, 0):
        for set_index in (2, 3):
            assert plugin.recon(context, "", _unit(location, set_index)) is None

    (first,) = plugin.recon(context, "", _unit(1, 0))
    (second,) = plugin.recon(context, "", _unit(0, 0))

    # One field for the stack, the slices in the order of their positions,
    # the end slices repeated; the stand-in displaces slice k by k voxels.
    padded = 2 + 2 * FIELD_EDGE_PARTITIONS
    assert correction.calls == [((6, 8, padded), (0.2 * 1e3 / 6, 25.0, 5.0), 0)]
    for result, location in ((first, 1), (second, 0)):
        assert result.reference.position[2] == 5.0 * location
        shift = location + FIELD_EDGE_PARTITIONS
        np.testing.assert_allclose(
            result.data, np.roll(images[(location, 0)], -shift, axis=0)
        )
        assert "HYSCO" in result.attributes["ImageProcessingHistory"]


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


@pytest.mark.parametrize("delay_us", [-3.0, 4.0])
def test_the_navigator_gives_the_read_gradient_delay_and_the_ramp_resampling_applies_it(
    delay_us,
):
    dwell_us, samples = 2.0, 2 * COLUMNS + 16
    t = np.arange(samples) * dwell_us
    rise, total = 40.0, t[-1]

    def designed(time):
        # Area under a trapezoid of unit amplitude, rising over `rise` us.
        up = np.clip(time, 0, rise) ** 2 / (2 * rise)
        flat = np.clip(time - rise, 0, total - 2 * rise)
        down = np.clip(time - (total - rise), 0, rise)
        down = down - down**2 / (2 * rise)
        return up + flat + down

    span = designed(total)
    to_pixels = 1.0 / span  # cycles per pixel, -1/2 to 1/2 across the line

    def line(sign):
        nominal = sign * (designed(t) * to_pixels - 0.5)
        played = sign * (designed(t - delay_us) * to_pixels - 0.5)
        flags = AcquisitionFlag.IS_NAVIGATION_DATA
        if sign < 0:
            flags |= AcquisitionFlag.IS_REVERSE
        acquisition = SimpleNamespace(
            flags=int(flags.value),
            traj=(nominal * COLUMNS / FOV)[:, None],
            encoding_space_ref=0,
            sample_time_us=dwell_us,
        )
        return acquisition, _sampled(played)[None]

    ramp = _ramp_gadget()
    ramp.delay_s = 0.0
    gadget = EpiPhaseCorrection()
    gadget.startup(SimpleNamespace(header=None))
    gadget.ramp = ramp
    for sign in (1, -1, 1):
        acquisition, data = line(sign)
        assert ramp(acquisition, data) is data
        assert gadget(acquisition, data) is None

    np.testing.assert_allclose(ramp.delay_s, 1e-6 * delay_us, rtol=0.03)


def _calibration_unit(location, polarity, kspace):
    space = SimpleNamespace(recon_fov=(0.005, 0.2, 0.2), recon_matrix=(1, 6, 8))
    reference = SimpleNamespace(
        position=(0.0, 0.0, 5.0 * location), slice_dir=(0.0, 0.0, 1.0)
    )
    ref = SimpleNamespace(
        space=space,
        reference=reference,
        image_shape=(6, 8),
        kspace=kspace,
        coils=kspace.shape[0],
    )
    return SimpleNamespace(
        counters={"slice": location, "set": polarity}, ref=ref, data=None
    )


def test_a_fleet_calibration_of_both_polarities_gives_no_field(monkeypatch, correction):
    from pulserver.recon.handlers import epi

    monkeypatch.setattr(epi, "coil_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(epi, "_zero_filled", lambda ref: ref.kspace)
    plugin, context = _stack_recon(monkeypatch, _images(1))
    rng = np.random.default_rng(3)
    plugin.calibrations = {
        polarity: [
            _calibration_unit(
                0, polarity, rng.random((2, 1, 6, 8)).astype(np.complex64)
            )
        ]
        for polarity in (0, 1)
    }

    plugin._calibrate(context)
    plugin.recon(context, "", _unit(0, 0))

    assert correction.calls == []
    assert plugin.field is None


def test_a_spin_echo_field_pair_is_sent_as_no_image_and_its_field_corrects_every_image(
    monkeypatch, correction
):
    images = _images(1, polarities=(0, 1, 2, 3))
    plugin, context = _stack_recon(monkeypatch, images)

    assert plugin.recon(context, "", _unit(0, 2)) is None
    assert plugin.recon(context, "", _unit(0, 3)) is None
    (first,) = plugin.recon(context, "", _unit(0, 0))
    plugin.recon(context, "", _unit(0, 1))
    (later,) = plugin.recon(context, "", _unit(0, 0))

    assert len(correction.calls) == 1
    np.testing.assert_allclose(correction.blip_up[0], images[(0, 2)])
    for result in (first, later):
        np.testing.assert_allclose(result.data, np.roll(images[(0, 0)], -1, axis=0))
        assert "HYSCO" in result.attributes["ImageProcessingHistory"]


def test_a_series_without_a_field_pair_takes_the_exam_field_of_a_pair_sampled_as_it_is(
    monkeypatch, correction
):
    from pulserver.recon.handlers.epi import FIELD_EDGE_PARTITIONS

    images = _images(2, polarities=(0, 2, 3))
    first, context = _stack_recon(monkeypatch, images)
    for location in (0, 1):
        for set_index in (2, 3):
            first.recon(context, "", _unit(location, set_index))
    first.recon(context, "", _unit(0, 0))
    later, _ = _stack_recon(monkeypatch, images, context=context)

    (result,) = later.recon(context, "", _unit(1, 0))

    assert len(correction.calls) == 1
    shift = 1 + FIELD_EDGE_PARTITIONS
    np.testing.assert_allclose(result.data, np.roll(images[(1, 0)], -shift, axis=0))


def test_a_series_sampled_otherwise_than_the_field_pair_is_left_uncorrected_saying_so(
    monkeypatch, correction, caplog
):
    import logging

    images = _images(1, polarities=(0, 2, 3))
    first, context = _stack_recon(monkeypatch, images)
    for set_index in (2, 3):
        first.recon(context, "", _unit(0, set_index))
    first.recon(context, "", _unit(0, 0))
    later, _ = _stack_recon(monkeypatch, images, context=context)
    unit = _unit(0, 0)
    unit.data.sample_time = 4e-6

    with caplog.at_level(logging.WARNING, logger="pulserver.recon"):
        (result,) = later.recon(context, "", unit)

    np.testing.assert_array_equal(result.data, images[(0, 0)])
    assert any("phase-encode sampling" in r.getMessage() for r in caplog.records)


def test_a_plane_between_those_of_a_stack_field_takes_their_linear_interpolation():
    from pulserver.recon.handlers.epi import PlaneField

    fields = np.stack([np.zeros((7, 8)), np.ones((7, 8)), 3 * np.ones((7, 8))])
    field = PlaneField(
        positions=np.array([0.0, 2.0, 4.0]),
        fields=fields,
        voxel=(2.0, 2.0),
        slice_dir=(0.0, 0.0, 1.0),
        train=None,
    )

    displacement, reason = field.at([0.5, 3.0, 4.8], None, (0, 0, 1), (6, 8))
    outside, why = field.at([5.5], None, (0, 0, 1), (6, 8))

    assert reason is None
    np.testing.assert_allclose(displacement[:, 0, 0], [0.25, 2.0, 3.0])
    assert outside is None and "outside" in why


def _slab_maps(context, polarity, centre_mm, partitions):
    from pulserver.recon._calibration import CoilSensitivities

    maps = np.zeros((2, partitions, 6, 8), dtype=np.complex64)
    maps[:] = (np.arange(partitions) + 10 * polarity)[None, :, None, None]
    key = (0, polarity) if polarity else 0
    context.coil_maps[key] = CoilSensitivities(
        maps=maps,
        coils=("0", "1"),
        noise=None,
        basis=None,
        geometry={
            "position": (0.0, 0.0, centre_mm),
            "slice_dir": (0.0, 0.0, 1.0),
            "extent": (2.0 * partitions, 200.0, 200.0),
        },
        method="stand-in",
        source=None,
    )


def _plane(position_mm, set_index):
    unit = _unit(0, set_index)
    unit.data.reference.position = (0.0, 0.0, position_mm)
    unit.data.image_shape = (6, 8)
    return unit


def test_a_plane_on_a_slab_partition_takes_that_partitions_maps_of_its_polarity():
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers.epi import FIELD_PAIR_SETS, slab_plane_maps

    context = ReconContext.offline()
    for polarity in (0, 1):
        _slab_maps(context, polarity, centre_mm=10.0, partitions=4)
    forward, reversed_ = FIELD_PAIR_SETS

    # Partition p of 4 lies at 10 + (p - 2) * 2 mm.
    assert slab_plane_maps(context, _plane(8.0, forward))[0, 0, 0, 0] == 1
    assert slab_plane_maps(context, _plane(12.0, reversed_))[0, 0, 0, 0] == 13
    assert slab_plane_maps(context, _plane(9.0, forward)) is None
    assert slab_plane_maps(context, _plane(20.0, forward)) is None


def test_field_pair_planes_read_before_the_slab_calibration_wait_for_its_maps(
    monkeypatch, correction
):
    from pulserver.recon import ReconContext
    from pulserver.recon.handlers import epi

    context = ReconContext.offline()
    plugin = epi.EpiRecon(
        gradient_unwarp=False, intensity_correction=False, kspace_filter=False
    )
    plugin.startup(context)
    plugin.hysco = None
    images = _images(1, polarities=(0, 2, 3))
    unfolded = []

    def image(kspace, shape, context_, data):
        key = (data.counters["slice"], data.counters["set"])
        if key[1] == 0:
            _slab_maps(context, 0, centre_mm=0.0, partitions=4)
        else:
            unfolded.append(int(epi.slab_plane_maps(context, data)[0, 0, 0, 0]))
        return images[key]

    monkeypatch.setattr(plugin, "image", image)
    monkeypatch.setattr(epi, "averaged", lambda buffer: buffer.kspace)
    monkeypatch.setattr(epi, "coil_maps", lambda *args, **kwargs: None)

    def unit(set_index):
        data = _plane(0.0, set_index)
        data.ref = None
        data.data.kspace = np.zeros((2, 6, 8), dtype=np.complex64)
        return data

    assert plugin.recon(context, "", unit(2)) is None
    assert plugin.recon(context, "", unit(3)) is None
    assert unfolded == []
    (result,) = plugin.recon(context, "", unit(0))

    assert unfolded == [2, 2]
    assert len(correction.calls) == 1
    assert "HYSCO" in result.attributes["ImageProcessingHistory"]


@pytest.mark.parametrize("unit_set", [3, 5])
def test_a_reversed_volume_of_a_slab_is_unfolded_by_the_slab_calibrations_reversed_maps(
    unit_set,
):
    data = ReconData("", counters={"slice": 0, "set": unit_set})
    volume = np.zeros((2, 4, 8, 8), np.complex64)

    assert _polarity_of(data, volume).counters["set"] == 1
    assert data.counters["set"] == unit_set


def test_a_forward_or_single_plane_unit_keeps_its_own_set_for_its_maps():
    forward = ReconData("", counters={"slice": 0, "set": 2})
    plane = ReconData("", counters={"slice": 0, "set": 3})

    assert _polarity_of(forward, np.zeros((2, 4, 8, 8))) is forward
    assert _polarity_of(plane, np.zeros((2, 1, 8, 8))) is plane
