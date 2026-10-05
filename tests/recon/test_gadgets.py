"""The readout gadgets that bring each Cartesian readout to a centred full echo on the reconstruction's readout, against arrays built here."""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver import recon

COILS = 2
LINES = 4
FOV_MM = 220.0
OFF_GRID = [
    ismrmrd.ACQ_IS_NAVIGATION_DATA,
    ismrmrd.ACQ_IS_RTFEEDBACK_DATA,
    ismrmrd.ACQ_IS_NOISE_MEASUREMENT,
]


@pytest.fixture
def bartorch():
    return pytest.importorskip("bartorch")


needs_bartorch = pytest.mark.usefixtures("bartorch")


def _space(columns, oversampling, lines=LINES):
    """An encoding space whose readout is sampled ``oversampling`` times over the reconstruction's."""

    def fov(x):
        return SimpleNamespace(x=x, y=FOV_MM, z=5.0)

    return SimpleNamespace(
        encodedSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(x=round(oversampling * columns), y=lines, z=1),
            fieldOfView_mm=fov(oversampling * FOV_MM),
        ),
        reconSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(x=columns, y=lines, z=1),
            fieldOfView_mm=fov(FOV_MM),
        ),
        encodingLimits=SimpleNamespace(
            kspace_encoding_step_1=SimpleNamespace(
                minimum=0, maximum=lines - 1, center=lines // 2
            )
        ),
        trajectory="cartesian",
    )


def _header(*spaces):
    return SimpleNamespace(
        encoding=list(spaces),
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def _readout(samples, centre, flag=None, pre=0, post=0, space=0, line=0):
    """The acquisition of a readout of ``samples`` samples with its echo at ``centre``."""
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(samples, COILS)
    acquisition.center_sample = centre
    acquisition.discard_pre = pre
    acquisition.discard_post = post
    acquisition.encoding_space_ref = space
    acquisition.idx.kspace_encode_step_1 = line
    if flag is not None:
        acquisition.setFlag(flag)
    return acquisition


def _started(gadget, *spaces):
    gadget.startup(recon.ReconContext.offline(_header(*spaces)))
    return gadget


def _centred(transform, data):
    return np.fft.fftshift(transform(np.fft.ifftshift(data, axes=-1)), axes=-1)


def _profile(readout):
    """The image profile of a centred readout under the unitary transform."""
    return _centred(np.fft.ifft, readout) * np.sqrt(readout.shape[-1])


def _readout_of(profile):
    """The centred readout whose unitary profile is ``profile``."""
    return _centred(np.fft.fft, profile) / np.sqrt(profile.shape[-1])


def _complex(shape, seed=0):
    generator = np.random.default_rng(seed)
    return generator.standard_normal(shape) + 1j * generator.standard_normal(shape)


# --------------------------------------------------------------------------
# AsymmetricEcho
# --------------------------------------------------------------------------


def test_an_echo_missing_its_start_is_zero_filled_before_the_readout():
    acquisition = _readout(5, 1)
    data = np.arange(1, 6, dtype=complex) * np.ones((COILS, 1))

    full = recon.AsymmetricEcho()(acquisition, data)

    np.testing.assert_array_equal(full[0].real, [0, 0, 0, 1, 2, 3, 4, 5])
    assert full.shape == (COILS, 8)
    assert (acquisition.discard_pre, acquisition.discard_post) == (3, 0)
    assert acquisition.center_sample == 4


def test_an_echo_missing_its_end_is_zero_filled_after_the_readout():
    acquisition = _readout(5, 3)
    data = np.arange(1, 6, dtype=complex) * np.ones((COILS, 1))

    full = recon.AsymmetricEcho()(acquisition, data)

    np.testing.assert_array_equal(full[0].real, [1, 2, 3, 4, 5, 0])
    assert (acquisition.discard_pre, acquisition.discard_post) == (0, 1)
    assert acquisition.center_sample == 3


@pytest.mark.parametrize(
    ("samples", "centre"),
    [(5, 1), (5, 3), (8, 2), (8, 6), (31, 14), (31, 16), (6, 0), (6, 9)],
)
def test_the_completed_echo_is_symmetric_about_the_sample_at_its_echo(samples, centre):
    """The readout's echo is at the middle of the full echo, which has as many
    samples on each side of it as the longer side of the readout, and the
    acquired samples keep their positions relative to it."""
    acquisition = _readout(samples, centre)
    data = _complex((COILS, samples))

    full = recon.AsymmetricEcho()(acquisition, data)

    first = acquisition.discard_pre
    assert full.shape[-1] == 2 * max(centre, samples - centre)
    assert acquisition.center_sample == full.shape[-1] // 2
    assert first + centre == acquisition.center_sample
    kept = slice(first, full.shape[-1] - acquisition.discard_post)
    np.testing.assert_array_equal(full[:, kept], data)


@pytest.mark.parametrize(("samples", "centre"), [(8, 4), (7, 3), (64, 32)])
def test_a_readout_already_centred_passes_unchanged(samples, centre):
    acquisition = _readout(samples, centre)
    data = _complex((COILS, samples))

    assert recon.AsymmetricEcho()(acquisition, data) is data
    assert (acquisition.discard_pre, acquisition.discard_post) == (0, 0)
    assert acquisition.center_sample == centre


def test_the_zeros_are_added_to_the_samples_a_readout_already_discards():
    acquisition = _readout(5, 1, pre=1, post=1)

    recon.AsymmetricEcho()(acquisition, np.ones((COILS, 5), dtype=complex))

    assert (acquisition.discard_pre, acquisition.discard_post) == (4, 1)


def test_an_acquisition_that_states_no_echo_is_a_centred_full_echo():
    acquisition = SimpleNamespace(flags=0)
    data = _complex((COILS, 5))

    assert recon.AsymmetricEcho()(acquisition, data) is data
    assert not hasattr(acquisition, "center_sample")


@pytest.mark.parametrize("flag", OFF_GRID)
def test_navigator_and_noise_readouts_are_not_completed(flag):
    acquisition = _readout(5, 1, flag)
    data = _complex((COILS, 5))

    assert recon.AsymmetricEcho()(acquisition, data) is data
    assert acquisition.center_sample == 1


def test_a_readout_whose_trajectory_moves_along_more_than_one_axis_is_not_completed():
    acquisition = _readout(5, 1)
    acquisition.resize(5, COILS, 3)
    data = _complex((COILS, 5))

    assert recon.AsymmetricEcho()(acquisition, data) is data
    assert acquisition.center_sample == 1


# --------------------------------------------------------------------------
# RemoveReadoutOversampling
# --------------------------------------------------------------------------


@needs_bartorch
def test_the_cropped_readout_is_the_central_part_of_the_image_profile():
    profile = _complex((COILS, 64))
    gadget = _started(recon.RemoveReadoutOversampling(), _space(32, 2.0))

    cropped = gadget(_readout(64, 32), _readout_of(profile))

    assert cropped.shape == (COILS, 32)
    np.testing.assert_allclose(_profile(cropped), profile[:, 16:48], atol=1e-4)


@needs_bartorch
def test_a_point_inside_the_reconstruction_field_of_view_is_kept_where_it_is():
    """The readout of a point object, sampled at the reconstruction's spacing,
    is the decimated oversampled readout: the same phase ramp in half the
    samples."""
    pixel, full, kept = 5, 64, 32
    readout = np.exp(-2j * np.pi * (np.arange(full) - full // 2) * pixel / full)
    gadget = _started(recon.RemoveReadoutOversampling(), _space(kept, 2.0))

    cropped = gadget(_readout(full, full // 2), readout[None])[0]

    expected = np.exp(-2j * np.pi * (np.arange(kept) - kept // 2) * pixel / kept)
    np.testing.assert_allclose(cropped / cropped[kept // 2], expected, atol=1e-4)


@needs_bartorch
@pytest.mark.parametrize(
    ("oversampling", "samples", "kept"),
    [(2.0, 64, 32), (1.5, 48, 32), (4.0, 128, 32), (2.5, 80, 32)],
)
def test_the_ratio_of_the_fields_of_view_sets_the_samples_kept(
    oversampling, samples, kept
):
    gadget = _started(recon.RemoveReadoutOversampling(), _space(kept, oversampling))
    acquisition = _readout(samples, samples // 2)

    cropped = gadget(acquisition, _complex((COILS, samples)))

    assert cropped.shape == (COILS, kept)
    assert acquisition.center_sample == kept // 2


@needs_bartorch
def test_the_echo_and_the_discards_are_divided_by_the_ratio():
    gadget = _started(recon.RemoveReadoutOversampling(), _space(32, 2.0))
    acquisition = _readout(64, 32, pre=16, post=8)

    gadget(acquisition, _complex((COILS, 64)))

    assert acquisition.center_sample == 16
    assert (acquisition.discard_pre, acquisition.discard_post) == (8, 4)


def test_a_readout_that_is_not_a_centred_full_echo_is_refused():
    gadget = _started(recon.RemoveReadoutOversampling(), _space(32, 2.0))

    with pytest.raises(ValueError, match="AsymmetricEcho"):
        gadget(_readout(64, 20), _complex((COILS, 64)))


def test_a_readout_of_a_space_without_oversampling_passes_unchanged():
    gadget = _started(recon.RemoveReadoutOversampling(), _space(32, 1.0))
    data = _complex((COILS, 32))

    assert gadget(_readout(32, 5), data) is data


def test_a_header_stating_no_field_of_view_leaves_the_readout_as_it_is():
    space = _space(32, 2.0)
    space.encodedSpace.fieldOfView_mm = None
    gadget = _started(recon.RemoveReadoutOversampling(), space)
    data = _complex((COILS, 64))

    assert gadget(_readout(64, 32), data) is data


@needs_bartorch
def test_each_encoding_space_is_cropped_by_its_own_ratio():
    gadget = _started(
        recon.RemoveReadoutOversampling(), _space(32, 2.0), _space(32, 1.0)
    )
    data = _complex((COILS, 32))

    assert gadget(_readout(32, 16, space=1), data) is data
    assert gadget(_readout(32, 16, space=0), data).shape == (COILS, 16)


@pytest.mark.parametrize("flag", OFF_GRID)
def test_navigator_and_noise_readouts_keep_their_oversampling(flag):
    gadget = _started(recon.RemoveReadoutOversampling(), _space(32, 2.0))
    data = _complex((COILS, 64))

    assert gadget(_readout(64, 32, flag), data) is data


# --------------------------------------------------------------------------
# Both, in a plugin
# --------------------------------------------------------------------------


class Collect(recon.ReconPlugin):
    """Keeps the data of every unit it is given."""

    def __init__(self, **options):
        super().__init__(**options)
        self.units = []

    def recon(self, context, branch, data):
        self.units.append(data)


def _played(readouts, oversampling=2.0, columns=32):
    """The unit that ``(acquisition, data)`` readouts make up under the two gadgets."""
    template = Collect(
        gadgets=[recon.AsymmetricEcho(), recon.RemoveReadoutOversampling()]
    )
    context = recon.ReconContext.offline(_header(_space(columns, oversampling)))
    plugin = template.spawn()
    plugin.startup(context)
    for acquisition, data in readouts:
        acquisition.data[:] = data
        plugin.receive(acquisition, context)
    plugin.flush(context)
    (unit,) = template.units
    return unit


@needs_bartorch
def test_a_partial_echo_is_completed_cropped_and_placed_without_its_zero_fill():
    """An echo at sample 16 of 48 is the end of a full echo of 64; cropped by
    the oversampling it is a full echo of 32 whose first 8 samples are fill."""
    data = _complex((COILS, 48))

    buffer = _played([(_readout(48, 16), data)]).data

    full = np.zeros((COILS, 64), dtype=complex)
    full[:, 16:] = data
    cropped = _readout_of(_profile(full)[:, 16:48])
    assert buffer.kspace.shape == (COILS, LINES, 32)
    assert buffer.readout == (8, 31)
    assert buffer.center_sample == 16
    np.testing.assert_allclose(buffer.kspace[:, 0, 8:], cropped[:, 8:], atol=1e-4)
    assert not buffer.kspace[:, 0, :8].any()
    assert list(buffer.mask[0]) == [False] * 8 + [True] * 24


@needs_bartorch
def test_the_acquisition_a_unit_lists_is_the_one_that_was_received():
    """The gadgets state the echo of the readout they return on a copy."""
    received = _readout(48, 16)

    unit = _played([(received, _complex((COILS, 48)))])

    assert unit.acquisitions[0] is received
    assert received.center_sample == 16
    assert (received.discard_pre, received.discard_post) == (0, 0)
    placed = unit.data.headers[0]
    assert placed is not received
    assert (placed.center_sample, placed.discard_pre, placed.discard_post) == (16, 8, 0)


@needs_bartorch
def test_a_scan_of_full_echoes_is_cropped_to_the_reconstruction_matrix():
    data = _complex((COILS, 64))

    buffer = _played([(_readout(64, 32), data)]).data

    expected = _readout_of(_profile(data)[:, 16:48])
    assert buffer.kspace.shape[-1] == 32
    np.testing.assert_allclose(buffer.kspace[:, 0], expected, atol=1e-4)
    assert buffer.mask[0].all()


def test_a_scan_without_oversampling_is_placed_whole():
    data = _complex((COILS, 32))

    buffer = _played([(_readout(32, 16), data)], oversampling=1.0).data

    np.testing.assert_array_equal(buffer.kspace[:, 0], data.astype(np.complex64))
