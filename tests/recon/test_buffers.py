"""``ReconBuffer`` and the unit that allocates it -- the header lays the buffers out, the counters fill them.

What is checked here is that a plugin gets sorted k-space without sorting
anything: that the layout comes off the header rather than off the data, that
each acquisition lands where its own counters say, and that the arrangements a
real scan produces -- several encoding spaces, a calibration region, a partial
echo, an undersampled grid -- come out of it read correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver.mrd import (
    LOOP_COUNTERS,
    AcquisitionBucket,
    EncodingSpace,
)
from pulserver.recon import (
    ReconBuffer,
    ReconContext,
    ReconPlugin,
)
from pulserver.recon._buffers import ReconUnit
from pulserver.recon._units import unit_key

N_X = 8
COILS = 4


def limits(**counters):
    """An ``encodingLimits`` stating the extent of each named counter."""
    return SimpleNamespace(
        **{
            name: SimpleNamespace(minimum=0, maximum=extent - 1, center=extent // 2)
            for name, extent in counters.items()
        }
    )


def space(
    x=N_X, y=N_X, z=1, recon=None, trajectory="cartesian", fov_mm=None, **counters
):
    fov = (
        None
        if fov_mm is None
        else SimpleNamespace(**dict(zip("xyz", fov_mm, strict=True)))
    )
    return SimpleNamespace(
        encodedSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(x=x, y=y, z=z), fieldOfView_mm=fov
        ),
        reconSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(**(recon or {"x": x, "y": y, "z": z})),
            fieldOfView_mm=fov,
        ),
        encodingLimits=limits(**counters),
        trajectory=trajectory,
    )


def header(*spaces, coils=COILS):
    return SimpleNamespace(
        encoding=list(spaces),
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=coils),
    )


def acquire(samples=N_X, coils=COILS, value=None, flags=(), **idx):
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(samples, coils)
    acquisition.data[:] = np.arange(samples) if value is None else value
    for name, index in idx.items():
        setattr(acquisition.idx, name, index)
    for flag in flags:
        acquisition.setFlag(getattr(ismrmrd, flag))
    return acquisition


def fill(hdr, *acquisitions, axes=LOOP_COUNTERS, dtype=np.complex64):
    """The data of the unit ``acquisitions`` make up, laid out along ``axes`` as its plugin declares."""
    spaces = {s.index: s for s in EncodingSpace.all_from_header(hdr, axes)}
    unit = ReconUnit(unit_key("imaging", acquisitions[0], axes), spaces, dtype=dtype)
    for acquisition in acquisitions:
        unit.add_acquisition(acquisition, acquisition.data)
    return unit.data


# --------------------------------------------------------------------------
# What the header says
# --------------------------------------------------------------------------


def test_the_layout_is_read_before_any_data_arrives():
    """The whole premise: the header describes every space of the scan, so
    nothing has to be inferred from what turns up."""
    spaces = EncodingSpace.all_from_header(
        header(space(slice=3, contrast=2), space(x=16, y=1))
    )
    assert [s.index for s in spaces] == [0, 1]
    assert spaces[0].axes == (
        "coil",
        "slice",
        "contrast",
        "phase_encode",
        "readout",
    )
    assert spaces[0].shape == (COILS, 3, 2, N_X, N_X)
    # A navigator space is one line: nothing varies but the samples.
    assert spaces[1].axes == ("coil", "readout")
    assert spaces[1].shape == (COILS, 16)


def test_only_the_counters_that_vary_become_axes():
    """A counter the scan never advances is not a dimension of it."""
    one = EncodingSpace.from_header(header(space(slice=1, repetition=1, average=4)))
    assert one.loops == ("average",)


def test_the_loop_axes_nest_outermost_first():
    every = EncodingSpace.from_header(
        header(space(average=2, set=2, contrast=2, slice=2, phase=2, repetition=2))
    )
    assert every.loops == ("repetition", "phase", "slice", "contrast", "set", "average")


def test_segment_is_not_a_loop_axis():
    """It names a piece of one readout train, not a separate image -- splitting
    on it would take an EPI shot apart."""
    assert "segment" not in EncodingSpace.from_header(header(space(segment=4))).loops


def test_segment_is_an_axis_when_it_is_asked_for():
    asked = EncodingSpace.from_header(header(space(segment=4)), loops=("segment",))
    assert (asked.loops, asked.loop_sizes) == (("segment",), (4,))


def test_only_the_counters_asked_for_become_axes():
    """A unit spans the counters its plugin declares, and holds the others fixed."""
    asked = EncodingSpace.from_header(
        header(space(slice=3, contrast=2)), loops=("contrast",)
    )
    assert asked.axes == ("coil", "contrast", "phase_encode", "readout")
    assert EncodingSpace.from_header(header(space(slice=3)), loops=()).loops == ()


def test_a_grid_wider_than_its_sampled_lines_keeps_the_grid():
    """An undersampled Cartesian scan acquires fewer lines than its matrix, and
    the buffer is the matrix -- the gaps are the point."""
    partial = EncodingSpace.from_header(header(space(y=32)))
    assert partial.phase_encodes == 32


@pytest.mark.parametrize("n_views", [201, 17])
def test_a_non_cartesian_space_is_sized_by_its_views_not_its_matrix(n_views):
    """``kspace_encoding_step_1`` counts views there, and a view count bears no
    relation to the image matrix -- it can be either side of it."""
    radial = EncodingSpace.from_header(
        header(space(x=64, y=64, trajectory="radial", kspace_encoding_step_1=n_views))
    )
    assert radial.phase_encodes == n_views


def test_a_cartesian_space_keeps_its_grid_however_few_lines_were_taken():
    """The counterpart: an undersampled grid is still the whole grid, and the
    limit counts only what the scan acquired."""
    accelerated = EncodingSpace.from_header(
        header(space(x=64, y=64, kspace_encoding_step_1=33))
    )
    assert accelerated.phase_encodes == 64


def test_a_header_that_does_not_say_is_read_as_cartesian():
    plain = EncodingSpace.from_header(
        header(space(x=64, y=64, trajectory=None, kspace_encoding_step_1=33))
    )
    assert plain.phase_encodes == 64


def test_a_header_describing_nothing_buffers_nothing():
    """An offline bucket assembled from arrays carries no header, and nothing
    said where its acquisitions go."""
    unit = fill(None, acquire())
    assert unit.data is None
    assert len(unit.acquisitions) == 1


# --------------------------------------------------------------------------
# Where an acquisition lands
# --------------------------------------------------------------------------


def test_each_acquisition_lands_where_its_own_counters_say():
    buffer = fill(
        header(space(slice=2, contrast=3)),
        *(
            acquire(
                value=sl * 100 + echo * 10 + line,
                slice=sl,
                contrast=echo,
                kspace_encode_step_1=line,
            )
            for sl in range(2)
            for echo in range(3)
            for line in range(N_X)
        ),
    ).data

    kspace = buffer.kspace
    assert buffer.axes == ("coil", "slice", "contrast", "phase_encode", "readout")
    assert kspace.shape == (COILS, 2, 3, N_X, N_X)
    assert buffer.mask.all()
    expected = np.arange(2)[:, None, None] * 100 + np.arange(3)[None, :, None] * 10
    assert np.array_equal(kspace[0, :, :, :, 0].real, expected + np.arange(N_X))


def test_a_line_that_never_arrives_stays_unsampled():
    """Which is how an undersampled scan is read: not by counting what came,
    but by asking the mask what did."""
    buffer = fill(
        header(space(y=8)),
        *(acquire(kspace_encode_step_1=line) for line in range(0, 8, 2)),
    ).data

    assert list(buffer.mask[:, 0]) == [True, False] * 4


def test_a_partial_echo_is_right_aligned():
    """Truncating the samples before the echo is what a partial echo does, so
    the acquired window ends where a full one would."""
    buffer = fill(
        header(space(x=8)), acquire(samples=5, value=1.0, kspace_encode_step_1=0)
    ).data

    assert list(buffer.mask[0]) == [False] * 3 + [True] * 5
    assert np.array_equal(buffer.kspace[0, 0].real, [0, 0, 0, 1, 1, 1, 1, 1])


def test_the_echo_position_follows_the_alignment():
    acquisition = acquire(samples=5)
    acquisition.center_sample = 1
    assert fill(header(space(x=8)), acquisition).data.center_sample == 4


def test_readout_oversampling_widens_the_buffer_rather_than_being_refused():
    """The encoded matrix need not have counted the oversampling, and the first
    acquisition is what actually says how wide a readout is."""
    buffer = fill(header(space(x=8)), acquire(samples=16, kspace_encode_step_1=0)).data
    assert buffer.kspace.shape[-1] == 16


def test_an_acquisition_that_does_not_fit_says_so():
    with pytest.raises(ValueError, match=r"kspace_encode_step_1=9"):
        fill(header(space(y=4)), acquire(kspace_encode_step_1=9))


# --------------------------------------------------------------------------
# Several encoding spaces
# --------------------------------------------------------------------------


def test_a_unit_is_laid_out_as_the_space_its_readouts_name():
    """One space per subsequence, and each acquisition says which it belongs
    to -- so a calibration subsequence and an imaging one sort themselves."""
    hdr = header(space(x=8, y=8), space(x=16, y=4))
    imaging = acquire(samples=8, kspace_encode_step_1=3)
    calibration = acquire(samples=16, kspace_encode_step_1=1)
    calibration.encoding_space_ref = 1

    first = fill(hdr, imaging).data
    second = fill(hdr, calibration).data

    assert first.kspace.shape == (COILS, 8, 8)
    assert second.kspace.shape == (COILS, 4, 16)
    assert first.mask.sum() == 8
    assert second.mask.sum() == 16


def test_a_buffer_is_allocated_by_the_first_readout_it_holds():
    """A navigator space is in the header of every scan that has one, whether
    or not this reconstruction reads it: nothing is allocated for it."""
    hdr = header(space(), space(x=1024, y=1024, z=64))
    spaces = {s.index: s for s in EncodingSpace.all_from_header(hdr, ())}
    readout = acquire(kspace_encode_step_1=0)
    unit = ReconUnit(unit_key("imaging", readout, ()), spaces)

    assert unit.data.data is None
    unit.add_acquisition(readout, readout.data)
    assert unit.data.data.space.index == 0


def test_a_space_the_header_never_described_is_refused():
    stray = acquire(kspace_encode_step_1=0)
    stray.encoding_space_ref = 7
    with pytest.raises(KeyError, match="no encoding space 7"):
        fill(header(space()), stray)


# --------------------------------------------------------------------------
# The calibration
# --------------------------------------------------------------------------


def _lines(*lines):
    """The data of one unit of an eight-line grid holding ``(line, flags)`` readouts."""
    return fill(
        header(space(y=8)),
        *(acquire(kspace_encode_step_1=line, flags=flags) for line, flags in lines),
    )


def _sampled(buffer):
    return [] if buffer is None else list(np.flatnonzero(buffer.mask.any(axis=-1)))


def test_a_calibration_readout_is_reference_only():
    """The imaging data is what was acquired to be imaged; a line acquired only
    to calibrate a coil sensitivity is not part of it."""
    unit = _lines((2, ()), (3, ("ACQ_IS_PARALLEL_CALIBRATION",)))

    assert _sampled(unit.data) == [2]
    assert _sampled(unit.ref) == [3]


def test_a_calibration_and_imaging_readout_is_in_both():
    unit = _lines((2, ()), (4, ("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",)))

    assert _sampled(unit.data) == [2, 4]
    assert _sampled(unit.ref) == [4]


def test_a_phase_correction_readout_is_in_neither():
    unit = _lines((2, ()), (3, ("ACQ_IS_PHASECORR_DATA",)))

    assert _sampled(unit.data) == [2]
    assert unit.ref is None


def test_a_unit_without_calibration_has_no_reference_buffer():
    assert _lines((2, ())).ref is None


def test_a_unit_that_places_nothing_still_lists_what_it_received():
    """What is not placed is still what the unit was given."""
    unit = _lines((3, ("ACQ_IS_PHASECORR_DATA",)))

    assert unit.data is None
    assert len(unit.acquisitions) == 1


# --------------------------------------------------------------------------
# Reading a buffer
# --------------------------------------------------------------------------


def test_select_names_the_position_instead_of_counting_axes():
    buffer = fill(
        header(space(slice=2, contrast=3)),
        *(
            acquire(
                value=sl * 10 + echo, slice=sl, contrast=echo, kspace_encode_step_1=0
            )
            for sl in range(2)
            for echo in range(3)
        ),
    ).data

    kspace, mask = buffer.select(slice=1, contrast=2)
    assert kspace.shape == (COILS, N_X, N_X)
    assert mask.shape == (N_X, N_X)
    assert kspace[0, 0, 0].real == 12


def test_an_axis_that_does_not_vary_still_has_a_position_zero():
    """So a plugin writes ``select(slice=index)`` without first asking whether
    the scan has more than one slice."""
    buffer = fill(header(space(slice=1)), acquire(kspace_encode_step_1=0)).data
    kspace, _ = buffer.select(slice=0, contrast=0)
    assert kspace.shape == (COILS, N_X, N_X)


def test_selecting_past_the_end_of_a_flat_axis_says_so():
    buffer = fill(header(space(slice=1)), acquire(kspace_encode_step_1=0)).data
    with pytest.raises(IndexError, match="slice=2"):
        buffer.select(slice=2)


def test_selecting_something_that_is_not_an_encoding_axis_says_so():
    buffer = fill(header(space(slice=2)), acquire(kspace_encode_step_1=0)).data
    with pytest.raises(KeyError, match="coil"):
        buffer.select(coil=0)


def test_an_axis_that_does_not_vary_is_not_an_axis():
    """A two-dimensional scan has no partition axis and a single-slice one no
    slice axis, so a buffer is the array a reconstruction would have built."""
    buffer = fill(header(space(slice=1, z=1)), acquire(kspace_encode_step_1=0)).data
    assert buffer.axes == ("coil", "phase_encode", "readout")


def test_the_buffer_names_its_own_axes():
    """So a plugin reads the layout off the buffer rather than knowing it by
    convention."""
    buffer = fill(
        header(space(slice=2, average=3)), acquire(kspace_encode_step_1=0)
    ).data
    assert dict(zip(buffer.axes, buffer.kspace.shape, strict=True)) == {
        "coil": COILS,
        "slice": 2,
        "average": 3,
        "phase_encode": N_X,
        "readout": N_X,
    }


def test_a_buffer_keeps_the_headers_it_placed():
    sent = [acquire(kspace_encode_step_1=line) for line in range(3)]
    unit = fill(header(space()), *sent)
    assert unit.data.headers == sent
    assert unit.acquisitions == sent


def test_the_dtype_is_the_callers_to_choose():
    buffer = fill(
        header(space()), acquire(kspace_encode_step_1=0), dtype=np.complex128
    ).data
    assert buffer.kspace.dtype == np.complex128


def test_a_buffer_can_be_laid_out_without_a_header():
    """The header is the usual source, not the only one -- a test or an offline
    caller states the space directly."""
    buffer = ReconBuffer(
        EncodingSpace(
            index=0,
            coils=2,
            readout=4,
            phase_encodes=4,
            partitions=1,
            loops=(),
            loop_sizes=(),
            recon_matrix=(4, 4),
        )
    )
    buffer.add(acquire(samples=4, coils=2, kspace_encode_step_1=2))
    assert buffer.kspace.shape == (2, 4, 4)
    assert buffer.mask[2].all()


# --------------------------------------------------------------------------
# The default plugin lifecycle
# --------------------------------------------------------------------------


class Collect(ReconPlugin):
    """A plugin overriding nothing but the reconstruction itself, which keeps what it is given."""

    def __init__(self, **options):
        super().__init__(**options)
        self.units = []

    def recon(self, context, branch, data):
        del context, branch
        self.units.append(data)


def test_a_plugin_that_overrides_nothing_still_gets_sorted_kspace():
    """The point of the whole layer: placing is not a plugin's work."""
    template = Collect(axes=("slice",))
    plugin = template.spawn()
    context = ReconContext.offline(header(space(slice=2)))
    plugin.startup(context)
    for sl in range(2):
        for line in range(N_X):
            plugin.receive(acquire(slice=sl, kspace_encode_step_1=line), context)
    plugin.flush(context)

    (unit,) = template.units
    assert unit.data.kspace.shape == (COILS, 2, N_X, N_X)
    assert unit.data.mask.all()


def test_replaying_a_bucket_offline_fills_the_same_buffers():
    """A plugin has one behaviour, not a streamed one and an assembled one."""
    recorder = Collect()
    bucket = AcquisitionBucket(
        data=tuple(acquire(kspace_encode_step_1=line) for line in range(N_X))
    )

    recorder(bucket, ReconContext.offline(header(space())))

    (unit,) = recorder.units
    assert unit.data.mask.all()


def test_two_streams_do_not_share_a_unit():
    """``spawn`` is what keeps concurrent connections apart, and the units
    are created by the lifecycle, so they are per-stream."""
    template = Collect()
    context = ReconContext.offline(header(space()))
    first, second = template.spawn(), template.spawn()
    first.startup(context)
    second.startup(context)
    first.receive(acquire(kspace_encode_step_1=0), context)

    assert len(first._units) == 1
    assert not second._units
    assert not template._units


# --------------------------------------------------------------------------
# Where the samples were taken
# --------------------------------------------------------------------------


def test_a_trajectory_is_placed_beside_the_data():
    """A non-Cartesian scan needs where each sample was taken, and the
    acquisition carries it -- so it is placed the same way the data is."""
    views = []
    for view in range(3):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(4, COILS, 2)
        acquisition.data[:] = view
        acquisition.traj[:] = np.stack([np.full(4, view), np.arange(4)], axis=-1)
        acquisition.idx.kspace_encode_step_1 = view
        views.append(acquisition)

    trajectory = fill(header(space(x=4, y=3)), *views).data.trajectory
    assert trajectory.shape == (2, 3, 4)
    assert np.array_equal(trajectory[0, :, 0], [0, 1, 2])
    assert np.array_equal(trajectory[1, 2], [0, 1, 2, 3])


def test_a_cartesian_scan_holds_no_trajectory():
    """Nothing to hold: its acquisitions carry none."""
    assert (
        fill(header(space()), acquire(kspace_encode_step_1=0)).data.trajectory is None
    )


def test_an_axis_a_readout_never_traversed_reads_back_as_the_zero_it_was():
    """MRD leaves a trailing axis off a readout that does not traverse it.

    The centre partition of a Cartesian slab traverses no kz and says so with
    two dimensions where its neighbours say three, so what a buffer holds is
    as wide as the widest acquisition placed in it either way, and the axis
    the narrow one left off reads back as zero.
    """
    for order in ([3, 2], [2, 3]):
        views = []
        for view, dimensions in enumerate(order):
            acquisition = ismrmrd.Acquisition()
            acquisition.resize(4, COILS, dimensions)
            acquisition.traj[:] = np.arange(1, 4 * dimensions + 1).reshape(
                4, dimensions
            )
            acquisition.idx.kspace_encode_step_1 = view
            views.append(acquisition)

        trajectory = fill(header(space(x=4, y=3)), *views).data.trajectory
        assert trajectory.shape == (3, 3, 4)
        narrow = order.index(2)
        assert np.array_equal(trajectory[2, narrow], np.zeros(4))
        assert np.array_equal(trajectory[1, narrow], [2, 4, 6, 8])


def test_the_field_of_view_is_read_in_metres_in_the_order_of_the_matrix():
    volume = EncodingSpace.from_header(header(space(z=4, fov_mm=(220, 200, 40))))
    plane = EncodingSpace.from_header(header(space(fov_mm=(220, 200, 5))))
    assert volume.recon_fov == pytest.approx((0.04, 0.2, 0.22))
    assert plane.recon_fov == pytest.approx((0.2, 0.22))
    assert EncodingSpace.from_header(header(space())).recon_fov is None


def radial(position, n=32, spokes=24, fov=0.2):
    """A radial buffer of a point at ``position`` (m), with k in 1/m as enrichment writes it."""
    radius = (np.arange(n) - n // 2) / fov
    views = []
    for view in range(spokes):
        angle = np.pi * view / spokes
        k = np.stack([np.cos(angle) * radius, np.sin(angle) * radius], axis=-1)
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(n, 1, 2)
        acquisition.traj[:] = k
        acquisition.data[:] = np.exp(-2j * np.pi * (k @ np.asarray(position)))
        acquisition.idx.kspace_encode_step_1 = view
        views.append(acquisition)
    return fill(
        header(
            space(
                x=n,
                y=n,
                trajectory="radial",
                fov_mm=(1e3 * fov, 1e3 * fov, 5.0),
                kspace_encoding_step_1=spokes,
            )
        ),
        *views,
    ).data


#: A point 5 pixels along x and -3 along y from the centre of a 32-point, 0.2 m grid.
POINT_M, POINT_PIXEL = (5 * 0.2 / 32, -3 * 0.2 / 32), (16 - 3, 16 + 5)


def test_a_point_is_found_where_it_was_placed_on_the_grid_trajectory():
    buffer = radial(POINT_M)
    grid = buffer.grid_trajectory()
    assert grid.shape == (24, 32, 3)
    assert not grid[..., 2].any()
    pixels = np.arange(32) - 16
    phase = (
        pixels[:, None, None] * grid[..., 1].ravel()
        + pixels[None, :, None] * grid[..., 0].ravel()
    )
    image = np.exp(2j * np.pi * phase / 32) @ buffer.kspace[0].ravel()
    assert np.unravel_index(np.abs(image).argmax(), image.shape) == POINT_PIXEL


def test_the_grid_trajectory_is_what_bartorch_nufft_takes():
    torch = pytest.importorskip("torch")
    linop = pytest.importorskip("bartorch.linop")
    buffer = radial(POINT_M)
    operator = linop.NUFFT(
        torch.from_numpy(buffer.grid_trajectory()), image_shape=buffer.image_shape
    )
    image = operator.adjoint(torch.from_numpy(buffer.kspace[0])).numpy()
    assert np.unravel_index(np.abs(image).argmax(), image.shape) == POINT_PIXEL


def test_a_trajectory_without_its_field_of_view_is_refused():
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(4, COILS, 2)
    buffer = fill(header(space(x=4, y=3)), acquisition).data
    with pytest.raises(ValueError, match="field of view"):
        buffer.grid_trajectory()


def test_a_readout_placed_over_another_warns_that_the_sequence_does_not_label_them():
    buffer = ReconBuffer(EncodingSpace.from_header(header(space())))
    buffer.add(acquire(kspace_encode_step_1=0))
    with pytest.warns(UserWarning, match="make_label"):
        buffer.add(acquire(kspace_encode_step_1=0))


def test_readouts_at_distinct_lines_place_without_a_warning():
    import warnings

    buffer = ReconBuffer(EncodingSpace.from_header(header(space())))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for line in range(N_X):
            buffer.add(acquire(kspace_encode_step_1=line))
