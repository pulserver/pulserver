"""``ReconData`` -- the header lays the buffers out, the counters fill them.

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

from pulserver.recon import (
    ReconBuffer,
    ReconContext,
    ReconData,
    ReconPlugin,
)
from pulserver.mrd import (
    AcquisitionBucket,
    EncodingSpace,
)

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


def space(x=N_X, y=N_X, z=1, recon=None, trajectory="cartesian", **counters):
    return SimpleNamespace(
        encodedSpace=SimpleNamespace(matrixSize=SimpleNamespace(x=x, y=y, z=z)),
        reconSpace=SimpleNamespace(
            matrixSize=SimpleNamespace(**(recon or {"x": x, "y": y, "z": z}))
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
    buffers = ReconData.from_header(None)
    buffers.add(acquire())
    assert len(buffers) == 0


# --------------------------------------------------------------------------
# Where an acquisition lands
# --------------------------------------------------------------------------


def test_each_acquisition_lands_where_its_own_counters_say():
    buffers = ReconData.from_header(header(space(slice=2, contrast=3)))
    for sl in range(2):
        for echo in range(3):
            for line in range(N_X):
                buffers.add(
                    acquire(
                        value=sl * 100 + echo * 10 + line,
                        slice=sl,
                        contrast=echo,
                        kspace_encode_step_1=line,
                    )
                )

    kspace = buffers[0].kspace
    assert buffers[0].axes == ("coil", "slice", "contrast", "phase_encode", "readout")
    assert kspace.shape == (COILS, 2, 3, N_X, N_X)
    assert buffers[0].mask.all()
    expected = np.arange(2)[:, None, None] * 100 + np.arange(3)[None, :, None] * 10
    assert np.array_equal(kspace[0, :, :, :, 0].real, expected + np.arange(N_X))


def test_a_line_that_never_arrives_stays_unsampled():
    """Which is how an undersampled scan is read: not by counting what came,
    but by asking the mask what did."""
    buffers = ReconData.from_header(header(space(y=8)))
    for line in range(0, 8, 2):
        buffers.add(acquire(kspace_encode_step_1=line))

    assert list(buffers[0].mask[:, 0]) == [True, False] * 4


def test_a_partial_echo_is_right_aligned():
    """Truncating the samples before the echo is what a partial echo does, so
    the acquired window ends where a full one would."""
    buffers = ReconData.from_header(header(space(x=8)))
    buffers.add(acquire(samples=5, value=1.0, kspace_encode_step_1=0))

    buffer = buffers[0]
    assert list(buffer.mask[0]) == [False] * 3 + [True] * 5
    assert np.array_equal(buffer.kspace[0, 0].real, [0, 0, 0, 1, 1, 1, 1, 1])


def test_the_echo_position_follows_the_alignment():
    buffers = ReconData.from_header(header(space(x=8)))
    acquisition = acquire(samples=5)
    acquisition.center_sample = 1
    buffers.add(acquisition)
    assert buffers[0].center_sample == 4


def test_readout_oversampling_widens_the_buffer_rather_than_being_refused():
    """The encoded matrix need not have counted the oversampling, and the first
    acquisition is what actually says how wide a readout is."""
    buffers = ReconData.from_header(header(space(x=8)))
    buffers.add(acquire(samples=16, kspace_encode_step_1=0))
    assert buffers[0].kspace.shape[-1] == 16


def test_an_acquisition_that_does_not_fit_says_so():
    buffers = ReconData.from_header(header(space(y=4)))
    with pytest.raises(ValueError, match=r"kspace_encode_step_1=9"):
        buffers.add(acquire(kspace_encode_step_1=9))


# --------------------------------------------------------------------------
# Several encoding spaces
# --------------------------------------------------------------------------


def test_acquisitions_are_routed_by_the_space_they_name():
    """One space per subsequence, and each acquisition says which it belongs
    to -- so a calibration subsequence and an imaging one sort themselves."""
    buffers = ReconData.from_header(header(space(x=8, y=8), space(x=16, y=4)))
    imaging = acquire(samples=8, kspace_encode_step_1=3)
    calibration = acquire(samples=16, kspace_encode_step_1=1)
    calibration.encoding_space_ref = 1
    buffers.add(imaging)
    buffers.add(calibration)

    assert buffers[0].kspace.shape == (COILS, 8, 8)
    assert buffers[1].kspace.shape == (COILS, 4, 16)
    assert buffers[0].mask.sum() == 8
    assert buffers[1].mask.sum() == 16


def test_a_space_nothing_fills_costs_nothing():
    """A navigator space is in the header of every scan that has one, whether
    or not this reconstruction reads it."""
    buffers = ReconData.from_header(header(space(), space(x=1024, y=1024, z=64)))
    buffers.add(acquire(kspace_encode_step_1=0))
    assert set(buffers) == {0}
    assert 1 not in buffers.data


def test_a_space_the_header_never_described_is_refused():
    buffers = ReconData.from_header(header(space()))
    stray = acquire(kspace_encode_step_1=0)
    stray.encoding_space_ref = 7
    with pytest.raises(KeyError, match="no encoding space 7"):
        buffers.add(stray)


# --------------------------------------------------------------------------
# The calibration
# --------------------------------------------------------------------------


def test_a_calibration_line_is_marked_where_it_landed_not_moved_elsewhere():
    """One grid per encoding space: a calibration acquired on a grid of its own
    is a subsequence of its own, and so a space of its own."""
    buffers = ReconData.from_header(header(space(y=8)))
    for line, flags in (
        (2, ()),
        (3, ("ACQ_IS_PARALLEL_CALIBRATION",)),
        (4, ("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",)),
    ):
        buffers.add(acquire(kspace_encode_step_1=line, flags=flags))

    buffer = buffers[0]
    assert (
        list(buffer.mask.any(axis=-1)) == [False, False, True, True, True] + [False] * 3
    )
    assert (
        list(buffer.reference.any(axis=-1)) == [False] * 3 + [True, True] + [False] * 3
    )


def test_calibration_lines_are_available_to_the_image_as_well():
    """A solve that takes an arbitrary sampling mask wants every line that was
    acquired, whichever flag the sequence put on it."""
    buffers = ReconData.from_header(header(space(y=8)))
    for line in range(0, 8, 2):
        buffers.add(acquire(kspace_encode_step_1=line))
    for line in (3, 5):
        buffers.add(
            acquire(kspace_encode_step_1=line, flags=("ACQ_IS_PARALLEL_CALIBRATION",))
        )

    assert list(buffers[0].mask.any(axis=-1)) == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


# --------------------------------------------------------------------------
# Reading a buffer
# --------------------------------------------------------------------------


def test_select_names_the_position_instead_of_counting_axes():
    buffers = ReconData.from_header(header(space(slice=2, contrast=3)))
    for sl in range(2):
        for echo in range(3):
            buffers.add(
                acquire(
                    value=sl * 10 + echo,
                    slice=sl,
                    contrast=echo,
                    kspace_encode_step_1=0,
                )
            )

    kspace, mask = buffers[0].select(slice=1, contrast=2)
    assert kspace.shape == (COILS, N_X, N_X)
    assert mask.shape == (N_X, N_X)
    assert kspace[0, 0, 0].real == 12


def test_an_axis_that_does_not_vary_still_has_a_position_zero():
    """So a plugin writes ``select(slice=index)`` without first asking whether
    the scan has more than one slice."""
    buffers = ReconData.from_header(header(space(slice=1)))
    buffers.add(acquire(kspace_encode_step_1=0))
    kspace, _ = buffers[0].select(slice=0, contrast=0)
    assert kspace.shape == (COILS, N_X, N_X)


def test_selecting_past_the_end_of_a_flat_axis_says_so():
    buffers = ReconData.from_header(header(space(slice=1)))
    buffers.add(acquire(kspace_encode_step_1=0))
    with pytest.raises(IndexError, match="slice=2"):
        buffers[0].select(slice=2)


def test_selecting_something_that_is_not_an_encoding_axis_says_so():
    buffers = ReconData.from_header(header(space(slice=2)))
    buffers.add(acquire(kspace_encode_step_1=0))
    with pytest.raises(KeyError, match="coil"):
        buffers[0].select(coil=0)


def test_an_axis_that_does_not_vary_is_not_an_axis():
    """A two-dimensional scan has no partition axis and a single-slice one no
    slice axis, so a buffer is the array a reconstruction would have built."""
    buffers = ReconData.from_header(header(space(slice=1, z=1)))
    buffers.add(acquire(kspace_encode_step_1=0))
    assert buffers[0].axes == ("coil", "phase_encode", "readout")


def test_the_buffer_names_its_own_axes():
    """So a plugin reads the layout off the buffer rather than knowing it by
    convention."""
    buffers = ReconData.from_header(header(space(slice=2, average=3)))
    buffers.add(acquire(kspace_encode_step_1=0))
    buffer = buffers[0]
    assert dict(zip(buffer.axes, buffer.kspace.shape, strict=True)) == {
        "coil": COILS,
        "slice": 2,
        "average": 3,
        "phase_encode": N_X,
        "readout": N_X,
    }


def test_a_buffer_keeps_the_headers_it_placed():
    buffers = ReconData.from_header(header(space()))
    sent = [acquire(kspace_encode_step_1=line) for line in range(3)]
    for acquisition in sent:
        buffers.add(acquisition)
    assert buffers[0].headers == sent


def test_the_dtype_is_the_callers_to_choose():
    buffers = ReconData.from_header(header(space()), dtype=np.complex128)
    buffers.add(acquire(kspace_encode_step_1=0))
    assert buffers[0].kspace.dtype == np.complex128


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


class Sink(ReconPlugin):
    """A plugin overriding nothing but the reconstruction itself."""

    def recon(self, bucket, context):
        del bucket, context
        return None


def test_a_plugin_that_overrides_nothing_still_gets_sorted_kspace():
    """The point of the whole layer: placing is not a plugin's work."""
    plugin = Sink().spawn()
    context = ReconContext.offline(header(space(slice=2)))
    plugin.startup(context)
    for sl in range(2):
        for line in range(N_X):
            plugin.receive(acquire(slice=sl, kspace_encode_step_1=line), context)

    assert plugin.buffers[0].kspace.shape == (COILS, 2, N_X, N_X)
    assert plugin.buffers[0].mask.all()


def test_replaying_a_bucket_offline_fills_the_same_buffers():
    """A plugin has one behaviour, not a streamed one and an assembled one."""
    seen = {}

    class Recorder(ReconPlugin):
        def recon(self, bucket, context):
            del bucket, context
            seen["mask"] = self.buffers[0].mask.copy()
            return None

    bucket = AcquisitionBucket(
        data=tuple(acquire(kspace_encode_step_1=line) for line in range(N_X))
    )
    Recorder()(bucket, ReconContext.offline(header(space())))
    assert seen["mask"].all()


def test_two_streams_do_not_share_a_buffer():
    """``spawn`` is what keeps concurrent connections apart, and the buffers
    are assigned by the lifecycle, so they are per-stream."""
    template = Sink()
    context = ReconContext.offline(header(space()))
    first, second = template.spawn(), template.spawn()
    first.startup(context)
    second.startup(context)
    first.receive(acquire(kspace_encode_step_1=0), context)

    assert first.buffers[0].mask.any()
    assert 0 not in second.buffers.data
    assert 0 not in template.buffers.data


# --------------------------------------------------------------------------
# Where the samples were taken
# --------------------------------------------------------------------------


def test_a_trajectory_is_placed_beside_the_data():
    """A non-Cartesian scan needs where each sample was taken, and the
    acquisition carries it -- so it is placed the same way the data is."""
    buffers = ReconData.from_header(header(space(x=4, y=3)))
    for view in range(3):
        acquisition = ismrmrd.Acquisition()
        acquisition.resize(4, COILS, 2)
        acquisition.data[:] = view
        acquisition.traj[:] = np.stack([np.full(4, view), np.arange(4)], axis=-1)
        acquisition.idx.kspace_encode_step_1 = view
        buffers.add(acquisition)

    trajectory = buffers[0].trajectory
    assert trajectory.shape == (2, 3, 4)
    assert np.array_equal(trajectory[0, :, 0], [0, 1, 2])
    assert np.array_equal(trajectory[1, 2], [0, 1, 2, 3])


def test_a_cartesian_scan_holds_no_trajectory():
    """Nothing to hold: its acquisitions carry none."""
    buffers = ReconData.from_header(header(space()))
    buffers.add(acquire(kspace_encode_step_1=0))
    assert buffers[0].trajectory is None
