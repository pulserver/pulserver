"""``ReconBuffer`` and the unit that allocates it -- the header lays the buffers out, the counters fill them.

What is checked here is that a plugin gets sorted k-space without sorting
anything: that the layout comes off the header rather than off the data, that
each acquisition lands where its echo and its counters, taken about the
centre the header's limits state, say, and that the arrangements a real scan
produces -- several encoding spaces, a calibration region, a partial echo, an
undersampled or partial-Fourier grid -- come out of it read correctly.
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
from pulserver.recon._buffers import ReconUnit, nearest_to_centre
from pulserver.recon._units import unit_key

N_X = 8
COILS = 4


def limits(**counters):
    """An ``encodingLimits`` stating each named counter's extent, or its ``(minimum, maximum, center)``."""
    return SimpleNamespace(
        **{
            name: SimpleNamespace(
                **dict(
                    zip(
                        ("minimum", "maximum", "center"),
                        (0, extent - 1, extent // 2)
                        if isinstance(extent, int)
                        else extent,
                        strict=True,
                    )
                )
            )
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


def acquire(samples=N_X, coils=COILS, value=None, flags=(), centre=None, **idx):
    """A readout of ``samples`` samples whose echo is at ``centre``, by default a centred full echo."""
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(samples, coils)
    acquisition.data[:] = np.arange(samples) if value is None else value
    acquisition.center_sample = samples // 2 if centre is None else centre
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
    return unit.close()


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


def test_a_non_cartesian_volume_stating_one_partition_lays_out_none():
    """A trajectory that encodes kz itself writes no partition counter, so its
    matrix z is not a stack."""
    kooshball = EncodingSpace.from_header(
        header(
            space(
                x=64,
                y=64,
                z=64,
                trajectory="radial",
                kspace_encoding_step_1=200,
                kspace_encoding_step_2=1,
            )
        )
    )
    stack = EncodingSpace.from_header(
        header(space(x=64, y=64, z=16, trajectory="radial", kspace_encoding_step_1=8))
    )
    assert kooshball.partitions == 1
    assert stack.partitions == 16


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


def test_an_echo_missing_its_start_is_placed_by_its_centre():
    """The echo is at sample 1 of 5, so the readout is the end of the full
    echo of 8 samples whose echo is at 4: it occupies samples 3 to 7."""
    buffer = fill(
        header(space(x=8)),
        acquire(samples=5, centre=1, value=np.arange(1, 6), kspace_encode_step_1=0),
    ).data

    assert list(buffer.mask[0]) == [False] * 3 + [True] * 5
    assert np.array_equal(buffer.kspace[0, 0].real, [0, 0, 0, 1, 2, 3, 4, 5])
    assert (buffer.readout, buffer.center_sample) == ((3, 7), 4)


def test_an_echo_missing_its_end_is_placed_by_its_centre():
    """The echo is at sample 3 of 5, so the readout is the start of the full
    echo of 6 samples whose echo is at 3: it occupies samples 0 to 4."""
    buffer = fill(
        header(space(x=6)),
        acquire(samples=5, centre=3, value=np.arange(1, 6), kspace_encode_step_1=0),
    ).data

    assert list(buffer.mask[0]) == [True] * 5 + [False]
    assert np.array_equal(buffer.kspace[0, 0].real, [1, 2, 3, 4, 5, 0])
    assert (buffer.readout, buffer.center_sample) == ((0, 4), 3)


def test_every_echo_is_placed_with_its_centre_at_the_middle_of_the_buffer():
    """Whichever side of the echo a readout is missing, its echo is at the same
    sample, so lines of different partial echoes line up."""
    centred, early, late = (
        acquire(samples=8, centre=4, kspace_encode_step_1=0),
        acquire(samples=6, centre=2, kspace_encode_step_1=1),
        acquire(samples=6, centre=4, kspace_encode_step_1=2),
    )
    buffer = fill(header(space(x=8, y=3)), centred, early, late).data

    # The echo of each line is at sample 4, which holds the sample at its centre.
    np.testing.assert_array_equal(buffer.kspace[0, :, 4].real, [4, 2, 4])
    assert [list(np.flatnonzero(row)) for row in buffer.mask] == [
        list(range(8)),
        list(range(2, 8)),
        list(range(6)),
    ]
    assert buffer.readout == (0, 7)


def test_discarded_samples_are_not_placed():
    """The samples a readout declares discarded are left where the buffer has
    zero fill, and the rest keep their places about the echo."""
    acquisition = acquire(
        samples=8, centre=4, value=np.arange(1, 9), kspace_encode_step_1=0
    )
    acquisition.discard_pre = 2
    acquisition.discard_post = 1

    buffer = fill(header(space(x=8)), acquisition).data

    assert list(buffer.mask[0]) == [False] * 2 + [True] * 5 + [False]
    assert np.array_equal(buffer.kspace[0, 0].real, [0, 0, 3, 4, 5, 6, 7, 0])
    assert buffer.readout == (2, 6)


def test_a_discarded_trajectory_is_cut_with_the_samples():
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(8, COILS, 1)
    acquisition.center_sample = 4
    acquisition.discard_pre = 2
    acquisition.discard_post = 1
    acquisition.traj[:] = np.arange(8, dtype=np.float32)[:, None]
    acquisition.idx.kspace_encode_step_1 = 0

    buffer = fill(header(space(x=8)), acquisition).data

    assert np.array_equal(buffer.trajectory[0, 0], [0, 0, 2, 3, 4, 5, 6, 0])


def test_an_echo_that_leaves_the_buffer_is_refused():
    """An echo at sample 1 of 5 needs a full echo of 8: a buffer of 6 cannot hold it."""
    with pytest.raises(ValueError, match="does not fit the 6 samples"):
        fill(
            header(space(x=6)),
            acquire(samples=5, centre=1, kspace_encode_step_1=0),
        )


def test_a_buffer_is_as_wide_as_the_first_readout_when_that_is_a_centred_full_echo():
    """A readout completed to its full echo, oversampling included, is the
    echo whatever width the header's encoded matrix states."""
    unit = fill(
        header(space(x=8)),
        acquire(samples=16, kspace_encode_step_1=0),
        acquire(samples=12, centre=6, kspace_encode_step_1=1),
    )

    assert unit.data.kspace.shape[-1] == 16
    assert list(unit.data.mask[1]) == [False] * 2 + [True] * 12 + [False] * 2


def test_a_buffer_is_as_wide_as_the_encoded_matrix_when_the_first_readout_is_a_partial_echo():
    unit = fill(
        header(space(x=16)),
        acquire(samples=12, centre=4, kspace_encode_step_1=0),
    )

    assert unit.data.kspace.shape[-1] == 16


def test_readout_oversampling_widens_the_buffer_rather_than_being_refused():
    """The encoded matrix need not have counted the oversampling, and the first
    acquisition is what actually says how wide a readout is."""
    buffer = fill(header(space(x=8)), acquire(samples=16, kspace_encode_step_1=0)).data
    assert buffer.kspace.shape[-1] == 16


def test_an_acquisition_that_does_not_fit_says_so():
    with pytest.raises(ValueError, match=r"kspace_encode_step_1=9"):
        fill(header(space(y=4)), acquire(kspace_encode_step_1=9))


def _lines_at(buffer):
    return [int(line) for line in np.flatnonzero(buffer.mask[:, 0])]


def test_lines_are_placed_by_the_limits_centre():
    """The counter of the k-space centre lands at the middle of the grid,
    wherever the counters start and however far the centre is from the middle
    of the lines acquired."""
    unit = fill(
        header(space(y=16, kspace_encoding_step_1=(3, 12, 7))),
        *(acquire(value=line, kspace_encode_step_1=line) for line in range(3, 13)),
    )

    assert _lines_at(unit.data) == list(range(4, 14))
    np.testing.assert_array_equal(
        unit.data.kspace[0, :, 0].real, [0] * 4 + list(range(3, 13)) + [0] * 2
    )
    assert unit.data.kspace[0, 16 // 2, 0] == 7


def test_a_partial_fourier_scan_keeps_its_lines_where_the_centre_puts_them():
    """Lines from the centre to the edge of the grid are the half of it
    beyond the centre, however the counters are numbered."""
    unit = fill(
        header(space(y=16, kspace_encoding_step_1=(8, 15, 8))),
        *(acquire(kspace_encode_step_1=line) for line in range(8, 16)),
    )

    assert _lines_at(unit.data) == list(range(8, 16))


def test_a_scan_whose_counters_start_at_one_is_placed_by_its_centre():
    unit = fill(
        header(space(y=8, kspace_encoding_step_1=(1, 8, 5))),
        *(acquire(kspace_encode_step_1=line) for line in range(1, 9)),
    )

    assert _lines_at(unit.data) == list(range(8))


def test_partitions_are_placed_by_the_limits_centre_in_a_volume():
    unit = fill(
        header(space(y=4, z=8, kspace_encoding_step_2=(0, 6, 3))),
        *(
            acquire(value=partition, kspace_encode_step_2=partition)
            for partition in range(7)
        ),
    )

    assert unit.data.axes == ("coil", "partition", "phase_encode", "readout")
    np.testing.assert_array_equal(
        unit.data.kspace[0, :, 0, 0].real, [0, 0, 1, 2, 3, 4, 5, 6]
    )


def test_a_slice_selects_no_partition_in_a_plane():
    """A plane has one partition, so the centre of ``kspace_encoding_step_2``
    places nothing."""
    unit = fill(
        header(space(y=4, z=1, kspace_encoding_step_2=(0, 0, 3))),
        acquire(kspace_encode_step_1=1),
    )

    assert unit.data.axes == ("coil", "phase_encode", "readout")
    assert _lines_at(unit.data) == [1]


def test_a_line_placed_outside_the_grid_by_the_limits_centre_is_refused():
    with pytest.raises(
        ValueError,
        match=r"kspace_encode_step_1=15 \(16 once placed about the limits' centre\)",
    ):
        fill(
            header(space(y=16, kspace_encoding_step_1=(0, 15, 7))),
            acquire(kspace_encode_step_1=15),
        )


def test_a_header_that_states_no_centre_places_a_counter_where_it_says():
    unit = fill(header(space(y=8)), acquire(kspace_encode_step_1=5))

    assert _lines_at(unit.data) == [5]


def test_a_non_cartesian_space_places_a_view_where_its_counter_says():
    """A view counter bears no relation to the centre of a grid, so no shift."""
    unit = fill(
        header(
            space(
                x=4,
                y=1,
                trajectory="radial",
                kspace_encoding_step_1=(0, 5, 3),
            )
        ),
        acquire(samples=4, kspace_encode_step_1=5),
    )

    assert _lines_at(unit.data) == [5]


def test_a_non_cartesian_readout_is_right_aligned_whatever_its_echo():
    """Gridded k-space has no echo to place: a trajectory says where each
    sample was taken, and the samples stay as they were acquired."""
    unit = fill(
        header(space(x=8, trajectory="radial", kspace_encoding_step_1=2)),
        acquire(samples=5, centre=0, value=np.arange(1, 6), kspace_encode_step_1=0),
    )

    assert np.array_equal(unit.data.kspace[0, 0].real, [0, 0, 0, 1, 2, 3, 4, 5])


def test_a_buffer_cropped_to_some_lines_holds_only_those():
    grid = EncodingSpace.from_header(header(space(y=16)))
    buffer = ReconBuffer(grid, crop={"phase_encode": (4, 8)})

    assert buffer.kspace.shape == (COILS, 4, N_X)
    assert buffer.origin == {"phase_encode": 4}
    buffer.add(acquire(kspace_encode_step_1=5))
    assert _lines_at(buffer) == [1]
    with pytest.raises(ValueError, match="outside the 4 from 4"):
        buffer.add(acquire(kspace_encode_step_1=9))


@pytest.mark.parametrize(
    ("crop", "message"),
    [
        ({"slice": (0, 1)}, "only"),
        ({"phase_encode": (4, 20)}, "cannot crop"),
        ({"phase_encode": (6, 6)}, "cannot crop"),
    ],
)
def test_a_crop_that_is_not_a_part_of_an_encoded_axis_is_refused(crop, message):
    grid = EncodingSpace.from_header(header(space(y=16, slice=2)))

    with pytest.raises(ValueError, match=message):
        ReconBuffer(grid, crop=crop)


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
    """The lines of the grid ``buffer`` holds samples of."""
    if buffer is None:
        return []
    first = buffer.origin.get("phase_encode", 0)
    return [first + int(line) for line in np.flatnonzero(buffer.mask.any(axis=-1))]


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


def test_the_reference_holds_the_lines_it_covers_and_says_where_they_are():
    """A calibration region is a few lines of a large grid: its buffer is those
    lines, and ``origin`` is where the first is on the grid."""
    calibration = ("ACQ_IS_PARALLEL_CALIBRATION",)
    unit = _lines((0, ()), (3, calibration), (5, calibration))

    assert unit.ref.origin == {"phase_encode": 3}
    assert unit.ref.kspace.shape == (COILS, 3, N_X)
    assert unit.ref.mask[:, 0].tolist() == [True, False, True]
    assert unit.data.origin == {"phase_encode": 0}
    assert unit.data.kspace.shape == (COILS, 8, N_X)


def test_a_reference_is_placed_about_the_limits_centre_before_it_is_cropped():
    """The reference's origin is a position on the same grid the imaging data
    is placed on, so the two are read against each other."""
    calibration = ("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING",)
    unit = fill(
        header(space(y=16, kspace_encoding_step_1=(3, 12, 7))),
        *(
            acquire(
                kspace_encode_step_1=line, flags=calibration if 6 <= line <= 8 else ()
            )
            for line in range(3, 13)
        ),
    )

    assert unit.ref.origin == {"phase_encode": 7}
    assert unit.ref.mask[:, 0].tolist() == [True] * 3
    assert _sampled(unit.data) == list(range(4, 14))
    assert _sampled(unit.ref) == [7, 8, 9]


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


def test_the_reference_of_a_buffer_is_the_placed_acquisition_nearest_the_k_space_centre():
    """A scan that starts and ends far from its centre still names the line at it."""
    sent = [acquire(kspace_encode_step_1=line) for line in range(1, 9)]
    unit = fill(header(space(y=8, kspace_encoding_step_1=(1, 8, 5))), *sent)

    assert unit.data.reference is sent[4]


def test_a_header_that_states_no_centre_has_the_reference_at_half_the_extent():
    sent = [acquire(kspace_encode_step_1=line) for line in range(N_X)]
    unit = fill(header(space(y=8)), *sent)

    assert unit.data.reference is sent[4]


def test_equally_near_acquisitions_have_the_earliest_placed_as_reference():
    sent = [acquire(kspace_encode_step_1=line) for line in (5, 3)]
    unit = fill(header(space(y=8, kspace_encoding_step_1=(0, 7, 4))), *sent)

    assert unit.data.reference is sent[0]


def test_the_reference_of_a_volume_is_nearest_in_line_and_partition_together():
    planes = [
        acquire(kspace_encode_step_1=line, kspace_encode_step_2=partition)
        for line, partition in ((0, 3), (1, 4), (2, 0), (2, 2))
    ]
    stated = space(
        y=4, z=8, kspace_encoding_step_1=(0, 3, 2), kspace_encoding_step_2=(0, 6, 3)
    )

    unit = fill(header(stated), *planes)

    assert unit.data.reference is planes[3]


def test_the_partition_does_not_count_toward_the_distance_in_a_space_without_partitions():
    on_the_centre_line, on_the_centre_partition = (
        acquire(kspace_encode_step_1=line, kspace_encode_step_2=partition)
        for line, partition in ((2, 5), (0, 0))
    )
    planar = EncodingSpace.from_header(
        header(space(y=4, kspace_encoding_step_1=(0, 3, 2)))
    )

    assert planar.partitions == 1
    assert (
        nearest_to_centre(planar, [on_the_centre_partition, on_the_centre_line])
        is on_the_centre_line
    )


def test_a_buffer_with_nothing_placed_has_no_reference():
    assert ReconBuffer(EncodingSpace.from_header(header(space()))).reference is None


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
        acquisition.center_sample = 2
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
            acquisition.center_sample = 2
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
    acquisition.center_sample = 2
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
