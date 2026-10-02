"""Reconstruction units: what joins one, what closes it, and that none outlives its reconstruction."""

from __future__ import annotations

import gc
import weakref
from itertools import chain
from types import SimpleNamespace

import ismrmrd
import numpy as np
import pytest

from pulserver.mrd import AcquisitionBucket, AcquisitionFlag
from pulserver.recon import Gadget, ReconContext, ReconPlugin
from pulserver.recon._runtime.application import _make_bucket, run_application
from pulserver.recon._units import unit_key

N_X = 8
LINES = 4
COILS = 2

SLICE = AcquisitionFlag.LAST_IN_SLICE
CLOSES_SLICE = ("ACQ_LAST_IN_SLICE",)


def space(x=N_X, y=LINES, **counters):
    matrix = SimpleNamespace(
        matrixSize=SimpleNamespace(x=x, y=y, z=1), fieldOfView_mm=None
    )
    limits = {
        name: SimpleNamespace(minimum=0, maximum=extent - 1, center=extent // 2)
        for name, extent in counters.items()
    }
    return SimpleNamespace(
        encodedSpace=matrix,
        reconSpace=matrix,
        encodingLimits=SimpleNamespace(**limits),
        trajectory="cartesian",
    )


def header(*spaces):
    return SimpleNamespace(
        encoding=list(spaces) or [space()],
        acquisitionSystemInformation=SimpleNamespace(receiverChannels=COILS),
    )


def acquire(*flags, encoding_space=0, **idx):
    acquisition = ismrmrd.Acquisition()
    acquisition.resize(N_X, COILS)
    acquisition.data[:] = 1.0
    acquisition.encoding_space_ref = encoding_space
    for name, index in idx.items():
        setattr(acquisition.idx, name, index)
    for flag in flags:
        acquisition.setFlag(getattr(ismrmrd, flag))
    return acquisition


def frame(last=CLOSES_SLICE, **idx):
    """The ``LINES`` readouts of one image, the last of them carrying the flags ``last``."""
    return [
        acquire(*(last if line == LINES - 1 else ()), kspace_encode_step_1=line, **idx)
        for line in range(LINES)
    ]


def sequential(*frames):
    """The readouts of ``frames`` taken one frame after another."""
    return list(chain.from_iterable(frames))


def interleaved(*frames):
    """The readouts of ``frames`` taken one line of each in turn."""
    return list(chain.from_iterable(zip(*frames, strict=True)))


def lines(acquisitions):
    return [acquisition.idx.kspace_encode_step_1 for acquisition in acquisitions]


class Recorder(ReconPlugin):
    """Logs each unit it reconstructs and each time it finishes."""

    def __init__(self, **options):
        super().__init__(**options)
        self.log = []

    def recon(self, context, branch, data):
        self.log.append(("recon", branch, data))
        return branch

    def finish(self, context):
        self.log.append(("finish",))
        return "finished"


class Weak(ReconPlugin):
    """Keeps weak references to what it is given, and nothing else of it."""

    def __init__(self, **options):
        super().__init__(**options)
        self.refs = []

    def recon(self, context, branch, data):
        self.refs += [weakref.ref(data), weakref.ref(data.data)]
        return branch


class Stream:
    """The connection ``run_application`` reads items from and sends outputs on."""

    def __init__(self, *items):
        self.items = items
        self.sent = []

    def __iter__(self):
        return iter(self.items)

    def send(self, item):
        self.sent.append(item)


def reconstructed(plugin):
    """The data of each unit a :class:`Recorder` reconstructed, in order."""
    return [entry[2] for entry in plugin.log if entry[0] == "recon"]


def branches(plugin):
    """The branch each unit of a :class:`Recorder` was reconstructed under, in order."""
    return [entry[1] for entry in plugin.log if entry[0] == "recon"]


def events(plugin):
    return [entry[0] for entry in plugin.log]


def play(plugin, hdr, acquisitions):
    """Take ``acquisitions`` in as the runtime does, then end the stream.

    Returns ``(index, data)`` for each unit the readout at ``index`` closed.
    """
    context = ReconContext.offline(hdr)
    plugin.startup(context)
    closed = []
    for index, acquisition in enumerate(acquisitions):
        for data, _ in plugin.receive(acquisition, context):
            if data is not None:
                closed.append((index, data))
    plugin.flush(context)
    return closed


def held(plugin):
    """Bytes of k-space and sampling mask the open units hold."""
    return sum(
        buffer.kspace.nbytes + buffer.mask.nbytes
        for unit in plugin._units.values()
        for buffer in (unit.data.data, unit.data.ref)
        if buffer is not None
    )


class Double(Gadget):
    def __call__(self, acquisition, data):
        return 2 * data


class DropLine(Gadget):
    """Consumes the readouts of one phase encode."""

    def __init__(self, line):
        self.line = line

    def __call__(self, acquisition, data):
        return None if acquisition.idx.kspace_encode_step_1 == self.line else data


class Spy(Gadget):
    def __init__(self):
        self.seen = []

    def __call__(self, acquisition, data):
        self.seen.append(acquisition)
        return data


# --------------------------------------------------------------------------
# What a unit is
# --------------------------------------------------------------------------


def test_readouts_that_differ_only_in_segment_or_user_counters_share_a_unit():
    first, second = acquire(segment=0), acquire(segment=1)
    second.idx.user[0] = 5

    assert unit_key("imaging", first, ()) == unit_key("imaging", second, ())


@pytest.mark.parametrize(
    "counter", ["repetition", "phase", "slice", "contrast", "set", "average"]
)
def test_an_image_counter_separates_units_unless_it_is_an_axis_of_the_unit(counter):
    first, second = acquire(**{counter: 0}), acquire(**{counter: 1})

    assert unit_key("imaging", first, ()) != unit_key("imaging", second, ())
    assert unit_key("imaging", first, (counter,)) == unit_key(
        "imaging", second, (counter,)
    )


def test_the_branch_and_the_encoding_space_separate_units():
    readout = acquire()

    assert unit_key("imaging", readout, ()) != unit_key("navigator", readout, ())
    assert unit_key("imaging", readout, ()) != unit_key(
        "imaging", acquire(encoding_space=1), ()
    )


def test_readouts_of_two_encoding_spaces_are_reconstructed_as_two_units():
    plugin = Recorder(triggers={"imaging": SLICE})
    readouts = [acquire(encoding_space=1, kspace_encode_step_1=0), *frame()]

    play(plugin, header(space(), space(x=16, y=1)), readouts)

    spaces = [data.data.space for data in reconstructed(plugin)]
    assert sorted(space.index for space in spaces) == [0, 1]


# --------------------------------------------------------------------------
# From the acquisition to the unit
# --------------------------------------------------------------------------


def test_gadgets_run_before_a_readout_is_buffered():
    plugin = Recorder(gadgets=[Double(), DropLine(1)], triggers={"imaging": SLICE})

    play(plugin, header(), frame())

    (data,) = reconstructed(plugin)
    assert data.data.kspace[:, 0].real.max() == 2.0
    assert not data.data.mask[1].any()
    assert lines(data.acquisitions) == [0, 2, 3]


def test_an_acquisition_that_is_rejected_reaches_no_gadget_and_no_unit():
    spy = Spy()
    plugin = Recorder(
        gadgets=[spy],
        require_flags=AcquisitionFlag.IS_REVERSE,
        reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT,
        triggers={"imaging": SLICE},
    )
    unflagged = acquire(kspace_encode_step_1=0)
    reverse = acquire("ACQ_IS_REVERSE", kspace_encode_step_1=1)
    noise = acquire("ACQ_IS_REVERSE", "ACQ_IS_NOISE_MEASUREMENT")

    play(plugin, header(), [unflagged, reverse, noise])

    assert spy.seen == [reverse]
    (data,) = reconstructed(plugin)
    assert data.acquisitions == [reverse]


def test_noise_and_unrouted_navigators_belong_to_no_branch():
    plugin = Recorder(triggers={"spiral": SLICE, "second": AcquisitionFlag.LAST_IN_SET})

    assert plugin.branch_for(acquire("ACQ_IS_NOISE_MEASUREMENT")) is None
    assert plugin.branch_for(acquire("ACQ_IS_NAVIGATION_DATA")) is None
    assert plugin.branch_for(acquire()) == "spiral"


def test_navigators_belong_to_the_navigator_branch_when_one_is_declared():
    plugin = Recorder(
        triggers={"imaging": SLICE, "navigator": AcquisitionFlag.IS_NAVIGATION_DATA}
    )

    assert plugin.branch_for(acquire("ACQ_IS_NAVIGATION_DATA")) == "navigator"
    assert plugin.branch_for(acquire()) == "imaging"


CALIBRATION_ROLES = [
    acquire(kspace_encode_step_1=0),
    acquire("ACQ_IS_PARALLEL_CALIBRATION", kspace_encode_step_1=1),
    acquire("ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING", kspace_encode_step_1=2),
    acquire("ACQ_IS_PHASECORR_DATA", "ACQ_LAST_IN_SLICE", kspace_encode_step_1=3),
]


def test_a_unit_places_each_readout_in_data_and_reference_by_its_flags():
    plugin = Recorder(triggers={"imaging": SLICE})

    play(plugin, header(), CALIBRATION_ROLES)

    (data,) = reconstructed(plugin)
    assert data.data.mask[:, 0].tolist() == [True, False, True, False]
    assert data.ref.mask[:, 0].tolist() == [False, True, True, False]


def test_the_emission_bucket_splits_readouts_as_the_unit_buffers_do():
    plugin = Recorder(triggers={"imaging": SLICE})
    play(plugin, header(), CALIBRATION_ROLES)
    (data,) = reconstructed(plugin)

    bucket = _make_bucket(data)

    assert lines(bucket.data) == [0, 2]
    assert lines(bucket.ref) == [1, 2]
    assert lines(bucket.acquisitions) == [0, 1, 2, 3]


def test_a_unit_of_calibration_alone_has_no_data():
    plugin = Recorder(triggers={"imaging": SLICE})

    play(plugin, header(), [acquire("ACQ_IS_PARALLEL_CALIBRATION", *CLOSES_SLICE)])

    (data,) = reconstructed(plugin)
    assert data.data is None
    assert data.ref.mask.any()


def test_a_plugin_that_is_not_buffered_is_given_the_readouts_as_they_arrived():
    plugin = Recorder(buffered=False, triggers={"imaging": SLICE})

    play(plugin, header(), frame())

    (data,) = reconstructed(plugin)
    assert data.data is None
    assert data.ref is None
    assert lines(data.acquisitions) == [0, 1, 2, 3]


# --------------------------------------------------------------------------
# What closes a unit
# --------------------------------------------------------------------------


def test_interleaved_slices_each_close_at_their_own_flag_with_their_own_readouts():
    plugin = Recorder(triggers={"imaging": SLICE})
    slices = [frame(slice=0), frame(slice=1)]
    readouts = interleaved(*slices)

    closed = play(plugin, header(space(slice=2)), readouts)

    assert [index for index, _ in closed] == [len(readouts) - 2, len(readouts) - 1]
    assert [data.counters["slice"] for _, data in closed] == [0, 1]
    for (_, data), expected in zip(closed, slices, strict=True):
        assert data.acquisitions == expected
        assert data.data.mask.all()


@pytest.mark.parametrize("order", ["sequential", "interleaved"])
def test_echoes_close_their_unit_once_whatever_order_they_arrive_in(order):
    plugin = Recorder(triggers={"imaging": SLICE}, axes=("contrast",))
    echoes = [frame(contrast=0), frame(contrast=1)]
    readouts = (interleaved if order == "interleaved" else sequential)(*echoes)

    closed = play(plugin, header(space(contrast=2)), readouts)

    ((index, data),) = closed
    assert index == len(readouts) - 1
    assert data.data.kspace.shape == (COILS, 2, LINES, N_X)
    assert data.data.mask.all()


def test_a_unit_waits_for_its_flag_at_every_position_along_its_axes():
    plugin = Recorder(triggers={"imaging": SLICE}, axes=("average",))
    context = ReconContext.offline(header(space(average=2)))
    plugin.startup(context)

    flags = [
        len(plugin.receive(readout, context))
        for readout in frame(average=0) + frame(average=1)
    ]

    assert flags == [0, 0, 0, 0, 0, 0, 0, 1]


def test_a_unit_that_is_not_buffered_closes_at_the_first_flag_whatever_its_axes():
    plugin = Recorder(buffered=False, triggers={"imaging": SLICE}, axes=("average",))

    closed = play(plugin, header(space(average=2)), frame(average=0) + frame(average=1))

    assert [index for index, _ in closed] == [LINES - 1, 2 * LINES - 1]


class Pairs:
    """A closure policy of its own: a unit is complete once it holds two readouts."""

    def closed(self, units, acquisition, branch):
        return [key for key, unit in units.items() if len(unit.data.acquisitions) >= 2]


class All:
    """A closure policy closing every open unit at a flagged readout, listing them last-opened first."""

    def closed(self, units, acquisition, branch):
        flagged = acquisition.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)
        return list(reversed(units)) if flagged else []


def test_units_close_through_the_closure_policy_the_plugin_holds():
    plugin = Recorder()
    context = ReconContext.offline(header())
    plugin.startup(context)
    plugin._closure = Pairs()

    for line in range(5):
        plugin.receive(acquire(kspace_encode_step_1=line % LINES), context)
    assert [len(data.acquisitions) for data in reconstructed(plugin)] == [2, 2]
    plugin.flush(context)

    assert [len(data.acquisitions) for data in reconstructed(plugin)] == [2, 2, 1]


def test_units_that_close_together_are_reconstructed_in_the_order_they_opened():
    plugin = Recorder()
    context = ReconContext.offline(header(space(slice=3)))
    plugin.startup(context)
    plugin._closure = All()

    for index in (2, 0, 1):
        plugin.receive(acquire(slice=index, kspace_encode_step_1=0), context)
    plugin.receive(acquire(*CLOSES_SLICE, slice=1, kspace_encode_step_1=1), context)

    assert [data.counters["slice"] for data in reconstructed(plugin)] == [2, 0, 1]


# --------------------------------------------------------------------------
# The end of the measurement
# --------------------------------------------------------------------------


def test_units_still_open_at_the_end_are_reconstructed_under_their_own_branch():
    plugin = Recorder(triggers={"spiral": SLICE})

    play(plugin, header(), frame(last=()))

    assert branches(plugin) == ["spiral"]


def test_units_still_open_at_the_end_are_reconstructed_in_the_order_they_opened():
    plugin = Recorder(triggers={"imaging": SLICE})
    readouts = [acquire(slice=index, kspace_encode_step_1=0) for index in (1, 0, 2)]

    play(plugin, header(space(slice=3)), readouts)

    assert [data.counters["slice"] for data in reconstructed(plugin)] == [1, 0, 2]


def test_the_last_readout_of_the_measurement_closes_every_open_unit_in_the_order_they_opened():
    plugin = Recorder()
    readouts = [acquire(slice=index, kspace_encode_step_1=0) for index in range(3)]
    readouts[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)

    closed = play(plugin, header(space(slice=3)), readouts)

    assert [index for index, _ in closed] == [2, 2, 2]
    assert [data.counters["slice"] for data in reconstructed(plugin)] == [0, 1, 2]


def test_a_stream_with_nothing_to_reconstruct_does_not_invent_a_branch():
    plugin = Recorder(triggers={"spiral": SLICE})

    play(plugin, header(), [acquire("ACQ_IS_NOISE_MEASUREMENT") for _ in range(3)])

    assert events(plugin) == ["finish"]


def test_a_unit_that_closed_is_not_reconstructed_again_at_the_end():
    plugin = Recorder(triggers={"imaging": SLICE})

    play(plugin, header(), frame())

    assert events(plugin) == ["recon", "finish"]


@pytest.mark.parametrize("ending", ["flagged readout", "empty marker", "no marker"])
def test_finish_runs_once_after_the_last_unit_however_the_stream_ends(ending):
    plugin = Recorder(triggers={"imaging": SLICE})
    readouts = frame(last=())
    if ending == "flagged readout":
        readouts[-1].setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    elif ending == "empty marker":
        marker = ismrmrd.Acquisition()
        marker.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        readouts.append(marker)

    play(plugin, header(), readouts)

    assert events(plugin) == ["recon", "finish"]


def test_finish_is_optional():
    class Plain(ReconPlugin):
        def recon(self, context, branch, data):
            return branch

    plugin = Plain(triggers={"imaging": SLICE})
    context = ReconContext.offline(header())
    plugin.startup(context)
    for readout in frame(last=()):
        plugin.receive(readout, context)

    assert [output for _, output in plugin.flush(context)] == ["imaging"]


# --------------------------------------------------------------------------
# What a unit holds, and for how long
# --------------------------------------------------------------------------


def test_a_unit_is_released_as_soon_as_its_reconstruction_returns():
    plugin = Weak(triggers={"imaging": SLICE})
    context = ReconContext.offline(header())
    plugin.startup(context)

    for readout in frame():
        plugin.receive(readout, context)
    gc.collect()

    assert len(plugin.refs) == 2
    assert all(ref() is None for ref in plugin.refs)


def test_a_unit_the_runtime_has_emitted_is_released_when_the_stream_ends():
    plugin = Weak(triggers={"imaging": SLICE})
    stream = Stream(*frame(), *frame(slice=1))

    run_application(plugin, stream, ReconContext.offline(header()))
    gc.collect()

    assert stream.sent == ["imaging", "imaging"]
    assert len(plugin.refs) == 4
    assert all(ref() is None for ref in plugin.refs)


def test_the_bytes_held_stay_those_of_one_frame_over_a_hundred_frames():
    plugin = Weak(triggers={"imaging": AcquisitionFlag.LAST_IN_REPETITION})
    context = ReconContext.offline(header())
    plugin.startup(context)
    while_open, once_closed = set(), set()

    for repetition in range(100):
        for readout in frame(("ACQ_LAST_IN_REPETITION",), repetition=repetition):
            plugin.receive(readout, context)
            if readout.idx.kspace_encode_step_1 == 0:
                while_open.add(held(plugin))
        once_closed.add(held(plugin))
    gc.collect()

    assert len(while_open) == 1
    assert while_open.pop() > 0
    assert once_closed == {0}
    assert len(plugin.refs) == 200
    assert all(ref() is None for ref in plugin.refs)


def test_a_unit_is_allocated_by_its_first_readout_and_not_before():
    plugin = Recorder(triggers={"imaging": SLICE})
    context = ReconContext.offline(header())
    plugin.startup(context)
    assert held(plugin) == 0

    plugin.receive(acquire(kspace_encode_step_1=0), context)

    assert held(plugin) > 0


def test_waveforms_go_with_the_next_unit_to_close_and_with_no_other():
    plugin = Recorder(triggers={"imaging": SLICE})
    context = ReconContext.offline(header(space(slice=2)))
    plugin.startup(context)
    first, second = SimpleNamespace(name="first"), SimpleNamespace(name="second")

    plugin.receive_waveform(first)
    for readout in frame(slice=0):
        plugin.receive(readout, context)
    plugin.receive_waveform(second)
    for readout in frame(slice=1):
        plugin.receive(readout, context)

    assert [data.waveforms for data in reconstructed(plugin)] == [(first,), (second,)]


def test_the_runtime_gives_a_waveform_to_the_next_unit_to_close():
    plugin = Recorder(triggers={"imaging": SLICE})
    waveform = ismrmrd.Waveform.from_array(np.zeros((2, 5), dtype=np.uint32))

    run_application(plugin, Stream(waveform, *frame()), ReconContext.offline(header()))

    (data,) = reconstructed(plugin)
    assert data.waveforms == (waveform,)


def test_replaying_a_bucket_closes_its_units_and_finishes_once():
    plugin = Recorder(triggers={"imaging": SLICE})
    bucket = AcquisitionBucket(data=tuple(frame(slice=0) + frame(slice=1)))

    result = plugin(bucket, ReconContext.offline(header()))

    assert events(plugin) == ["recon", "recon", "finish"]
    assert result == "finished"


# --------------------------------------------------------------------------
# Declaring a plugin
# --------------------------------------------------------------------------


def test_a_plugin_without_recon_cannot_be_instantiated():
    class Nothing(ReconPlugin):
        pass

    with pytest.raises(TypeError, match="recon"):
        Nothing()


def test_a_plugin_closes_at_the_end_of_the_measurement_unless_it_says_otherwise():
    plugin = Recorder()

    assert plugin.triggers == {"imaging": AcquisitionFlag.LAST_IN_MEASUREMENT}
    assert plugin.axes == ()


def test_an_axis_a_unit_cannot_be_laid_out_along_is_refused():
    with pytest.raises(ValueError, match="kspace_encode_step_1"):
        Recorder(axes=("kspace_encode_step_1",))


def test_chain_is_a_deprecated_alias_of_gadgets():
    gadget = Double()

    with pytest.warns(DeprecationWarning, match="gadgets"):
        plugin = Recorder(chain=[gadget])

    assert plugin.gadgets == (gadget,)


def test_branches_is_a_deprecated_alias_of_triggers_with_its_mapping_inverted():
    with pytest.warns(DeprecationWarning, match="triggers"):
        plugin = Recorder(
            branches={SLICE: "imaging", AcquisitionFlag.LAST_IN_SET: "second"}
        )

    assert plugin.triggers == {"imaging": SLICE, "second": AcquisitionFlag.LAST_IN_SET}


def test_two_flags_of_one_branch_are_declared_combined_in_triggers():
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="triggers"):
        Recorder(branches={SLICE: "imaging", AcquisitionFlag.LAST_IN_SET: "imaging"})


@pytest.mark.parametrize(
    "options",
    [
        {"gadgets": [Double()], "chain": [Double()]},
        {"triggers": {"imaging": SLICE}, "branches": {SLICE: "imaging"}},
    ],
)
def test_an_alias_and_its_replacement_are_not_both_accepted(options):
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="not both"):
        Recorder(**options)
