"""The compiled block writer against `_add_block_py`, the Python it replaces.

`PeriodicSequence.add_block` runs once per block of a scan that may hold
millions, so it is the one method worth writing twice. This file is what keeps
the two honest: every scan here is built through both, and the resulting
sequences are compared as bit patterns rather than approximately -- including
the refusals, whose wording and type are part of what a user relies on.

Skipped, not failed, when the extension was not built.
"""

import numpy as np
import pypulseq as pp
import pytest

from fastseq import modules as M
from fastseq.modules import PeriodicSequence, SequenceModule
from fastseq.sequence import make_rf_shim, make_rotation

pytestmark = pytest.mark.skipif(
    M._accel is None, reason='the fastseq accelerator was not built'
)

SYSTEM = pp.Opts(rf_raster_time=2e-6, adc_raster_time=2e-6,
                 grad_raster_time=4e-6, block_duration_raster=4e-6)


@pytest.fixture
def system():
    return SYSTEM


class both_ways:
    """Run a scan-building callable with the writer, then without it.

    The switch is `fastseq.modules._accel`: with it in place the constructor
    binds the compiled writer onto the instance, without it `add_block` is the
    class's Python method. Which means the same source builds both, and any
    difference between the two is a difference in the writer.
    """

    def __init__(self, build):
        self.build = build

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self):
        saved = M._accel
        try:
            M._accel = saved
            fast = self.build()
            M._accel = None
            slow = self.build()
        finally:
            M._accel = saved
        return fast, slow


def identical(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def assert_same_sequence(fast, slow):
    """Two built sequences, library by library and block by block."""
    for name in ('rf_library', 'grad_library', 'adc_library', 'shape_library',
                 'extensions_library', 'label_set_library', 'label_inc_library',
                 'trigger_library', 'rotation_library', 'rf_shim_library',
                 'soft_delay_library'):
        mine = getattr(fast, name, None)
        theirs = getattr(slow, name, None)
        if mine is None or theirs is None:
            assert (mine is None) == (theirs is None), name
            continue
        assert mine.data.keys() == theirs.data.keys(), name
        for key in mine.data:
            a, b = mine.data[key], theirs.data[key]
            if isinstance(a, tuple) and any(isinstance(v, str) for v in a):
                assert a == b, (name, key)          # a soft delay carries a hint
            else:
                assert identical(np.asarray(a, dtype=float),
                                 np.asarray(b, dtype=float)), (name, key)

    assert fast.block_events.keys() == slow.block_events.keys()
    for key in fast.block_events:
        assert identical(fast.block_events[key], slow.block_events[key]), key
    assert fast.block_durations == slow.block_durations
    assert fast.definitions == slow.definitions
    assert getattr(fast, 'soft_delay_hints', {}) == getattr(slow, 'soft_delay_hints', {})


# ---------------------------------------------------------------------------
# Modules the scans below are built from
# ---------------------------------------------------------------------------


class Readout(SequenceModule):
    """Trapezoids, an RF pulse, an ADC and two labels: the ordinary case."""

    def init_module(self, system):
        rf = pp.make_block_pulse(flip_angle=np.deg2rad(12), duration=1e-3,
                                 system=system, use='excitation')
        gx_pre = pp.make_trapezoid(channel='x', area=-100.0, system=system)
        gy_pre = pp.make_trapezoid(channel='y', area=100.0, system=system)
        gz_pre = pp.make_trapezoid(channel='z', area=100.0, system=system)
        gx = pp.make_trapezoid(channel='x', flat_area=200.0, flat_time=400e-6,
                               system=system)
        adc = pp.make_adc(num_samples=64, duration=400e-6, delay=gx.rise_time,
                          system=system)
        lin = pp.make_label(type='SET', label='LIN', value=0)
        par = pp.make_label(type='SET', label='PAR', value=0)
        self.seq = pp.Sequence(system=system)
        self.seq.add_block(rf)
        self.seq.add_block(gx_pre, gy_pre, gz_pre)
        self.seq.add_block(gx, adc, lin, par)


class Spiral(SequenceModule):
    """An arbitrary waveform per shot: one template block, six shape IDs.

    Every shot is laid out, which is what registers its waveform, but the period
    is a single block -- so the shape ID the loop writes is not the one the
    template carries, and that is the case the writer has to get right.
    """

    def init_module(self, system, num_shots=6):
        self.num_shots = num_shots
        samples = 200
        t = np.arange(samples) * system.grad_raster_time
        # Zero at both ends, so consecutive shots connect -- PyPulseq refuses a
        # block whose gradient does not start where the last one stopped.
        radius = 1e5 * np.sin(np.pi * np.arange(samples) / (samples - 1))

        gx_read, gy_read = [], []
        for shot in range(num_shots):
            angle = 2 * np.pi * shot / num_shots
            gx_read.append(pp.make_arbitrary_grad(
                channel='x', waveform=radius * np.cos(20 * t * 1e3 + angle),
                system=system))
            gy_read.append(pp.make_arbitrary_grad(
                channel='y', waveform=radius * np.sin(20 * t * 1e3 + angle),
                system=system))
        adc = pp.make_adc(num_samples=samples, dwell=system.grad_raster_time,
                          system=system)

        self.seq = pp.Sequence(system=system)
        for shot in range(num_shots):
            self.seq.add_block(gx_read[shot], gy_read[shot], adc)


class Rotated(SequenceModule):
    """A rotation and an RF shim, the two extensions PyPulseq cannot write."""

    def init_module(self, system):
        rf = pp.make_block_pulse(flip_angle=np.deg2rad(8), duration=1e-3,
                                 system=system, use='excitation')
        shim = make_rf_shim(np.ones(4, dtype=complex))
        gx = pp.make_trapezoid(channel='x', flat_area=200.0, flat_time=400e-6,
                               system=system)
        rotation = make_rotation(np.array([1.0, 0.0, 0.0, 0.0]))
        self.seq = pp.Sequence(system=system)
        self.seq.add_block(rf, shim)
        self.seq.add_block(gx, rotation)


class Triggered(SequenceModule):
    """A trigger and a soft delay."""

    def init_module(self, system, hint):
        gz = pp.make_trapezoid(channel='z', area=100.0, system=system)
        trigger = pp.make_trigger(channel='physio1', delay=100e-6, duration=1e-3)
        wait = pp.make_soft_delay(hint=hint, offset=0.0, factor=1.0,
                                  default_duration=2e-3)
        self.seq = pp.Sequence(system=system)
        self.seq.add_block(gz, trigger)
        self.seq.add_block(wait)


# ---------------------------------------------------------------------------
# The same scan, both ways
# ---------------------------------------------------------------------------


def test_a_cartesian_scan(system):
    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 12, readout)
        scale = ((2 * np.arange(12) - 12) / 12).tolist()
        gy, gz = readout.gy_pre, readout.gz_pre
        gy_amplitude, gz_amplitude = float(gy.amplitude), float(gz.amplitude)
        phase = 0.0
        for n in range(12):
            gy.amplitude = scale[n] * gy_amplitude
            gz.amplitude = -scale[n] * gz_amplitude
            readout.rf.phase_offset = phase
            readout.adc.phase_offset = phase
            readout.adc.freq_offset = 100.0 * n
            readout.lin.value = n
            readout.par.value = n // 4
            seq.add_block(readout.rf)
            seq.add_block(readout.gx_pre, gy, gz)
            seq.add_block(readout.gx, readout.adc, readout.lin, readout.par)
            phase = (phase + 0.7) % (2 * np.pi)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_switching_events_off(system):
    """Dummy shots: the same template with the ADC, a label, or a whole
    gradient left out."""

    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 8, readout)
        for n in range(8):
            readout.lin.value = n
            seq.add_block(readout.rf)
            if n % 2:
                seq.add_block(readout.gx_pre, readout.gy_pre)   # no gz
            else:
                seq.add_block(readout.gx_pre, readout.gy_pre, readout.gz_pre)
            if n < 2:
                seq.add_block(readout.gx)                        # no ADC, no labels
            elif n < 4:
                seq.add_block(readout.gx, readout.adc, readout.lin)   # one label
            else:
                seq.add_block(readout.gx, readout.adc, readout.lin, readout.par)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_a_multishot_arbitrary_waveform(system):
    def build():
        spiral = Spiral(system)
        seq = PeriodicSequence(system, 4 * spiral.num_shots, spiral)
        gx_read, gy_read, adc = spiral.gx_read, spiral.gy_read, spiral.adc
        for period in range(4):
            for shot in range(spiral.num_shots):
                gx, gy = gx_read[shot], gy_read[shot]
                gx.amplitude = 1.0 - 0.1 * period
                gy.amplitude = 1.0 - 0.1 * period
                adc.phase_offset = 0.3 * period
                seq.add_block(gx, gy, adc)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_rotations_and_rf_shims(system):
    def build():
        rotated = Rotated(system)
        seq = PeriodicSequence(system, 10, rotated)
        angles = (np.pi * np.arange(10) / 10)
        quaternions = np.column_stack([
            np.cos(angles / 2), np.zeros(10), np.zeros(10), np.sin(angles / 2),
        ]).tolist()
        for n in range(10):
            rotated.rotation.rot_quaternion = quaternions[n]
            rotated.shim.shim_vector = np.exp(1j * angles[n]) * np.arange(1, 5) / 4
            seq.add_block(rotated.rf, rotated.shim)
            seq.add_block(rotated.gx, rotated.rotation)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_a_scipy_rotation_object_is_converted(system):
    """The slow path: a `Rotation` rather than four numbers."""
    Rotation = pytest.importorskip('scipy.spatial.transform').Rotation

    def build():
        rotated = Rotated(system)
        seq = PeriodicSequence(system, 5, rotated)
        for n in range(5):
            rotated.rotation.rot_quaternion = Rotation.from_euler('z', 0.3 * n)
            seq.add_block(rotated.rf, rotated.shim)
            seq.add_block(rotated.gx, rotated.rotation)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_triggers_and_soft_delays(system):
    def build():
        te = Triggered(system, hint='TE')
        tr = Triggered(system, hint='TR')
        seq = PeriodicSequence(system, 6, te, tr)
        for n in range(6):
            te.trigger.delay = 100e-6 + 10e-6 * n
            te.trigger.duration = 1e-3 + 10e-6 * n
            seq.add_block(te.gz, te.trigger)
            seq.add_block(te.wait)
            if n % 2:
                seq.add_block(tr.gz)                 # trigger switched off
            else:
                seq.add_block(tr.gz, tr.trigger)
            seq.add_block(tr.wait)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_a_bare_delay_sets_the_block_duration(system):
    def build():
        readout = Readout(system)
        wait = pp.make_delay(5e-3)
        seq = PeriodicSequence(system, 4, readout, wait)
        for n in range(4):
            seq.add_block(readout.rf)
            seq.add_block(readout.gx_pre, readout.gy_pre, readout.gz_pre)
            seq.add_block(readout.gx, readout.adc, readout.lin, readout.par)
            wait.delay = 5e-3 + 1e-3 * n
            seq.add_block(wait)
        return seq.seq

    assert_same_sequence(*both_ways(build).run())


def test_a_label_read_through_the_registry(system):
    """The writer maps a label name to a number through `LABEL_ID`.

    `ONCE` and `TRID` sit at the far end of that registry, past the counters
    every other test here uses, so a writer holding a stale or truncated copy of
    it gets them wrong. (A name the registry learned at runtime cannot be
    reached from a module: `pp.Sequence.add_block` refuses any label Pulseq did
    not ship with, so the custom-label branch downstream is unreachable through
    this path. The registry itself is checked directly below.)
    """
    from fastseq.sequence import LABEL_ID, label_id

    class Marked(SequenceModule):
        def init_module(self, system):
            gx = pp.make_trapezoid(channel='x', flat_area=200.0, flat_time=400e-6,
                                   system=system)
            once = pp.make_label(type='SET', label='ONCE', value=1)
            trid = pp.make_label(type='SET', label='TRID', value=0)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(gx, once, trid)

    def build():
        marked = Marked(system)
        seq = PeriodicSequence(system, 5, marked)
        for n in range(5):
            marked.once.value = int(n == 0)
            marked.trid.value = n * 3
            seq.add_block(marked.gx, marked.once, marked.trid)
        return seq.seq

    fast, slow = both_ways(build).run()
    assert_same_sequence(fast, slow)

    assert label_id('ONCE') == LABEL_ID['ONCE']
    assert label_id('MYFLAG') == len(LABEL_ID)      # registered on the way past
    assert label_id('MYFLAG') == LABEL_ID['MYFLAG']  # and remembered


# ---------------------------------------------------------------------------
# Refusals -- same type, same wording
# ---------------------------------------------------------------------------


def refusal(build):
    """The exception a build raises, through the writer and through Python."""
    out = []
    saved = M._accel
    try:
        for accelerator in (saved, None):
            M._accel = accelerator
            with pytest.raises(Exception) as caught:   # noqa: B017 - the point is which
                build()
            out.append(caught.value)
    finally:
        M._accel = saved
    return out


def assert_same_refusal(build):
    fast, slow = refusal(build)
    assert type(fast) is type(slow), (type(fast), type(slow))
    assert str(fast) == str(slow), (str(fast), str(slow))
    return fast


def test_one_block_too_many(system):
    """The period filled exactly, and then one block more."""

    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 1, readout)
        seq.add_block(readout.rf)
        seq.add_block(readout.gx_pre, readout.gy_pre, readout.gz_pre)
        seq.add_block(readout.gx, readout.adc, readout.lin, readout.par)
        seq.add_block(readout.rf)          # the scan is over

    assert isinstance(assert_same_refusal(build), IndexError)


def test_an_event_the_template_has_no_slot_for(system):
    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 2, readout)
        seq.add_block(readout.rf, readout.gx_pre)     # block 0 carries no gradient

    error = assert_same_refusal(build)
    assert isinstance(error, ValueError)
    assert 'trapezoidal gx' in str(error)


def test_an_unregistered_event(system):
    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 2, readout)
        stray = pp.make_trapezoid(channel='x', area=10.0, system=system)
        seq.add_block(stray)

    error = assert_same_refusal(build)
    assert isinstance(error, ValueError)
    assert 'unregistered' in str(error)


def test_more_extensions_than_the_block_has_room_for(system):
    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 2, readout)
        seq.add_block(readout.rf)
        seq.add_block(readout.gx_pre, readout.gy_pre, readout.gz_pre)
        third = pp.make_label(type='SET', label='SLC', value=1)
        seq.add_block(readout.gx, readout.adc, readout.lin, readout.par, third)

    assert isinstance(assert_same_refusal(build), ValueError)


def test_an_extension_of_a_type_the_template_never_carried(system):
    def build():
        readout = Readout(system)
        seq = PeriodicSequence(system, 2, readout)
        seq.add_block(readout.rf, pp.make_trigger(channel='osc0', duration=1e-3))

    assert isinstance(assert_same_refusal(build), ValueError)


def test_an_rf_shim_with_the_wrong_number_of_channels(system):
    def build():
        rotated = Rotated(system)
        seq = PeriodicSequence(system, 2, rotated)
        rotated.shim.shim_vector = np.ones(3, dtype=complex)   # template has four
        seq.add_block(rotated.rf, rotated.shim)

    assert isinstance(assert_same_refusal(build), ValueError)


def test_a_waveform_that_was_never_stamped_with_a_shape(system):
    """An event carrying a registration but no shape ID.

    Both paths raise the same `AttributeError` here rather than a worded
    refusal, which is the writer matching Python rather than improving on it --
    the two agreeing is what this file is for. Reaching this state takes
    deleting the attribute by hand; an ordinary `deepcopy` keeps it, which is
    why the copies in `make_mprage` work.
    """
    import copy as copy_module

    def build():
        spiral = Spiral(system)
        stale = copy_module.deepcopy(spiral.gx_read[0])
        del stale._ps_shape
        seq = PeriodicSequence(system, 2, spiral)
        seq.add_block(stale, spiral.gy_read[0], spiral.adc)

    error = assert_same_refusal(build)
    assert isinstance(error, AttributeError)
    assert '_ps_shape' in str(error)


def test_the_cursor_is_visible_from_python_after_every_block(system):
    """Other methods read `seq.cursor`; the writer keeps it current."""
    readout = Readout(system)
    seq = PeriodicSequence(system, 3, readout)
    assert seq.cursor == 0
    for n in range(3):
        seq.add_block(readout.rf)
        assert seq.cursor == 3 * n + 1
        seq.add_block(readout.gx_pre, readout.gy_pre, readout.gz_pre)
        assert seq.cursor == 3 * n + 2
        seq.add_block(readout.gx, readout.adc, readout.lin, readout.par)
        assert seq.cursor == 3 * n + 3
    assert isinstance(seq.cursor, int)


def test_a_refused_block_does_not_advance_the_cursor(system):
    readout = Readout(system)
    seq = PeriodicSequence(system, 2, readout)
    seq.add_block(readout.rf)
    assert seq.cursor == 1
    with pytest.raises(ValueError):
        seq.add_block(readout.rf, readout.gx_pre)    # block 1 has no RF slot
    assert seq.cursor == 1
