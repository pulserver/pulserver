"""SequenceModule ergonomics and multishot template inference."""

import numpy as np
import pypulseq as pp
import pytest

from fastseq.sequence import Sequence, make_rf_shim, make_rotation
from fastseq.modules import (
    LineReadout3D,
    NonSelectiveExcitation,
    PeriodicSequence,
    SequenceModule,
)


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


class _MultishotReadout(SequenceModule):
    def init_module(self, system, num_shots=3):
        self.num_shots = num_shots
        gradients = [
            pp.make_arbitrary_grad(
                channel="x",
                waveform=np.array([0.0, peak, 0.5 * peak, 0.0]),
                system=system,
            )
            for peak in np.linspace(1_000.0, 3_000.0, num_shots)
        ]
        adc = pp.make_adc(num_samples=8, duration=40e-6, system=system)
        recovery = pp.make_delay(1e-3)
        unused = pp.make_trapezoid(channel="z", area=1.0, system=system)

        self.seq = pp.Sequence(system=system)
        for shot in range(num_shots):
            self.seq.add_block(gradients[shot], adc)
            self.seq.add_block(recovery)


def test_module_automatically_publishes_played_constructor_locals(system):
    module = _MultishotReadout(system, num_shots=3)

    assert module.events.gradients is not None
    assert len(module.events.gradients) == 3
    assert module.events.adc.type == "adc"
    assert module.events.recovery.type == "delay"
    assert not hasattr(module.events, "unused")
    assert all(hasattr(gradient, "id") for gradient in module.events.gradients)


def test_num_shots_argument_infers_one_shot_template_length(system):
    module = _MultishotReadout(system, num_shots=3)

    assert module.num_shots == 3
    assert module.num_blocks == 2
    assert len(module.blocks) == 6

    sequence = PeriodicSequence(system, 4, module)
    assert len(sequence._seq.block_events) == 2
    assert sequence.num_period_blocks == 2
    assert sequence.num_blocks == 8


def test_num_shots_is_explicit_module_state_not_an_argument_name(system):
    class ParameterOnly(SequenceModule):
        def init_module(self, system, num_shots=3):
            delay = pp.make_delay(1e-3)
            self.seq = pp.Sequence(system=system)
            for _ in range(num_shots):
                self.seq.add_block(delay)

    module = ParameterOnly(system)

    assert module.num_shots == 1
    assert module.num_blocks == 3


def test_explicit_num_blocks_still_infers_laid_out_shots(system):
    class ExplicitPeriod(SequenceModule):
        def init_module(self, system):
            delay = pp.make_delay(1e-3)
            self.seq = pp.Sequence(system=system)
            for _ in range(6):
                self.seq.add_block(delay)
            self.num_blocks = 2

    module = ExplicitPeriod(system)

    assert module.num_blocks == 2
    assert module.num_shots == 3


def test_num_shots_must_evenly_partition_laid_out_blocks(system):
    class UnevenShots(SequenceModule):
        def init_module(self, system, num_shots=2):
            self.num_shots = num_shots
            delay = pp.make_delay(1e-3)
            self.seq = pp.Sequence(system=system)
            for _ in range(3):
                self.seq.add_block(delay)

    with pytest.raises(ValueError, match="cannot be divided into 2 equal shots"):
        UnevenShots(system)


# -----------------------------------------------------------------------------
# Publication reads the constructor's *final* namespace
# -----------------------------------------------------------------------------


def test_publication_sees_names_bound_after_the_last_add_block(system):
    """The same namespace a closing `self.publish()` would have read."""

    class LateRebind(SequenceModule):
        def init_module(self, system):
            gz = pp.make_trapezoid(channel="z", area=1.0, system=system)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(gz)
            gz_spoil = gz  # named only once every block is laid out

    module = LateRebind(system)

    assert module.events.gz_spoil is module.events.gz


def test_a_module_built_entirely_by_a_helper_says_it_published_nothing(system):
    class Delegating(SequenceModule):
        def init_module(self, system):
            self.seq = pp.Sequence(system=system)
            self._build(system)

        def _build(self, system):
            gz = pp.make_trapezoid(channel="z", area=1.0, system=system)
            self.seq.add_block(gz)

    with pytest.warns(UserWarning, match="published no events"):
        module = Delegating(system)

    assert not vars(module.events)


def test_a_helper_can_still_publish_for_itself(system):
    class Delegating(SequenceModule):
        def init_module(self, system):
            self.seq = pp.Sequence(system=system)
            self._build(system)

        def _build(self, system):
            gz_spoil = pp.make_trapezoid(channel="z", area=1.0, system=system)
            self.seq.add_block(gz_spoil)
            self.publish()

    module = Delegating(system)

    assert module.gz_spoil.type == "trap"


def test_line_readout_publishes_its_labels_without_an_explicit_publish(system):
    excitation = NonSelectiveExcitation(system=system, flip_angle_deg=10.0, duration_s=1e-3)
    readout = LineReadout3D(
        system=system,
        excitation=excitation,
        FOV_m=(0.225, 0.225, 0.225),
        matrix=(64, 64, 64),
        spoiling_area=3e3,
        labels=("LIN", "PAR"),
    )

    assert [label.label for label in readout.events.adc_labels] == ["LIN", "PAR"]
    assert readout.events.gx_read.type in ("grad", "trap")
    assert not hasattr(readout.events, "gx_read_full")   # an intermediate stays private


# -----------------------------------------------------------------------------
# Events are reachable straight off the module
# -----------------------------------------------------------------------------


def test_module_forwards_attribute_lookup_to_its_events(system):
    module = _MultishotReadout(system, num_shots=3)

    assert module.adc is module.events.adc
    assert module.gradients is module.events.gradients
    assert "gradients" in dir(module)          # what completion offers

    with pytest.raises(AttributeError, match="no attribute or event 'gx_read'"):
        module.gx_read


def test_module_attributes_win_over_events_and_the_clash_is_reported(system):
    class Clashing(SequenceModule):
        def init_module(self, system):
            duration = pp.make_trapezoid(channel="z", area=1.0, system=system)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(duration)

    with pytest.warns(UserWarning, match="also a module attribute"):
        module = Clashing(system)

    assert isinstance(module.duration, float)                  # the module's own
    assert module.events.duration.type == "trap"               # the event


# -----------------------------------------------------------------------------
# Scaling reads `amplitude`, which registration is what puts there
# -----------------------------------------------------------------------------


def test_registration_gives_rf_and_arbitrary_gradients_an_amplitude(system):
    module = _MultishotReadout(system, num_shots=3)

    for gradient, peak in zip(module.gradients, np.linspace(1_000.0, 3_000.0, 3)):
        assert gradient.amplitude == pytest.approx(peak)
        assert gradient.waveform is not None       # the samples are still there

    excitation = NonSelectiveExcitation(system=system, flip_angle_deg=10.0, duration_s=1e-3)
    assert excitation.rf.amplitude == pytest.approx(np.abs(excitation.rf.signal).max())


# -----------------------------------------------------------------------------
# ADC phase modulation: one shape per shot, exactly as for gradient waveforms
# -----------------------------------------------------------------------------


class _PhaseModulatedReadout(SequenceModule):
    """A shot's demodulation travels with the shot's waveform."""

    num_samples = 8

    def init_module(self, system, num_shots=4):
        self.num_shots = num_shots
        gx = [
            pp.make_arbitrary_grad(
                channel="x",
                waveform=np.array([0.0, 1e3 * (shot + 1), 5e2 * (shot + 1), 0.0]),
                system=system,
            )
            for shot in range(num_shots)
        ]
        adc = [
            pp.make_adc(
                num_samples=self.num_samples,
                duration=16e-6,
                system=system,
                phase_modulation=np.linspace(0.0, 0.1 * (shot + 1), self.num_samples),
            )
            for shot in range(num_shots)
        ]
        self.seq = pp.Sequence(system=system)
        for shot in range(num_shots):
            self.seq.add_block(gx[shot], adc[shot])


def test_each_shot_registers_its_own_phase_modulation_shape(system):
    module = _PhaseModulatedReadout(system, num_shots=4)

    assert [adc.shape_IDs for adc in module.adc] == [[3], [4], [5], [6]]
    assert all(adc.shape_id == adc.shape_IDs[0] for adc in module.adc)


def test_the_loop_picks_a_shot_by_handing_in_its_adc(system):
    module = _PhaseModulatedReadout(system, num_shots=4)
    sequence = PeriodicSequence(system, 4, module)

    # Rebased: the shapes are the parent's now, and still one per shot.
    assert len({adc._ps_shape for adc in module.adc}) == 4

    order = [3, 2, 1, 0]
    for shot in order:
        sequence.add_block(module.gx[shot], module.adc[shot])

    built = sequence.build()
    for block, shot in enumerate(order, start=1):
        played = built.get_block(block)
        assert np.allclose(
            np.asarray(played.adc.phase_modulation).ravel(),
            np.linspace(0.0, 0.1 * (shot + 1), _PhaseModulatedReadout.num_samples),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(played.gx.waveform),
            np.array([0.0, 1e3 * (shot + 1), 5e2 * (shot + 1), 0.0]),
            rtol=1e-6,
        )


def test_phase_modulation_survives_a_seq_file_round_trip(system, tmp_path):
    module = _PhaseModulatedReadout(system, num_shots=4)
    sequence = PeriodicSequence(system, 4, module)
    for shot in range(4):
        sequence.add_block(module.gx[shot], module.adc[shot])

    path = tmp_path / "phase_modulation.seq"
    sequence.write(str(path))

    reread = pp.Sequence(system=system)
    reread.read(str(path))
    for shot in range(4):
        assert np.allclose(
            np.asarray(reread.get_block(shot + 1).adc.phase_modulation).ravel(),
            np.linspace(0.0, 0.1 * (shot + 1), _PhaseModulatedReadout.num_samples),
            atol=1e-5,
        )


def test_an_adc_without_phase_modulation_names_no_shape(system):
    class Plain(SequenceModule):
        def init_module(self, system):
            adc = pp.make_adc(num_samples=8, duration=16e-6, system=system)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(adc)

    module = Plain(system)
    assert module.adc.shape_IDs == [0]

    sequence = PeriodicSequence(system, 2, module)
    for _ in range(2):
        sequence.add_block(module.adc)

    built = sequence.build()
    assert list(built.adc_library.data.values())[0][7] == 0
    assert len(built.shape_library.data) == 0


def test_an_adc_handed_to_another_module_is_registered_from_scratch(system):
    """`_fresh` has to drop Pulseq's own `shape_id` cache, not just ours."""
    from fastseq.modules import _fresh

    donor = _PhaseModulatedReadout(system, num_shots=2)

    class Borrower(SequenceModule):
        def init_module(self, system, adc):
            adc = _fresh(adc)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(adc)

    borrower = Borrower(system, donor.adc[1])

    # Its own sequence, so its own numbering -- and the samples the donor had.
    assert borrower.adc.shape_IDs == [1]
    assert np.allclose(
        np.asarray(borrower.adc.phase_modulation).ravel(),
        np.linspace(0.0, 0.2, _PhaseModulatedReadout.num_samples),
    )


# -----------------------------------------------------------------------------
# Extensions: a block's labels, triggers and soft delays
# -----------------------------------------------------------------------------


class _LabelledReadout(SequenceModule):
    """Two label slots on the readout block, one on the excitation."""

    def init_module(self, system):
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(10), duration=1e-3, system=system, use="excitation"
        )
        gx_read = pp.make_trapezoid(channel="x", flat_area=200.0, flat_time=400e-6, system=system)
        adc = pp.make_adc(num_samples=8, duration=400e-6, delay=gx_read.rise_time, system=system)
        once_label = pp.make_label(type="SET", label="ONCE", value=0)
        adc_labels = [pp.make_label(type="SET", label=name, value=0) for name in ("LIN", "PAR")]

        self.seq = pp.Sequence(system=system)
        self.seq.add_block(rf, once_label)
        self.seq.add_block(gx_read, adc, *adc_labels)


def _labels_of(seq, block):
    played = seq.get_block(block)
    labels = getattr(played, "label", None)
    if not labels:
        return {}
    return {label.label: int(label.value) for label in labels.values()}


def test_every_iteration_writes_its_own_label_values(system):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 6, module)
    lin, par = module.adc_labels

    for line in range(6):
        lin.value, par.value = line, 2 * line
        sequence.add_block(module.rf)
        sequence.add_block(module.gx_read, module.adc, lin, par)

    built = sequence.seq
    for line in range(6):
        assert _labels_of(built, 2 * line + 2) == {"LIN": line, "PAR": 2 * line}

    # LIN 0..5 and PAR 0,2,..10 -- twelve rows, and not one for every block.
    assert len(built.label_set_library.data) == 12


def test_an_extension_left_out_is_switched_off(system):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 4, module)
    lin, par = module.adc_labels

    for line in range(4):
        lin.value, par.value = line, 0
        sequence.add_block(module.rf)              # the ONCE slot goes unused
        if line == 0:
            sequence.add_block(module.gx_read, module.adc)      # and both labels
        else:
            sequence.add_block(module.gx_read, module.adc, lin, par)

    built = sequence.seq
    assert _labels_of(built, 1) == {}              # no ONCE anywhere
    assert _labels_of(built, 2) == {}              # the unlabelled readout
    assert _labels_of(built, 4) == {"LIN": 1, "PAR": 0}


def test_a_block_cannot_be_handed_more_labels_than_it_was_built_with(system):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 1, module)
    lin, par = module.adc_labels
    extra = pp.make_label(type="SET", label="SLC", value=1)

    with pytest.raises(ValueError, match="was built to carry 1 LABELSET"):
        sequence.add_block(module.rf, lin, par)

    # A fresh sequence, because the refused block is still the one at the cursor.
    sequence = PeriodicSequence(system, 1, module)
    sequence.add_block(module.rf)
    with pytest.raises(ValueError, match="was built to carry 2 LABELSET"):
        sequence.add_block(module.gx_read, module.adc, lin, par, extra)


def test_a_block_cannot_be_handed_an_extension_type_it_has_no_slot_for(system):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 1, module)
    trigger = pp.make_trigger(channel="physio1", duration=100e-6, system=system)

    with pytest.raises(ValueError, match="carries no TRIGGERS extension"):
        sequence.add_block(module.rf, trigger)


def test_a_trigger_slot_takes_the_trigger_it_is_handed(system):
    class Triggered(SequenceModule):
        def init_module(self, system):
            gz = pp.make_trapezoid(channel="z", area=200.0, duration=1e-3, system=system)
            trigger = pp.make_trigger(channel="physio1", duration=100e-6, system=system)
            self.seq = pp.Sequence(system=system)
            self.seq.add_block(gz, trigger)

    module = Triggered(system)
    sequence = PeriodicSequence(system, 2, module)

    late = pp.make_trigger(channel="physio2", duration=200e-6, delay=50e-6, system=system)
    sequence.add_block(module.gz, module.trigger)
    sequence.add_block(module.gz, late)

    built = sequence.seq
    first = next(iter(built.get_block(1).trigger.values()))
    assert first.channel == "physio1"
    second = next(iter(built.get_block(2).trigger.values()))
    assert second.channel == "physio2"
    assert second.duration == pytest.approx(200e-6)
    assert second.delay == pytest.approx(50e-6)


class _SoftDelayed(SequenceModule):
    def init_module(self, system, hint):
        gz = pp.make_trapezoid(channel="z", area=200.0, duration=1e-3, system=system)
        wait = pp.make_soft_delay(hint=hint, offset=0.0, factor=1.0, default_duration=1e-3)
        self.seq = pp.Sequence(system=system)
        self.seq.add_block(gz)
        self.seq.add_block(wait)


def test_composing_soft_delays_renumbers_them_by_hint(system):
    """Both modules called their delay numID 0; the hints are what must survive."""
    te = _SoftDelayed(system, hint="TE")
    tr = _SoftDelayed(system, hint="TR")
    assert te.seq.soft_delay_hints == {"TE": 0}
    assert tr.seq.soft_delay_hints == {"TR": 0}

    sequence = PeriodicSequence(system, 2, te, tr)
    for _ in range(2):
        sequence.add_block(te.gz)
        sequence.add_block(te.wait)
        sequence.add_block(tr.gz)
        sequence.add_block(tr.wait)

    built = sequence.seq
    assert built.soft_delay_hints == {"TE": 0, "TR": 1}
    assert len(built.soft_delay_library.data) == 2
    assert built.get_block(2).soft_delay.hint == "TE"
    assert built.get_block(4).soft_delay.hint == "TR"


def test_a_soft_delay_can_be_switched_off_like_any_other_extension(system):
    te = _SoftDelayed(system, hint="TE")
    sequence = PeriodicSequence(system, 2, te)

    sequence.add_block(te.gz)
    sequence.add_block(te.wait)
    sequence.add_block(te.gz)
    sequence.add_block(pp.make_delay(1e-3))

    built = sequence.seq
    carried = [
        block for block in range(1, len(built.block_events) + 1)
        if getattr(built.get_block(block), "soft_delay", None) is not None
    ]
    assert carried == [2]


def test_soft_delays_survive_a_seq_file_round_trip(system, tmp_path):
    te = _SoftDelayed(system, hint="TE")
    sequence = PeriodicSequence(system, 1, te)
    sequence.add_block(te.gz)
    sequence.add_block(te.wait)

    path = tmp_path / "soft.seq"
    sequence.write(str(path))

    reread = pp.Sequence(system=system)
    reread.read(str(path))
    assert reread.soft_delay_hints == {"TE": 0}
    assert reread.get_block(2).soft_delay.hint == "TE"


# -----------------------------------------------------------------------------
# `seq`: the finished, ordinary Pulseq sequence
# -----------------------------------------------------------------------------


def test_seq_builds_once_and_is_an_ordinary_pulseq_sequence(system):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 3, module)
    lin, par = module.adc_labels
    for line in range(3):
        lin.value = line
        sequence.add_block(module.rf)
        sequence.add_block(module.gx_read, module.adc, lin, par)

    built = sequence.seq
    assert isinstance(built, Sequence)             # the wrapper
    assert isinstance(built._seq, pp.Sequence)     # wrapping an ordinary one
    assert built is sequence.seq                   # built once, then kept
    assert built.system is system
    assert len(built.block_events) == sequence.num_blocks
    assert built.check_timing()[0]


# -----------------------------------------------------------------------------
# The Sequence wrapper: rotations, RF shims, and the 1.5.1 file format
# -----------------------------------------------------------------------------


class _RadialSpoke(SequenceModule):
    """One base spoke rotated per shot, one RF pulse re-shimmed per shot."""

    channels = 4

    def init_module(self, system):
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(10), duration=1e-3, system=system, use="excitation"
        )
        shim = make_rf_shim(np.ones(self.channels, dtype=complex))
        gx_read = pp.make_trapezoid(channel="x", flat_area=200.0, flat_time=400e-6, system=system)
        adc = pp.make_adc(num_samples=8, duration=400e-6, delay=gx_read.rise_time, system=system)
        lin = pp.make_label(type="SET", label="LIN", value=0)
        rotation = make_rotation((1.0, 0.0, 0.0, 0.0))

        self.seq = pp.Sequence(system=system)
        self.seq.add_block(rf, shim)
        self.seq.add_block(gx_read, adc, lin, rotation)


def _z_quaternion(angle):
    return (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))


def test_a_module_assigned_a_plain_pypulseq_sequence_gets_a_wrapped_one(system):
    module = _RadialSpoke(system)

    assert isinstance(module.seq, Sequence)
    assert isinstance(module.seq._seq, pp.Sequence)
    assert len(module.seq.rotation_library.data) == 1
    assert len(module.seq.rf_shim_library.data) == 1


def test_rotations_and_shims_vary_per_iteration_like_any_other_extension(system):
    module = _RadialSpoke(system)
    spokes = 8
    sequence = PeriodicSequence(system, spokes, module)

    angles = np.pi * np.arange(spokes) / spokes
    for shot, angle in enumerate(angles):
        module.rotation.rot_quaternion = _z_quaternion(angle)
        module.shim.shim_vector = np.exp(1j * angle) * np.arange(1, module.channels + 1)
        module.lin.value = shot
        sequence.add_block(module.rf, module.shim)
        sequence.add_block(module.gx_read, module.adc, module.lin, module.rotation)

    built = sequence.seq
    assert len(built.rotation_library.data) == spokes
    assert len(built.rf_shim_library.data) == spokes

    for shot, angle in enumerate(angles):
        excitation = built.get_block(2 * shot + 1)
        readout = built.get_block(2 * shot + 2)

        assert excitation.rotation is None and excitation.rf_shim is not None
        assert readout.rf_shim is None
        assert np.allclose(readout.rotation, _z_quaternion(angle), atol=1e-9)

        wanted = np.exp(1j * angle) * np.arange(1, module.channels + 1)
        shim = np.asarray(excitation.rf_shim)
        assert np.allclose(shim[0::2], np.abs(wanted), atol=1e-9)
        assert np.allclose(shim[1::2], np.angle(wanted), atol=1e-9)


def test_a_rotation_or_shim_left_out_is_switched_off(system):
    module = _RadialSpoke(system)
    sequence = PeriodicSequence(system, 2, module)

    sequence.add_block(module.rf, module.shim)
    sequence.add_block(module.gx_read, module.adc, module.lin, module.rotation)
    sequence.add_block(module.rf)                                    # no shim
    sequence.add_block(module.gx_read, module.adc, module.lin)       # no rotation

    built = sequence.seq
    assert built.get_block(1).rf_shim is not None
    assert built.get_block(2).rotation is not None
    assert built.get_block(3).rf_shim is None
    assert built.get_block(4).rotation is None


def test_a_shim_for_the_wrong_number_of_channels_is_refused(system):
    module = _RadialSpoke(system)
    sequence = PeriodicSequence(system, 1, module)
    module.shim.shim_vector = np.ones(2, dtype=complex)

    with pytest.raises(ValueError, match="RF shim for 2 channels"):
        sequence.add_block(module.rf, module.shim)


def test_native_bakes_a_rotation_into_the_waveforms(system):
    module = _RadialSpoke(system)
    sequence = PeriodicSequence(system, 2, module)
    base = module.gx_read.amplitude

    for angle in (0.0, np.pi / 3):
        module.rotation.rot_quaternion = _z_quaternion(angle)
        sequence.add_block(module.rf, module.shim)
        sequence.add_block(module.gx_read, module.adc, module.lin, module.rotation)

    native = sequence.seq.native()
    assert isinstance(native, pp.Sequence)

    turned = native.get_block(4)
    assert np.isclose(turned.gx.amplitude, np.cos(np.pi / 3) * base, rtol=1e-6)
    assert np.isclose(turned.gy.amplitude, np.sin(np.pi / 3) * base, rtol=1e-6)

    # A shimmed pulse becomes its channels, laid end to end in one signal.
    excitation = native.get_block(1)
    assert len(excitation.rf.signal) == module.channels * len(module.rf.signal)


def test_the_wrapper_writes_and_reads_the_extensions_pypulseq_cannot(system, tmp_path):
    module = _RadialSpoke(system)
    spokes = 4
    sequence = PeriodicSequence(system, spokes, module)
    angles = np.pi * np.arange(spokes) / spokes
    for shot, angle in enumerate(angles):
        module.rotation.rot_quaternion = _z_quaternion(angle)
        module.shim.shim_vector = np.exp(1j * angle) * np.ones(module.channels)
        module.lin.value = shot
        sequence.add_block(module.rf, module.shim)
        sequence.add_block(module.gx_read, module.adc, module.lin, module.rotation)

    path = tmp_path / "radial.seq"
    sequence.write(str(path))
    text = path.read_text()
    assert "extension ROTATIONS" in text
    assert "extension RF_SHIMS" in text
    assert "revision 1" in text          # 1.5.1, because it used them

    reread = Sequence.read(str(path), system=system)
    assert len(reread.rotation_library.data) == spokes
    assert len(reread.rf_shim_library.data) == spokes
    for shot, angle in enumerate(angles):
        readout = reread.get_block(2 * shot + 2)
        assert np.allclose(readout.rotation, _z_quaternion(angle), atol=1e-5)


def test_a_sequence_using_neither_is_still_written_as_plain_1_5_0(system, tmp_path):
    module = _LabelledReadout(system)
    sequence = PeriodicSequence(system, 1, module)
    sequence.add_block(module.rf)
    sequence.add_block(module.gx_read, module.adc, *module.adc_labels)

    path = tmp_path / "plain.seq"
    sequence.write(str(path))
    text = path.read_text()
    assert "revision 0" in text
    assert "ROTATIONS" not in text and "RF_SHIMS" not in text

    # And upstream PyPulseq reads it, which is the point of not raising the
    # revision when nothing needed it.
    upstream = pp.Sequence(system=system)
    upstream.read(str(path))
    assert len(upstream.block_events) == len(sequence.seq.block_events)


def test_a_plain_pypulseq_file_reads_back_through_the_wrapper(system, tmp_path):
    plain = pp.Sequence(system=system)
    plain.add_block(pp.make_trapezoid(channel="z", area=200.0, duration=1e-3, system=system))
    plain.add_block(pp.make_delay(1e-3))
    path = tmp_path / "upstream.seq"
    plain.write(str(path))

    wrapped = Sequence.read(str(path), system=system)
    assert len(wrapped.block_events) == 2
    assert len(wrapped.rotation_library.data) == 0
    assert wrapped.get_block(1).rotation is None


def test_the_wrapper_delegates_everything_it_does_not_own(system):
    sequence = Sequence(system=system)
    sequence.add_block(pp.make_delay(1e-3))

    assert sequence.duration()[0] == pytest.approx(1e-3)
    assert len(sequence.block_events) == 1          # read through
    sequence.definitions["Custom"] = 3.0            # written through
    assert sequence._seq.definitions["Custom"] == 3.0
    assert "rotation_library" in dir(sequence)
    assert "block_durations" in dir(sequence)
