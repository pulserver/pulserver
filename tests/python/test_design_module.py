"""What a `SequenceModule` publishes, and what it refuses to.

A module's whole contract is that the events it designed come back under the
names its constructor gave them, collapsed to one object per name when only
one object ever wore it. Everything here pins one half of that rule.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver import SequenceModule


@pytest.fixture
def system():
    return pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")


def _rf(system, duration=1e-3):
    # `use` is not decoration: the trajectory core needs to know which pulses
    # start a readout period, and refuses a sequence whose pulses do not say.
    return pp.make_block_pulse(
        flip_angle=np.deg2rad(15), duration=duration, use="excitation", system=system
    )


def _trap(system, area=500.0, channel="x"):
    return pp.make_trapezoid(channel, area=area, system=system)


class Arms(SequenceModule):
    """One pulse replayed per arm, one gradient per arm, one ADC throughout."""

    def init_module(self, system, num_arms=3, share_gradient=False):
        rf = _rf(system)
        adc = pp.make_adc(num_samples=64, dwell=4e-6, system=system)
        if share_gradient:
            shared = _trap(system)
            gread = [shared] * num_arms
        else:
            gread = [_trap(system, area=500.0 * (n + 1)) for n in range(num_arms)]
        self.seq = pp.Sequence(system)
        for n in range(num_arms):
            self.seq.add_block(rf)
            self.seq.add_block(gread[n], adc)
        self.center = rf.center


def test_an_event_played_many_times_is_published_once(system):
    arms = Arms(system, num_arms=4)
    assert arms.rf is arms.events.rf
    assert getattr(arms.rf, "type", None) == "rf"
    assert arms.adc.num_samples == 64


def test_distinct_objects_under_one_name_stay_a_list(system):
    arms = Arms(system, num_arms=3)
    assert isinstance(arms.gread, list)
    assert [round(g.area) for g in arms.gread] == [500, 1000, 1500]


def test_one_object_repeated_under_one_name_collapses(system):
    """A trajectory whose arms are the same waveform has one waveform.

    This is the case the identity dedup exists for: a constructor may build its
    per-arm list by repetition, and the module must say so rather than hand
    back N references to one gradient.
    """
    arms = Arms(system, num_arms=3, share_gradient=True)
    assert not isinstance(arms.gread, list)
    assert round(arms.gread.area) == 500


def test_blocks_are_the_published_events_in_play_order(system):
    arms = Arms(system, num_arms=2)
    blocks = arms.blocks
    assert len(blocks) == 4
    assert blocks[0] == (arms.rf,)
    assert blocks[1] == (arms.gread[0], arms.adc)
    assert blocks[3] == (arms.gread[1], arms.adc)


def test_mutating_a_published_event_shows_through_the_blocks(system):
    """The blocks hold the very objects that were published, not copies.

    Pulseq events are C extension types and cannot be copied, so this is not an
    optimisation -- it is the only thing a per-shot loop can do.
    """
    arms = Arms(system, num_arms=1)
    arms.rf.phase_offset = 1.25
    assert arms.blocks[0][0].phase_offset == 1.25


def test_duration_and_center_describe_the_module(system):
    arms = Arms(system, num_arms=2)
    assert arms.duration == pytest.approx(arms.seq.duration()[0])
    assert arms.center == pytest.approx(arms.rf.center)

    arms.duration = 0.5
    assert arms.duration == 0.5


class BuiltInHelper(SequenceModule):
    """Events built where `init_module` has no local holding them."""

    def init_module(self, system):
        self.seq = pp.Sequence(system)
        self._build(system)

    def _build(self, system):
        gspoil = _trap(system, area=2000.0, channel="z")
        self.seq.add_block(gspoil)
        self.publish()


def test_publish_reaches_a_helper_functions_locals(system):
    module = BuiltInHelper(system)
    assert round(module.gspoil.area) == 2000


class Unnamed(SequenceModule):
    def init_module(self, system):
        self.seq = pp.Sequence(system)
        self.seq.add_block(pp.make_delay(1e-3))


def test_a_module_that_names_nothing_says_so(system):
    with pytest.warns(UserWarning, match="published no events"):
        Unnamed(system)


class Registered(SequenceModule):
    """A bank the loop chooses from, only one of whose entries is laid out."""

    def init_module(self, system):
        bank = [_trap(system, area=100.0 * (n + 1)) for n in range(3)]
        self.seq = pp.Sequence(system)
        self.seq.add_block(bank[0])
        self.register(bank=bank, first=bank[0])


def test_register_publishes_events_no_block_played(system):
    module = Registered(system)
    assert len(module.bank) == 3
    assert module.first is module.bank[0]


class Shadowing(SequenceModule):
    def init_module(self, system):
        duration = _trap(system)
        self.seq = pp.Sequence(system)
        self.seq.add_block(duration)


def test_shadowing_a_module_attribute_warns_and_stays_reachable(system):
    with pytest.warns(UserWarning, match="also a module attribute"):
        module = Shadowing(system)
    assert getattr(module.events.duration, "type", None) == "trap"
    # The module's own attribute wins, which is why the warning is worth making.
    assert isinstance(module.duration, float)


class NoSequence(SequenceModule):
    def init_module(self, system):
        pass


class NoBlocks(SequenceModule):
    def init_module(self, system):
        self.seq = pp.Sequence(system)


@pytest.mark.parametrize(
    ("module", "message"),
    [(NoSequence, "never assigned self.seq"), (NoBlocks, "added no blocks")],
)
def test_an_incomplete_module_is_refused(system, module, message):
    with pytest.raises(TypeError, match=message):
        module(system)


class WrongSequence(SequenceModule):
    def init_module(self, system):
        import pypulseq

        self.seq = pypulseq.Sequence(system=system)


def test_an_upstream_sequence_is_refused(system):
    """Upstream's Sequence cannot record, and its events are not the fast ones."""
    with pytest.raises(TypeError, match=r"pulserver\.pypulseq\.Sequence"):
        WrongSequence(system)


class Readable(SequenceModule):
    """A whole TR, so every analysis has something to answer about."""

    def init_module(self, system):
        rf = _rf(system)
        gread = pp.make_trapezoid(
            "x", flat_area=2000.0, flat_time=2.56e-3, system=system
        )
        gpre = pp.make_trapezoid("x", area=-0.5 * gread.area, system=system)
        adc = pp.make_adc(
            num_samples=64, dwell=4e-5, delay=gread.rise_time, system=system
        )
        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)
        self.seq.add_block(gpre)
        self.seq.add_block(gread, adc)


def test_analyses_are_answered_by_the_inner_sequence(system):
    module = Readable(system)
    assert module.check_timing()[0]
    k_traj_adc = module.calculate_kspace()[0]
    assert k_traj_adc.shape[1] == 64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(module.test_report(), str)


def test_dir_offers_events_and_analyses(system):
    module = Arms(system)
    listing = dir(module)
    assert {"rf", "adc", "gread", "plot_kspace", "calculate_pns", "duration"} <= set(
        listing
    )


def test_an_unknown_name_names_the_module(system):
    module = Arms(system)
    with pytest.raises(AttributeError, match="no attribute or event 'gphase'"):
        module.gphase  # noqa: B018 -- the attribute access is what must raise


class Base(SequenceModule):
    """A family that lays out the blocks, given what a subclass designed."""

    def init_module(self, system, gread):
        rf = _rf(system)
        adc = pp.make_adc(num_samples=64, dwell=4e-6, system=system)
        self.seq = pp.Sequence(system)
        self.seq.add_block(rf)
        self.seq.add_block(gread, adc)


class Narrowed(Base):
    """A subclass that designs the waveform and delegates the layout."""

    def init_module(self, system):
        gspecific = _trap(system, area=1234.0)
        super().init_module(system, gspecific)


def test_publication_reaches_every_init_module_in_the_chain(system):
    """Both frames are the constructor, so neither needs an explicit publish().

    A subclass that narrows a family builds some events itself and delegates
    the rest. Reading only the frame the walk stopped at would leave half the
    module unpublished and make the author say so by hand.
    """
    module = Narrowed(system)
    assert round(module.gspecific.area) == 1234  # the subclass's frame
    assert module.adc.num_samples == 64  # the base's frame
    assert getattr(module.rf, "type", None) == "rf"
    assert module.gread is module.gspecific
